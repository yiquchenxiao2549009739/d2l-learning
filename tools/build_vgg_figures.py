"""7.2结构与原理图；已有运行环境，无训练、下载或依赖安装。"""
import runpy
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def block_demo():
    # 使用valid边界且无中间激活时，两次3×3线性互相关可合成5×5。
    gen = torch.Generator().manual_seed(7205)
    X = torch.randn(1, 1, 9, 9, generator=gen, dtype=torch.float64)
    K1 = torch.randn(1, 1, 3, 3, generator=gen, dtype=torch.float64)
    K2 = torch.randn(1, 1, 3, 3, generator=gen, dtype=torch.float64)
    combined = torch.zeros(1, 1, 5, 5, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            combined[0, 0, i:i + 3, j:j + 3] += K2[0, 0, i, j] * K1[0, 0]
    sequential = F.conv2d(F.conv2d(X, K1), K2)
    one = F.conv2d(X, combined)
    error = (sequential - one).abs().max().item()
    ones = torch.ones(1, 1, 3, 3, dtype=torch.float64)
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float64)
    coefficients = torch.tensor([1., 2., 3., 2., 1.], dtype=torch.float64)
    five = (coefficients[:, None] * coefficients[None, :])[None, None]
    two_padded = F.conv2d(F.conv2d(ones, kernel, padding=1), kernel, padding=1)
    one_padded = F.conv2d(ones, five, padding=2)
    identity3 = torch.zeros_like(kernel)
    identity3[0, 0, 1, 1] = 1
    nonlinear = F.conv2d(F.relu(F.conv2d(X, identity3)), identity3)
    linear = F.conv2d(F.conv2d(X, identity3), identity3)
    assert error < 1e-12
    assert two_padded[0, 0, 0, 0] == 25 and one_padded[0, 0, 0, 0] == 36
    return dict(valid_linear_max_error=error, padded_two_corner=25, padded_combined_corner=36,
                relu_vs_linear_max_difference=(nonlinear - linear).abs().max().item(),
                two3_vs_one5_equal_width_weight_coefficients=[18, 25],
                three3_vs_one7_equal_width_weight_coefficients=[27, 49])


def main():
    H = runpy.run_path(str(ROOT / "lessons/ch07/02_vgg.py"))
    H["COMMON"]["configure"]("cpu")
    data = H["structures"]()
    H["write_json"]("7_2_architectures.json", data)
    H["write_csv"]("7_2_weighted_layers.csv", [dict(model=name, **row) for name, item in data.items() for row in item["weighted_layers"]])
    first = H["vgg_block"](2, 3, 8)
    second = H["vgg_block"](2, 3, 8)
    assert first[0].weight is not second[0].weight
    demo = block_demo()
    demo.update(independent_block_parameters=True, two_conv_block_parameters=sum(p.numel() for p in first.parameters()),
                two_conv_block_shape=list(first(torch.zeros(2, 3, 16, 16)).shape))
    H["write_json"]("7_2_block_demo.json", demo)
    plt = H["plot_setup"]()
    folder = ROOT / "reports/ch07/figures"
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    for i, (count, channels) in enumerate(zip([1, 1, 2, 2, 2], [64, 128, 256, 512, 512])):
        ax.text(i, 1, f"块{i+1}\n{count}×(3×3+ReLU)\n{channels}通道 + 汇聚", ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=.5", fc="#e8f4f3", ec="#147e83"))
        ax.text(i, 0, str([112, 56, 28, 14, 7][i]) + "×" + str([112, 56, 28, 14, 7][i]), ha="center", fontsize=12)
        if i < 4:
            ax.annotate("", xy=(i + .63, 1), xytext=(i + .38, 1), arrowprops=dict(arrowstyle="->"))
    ax.text(2, -.48, "完整版：512×7×7 → 4096 → 4096 → 10", ha="center", fontsize=12)
    ax.text(2, -.93, "教材训练版通道/4：128×7×7 → 4096 → 4096 → 10", ha="center", fontsize=11)
    ax.set(xlim=(-.65, 4.65), ylim=(-1.2, 1.8))
    ax.axis("off")
    fig.savefig(folder / "7_2_architecture.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 3.3), constrained_layout=True)
    left = ["顶层 0: Sequential（块1）", "顶层 1: Sequential（块2）", "顶层 2: Sequential（块3）",
            "顶层 3: Sequential（块4）", "顶层 4: Sequential（块5）", "顶层 6/9/12: 三个 Linear"]
    right = ["0.0 Conv：第1个", "1.0 Conv：第2个", "2.0 / 2.2 Conv：第3、4个",
             "3.0 / 3.2 Conv：第5、6个", "4.0 / 4.2 Conv：第7、8个", "三个 Linear：第9、10、11个"]
    for i, (a, b) in enumerate(zip(left, right)):
        y = 5 - i
        ax.text(0, y, a, fontsize=11, va="center")
        ax.text(1.22, y, "→", fontsize=14, va="center")
        ax.text(1.5, y, b, fontsize=11, va="center", color="#147e83")
    ax.set(xlim=(-.05, 3.5), ylim=(-.6, 5.7))
    ax.axis("off")
    fig.savefig(folder / "7_2_counting.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), constrained_layout=True)
    for ax, size in zip(axes, [3, 5, 7]):
        image = torch.zeros(9, 9)
        start = (9 - size) // 2
        image[start:start + size, start:start + size] = 1
        ax.imshow(image, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{(size-1)//2}层3×3：{size}×{size}", fontsize=12)
        ax.set(xticks=[], yticks=[])
        ax.plot(4, 4, "o", color="#e6ab52", markersize=7)
    fig.savefig(folder / "7_2_receptive.png", dpi=180)
    plt.close(fig)
    names = ["Alex224", "VGG全224", "VGG缩224", "VGG缩96"]
    alex = H["PREVIOUS"]["inspect_net"]()
    values = [alex, data["full11"], data["quarter11"], data["quarter96"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    axes[0].bar(names, [v["parameters"] / 1e6 for v in values], color="#147e83")
    axes[0].set(ylabel="参数（百万）")
    axes[1].bar(names, [v["weight_multiplications"] / 1e9 if "weight_multiplications" in v else 0 for v in values], color="#e6ab52")
    axes[1].set(ylabel="单图权重乘法（十亿）")
    fig.savefig(folder / "7_2_cost.png", dpi=180)
    plt.close(fig)
    print("VGG four structure/principle figures and block checks ready")


if __name__ == "__main__":
    main()
