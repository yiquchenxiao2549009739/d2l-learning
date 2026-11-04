"""6.3 填充和步幅。

接着 6.2 的窗口乘加，先用 output_shape 推导能放下多少个完整窗口。
corr2d 先显式补零，再将输出索引乘以步幅来定位窗口。
padding_demo、stride_demo 和 shape_demo 核对边界、对齐及尺寸。
gradient_demo 检查可训练性，audio_demo 解释跨步抽样与混叠。

加 --guided 阅读分阶段中文结果，加 --record 更新实测 JSON 和四张图。
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/ch06"
DTYPE = torch.float64


def output_shape(input_shape, kernel_shape, padding=(0, 0, 0, 0), stride=(1, 1)):
    """padding 按 F.pad 的左、右、上、下顺序，stride 按高、宽顺序。"""
    left, right, top, bottom = padding
    sh, sw = stride
    if min(padding) < 0 or min(stride) < 1 or min(kernel_shape) < 1:
        raise ValueError("本节只接受非负整数填充、正整数步幅和非空核")
    height = input_shape[0] + top + bottom - kernel_shape[0]
    width = input_shape[1] + left + right - kernel_shape[1]
    if min(height, width) < 0:
        raise ValueError("填充后的输入必须能容纳完整卷积核")
    return height // sh + 1, width // sw + 1


def corr2d(X, K, padding=(0, 0, 0, 0), stride=(1, 1)):
    """将 6.2 的逐窗口互相关扩展到显式零填充和步幅。"""
    rows, cols = output_shape(X.shape, K.shape, padding, stride)
    padded = F.pad(X, padding)
    kh, kw = K.shape
    sh, sw = stride
    outputs = []
    for i in range(rows):
        line = []
        for j in range(cols):
            row, col = i * sh, j * sw
            patch = padded[row:row + kh, col:col + kw]
            line.append((patch * K).sum())
        outputs.append(torch.stack(line))
    return torch.stack(outputs)


def shape_demo():
    X = torch.arange(64, dtype=DTYPE).reshape(8, 8)
    cases = [
        ("无填充", (3, 3), (0, 0), (1, 1), (6, 6)),
        ("3×3 保持尺寸", (3, 3), (1, 1), (1, 1), (8, 8)),
        ("5×3 保持尺寸", (5, 3), (2, 1), (1, 1), (8, 8)),
        ("步幅 2", (3, 3), (1, 1), (2, 2), (4, 4)),
        ("教材最后一例", (3, 5), (0, 1), (3, 4), (2, 2)),
        ("有余数", (3, 3), (0, 0), (2, 2), (3, 3)),
    ]
    records = []
    for name, kernel, padding, stride, expected in cases:
        ph, pw = padding
        pad4 = (pw, pw, ph, ph)
        K = torch.ones(kernel, dtype=DTYPE)
        hand = corr2d(X, K, pad4, stride)
        builtin = F.conv2d(X[None, None], K[None, None], padding=padding, stride=stride)[0, 0]
        torch.testing.assert_close(hand, builtin)
        assert tuple(hand.shape) == output_shape(X.shape, kernel, pad4, stride) == expected
        records.append({
            "name": name, "kernel": kernel, "padding_per_side_hw": padding,
            "stride_hw": stride, "output_shape": list(hand.shape),
            "max_error": (hand - builtin).abs().max().item(),
        })
    # 奇数输入：保持尺寸所需的总填充配步幅 2，得到 ceil(7/2)=4。
    odd = corr2d(X[:7, :7], torch.ones(3, 3, dtype=DTYPE), (1, 1, 1, 1), (2, 2))
    assert odd.shape == (4, 4)
    return records


def padding_demo():
    X = torch.arange(9, dtype=DTYPE).reshape(3, 3)
    K = torch.arange(4, dtype=DTYPE).reshape(2, 2)
    border = corr2d(X, K, (1, 1, 1, 1))
    assert border.shape == (4, 4) and border[0, 0] == 0
    # 总填充均为 1，教材给出上/左多一格的可选对齐。
    before = corr2d(X, K, (1, 0, 1, 0))
    after = corr2d(X, K, (0, 1, 0, 1))
    same_layer = nn.Conv2d(1, 1, 2, padding="same", bias=False, dtype=DTYPE)
    with torch.no_grad():
        same_layer.weight.copy_(K[None, None])
    same = same_layer(X[None, None])[0, 0]
    torch.testing.assert_close(same, after)
    assert before.shape == after.shape == X.shape
    try:
        nn.Conv2d(1, 1, 3, stride=2, padding="same")
    except ValueError as error:
        same_stride_error = str(error)
    else:
        raise AssertionError("本机版本预期拒绝 same 与步幅 2 的组合")
    return {
        "padded_response": border.tolist(),
        "even_kernel_more_before": before.tolist(),
        "even_kernel_more_after": after.tolist(),
        "pytorch_same_matches_more_after": True,
        "same_stride_2_error": same_stride_error,
    }, (X, F.pad(X, (1, 1, 1, 1)), border, before, after)


def stride_demo():
    X = torch.arange(1, 21, dtype=DTYPE).reshape(4, 5)
    K = torch.tensor([[1., 2.], [3., 4.]], dtype=DTYPE)
    padding, stride = (1, 1, 1, 1), (2, 3)
    dense = corr2d(X, K, padding)
    sparse = corr2d(X, K, padding, stride)
    torch.testing.assert_close(sparse, dense[::2, ::3])
    # 边界像素仍参与填充后的窗口，但只出现在窗口角落。
    coverage = torch.zeros_like(F.pad(X, padding))
    for row in range(dense.shape[0]):
        for col in range(dense.shape[1]):
            coverage[row:row + 2, col:col + 2] += 1
    return {
        "input": X.tolist(), "kernel": K.tolist(),
        "dense_response": dense.tolist(), "strided_response": sparse.tolist(),
        "padded_window_starts": [[i * 2, j * 3] for i in range(3) for j in range(2)],
        "original_pixel_coverage": coverage[1:-1, 1:-1].tolist(),
    }, (dense, sparse)


def gradient_demo():
    generator = torch.Generator().manual_seed(6301)
    X = torch.randn(4, 5, generator=generator, dtype=DTYPE, requires_grad=True)
    K = torch.randn(2, 3, generator=generator, dtype=DTYPE, requires_grad=True)
    padding, stride = (1, 2, 2, 0), (2, 3)
    builtin_X = X.detach().clone().requires_grad_()
    builtin_K = K.detach().clone().requires_grad_()
    hand = corr2d(X, K, padding, stride)
    builtin = F.conv2d(F.pad(builtin_X[None, None], padding), builtin_K[None, None], stride=stride)[0, 0]
    hand.square().mean().backward()
    builtin.square().mean().backward()
    errors = {
        "output": (hand - builtin).abs().max().item(),
        "input_gradient": (X.grad - builtin_X.grad).abs().max().item(),
        "kernel_gradient": (K.grad - builtin_K.grad).abs().max().item(),
    }
    assert max(errors.values()) < 1e-12
    return errors


def audio_demo():
    n = torch.arange(64, dtype=DTYPE)
    low = torch.cos(2 * torch.pi * .125 * n)
    high = torch.cos(2 * torch.pi * .375 * n)
    alias_error = (low[::2] - high[::2]).abs().max().item()
    assert alias_error < 1e-12
    smooth = torch.tensor([[[.25, .5, .25]]], dtype=DTYPE)
    filtered_low = F.conv1d(low[None, None], smooth, stride=2).flatten()
    filtered_high = F.conv1d(high[None, None], smooth, stride=2).flatten()
    ratio = (filtered_high.square().mean() / filtered_low.square().mean()).sqrt().item()
    return {
        "original_sample_rate_hz": 8000, "output_sample_rate_hz": 4000,
        "frequencies_hz": [1000, 3000], "downsampled_max_difference": alias_error,
        "three_tap_filtered_high_to_low_rms_ratio": ratio,
        "macs_8x8_k3_stride1": 8 * 8 * 9,
        "macs_8x8_k3_stride2": 4 * 4 * 9,
    }, (low, high)


def record(metrics, padding_images, stride_images, audio_images):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out = REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (REPORTS / "6_3_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})

    def grid(ax, values, title):
        ax.imshow(values.detach().numpy(), cmap="Blues")
        ax.set(title=title, xticks=[], yticks=[])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=11,
                        color="white" if values[i, j] > values.max() * .65 else "#173448")

    X, padded, border, before, after = padding_images
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.85), layout="constrained")
    for ax, values, title in zip(axes, (X, padded, border), ("输入 3×3", "每侧补 1 → 5×5", "2×2 核 → 4×4")):
        grid(ax, values, title)
    axes[1].add_patch(Rectangle((.5, .5), 3, 3, fill=False, edgecolor="#c16d22", lw=2))
    fig.savefig(out / "6_3_padding.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.85), layout="constrained")
    grid(axes[0], before, "2×2 核：上/左各补 1")
    grid(axes[1], after, "2×2 核：下/右各补 1（same）")
    fig.savefig(out / "6_3_alignment.png", dpi=180)
    plt.close(fig)

    dense, sparse = stride_images
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.05), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
    grid(axes[0], dense, "步幅 (1,1)：5×6 响应")
    grid(axes[1], sparse, "步幅 (2,3)：3×2")
    for i in range(0, 5, 2):
        for j in range(0, 6, 3):
            axes[0].add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="#e69a39", lw=2.3))
    fig.savefig(out / "6_3_stride.png", dpi=180)
    plt.close(fig)

    low, high = audio_images
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 3.6), layout="constrained")
    axes[0].plot(range(16), low[:16], "o-", label="原始 1 kHz")
    axes[0].plot(range(16), high[:16], "s--", label="原始 3 kHz", alpha=.8)
    axes[1].plot(range(16), low[::2][:16], "o-", label="1 kHz 每隔 2 点取样")
    axes[1].plot(range(16), high[::2][:16], "x--", label="3 kHz 每隔 2 点取样")
    for ax in axes:
        ax.set(ylabel="幅度", ylim=(-1.3, 1.3))
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=.2)
    axes[0].set(xlabel="原始采样点 n")
    axes[1].set(xlabel="输出点 m，对应原始 n=2m")
    fig.savefig(out / "6_3_audio.png", dpi=180)
    plt.close(fig)


def print_guided(metrics):
    """按边界、尺寸、位置、可训练性和采样的顺序连接结果。"""
    print("\n6.3 填充和步幅\n")
    print("步骤 1 看清补零如何增加可用窗口")
    print("3×3 输入每侧补 1 后，用 2×2 核得到")
    print(metrics["padding"]["padded_response"])
    print("2×2 核保尺寸的两种对齐")
    print("上/左多补", metrics["padding"]["even_kernel_more_before"])
    print("下/右多补", metrics["padding"]["even_kernel_more_after"], "\n")
    print("步骤 2 用完整窗口条件核对输出形状")
    for case in metrics["shapes"]:
        print(case["name"], "核", case["kernel"], "单侧填充", case["padding_per_side_hw"],
              "步幅", case["stride_hw"], "输出", case["output_shape"])
    print("教材总填充量等于两侧之和。起点从 0 开始，所以整除后还要加 1。\n")
    print("步骤 3 把输出索引换成输入窗口起点")
    print("填充网格中的窗口起点", metrics["stride"]["padded_window_starts"])
    print("步幅 (2,3) 的响应", metrics["stride"]["strided_response"])
    print("每个窗口内部仍用同一核逐元素乘加。\n")
    print("步骤 4 核对梯度并理解跨步抽样")
    print("前向和梯度误差", metrics["gradient_errors"])
    audio = metrics["audio"]
    print("输出采样率", audio["output_sample_rate_hz"], "Hz")
    print("1/3 kHz 隔点采样后最大差", audio["downsampled_max_difference"])
    print("三点平滑后高频/低频 RMS 比", audio["three_tap_filtered_high_to_low_rms_ratio"])
    print("步幅 1/2 的理论乘加次数", audio["macs_8x8_k3_stride1"], audio["macs_8x8_k3_stride2"])
    print("接着按讲义逐项改参数，再打开 exercises/ch06/03_padding_and_stride.py。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="保存实测 JSON 和四幅图")
    parser.add_argument("--guided", action="store_true", help="按讲义步骤显示中文结果和解释")
    args = parser.parse_args()
    torch.set_num_threads(1)
    padding, padding_images = padding_demo()
    stride, stride_images = stride_demo()
    audio, audio_images = audio_demo()
    metrics = {
        "section": "6.3", "date": "2026-11-04", "torch": torch.__version__,
        "device": "cpu", "dtype": str(DTYPE), "shapes": shape_demo(),
        "padding": padding, "stride": stride, "gradient_errors": gradient_demo(), "audio": audio,
    }
    if args.record:
        record(metrics, padding_images, stride_images, audio_images)
    if args.guided:
        print_guided(metrics)
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
