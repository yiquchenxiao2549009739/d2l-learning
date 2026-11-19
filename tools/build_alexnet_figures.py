"""使用现有环境生成7.1结构、资源和插值图，不训练、不下载。"""
import runpy
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def main():
    helper = runpy.run_path(str(ROOT / "lessons/ch07/01_alexnet.py"))
    helper["COMMON"]["configure"]("cpu")
    result = helper["inspect_net"]()
    helper["write_json"]("7_1_architecture.json", result)
    helper["write_csv"]("7_1_layers.csv", result["layers"])
    plt = helper["plot_setup"]()
    destination = ROOT / "reports/ch07/figures"
    labels = ["卷积1 + 汇聚\n96×26×26", "卷积2 + 汇聚\n256×12×12",
              "卷积3 + ReLU\n384×12×12", "卷积4 + ReLU\n384×12×12",
              "卷积5 + 汇聚\n256×5×5", "展平 → FC1\n6400 → 4096", "FC2\n4096 → 4096", "logits\n4096 → 10"]
    fig, ax = plt.subplots(figsize=(8.5, 3.3), constrained_layout=True)
    for index, label in enumerate(labels):
        x, y = index % 4, 1 - index // 4
        ax.text(x, y, label, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=.6", fc="#e8f4f3", ec="#147e83"))
        if index % 4 < 3:
            ax.annotate("", xy=(x + .66, y), xytext=(x + .4, y), arrowprops=dict(arrowstyle="->"))
    ax.text(1.5, .5, "上行从左到右，再读下行；FC1/FC2 后均有 ReLU 与 Dropout", ha="center", fontsize=10)
    ax.set(xlim=(-.65, 3.65), ylim=(-.45, 1.45))
    ax.axis("off")
    fig.savefig(destination / "7_1_architecture.png", dpi=180)
    plt.close(fig)
    rows = [r for r in result["layers"] if r["parameters"]]
    names = ["C1", "C2", "C3", "C4", "C5", "FC1", "FC2", "FC3"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), constrained_layout=True)
    axes[0].bar(names, [r["parameters"] / 1e6 for r in rows], color="#147e83")
    axes[0].set(ylabel="参数（百万）", title="全连接层拥有多数参数")
    axes[1].bar(names, [r["weight_multiplications"] / 1e6 for r in rows], color="#e6ab52")
    axes[1].set(ylabel="单图权重乘法（百万）", title="卷积层承担多数乘法")
    fig.savefig(destination / "7_1_cost.png", dpi=180)
    plt.close(fig)
    splits, _ = helper["COMMON"]["load_data"](include_test=False)
    X, label = splits["validation"][0]
    enlarged = F.interpolate(X[None], size=(224, 224), mode="bilinear", align_corners=False, antialias=True)[0, 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), constrained_layout=True)
    axes[0].imshow(X[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].imshow(enlarged, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("原始 28×28")
    axes[1].set_title("插值 224×224；没有新增采集细节")
    for ax in axes:
        ax.set(xticks=[], yticks=[])
    fig.savefig(destination / "7_1_resize.png", dpi=180)
    plt.close(fig)
    helper["write_json"]("7_1_resize_example.json", dict(validation_position=0,
        source_train_index=int(splits["validation"].indices[0]), label=int(label), original_shape=list(X.shape),
        resized_shape=list(enlarged.shape), method="float32 /255 then bilinear, align_corners=False, antialias=True"))
    print("7.1 structure/cost/resize figures ready")


if __name__ == "__main__":
    main()
