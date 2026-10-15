"""用已有学习环境的 Matplotlib 重建 2.3 配图，不下载或安装软件。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ch02" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "font.size": 11,
    "savefig.facecolor": "white",
})
NAVY, TEAL, ORANGE, GRAY = "#173448", "#147E83", "#D07632", "#657987"


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(name)


fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.65))
for ax, y, title in zip(axes, [(1.3, 1.3), (0, 1.8), (-1.3, 1.3)],
                        ["锐角：点积 > 0", "直角：点积 = 0", "钝角：点积 < 0"]):
    ax.axhline(0, color="#D7E2E9", lw=0.8)
    ax.axvline(0, color="#D7E2E9", lw=0.8)
    ax.annotate("", (1.8, 0), (0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5))
    ax.annotate("", y, (0, 0), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5))
    ax.text(1.83, -0.22, "x", color=TEAL, fontsize=13)
    ax.text(y[0] + 0.12, y[1] + 0.05, "y", color=ORANGE, fontsize=13)
    ax.set(title=title, xlim=(-1.7, 2.2), ylim=(-0.45, 2.2), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.tight_layout(w_pad=2)
save(fig, "2_3_dot_geometry.png")


fig, ax = plt.subplots(figsize=(10, 3.05))
ax.set(xlim=(-0.1, 12.6), ylim=(-1.0, 2.9))
ax.axis("off")


def matrix(values, left, label, highlight):
    rows, cols = len(values), len(values[0])
    size = 0.72
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            selected = highlight(i, j)
            color = "#E4F2F0" if selected else "#F5F7FA"
            box = FancyBboxPatch((left + j * size, 1.4 - i * size), size-0.04, size-0.04,
                                 boxstyle="round,pad=0.015", ec="#D7E2E9", fc=color)
            ax.add_patch(box)
            ax.text(left + j*size + (size-0.04)/2, 1.4 - i*size + (size-0.04)/2,
                    str(value), ha="center", va="center", color=TEAL if selected else NAVY,
                    fontsize=14, fontweight="bold" if selected else "normal")
    ax.text(left+cols*size/2, 2.48, label, ha="center", color=NAVY, fontsize=13)


matrix([[1, 2, 3], [4, 5, 6]], 0.4, "A：2 条样本 × 3 特征", lambda i, j: i == 0)
matrix([[1, 2], [0, -1], [1, 0.5]], 4.55, "W：3 特征 × 2 输出", lambda i, j: j == 1)
matrix([[4, 1.5], [10, 6]], 9.3, "C：2 条样本 × 2 输出", lambda i, j: i == 0 and j == 1)
ax.text(3.4, 1.3, "@", ha="center", fontsize=23, color=ORANGE)
ax.text(7.6, 1.3, "=", ha="center", fontsize=23, color=ORANGE)
ax.text(6.25, -0.65, "C[0,1] = 左第 0 行 · 右第 1 列 = 1×2 + 2×(-1) + 3×0.5 = 1.5",
        ha="center", color=NAVY, fontsize=12)
save(fig, "2_3_matrix_product.png")


fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.25))
ax = axes[0]
ax.plot([0, 3, 3], [0, 0, -4], "--", color=ORANGE, lw=2.2, label="L1：3 + 4 = 7")
ax.annotate("", (3, -4), (0, 0), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5))
ax.plot([], [], color=TEAL, lw=2.5, label="L2：直线长度 5")
ax.scatter([3], [-4], color=NAVY, s=24, zorder=4)
ax.text(2.9, -4.6, "(3,-4)", color=NAVY)
ax.set(xlim=(-0.6, 4.5), ylim=(-5.0, 2.2), aspect="equal", title="同一个向量，两种大小")
ax.set_xticks([0, 3])
ax.set_yticks([-4, 0])
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax = axes[1]
ax.add_patch(Circle((0, 0), 1, fill=False, ec=TEAL, lw=2.3, label="L2 = 1：圆"))
ax.add_patch(Polygon([(1, 0), (0, 1), (-1, 0), (0, -1)], fill=False, ec=ORANGE,
                     lw=2.0, linestyle="--", label="L1 = 1：菱形"))
ax.set(xlim=(-1.45, 1.65), ylim=(-1.3, 2.0), aspect="equal", title="大小等于 1 的向量集合")
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
for ax in axes:
    ax.axhline(0, color="#D7E2E9", lw=0.8, zorder=0)
    ax.axvline(0, color="#D7E2E9", lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.tight_layout(w_pad=3)
save(fig, "2_3_norm_geometry.png")
