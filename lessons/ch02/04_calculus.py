"""完整 2.4 微积分：数值逼近 -> 手工求导 -> 梯度 -> 链式法则 -> 更新。

原教材 PDF 第 81-87 页，正文第 63-69 页。本节不调用自动求导；
用现有 PyTorch / Matplotlib 计算并生成三张原理图，无需 d2l 或 Notebook。
"""

import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

REPORT = ROOT / "reports" / "ch02"
FIGURES = REPORT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
torch.set_printoptions(precision=6, sci_mode=False)


def f(x):
    return 3 * x ** 2 - 4 * x


def df(x):
    return 6 * x - 4


def forward_difference(fn, x, h):
    """有限差分是近似，用来核对解析求导，不是自动微分。"""
    return (fn(x + h) - fn(x)) / h


def central_difference(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2 * h)


def batch_loss(X, w, b, y):
    return ((X @ w + b - y) ** 2).mean()


print("===== 1. 导数：观察越来越短区间上的变化率 =====")
for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    print("h=%.5f，差商=%.5f" % (h, forward_difference(f, 1.0, h)))
print("解析求导 f'(x)=6x-4；f'(1)=", df(1.0))
print("过小的 h 可能丢失有效数字，h=1e-16 时差商=",
      forward_difference(f, 1.0, 1e-16))

print("\n===== 2. 切线和微分：局部变化的线性近似 =====")
delta = 0.1
approx = f(1.0) + df(1.0) * delta
print("x 从 1 变为 1.1，切线估计 f(x)=", approx)
print("真实 f(1.1)=", f(1.0 + delta), "; 近似不是精确等式。")
print("abs(x) 在 0 的左右差商:",
      forward_difference(abs, 0.0, -1e-4), forward_difference(abs, 0.0, 1e-4))

print("\n===== 3. 常见求导法则：求值不等于求导 =====")
z = 2.0
print("常数 7 的导数=0；x^3 的导数=3x^2，在 2 处为", 3 * z ** 2)
print("exp(x) 的导数仍为 exp(x)，在 2 处为", math.exp(z))
print("log(x) 的导数=1/x，在 2 处为", 1 / z)
print("乘积 x^2*exp(x) 的导数=exp(x)*(2x+x^2)，在 2 处为",
      math.exp(z) * (2 * z + z ** 2))

print("\n===== 4. 偏导与梯度：一次只改变一个坐标 =====")
point = torch.tensor([1.0, 0.0], dtype=torch.float64)


def g(v):
    return 3 * v[0] ** 2 + 5 * torch.exp(v[1])


gradient = torch.stack([6 * point[0], 5 * torch.exp(point[1])])
numeric_components = []
for i in range(point.numel()):
    step = torch.zeros_like(point)
    step[i] = 1e-5
    numeric_components.append((g(point + step) - g(point - step)) / (2e-5))
numeric_gradient = torch.stack(numeric_components)
print("g(x1,x2)=3x1^2+5exp(x2)，在 [1,0] 的函数值:", g(point).item())
print("手工梯度:", gradient)
print("逐坐标中心差分核对:", numeric_gradient)

print("\n===== 5. 梯度与局部下降方向 =====")
q_point = torch.tensor([1.0, -2.0], dtype=torch.float64)
q_gradient = torch.tensor([2.0, -8.0], dtype=torch.float64)
q_next = q_point - 0.1 * q_gradient
q_old = q_point[0] ** 2 + 2 * q_point[1] ** 2
q_new = q_next[0] ** 2 + 2 * q_next[1] ** 2
print("q=x1^2+2x2^2；当前点:", q_point, "; 梯度:", q_gradient)
print("沿负梯度走 0.1 倍:", q_next)
print("旧值 / 新值:", q_old.item(), q_new.item())
print("一阶近似的新值:", (q_old - 0.1 * q_gradient.square().sum()).item())
u = torch.tensor([3.0, -4.0], dtype=torch.float64)
print("L2 范数的梯度（非零点）:", u / torch.linalg.vector_norm(u))
print("L2 平方的梯度:", 2 * u)

print("\n===== 6. 链式法则：沿每条路径相乘，多条路径相加 =====")
chain_x = 2.0
a = chain_x ** 2
c = 3 * chain_x
branch_gradient = c * (2 * chain_x) + a * 3
print("a=x^2，c=3x，z=a*c；x=2 时 dz/dx=", branch_gradient)
print("直接合并 z=3x^3 后求导=", 9 * chain_x ** 2)

print("\n===== 7. 重现你做过的实验：固定 b，只更新 w =====")
sample_x, target, old_w, fixed_b = 3.0, 10.0, 2.0, 1.0
old_pred = sample_x * old_w + fixed_b
old_error = old_pred - target
old_loss = old_error ** 2
grad_w = 2 * old_error * sample_x
print("预测 / 误差 / 损失:", old_pred, old_error, old_loss)
print("dL/de * de/dpred * dpred/dw = (-6)*1*3 =", grad_w)
updates = []
for lr in [0.01, 0.2]:
    new_w = old_w - lr * grad_w  # 两次都从同一个旧 w=2 出发。
    new_pred = sample_x * new_w + fixed_b
    new_loss = (new_pred - target) ** 2
    updates.append(dict(lr=lr, old_w=old_w, gradient=grad_w,
                        new_w=new_w, prediction=new_pred, loss=new_loss))
    print("lr=%.2f：w=%.4f，预测=%.4f，损失=%.4f" %
          (lr, new_w, new_pred, new_loss))
with (REPORT / "2_4_learning_rates.csv").open("w", encoding="utf-8", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(updates[0]))
    writer.writeheader()
    writer.writerows(updates)

print("\n===== 8. 将偏导和链式法则合成一次小批量更新 =====")
X = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
w = torch.tensor([[0.5], [-1.0]], dtype=torch.float64)
b = torch.tensor(0.5, dtype=torch.float64)
y = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
pred = X @ w + b
error = pred - y
loss = error.square().mean()
dw = 2 / X.shape[0] * (X.T @ error)
db = 2 * error.mean()
lr = 0.01
w_after = w - lr * dw
b_after = b - lr * db  # 与 dw 一样，db 来自更新前的参数。
loss_after = batch_loss(X, w_after, b_after, y)
numeric_dw = torch.zeros_like(w)
for i in range(w.numel()):
    step = torch.zeros_like(w)
    step[i, 0] = 1e-5
    numeric_dw[i, 0] = (batch_loss(X, w + step, b, y)
                        - batch_loss(X, w - step, b, y)) / (2e-5)
numeric_db = central_difference(lambda value: batch_loss(X, w, value, y), b)
print("预测:\n", pred)
print("手工 dw / db:\n", dw, "\n", db)
print("中心差分 dw / db:\n", numeric_dw, "\n", numeric_db)
print("更新前 / 后损失:", loss.item(), loss_after.item())
print("本节手工写求导公式；2.5 再学习让框架自动求导。")


def make_figures():
    """绘图辅助区；先理解上面的数学，不要求自己重写排版代码。"""
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "savefig.facecolor": "white"})
    navy, teal, orange = "#173448", "#147E83", "#D07632"

    def save(fig, name):
        fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print("配图:", name)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.05))
    grid = np.linspace(0, 2.1, 200)
    for ax in axes:
        ax.plot(grid, f(grid), color=teal, label="f(x)=3x²-4x")
        ax.plot(grid, 2 * grid - 3, "--", color=orange, label="x=1 的切线")
        ax.scatter([1], [-1], color=navy, zorder=4)
        ax.set(xlabel="x", ylabel="函数值")
        ax.grid(alpha=0.16)
    axes[0].set(title="导数是切线斜率：f'(1)=2", ylim=(-2.2, 5.0))
    axes[1].set(title="放大局部：曲线接近切线", xlim=(0.8, 1.2), ylim=(-1.5, -0.35))
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(w_pad=2.5)
    save(fig, "2_4_tangent.png")

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    gx, gy = np.meshgrid(np.linspace(-2, 2, 180), np.linspace(-2.5, 2.5, 180))
    contours = ax.contour(gx, gy, gx**2 + 2*gy**2, levels=[0.5, 2, 4, 6, 9, 12],
                          colors=teal, linewidths=0.8, alpha=0.65)
    ax.clabel(contours, inline=True, fontsize=8)
    ax.scatter([1, 0.8, 0], [-2, -1.2, 0], c=[navy, orange, teal], s=35, zorder=5)
    ax.annotate("", (0.8, -1.2), (1, -2),
                arrowprops=dict(arrowstyle="->", color=orange, lw=2.3))
    ax.annotate("起点 (1,-2)，值 9", (1, -2), (1.25, -2.25), fontsize=9, color=navy)
    ax.annotate("更新后 (0.8,-1.2)\n值 3.52", (0.8, -1.2), (1.12, -0.95),
                fontsize=9, color=orange)
    ax.text(-0.95, 0.12, "最低点 (0,0)", color=teal)
    for label in ax.texts:
        if label.get_text().startswith(("起点", "更新后", "最低点")):
            label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
    ax.set(title="等高线：q=x1²+2x2²；沿负梯度走一小步", xlabel="x1", ylabel="x2",
           xlim=(-2, 2.5), ylim=(-2.6, 2.6), aspect="equal")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    save(fig, "2_4_gradient.png")

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set(xlim=(-0.15, 12.2), ylim=(-0.8, 2.3))
    ax.axis("off")
    labels = ["w=2", "预测=3w+1\n=7", "误差=预测-10\n=-3", "损失=误差²\n=9"]
    positions = [0.1, 3.1, 6.1, 9.1]
    for left, label in zip(positions, labels):
        ax.add_patch(FancyBboxPatch((left, 0.8), 2.4, 0.85, boxstyle="round,pad=0.03",
                                    fc="#F1F7F8", ec="#D7E2E9"))
        ax.text(left + 1.2, 1.225, label, ha="center", va="center", color=navy, fontsize=11)
    for i, local in enumerate(["×3", "×1", "×(-6)"]):
        start, end = positions[i] + 2.45, positions[i+1] - 0.05
        ax.annotate("", (end, 1.23), (start, 1.23),
                    arrowprops=dict(arrowstyle="->", color=teal, lw=1.6))
        ax.annotate("", (positions[i]+1.2, 0.32), (positions[i+1]+1.2, 0.32),
                    arrowprops=dict(arrowstyle="->", color=orange, lw=1.8))
        ax.text((positions[i]+positions[i+1])/2 + 1.2, 0.05, local,
                color=orange, ha="center", fontsize=11)
    ax.text(6, 2.0, "向右计算函数值；沿路径向左传递变化率", ha="center", color=navy, fontsize=12)
    ax.text(6, -0.5, "dL/dw = (-6) × 1 × 3 = -18", ha="center", color=orange, fontsize=13)
    save(fig, "2_4_chain_rule.png")


make_figures()
print("\n整节结束：综合练习集中填写八项，再一起反馈输出和三个解释。")
