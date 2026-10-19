"""完整 2.5 自动微分：记录计算 -> 反向传播 -> 查看/清理梯度 -> 更新。

原教材 PDF 第 87-91 页，正文第 69-73 页（含节末练习）。
复用已有 openmmlab，一次运行整节，不安装、不下载。
"""

import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

FIGURES = ROOT / "reports" / "ch02" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
torch.set_printoptions(precision=6, sci_mode=False)


def grad_of(tensor):
    """统一检查，避免对尚未产生的 None 梯度调用张量方法。"""
    if tensor.grad is None:
        raise RuntimeError("这个叶张量还没有梯度，请检查是否记录并执行了反向传播。")
    return tensor.grad


print("===== 1. 从上一节的手算导数开始 =====")
x = torch.arange(4.0, dtype=torch.float64)
x.requires_grad_(True)
print("x:", x)
print("反向前 x.grad:", x.grad)
y = 2 * torch.dot(x, x)
print("y=2*x点积x:", y.item(), "; x.is_leaf:", x.is_leaf)
print("y.requires_grad / y.is_leaf:", y.requires_grad, y.is_leaf)
print("y 的反向操作:", type(y.grad_fn).__name__)
y.backward()
basic_gradient = grad_of(x).clone()
print("自动梯度:", basic_gradient, "; 手算 4*x:", 4 * x.detach())
print("backward 后 x 的数值仍是:", x.detach())

print("\n===== 2. 新前向仍会累加梯度；清理后只保留本次贡献 =====")
x.grad = None
(x * x).sum().backward()
first_gradient = grad_of(x).clone()
(x * x).sum().backward()  # 新建计算图，但写入同一份 x.grad。
accumulated_gradient = grad_of(x).clone()
print("第一次 2*x:", first_gradient)
print("两次相加 4*x:", accumulated_gradient)
grad_of(x).zero_()
x.sum().backward()
print("清零后对 sum(x) 求导:", grad_of(x))
x.grad = None
print("另一种清理方式 x.grad=None:", x.grad)

print("\n===== 3. 同一张图重复反向，与梯度累加是两回事 =====")
a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
square = a * a
square.backward()
repeated_backward_failed = False
try:
    square.backward()
except RuntimeError:
    repeated_backward_failed = True
    print("预期错误：a*a 的中间缓存已释放；重新前向计算再 backward。")
a.grad = None
square = a * a
square.backward()
print("重新前向后的梯度:", grad_of(a).item())

print("\n===== 4. 多元素输出需要先说明怎样汇总 =====")
sum_x = torch.tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
(sum_x * sum_x).sum().backward()
sum_gradient = grad_of(sum_x).clone()
mean_x = sum_x.detach().clone().requires_grad_(True)
(mean_x * mean_x).mean().backward()
mean_gradient = grad_of(mean_x).clone()
weighted_x = sum_x.detach().clone().requires_grad_(True)
seed = torch.tensor([1., 2., 3.], dtype=torch.float64)
weighted_y = weighted_x * weighted_x
weighted_y.backward(seed)
weighted_gradient = grad_of(weighted_x).clone()
print("sum 的梯度:", sum_gradient)
print("mean 的梯度:", mean_gradient)
print("权重 [1,2,3] 的加权和梯度:", weighted_gradient)
print("backward(seed) 得到 Jacobian.T@seed 对应的梯度，不是完整 Jacobian。")
coupled_x = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
coupled_y = torch.stack([coupled_x[0] + coupled_x[1], coupled_x[0] * coupled_x[1]])
coupled_y.backward(torch.tensor([2., 3.], dtype=torch.float64))
print("耦合输出 [x0+x1,x0*x1] 加权 [2,3] 的梯度:", grad_of(coupled_x))

print("\n===== 5. detach：保持数值，切断一条求导路径 =====")
cut_x = torch.tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
cut_y = cut_x * cut_x
constant = cut_y.detach()
cut_z = constant * cut_x
cut_z.sum().backward()
detached_gradient = grad_of(cut_x).clone()
print("分离 y=x*x 后，z=y.detach()*x 的梯度:", detached_gradient)
cut_x.grad = None
cut_y.sum().backward()  # 上一次没有沿 constant 回到 cut_y 的原图。
print("原来的 y=x*x 仍可单独求导:", grad_of(cut_x))
full_x = cut_x.detach().clone().requires_grad_(True)
((full_x * full_x) * full_x).sum().backward()
full_gradient = grad_of(full_x).clone()
print("未分离 x*x*x 的梯度:", full_gradient)
print("detach 与原 y 共享存储:", constant.data_ptr() == cut_y.data_ptr())
independent = cut_y.detach().clone()
print("detach().clone() 获得独立存储:", independent.data_ptr() != cut_y.data_ptr())


def controlled(a):
    """保留教材控制流，补上全零输入保护，避免循环永不结束。"""
    if a.detach().norm().item() == 0:
        raise ValueError("教材的倍增循环不能从全零输入开始。")
    b = a * 2
    factor = 2
    loops = 0
    while b.norm().item() < 1000:
        b = b * 2
        factor *= 2
        loops += 1
    if b.sum().item() > 0:
        c = b
    else:
        c = 100 * b
        factor *= 100
    return c, factor, loops


print("\n===== 6. 控制流：记录本次实际执行的张量计算 =====")
control_records = []
for value in [1.25, -1.25]:
    control_a = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    result, factor, loops = controlled(control_a)
    result.backward()
    actual = grad_of(control_a).item()
    control_records.append((value, factor, actual))
    print("a=%g，循环=%d，系数=%d，结果=%g，梯度=%g" %
          (value, loops, factor, result.item(), actual))
matrix_a = torch.tensor([[1., 0.], [-2., 3.]], dtype=torch.float64, requires_grad=True)
matrix_result, matrix_factor, _ = controlled(matrix_a)
matrix_result.sum().backward()
matrix_gradient = grad_of(matrix_a).clone()
print("矩阵输入先对输出求和，梯度:\n", matrix_gradient)
print("全部等于本路径系数:", torch.equal(matrix_gradient, torch.full_like(matrix_a, matrix_factor)))
print("含零元素时 result/a 会出现 0/0，不能用比值普遍验证。")
alternative_records = []
for value in [2.0, -2.0]:
    point = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    result = point.square() if point.item() > 0 else 3 * point
    result.backward()
    alternative_records.append((value, grad_of(point).item()))
print("自定义正分支 a²、非正分支 3a，在 2/-2 的梯度:", alternative_records)

print("\n===== 7. 二阶导数：让一阶导数的计算也进入图 =====")
q = torch.tensor(2., dtype=torch.float64, requires_grad=True)
value = q ** 4
first_derivative = torch.autograd.grad(value, q, create_graph=True)[0]
second_derivative = torch.autograd.grad(first_derivative, q)[0]
print("q^4 在 q=2 的一阶/二阶导数:", first_derivative.item(), second_derivative.item())
print("autograd.grad 返回导数；q.grad 仍为:", q.grad)

print("\n===== 8. 回到上一节：自动求导完成同一次小批量更新 =====")
X = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
w = torch.tensor([[0.5], [-1.]], dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
target = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
prediction = X @ w + b
error = prediction - target
loss = error.square().mean()
loss.backward()
auto_dw = grad_of(w).clone()
auto_db = grad_of(b).clone()
manual_dw = 2 / X.shape[0] * (X.T @ error.detach())
manual_db = 2 * error.detach().mean()
print("自动 / 手算 dw:\n", auto_dw, "\n", manual_dw, sep="")
print("自动 / 手算 db:", auto_db.item(), manual_db.item())
lr = 0.01
with torch.no_grad():
    w -= lr * grad_of(w)
    b -= lr * grad_of(b)
    after_loss = (X @ w + b - target).square().mean()
print("更新前/后损失:", loss.item(), after_loss.item())
print("更新后的参数仍是可求导叶张量:", w.is_leaf, w.requires_grad, b.is_leaf)
w.grad = None
b.grad = None
print("为下一轮清理梯度:", w.grad, b.grad)


def make_figures():
    """绘图辅助区；sin 的导数数据由 autograd 产生。"""
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "savefig.facecolor": "white"})
    teal, orange, navy = "#147E83", "#D07632", "#173448"

    def save(fig, name):
        fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print("配图:", name)

    grid = torch.linspace(-2 * math.pi, 2 * math.pi, 400, dtype=torch.float64,
                          requires_grad=True)
    sine = torch.sin(grid)
    sine_gradient = torch.autograd.grad(sine.sum(), grid)[0]
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.plot(grid.detach().numpy(), sine.detach().numpy(), color=teal, label="sin(x)")
    ax.plot(grid.detach().numpy(), sine_gradient.numpy(), "--", color=orange,
            label="通过自动微分计算的导数")
    ax.set(xlabel="x", ylabel="函数值 / 导数", title="sin(x) 与自动微分得到的斜率")
    ax.legend(frameon=True, facecolor="white", edgecolor="#D7E2E9", framealpha=1,
              loc="upper right", fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    save(fig, "2_5_sin_autograd.png")

    fig, ax = plt.subplots(figsize=(9.6, 3.25))
    ax.set(xlim=(-0.3, 10), ylim=(-1.3, 2.6))
    ax.axis("off")
    nodes = [(0, 0, "x"), (3.4, 1, "y=x²"), (6.8, 1, "u=y.detach()"), (6.8, -0.4, "z=u*x")]
    for left, bottom, label in nodes:
        ax.add_patch(FancyBboxPatch((left, bottom), 2.6, 0.7, boxstyle="round,pad=0.04",
                                    fc="#F1F7F8", ec="#D7E2E9"))
        ax.text(left+1.3, bottom+0.35, label, ha="center", va="center", color=navy, fontsize=12)
    ax.annotate("", (3.35, 1.35), (2.65, 0.35), arrowprops=dict(arrowstyle="->", color=teal))
    ax.annotate("", (6.75, 1.35), (6.05, 1.35), arrowprops=dict(arrowstyle="->", color=teal))
    ax.annotate("", (8.1, 0.35), (8.1, 0.95), arrowprops=dict(arrowstyle="->", color=teal))
    ax.annotate("", (6.75, -0.05), (2.65, 0.15), arrowprops=dict(arrowstyle="->", color=orange, lw=2))
    ax.text(6.4, 1.9, "数值保留；反向路径在此切断", color=orange, ha="center", fontsize=10)
    ax.text(4.5, -0.55, "仍通过直接的 x 路径求导", color=orange, ha="center", fontsize=10)
    ax.text(4.7, -1.05, "不分离：梯度 3x²    |    分离 y：梯度 x²", ha="center", color=navy, fontsize=12)
    save(fig, "2_5_detach_paths.png")

    fig, ax = plt.subplots(figsize=(9.6, 2.7))
    ax.set(xlim=(-0.2, 12), ylim=(-0.8, 2.0))
    ax.axis("off")
    labels = ["清理旧梯度", "前向计算损失", "backward 求梯度", "no_grad 中更新"]
    for i, label in enumerate(labels):
        left = i * 3
        ax.add_patch(FancyBboxPatch((left, 0.65), 2.55, 0.7, boxstyle="round,pad=0.03",
                                    fc="#F1F7F8", ec="#D7E2E9"))
        ax.text(left+1.275, 1.0, label, ha="center", va="center", fontsize=11, color=navy)
        if i < 3:
            ax.annotate("", (left+2.95, 1), (left+2.6, 1),
                        arrowprops=dict(arrowstyle="->", color=teal, lw=1.8))
    ax.text(5.9, 1.65, "每轮重新前向，建立本轮计算图", ha="center", color=navy, fontsize=12)
    ax.text(5.9, -0.05, "backward 只计算梯度；参数更新由我们或优化器执行", ha="center", color=orange, fontsize=12)
    save(fig, "2_5_update_cycle.png")
    return grid.detach(), sine_gradient


sine_grid, sine_auto_gradient = make_figures()
print("\n整节结束：去综合练习填写八处 TODO，再一次反馈结果与三个解释。")
