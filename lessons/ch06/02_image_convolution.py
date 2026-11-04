"""6.2 图像卷积。

先在 window_demo 手算一个输出，再读 corr2d 的切片和逐元素乘加。
Conv2D 将核注册为可学习参数，edge_demo 给出差分检测的目标。
learn_kernel 串起前向、损失、清梯度、反传和 SGD 更新。
receptive_field_demo 最后把一层窗口衔接到两层网络。

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
STEPS = 36
LR = .03


def corr2d(X, K):
    """单张二维图像，步幅 1，只计算完整窗口。"""
    height, width = K.shape
    rows = X.shape[0] - height + 1
    cols = X.shape[1] - width + 1
    if not (height > 0 and width > 0 and rows > 0 and cols > 0):
        raise ValueError("核须非空且能放入图像")
    outputs = []
    for row in range(rows):
        line = []
        for col in range(cols):
            patch = X[row:row + height, col:col + width]
            value = (patch * K).sum()
            line.append(value)
        outputs.append(torch.stack(line))
    # 内层 stack 拼接一行，外层 stack 拼成整张响应图，计算图保持连通。
    return torch.stack(outputs)


class Conv2D(nn.Module):
    """一个核和一个共享标量偏置，先登记参数，再在 forward 中使用。"""
    def __init__(self, kernel_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size, dtype=DTYPE))
        self.bias = nn.Parameter(torch.zeros((), dtype=DTYPE)) if bias else None

    def forward(self, X):
        response = corr2d(X, self.weight)
        return response + self.bias if self.bias is not None else response


def window_demo():
    X = torch.arange(9, dtype=DTYPE).reshape(3, 3)
    K = torch.arange(4, dtype=DTYPE).reshape(2, 2)
    Y = corr2d(X, K)
    torch.testing.assert_close(Y, torch.tensor([[19., 25.], [37., 43.]], dtype=DTYPE))
    columns = F.unfold(X[None, None], K.shape)[0]
    product = (K.reshape(1, -1) @ columns).reshape(Y.shape)
    flipped_kernel = K.flip((0, 1))
    flipped = corr2d(X, flipped_kernel)
    torch.testing.assert_close(product, Y)
    torch.testing.assert_close(corr2d(X, flipped_kernel.flip((0, 1))), Y)

    # 核、偏置、输入三路梯度在相同前向点对照。
    hand = Conv2D((2, 2))
    builtin = nn.Conv2d(1, 1, 2, dtype=DTYPE)
    with torch.no_grad():
        hand.weight.copy_(K)
        hand.bias.fill_(.5)
        builtin.weight.copy_(K[None, None])
        builtin.bias.fill_(.5)
    x_hand = X.clone().requires_grad_()
    x_builtin = X.clone().requires_grad_()
    first = hand(x_hand)
    second = builtin(x_builtin[None, None])[0, 0]
    first.sum().backward()
    second.sum().backward()
    differences = {
        "output": (first - second).abs().max().item(),
        "kernel_gradient": (hand.weight.grad - builtin.weight.grad[0, 0]).abs().max().item(),
        "bias_gradient": (hand.bias.grad - builtin.bias.grad[0]).abs().item(),
        "input_gradient": (x_hand.grad - x_builtin.grad).abs().max().item(),
    }
    assert max(differences.values()) == 0

    # 教材的逐格赋值在本机生成 CopySlices，仍能反传。
    legacy_K = K.clone().requires_grad_()
    legacy = X.new_zeros(Y.shape)
    for row in range(Y.shape[0]):
        for col in range(Y.shape[1]):
            legacy[row, col] = (X[row:row + 2, col:col + 2] * legacy_K).sum()
    legacy.sum().backward()
    torch.testing.assert_close(legacy_K.grad, hand.weight.grad)
    # 重新用 torch.tensor 封装数值则切断计算图。
    detached = torch.tensor(first.detach().tolist(), dtype=DTYPE)
    try:
        detached.sum().backward()
    except RuntimeError as error:
        detach_error = str(error)
    else:
        raise AssertionError("预期脱离计算图的输出无法反传")

    return {
        "response": Y.tolist(), "flipped_response": flipped.tolist(),
        "window_matrix": columns.tolist(), "hand_vs_builtin": differences,
        "kernel_gradient_for_sum": hand.weight.grad.tolist(),
        "bias_gradient_for_sum": hand.bias.grad.item(),
        "assignment_grad_fn": type(legacy.grad_fn).__name__,
        "detached_error": detach_error,
    }, (X, K, Y)


def edge_demo():
    X = torch.ones(6, 8, dtype=DTYPE)
    X[:, 2:6] = 0
    K = torch.tensor([[1., -1.]], dtype=DTYPE)
    target = corr2d(X, K)
    rotated_input = X.T
    mismatched = corr2d(rotated_input, K)
    matching = corr2d(rotated_input, K.T)
    torch.testing.assert_close(matching, target.T)
    assert mismatched.count_nonzero() == 0

    axis = torch.arange(6)
    diagonal = (axis[None, :] >= axis[:, None]).to(DTYPE)
    diag_x = corr2d(diagonal, K)
    diag_transposed = corr2d(diagonal.T, K)
    diag_y = corr2d(diagonal, K.T)
    torch.testing.assert_close(diag_y, diag_transposed.T)
    return {
        "stripe_response_first_row": target[0].tolist(),
        "transposed_stripe_wrong_kernel_nonzero": mismatched.count_nonzero().item(),
        "diagonal_horizontal_values": sorted(diag_x[diag_x != 0].unique().tolist()),
        "diagonal_transposed_values": sorted(diag_transposed[diag_transposed != 0].unique().tolist()),
        "diagonal_vertical_values": sorted(diag_y[diag_y != 0].unique().tolist()),
        "diagonal_nonzero_counts": [v.count_nonzero().item() for v in (diag_x, diag_transposed, diag_y)],
    }, (X, K, target, mismatched, matching)


def learn_kernel(X, K, target):
    """两个实现复制同一初值，用同一目标和损失，逐步核对梯度与更新。"""
    hand = Conv2D((1, 2), bias=False)
    builtin = nn.Conv2d(1, 1, (1, 2), bias=False, dtype=DTYPE)
    with torch.no_grad():
        builtin.weight.copy_(hand.weight[None, None])
    opt_hand = torch.optim.SGD(hand.parameters(), lr=LR)
    opt_builtin = torch.optim.SGD(builtin.parameters(), lr=LR)
    initial = hand.weight.detach().clone()
    history = []
    max_grad_error = 0.
    max_weight_error = 0.
    for step in range(STEPS + 1):
        left = (hand(X) - target).square().sum()
        right = (builtin(X[None, None])[0, 0] - target).square().sum()
        torch.testing.assert_close(left, right)
        history.append(left.item())
        if step == STEPS:
            break
        opt_hand.zero_grad(set_to_none=True)
        opt_builtin.zero_grad(set_to_none=True)
        # backward 写入梯度，step 才更新参数。每步先清除上一步梯度。
        left.backward()
        right.backward()
        max_grad_error = max(max_grad_error, (hand.weight.grad - builtin.weight.grad[0, 0]).abs().max().item())
        opt_hand.step()
        opt_builtin.step()
        max_weight_error = max(max_weight_error, (hand.weight - builtin.weight[0, 0]).abs().max().item())

    with torch.no_grad():
        test = torch.tensor([[.2, .85, .85, .4, .2]], dtype=DTYPE)
        test_error = (hand(test) - corr2d(test, K)).abs().max().item()
        kernel_error = (hand.weight - K).abs().max().item()
    assert max_grad_error < 1e-12 and max_weight_error < 1e-12
    assert history[-1] < 1e-4 and test_error < .003
    return {
        "updates": STEPS, "learning_rate": LR, "loss": "sum_squared_error",
        "initial_kernel": initial.flatten().tolist(),
        "learned_kernel": hand.weight.detach().flatten().tolist(),
        "max_kernel_error": kernel_error, "max_new_row_error": test_error,
        "sse_by_step": history, "max_gradient_difference": max_grad_error,
        "max_weight_difference": max_weight_error,
    }


def receptive_field_demo():
    X = torch.arange(9, dtype=DTYPE).reshape(3, 3).requires_grad_()
    K = torch.ones(2, 2, dtype=DTYPE)
    hidden = corr2d(X, K)
    grad_one, = torch.autograd.grad(hidden[0, 0], X, retain_graph=True)
    grad_two, = torch.autograd.grad(corr2d(hidden, K).sum(), X)
    return {
        "one_layer_input_gradient": grad_one.tolist(),
        "two_layer_input_gradient": grad_two.tolist(),
    }, (grad_one, grad_two)


def record(metrics, windows, edges, fields):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out = REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (REPORTS / "6_2_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.05), layout="constrained")
    for ax, values, title, color in zip(
        axes, windows, ("输入 X", "卷积核 K", "输出 Y"), ("#e0eef0", "#f8ead5", "#e0e8f6")
    ):
        ax.imshow(torch.zeros_like(values), cmap=matplotlib.colors.ListedColormap([color]))
        ax.set_title(title, fontsize=12, pad=10)
        ax.set(xticks=[], yticks=[])
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, fill=False, edgecolor="white"))
                ax.text(col, row, f"{values[row, col]:.0f}", ha="center", va="center", fontsize=15, color="#183047")
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax, size in zip(axes, (2, 2, 1)):
        ax.add_patch(Rectangle((-.5, -.5), size, size, fill=False, edgecolor="#b85b1b", lw=2.7, clip_on=False))
    fig.savefig(out / "6_2_windows.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.7), layout="constrained")
    X, _, target, wrong, matched = edges
    entries = [
        (X, "条纹输入", "gray", 0), (target, "横着做差，找到竖边", "RdBu_r", -1),
        (wrong, "转置图像，仍横着做差", "RdBu_r", -1), (matched, "转置图像，改为竖着做差", "RdBu_r", -1),
    ]
    for ax, (values, title, cmap, low) in zip(axes.flat, entries):
        chart = ax.imshow(values.numpy(), cmap=cmap, vmin=low, vmax=1)
        ax.set(title=title, xticks=[], yticks=[])
    fig.colorbar(chart, ax=axes[:, 1], ticks=[-1, 0, 1], shrink=.82, label="响应")
    fig.savefig(out / "6_2_edges.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.7, 3.45), layout="constrained")
    sse = metrics["learning"]["sse_by_step"]
    ax.semilogy(range(len(sse)), sse, color="#177d85", lw=2.5)
    ax.scatter([0, 10, STEPS], [sse[0], sse[10], sse[-1]], color="#b85b1b", zorder=3)
    ax.set(xlabel="更新次数", ylabel="平方误差和 SSE（对数轴）", title="让条纹图的预测响应逐渐贴近目标")
    ax.grid(alpha=.2)
    fig.savefig(out / "6_2_fit.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), layout="constrained")
    for ax, values, title in zip(axes, fields, ("一层：左上输出", "两层：唯一输出")):
        ax.imshow(values.numpy(), cmap="Blues", vmin=0, vmax=4)
        ax.set(title=title, xticks=range(3), yticks=range(3))
        for row in range(3):
            for col in range(3):
                value = values[row, col].item()
                ax.text(col, row, f"{value:.0f}", ha="center", va="center",
                        color="white" if value >= 3 else "#183047", fontsize=14)
    fig.savefig(out / "6_2_receptive.png", dpi=170)
    plt.close(fig)


def print_guided(metrics):
    """把手算、边缘、训练和感受野连续展示。"""
    scan, edges = metrics["scan"], metrics["edges"]
    learning, field = metrics["learning"], metrics["receptive_field"]
    print("\n6.2 图像卷积\n")
    print("步骤 1 先手算第一个窗口，再核对整张输出")
    print("响应", scan["response"])
    print("翻核后响应", scan["flipped_response"])
    print("手写与内置层的前向和三路梯度差", scan["hand_vs_builtin"])
    print("核梯度会汇总所有使用同一参数的窗口。\n")
    print("步骤 2 用相邻像素之差寻找亮度变化")
    print("条纹响应第一行", edges["stripe_response_first_row"])
    print("转置输入后继续横向做差的非零数", edges["transposed_stripe_wrong_kernel_nonzero"])
    print("水平亮度变化对应竖直延伸的边缘。\n")
    print("步骤 3 用目标响应反过来学习核")
    print("初始核", learning["initial_kernel"])
    for step in (0, 10, learning["updates"]):
        print("更新次数", step, "SSE", learning["sse_by_step"][step])
    print("最终核", learning["learned_kernel"])
    print("新灰度序列的最大响应误差", learning["max_new_row_error"])
    print("前向预测后计算 SSE，再清梯度、反传、更新，共执行 36 次。\n")
    print("步骤 4 将两个局部窗口层叠起来")
    print("两层对原始输入的梯度", field["two_layer_input_gradient"])
    print("3×3 区域共同影响最终输出，中心像素通过四条路径贡献。")
    print("接着按讲义改变输入方向，再打开 exercises/ch06/02_image_convolution.py。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="更新实测 JSON 和四张图")
    parser.add_argument("--guided", action="store_true", help="按讲义步骤显示中文结果和解释")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(6201)
    scan, windows = window_demo()
    edges, edge_images = edge_demo()
    fields, field_images = receptive_field_demo()
    metrics = {
        "section": "6.2", "date": "2026-11-04", "torch": torch.__version__,
        "device": "cpu", "dtype": str(DTYPE), "seed": 6201, "scan": scan,
        "edges": edges, "learning": learn_kernel(*edge_images[:3]), "receptive_field": fields,
    }
    if args.record:
        record(metrics, windows, edge_images, field_images)
    if args.guided:
        print_guided(metrics)
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
