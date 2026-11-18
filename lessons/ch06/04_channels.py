"""6.4 多输入多输出通道。

先对每个输入通道做 6.3 的互相关，再求和得到一个输出通道。
多组这样的核沿输出通道轴叠起来，构成完整卷积层。
随后核对 1×1 矩阵乘法、unfold、线性层合并以及一个小型通道映射训练。
--guided 显示中文步骤，--record 保存实测 JSON 和四张图。
"""
import argparse
import json
import os
import runpy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/ch06"
DTYPE = torch.float64
# 读取已有的纯张量函数，原文件的 main 入口在这里不会执行。
corr2d = runpy.run_path(str(ROOT / "lessons/ch06/03_padding_and_stride.py"))["corr2d"]


def corr2d_multi_in(X, K, padding=(0, 0, 0, 0), stride=(1, 1)):
    """CHW 输入配同样输入通道数的核，通道结果相加为 HW。"""
    if X.ndim != 3 or K.ndim != 3 or X.shape[0] != K.shape[0] or X.shape[0] == 0:
        raise ValueError("输入和核应为非空 CHW，输入通道数量必须相同")
    contributions = []
    for channel in range(X.shape[0]):
        contributions.append(corr2d(X[channel], K[channel], padding, stride))
    return torch.stack(contributions).sum(dim=0)


def corr2d_multi_in_out(X, K, bias=None, padding=(0, 0, 0, 0), stride=(1, 1)):
    """每个输出通道使用一整组输入通道核，最后沿新轴叠放。"""
    if K.ndim != 4 or K.shape[0] == 0:
        raise ValueError("核应为非空 OIHW")
    if bias is not None and tuple(bias.shape) != (K.shape[0],):
        raise ValueError("偏置应为每个输出通道一个值")
    outputs = []
    for output_channel in range(K.shape[0]):
        response = corr2d_multi_in(X, K[output_channel], padding, stride)
        if bias is not None:
            response = response + bias[output_channel]
        outputs.append(response)
    return torch.stack(outputs)


def textbook_demo():
    X = torch.stack((torch.arange(9, dtype=DTYPE).reshape(3, 3),
                     torch.arange(1, 10, dtype=DTYPE).reshape(3, 3)))
    K = torch.stack((torch.arange(4, dtype=DTYPE).reshape(2, 2),
                     torch.arange(1, 5, dtype=DTYPE).reshape(2, 2)))
    parts = torch.stack([corr2d(X[channel], K[channel]) for channel in range(2)])
    single = corr2d_multi_in(X, K)
    kernels = torch.stack((K, K + 1, K + 2))
    multiple = corr2d_multi_in_out(X, kernels)
    expected = torch.tensor([
        [[56., 72.], [104., 120.]], [[76., 100.], [148., 172.]],
        [[96., 128.], [192., 224.]],
    ], dtype=DTYPE)
    torch.testing.assert_close(multiple, expected)
    builtin = F.conv2d(X[None], kernels)[0]
    torch.testing.assert_close(multiple, builtin)
    bias = torch.tensor([.5, -1., 2.], dtype=DTYPE)
    torch.testing.assert_close(corr2d_multi_in_out(X, kernels, bias), builtin + bias[:, None, None])
    return {
        "input_shape": list(X.shape), "kernel_shape": list(kernels.shape),
        "channel_contributions": parts.tolist(), "single_output": single.tolist(),
        "multiple_output": multiple.tolist(), "bias": bias.tolist(),
        "parameter_count_with_bias": kernels.numel() + bias.numel(),
    }, (parts, single, multiple)


def gradient_demo():
    generator = torch.Generator().manual_seed(6401)
    values = [torch.randn(shape, generator=generator, dtype=DTYPE)
              for shape in ((2, 3, 5, 6), (4, 3, 2, 3), (4,))]
    left = [value.clone().requires_grad_() for value in values]
    right = [value.clone().requires_grad_() for value in values]
    padding, stride = (1, 0, 0, 1), (2, 1)
    hand = torch.stack([corr2d_multi_in_out(image, left[1], left[2], padding, stride)
                        for image in left[0]])
    builtin = F.conv2d(F.pad(right[0], padding), right[1], right[2], stride=stride)
    hand.square().mean().backward()
    builtin.square().mean().backward()
    errors = {"output": (hand - builtin).abs().max().item()}
    for name, first, second in zip(("input", "weight", "bias"), left, right):
        errors[name + "_gradient"] = (first.grad - second.grad).abs().max().item()
    assert max(errors.values()) < 1e-12
    return errors


def pointwise_demo():
    generator = torch.Generator().manual_seed(6402)
    X = torch.randn(3, 3, 4, generator=generator, dtype=DTYPE)
    K = torch.randn(2, 3, 1, 1, generator=generator, dtype=DTYPE)
    bias = torch.tensor([.2, -.3], dtype=DTYPE)
    matrix = K.reshape(2, 3) @ X.reshape(3, -1) + bias[:, None]
    hand = corr2d_multi_in_out(X, K, bias)
    builtin = F.conv2d(X[None], K, bias)[0]
    linear = F.linear(X.movedim(0, -1), K[:, :, 0, 0], bias).movedim(-1, 0)
    errors = {
        "matrix": (matrix.reshape_as(hand) - builtin).abs().max().item(),
        "windows": (hand - builtin).abs().max().item(),
        "linear": (linear - builtin).abs().max().item(),
    }
    assert max(errors.values()) < 1e-12
    return {"input_shape": list(X.shape), "matrix_shape": [2, 3],
            "position_columns": list(X.reshape(3, -1).shape), "errors": errors}, K[:, :, 0, 0]


def matrix_and_composition_demo():
    generator = torch.Generator().manual_seed(6403)
    X = torch.randn(2, 3, 5, 6, generator=generator, dtype=DTYPE)
    K = torch.randn(4, 3, 2, 3, generator=generator, dtype=DTYPE)
    bias = torch.randn(4, generator=generator, dtype=DTYPE)
    columns = F.unfold(X, kernel_size=(2, 3))
    product = K.reshape(4, -1) @ columns + bias[None, :, None]
    builtin = F.conv2d(X, K, bias)
    torch.testing.assert_close(product.reshape_as(builtin), builtin)

    # 两层无偏置、无激活、步幅 1、valid 卷积合并为一个 3×3 核。
    first = torch.randn(3, 2, 2, 2, generator=generator, dtype=DTYPE)
    second = torch.randn(2, 3, 2, 2, generator=generator, dtype=DTYPE)
    image = torch.randn(1, 2, 6, 7, generator=generator, dtype=DTYPE)
    combined = torch.zeros(2, 2, 3, 3, dtype=DTYPE)
    for a in range(2):
        for b in range(2):
            # 对中间通道求和，第一层核按第二层的空间偏移累加。
            combined[:, :, a:a + 2, b:b + 2] += torch.einsum(
                "om,mihw->oihw", second[:, :, a, b], first
            )
    two_layers = F.conv2d(F.conv2d(image, first), second)
    one_layer = F.conv2d(image, combined)
    composition_error = (two_layers - one_layer).abs().max().item()
    assert composition_error < 1e-12
    # 1 个中间通道的 1×1 两层，其等效通道矩阵秩最多为 1。
    bottleneck = torch.tensor([[1.], [2.]], dtype=DTYPE) @ torch.tensor([[3., 4.]], dtype=DTYPE)
    return {
        "unfold_shape": list(columns.shape), "output_shape": list(builtin.shape),
        "matrix_max_error": (product.reshape_as(builtin) - builtin).abs().max().item(),
        "combined_kernel_shape": list(combined.shape), "composition_max_error": composition_error,
        "rank_one_bottleneck": torch.linalg.matrix_rank(bottleneck).item(),
        "identity_rank": torch.linalg.matrix_rank(torch.eye(2, dtype=DTYPE)).item(),
        "multiplications": builtin.numel() * K[0].numel(),
        "additions_without_bias": builtin.numel() * (K[0].numel() - 1),
        "bias_additions": builtin.numel(),
        "input_weight_bias_output_bytes": (X.numel() + K.numel() + bias.numel() + builtin.numel()) * X.element_size(),
        "unfold_temporary_bytes": columns.numel() * columns.element_size(),
        "doubling_both_channels_main_cost_factor": 4,
        "padding_1_to_2_output_area_ratio_8x8_k3_s1": 100 / 64,
    }


def train_pointwise():
    generator = torch.Generator().manual_seed(6404)
    X = torch.randn(24, 3, 3, 4, generator=generator, dtype=DTYPE)
    validation = torch.randn(8, 3, 3, 4, generator=generator, dtype=DTYPE)
    true_weight = torch.tensor([[1., -.5, .25], [-.75, .5, 1.25]], dtype=DTYPE)[:, :, None, None]
    true_bias = torch.tensor([.1, -.2], dtype=DTYPE)
    target = F.conv2d(X, true_weight, true_bias)
    validation_target = F.conv2d(validation, true_weight, true_bias)
    model = nn.Conv2d(3, 2, 1, dtype=DTYPE)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.15)
    losses = []
    for step in range(121):
        loss = (model(X) - target).square().mean()
        losses.append(loss.item())
        if step == 120:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        validation_mse = (model(validation) - validation_target).square().mean().item()
        error = (model.weight - true_weight).abs().max().item()
    assert losses[-1] < 1e-10 and validation_mse < 1e-10
    return {"seed": 6404, "train_images": 24, "validation_images": 8,
            "image_shape": [3, 3, 4], "parameters": 8, "updates": 120, "learning_rate": .15,
            "loss": "mean_squared_error", "train_mse_by_step": losses,
            "validation_mse": validation_mse, "max_weight_error": error,
            "learned_weight": model.weight.detach()[:, :, 0, 0].tolist(),
            "learned_bias": model.bias.detach().tolist()}


def record(metrics, textbook_images, pointwise_matrix):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (REPORTS / "6_4_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})

    def grid(ax, values, title):
        ax.imshow(values.detach().numpy(), cmap="Blues")
        ax.set(title=title, xticks=[], yticks=[])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i,j]:.0f}", ha="center", va="center", fontsize=13,
                        color="white" if values[i,j] > values.max() * .7 else "#173448")

    parts, summed, multiple = textbook_images
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), layout="constrained")
    for ax, value, title in zip(axes, (parts[0], parts[1], summed), ("输入通道 0 的贡献", "输入通道 1 的贡献", "相加得到一个输出通道")):
        grid(ax, value, title)
    fig.savefig(out / "6_4_channel_sum.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), layout="constrained")
    for channel, ax in enumerate(axes):
        grid(ax, multiple[channel], f"输出通道 {channel}")
    fig.savefig(out / "6_4_output_maps.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 3.0), layout="constrained")
    matrix = pointwise_matrix
    ax.imshow(matrix.numpy(), cmap="RdBu_r", vmin=-2, vmax=2)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=14)
    ax.set(xticks=range(3), yticks=range(2), xlabel="输入通道", ylabel="输出通道",
           title="同一个 2×3 通道矩阵用于全部 12 个空间位置")
    fig.savefig(out / "6_4_pointwise.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 3.0), layout="constrained")
    ax.semilogy(metrics["training"]["train_mse_by_step"], color="#147e83", lw=2.3)
    ax.set(xlabel="更新次数", ylabel="训练 MSE（对数轴）", title="学习 3 通道到 2 通道的共享 1×1 映射")
    ax.grid(alpha=.2)
    fig.savefig(out / "6_4_training.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guided", action="store_true", help="按中文步骤讲解结果")
    parser.add_argument("--record", action="store_true", help="更新实测 JSON 和四图")
    args = parser.parse_args()
    torch.set_num_threads(1)
    textbook, images = textbook_demo()
    pointwise, pointwise_matrix = pointwise_demo()
    metrics = {"section": "6.4", "date": "2026-11-18", "torch": torch.__version__,
               "device": "cpu", "dtype": str(DTYPE), "textbook": textbook,
               "gradient_errors": gradient_demo(), "pointwise": pointwise,
               "matrix_and_composition": matrix_and_composition_demo(), "training": train_pointwise()}
    if args.record:
        record(metrics, images, pointwise_matrix)
    if args.guided:
        print("6.4 多输入多输出通道\n步骤 1 每个输入通道分别计算，再相加")
        print("两个通道的首格贡献", [plane[0][0] for plane in textbook["channel_contributions"]])
        print("单输出响应", textbook["single_output"])
        print("步骤 2 多组核沿输出通道轴叠放")
        print("核形状", textbook["kernel_shape"], "含偏置参数数", textbook["parameter_count_with_bias"])
        print("多输出响应", textbook["multiple_output"])
        print("步骤 3 将 1×1 和一般卷积写成矩阵运算")
        print("1×1 三种写法误差", pointwise["errors"])
        print("前向与三路梯度误差", metrics["gradient_errors"])
        print("unfold 形状", metrics["matrix_and_composition"]["unfold_shape"])
        print("两层合并核形状", metrics["matrix_and_composition"]["combined_kernel_shape"])
        analysis = metrics["matrix_and_composition"]
        print("合并输出最大差", analysis["composition_max_error"])
        print("前向乘法/窗口加法/偏置加法", analysis["multiplications"],
              analysis["additions_without_bias"], analysis["bias_additions"])
        print("瓶颈矩阵秩/单位矩阵秩", analysis["rank_one_bottleneck"], analysis["identity_rank"])
        print("步骤 4 从零学习一个通道混合规则")
        training = metrics["training"]
        print("初始/最终训练 MSE", training["train_mse_by_step"][0], training["train_mse_by_step"][-1])
        print("验证 MSE", training["validation_mse"], "最终权重", training["learned_weight"])
        print("按讲义在 .cache 跟做，再完成 exercises/ch06/04_channels.py 的八处 TODO。")
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
