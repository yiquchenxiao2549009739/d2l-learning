"""6.5 汇聚层。

沿用窗口、填充和步幅，但将窗口内乘权求和改为固定的最大值或平均值。
pool2d 用显式循环处理有限浮点二维输入，最大值并列时选择第一个位置。
依次核对教材、边界约定、通道独立、梯度、位置敏感性和卷积到汇聚的流水线。
--guided 显示中文步骤，--record 保存实测 JSON 和四张图。
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/ch06"
DTYPE = torch.float64


def pool2d(X, pool_size, mode="max", stride=(1, 1), padding=(0, 0), count_include_pad=True):
    """手写二维汇聚，显式步幅默认 1，填充按单侧高宽给定。"""
    if X.ndim != 2 or X.numel() == 0 or not X.is_floating_point() or not torch.isfinite(X).all():
        raise ValueError("本函数需要非空、有限浮点二维输入")
    if mode not in ("max", "avg"):
        raise ValueError("mode 只能是 max 或 avg")
    kh, kw = pool_size
    sh, sw = stride
    ph, pw = padding
    if min(kh, kw, sh, sw) < 1 or min(ph, pw) < 0 or ph > kh // 2 or pw > kw // 2:
        raise ValueError("窗口和步幅应为正整数，单侧填充不超过半个窗口")
    rows = (X.shape[0] + 2 * ph - kh) // sh + 1
    cols = (X.shape[1] + 2 * pw - kw) // sw + 1
    if min(rows, cols) < 1:
        raise ValueError("填充后的输入须能放下完整窗口")
    pad4 = (pw, pw, ph, ph)
    fill = float("-inf") if mode == "max" else 0.
    padded = F.pad(X, pad4, value=fill)
    valid = F.pad(torch.ones_like(X), pad4)
    outputs = []
    for i in range(rows):
        line = []
        for j in range(cols):
            row, col = i * sh, j * sw
            patch = padded[row:row + kh, col:col + kw]
            if mode == "max":
                flat = patch.reshape(-1)
                value = flat[flat.argmax()]
            else:
                divisor = kh * kw if count_include_pad else valid[row:row + kh, col:col + kw].sum()
                value = patch.sum() / divisor
            line.append(value)
        outputs.append(torch.stack(line))
    return torch.stack(outputs)


def textbook_demo():
    X = torch.arange(9, dtype=DTYPE).reshape(3, 3)
    maximum = pool2d(X, (2, 2))
    average = pool2d(X, (2, 2), mode="avg")
    torch.testing.assert_close(maximum, torch.tensor([[4., 5.], [7., 8.]], dtype=DTYPE))
    torch.testing.assert_close(average, torch.tensor([[2., 3.], [5., 6.]], dtype=DTYPE))
    source = torch.arange(16, dtype=DTYPE).reshape(1, 1, 4, 4)
    default = nn.MaxPool2d(3)(source)
    padded = nn.MaxPool2d(3, stride=2, padding=1)(source)
    rectangular = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))(source)
    torch.testing.assert_close(padded, rectangular)
    channels = torch.cat((source, source + 1), dim=1)
    channel_output = F.max_pool2d(channels, 3, stride=2, padding=1)
    torch.testing.assert_close(channel_output[:, 1], channel_output[:, 0] + 1)
    return {"maximum": maximum.tolist(), "average": average.tolist(),
            "default_window3_output": default[0, 0].tolist(),
            "padding1_stride2_output": padded[0, 0].tolist(),
            "rectangular_output": rectangular[0, 0].tolist(),
            "multi_channel_input_shape": list(channels.shape),
            "multi_channel_output": channel_output.tolist(),
            "max_pool_parameters": sum(p.numel() for p in nn.MaxPool2d(2).parameters()),
            "avg_pool_parameters": sum(p.numel() for p in nn.AvgPool2d(2).parameters())}, (X, maximum, average)


def padding_demo():
    negative = -torch.tensor([[4., 3.], [2., 1.]], dtype=DTYPE)
    proper = F.max_pool2d(negative[None, None], 3, stride=1, padding=1)[0, 0]
    zero_padded = F.max_pool2d(F.pad(negative[None, None], (1, 1, 1, 1)), 3, stride=1)[0, 0]
    torch.testing.assert_close(pool2d(negative, (3, 3), padding=(1, 1)), proper)
    positive = -negative
    include = F.avg_pool2d(positive[None, None], 3, stride=1, padding=1, count_include_pad=True)[0, 0]
    exclude = F.avg_pool2d(positive[None, None], 3, stride=1, padding=1, count_include_pad=False)[0, 0]
    torch.testing.assert_close(pool2d(positive, (3, 3), mode="avg", padding=(1, 1)), include)
    torch.testing.assert_close(pool2d(positive, (3, 3), mode="avg", padding=(1, 1), count_include_pad=False), exclude)
    assert proper.max() == -1 and zero_padded.max() == 0
    return {"max_implicit_padding": proper.tolist(), "max_explicit_zero_padding": zero_padded.tolist(),
            "avg_include_pad": include.tolist(), "avg_exclude_pad": exclude.tolist()}


def gradient_demo():
    X = torch.arange(9, dtype=DTYPE).reshape(3, 3)
    gradients = {}
    errors = {}
    for mode in ("max", "avg"):
        hand_input = X.clone().requires_grad_()
        builtin_input = X.clone().requires_grad_()
        hand = pool2d(hand_input, (2, 2), mode=mode)
        function = F.max_pool2d if mode == "max" else F.avg_pool2d
        builtin = function(builtin_input[None, None], 2, stride=1)[0, 0]
        hand.sum().backward()
        builtin.sum().backward()
        gradients[mode] = hand_input.grad
        errors[mode] = (hand_input.grad - builtin_input.grad).abs().max().item()
    tie = torch.ones(2, 2, dtype=DTYPE, requires_grad=True)
    pool2d(tie, (2, 2)).sum().backward()
    builtin_tie = torch.ones(1, 1, 2, 2, dtype=DTYPE, requires_grad=True)
    F.max_pool2d(builtin_tie, 2).sum().backward()
    torch.testing.assert_close(tie.grad, builtin_tie.grad[0, 0])
    scalar_tie = torch.ones(2, 2, dtype=DTYPE, requires_grad=True)
    scalar_tie.max().backward()
    assert max(errors.values()) == 0
    return {"max_input_gradient": gradients["max"].tolist(), "avg_input_gradient": gradients["avg"].tolist(),
            "gradient_errors": errors, "tie_first_index_gradient": tie.grad.tolist(),
            "scalar_max_tie_gradient": scalar_tie.grad.tolist()}, (gradients["max"], gradients["avg"])


def shift_demo():
    positions = [(0, 0), (0, 1), (0, 2)]
    images = torch.zeros(3, 1, 4, 4, dtype=DTYPE)
    for index, (row, col) in enumerate(positions):
        images[index, 0, row, col] = 1
    output = F.max_pool2d(images, 2, stride=2)
    assert torch.equal(output[0], output[1]) and not torch.equal(output[1], output[2])
    return {"impulse_positions": positions, "outputs": output[:, 0].tolist(),
            "within_cell_max_difference": (output[0] - output[1]).abs().max().item(),
            "cross_cell_max_difference": (output[1] - output[2]).abs().max().item()}, (images[:, 0], output[:, 0])


def alternatives_demo():
    generator = torch.Generator().manual_seed(6501)
    X = torch.randn(2, 3, 6, 8, generator=generator, dtype=DTYPE)
    average = F.avg_pool2d(X, 2, stride=2)
    fixed_weights = torch.full((3, 1, 2, 2), .25, dtype=DTYPE)
    fixed_conv = F.conv2d(X, fixed_weights, stride=2, groups=3)
    torch.testing.assert_close(average, fixed_conv)
    minimum = -F.max_pool2d(-X, 2, stride=2)
    manual_minimum = torch.stack([torch.stack([
        -pool2d(-channel, (2, 2), stride=(2, 2)) for channel in image
    ]) for image in X])
    torch.testing.assert_close(minimum, manual_minimum)
    a, b = torch.tensor([1., 0.]), torch.tensor([0., 1.])
    non_additive = [a.max().item() + b.max().item(), (a + b).max().item()]
    patch = torch.tensor([0., 1., 3., 4.], dtype=DTYPE)
    temperatures = [.01, .1, 1., 10., 100.]
    soft_values = [temperature * (torch.logsumexp(patch / temperature, dim=0) - math.log(patch.numel())).item()
                   for temperature in temperatures]
    # 用 ReLU 构造两个数的 max，再合并四个数。
    pair_a = F.relu(patch[0] - patch[1]) + patch[1]
    pair_b = F.relu(patch[2] - patch[3]) + patch[3]
    relu_max = F.relu(pair_a - pair_b) + pair_b
    assert relu_max == patch.max()
    return {"average_vs_fixed_depthwise_conv_error": (average - fixed_conv).abs().max().item(),
            "cost_example_input_shape": list(X.shape), "cost_example_output_shape": list(average.shape),
            "cost_example_output_elements": average.numel(), "cost_example_max_comparisons": average.numel() * 3,
            "minimum_max_error": (minimum - manual_minimum).abs().max().item(),
            "max_additivity_counterexample": non_additive, "relu_constructed_max": relu_max.item(),
            "temperatures": temperatures, "log_mean_exp_values": soft_values,
            "patch_mean": patch.mean().item(), "patch_max": patch.max().item()}


def pipeline_demo():
    X = torch.zeros(1, 1, 6, 6, dtype=DTYPE, requires_grad=True)
    with torch.no_grad():
        X[:, :, 2:5, 1:4] = 1
    convolution = nn.Conv2d(1, 2, 3, bias=False, dtype=DTYPE)
    kernels = torch.tensor([[[[0., 0., 0.], [1., 0., -1.], [0., 0., 0.]]],
                            [[[0., 1., 0.], [0., 0., 0.], [0., -1., 0.]]]], dtype=DTYPE)
    with torch.no_grad():
        convolution.weight.copy_(kernels)
    response = convolution(X)
    activated = F.relu(response)
    output = F.max_pool2d(activated, 2, stride=2)
    output.sum().backward()
    assert convolution.weight.grad.abs().sum() > 0
    return {"input_shape": list(X.shape), "convolution_shape": list(response.shape),
            "pooling_shape": list(output.shape), "output": output.detach().tolist(),
            "conv_parameters": convolution.weight.numel(), "pool_parameters": 0,
            "conv_gradient_l1": convolution.weight.grad.abs().sum().item(),
            "input_gradient_l1": X.grad.abs().sum().item()}, (X.detach()[0, 0], response.detach()[0, 0], activated.detach()[0, 0], output.detach()[0, 0])


def record(metrics, textbook_images, gradients, shifts, stages):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (REPORTS / "6_5_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})

    def grid(ax, values, title):
        ax.imshow(values.numpy(), cmap="Blues")
        ax.set(title=title, xticks=[], yticks=[])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j].item()
                ax.text(j, i, f"{value:g}", ha="center", va="center", fontsize=12,
                        color="white" if value > max(values.max().item() * .7, 0) else "#173448")

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), layout="constrained")
    for ax, values, title in zip(axes, textbook_images, ("输入 3×3", "最大汇聚 2×2", "平均汇聚 2×2")):
        grid(ax, values, title)
    fig.savefig(out / "6_5_windows.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), layout="constrained")
    for ax, values, title in zip(axes, gradients, ("最大汇聚的输入梯度", "平均汇聚的输入梯度")):
        grid(ax, values, title)
    fig.savefig(out / "6_5_gradients.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), layout="constrained")
    images, outputs = shifts
    for index, position in enumerate(metrics["shift"]["impulse_positions"]):
        grid(axes[0, index], images[index], f"亮点 {tuple(position)}")
        grid(axes[1, index], outputs[index], "窗口 2×2 / 步幅 2")
    fig.savefig(out / "6_5_shift.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.5), layout="constrained")
    for ax, values, title in zip(axes, stages, ("输入 6×6", "通道 0 响应 4×4", "ReLU 后 4×4", "汇聚后 2×2")):
        grid(ax, values, title)
    fig.savefig(out / "6_5_pipeline.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guided", action="store_true", help="按中文步骤讲解结果")
    parser.add_argument("--record", action="store_true", help="更新实测 JSON 和四图")
    args = parser.parse_args()
    torch.set_num_threads(1)
    textbook, images = textbook_demo()
    gradients, grad_images = gradient_demo()
    shifts, shift_images = shift_demo()
    pipeline, stages = pipeline_demo()
    metrics = {"section": "6.5", "date": "2026-11-18", "torch": torch.__version__,
               "device": "cpu", "dtype": str(DTYPE), "textbook": textbook,
               "padding": padding_demo(), "gradients": gradients, "shift": shifts,
               "alternatives": alternatives_demo(), "pipeline": pipeline}
    if args.record:
        record(metrics, images, grad_images, shift_images, stages)
    if args.guided:
        print("6.5 汇聚层\n步骤 1 将窗口乘加改为最大值或平均值")
        print("最大/平均响应", textbook["maximum"], textbook["average"])
        print("默认 3×3 窗口结果", textbook["default_window3_output"])
        print("步骤 2 核对边界、步幅和通道")
        print("教材填充步幅结果", textbook["padding1_stride2_output"])
        print("多通道响应", textbook["multi_channel_output"])
        print("负数输入的正确最大汇聚/显式补零", metrics["padding"]["max_implicit_padding"], metrics["padding"]["max_explicit_zero_padding"])
        print("步骤 3 观察位置变化与梯度")
        print("同窗移动/跨窗移动差", shifts["within_cell_max_difference"], shifts["cross_cell_max_difference"])
        print("最大/平均输入梯度", gradients["max_input_gradient"], gradients["avg_input_gradient"])
        print("最大值并列时选择第一个位置", gradients["tie_first_index_gradient"])
        print("平均值计入/排除边界", metrics["padding"]["avg_include_pad"], metrics["padding"]["avg_exclude_pad"])
        alternatives = metrics["alternatives"]
        print("平均汇聚与固定逐通道卷积最大差", alternatives["average_vs_fixed_depthwise_conv_error"])
        print("最小汇聚最大差", alternatives["minimum_max_error"])
        print("光滑汇总温度", alternatives["temperatures"])
        print("光滑汇总输出", alternatives["log_mean_exp_values"])
        print("步骤 4 接通卷积、激活和汇聚")
        print("输入、卷积、汇聚形状", pipeline["input_shape"], pipeline["convolution_shape"], pipeline["pooling_shape"])
        print("卷积参数/汇聚参数", pipeline["conv_parameters"], pipeline["pool_parameters"])
        print("卷积权重收到的梯度绝对值之和", pipeline["conv_gradient_l1"])
        print("按讲义完成跟做，再填写 exercises/ch06/05_pooling.py 的八处 TODO。")
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
