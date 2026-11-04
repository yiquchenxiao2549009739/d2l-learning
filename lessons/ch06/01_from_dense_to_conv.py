"""6.1 从全连接层到卷积。

阅读顺序与讲义一致
1. DenseMap、LocalMap、SharedMap 表达三种连接规则。
2. spatial_demo 用同一个核寻找不同位置的 L 形。
3. math_and_channels 核对互相关、数学卷积和通道。
4. train_maps 只监督一个输出位置，检验共享参数的作用。

加 --guided 按中文步骤展示结果，加 --record 更新实测 JSON 和配图。
所有图像在内存中生成。坐标均按从 0 开始的行、列排列。
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
SIDE = 8
WINDOW = 3
OUT = SIDE - WINDOW + 1
TRAIN_POSITION = (2, 2)
STEPS = 200
LEARNING_RATE = .08


class DenseMap(nn.Module):
    """36 个输出各自读取全部 64 个像素，共 36×65 个参数。"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(SIDE * SIDE, OUT * OUT, dtype=DTYPE)

    def forward(self, X):
        return self.layer(X.flatten(1)).reshape(-1, OUT, OUT)


class LocalMap(nn.Module):
    """每个输出读取自己的 3×3 窗口，36 组权重分别学习。"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(OUT * OUT, WINDOW**2, dtype=DTYPE))
        self.bias = nn.Parameter(torch.zeros(OUT * OUT, dtype=DTYPE))

    def forward(self, X):
        # 每列是一个窗口，列的顺序与输出图的位置一致。
        windows = F.unfold(X, WINDOW)
        scores = torch.einsum("nkp,pk->np", windows, self.weight) + self.bias
        return scores.reshape(-1, OUT, OUT)


class SharedMap(nn.Module):
    """全部输出共用 9 个核权重和一个偏置。"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(1, 1, WINDOW, dtype=DTYPE)

    def forward(self, X):
        return self.layer(X).squeeze(1)


def make_image(position, motif):
    image = torch.zeros(SIDE, SIDE, dtype=DTYPE)
    row, col = position
    image[row:row + WINDOW, col:col + WINDOW] = motif
    return image


def spatial_demo(kernel, bias):
    """先手算匹配分数，再比较平移前后的响应及边界。"""
    motif = (kernel > 0).to(DTYPE)
    positions = [(1, 1), (3, 2)]
    X = torch.stack([make_image(position, motif) for position in positions])
    responses = F.conv2d(X[:, None], kernel[None, None], bias).squeeze(1)
    peaks = [list(divmod(int(response.argmax()), OUT)) for response in responses]
    assert peaks == [list(position) for position in positions]

    dr = positions[1][0] - positions[0][0]
    dc = positions[1][1] - positions[0][1]
    overlap_error = (responses[1, dr:, dc:] - responses[0, :-dr, :-dc]).abs().max().item()
    assert overlap_error == 0
    row, col = positions[0]
    hand_score = (X[0, row:row + WINDOW, col:col + WINDOW] * kernel).sum() + bias[0]
    assert hand_score == responses[0, row, col]

    # 有限画布做零填充时，先移动输入与先计算再移动输出可能不同。
    ones = torch.ones(1, 1, SIDE, SIDE, dtype=DTYPE)
    sum_window = torch.ones(1, 1, 3, 3, dtype=DTYPE)
    moved_input = F.pad(ones[..., :-1], (1, 0))
    response_after_move = F.conv2d(moved_input, sum_window, padding=1)
    moved_response = F.pad(F.conv2d(ones, sum_window, padding=1)[..., :-1], (1, 0))
    edge = [response_after_move[0, 0, 3, 0].item(), moved_response[0, 0, 3, 0].item()]
    assert edge == [3., 0.]
    details = {
        "input_shape": list(X.shape[1:]), "response_shape": list(responses.shape[1:]),
        "motif_positions": positions, "peak_positions": peaks,
        "peak_values": responses.amax(dim=(1, 2)).tolist(),
        "overlap_max_error": overlap_error,
        "edge_after_move_input_vs_after_move_output": edge,
    }
    return details, X, responses


def math_and_channels(generator):
    X = torch.arange(1., 10., dtype=DTYPE).reshape(3, 3)
    kernel = torch.tensor([[1., 2.], [3., 4.]], dtype=DTYPE)
    correlation = F.conv2d(X[None, None], kernel[None, None])[0, 0]
    convolution = F.conv2d(X[None, None], kernel.flip((0, 1))[None, None])[0, 0]

    def full_convolution(first, second):
        padding = (second.shape[0] - 1, second.shape[1] - 1)
        return F.conv2d(first[None, None], second.flip((0, 1))[None, None], padding=padding)[0, 0]

    swap_error = (full_convolution(X, kernel) - full_convolution(kernel, X)).abs().max().item()
    assert swap_error == 0
    assert correlation[0, 0] == 37 and convolution[0, 0] == 23

    images = torch.randn(2, 3, SIDE, SIDE, generator=generator, dtype=DTYPE)
    weights = torch.randn(2, 3, 3, 3, generator=generator, dtype=DTYPE)
    bias = torch.tensor([.25, -.5], dtype=DTYPE)
    scores = F.conv2d(images, weights, bias)
    manual = (images[0, :, :3, :3] * weights[0]).sum() + bias[0]
    torch.testing.assert_close(scores[0, 0, 0, 0], manual)

    point_weights = weights[:, :, 1, 1].contiguous()
    point_conv = F.conv2d(images, point_weights[:, :, None, None], bias)
    point_linear = F.linear(images.movedim(1, -1), point_weights, bias).movedim(-1, 1)
    torch.testing.assert_close(point_conv, point_linear)
    return {
        "correlation_valid": correlation.tolist(), "convolution_flipped_valid": convolution.tolist(),
        "full_convolution_swap_max_error": swap_error,
        "input_nchw": list(images.shape), "weights_oihw": list(weights.shape),
        "output_nchw": list(scores.shape), "multi_channel_parameters": weights.numel() + bias.numel(),
        "multi_channel_one_score_error": abs((scores[0, 0, 0, 0] - manual).item()),
        "pointwise_output_shape": list(point_conv.shape),
        "pointwise_max_error": (point_conv - point_linear).abs().max().item(),
    }


def train_maps(kernel, bias, generator):
    """固定目标生成规则，仅在 (2,2) 计算损失，再评估全部位置。"""
    X = torch.randn(160, 1, SIDE, SIDE, generator=generator, dtype=DTYPE)
    test = torch.randn(64, 1, SIDE, SIDE, generator=generator, dtype=DTYPE)
    targets = F.conv2d(X, kernel[None, None], bias).squeeze(1)
    test_targets = F.conv2d(test, kernel[None, None], bias).squeeze(1)
    row, col = TRAIN_POSITION
    unseen = torch.ones(OUT, OUT, dtype=torch.bool)
    unseen[row, col] = False
    records, error_maps = {}, {}
    for name, model in [("dense", DenseMap()), ("local", LocalMap()), ("shared", SharedMap())]:
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        initial_mse = targets[:, row, col].square().mean().item()
        for _ in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            # 切片决定哪些输出提供监督。其余位置能否改变取决于参数是否共享。
            error = model(X)[:, row, col] - targets[:, row, col]
            loss = error.square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_mse = (model(X)[:, row, col] - targets[:, row, col]).square().mean().item()
            test_outputs = model(test)
            error_map = (test_outputs - test_targets).square().mean(dim=0)
            records[name] = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "initial_train_mse": initial_mse, "train_mse": train_mse,
                "test_seen_position_mse": error_map[row, col].item(),
                "test_unseen_positions_mse": error_map[unseen].mean().item(),
                "max_unseen_prediction": test_outputs[:, unseen].abs().max().item(),
            }
            error_maps[name] = error_map
            if name == "shared":
                records[name]["kernel_max_error"] = (model.layer.weight[0, 0] - kernel).abs().max().item()
    assert records["shared"]["test_unseen_positions_mse"] < 1e-12
    assert all(records[name]["max_unseen_prediction"] == 0 for name in ("dense", "local"))
    return records, error_maps


def record(metrics, images, responses, error_maps):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figures = REPORTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (REPORTS / "6_1_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 4, figsize=(9.8, 2.9), layout="constrained")
    for i, (row, col) in enumerate(metrics["spatial"]["peak_positions"]):
        image_ax, response_ax = axes[2 * i:2 * i + 2]
        image_ax.imshow(images[i].numpy(), cmap="Blues", vmin=0, vmax=1)
        image_ax.add_patch(Rectangle((col - .5, row - .5), 3, 3, fill=False, edgecolor="#d97706", linewidth=2))
        plotted = response_ax.imshow(responses[i].numpy(), cmap="RdBu_r", vmin=-5.5, vmax=5.5)
        response_ax.scatter(col, row, marker="s", s=150, facecolors="none", edgecolors="#f59e0b", linewidths=2)
        image_ax.set_title("原图" if i == 0 else "向下 2、向右 1")
        response_ax.set_title(f"响应峰值 ({row}, {col})")
        for ax in (image_ax, response_ax):
            ax.set_xlabel("列")
            ax.set_ylabel("行")
            ax.tick_params(labelsize=8)
    fig.colorbar(plotted, ax=axes, shrink=.74, label="共享核的响应")
    fig.savefig(figures / "6_1_translation.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2), layout="constrained")
    max_mse = max(values.max().item() for values in error_maps.values())
    row, col = TRAIN_POSITION
    for ax, name, title in zip(axes, ("dense", "local", "shared"), ("全连接", "局部独立", "共享卷积")):
        heatmap = ax.imshow(error_maps[name].numpy(), cmap="YlGnBu", vmin=0, vmax=max_mse)
        ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, fill=False, edgecolor="#ea580c", linewidth=2))
        ax.set(title=title, xlabel="输出列", ylabel="输出行")
    fig.colorbar(heatmap, ax=axes, shrink=.75, label="新图片的逐位置 MSE")
    fig.savefig(figures / "6_1_position_generalization.png", dpi=150)
    plt.close(fig)


def print_guided(metrics):
    """直接使用已核对的结果，按原理、观察、解释的顺序阅读。"""
    spatial = metrics["spatial"]
    math = metrics["math_and_channels"]
    training = metrics["training"]
    print("\n6.1 从全连接层到卷积\n")
    print("步骤 1 先观察同一组核怎样找到不同位置的图案")
    print("输入与输出形状", spatial["input_shape"], spatial["response_shape"])
    print("图案位置", spatial["motif_positions"])
    print("响应峰值位置", spatial["peak_positions"])
    print("峰值", spatial["peak_values"])
    print("平移重叠区域最大差", spatial["overlap_max_error"])
    print("同一个局部规则在新位置继续使用，响应位置随图案移动。\n")
    print("步骤 2 将边界条件与卷积约定一起核对")
    print("边界两条计算路径", spatial["edge_after_move_input_vs_after_move_output"])
    print("互相关与翻核后的左上输出", math["correlation_valid"][0][0], math["convolution_flipped_valid"][0][0])
    print("完整卷积交换输入顺序的最大差", math["full_convolution_swap_max_error"])
    print("步骤 3 读懂批量、输入通道和输出通道")
    print("输入、权重、输出形状", math["input_nchw"], math["weights_oihw"], math["output_nchw"])
    print("三输入两输出通道的参数量", math["multi_channel_parameters"])
    print("1×1 卷积与逐像素线性层最大差", math["pointwise_max_error"], "\n")
    print("步骤 4 只教一个位置，检查其余位置")
    print("训练图、测试图、监督位置", training["training_images"], training["test_images"], training["supervised_position"])
    print("模型          参数数       训练 MSE       其余位置测试 MSE")
    for name, values in training["models"].items():
        print(f"{name:8s} {values['parameters']:9d} {values['train_mse']:14.6g} {values['test_unseen_positions_mse']:18.6g}")
    print("目标由同一共享核生成。共享结构把监督位置学到的规则用于其余位置。")
    print("接着按讲义修改图案位置，再打开 exercises/ch06/01_from_dense_to_conv.py。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="更新仓库中的实测 JSON 和图")
    parser.add_argument("--guided", action="store_true", help="按讲义步骤显示中文结果和解释")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(6100)
    generator = torch.Generator().manual_seed(6101)
    kernel = torch.tensor([[1., -1., -1.], [1., -1., -1.], [1., 1., 1.]], dtype=DTYPE)
    bias = torch.tensor([.25], dtype=DTYPE)
    spatial, images, responses = spatial_demo(kernel, bias)
    math = math_and_channels(generator)
    training, error_maps = train_maps(kernel, bias, generator)
    metrics = {
        "section": "6.1", "date": "2026-11-04", "torch": torch.__version__,
        "device": "cpu", "dtype": str(DTYPE),
        "spatial": spatial, "math_and_channels": math,
        "training": {
            "seed": 6101, "training_images": 160, "test_images": 64,
            "supervised_position": TRAIN_POSITION, "unseen_position_count": OUT * OUT - 1,
            "steps": STEPS, "learning_rate": LEARNING_RATE, "models": training,
        },
    }
    if args.record:
        record(metrics, images, responses, error_maps)
    if args.guided:
        print_guided(metrics)
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
