"""5.4 自定义层。复用 openmmlab，在 CPU 上运行小型人工实验。"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

REPORT = ROOT / "reports/ch05"
FIGURES = REPORT / "figures"
DTYPE = torch.float64


class CenteredLayer(nn.Module):
    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    """按教材方向保存权重 (输入维, 输出维)，包含 ReLU。"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight.T, nonlinearity="relu")

    def forward(self, X):
        return torch.relu(X @ self.weight + self.bias)


class QuadraticLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, in_features, out_features))

    def forward(self, X):
        return torch.einsum("...i,ijk,...j->...k", X, self.weight, X)


class FourierLayer(nn.Module):
    def forward(self, X):
        return torch.fft.rfft(X, dim=-1)


def centering_demo():
    layer = CenteredLayer()
    X = torch.arange(1, 6, dtype=DTYPE, requires_grad=True)
    Z = layer(X)
    upstream = torch.tensor([1., 0., 2., -1., 3.], dtype=DTYPE)
    (Z * upstream).sum().backward()
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer()).double()
    batch = torch.tensor([[1., 2.], [9., 10.]], dtype=DTYPE)
    return {
        "parameter_count": sum(p.numel() for p in layer.parameters()),
        "centered_vector": Z.detach().tolist(),
        "gradient_error": (X.grad - (upstream - upstream.mean())).abs().max().item(),
        "sequential_output_mean": net(torch.rand(4, 8, dtype=DTYPE)).mean().item(),
        "global_centered": layer(batch).tolist(),
        "row_centered": (batch - batch.mean(dim=-1, keepdim=True)).tolist(),
    }


def linear_demo():
    layer = MyLinear(5, 3).double()
    reference = nn.Sequential(nn.Linear(5, 3), nn.ReLU()).double()
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(.05, .75, 15, dtype=DTYPE).reshape(5, 3))
        layer.bias.copy_(torch.tensor([.1, .2, .3], dtype=DTYPE))
        reference[0].weight.copy_(layer.weight.T)
        reference[0].bias.copy_(layer.bias)
    X = (torch.arange(1, 11, dtype=DTYPE).reshape(2, 5) / 10).requires_grad_()
    other_X = X.detach().clone().requires_grad_()
    Y, other_Y = layer(X), reference(other_X)
    Y.square().mean().backward()
    other_Y.square().mean().backward()
    result = {
        "parameter_count": sum(p.numel() for p in layer.parameters()),
        "output_shape": list(Y.shape),
        "output_error": (Y - other_Y).abs().max().item(),
        "weight_gradient_error": (layer.weight.grad.T - reference[0].weight.grad).abs().max().item(),
        "bias_gradient_error": (layer.bias.grad - reference[0].bias.grad).abs().max().item(),
        "input_gradient_error": (X.grad - other_X.grad).abs().max().item(),
    }
    optimizer = torch.optim.SGD(layer.parameters(), lr=.01)
    before = layer.weight.detach().clone()
    optimizer.step()
    result["one_step_weight_change"] = (before - layer.weight).norm().item()
    optimizer.zero_grad(set_to_none=True)

    # 局部复现教材的 .data 写法，观察参数梯度。
    broken = torch.relu(X.detach() @ layer.weight.data + layer.bias.data)
    probe = X.detach().clone().requires_grad_()
    torch.relu(probe @ layer.weight.data + layer.bias.data).sum().backward()
    result["book_output_requires_grad_with_plain_input"] = broken.requires_grad
    result["book_input_has_gradient_when_requested"] = probe.grad is not None
    result["book_parameter_gradients_are_none"] = all(p.grad is None for p in layer.parameters())
    chain = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1)).double()
    result["composed_output_shape"] = list(chain(torch.ones(2, 64, dtype=DTYPE)).shape)
    return result


def make_quadratic_data():
    generator = torch.Generator().manual_seed(5410)
    data = {}
    for name, count in (("train", 64), ("validation", 32), ("test", 32)):
        X = 2 * torch.rand(count, 2, generator=generator, dtype=DTYPE) - 1
        first, second = X[:, :1], X[:, 1:]
        y = .8 * first.square() + .6 * first * second - .4 * second.square()
        data[name] = (X, y)
    return data


def train_quadratic():
    data = make_quadratic_data()
    layer = QuadraticLayer(2, 1).double()
    optimizer = torch.optim.SGD(layer.parameters(), lr=.2)
    X, y = data["train"]
    VX, vy = data["validation"]
    history = []
    for step in range(201):
        loss = (layer(X) - y).square().mean()
        with torch.no_grad():
            validation_loss = (layer(VX) - vy).square().mean().item()
        history.append({"step": step, "train_mse": loss.item(), "validation_mse": validation_loss})
        if step == 200:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    TX, ty = data["test"]
    with torch.no_grad():
        test_mse = (layer(TX) - ty).square().mean().item()
    weight = layer.weight.detach()[:, :, 0]
    return {
        "updates": 200, "learning_rate": .2, "parameter_count": layer.weight.numel(),
        "initial_train_mse": history[0]["train_mse"],
        "final_train_mse": history[-1]["train_mse"],
        "final_validation_mse": history[-1]["validation_mse"],
        "test_mse": test_mse, "weight": weight.tolist(),
        "interaction_coefficient": (weight[0, 1] + weight[1, 0]).item(),
    }, history


def fourier_demo():
    layer = FourierLayer()
    sample = torch.arange(8, dtype=DTYPE)
    signal = 3 + torch.sin(2 * torch.pi * 2 * sample / 8)
    spectrum = layer(signal)
    odd_signal = torch.linspace(-1, 1, 7, dtype=DTYPE)
    odd_spectrum = layer(odd_signal)
    info = {
        "even_bins": spectrum.numel(), "odd_bins": odd_spectrum.numel(),
        "dc_real": spectrum[0].real.item(),
        "bin_2_real": spectrum[2].real.item(), "bin_2_imag": spectrum[2].imag.item(),
        "even_reconstruction_error": (torch.fft.irfft(spectrum, n=8) - signal).abs().max().item(),
        "odd_reconstruction_error": (torch.fft.irfft(odd_spectrum, n=7) - odd_signal).abs().max().item(),
        "coefficients_real": spectrum.real.tolist(),
        "coefficients_imag": spectrum.imag.tolist(),
    }
    return info, signal, spectrum


def draw_figures(centering, signal, spectrum, history):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})
    teal, gold = "#147e83", "#c57725"
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, title, values in zip(axes, ("原输入，均值3", "全局中心化，均值0"),
                                 ([1, 2, 3, 4, 5], centering["centered_vector"])):
        ax.bar(range(1, 6), values, color=teal)
        ax.axhline(0, color="#334155", lw=.8)
        ax.set(title=title, xlabel="元素位置", ylim=(-2.5, 5.5))
    fig.tight_layout()
    fig.savefig(FIGURES / "5_4_centering.png", dpi=145)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    axes[0].plot(range(8), signal.tolist(), "o-", color=teal)
    axes[0].set(title="8点实信号，常量加正弦", xlabel="采样位置", ylabel="信号值")
    axes[1].bar(range(5), spectrum.abs().tolist(), color=gold)
    axes[1].set(title="rfft保留5个频点", xlabel="频点索引 k", ylabel="系数模长")
    fig.tight_layout()
    fig.savefig(FIGURES / "5_4_fourier.png", dpi=145)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    ax.semilogy([r["step"] for r in history], [r["train_mse"] for r in history], label="训练64条", color=teal)
    ax.semilogy([r["step"] for r in history], [r["validation_mse"] for r in history], label="验证32条", color=gold)
    ax.set(title="自定义二次层拟合平方与交叉项", xlabel="SGD更新次数", ylabel="MSE（对数坐标）")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "5_4_training.png", dpi=145)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(5401)
    FIGURES.mkdir(parents=True, exist_ok=True)
    centering = centering_demo()
    linear = linear_demo()
    quadratic, history = train_quadratic()
    fourier, signal, spectrum = fourier_demo()
    metrics = {
        "section": "5.4", "date": "2026-11-03", "torch": torch.__version__,
        "dtype": str(DTYPE), "device": "cpu", "centering": centering,
        "linear": linear, "quadratic": quadratic, "fourier": fourier,
    }
    with (REPORT / "5_4_epochs.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (REPORT / "5_4_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    draw_figures(centering, signal, spectrum, history)
    for name in ("centering", "linear", "quadratic", "fourier"):
        print(name, json.dumps(metrics[name], ensure_ascii=False))


if __name__ == "__main__":
    main()
