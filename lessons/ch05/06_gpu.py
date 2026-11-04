"""5.6 GPU。贯通设备检查、单卡训练、跨设备恢复和公平计时。"""
import argparse
import copy
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/ch05/5_6"
REPORTS = ROOT / "reports/ch05"
TRIALS = 7


def try_gpu(index=0):
    """返回指定 CUDA 设备，设备不可用时回退到 CPU。"""
    if 0 <= index < torch.cuda.device_count() and torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def try_all_gpus():
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def tensor_demo(device):
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    moved = x.to(device)
    y = torch.ones((2, 3), device=device)
    mismatch = None
    if device.type == "cuda":
        try:
            x + y
        except RuntimeError as error:
            mismatch = str(error)
        if mismatch is None:
            raise AssertionError("预期两个非标量张量相加触发设备错误")
    result = moved + y
    torch.testing.assert_close(result.cpu(), x + 1)
    return {
        "original_device": str(x.device), "moved_device": str(moved.device),
        "to_same_device_returns_self": moved.to(device) is moved,
        "cross_device_add_error": mismatch,
        "result_device": str(result.device), "result": result.cpu().tolist(),
    }


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor([[.25, -.25]]))
        self.layers = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, X):
        return self.layers(X - self.offset)


def train(model, X, y, device):
    model.to(device)
    X, y = X.to(device), y.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.08)
    losses = []
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(X) - y).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    with torch.no_grad():
        prediction = model(X)
        final_loss = (prediction - y).square().mean()
        losses.append(final_loss)
    return prediction, torch.stack(losses).cpu()


def model_demo(device):
    torch.manual_seed(5601)
    X = torch.rand(64, 2) * 2 - 1
    y = 1.5 * X[:, :1] - .75 * X[:, 1:] + .25
    cpu_model = Regressor()
    model = copy.deepcopy(cpu_model)
    reference, cpu_losses = train(cpu_model, X, y, torch.device("cpu"))
    prediction, losses = train(model, X, y, device)
    torch.testing.assert_close(losses, cpu_losses, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(prediction.cpu(), reference, rtol=1e-4, atol=1e-6)

    path = ARTIFACTS / f"regressor_{device.type}.pt"
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored = Regressor().eval()
    restored.load_state_dict(state)
    with torch.no_grad():
        cpu_output = restored(X)
        restored.to(device)
        same_device_output = restored(X.to(device))
    torch.testing.assert_close(cpu_output, prediction.cpu(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(same_device_output, prediction, rtol=1e-4, atol=1e-6)
    return {
        "parameter_device": str(model.layers[0].weight.device),
        "buffer_device": str(model.offset.device), "output_device": str(prediction.device),
        "gradient_device": str(model.layers[0].weight.grad.device),
        "initial_mse": losses[0].item(), "final_mse": losses[-1].item(),
        "cpu_final_mse": cpu_losses[-1].item(),
        "cpu_comparison_max_loss_error": (losses - cpu_losses).abs().max().item(),
        "cpu_comparison_max_prediction_error": (prediction.cpu() - reference).abs().max().item(),
        "loaded_state_device": str(state["layers.0.weight"].device),
        "reload_cpu_max_prediction_error": (cpu_output - prediction.cpu()).abs().max().item(),
        "reload_same_device_max_prediction_error": (same_device_output - prediction).abs().max().item(),
        "local_weights": path.relative_to(ROOT).as_posix(),
    }


def wait_for(*devices):
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def measure(work, *devices, repeat=TRIALS):
    work()  # 预热一次完整工作负载。
    wait_for(*devices)
    durations = []
    for _ in range(repeat):
        wait_for(*devices)
        start = perf_counter()
        work()
        wait_for(*devices)
        durations.append((perf_counter() - start) * 1000)
    return {"median_ms": median(durations), "trials_ms": durations}


def matmul_demo(device):
    results = []
    for n, repeat in [(8, 1000), (2048, 3)]:
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        out = torch.empty_like(a)

        def cpu_work():
            for _ in range(repeat):
                torch.mm(a, b, out=out)

        row = {"n": n, "products_per_trial": repeat, "cpu": measure(cpu_work)}
        if device.type == "cuda":
            a_gpu, b_gpu = a.to(device), b.to(device)
            out_gpu = torch.empty_like(a_gpu)

            def gpu_work():
                for _ in range(repeat):
                    torch.mm(a_gpu, b_gpu, out=out_gpu)

            def round_trip():
                for _ in range(repeat):
                    torch.mm(a.to(device), b.to(device), out=out_gpu).cpu()

            row["gpu_resident"] = measure(gpu_work, device)
            row["gpu_round_trip"] = measure(round_trip, device)
            torch.testing.assert_close(out_gpu.cpu(), out, rtol=1e-4, atol=3e-4)
            row["max_absolute_error"] = (out_gpu.cpu() - out).abs().max().item()
        results.append(row)
    return results


def log_demo(device):
    a = torch.randn(1000, 100, 100).to(device)
    b = torch.randn(1000, 100, 100).to(device)
    out = torch.empty_like(a[0])
    log = torch.empty(len(a), device=device)

    def each_scalar():
        return [
            torch.linalg.matrix_norm(torch.mm(a[i], b[i], out=out)).item()
            for i in range(len(a))
        ]

    def once():
        for i in range(len(a)):
            log[i] = torch.linalg.matrix_norm(torch.mm(a[i], b[i], out=out))
        return log.cpu().tolist()

    each_stats = measure(each_scalar, device)
    once_stats = measure(once, device)
    per_item = torch.tensor(each_scalar())
    in_bulk = torch.tensor(once())
    torch.testing.assert_close(per_item, in_bulk, rtol=0, atol=0)
    manual = (a[0].cpu().double() @ b[0].cpu().double()).square().sum().sqrt().item()
    assert abs(in_bulk[0].item() - manual) < 1e-3
    return {
        "products": len(a), "n": 100, "device": str(device),
        "each_scalar": each_stats, "once": once_stats,
        "max_norm_difference": (per_item - in_bulk).abs().max().item(),
        "first_norm": in_bulk[0].item(), "first_norm_float64_reference": manual,
    }


def two_gpu_demo(devices):
    if len(devices) < 2:
        return {"status": "skipped", "reason": "需要两张可见的 CUDA GPU"}
    gpu0, gpu1 = devices[:2]
    a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
    a0, b0 = a.to(gpu0), b.to(gpu0)
    a1, b1 = a.to(gpu1), b.to(gpu1)
    out0, out0_second, out1 = torch.empty_like(a0), torch.empty_like(a0), torch.empty_like(a1)
    torch.testing.assert_close((a0.to(gpu1) + a1).cpu(), a + a, rtol=0, atol=0)

    def serial():
        torch.mm(a0, b0, out=out0)
        torch.mm(a0, b0, out=out0_second)

    def parallel():
        torch.mm(a0, b0, out=out0)
        torch.mm(a1, b1, out=out1)

    serial_stats = measure(serial, gpu0)
    parallel_stats = measure(parallel, gpu0, gpu1)
    torch.testing.assert_close(out0_second.cpu(), out1.cpu(), rtol=1e-4, atol=3e-4)
    return {
        "status": "measured", "devices": [str(gpu0), str(gpu1)],
        "serial": serial_stats, "parallel": parallel_stats,
        "speedup": serial_stats["median_ms"] / parallel_stats["median_ms"],
    }


def record(metrics):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "5_6_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    figure = REPORTS / "figures/5_6_timing.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.7), layout="constrained")
    for ax, row in zip(axes[:2], metrics["matmul"]):
        keys = ["cpu", "gpu_resident", "gpu_round_trip"]
        labels = ["CPU", "GPU常驻", "GPU往返"]
        pairs = [(label, row[key]["median_ms"]) for key, label in zip(keys, labels) if key in row]
        ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color=["#346491", "#147e83", "#b76b1b"][:len(pairs)])
        ax.set_title(f'{row["n"]}×{row["n"]} 矩阵乘法 ×{row["products_per_trial"]}')
    log = metrics["log"]
    axes[2].barh(["逐次读出", "末尾读出"], [log["each_scalar"]["median_ms"], log["once"]["median_ms"]],
                 color=["#b76b1b", "#147e83"])
    axes[2].set_title(f'1000 个范数（{log["device"]}）')
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)
        ax.set_xlabel("毫秒（7 次中位数）")
    fig.savefig(figure, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--record", action="store_true", help="更新仓库中的实测报告和图")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(5600)
    devices = try_all_gpus()
    device = try_gpu() if args.device == "auto" else torch.device("cpu")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    metrics = {
        "section": "5.6", "python": platform.python_version(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "selected_device": str(device),
        "available_devices": [str(d) for d in devices],
        "gpu_names": [torch.cuda.get_device_name(d) for d in devices if d.type == "cuda"],
        "cpu_threads": torch.get_num_threads(), "matmul_precision": torch.get_float32_matmul_precision(),
        "try_gpu_0": str(try_gpu()), "try_gpu_1": str(try_gpu(1)),
        "tensor": tensor_demo(device), "model": model_demo(device),
        "matmul": matmul_demo(device), "log": log_demo(device),
        "two_gpu": two_gpu_demo(devices if args.device == "auto" else []),
    }
    if args.record:
        record(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
