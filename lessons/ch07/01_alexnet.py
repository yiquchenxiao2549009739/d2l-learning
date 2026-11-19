"""7.1 AlexNet：教材完整结构、真实训练、简化/批量/LeNet对照与GPU测量。

默认十五轮：先完成教材十轮，再按预定方案延长五轮，两个终点各测试一次。
--inspect 不读数据；--experiments 只用固定子集验证；--benchmark 测本机。
--record 写公开报告与图，本地权重只保存在 artifacts/ch07。原始数据离线只读。
--native 是观察子集优化不足后追加的固定28图方案（lr=.1、完整十五轮）。
"""
import argparse
import csv
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

COMMON = runpy.run_path(str(ROOT / "lessons/ch06/06_lenet.py"))
REPORT = ROOT / "reports/ch07"
BASE = dict(architecture="alex", channels=[96, 256, 384, 384, 256], hidden=[4096, 4096],
            size=224, dropout=.5, activation="relu", standardize=False, lr=.01, batch_size=128)


def build_net(config=None, seed=7100):
    cfg = dict(BASE, **(config or {}))
    if cfg["architecture"] not in ("alex", "native", "lenet"):
        raise ValueError("architecture 须为 alex/native/lenet")
    if cfg["activation"] not in ("relu", "sigmoid") or not 0 <= cfg["dropout"] < 1:
        raise ValueError("检查激活名称与 dropout 概率")
    activation = nn.ReLU if cfg["activation"] == "relu" else nn.Sigmoid
    with torch.random.fork_rng(devices=[]):
        # 只设置CPU生成器；构造CPU参数时不扰动调用者的CUDA Dropout随机流。
        torch.default_generator.manual_seed(seed)
        if cfg["architecture"] == "alex":
            if cfg["size"] != 224:
                raise ValueError("本节教材 AlexNet 固定 224 输入；28 输入请用 native")
            a, b, c, d, e = cfg["channels"]
            layers = [nn.Conv2d(1, a, 11, stride=4, padding=1), activation(), nn.MaxPool2d(3, 2),
                      nn.Conv2d(a, b, 5, padding=2), activation(), nn.MaxPool2d(3, 2),
                      nn.Conv2d(b, c, 3, padding=1), activation(),
                      nn.Conv2d(c, d, 3, padding=1), activation(),
                      nn.Conv2d(d, e, 3, padding=1), activation(), nn.MaxPool2d(3, 2), nn.Flatten()]
            width = e * 5 * 5
            hidden = cfg["hidden"]
        elif cfg["architecture"] == "native":
            if cfg["size"] != 28:
                raise ValueError("native 固定直接读取 28×28")
            if len(cfg["channels"]) != 3:
                raise ValueError("native 需要三个卷积通道数")
            layers, width_in = [], 1
            for width_out in cfg["channels"]:
                layers.extend([nn.Conv2d(width_in, width_out, 3, padding=1), activation(), nn.MaxPool2d(2, 2)])
                width_in = width_out
            layers.append(nn.Flatten())
            width, hidden = cfg["channels"][-1] * 3 * 3, cfg["hidden"]
        else:
            if cfg["size"] != 28:
                raise ValueError("lenet 固定 28×28")
            c1, c2 = cfg["channels"]
            layers = [nn.Conv2d(1, c1, 5, padding=2), activation(), nn.AvgPool2d(2, 2),
                      nn.Conv2d(c1, c2, 5), activation(), nn.AvgPool2d(2, 2), nn.Flatten()]
            width, hidden = c2 * 5 * 5, cfg["hidden"]
        for next_width in hidden:
            layers.extend([nn.Linear(width, next_width), activation(), nn.Dropout(cfg["dropout"])])
            width = next_width
        layers.append(nn.Linear(width, 10))
        net = nn.Sequential(*layers)
        for layer in net:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight, gain=1.)
                nn.init.zeros_(layer.bias)
    return net


def parameter_hash(net):
    # 按参数顺序而非模块索引命名，便于比较不同激活/Dropout而相同形状的网络。
    digest = hashlib.sha256()
    for value in net.parameters():
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def inspect_net(config=None):
    cfg = dict(BASE, **(config or {}))
    net = build_net(cfg).eval()
    X = torch.zeros(1, 1, cfg["size"], cfg["size"])
    rows = []
    receptive, jump = 1, 1
    with torch.no_grad():
        for index, layer in enumerate(net):
            shape_in = list(X.shape)
            X = layer(X)
            params = sum(p.numel() for p in layer.parameters())
            multiply = 0
            if isinstance(layer, nn.Conv2d):
                multiply = X[0].numel() * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            elif isinstance(layer, nn.Linear):
                multiply = layer.in_features * layer.out_features
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
                s = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
                receptive += (k - 1) * jump
                jump *= s
            rows.append(dict(index=index, layer=type(layer).__name__, input_shape=shape_in, output_shape=list(X.shape),
                             parameters=params, weight_multiplications=multiply,
                             output_float32_bytes=X.numel() * 4,
                             receptive_field=receptive if X.ndim == 4 else None,
                             jump=jump if X.ndim == 4 else None))
    count = sum(p.numel() for p in net.parameters())
    return dict(config=cfg, layers=rows, parameters=count, parameter_float32_bytes=count * 4,
                weight_multiplications=sum(r["weight_multiplications"] for r in rows),
                convolution_parameters=sum(r["parameters"] for r in rows if r["layer"] == "Conv2d"),
                linear_parameters=sum(r["parameters"] for r in rows if r["layer"] == "Linear"))


def prepare(X, cfg, device, stats):
    X = X.to(device)
    if cfg["size"] != 28:
        # 批内按需插值，不在CPU预存全部224图。先float/255，再插值，与PIL先Resize略有差别。
        X = F.interpolate(X, size=(cfg["size"], cfg["size"]), mode="bilinear", align_corners=False, antialias=True)
    if cfg["standardize"]:
        X = (X - stats["mean"]) / stats["std"]
    return X


def training_stats(dataset):
    # 只从本次训练划分求像素均值、总体标准差；不拟合验证或测试数据。
    total, squares, count = 0., 0., 0
    for X, _ in COMMON["make_loader"](dataset):
        X = X.double()
        total += X.sum().item()
        squares += X.square().sum().item()
        count += X.numel()
    mean = total / count
    return dict(mean=mean, std=(squares / count - mean * mean) ** .5, count=count)


@torch.no_grad()
def evaluate(net, dataset, cfg, device, stats):
    net.eval()
    total, correct, count = 0., 0, 0
    for X, y in COMMON["make_loader"](dataset, batch_size=cfg["batch_size"]):
        X, y = prepare(X, cfg, device, stats), y.to(device)
        logits = net(X)
        total += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return dict(ce=total / count, accuracy=correct / count, correct=correct, samples=count)


def train_epoch(net, loader, optimizer, cfg, device, stats):
    net.train()
    total, correct, count, updates = 0., 0, 0, 0
    for X, y in loader:
        X, y = prepare(X, cfg, device, stats), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = net(X)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("损失非有限，检查输入、初值与学习率")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        correct += (logits.detach().argmax(1) == y).sum().item()
        count += len(y)
        updates += 1
    return dict(online_train_ce=total / count, online_train_accuracy=correct / count, updates=updates, samples=count)


def save_checkpoint(net, optimizer, cfg, epoch, prefix):
    folder = ROOT / "artifacts/ch07"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (prefix + "_epoch{}.pt".format(epoch))
    torch.save(dict(config=cfg, epoch=epoch, model={k: v.detach().cpu() for k, v in net.state_dict().items()},
                    optimizer=optimizer.state_dict()), path)
    return path


def fit(cfg, splits, epochs, device, name, test_epochs=(), record=False, prefix=None):
    net = build_net(cfg).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg["lr"])
    loader = COMMON["make_loader"](splits["train"], shuffle=True, batch_size=cfg["batch_size"])
    stats = training_stats(splits["train"]) if cfg["standardize"] else dict(mean=0., std=1.)
    torch.manual_seed(7103)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7103)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    result = dict(config=cfg, model_seed=7100, dropout_seed=7103, loader_seed=3603,
                  initial_parameter_sha256=parameter_hash(net), parameters=sum(p.numel() for p in net.parameters()),
                  runtime=COMMON["runtime_metadata"](device), epochs=epochs, normalization=stats,
                  train_samples=len(splits["train"]), validation_samples=len(splits["validation"]), history=[], tests={})
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        online = {} if epoch == 0 else train_epoch(net, loader, optimizer, cfg, device, stats)
        validation = evaluate(net, splits["validation"], cfg, device, stats)
        row = dict(epoch=epoch, **online, **{"validation_" + k: v for k, v in validation.items()})
        if epoch in test_epochs or epoch == epochs:
            row.update({"train_eval_" + k: v for k, v in evaluate(net, splits["train"], cfg, device, stats).items()})
        result["history"].append(row)
        if epoch in test_epochs:
            result["tests"][str(epoch)] = evaluate(net, splits["test"], cfg, device, stats)
            if record:
                save_checkpoint(net, optimizer, cfg, epoch, prefix)
        print("{} epoch {:2d}: online CE {}, val CE {:.6f}, val acc {:.2%}".format(
            name, epoch, "-" if not online else "{:.6f}".format(online["online_train_ce"]),
            validation["ce"], validation["accuracy"]), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result.update(seconds=time.perf_counter() - started, updates=sum(r.get("updates", 0) for r in result["history"]),
                  last_batch_size=(len(splits["train"]) - 1) % cfg["batch_size"] + 1,
                  final_parameter_sha256=parameter_hash(net),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    if record and not test_epochs:
        save_checkpoint(net, optimizer, cfg, epochs, prefix)
    return result, net


def write_json(name, value):
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def write_csv(name, rows):
    REPORT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (REPORT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidates():
    return {"A": dict(BASE),
            "B": dict(BASE, channels=[24, 64, 96, 96, 64], hidden=[256, 128]),
            "C": dict(BASE, architecture="native", channels=[16, 32, 64], hidden=[64], size=28, dropout=0.),
            "D": dict(BASE, batch_size=32),
            "L0": dict(BASE, architecture="lenet", channels=[6, 16], hidden=[120, 84], size=28, activation="sigmoid", dropout=0., lr=.1),
            "L1": dict(BASE, architecture="lenet", channels=[6, 16], hidden=[120, 84], size=28, dropout=0., lr=.1),
            "L2": dict(BASE, architecture="lenet", channels=[6, 16], hidden=[120, 84], size=28, dropout=.5, lr=.1),
            "L3": dict(BASE, architecture="lenet", channels=[6, 16], hidden=[120, 84], size=28, dropout=.5, lr=.1, standardize=True)}


def plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    return plt


def plot_formal(result):
    plt = plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    h = result["history"]
    axes[0].plot([r["epoch"] for r in h[1:]], [r["online_train_ce"] for r in h[1:]], "o-", label="在线训练，Dropout开启")
    axes[0].plot([r["epoch"] for r in h], [r["validation_ce"] for r in h], "o-", label="验证，Dropout关闭")
    axes[1].plot([r["epoch"] for r in h[1:]], [r["online_train_accuracy"] for r in h[1:]], "o-", label="在线训练")
    axes[1].plot([r["epoch"] for r in h], [r["validation_accuracy"] for r in h], "o-", label="验证")
    axes[0].set(ylabel="平均交叉熵", xlabel="epoch")
    axes[1].set(ylabel="准确率", xlabel="epoch", ylim=(0, 1))
    for ax in axes:
        ax.axvline(10, ls=":", color="gray")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(REPORT / "figures/7_1_training.png", dpi=180)
    plt.close(fig)


def run_experiments(device, record):
    splits, hashes = COMMON["load_data"](include_test=False)
    splits = dict(train=COMMON["balanced_subset"](splits["train"], 2000, 7101),
                  validation=COMMON["balanced_subset"](splits["validation"], 1000, 7102))
    results, rows = {}, []
    for name, cfg in candidates().items():
        result, net = fit(cfg, splits, 10, device, name)
        results[name] = result
        rows.extend(dict(candidate=name, **r) for r in result["history"])
        del net
    assert len({results[k]["initial_parameter_sha256"] for k in ("A", "D")}) == 1
    assert len({results[k]["initial_parameter_sha256"] for k in ("L0", "L1", "L2", "L3")}) == 1
    output = dict(split_hashes=hashes, train_subset_seed=7101, validation_subset_seed=7102,
                  train_per_class=200, validation_per_class=100, test_evaluated=False,
                  protocol="预定八方案十轮；只比较各自末轮验证指标，不按测试调参", results=results)
    if record:
        write_json("7_1_experiments.json", output)
        write_csv("7_1_experiments_epochs.csv", rows)
        plt = plot_setup()
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
        for name, r in results.items():
            ax = axes[0] if name in ("A", "B", "C", "D") else axes[1]
            ax.plot(range(11), [v["validation_ce"] for v in r["history"]], label=name)
        for ax, title in zip(axes, ("结构/批量对照", "同起点 LeNet 对照")):
            ax.set(xlabel="epoch", ylabel="验证 CE", title=title)
            ax.legend()
            ax.grid(alpha=.2)
        fig.savefig(REPORT / "figures/7_1_experiments.png", dpi=180)
        plt.close(fig)
    return output


def run_native(device, record):
    """八方案验证反馈后的追加实测，独立记录，不冒称预注册候选或测试选择。"""
    splits, hashes = COMMON["load_data"]()
    cfg = dict(candidates()["C"], lr=.1)
    result, _ = fit(cfg, splits, 15, device, "Native28", test_epochs=(10, 15),
                    record=record, prefix="7_1_native")
    result.update(split_hashes=hashes, split_seed=3601,
                  protocol="观察C在lr=.01/子集160更新的验证学习不足后，追加原始28模型lr=.1完整15轮；固定10/15终点，不按测试选模型",
                  adaptive_followup=True, preprocessing="float32 /255, original28; no resize")
    if record:
        write_json("7_1_native_metrics.json", result)
        write_csv("7_1_native_epochs.csv", result["history"])
    return result


def benchmark(device, record):
    if device.type != "cuda":
        raise RuntimeError("显存与CUDA事件计时需要GPU；CPU可运行 --inspect 或自定义小训练")
    torch.manual_seed(7104)
    torch.cuda.manual_seed_all(7104)
    results = []
    for batch in (16, 64, 128):
        net = build_net().to(device).train()
        optimizer = torch.optim.SGD(net.parameters(), lr=.01)
        X = torch.rand(batch, 1, 224, 224, device=device)
        y = torch.arange(batch, device=device) % 10
        def step():
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(net(X), y).backward()
            optimizer.step()
        for _ in range(2):
            step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            step()
        end.record()
        torch.cuda.synchronize()
        results.append(dict(batch_size=batch, milliseconds_per_step=start.elapsed_time(end) / 5,
                            examples_per_second=batch * 5000 / start.elapsed_time(end),
                            baseline_allocated_bytes=before, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
        del net, optimizer, X, y
        torch.cuda.empty_cache()
    net = build_net().to(device).eval()
    X = torch.rand(8, 1, 224, 224, device=device)
    layer_rows = []
    with torch.no_grad():
        for index, layer in enumerate(net):
            for _ in range(2):
                Y = layer(X)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                Y = layer(X)
            end.record()
            torch.cuda.synchronize()
            milliseconds = start.elapsed_time(end) / 10
            weights = sum(p.numel() for p in layer.parameters())
            logical_bytes = 4 * (X.numel() + Y.numel() + weights)
            layer_rows.append(dict(index=index, layer=type(layer).__name__, milliseconds=milliseconds,
                                   idealized_logical_bytes=logical_bytes,
                                   logical_GB_per_second=logical_bytes / (milliseconds * 1e6)
                                   if milliseconds > 0 and isinstance(layer, (nn.Conv2d, nn.Linear)) else None))
            X = Y
    output = dict(runtime=COMMON["runtime_metadata"](device), synthetic_seed=7104,
                  warmup_steps=2, timed_steps=5, batches=results,
                  forward_layer_batch=8, forward_layers=layer_rows,
                  bandwidth_note="逻辑输入/输出/参数字节除时间是算子流量代理，不是实测DRAM带宽；未计缓存/工作区/融合")
    if record:
        write_json("7_1_benchmark.json", output)
        plt = plot_setup()
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
        axes[0].bar([str(r["batch_size"]) for r in results], [r["peak_allocated_bytes"] / 2**20 for r in results], color="#147e83")
        axes[0].set(xlabel="batch", ylabel="实测分配峰值 MiB")
        axes[1].bar([str(r["batch_size"]) for r in results], [r["examples_per_second"] for r in results], color="#e6ab52")
        axes[1].set(xlabel="batch", ylabel="训练 step 图像数/秒")
        fig.savefig(REPORT / "figures/7_1_benchmark.png", dpi=180)
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ("inspect", "experiments", "benchmark", "native"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("轮数与批量大小须为正整数")
    custom = (args.epochs != 15 or args.batch_size != 128 or args.train_size is not None or args.validation_size is not None)
    if (args.inspect or args.experiments or args.benchmark or args.native) and custom:
        parser.error("预定检查/对照模式不接受自定义训练参数")
    device = COMMON["configure"](args.device)
    if args.guided:
        print("7.1：完整AlexNet用五个卷积层提取表示，三个全连接层分类。", flush=True)
        if args.native:
            print("追加28图小CNN：直接输入28，lr=.1，固定十五轮；当前设备", device, flush=True)
        else:
            print("默认完整模型十五轮含教材十轮和五轮延长；28图按批插值224，当前设备", device, flush=True)
    if args.inspect:
        result = inspect_net()
    elif args.experiments:
        result = run_experiments(device, args.record)
    elif args.benchmark:
        result = benchmark(device, args.record)
    elif args.native:
        result = run_native(device, args.record)
    else:
        splits, hashes = COMMON["load_data"](include_test=not custom)
        for name, size, seed in (("train", args.train_size, 7101), ("validation", args.validation_size, 7102)):
            if size is not None:
                splits[name] = COMMON["balanced_subset"](splits[name], size, seed)
        cfg = dict(BASE, batch_size=args.batch_size)
        prefix = "7_1_custom" if custom else "7_1"
        result, net = fit(cfg, splits, args.epochs, device, "AlexNet", test_epochs=() if custom else (10, 15),
                          record=args.record, prefix=prefix)
        result.update(custom=custom, split_hashes=hashes, split_seed=3601,
                      train_subset_seed=7101 if args.train_size is not None else None,
                      validation_subset_seed=7102 if args.validation_size is not None else None,
                      preprocessing="float32 /255 then batched bilinear resize224, align_corners=False, antialias=True")
        if args.record:
            write_json(prefix + "_metrics.json", result)
            write_csv(prefix + "_epochs.csv", result["history"])
            if not custom:
                plot_formal(result)
    if args.guided:
        if args.inspect:
            for row in result["layers"]:
                print(row["index"], row["layer"], row["output_shape"], "参数", row["parameters"])
            print("参数总数", result["parameters"], "单图权重乘法", result["weight_multiplications"])
        elif args.experiments:
            for name, item in result["results"].items():
                print(name, item["history"][-1])
        elif args.benchmark:
            print(json.dumps(result["batches"], ensure_ascii=False, indent=2))
        else:
            print("预定终点测试", result["tests"], "；custom短实验不评估测试集。")
        print("教师工程结果不代替学习者作答；综合练习仍保留八处TODO。")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
