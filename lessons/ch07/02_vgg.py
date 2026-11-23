"""7.2 VGG：复用块的代码、保留独立参数，核对完整结构并训练教材缩通道版。

默认VGG-11通道/4、224输入、隐藏4096/4096、十轮；--size 96另跑固定尺寸对照。
--inspect用meta张量核对11/16/19与两种尺寸，不读数据；--experiments只用验证子集；
--benchmark单独测完整/缩通道VGG与AlexNet。--record才写报告、图和本地模型。
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

PREVIOUS = runpy.run_path(str(ROOT / "lessons/ch07/01_alexnet.py"))
COMMON = PREVIOUS["COMMON"]
REPORT = ROOT / "reports/ch07"
DEPTHS = {"11": [1, 1, 2, 2, 2], "16": [2, 2, 3, 3, 3], "19": [2, 2, 4, 4, 4]}
CHANNELS = [64, 128, 256, 512, 512]
BASE = dict(version="11", ratio=4, size=224, hidden=[4096, 4096], dropout=.5,
            lr=.05, batch_size=128)


def checked(config=None):
    cfg = dict(BASE, **(config or {}))
    if cfg["version"] not in DEPTHS or cfg["ratio"] not in (1, 2, 4, 8, 16):
        raise ValueError("version须为11/16/19，ratio须为1/2/4/8/16")
    if cfg["size"] < 32 or len(cfg["hidden"]) != 2 or min(cfg["hidden"]) < 1:
        raise ValueError("五次汇聚需要至少32输入，并给出两个正整数隐藏宽度")
    if not 0 <= cfg["dropout"] < 1 or cfg["lr"] <= 0 or cfg["batch_size"] < 1:
        raise ValueError("检查Dropout、学习率和批量大小")
    return cfg


def vgg_block(num_convs, in_channels, out_channels, device=None):
    """每次调用构造新层；共享代码模板，不绑定不同块的参数。"""
    if num_convs < 1 or min(in_channels, out_channels) < 1:
        raise ValueError("卷积数和通道数须为正整数")
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


def build_net(config=None, device="cpu", seed=7200):
    cfg = checked(config)
    if torch.device(device).type not in ("cpu", "meta"):
        raise ValueError("先在CPU构造并初始化，再整体to目标设备；meta仅用于形状检查")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        blocks, in_channels = [], 1
        for count, channels in zip(DEPTHS[cfg["version"]], CHANNELS):
            out_channels = channels // cfg["ratio"]
            blocks.append(vgg_block(count, in_channels, out_channels, device=device))
            in_channels = out_channels
        side = cfg["size"] // 32
        a, b = cfg["hidden"]
        net = nn.Sequential(*blocks, nn.Flatten(),
                            nn.Linear(in_channels * side * side, a, device=device), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                            nn.Linear(a, b, device=device), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                            nn.Linear(b, 10, device=device))
        if str(device) != "meta":
            # 同形状层按7200+j独立初始化，224/96共用卷积及同形状FC的初值。
            for index, layer in enumerate(m for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))):
                torch.default_generator.manual_seed(seed + index)
                nn.init.xavier_uniform_(layer.weight, gain=1.)
                nn.init.zeros_(layer.bias)
    return net


def parameter_hash(net, convolution_only=False):
    digest = hashlib.sha256()
    for module in net.modules():
        if isinstance(module, nn.Conv2d) or (isinstance(module, nn.Linear) and not convolution_only):
            for p in module.parameters(recurse=False):
                digest.update(str(tuple(p.shape)).encode())
                digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def inspect_net(config=None):
    cfg = checked(config)
    net = build_net(cfg, device="meta").eval()
    weighted, outer, handles = [], [], []
    def weighted_hook(name):
        def hook(layer, inputs, output):
            if isinstance(layer, nn.Conv2d):
                multiply = output.numel() * layer.in_channels * 9
            else:
                multiply = layer.in_features * layer.out_features
            weighted.append(dict(name=name, layer=type(layer).__name__, input_shape=list(inputs[0].shape),
                                 output_shape=list(output.shape), parameters=sum(p.numel() for p in layer.parameters()),
                                 weight_multiplications=multiply, output_float32_bytes=output.numel() * 4))
        return hook
    def outer_hook(name):
        def hook(layer, inputs, output):
            outer.append(dict(name=name, layer=type(layer).__name__, shape=list(output.shape),
                              recursive_parameters=sum(p.numel() for p in layer.parameters())))
        return hook
    for name, layer in net.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(weighted_hook(name)))
    for name, layer in net.named_children():
        handles.append(layer.register_forward_hook(outer_hook(name)))
    with torch.no_grad():
        net(torch.zeros(1, 1, cfg["size"], cfg["size"], device="meta"))
    for handle in handles:
        handle.remove()
    count = sum(p.numel() for p in net.parameters())
    return dict(config=cfg, block_architecture=list(zip(DEPTHS[cfg["version"]], [c // cfg["ratio"] for c in CHANNELS])),
                outer=outer, weighted_layers=weighted, parameters=count, parameter_float32_bytes=count * 4,
                weight_multiplications=sum(r["weight_multiplications"] for r in weighted),
                conv_count=sum(r["layer"] == "Conv2d" for r in weighted), linear_count=3,
                note="meta仅作形状和理论计数，不代表真实计算/分配峰值")


def structures():
    return {"full11": inspect_net(dict(BASE, ratio=1)), "quarter11": inspect_net(),
            "quarter96": inspect_net(dict(BASE, size=96)),
            "full16": inspect_net(dict(BASE, version="16", ratio=1)),
            "full19": inspect_net(dict(BASE, version="19", ratio=1))}


def prepare(X, cfg, device):
    return F.interpolate(X.to(device), size=(cfg["size"], cfg["size"]), mode="bilinear",
                         align_corners=False, antialias=True)


@torch.no_grad()
def evaluate(net, dataset, cfg, device):
    net.eval()
    total, correct, count = 0., 0, 0
    for X, y in COMMON["make_loader"](dataset, batch_size=cfg["batch_size"]):
        X, y = prepare(X, cfg, device), y.to(device)
        logits = net(X)
        total += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return dict(ce=total / count, accuracy=correct / count, correct=correct, samples=count)


def train_epoch(net, loader, optimizer, cfg, device):
    net.train()
    total, correct, count, updates = 0., 0, 0, 0
    for X, y in loader:
        X, y = prepare(X, cfg, device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = net(X)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("损失非有限，请检查输入、学习率与初值")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        correct += (logits.detach().argmax(1) == y).sum().item()
        count += len(y)
        updates += 1
    return dict(online_train_ce=total / count, online_train_accuracy=correct / count, updates=updates, samples=count)


def fit(cfg, splits, epochs, device, name, use_test=False, record=False, prefix=None):
    net = build_net(cfg).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg["lr"])
    loader = COMMON["make_loader"](splits["train"], shuffle=True, batch_size=cfg["batch_size"])
    torch.manual_seed(7203)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7203)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    result = dict(config=cfg, model_seed_base=7200, initialization="Xavier gain1, per-weighted-layer seed7200+j; zero bias",
                  dropout_seed=7203, loader_seed=3603, initial_parameter_sha256=parameter_hash(net),
                  initial_conv_sha256=parameter_hash(net, True), parameters=sum(p.numel() for p in net.parameters()),
                  runtime=COMMON["runtime_metadata"](device), epochs=epochs, train_samples=len(splits["train"]),
                  validation_samples=len(splits["validation"]), history=[])
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        online = {} if epoch == 0 else train_epoch(net, loader, optimizer, cfg, device)
        validation = evaluate(net, splits["validation"], cfg, device)
        row = dict(epoch=epoch, **online, **{"validation_" + k: v for k, v in validation.items()})
        if epoch == epochs:
            row.update({"train_eval_" + k: v for k, v in evaluate(net, splits["train"], cfg, device).items()})
        result["history"].append(row)
        print("{} epoch {:2d}: online CE {}, val CE {:.6f}, val acc {:.2%}".format(name, epoch,
              "-" if not online else "{:.6f}".format(online["online_train_ce"]), validation["ce"], validation["accuracy"]), flush=True)
    if use_test:
        result["test"] = evaluate(net, splits["test"], cfg, device)
    result.update(updates=sum(r.get("updates", 0) for r in result["history"]),
                  last_batch_size=(len(splits["train"]) - 1) % cfg["batch_size"] + 1,
                  final_parameter_sha256=parameter_hash(net), test_evaluated=use_test,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    if record:
        folder = ROOT / "artifacts/ch07"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(config=cfg, epoch=epochs, model={k: v.detach().cpu() for k, v in net.state_dict().items()},
                        optimizer=optimizer.state_dict()), folder / (prefix + ".pt"))
    if device.type == "cuda":
        torch.cuda.synchronize()
    result["seconds_including_evaluation_and_save"] = time.perf_counter() - started
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
    small = dict(BASE, hidden=[256, 128])
    return {"A": small, "B": dict(small, version="16"), "C": dict(small, version="19"),
            "D": dict(small, size=96)}


def plot_setup():
    return PREVIOUS["plot_setup"]()


def plot_training():
    plt = plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    for size, filename in ((224, "7_2_metrics.json"), (96, "7_2_96_metrics.json")):
        path = REPORT / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf8"))
        h = data["history"]
        axes[0].plot([r["epoch"] for r in h], [r["validation_ce"] for r in h], "o-", label=str(size) + "验证")
        axes[1].plot([r["epoch"] for r in h], [r["validation_accuracy"] for r in h], "o-", label=str(size) + "验证")
    axes[0].set(xlabel="epoch", ylabel="验证CE")
    axes[1].set(xlabel="epoch", ylabel="验证准确率", ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(REPORT / "figures/7_2_training.png", dpi=180)
    plt.close(fig)


def run_experiments(device, record):
    splits, hashes = COMMON["load_data"](include_test=False)
    splits = dict(train=COMMON["balanced_subset"](splits["train"], 2000, 7201),
                  validation=COMMON["balanced_subset"](splits["validation"], 1000, 7202))
    results, rows = {}, []
    for name, cfg in candidates().items():
        result, net = fit(cfg, splits, 20, device, name)
        results[name] = result
        rows.extend(dict(candidate=name, **r) for r in result["history"])
        del net
    assert results["A"]["initial_conv_sha256"] == results["D"]["initial_conv_sha256"]
    output = dict(split_hashes=hashes, train_subset_seed=7201, validation_subset_seed=7202,
                  train_per_class=200, validation_per_class=100, test_evaluated=False, results=results,
                  protocol="预定四方案20轮、320更新；小分类头256/128；按末轮验证比较，不读测试")
    if record:
        write_json("7_2_experiments.json", output)
        write_csv("7_2_experiments_epochs.csv", rows)
        plt = plot_setup()
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
        for name, result in results.items():
            h = result["history"]
            axes[0].plot(range(21), [r["validation_ce"] for r in h], label=name)
        axes[0].set(xlabel="epoch", ylabel="验证CE")
        axes[0].legend()
        axes[1].bar(list(results), [v["history"][-1]["validation_accuracy"] for v in results.values()], color="#147e83")
        axes[1].set(ylabel="末轮验证准确率", ylim=(0, 1))
        fig.savefig(REPORT / "figures/7_2_experiments.png", dpi=180)
        plt.close(fig)
    return output


def benchmark(device, record):
    if device.type != "cuda":
        raise RuntimeError("CUDA计时与显存测量需要GPU；CPU可做inspect或custom小训练")
    specs = [("AlexNet224", None), ("VGG11-full224", dict(BASE, ratio=1)),
             ("VGG11-quarter224", dict(BASE)), ("VGG11-quarter96", dict(BASE, size=96))]
    results = []
    for name, cfg in specs:
        net = PREVIOUS["build_net"]() if cfg is None else build_net(cfg)
        net = net.to(device).train()
        size = 224 if cfg is None else cfg["size"]
        torch.manual_seed(7204)
        torch.cuda.manual_seed_all(7204)
        X = torch.rand(8, 1, size, size, device=device)
        y = torch.arange(8, device=device) % 10
        optimizer = torch.optim.SGD(net.parameters(), lr=.05)
        def step():
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(net(X), y).backward()
            optimizer.step()
        for _ in range(2):
            step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            step()
        end.record()
        torch.cuda.synchronize()
        results.append(dict(model=name, batch_size=8, size=size, milliseconds_per_step=start.elapsed_time(end) / 5,
                            baseline_allocated_bytes=baseline, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
        del net, optimizer, X, y
        torch.cuda.empty_cache()
    output = dict(runtime=COMMON["runtime_metadata"](device), synthetic_seed=7204, warmup=2, repeats=5, results=results,
                  protocol="单独顺序运行；随机GPU驻留输入，不含加载/Resize；不能当分类精度或长程稳定吞吐量")
    if record:
        write_json("7_2_benchmark.json", output)
        plt = plot_setup()
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
        labels = ["Alex224", "VGG全224", "VGG缩224", "VGG缩96"]
        axes[0].bar(labels, [r["milliseconds_per_step"] for r in results], color="#147e83")
        axes[0].set(ylabel="训练step毫秒")
        axes[1].bar(labels, [r["peak_allocated_bytes"] / 2**20 for r in results], color="#e6ab52")
        axes[1].set(ylabel="分配峰值MiB")
        fig.savefig(REPORT / "figures/7_2_benchmark.png", dpi=180)
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ("inspect", "experiments", "benchmark"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--size", type=int, choices=(224, 96), default=224)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("轮数与批量须为正整数")
    custom = args.epochs != 10 or args.batch_size != 128 or args.train_size is not None or args.validation_size is not None
    if (args.inspect or args.experiments or args.benchmark) and (custom or args.size != 224):
        parser.error("结构汇总/预定对照不接受额外训练或尺寸参数")
    device = COMMON["configure"](args.device)
    if args.guided:
        print("7.2：Conv3×3→ReLU重复，再MaxPool2×2；复用块模板不共享不同层的参数。", flush=True)
        print("默认教材通道/4，两个隐藏层仍4096；实际数据入口尺寸", args.size, "设备", device, flush=True)
    if args.inspect:
        result = structures()
        if args.record:
            write_json("7_2_architectures.json", result)
    elif args.experiments:
        result = run_experiments(device, args.record)
    elif args.benchmark:
        result = benchmark(device, args.record)
    else:
        splits, hashes = COMMON["load_data"](include_test=not custom)
        for name, size, seed in (("train", args.train_size, 7201), ("validation", args.validation_size, 7202)):
            if size is not None:
                splits[name] = COMMON["balanced_subset"](splits[name], size, seed)
        cfg = dict(BASE, size=args.size, batch_size=args.batch_size)
        prefix = "7_2" + ("_96" if args.size == 96 else "") + ("_custom" if custom else "")
        result, _ = fit(cfg, splits, args.epochs, device, "VGG11/4-" + str(args.size),
                        use_test=not custom, record=args.record, prefix=prefix)
        result.update(custom=custom, split_hashes=hashes, split_seed=3601,
                      train_subset_seed=7201 if args.train_size is not None else None,
                      validation_subset_seed=7202 if args.validation_size is not None else None,
                      preprocessing="float32 /255 then batched bilinear, align_corners=False, antialias=True")
        if args.record:
            write_json(prefix + "_metrics.json", result)
            write_csv(prefix + "_epochs.csv", result["history"])
            if not custom:
                plot_training()
    if args.guided:
        if args.inspect:
            for name, value in result.items():
                print(name, "参数", value["parameters"], "含权重层", len(value["weighted_layers"]),
                      "单图乘法", value["weight_multiplications"])
            print("full11顶层实际执行行数", len(result["full11"]["outer"]), "；叶子卷积8加Linear3共11。")
        elif args.experiments:
            for name, value in result["results"].items():
                print(name, value["history"][-1])
        elif args.benchmark:
            print(json.dumps(result["results"], ensure_ascii=False, indent=2))
        else:
            print("最终测试", result.get("test", "custom不读取测试集"))
        print("教师运行不等于学习者掌握；综合练习八处TODO待自己完成。")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
