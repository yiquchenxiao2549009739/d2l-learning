"""6.6 教材 LeNet 风格网络：形状、真实训练、受控对照和激活图。

默认 Fashion-MNIST 十轮；--inspect 不读数据；--experiments 只比较验证集；
--mnist 运行预定数字识别双方案。--record 才写报告、图和本地检查点。
原始数据只读且 download=False，不安装软件。短训练报告使用 custom 前缀。
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

REPORT = ROOT / "reports/ch06"
NAMES = ["T恤/上衣", "长裤", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
BASE = dict(pool="avg", activation="sigmoid", kernels=[5, 5], channels=[6, 16],
            hidden=[120, 84], extra_conv=False, initializer="xavier", lr=.9)


def configure(device="auto"):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，可用 --device cpu")
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                        else "cpu" if device == "auto" else device)


def build_net(config=None, seed=6600):
    """先在 CPU 确定初值。相同参数形状和初始化规则的候选得到相同权重。"""
    cfg = dict(BASE, **(config or {}))
    if cfg["pool"] not in ("avg", "max") or cfg["activation"] not in ("sigmoid", "relu"):
        raise ValueError("pool 只能为 avg/max，activation 只能为 sigmoid/relu")
    if cfg["initializer"] not in ("xavier", "kaiming"):
        raise ValueError("initializer 只能为 xavier/kaiming")
    if cfg["initializer"] == "kaiming" and cfg["activation"] != "relu":
        raise ValueError("本节 Kaiming 对照只用于 ReLU 隐藏层")
    c1, c2 = cfg["channels"]
    k1, k2 = cfg["kernels"]
    if k1 % 2 != 1 or min(k1, k2, c1, c2) < 1:
        raise ValueError("首核须为正奇数，核尺寸与通道数须为正")
    activation = nn.Sigmoid if cfg["activation"] == "sigmoid" else nn.ReLU
    pooling = nn.AvgPool2d if cfg["pool"] == "avg" else nn.MaxPool2d
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers = [nn.Conv2d(1, c1, k1, padding=k1 // 2), activation(), pooling(2, 2),
                  nn.Conv2d(c1, c2, k2), activation()]
        if cfg["extra_conv"]:
            layers.extend([nn.Conv2d(c2, c2, 3, padding=1), activation()])
        layers.extend([pooling(2, 2), nn.Flatten()])
        side = (14 - k2 + 1) // 2
        if side < 1:
            raise ValueError("第二次汇聚前的空间尺寸太小")
        width = c2 * side * side
        for next_width in cfg["hidden"]:
            layers.extend([nn.Linear(width, next_width), activation()])
            width = next_width
        layers.append(nn.Linear(width, 10))
        net = nn.Sequential(*layers)
        weighted = [layer for layer in net if isinstance(layer, (nn.Conv2d, nn.Linear))]
        for layer in weighted:
            if cfg["initializer"] == "kaiming" and layer is not weighted[-1]:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(layer.weight, gain=1.)
            nn.init.zeros_(layer.bias)
    return net


def state_hash(net):
    digest = hashlib.sha256()
    for name, value in net.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def runtime_metadata(device):
    return dict(device=str(device), torch_version=str(torch.__version__), dtype="float32",
                accelerator=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                deterministic_algorithms=True, tf32=False, num_threads=torch.get_num_threads())


def inspect_net(config=None):
    net = build_net(config)
    X = torch.zeros(2, 1, 28, 28)
    rows = []
    with torch.no_grad():
        for index, layer in enumerate(net):
            X = layer(X)
            params = sum(p.numel() for p in layer.parameters())
            multiplications = 0
            if isinstance(layer, nn.Conv2d):
                multiplications = X[0].numel() * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            elif isinstance(layer, nn.Linear):
                multiplications = layer.in_features * layer.out_features
            rows.append(dict(index=index, layer=type(layer).__name__, shape=list(X.shape),
                             parameters=params, weight_multiplications_per_image=multiplications))
    return dict(input_shape=[2, 1, 28, 28], layers=rows,
                parameters=sum(p.numel() for p in net.parameters()),
                weight_multiplications_per_image=sum(r["weight_multiplications_per_image"] for r in rows),
                initial_state_sha256=state_hash(net))


def make_loader(dataset, shuffle=False, batch_size=256):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=False, generator=torch.Generator().manual_seed(3603))


def load_data(dataset="fashion", include_test=True):
    from torchvision.datasets import FashionMNIST, MNIST
    dataset_class = FashionMNIST if dataset == "fashion" else MNIST
    raw = dataset_class(ROOT / "data", train=True, download=False)
    generator = torch.Generator().manual_seed(3601)
    train_ids, validation_ids = [], []
    for label in range(10):
        ids = torch.where(raw.targets == label)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        validation_ids.append(ids[:600])
        train_ids.append(ids[600:])
    train_ids, validation_ids = torch.cat(train_ids), torch.cat(validation_ids)
    # float() 创建新存储，下面的原地除法不会修改 uint8 原始图像。
    base = TensorDataset(raw.data.unsqueeze(1).float().div_(255.), raw.targets)
    splits = dict(train=Subset(base, train_ids), validation=Subset(base, validation_ids))
    if include_test:
        raw_test = dataset_class(ROOT / "data", train=False, download=False)
        splits["test"] = TensorDataset(raw_test.data.unsqueeze(1).float().div_(255.), raw_test.targets)
    hashes = {name: hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()
              for name, ids in (("train", train_ids), ("validation", validation_ids))}
    return splits, hashes


def balanced_subset(dataset, size, seed):
    if size < 1 or size > len(dataset):
        raise ValueError("子集大小须在 1 与原划分大小之间")
    if isinstance(dataset, Subset):
        labels = dataset.dataset.tensors[1][dataset.indices]
    else:
        labels = dataset.tensors[1]
    generator = torch.Generator().manual_seed(seed)
    ids = []
    for label in range(10):
        candidates = torch.where(labels == label)[0]
        count = size // 10 + int(label < size % 10)
        if count > len(candidates):
            raise ValueError("某类不足以建立指定平衡子集")
        ids.append(candidates[torch.randperm(len(candidates), generator=generator)[:count]])
    ids = torch.cat(ids)
    return Subset(dataset, ids)


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    loss_sum, correct, count = 0., 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = net(X)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return dict(ce=loss_sum / count, accuracy=correct / count, correct=correct, samples=count)


def train_epoch(net, loader, optimizer, device):
    net.train()
    loss_sum, correct, count, updates = 0., 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = net(X)
        loss = F.cross_entropy(logits, y)  # 批平均，只平均这一次。
        if not torch.isfinite(loss):
            raise FloatingPointError("交叉熵非有限；先检查输入、初值和学习率")
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.detach().argmax(1) == y).sum().item()
        count += len(y)
        updates += 1
    return dict(online_train_ce=loss_sum / count, online_train_accuracy=correct / count,
                updates=updates, samples=count)


def fit(config, splits, epochs, device, tag, use_test=False):
    net = build_net(config).to(device)
    initial_hash = state_hash(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"])
    training = make_loader(splits["train"], shuffle=True)
    evaluations = {name: make_loader(splits[name]) for name in ("train", "validation")}
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        online = {} if epoch == 0 else train_epoch(net, training, optimizer, device)
        row = dict(epoch=epoch, **online)
        for name, loader in evaluations.items():
            row.update({name + "_" + key: value for key, value in evaluate(net, loader, device).items()})
        history.append(row)
        print("{} epoch {:2d}: train CE {:.6f}, val CE {:.6f}, val acc {:.2%}".format(
            tag, epoch, row["train_ce"], row["validation_ce"], row["validation_accuracy"]), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = dict(config=config, model_seed=6600, loader_seed=3603, epochs=epochs,
                  runtime=runtime_metadata(device),
                  parameters=sum(p.numel() for p in net.parameters()), initial_state_sha256=initial_hash,
                  updates=sum(row.get("updates", 0) for row in history), history=history,
                  train_samples=len(splits["train"]), validation_samples=len(splits["validation"]),
                  batch_size=256, last_batch_size=(len(splits["train"]) - 1) % 256 + 1,
                  training_and_validation_seconds=time.perf_counter() - started)
    if use_test:
        result["test"] = evaluate(net, make_loader(splits["test"]), device)
    result["final_state_sha256"] = state_hash(net)
    return result, net, optimizer


def write_json(name, value):
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def write_csv(name, rows):
    REPORT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (REPORT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    return plt


def record_formal(result, net, optimizer, splits, device, custom=False):
    prefix = "6_6_custom" if custom else "6_6"
    destination = ROOT / "artifacts/ch06"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / (prefix + "_lenet.pt")
    cpu_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    torch.save(dict(config=result["config"], model=cpu_state, optimizer=optimizer.state_dict(),
                    epochs=result["epochs"], model_seed=6600, split_hashes=result["split_hashes"]), checkpoint)
    restored = build_net(result["config"])
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    restored.eval()
    X, _ = next(iter(make_loader(splits["validation"])))
    with torch.no_grad():
        difference = (restored(X[:32]) - net(X[:32].to(device)).cpu()).abs().max().item()
    result["cpu_checkpoint_max_logit_difference"] = difference
    if difference > 1e-4:
        raise AssertionError("CPU 恢复预测差超过 float32 核对容差")
    write_csv(prefix + "_epochs.csv", result["history"])
    if custom:
        write_json(prefix + "_metrics.json", result)
        return
    plt = plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), constrained_layout=True)
    epochs = [row["epoch"] for row in result["history"]]
    for label, name in (("轮末训练", "train"), ("验证", "validation")):
        axes[0].plot(epochs, [row[name + "_ce"] for row in result["history"]], "o-", label=label)
        axes[1].plot(epochs, [row[name + "_accuracy"] for row in result["history"]], "o-", label=label)
    axes[0].set(ylabel="平均交叉熵", xlabel="epoch")
    axes[1].set(ylabel="准确率", xlabel="epoch", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(REPORT / "figures/6_6_training.png", dpi=170)
    plt.close(fig)
    labels = ["输入\n1×28×28", "卷积+σ\n6×28×28", "平均汇聚\n6×14×14",
              "卷积+σ\n16×10×10", "平均汇聚\n16×5×5", "展平\n400", "全连接+σ\n120 → 84", "logits\n10"]
    fig, ax = plt.subplots(figsize=(10.5, 3.7), constrained_layout=True)
    for i, label in enumerate(labels):
        x, y = i % 4, 1 - i // 4
        ax.text(x, y, label, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=.65", fc="#e7f3f3", ec="#147e83"))
        if i % 4 < 3:
            ax.annotate("", xy=(x + .65, y), xytext=(x + .36, y), arrowprops=dict(arrowstyle="->"))
    ax.text(1.5, .5, "先按上行从左到右，再按下行从左到右", ha="center", color="#536775")
    ax.set(xlim=(-.6, 3.6), ylim=(-.5, 1.5))
    ax.axis("off")
    fig.savefig(REPORT / "figures/6_6_architecture.png", dpi=170)
    plt.close(fig)
    result["feature_maps"] = feature_figures(net, splits["validation"], device, plt)
    write_json("6_6_metrics.json", result)


def feature_figures(net, validation, device, plt):
    # 预定规则：验证划分中套衫/外套的第一张，不按预测或图像效果筛选。
    chosen = []
    for label in (2, 4):
        for index in range(len(validation)):
            if int(validation[index][1]) == label:
                chosen.append(index)
                break
    X = torch.stack([validation[i][0] for i in chosen])
    net.eval()
    with torch.no_grad():
        hidden1 = net[:2](X.to(device)).cpu()
        hidden2 = net[:5](X.to(device)).cpu()
        predictions = net(X.to(device)).argmax(1).cpu().tolist()
    details = dict(validation_positions=chosen, source_train_indices=[int(validation.indices[i]) for i in chosen],
                   labels=[2, 4], predictions=predictions, displayed_channels=list(range(6)), layers={})
    for layer, feature in ((1, hidden1), (2, hidden2)):
        lo, hi = float(feature.min()), float(feature.max())
        details["layers"][str(layer)] = dict(shape=list(feature.shape), vmin=lo, vmax=hi,
                                              stage="post_sigmoid_before_pool")
        fig, axes = plt.subplots(2, 7, figsize=(8.5, 3.3), constrained_layout=True)
        for row, label in enumerate((2, 4)):
            axes[row, 0].imshow(X[row, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(NAMES[label] + "\n预测：" + NAMES[predictions[row]], fontsize=10)
            for channel in range(6):
                im = axes[row, channel + 1].imshow(feature[row, channel], cmap="viridis", vmin=lo, vmax=hi)
                axes[row, channel + 1].set_title("通道 {}".format(channel), fontsize=12)
        for ax in axes.flat:
            ax.set(xticks=[], yticks=[])
        fig.colorbar(im, ax=axes[:, 1:], shrink=.8, label="sigmoid 激活")
        fig.savefig(REPORT / ("figures/6_6_features{}.png".format(layer)), dpi=170)
        plt.close(fig)
    return details


def candidates():
    return {"A": dict(BASE), "B": dict(BASE, pool="max"),
            "C": dict(BASE, kernels=[3, 3]), "D": dict(BASE, channels=[8, 24]),
            "E": dict(BASE, lr=.1), "F": dict(BASE, activation="relu", lr=.1),
            "G": dict(BASE, extra_conv=True), "H": dict(BASE, hidden=[120, 84, 32]),
            "I": dict(BASE, activation="relu", lr=.1, initializer="kaiming")}


def run_experiments(device, record):
    splits, hashes = load_data(include_test=False)
    splits = dict(train=balanced_subset(splits["train"], 2000, 6601),
                  validation=balanced_subset(splits["validation"], 1000, 6602))
    results, rows = {}, []
    for name, config in candidates().items():
        result, _, _ = fit(config, splits, 20, device, name)
        results[name] = result
        rows.extend(dict(candidate=name, **row) for row in result["history"])
    best = min(results, key=lambda name: results[name]["history"][-1]["validation_ce"])
    output = dict(dataset="Fashion-MNIST", split_hashes=hashes, train_subset_seed=6601,
                  validation_subset_seed=6602, train_per_class=200, validation_per_class=100,
                  selection_metric="epoch20 validation CE", selected=best, test_evaluated=False, results=results)
    assert len({results[name]["initial_state_sha256"] for name in ("A", "B", "E", "F")}) == 1
    if record:
        write_json("6_6_experiments.json", output)
        write_csv("6_6_experiments_epochs.csv", rows)
        plt = plot_setup()
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
        for name in results:
            axes[0].plot(range(21), [r["validation_ce"] for r in results[name]["history"]], label=name)
        axes[0].set(xlabel="epoch", ylabel="验证 CE", title="同一平衡子集：九个预定候选")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].bar(list(results), [r["history"][-1]["validation_accuracy"] for r in results.values()], color="#147e83")
        axes[1].set(ylabel="验证准确率", ylim=(0, 1), title="只比较 epoch 20；不读取测试集")
        fig.savefig(REPORT / "figures/6_6_experiments.png", dpi=170)
        plt.close(fig)
    return output


def run_mnist(device, record):
    splits, hashes = load_data("mnist")
    results, rows = {}, []
    for name in ("A", "F"):
        result, _, _ = fit(candidates()[name], splits, 5, device, "MNIST-" + name, use_test=True)
        results[name] = result
        rows.extend(dict(candidate=name, **row) for row in result["history"])
    output = dict(dataset="MNIST", split_hashes=hashes, results=results,
                  protocol="预定 A/F 各五轮；各自最终测试一次；不按测试结果调参")
    if record:
        write_json("6_6_mnist_metrics.json", output)
        write_csv("6_6_mnist_epochs.csv", rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inspect", action="store_true")
    modes.add_argument("--experiments", action="store_true")
    modes.add_argument("--mnist", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if (args.inspect or args.experiments or args.mnist) and any(
            v is not None for v in (args.epochs, args.train_size, args.validation_size)):
        parser.error("形状检查和预定对照不接受自定义轮数或子集参数")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs 须为正整数")
    device = configure(args.device)
    if args.guided:
        print("6.6：卷积检测局部特征，汇聚降低高宽，全连接把特征变成十类 logits。")
        print("先用 --inspect 看 NCHW 和 61706 个参数，再读讲义逐段跟做。")
        print("交叉熵接 logits；模型、图像和 long 标签放在同一设备。当前设备：", device)
    if args.inspect:
        result = inspect_net()
    elif args.experiments:
        result = run_experiments(device, args.record)
    elif args.mnist:
        result = run_mnist(device, args.record)
    else:
        custom = any(v is not None for v in (args.train_size, args.validation_size)) or args.epochs not in (None, 10)
        splits, hashes = load_data(include_test=not custom)
        for name, size, seed in (("train", args.train_size, 6601), ("validation", args.validation_size, 6602)):
            if size is not None:
                splits[name] = balanced_subset(splits[name], size, seed)
        result, net, optimizer = fit(dict(BASE), splits, args.epochs or 10, device, "LeNet", use_test=not custom)
        result.update(dataset="Fashion-MNIST", split_seed=3601, split_hashes=hashes, custom=custom,
                      device=str(device), torch_version=str(torch.__version__), dtype="float32", tf32=False,
                      deterministic_algorithms=True, test_evaluated=not custom,
                      train_subset_seed=6601 if args.train_size is not None else None,
                      validation_subset_seed=6602 if args.validation_size is not None else None)
        if args.record:
            record_formal(result, net, optimizer, splits, device, custom)
    if args.guided:
        if args.inspect:
            for row in result["layers"]:
                print(row["index"], row["layer"], row["shape"], "参数", row["parameters"])
            print("总参数", result["parameters"], "单图权重乘法", result["weight_multiplications_per_image"])
        elif args.experiments:
            print("按本次验证 CE 选", result["selected"], "；只限这九种配置和当前预算。")
        elif args.mnist:
            for name, item in result["results"].items():
                print("MNIST", name, "最终测试", item["test"])
        else:
            print("最终轮末训练/验证：", result["history"][-1])
            print("测试：", result.get("test", "自定义短练习不评估测试集"))
        print("结果是教师实验记录；综合练习仍需你自己完成八处 TODO 并解释。")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
