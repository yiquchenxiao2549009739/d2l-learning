# -*- coding: utf-8 -*-
"""3.7：用 Flatten、Linear、CrossEntropyLoss 和 SGD 完成图像分类。

复用已有 Fashion-MNIST，无下载、无安装。先读 build_model、train_epoch、evaluate。
默认运行正式十轮训练、同起点手写对照和只使用训练/验证数据的小型调参实验。
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST

REPORT = ROOT / "reports/ch03"
FIGURES = REPORT / "figures"


def make_loader(dataset, batch_size=256, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=False, generator=torch.Generator().manual_seed(3603))


def load_data():
    raw = FashionMNIST(ROOT / "data", train=True, download=False)
    # 转换在内存中完成，不改写与其他项目共享的原始 uint8 文件。
    base = TensorDataset(raw.data.unsqueeze(1).float() / 255., raw.targets)
    generator = torch.Generator().manual_seed(3601)
    train_ids, validation_ids = [], []
    for class_id in range(10):
        ids = torch.where(raw.targets == class_id)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        validation_ids.append(ids[:600])
        train_ids.append(ids[600:])
    train_ids, validation_ids = torch.cat(train_ids), torch.cat(validation_ids)
    hashes = {name: hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()
              for name, ids in (("train", train_ids), ("validation", validation_ids))}
    return Subset(base, train_ids), Subset(base, validation_ids), hashes


def build_model(seed=3700, dtype=torch.float32):
    # fork_rng 让每个实验可复用相同起点，且不影响其他随机数流。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(dtype=dtype)

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0., std=.01)
                nn.init.zeros_(module.bias)

        net.apply(init_weights)
    return net


def train_epoch(net, loader, loss_fn, optimizer):
    net.train()
    loss_total, correct, samples = 0., 0, 0
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = net(images)
        losses = loss_fn(logits, labels)  # reduction='none'，每张图片一项。
        losses.mean().backward()
        optimizer.step()  # 梯度已经取平均，不再除批量大小。
        loss_total += losses.detach().sum().item()
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        samples += len(labels)
    return {"online_train_ce": loss_total / samples,
            "online_train_accuracy": correct / samples}


def evaluate(net, loader, loss_fn):
    net.eval()
    loss_total, correct, samples = 0., 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = net(images)
            loss_total += loss_fn(logits, labels).sum().item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += len(labels)
    return {"ce": loss_total / samples, "accuracy": correct / samples,
            "correct": int(correct), "samples": samples}


def fit(net, train_set, validation_set, epochs, lr, batch_size, verbose=False):
    loader = make_loader(train_set, batch_size, shuffle=True)
    train_eval = make_loader(train_set, batch_size)
    validation_loader = make_loader(validation_set, batch_size)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    history = []
    for epoch in range(epochs + 1):
        online = train_epoch(net, loader, loss_fn, optimizer) if epoch else {
            "online_train_ce": None, "online_train_accuracy": None}
        row = {"epoch": epoch, "updates": epoch * len(loader), **online}
        for name, current_loader in (("train", train_eval), ("validation", validation_loader)):
            result = evaluate(net, current_loader, loss_fn)
            row[name + "_ce"] = result["ce"]
            row[name + "_accuracy"] = result["accuracy"]
        history.append(row)
        if verbose:
            print(row, flush=True)
    return history


def audit_equivalence(images, labels):
    """三轮、23张真实图像、8/8/7尾批；同起点逐步比较手写与框架。"""
    images, labels = images[:23].double(), labels[:23]
    generator = torch.Generator().manual_seed(3710)
    weight = (.01 * torch.randn(784, 10, generator=generator, dtype=torch.float64)).requires_grad_()
    bias = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    net = build_model(dtype=torch.float64)
    with torch.no_grad():
        net[1].weight.copy_(weight.T)
        net[1].bias.copy_(bias)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    errors = {"logits": 0., "loss": 0., "gradient": 0., "parameter": 0.}
    sizes = []
    for _ in range(3):
        ids = torch.randperm(23, generator=generator)
        for start in range(0, 23, 8):
            x, y = images[ids[start:start+8]], labels[ids[start:start+8]]
            scores = x.flatten(start_dim=1) @ weight + bias
            shifted = scores - scores.max(dim=1, keepdim=True).values
            log_probs = shifted - shifted.exp().sum(dim=1, keepdim=True).log()
            manual_losses = -log_probs[torch.arange(len(y)), y]
            optimizer.zero_grad(set_to_none=True)
            framework_scores = net(x)
            framework_losses = criterion(framework_scores, y)
            errors["logits"] = max(errors["logits"], (scores-framework_scores).abs().max().item())
            errors["loss"] = max(errors["loss"], (manual_losses-framework_losses).abs().max().item())
            manual_losses.sum().backward()
            framework_losses.mean().backward()
            assert weight.grad is not None and bias.grad is not None
            assert net[1].weight.grad is not None and net[1].bias.grad is not None
            errors["gradient"] = max(errors["gradient"],
                (weight.grad/len(y)-net[1].weight.grad.T).abs().max().item(),
                (bias.grad/len(y)-net[1].bias.grad).abs().max().item())
            with torch.no_grad():
                for parameter in (weight, bias):
                    assert parameter.grad is not None
                    parameter -= .1 * parameter.grad / len(y)
                    parameter.grad.zero_()
            optimizer.step()
            errors["parameter"] = max(errors["parameter"],
                (weight-net[1].weight.T).abs().max().item(), (bias-net[1].bias).abs().max().item())
            sizes.append(len(y))
    assert max(errors.values()) < 1e-12
    extreme = criterion(torch.tensor([[1000., 0., -1000.]]), torch.tensor([2])).item()
    assert extreme == 2000.
    example = torch.tensor([[8., 0., -3.]], dtype=torch.float64)
    target = torch.tensor([0])
    correct_loss = criterion(example, target).item()
    double_softmax_loss = criterion(example.softmax(dim=1), target).item()
    return {"dtype": "float64", "updates": len(sizes), "batch_sizes": sizes,
            "maximum_absolute_errors": errors, "extreme_stable_ce": extreme,
            "correct_logits_ce": correct_loss, "wrong_softmax_then_ce": double_softmax_loss,
            "zero_grad_set_to_none": True}


def balanced_subset(dataset, per_class, seed):
    labels = torch.tensor([int(dataset[i][1]) for i in range(len(dataset))])
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for class_id in range(10):
        ids = torch.where(labels == class_id)[0]
        selected.append(ids[torch.randperm(len(ids), generator=generator)[:per_class]])
    ids = torch.cat(selected)
    return Subset(dataset, ids), hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()


def tuning_experiments(train_set, validation_set):
    # 独立的小教学实验，只用训练/验证子集；不读取官方测试数据。
    small_train, train_hash = balanced_subset(train_set, 50, 3701)
    small_validation, validation_hash = balanced_subset(validation_set, 100, 3702)
    cases = [("baseline", .1, 256, 60), ("small_lr", .01, 256, 10),
             ("large_lr", 1., 256, 10), ("small_batch", .1, 64, 10)]
    summaries, all_rows = [], []
    long_history = None
    for name, lr, batch_size, epochs in cases:
        history = fit(build_model(), small_train, small_validation, epochs, lr, batch_size)
        for row in history:
            all_rows.append({"case": name, "lr": lr, "batch_size": batch_size, **row})
        checkpoints = (10, epochs) if epochs != 10 else (10,)
        for epoch in checkpoints:
            row = history[epoch]
            summaries.append({"case": name, "lr": lr, "batch_size": batch_size,
                              **{k:v for k,v in row.items() if not k.startswith("online_")}})
        if name == "baseline":
            long_history = history
        print("小型训练/验证实验:", summaries[-1], flush=True)
    return {"train_samples": 500, "validation_samples": 1000,
            "per_class": {"train": 50, "validation": 100},
            "subset_seeds": [3701, 3702],
            "subset_index_sha256": {"train": train_hash, "validation": validation_hash},
            "model_seed": 3700, "test_evaluations": 0, "summaries": summaries}, all_rows, long_history


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(history, tuning, long_history):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "font.size": 10,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    teal, orange = "#147E83", "#D47F37"
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set(xlim=(0, 10), ylim=(0, 3.5)); ax.axis("off")
    nodes = [(0.1, "图片", "(n,1,28,28)"), (2.6, "Flatten", "(n,784)"),
             (5.1, "Linear", "logits (n,10)"), (7.6, "CrossEntropyLoss", "losses (n,)")]
    for x, title, shape in nodes:
        ax.add_patch(FancyBboxPatch((x, 1.8), 2.15, 1.05, boxstyle="round,pad=.04",
                                   facecolor="#F1F7F8", edgecolor=teal))
        ax.text(x+1.075, 2.48, title, ha="center", fontsize=11, weight="bold")
        ax.text(x+1.075, 2.06, shape, ha="center", fontsize=10)
        if x < 7:
            ax.annotate("", xy=(x+2.45, 2.32), xytext=(x+2.19, 2.32), arrowprops={"arrowstyle":"->", "color":teal})
    ax.text(8.7, 3.16, "真实类别 y：long / (n,)", ha="center", fontsize=10)
    ax.annotate("", xy=(8.7,2.89), xytext=(8.7,3.02), arrowprops={"arrowstyle":"->"})
    ax.text(5, 1.05, "losses.mean().backward() → 计算梯度", ha="center", color=teal, fontsize=12)
    ax.text(5, .48, "optimizer.step() → 更新 Linear 的 weight 和 bias", ha="center", color=teal, fontsize=12)
    ax.text(5, .02, "模型末尾没有 Softmax；查看概率时另用 logits.softmax(dim=1)。", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES/"3_7_components.png",dpi=170);plt.close(fig)

    def curves(rows, path, title):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for name, label, color in (("train", "训练集重算", teal), ("validation", "验证集", orange)):
            for ax, key in zip(axes, ("ce", "accuracy")):
                ax.plot([r["epoch"] for r in rows], [r[name+"_"+key] for r in rows], label=label, color=color)
        axes[0].set(xlabel="epoch", ylabel="平均交叉熵", title=title)
        axes[1].set(xlabel="epoch", ylabel="准确率", ylim=(0,1), title="固定轮末参数重新评估")
        for ax in axes:
            ax.grid(alpha=.2); ax.legend()
        fig.tight_layout();fig.savefig(FIGURES/path,dpi=170);plt.close(fig)
    curves(history, "3_7_training.png", "正式实验：54,000 训练 / 6,000 验证")
    if tuning is not None:
        curves(long_history, "3_7_epochs.png", "教学子集：500 训练 / 1,000 验证")
        rows = [r for r in tuning["summaries"] if r["epoch"] == 10]
        labels = ["lr=.1\nbatch256", "lr=.01\nbatch256", "lr=1\nbatch256", "lr=.1\nbatch64"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for name, label, color, offset in (("train", "训练", teal, -.18), ("validation", "验证", orange, .18)):
            for ax, key in zip(axes, ("ce", "accuracy")):
                ax.bar([i+offset for i in range(4)], [r[name+"_"+key] for r in rows], width=.35, label=label, color=color)
        for ax in axes:
            ax.set_xticks(range(4), labels);ax.legend();ax.grid(axis="y",alpha=.2)
        axes[0].set(ylabel="平均交叉熵",title="相同子集与初始化，均训练十轮")
        axes[1].set(ylabel="准确率",ylim=(0,1),title="batch64 的每轮更新次数更多")
        fig.tight_layout();fig.savefig(FIGURES/"3_7_tuning.png",dpi=170);plt.close(fig)


def main(epochs=10, lr=.1, batch_size=256, skip_tuning=False):
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True, exist_ok=True);FIGURES.mkdir(parents=True, exist_ok=True)
    train_set, validation_set, hashes = load_data()
    net = build_model()
    audit = audit_equivalence(train_set.dataset.tensors[0], train_set.dataset.tensors[1])
    start = time.perf_counter()
    history = fit(net, train_set, validation_set, epochs, lr, batch_size, verbose=True)
    training_seconds = time.perf_counter() - start
    # 正式配置固定后才读取并评估测试数据；小型调参实验不访问测试集。
    raw_test = FashionMNIST(ROOT/"data", train=False, download=False)
    test_set = TensorDataset(raw_test.data.unsqueeze(1).float()/255., raw_test.targets)
    test = evaluate(net, make_loader(test_set, batch_size), nn.CrossEntropyLoss(reduction="none"))
    checkpoint = ROOT/"artifacts/ch03/3_7_softmax_concise.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "epochs": epochs, "lr": lr,
                "batch_size": batch_size, "model_seed": 3700, "split_seed": 3601}, checkpoint)
    tuning, tuning_rows, long_history = (None, None, None) if skip_tuning else tuning_experiments(train_set, validation_set)
    draw_figures(history, tuning, long_history)
    metrics = {"purpose":"teacher demonstration, not learner completion", "date":"2026-10-20",
        "torch":torch.__version__, "device":"cpu", "dtype":"float32", "download":False,
        "split_sizes":{"train":len(train_set),"validation":len(validation_set),"test":len(test_set)},
        "split_seed":3601,"loader_seed":3603,"model_seed":3700,"split_index_sha256":hashes,
        "epochs":epochs,"learning_rate":lr,"batch_size":batch_size,"updates":history[-1]["updates"],
        "training_tail":len(train_set)%batch_size or batch_size,"weight_shape":list(net[1].weight.shape),
        "bias_shape":list(net[1].bias.shape),"parameter_count":sum(p.numel() for p in net.parameters()),
        "bias_initialization":"explicit zeros; textbook leaves default bias unchanged",
        "initialization_vs_3_6":"different random starting parameters; equivalence audit uses identical parameters",
        "loss_reduction":"none then mean backward; sum only for reporting", "final":history[-1],
        "test":test,"test_evaluations":1,"training_and_epoch_eval_seconds":training_seconds,
        "equivalence_audit":audit,"tuning":tuning,"checkpoint_local_only":"artifacts/ch03/3_7_softmax_concise.pt",
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (REPORT/"3_7_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write_csv(REPORT/"3_7_epochs.csv",history)
    if tuning_rows is not None:
        write_csv(REPORT/"3_7_tuning.csv",tuning_rows)
    print("最终测试:", test, "| 正式训练和逐轮评估秒:", training_seconds)
    print("同起点逐步核对:", audit)
    print("数据与检查点仅本地；小型调参不使用官方测试集。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--lr",type=float,default=.1)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--skip-tuning",action="store_true",help="只跳过额外小型调参，正式训练仍运行")
    args=parser.parse_args()
    if args.epochs<=0 or args.lr<=0 or args.batch_size<=0:
        parser.error("epochs、lr、batch-size 必须为正")
    main(args.epochs,args.lr,args.batch_size,args.skip_tuning)
