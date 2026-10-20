# -*- coding: utf-8 -*-
"""3.4 softmax 回归：得分、概率、交叉熵、梯度和分类评估。

人工二维数据仅用于说明原理，不是真实猫/鸡/狗图像。无下载、无安装。
先读 stable_softmax、audit_theory 和 train，再读报告与绘图辅助部分。
"""
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

DTYPE = torch.float64
NAMES = ["猫", "鸡", "狗"]
REPORT = ROOT / "reports/ch03"
FIGURES = REPORT / "figures"
DATA = ROOT / "datasets/ch03"


def stable_softmax(logits):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    numerator = shifted.exp()
    return numerator / numerator.sum(dim=1, keepdim=True)


def create_splits():
    generator = torch.Generator().manual_seed(3401)
    centers = torch.tensor([[-2., -1.], [2., -1.], [0., 2.]], dtype=DTYPE)
    splits = {}
    for name, per_class in (("train", 80), ("validation", 30), ("test", 30)):
        labels = torch.arange(3).repeat_interleave(per_class)
        features = centers[labels] + .85 * torch.randn(
            len(labels), 2, generator=generator, dtype=DTYPE)
        order = torch.randperm(len(labels), generator=generator)
        splits[name] = (features[order], labels[order])
    return splits


def make_model():
    net = nn.Linear(2, 3, dtype=DTYPE)
    nn.init.normal_(net.weight, std=.01,
                    generator=torch.Generator().manual_seed(3402))
    nn.init.zeros_(net.bias)
    return net


def make_loader(features, labels):
    return DataLoader(TensorDataset(features, labels), batch_size=24,
                      shuffle=True, num_workers=0, drop_last=False,
                      generator=torch.Generator().manual_seed(3403))


def evaluate(net, features, labels):
    with torch.no_grad():
        logits = net(features)
        return {"ce": F.cross_entropy(logits, labels).item(),
                "accuracy": (logits.argmax(dim=1) == labels).double().mean().item()}


def train(splits, epochs=40, lr=.1):
    features, labels = splits["train"]
    loader = make_loader(features, labels)
    net = make_model()
    loss_fn = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    history, step = [], 0

    def record(epoch):
        row = {"epoch": epoch, "updates": step}
        for name in ("train", "validation"):
            for key, value in evaluate(net, *splits[name]).items():
                row[name + "_" + key] = value
        history.append(row)

    record(0)
    for epoch in range(epochs):
        for x, y in loader:
            trainer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = loss_fn(logits, y)
            loss.backward()
            trainer.step()
            step += 1
        record(epoch + 1)
    return net, history


def audit_theory():
    logits = torch.tensor([[2., 1., 0.], [0., -1., 2.]],
                          dtype=DTYPE, requires_grad=True)
    labels = torch.tensor([0, 2], dtype=torch.long)
    one_hot = F.one_hot(labels, num_classes=3).to(DTYPE)
    probabilities = stable_softmax(logits)
    log_probabilities = F.log_softmax(logits, dim=1)
    manual = -log_probabilities[torch.arange(2), labels].mean()
    builtin = F.cross_entropy(logits, labels)
    soft_target = F.cross_entropy(logits, one_hot)
    assert torch.allclose(manual, builtin) and torch.allclose(builtin, soft_target)
    builtin.backward()
    expected = (probabilities.detach() - one_hot) / len(labels)
    assert torch.allclose(logits.grad, expected, atol=1e-14)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, dtype=DTYPE))
    assert torch.equal(logits.argmax(dim=1), probabilities.argmax(dim=1))
    assert torch.allclose(stable_softmax(logits + 1234.), probabilities)

    extreme = torch.tensor([[1000., 1001., 1002.], [-1000., -1001., -1002.]], dtype=DTYPE)
    naive = extreme.exp() / extreme.exp().sum(dim=1, keepdim=True)
    stable = stable_softmax(extreme)
    assert not torch.isfinite(naive).all() and torch.isfinite(stable).all()
    hard_wrong = torch.tensor([[1000., 0., -1000.]], dtype=DTYPE)
    stable_wrong_ce = F.cross_entropy(hard_wrong, torch.tensor([2])).item()
    assert stable_wrong_ce == 2000.
    assert torch.softmax(hard_wrong, dim=1)[0, 2] == 0

    point = logits[0].detach()
    p = torch.softmax(point, dim=0)
    hessian = torch.autograd.functional.hessian(
        lambda z: torch.logsumexp(z, dim=0) - z[0], point)
    analytic = torch.diag(p) - torch.outer(p, p)
    centered = torch.eye(3, dtype=DTYPE) - p
    covariance = (centered.T * p) @ centered
    assert torch.allclose(hessian, analytic, atol=1e-14)
    assert torch.allclose(hessian, covariance, atol=1e-14)
    assert torch.linalg.eigvalsh(hessian).min() >= -1e-14
    assert torch.allclose(hessian.sum(dim=1), torch.zeros(3, dtype=DTYPE), atol=1e-14)

    true_p = torch.tensor([.1, .2, .7], dtype=DTYPE)
    assumed_q = torch.tensor([.2, .2, .6], dtype=DTYPE)
    entropy = -(true_p * true_p.log()).sum().item()
    cross_entropy = -(true_p * assumed_q.log()).sum().item()
    kl = (true_p * (true_p / assumed_q).log()).sum().item()
    assert abs(cross_entropy - entropy - kl) < 1e-14
    coding = [{"block_length": n, "fixed_bits_per_block": math.ceil(n * math.log2(3)),
               "bits_per_symbol": math.ceil(n * math.log2(3)) / n}
              for n in (1, 2, 3, 5, 10, 100)]
    lse = []
    for scale in (1., 2., 5., 10., 100.):
        smooth_max = (torch.logsumexp(scale * point, dim=0) / scale).item()
        smooth_min = (-torch.logsumexp(-scale * point, dim=0) / scale).item()
        assert 2. - 1e-14 <= smooth_max <= 2. + math.log(3) / scale
        assert -math.log(3) / scale <= smooth_min <= 1e-14
        lse.append({"lambda": scale, "smooth_max": smooth_max, "smooth_min": smooth_min})
    temperatures = []
    for temperature in (.25, .5, 1., 2., 5.):
        pt = torch.softmax(point / temperature, dim=0)
        temperatures.append({"temperature": temperature,
                             "p0": pt[0].item(), "p1": pt[1].item(), "p2": pt[2].item(),
                             "entropy_nats": -(pt * pt.log()).sum().item()})
    result = {"logits": logits.detach().tolist(), "labels": labels.tolist(),
              "probabilities": probabilities.detach().tolist(), "mean_ce": builtin.item(),
              "logit_gradient": logits.grad.tolist(), "hessian_single_sample": hessian.tolist(),
              "hessian_equals_one_hot_covariance": True,
              "naive_extreme_nonfinite": True, "stable_extreme_probabilities": stable.tolist(),
              "underflowed_probability_but_stable_ce": stable_wrong_ce,
              "double_softmax_wrong_ce": F.cross_entropy(probabilities, labels).item(),
              "entropy_nats": entropy, "entropy_bits": entropy / math.log(2),
              "cross_entropy_nats": cross_entropy, "kl_nats": kl,
              "uniform_3_entropy_bits": math.log2(3)}
    return result, coding, lse, temperatures


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def draw_figures(net, splits, history, audit, temperatures):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "font.size": 12,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    palette = ["#147E83", "#D47F37", "#536FA4"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(NAMES, audit["logits"][0], color=palette)
    axes[0].set(title="同一样本：类别得分", ylabel="logit，可正可负")
    axes[1].bar(NAMES, audit["probabilities"][0], color=palette)
    axes[1].set(title="softmax 后：总和为 1", ylabel="预测概率", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(FIGURES / "3_4_scores_probabilities.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    probabilities = np.linspace(.005, 1., 500)
    axes[0].plot(probabilities, -np.log(probabilities), color=palette[0])
    for v in (.1, .5, .9):
        axes[0].scatter(v, -math.log(v), color=palette[1])
    axes[0].set(xlabel="给真实类别的预测概率", ylabel="交叉熵（nat）", title="越自信地预测错，惩罚越大")
    for j, (name, color) in enumerate(zip(NAMES, palette)):
        axes[1].plot([r["temperature"] for r in temperatures],
                     [r[f"p{j}"] for r in temperatures], "o-", label=name, color=color)
    axes[1].set(xlabel="温度 T", ylabel="softmax([2,1,0] / T)", title="概率改变，最大项类别不变")
    axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "3_4_loss_temperature.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, label, color in (("train", "训练", palette[0]), ("validation", "验证", palette[1])):
        axes[0].plot([r["epoch"] for r in history], [r[name + "_ce"] for r in history], label=label, color=color)
        axes[1].plot([r["epoch"] for r in history], [r[name + "_accuracy"] for r in history], label=label, color=color)
    axes[0].set(xlabel="epoch", ylabel="平均交叉熵（nat）", title="损失追踪概率质量")
    axes[1].set(xlabel="epoch", ylabel="准确率", ylim=(0, 1.03), title="准确率只看类别是否选对")
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "3_4_training.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    gx, gy = np.meshgrid(np.linspace(-4.6, 4.6, 180), np.linspace(-4., 4.6, 180))
    points = torch.tensor(np.column_stack((gx.ravel(), gy.ravel())), dtype=DTYPE)
    with torch.no_grad(): region = net(points).argmax(dim=1).numpy().reshape(gx.shape)
    from matplotlib.colors import ListedColormap
    axes[0].contourf(gx, gy, region, levels=[-.5, .5, 1.5, 2.5],
                     cmap=ListedColormap(palette), alpha=.18)
    x, y = splits["validation"]
    for j in range(3):
        axes[0].scatter(x[y == j, 0], x[y == j, 1], s=23, label=NAMES[j], color=palette[j])
    axes[0].set(xlabel="人工特征 x1", ylabel="人工特征 x2", title="验证样本与学到的分类区域")
    axes[0].legend(fontsize=10)
    xt, yt = splits["test"]
    with torch.no_grad(): predicted = net(xt).argmax(dim=1)
    matrix = torch.bincount(yt * 3 + predicted, minlength=9).reshape(3, 3).numpy()
    axes[1].imshow(matrix, cmap="Blues", vmin=0, vmax=30)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center",
                         color="white" if matrix[i, j] > 15 else "#173448")
    axes[1].set(xticks=range(3), yticks=range(3), xticklabels=NAMES, yticklabels=NAMES,
                xlabel="预测类别", ylabel="真实类别", title="固定方案：测试集计数")
    fig.tight_layout(); fig.savefig(FIGURES / "3_4_decisions.png", dpi=170); plt.close(fig)
    return matrix.tolist()


def main():
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    splits = create_splits()
    for name, (x, y) in splits.items():
        write_csv(DATA / f"softmax_{name}.csv",
                  [{"x1": row[0].item(), "x2": row[1].item(), "class_id": int(label)}
                   for row, label in zip(x, y)])
    net, history = train(splits)
    audit, coding, lse, temperatures = audit_theory()
    matrix = draw_figures(net, splits, history, audit, temperatures)
    metrics = {"purpose": "teacher demonstration, not learner completion",
               "date": "2026-10-20", "torch": torch.__version__, "device": "cpu", "dtype": "float64",
               "artificial_class_names": NAMES, "split_sizes": {k: len(v[1]) for k, v in splits.items()},
               "data_seed": 3401, "init_seed": 3402, "loader_seed": 3403,
               "batch_size": 24, "epochs": 40, "updates": 400, "learning_rate": .1,
               "loss": "mean cross entropy on raw logits, class indices",
               "weight": net.weight.detach().tolist(), "bias": net.bias.detach().tolist(),
               "final_train": evaluate(net, *splits["train"]),
               "final_validation": evaluate(net, *splits["validation"]),
               "final_test": evaluate(net, *splits["test"]),
               "test_confusion_true_rows": matrix, "theory_checks": audit}
    with (REPORT / "3_4_metrics.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for filename, values in (("epochs", history), ("coding", coding),
                             ("smooth_max", lse), ("temperature", temperatures)):
        write_csv(REPORT / f"3_4_{filename}.csv", values)
    print("3.4 softmax 回归 |", torch.__version__, "| CPU / float64")
    print("人工数据，不是真实图像；类别映射:", dict(enumerate(NAMES)))
    for row in history:
        if row["epoch"] in (0, 1, 10, 20, 40): print(row)
    print("测试集固定方案:", metrics["final_test"])
    print("第一条示例概率:", audit["probabilities"][0], "| 示例平均交叉熵:", audit["mean_ce"])
    print("示例 logits 梯度:", audit["logit_gradient"])
    print("原始指数不稳定，稳定交叉熵可处理真类概率下溢，CE:", audit["underflowed_probability_but_stable_ce"])
    print("单样本 Hessian = 独热随机向量协方差:", audit["hessian_equals_one_hot_covariance"])
    print("三类等概率熵（bit）:", audit["uniform_3_entropy_bits"], "| 分组编码:", coding)
    print("指标、四张图、人工 CSV 已保存在本项目。")


if __name__ == "__main__":
    main()
