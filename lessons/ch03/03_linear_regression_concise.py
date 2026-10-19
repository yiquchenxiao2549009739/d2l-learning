# -*- coding: utf-8 -*-
"""3.3 简洁实现：DataLoader、Linear、MSELoss 与 optim.SGD。

复用 3.2 已同步的人工 CSV，核心训练函数与报告辅助代码分开。
已有 openmmlab 即可运行，无安装、无下载。
"""

import csv
import json
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPORT = ROOT / "reports" / "ch03"
FIGURES = REPORT / "figures"
DTYPE = torch.float64


def load_splits():
    splits = {}
    for name in ("train", "validation", "test"):
        with (ROOT / "datasets" / "ch03" / f"scratch_{name}.csv").open(encoding="utf-8") as f:
            values = [[float(row[key]) for key in ("x1", "x2", "y")]
                      for row in csv.DictReader(f)]
        array = torch.tensor(values, dtype=DTYPE)
        splits[name] = (array[:, :2], array[:, 2:])
    return splits


def make_loader(features, labels, batch_size=10, shuffle=True, drop_last=False):
    dataset = TensorDataset(features, labels)
    generator = torch.Generator().manual_seed(3303)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=0, generator=generator)


def make_model():
    net = nn.Sequential(nn.Linear(2, 1, dtype=DTYPE))
    generator = torch.Generator().manual_seed(3302)
    nn.init.normal_(net[0].weight, mean=0., std=0.01, generator=generator)
    nn.init.zeros_(net[0].bias)
    return net


def evaluate(net, features, labels):
    with torch.no_grad():
        prediction = net(features)
        assert prediction.shape == labels.shape
        return (prediction - labels).square().mean().item()


def train(splits, loss_fn=None, lr=0.03, epochs=3):
    features, labels = splits["train"]
    xv, yv = splits["validation"]
    loader = make_loader(features, labels)
    net = make_model()
    if loss_fn is None:
        loss_fn = nn.MSELoss(reduction="mean")
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    epoch_rows, step_rows = [], []
    step = 0

    def record_epoch(epoch):
        epoch_rows.append({"epoch": epoch, "steps": step,
                           "train_mse": evaluate(net, features, labels),
                           "validation_mse": evaluate(net, xv, yv)})

    record_epoch(0)
    step_rows.append({"step": 0, "train_mse_after": epoch_rows[0]["train_mse"],
                      "w1": net[0].weight[0, 0].item(),
                      "w2": net[0].weight[0, 1].item(), "b": net[0].bias.item()})
    for epoch in range(epochs):
        for x, y in loader:
            trainer.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            loss.backward()
            trainer.step()
            step += 1
            step_rows.append({"step": step, "train_mse_after": evaluate(net, features, labels),
                              "w1": net[0].weight[0, 0].item(),
                              "w2": net[0].weight[0, 1].item(), "b": net[0].bias.item()})
        record_epoch(epoch + 1)
    return net, epoch_rows, step_rows


def compare_with_scratch(features, labels):
    """同批、同起点：MSE/lr=.015 对照手写半平方损失/lr=.03。"""
    net = make_model()
    w = net[0].weight.detach().T.clone().requires_grad_()
    b = net[0].bias.detach().clone().requires_grad_()
    trainer = torch.optim.SGD(net.parameters(), lr=0.015)
    loss_fn = nn.MSELoss()
    loader = make_loader(features, labels)
    rows, step = [], 0
    for _ in range(3):
        for x, y in loader:
            trainer.zero_grad(set_to_none=True)
            loss_fn(net(x), y).backward()
            trainer.step()
            half_losses = (x @ w + b - y).square() / 2
            half_losses.sum().backward()
            with torch.no_grad():
                for param in (w, b):
                    assert param.grad is not None
                    param -= 0.03 * param.grad / len(x)
                    param.grad.zero_()
                difference = max((net[0].weight.T - w).abs().max().item(),
                                 (net[0].bias - b).abs().max().item())
                step += 1
                rows.append({"step": step, "max_parameter_difference": difference,
                             "framework_mse": evaluate(net, features, labels),
                             "scratch_mse": (features @ w + b - labels).square().mean().item()})
    assert max(row["max_parameter_difference"] for row in rows) < 1e-12
    return rows


def audit_api(features, labels):
    ids = torch.arange(23, dtype=DTYPE).reshape(-1, 1)
    kept = list(make_loader(ids, ids + 100))
    dropped = list(make_loader(ids, ids + 100, drop_last=True))
    assert sorted(int(v) for x, _ in kept for v in x.ravel()) == list(range(23))
    assert all(torch.equal(y, x + 100) for x, y in kept)
    x, y = next(iter(make_loader(features, labels)))
    net = make_model()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    trainer.zero_grad(set_to_none=True)
    residual = net(x) - y
    nn.MSELoss()(net(x), y).backward()
    assert net[0].weight.grad is not None and net[0].bias.grad is not None
    gradient = net[0].weight.grad.detach().clone()
    assert torch.allclose(gradient, (2 * x.T @ residual.detach() / len(x)).T)
    trainer.step()
    assert torch.equal(net[0].weight.grad, gradient)
    trainer.zero_grad(set_to_none=True)
    assert net[0].weight.grad is None and net[0].bias.grad is None

    prediction = torch.tensor([[1.], [2.], [3.]], dtype=DTYPE)
    target = torch.tensor([1., 2., 3.], dtype=DTYPE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrong_mse = nn.MSELoss()(prediction, target).item()
    correct_mse = nn.MSELoss()(prediction, target.reshape_as(prediction)).item()
    assert caught and correct_mse == 0 and wrong_mse > 0
    beta = 0.5
    r = torch.tensor([-2., -0.5, 0., 0.5, 2.], dtype=DTYPE)
    manual = torch.where(r.abs() < beta, r.square() / (2 * beta), r.abs() - beta / 2)
    smooth = nn.SmoothL1Loss(beta=beta, reduction="none")(r, torch.zeros_like(r))
    huber = nn.HuberLoss(delta=beta, reduction="none")(r, torch.zeros_like(r))
    assert torch.equal(manual, smooth) and torch.equal(huber, beta * smooth)
    return {"weight_shape": list(net[0].weight.shape),
            "bias_shape": list(net[0].bias.shape),
            "kept_batch_sizes": [len(x) for x, _ in kept],
            "dropped_batch_sizes": [len(x) for x, _ in dropped],
            "weight_gradient_before_step": gradient.tolist(),
            "step_preserves_gradient": True, "zero_grad_sets_none": True,
            "wrong_broadcast_mse": wrong_mse, "correct_aligned_mse": correct_mse,
            "broadcast_warning_captured": True, "smooth_l1_matches_book": True,
            "huber_equals_beta_times_smooth_l1": True}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(step_rows, comparison, robust_rows):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "font.size": 12,
                         "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    teal, orange, navy = "#147E83", "#D47F37", "#536FA4"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = [r["step"] for r in step_rows]
    axes[0].semilogy(steps, [r["train_mse_after"] for r in step_rows], color=teal)
    axes[0].set(xlabel="参数更新次数", ylabel="MSE（不含 1/2）", title="框架实现：3 轮、300 次更新")
    for key, target, color in (("w1", 2., teal), ("w2", -3.4, orange), ("b", 4.2, navy)):
        axes[1].plot(steps, [r[key] for r in step_rows], label=key, color=color)
        axes[1].axhline(target, color=color, linestyle="--", alpha=0.6)
    axes[1].set(xlabel="参数更新次数", ylabel="参数值", title="实线为估计，虚线为生成参数")
    axes[1].legend()
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIGURES / "3_3_training.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = [r["step"] for r in comparison]
    axes[0].semilogy(steps, [r["framework_mse"] for r in comparison], color=teal, label="MSE + SGD lr=0.015")
    axes[0].semilogy(steps, [r["scratch_mse"] for r in comparison], color=orange,
                     linestyle="--", label="手写半平方损失 lr=0.03")
    axes[0].set(xlabel="更新次数", ylabel="统一按 MSE 评估", title="同初始参数、同一批数据，曲线重合")
    axes[0].legend(fontsize=10)
    axes[1].plot(steps, [r["max_parameter_difference"] for r in comparison], color=navy)
    axes[1].set(xlabel="更新次数", ylabel="最大参数绝对差", title="差异仅在浮点舍入量级")
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIGURES / "3_3_equivalence.png", dpi=170); plt.close(fig)

    r = torch.linspace(-3., 3., 301, dtype=DTYPE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(r.numpy(), (r.square() / 2).numpy(), label="半平方误差", color=teal)
    axes[0].plot(r.numpy(), nn.SmoothL1Loss(beta=.5, reduction="none")(r, torch.zeros_like(r)).numpy(),
                 label="教材公式：Smooth L1，beta=0.5", color=orange)
    axes[0].set(xlabel="残差", ylabel="单样本损失", title="小残差二次，大残差线性")
    axes[0].legend(fontsize=10)
    axes[1].bar(["MSE 训练", "Smooth L1 训练"], [row["clean_validation_mse"] for row in robust_rows],
                 color=[teal, orange])
    axes[1].set(ylabel="干净验证集上的 MSE", title="仅训练标签加三个 +50 离群值")
    for ax in axes: ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(FIGURES / "3_3_smooth_l1.png", dpi=170); plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.set_printoptions(precision=8, sci_mode=False)
    REPORT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    features, labels = splits["train"]
    net, epoch_rows, step_rows = train(splits)
    comparison = compare_with_scratch(features, labels)
    api = audit_api(features, labels)
    contaminated = dict(splits)
    noisy_labels = labels.clone(); noisy_labels[:3] += 50.
    contaminated["train"] = (features, noisy_labels)
    robust_rows = []
    for name, criterion in (("MSELoss", nn.MSELoss()), ("SmoothL1Loss(beta=0.5)", nn.SmoothL1Loss(beta=.5))):
        robust_net, _, _ = train(contaminated, loss_fn=criterion, epochs=6)
        xv, yv = splits["validation"]
        robust_rows.append({"training_loss": name, "epochs": 6,
                            "clean_validation_mse": evaluate(robust_net, xv, yv),
                            "w1": robust_net[0].weight[0, 0].item(),
                            "w2": robust_net[0].weight[0, 1].item(), "b": robust_net[0].bias.item()})
    xt, yt = splits["test"]
    metrics = {"purpose": "teacher demonstration, not learner completion",
               "torch": torch.__version__, "device": "cpu", "dtype": "float64",
               "data_source": "reuse datasets/ch03/scratch_*.csv from section 3.2",
               "init_seed": 3302, "loader_seed": 3303, "batch_size": 10,
               "epochs": 3, "learning_rate": 0.03, "loss": "mean squared error without 1/2",
               "updates": len(step_rows) - 1,
               "learned_weight": net[0].weight.detach().tolist(), "learned_bias": net[0].bias.item(),
               "final_train_mse": epoch_rows[-1]["train_mse"],
               "final_validation_mse": epoch_rows[-1]["validation_mse"],
               "final_test_mse": evaluate(net, xt, yt),
               "equivalence_max_parameter_difference": max(r["max_parameter_difference"] for r in comparison),
               "api_checks": api,
               "outlier_design": {"training_indices": [0, 1, 2], "label_shift": 50., "beta": .5}}
    print("3.3 线性回归的简洁实现 |", torch.__version__, "| CPU / float64")
    print("复用 3.2 数据；Linear.weight 形状:", tuple(net[0].weight.shape))
    for row in epoch_rows: print(row)
    print("学到 weight、bias:", net[0].weight.detach(), net[0].bias.detach())
    print("最终测试 MSE:", metrics["final_test_mse"])
    print("\n手写/框架对齐后的最大参数差:", metrics["equivalence_max_parameter_difference"])
    print("API、尾批、梯度和形状检查:", api)
    print("\n离群值对照，使用统一的干净验证集 MSE:")
    for row in robust_rows: print(row)
    (REPORT / "3_3_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(REPORT / "3_3_epochs.csv", epoch_rows)
    write_csv(REPORT / "3_3_steps.csv", step_rows)
    write_csv(REPORT / "3_3_equivalence.csv", comparison)
    write_csv(REPORT / "3_3_robustness.csv", robust_rows)
    draw_figures(step_rows, comparison, robust_rows)
    print("\n结果与三张图已保存在 reports/ch03；原有数据没有重写。")


if __name__ == "__main__":
    main()
