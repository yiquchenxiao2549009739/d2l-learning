"""3.2 从零实现线性回归：人工数据 -> 生成器 -> 模型/损失 -> 手写 SGD。

核心仅用张量与自动求导；不使用 nn.Linear、DataLoader 或 optim.SGD。
复用已有环境，无安装、无下载。绘图和结果保存放在主训练之外。
"""

import csv
import json
import math
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

REPORT = ROOT / "reports" / "ch03"
FIGURES = REPORT / "figures"
DATA = ROOT / "datasets" / "ch03"
DTYPE = torch.float64
TRUE_W = torch.tensor([[2.], [-3.4]], dtype=DTYPE)
TRUE_B = 4.2


def synthetic_data():
    g = torch.Generator().manual_seed(3201)
    features = torch.randn(1400, 2, generator=g, dtype=DTYPE)
    labels = features @ TRUE_W + TRUE_B
    labels += 0.01 * torch.randn(1400, 1, generator=g, dtype=DTYPE)
    return {"train": (features[:1000], labels[:1000]),
            "validation": (features[1000:1200], labels[1000:1200]),
            "test": (features[1200:], labels[1200:])}


def data_iter(batch_size, features, labels, rng=None):
    """同一索引同时选特征和标签；不丢最后不足一批的样本。"""
    if batch_size <= 0 or len(features) != len(labels):
        raise ValueError("批量大小应为正数，特征与标签的行数应一致")
    indices = list(range(len(features)))
    (rng if rng is not None else random).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.tensor(indices[start:start + batch_size],
                                     dtype=torch.long)
        yield features[batch_indices], labels[batch_indices]


def initialize(mode="random"):
    g = torch.Generator().manual_seed(3202)
    if mode == "random":
        w = (0.01 * torch.randn((2, 1), generator=g, dtype=DTYPE)).requires_grad_()
    elif mode == "zero":
        w = torch.zeros((2, 1), dtype=DTYPE, requires_grad=True)
    else:
        raise ValueError("mode 只能为 random 或 zero")
    b = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    return w, b


def linreg(x, w, b):
    return x @ w + b


def squared_loss(y_hat, y):
    """返回每条样本的半平方误差，不在函数内求和或平均。"""
    if y_hat.numel() != y.numel():
        raise ValueError("预测和标签的元素数不同，不能仅靠 reshape 修复")
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, num_samples):
    """与 loss.sum().backward() 配套：除以本批实际样本数。"""
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                raise RuntimeError("缺少梯度，请先执行 backward")
            param -= lr * param.grad / num_samples
            param.grad.zero_()


def evaluate(features, labels, w, b):
    with torch.no_grad():
        return squared_loss(linreg(features, w, b), labels).mean().item()


def train(splits, lr=0.03, batch_size=10, epochs=3, init="random"):
    features, labels = splits["train"]
    xv, yv = splits["validation"]
    w, b = initialize(init)
    rng = random.Random(3203)
    steps, stopped = 0, False
    epoch_rows, step_rows = [], []

    def record_epoch(epoch):
        epoch_rows.append({"epoch": epoch, "steps": steps,
                           "train_half_mse": evaluate(features, labels, w, b),
                           "validation_half_mse": evaluate(xv, yv, w, b)})

    record_epoch(0)
    step_rows.append({"step": 0, "epoch": 0, "batch_size": 0,
                      "batch_half_mse_before": None,
                      "train_half_mse_after": epoch_rows[0]["train_half_mse"],
                      "w1": w[0, 0].item(), "w2": w[1, 0].item(), "b": b.item()})
    for epoch in range(epochs):
        for x, y in data_iter(batch_size, features, labels, rng):
            per_sample = squared_loss(linreg(x, w, b), y)
            batch_loss = per_sample.mean().item()
            if not math.isfinite(batch_loss) or batch_loss > 1e12:
                stopped = True  # 只为过大学习率对照设置的演示保护阈值。
                break
            per_sample.sum().backward()
            sgd([w, b], lr, len(x))
            steps += 1
            step_rows.append({"step": steps, "epoch": epoch + 1,
                              "batch_size": len(x), "batch_half_mse_before": batch_loss,
                              "train_half_mse_after": evaluate(features, labels, w, b),
                              "w1": w[0, 0].item(), "w2": w[1, 0].item(), "b": b.item()})
        record_epoch(epoch + 1)
        if stopped:
            break
    return {"w": w.detach().clone(), "b": b.detach().clone(),
            "epochs": epoch_rows, "steps": step_rows,
            "status": "stopped_divergence_guard" if stopped else "complete",
            "lr": lr, "init": init}


def audit_tail_and_scaling(features, labels):
    ids = torch.arange(23, dtype=DTYPE).reshape(-1, 1)
    batches = list(data_iter(10, ids, ids + 100, random.Random(3203)))
    order = [int(v) for x, _ in batches for v in x.ravel().tolist()]
    assert sorted(order) == list(range(23))
    assert all(torch.equal(y, x + 100) for x, y in batches)
    tail_indices = batches[-1][0].ravel().long()
    x, y = features[tail_indices], labels[tail_indices]
    w_sum, b_sum = initialize()
    w_mean = w_sum.detach().clone().requires_grad_()
    b_mean = b_sum.detach().clone().requires_grad_()
    initial = torch.cat((w_sum.detach().ravel(), b_sum.detach().ravel()))
    squared_loss(linreg(x, w_sum, b_sum), y).sum().backward()
    sgd([w_sum, b_sum], 0.03, len(x))
    squared_loss(linreg(x, w_mean, b_mean), y).mean().backward()
    with torch.no_grad():
        assert w_mean.grad is not None and b_mean.grad is not None
        w_mean -= 0.03 * w_mean.grad
        b_mean -= 0.03 * b_mean.grad
    sum_result = torch.cat((w_sum.detach().ravel(), b_sum.detach().ravel()))
    mean_result = torch.cat((w_mean.detach().ravel(), b_mean.detach().ravel()))
    assert torch.allclose(sum_result, mean_result, rtol=0, atol=1e-15)
    correct_delta = sum_result - initial
    wrong_delta = correct_delta * len(x) / 10
    return {"sizes": [len(x) for x, _ in batches], "order": order,
            "each_sample_once": True, "feature_label_alignment": True,
            "sum_mean_max_difference": (sum_result - mean_result).abs().max().item(),
            "wrong_divisor_update_ratio": (wrong_delta.norm() / correct_delta.norm()).item()}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(baseline, experiments, tail):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                         "axes.unicode_minus": False, "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#147E83", "#D47F37", "#536FA4"]
    rows = baseline["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy([r["step"] for r in rows],
                     [r["train_half_mse_after"] for r in rows], color=colors[0])
    axes[0].set(xlabel="小批量更新次数", ylabel="全训练集半均方误差",
                title="3 个 epoch，合计 300 次更新")
    for key, target, color in zip(("w1", "w2", "b"), (2., -3.4, 4.2), colors):
        axes[1].plot([r["step"] for r in rows], [r[key] for r in rows],
                     label=key, color=color)
        axes[1].axhline(target, color=color, linestyle="--", alpha=0.6)
    axes[1].set(xlabel="小批量更新次数", ylabel="参数值", title="实线为估计值，虚线为生成参数")
    axes[1].legend()
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIGURES / "3_2_training.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for run, color in zip(experiments[:3], colors):
        axes[0].semilogy([r["step"] for r in run["steps"]],
                         [r["train_half_mse_after"] for r in run["steps"]],
                         label=f"lr={run['lr']:g}", color=color)
    high = experiments[3]
    axes[1].semilogy([r["step"] for r in high["steps"]],
                     [r["train_half_mse_after"] for r in high["steps"]], color="#B64C45")
    axes[0].set(title="同数据、初始化与打乱顺序", xlabel="更新次数", ylabel="训练半均方误差")
    axes[0].legend()
    axes[1].set(title="lr=3：触发演示保护阈值后停止", xlabel="更新次数", ylabel="训练半均方误差")
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIGURES / "3_2_learning_rates.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    ax.set(xlim=(-2, 10), ylim=(-0.5, 3.4)); ax.axis("off")
    start = 0
    for row, size in enumerate(tail["sizes"]):
        y = 2.2 - row
        ax.text(-0.3, y + 0.3, f"批次 {row + 1}\n{size} 条", ha="right", va="center")
        for col, idx in enumerate(tail["order"][start:start + size]):
            ax.add_patch(FancyBboxPatch((col * 0.95, y), 0.82, 0.6,
                         boxstyle="round,pad=0.03", facecolor=colors[row], linewidth=0))
            ax.text(col * 0.95 + 0.41, y + 0.3, str(idx), ha="center", va="center", color="white")
        start += size
    ax.text(4, 3.1, "23 条样本：打乱索引后分批，特征和标签共享同一组索引", ha="center")
    fig.tight_layout(); fig.savefig(FIGURES / "3_2_batches.png", dpi=170); plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.set_printoptions(precision=7, sci_mode=False)
    for folder in (REPORT, FIGURES, DATA): folder.mkdir(parents=True, exist_ok=True)
    splits = synthetic_data()
    for name, (x, y) in splits.items():
        write_csv(DATA / f"scratch_{name}.csv",
                  [{"x1": row[0], "x2": row[1], "y": row[2]}
                   for row in torch.cat((x, y), dim=1).tolist()])
    x, y = splits["train"]
    preview_x, preview_y = next(data_iter(10, x, y, random.Random(3203)))
    print("3.2 从零实现线性回归 |", torch.__version__, "| CPU / float64")
    print("训练数据形状:", tuple(x.shape), tuple(y.shape), "| 噪声标准差 0.01")
    print("第一批形状:", tuple(preview_x.shape), tuple(preview_y.shape))
    print("第一批前两条:\n", preview_x[:2], "\n", preview_y[:2])
    baseline = train(splits)
    print("\n默认设置：batch_size=10，lr=0.03，epochs=3")
    for row in baseline["epochs"]: print(row)
    w, b = baseline["w"], baseline["b"]
    print("学到 w,b:", w.ravel(), b.item())
    print("与生成参数的差（真实-估计）:", (TRUE_W - w).ravel(), TRUE_B - b.item())
    test_x, test_y = splits["test"]
    test_loss = evaluate(test_x, test_y, w, b)
    tail = audit_tail_and_scaling(x, y)
    print("\n尾批检查:", tail)
    experiments = [train(splits, lr=0.003), baseline,
                   train(splits, lr=0.3), train(splits, lr=3.)]
    zero = train(splits, init="zero")
    comparisons = []
    for run in experiments + [zero]:
        last = run["epochs"][-1]
        comparisons.append({"learning_rate": run["lr"], "init": run["init"],
                            "status": run["status"], "updates": len(run["steps"]) - 1,
                            "train_half_mse": last["train_half_mse"],
                            "validation_half_mse": last["validation_half_mse"],
                            "w1": run["w"][0, 0].item(), "w2": run["w"][1, 0].item(),
                            "b": run["b"].item()})
    print("\n学习率与初始化对照（每次重新开始）:")
    for row in comparisons: print(row)
    print("默认模型最终测试半均方误差:", test_loss)
    metrics = {"purpose": "teacher demonstration, not learner completion",
               "torch": torch.__version__, "seeds": {"data": 3201, "init": 3202, "shuffle": 3203},
               "device": "cpu", "dtype": "float64", "noise_std": 0.01,
               "samples": {name: len(pair[0]) for name, pair in splits.items()},
               "batch_size": 10, "learning_rate": 0.03, "epochs": 3,
               "updates": len(baseline["steps"]) - 1,
               "learned_w": w.ravel().tolist(), "learned_b": b.item(),
               "final_train_half_mse": baseline["epochs"][-1]["train_half_mse"],
               "final_validation_half_mse": baseline["epochs"][-1]["validation_half_mse"],
               "final_test_half_mse": test_loss, "tail_check": tail}
    (REPORT / "3_2_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(REPORT / "3_2_steps.csv", baseline["steps"])
    write_csv(REPORT / "3_2_epochs.csv", baseline["epochs"])
    write_csv(REPORT / "3_2_comparisons.csv", comparisons)
    draw_figures(baseline, experiments, tail)
    print("\n数据、实际结果、对照与三张图已保存到项目中。")


if __name__ == "__main__":
    main()
