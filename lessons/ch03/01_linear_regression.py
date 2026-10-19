"""3.1 线性回归整节演示：模型、损失、优化、矢量化与概率解释。

此处用全批量梯度下降观察原理；3.2 再组织小批量数据流水线。
所有数据均为本地人工生成，复用 openmmlab，不安装或下载任何内容。
"""

import csv
import json
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import torch

REPORT = ROOT / "reports" / "ch03"
FIGURES = REPORT / "figures"
DATA = ROOT / "datasets" / "ch03"
SEED = 3101
DTYPE = torch.float64
TRUE_W = torch.tensor([[2.], [-3.4]], dtype=DTYPE)
TRUE_B = 4.2
SIGMA = 0.2


def make_data():
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(256, 2, generator=g, dtype=DTYPE)
    noise = SIGMA * torch.randn(256, 1, generator=g, dtype=DTYPE)
    y = x @ TRUE_W + TRUE_B + noise
    return {"train": (x[:128], y[:128]),
            "validation": (x[128:192], y[128:192]),
            "test": (x[192:], y[192:])}


def half_mse(prediction, target):
    assert prediction.shape == target.shape
    return ((prediction - target) ** 2).mean() / 2


def train_full_batch(splits, lr=0.1, steps=200):
    x, y = splits["train"]
    xv, yv = splits["validation"]
    w = torch.zeros((2, 1), dtype=DTYPE, requires_grad=True)
    b = torch.zeros(1, dtype=DTYPE, requires_grad=True)
    history = []

    def record(step):
        with torch.no_grad():
            history.append({"step": step,
                            "train_half_mse": half_mse(x @ w + b, y).item(),
                            "validation_half_mse": half_mse(xv @ w + b, yv).item(),
                            "w1": w[0, 0].item(), "w2": w[1, 0].item(),
                            "b": b.item()})

    record(0)
    for step in range(1, steps + 1):
        prediction = x @ w + b
        loss = half_mse(prediction, y)
        loss.backward()
        assert w.grad is not None and b.grad is not None
        if step == 1:
            residual = prediction.detach() - y
            assert torch.allclose(w.grad, x.T @ residual / len(x))
            assert torch.allclose(b.grad, residual.mean().reshape(1))
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None
        b.grad = None
        record(step)
    return w.detach().clone(), b.detach().clone(), history


def least_squares(x, y):
    augmented = torch.cat((x, torch.ones((len(x), 1), dtype=DTYPE)), dim=1)
    solution = torch.linalg.lstsq(augmented, y, rcond=1e-12, driver="gelsd")
    return augmented, solution


def vectorization_benchmark(n=10000, repeats=5, vector_inner=1000):
    a, b = torch.ones(n), torch.ones(n)
    c = torch.empty_like(a)
    a + b  # 预热；全部在 CPU 上计时，避免异步 GPU 时间的解释问题。
    rows = []
    for trial in range(1, repeats + 1):
        started = perf_counter()
        for i in range(n):
            c[i] = a[i] + b[i]
        loop_time = perf_counter() - started
        started = perf_counter()
        for _ in range(vector_inner):
            d = a + b
        vector_time = (perf_counter() - started) / vector_inner
        assert torch.equal(c, d)
        rows.append({"trial": trial, "elements": n,
                     "loop_seconds_per_call": loop_time,
                     "vector_seconds_per_call": vector_time,
                     "vector_calls_averaged": vector_inner})
    return rows


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(splits, w, b, history):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"],
                         "axes.unicode_minus": False, "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    teal, orange, navy = "#147E83", "#D47F37", "#173448"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    steps = [row["step"] for row in history]
    axes[0].semilogy(steps, [row["train_half_mse"] for row in history],
                     label="训练集", color=teal)
    axes[0].semilogy(steps, [row["validation_half_mse"] for row in history],
                     label="验证集", color=orange, linestyle="--")
    axes[0].set(xlabel="参数更新次数", ylabel="平均平方误差 / 2",
                title="固定学习率 0.1，全批量更新")
    axes[0].legend()
    xt, yt = splits["test"]
    prediction = xt @ w + b
    axes[1].scatter(yt.numpy().ravel(), prediction.numpy().ravel(),
                    color=teal, alpha=0.65, s=22)
    lower = min(yt.min().item(), prediction.min().item())
    upper = max(yt.max().item(), prediction.max().item())
    axes[1].plot([lower, upper], [lower, upper], color=orange,
                 linestyle="--", label="预测 = 观测")
    axes[1].set(xlabel="测试集观测标签", ylabel="测试集预测值",
                title="训练结束后，测试集评估一次")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "3_1_training.png", dpi=170)
    plt.close(fig)

    grid = torch.linspace(-7., 7., 500, dtype=DTYPE)
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    for mu, sigma, color in ((0., 1., teal), (0., 2., orange), (3., 1., navy)):
        density = torch.exp(-0.5 * ((grid - mu) / sigma) ** 2)
        density /= math.sqrt(2 * math.pi) * sigma
        ax.plot(grid.numpy(), density.numpy(), color=color,
                 label=f"均值 {mu:g}，标准差 {sigma:g}")
    ax.set(xlabel="随机变量的取值", ylabel="概率密度", title="均值改变位置，标准差改变分散程度")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "3_1_normal.png", dpi=170)
    plt.close(fig)

    residual = torch.linspace(-3., 3., 301)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    axes[0].plot(residual.numpy(), (residual ** 2 / 2).numpy(),
                 label="平方损失 r²/2", color=teal)
    axes[0].plot(residual.numpy(), residual.abs().numpy(),
                 label="绝对误差 |r|", color=orange)
    axes[0].set(xlabel="残差 r = 预测 - 标签", ylabel="单样本损失",
                title="大残差在平方损失中增长更快")
    values = [0., 0., 1., 1., 2., 25.]
    axes[1].scatter(values, [0.] * len(values), s=38, color=navy, alpha=0.7)
    axes[1].axvline(sum(values) / len(values), color=teal, label="均值：最小平方误差")
    axes[1].axvline(median(values), color=orange, linestyle="--", label="中位数：最小绝对误差")
    axes[1].set(xlabel="只预测一个常数时的样本值", yticks=[],
                title="单个离群值会明显移动均值", ylim=(-0.4, 0.4))
    for ax in axes:
        ax.legend(fontsize=11)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "3_1_losses.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for center, label in (((0.17, 0.73), "x1"), ((0.17, 0.28), "x2"),
                          ((0.76, 0.5), "ŷ")):
        ax.add_patch(Circle(center, 0.085, facecolor="#F1F7F8", edgecolor=teal, linewidth=2))
        ax.text(*center, label, ha="center", va="center", fontsize=16, color=navy)
    for origin in ((0.25, 0.70), (0.25, 0.31)):
        ax.add_patch(FancyArrowPatch(origin, (0.675, 0.5), arrowstyle="->",
                                     mutation_scale=16, linewidth=1.7, color=teal))
    ax.text(0.44, 0.66, "权重 w1", ha="center", color=teal)
    ax.text(0.44, 0.32, "权重 w2", ha="center", color=teal)
    ax.text(0.76, 0.86, "偏置 b", ha="center", color=orange)
    ax.add_patch(FancyArrowPatch((0.76, 0.81), (0.76, 0.6), arrowstyle="->",
                                 mutation_scale=16, color=orange, linewidth=1.5))
    ax.text(0.17, 0.05, "给定的输入\n不计入计算层数", ha="center", va="bottom")
    ax.text(0.76, 0.05, "ŷ = w1·x1 + w2·x2 + b\n一个计算层，单个输出", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES / "3_1_network.png", dpi=170)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.set_printoptions(precision=6, sci_mode=False)
    for path in (REPORT, FIGURES, DATA):
        path.mkdir(parents=True, exist_ok=True)
    splits = make_data()
    for name, (x, y) in splits.items():
        rows = [{"x1": row[0], "x2": row[1], "y": row[2]}
                for row in torch.cat((x, y), dim=1).tolist()]
        write_csv(DATA / f"linear_{name}.csv", rows, ["x1", "x2", "y"])
    print("3.1 线性回归 | PyTorch", torch.__version__, "| CPU / float64")
    print("1. 人工数据：训练 128、验证 64、测试 64；每条样本两个特征。")
    print("真实参数:", TRUE_W.ravel(), TRUE_B, "| 噪声标准差:", SIGMA)
    w, b, history = train_full_batch(splits)
    x, y = splits["train"]
    augmented, solution = least_squares(x, y)
    theta = solution.solution
    print("\n2. 同一目标：平均平方误差 / 2；lr=0.1，200 次全批量更新。")
    print("初始训练损失:", history[0]["train_half_mse"])
    print("最终训练损失:", history[-1]["train_half_mse"])
    print("最终验证损失:", history[-1]["validation_half_mse"])
    print("梯度下降 w,b:", w.ravel(), b.item())
    print("最小二乘 w,b:", theta.ravel(), "| 增广矩阵秩:", solution.rank.item())
    test_x, test_y = splits["test"]
    test_prediction = test_x @ w + b
    test_loss = half_mse(test_prediction, test_y).item()
    print("最终测试损失（固定训练结束后）:", test_loss)
    new_x = torch.tensor([[1., -1.]], dtype=DTYPE)
    print("新输入 [1,-1] 的预测:", (new_x @ w + b).item())

    print("\n3. 重复特征：预测可相同，参数却不唯一。")
    rank_deficient = torch.tensor([[1., 2., 1.], [2., 4., 1.], [3., 6., 1.]], dtype=DTYPE)
    theta1 = torch.tensor([[1.], [1.], [0.]], dtype=DTYPE)
    theta2 = theta1 + torch.tensor([[2.], [-1.], [0.]], dtype=DTYPE)
    print("秩:", torch.linalg.matrix_rank(rank_deficient).item(),
          "| 不同参数预测相同:", torch.equal(rank_deficient @ theta1, rank_deficient @ theta2))

    print("\n4. 正态噪声负对数似然 = 常数 + SSE/(2σ²)。")
    residual = x @ w + b - y
    formula_nll = len(x) / 2 * math.log(2 * math.pi * SIGMA ** 2)
    formula_nll += residual.square().sum().item() / (2 * SIGMA ** 2)
    api_nll = -torch.distributions.Normal(x @ w + b, SIGMA).log_prob(y).sum().item()
    print("公式 NLL / 分布 API NLL:", formula_nll, api_nll)
    assert math.isclose(formula_nll, api_nll, rel_tol=1e-12, abs_tol=1e-12)
    values = torch.tensor([0., 0., 1., 1., 2., 25.], dtype=DTYPE)
    print("只预测常数：平方误差选均值", values.mean().item(),
          "；绝对误差可选中位数", median(values.tolist()))
    laplace_nll = -torch.distributions.Laplace(torch.zeros_like(values), 1.).log_prob(values).sum()
    assert torch.allclose(laplace_nll, len(values) * math.log(2) + values.abs().sum())
    point, oscillation = 0.2, [0.2]
    for _ in range(6):
        point -= 0.3 * (1 if point > 0 else -1 if point < 0 else 0)
        oscillation.append(round(point, 6))
    print("|b| 用固定步长 0.3 的轨迹:", oscillation)

    print("\n5. 矢量化：结果相同，减少逐元素 Python 调用。")
    benchmark = vectorization_benchmark()
    loop = median(row["loop_seconds_per_call"] for row in benchmark)
    vector = median(row["vector_seconds_per_call"] for row in benchmark)
    print(f"循环 / 矢量化中位耗时: {loop:.8f}s / {vector:.8f}s")
    print(f"本次运行比值: {loop / vector:.1f} 倍；实际时间随机器负载变化。")

    print("\n6. 堆叠两个仿射层，仍可合并成一个仿射层。")
    first = torch.tensor([[1., 2., 0.], [0., 1., 3.]], dtype=DTYPE)
    first_b = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)
    second = torch.tensor([[1.], [-1.], [0.5]], dtype=DTYPE)
    second_b = torch.tensor([0.4], dtype=DTYPE)
    stacked = (x @ first + first_b) @ second + second_b
    collapsed = x @ (first @ second) + (first_b @ second + second_b)
    assert torch.allclose(stacked, collapsed)
    print("两种计算一致:", torch.allclose(stacked, collapsed))

    metrics = {"purpose": "teacher demonstration, not learner completion",
               "torch": torch.__version__, "seed": SEED, "device": "cpu",
               "dtype": "float64", "noise_std": SIGMA, "learning_rate": 0.1,
               "updates": 200, "samples": {name: len(pair[0]) for name, pair in splits.items()},
               "true_w": TRUE_W.ravel().tolist(), "true_b": TRUE_B,
               "learned_w": w.ravel().tolist(), "learned_b": b.item(),
               "least_squares_theta": theta.ravel().tolist(),
               "initial_train_half_mse": history[0]["train_half_mse"],
               "final_train_half_mse": history[-1]["train_half_mse"],
               "final_validation_half_mse": history[-1]["validation_half_mse"],
               "final_test_half_mse": test_loss,
               "new_prediction": (new_x @ w + b).item(),
               "gaussian_nll": formula_nll,
               "loop_median_seconds": loop, "vector_median_seconds": vector}
    (REPORT / "3_1_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(REPORT / "3_1_training_history.csv", history, list(history[0]))
    write_csv(REPORT / "3_1_vectorization.csv", benchmark, list(benchmark[0]))
    draw_figures(splits, w, b, history)
    print("\n数据、指标、计时与四张图已保存到 datasets/ch03 和 reports/ch03。")
    return splits, w, b, history, metrics


if __name__ == "__main__":
    main()
