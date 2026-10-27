# -*- coding: utf-8 -*-
"""4.4 整节：泛化、三份数据、K折，以及多项式的容量/优化/数值对照。

人工数据、CPU/float64，复用已有环境。教材高斯数据的SGD实验与额外区间
控制实验分开；所有候选只用验证数据比较，独立测试仅评估最终CV方案一次。
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
from matplotlib.patches import Rectangle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPORT = ROOT / "reports/ch04"
FIGURES = REPORT / "figures"
DTYPE = torch.float64
TRUE_W = [5., 1.2, -3.4, 5.6]
SIGMA = .1


def factorial_features(x, degree):
    """degree=3有4列；常数列已包含截距，不再额外加bias。"""
    columns = [torch.ones_like(x)]
    for power in range(1, degree + 1):
        columns.append(columns[-1] * x / power)
    return torch.cat(columns, dim=1)


def truth(x):
    return factorial_features(x, 3) @ torch.tensor(TRUE_W, dtype=DTYPE).reshape(-1, 1)


def legendre_features(x, degree):
    """区间控制实验用的等价多项式基；改善病态，不改变最高次数。"""
    columns = [torch.ones_like(x)]
    if degree:
        columns.append(x)
    for power in range(2, degree + 1):
        columns.append(((2*power-1)*x*columns[-1] - (power-1)*columns[-2]) / power)
    return torch.cat(columns, dim=1)


def make_gaussian_data():
    generator = torch.Generator().manual_seed(4401)
    x = torch.randn(300, 1, generator=generator, dtype=DTYPE)
    y = truth(x) + SIGMA*torch.randn(300, 1, generator=generator, dtype=DTYPE)
    return {name: (x[start:start+100], y[start:start+100])
            for name, start in (("train", 0), ("validation", 100), ("test", 200))}


def mse(prediction, target):
    assert prediction.shape == target.shape
    return ((prediction-target)**2).mean().item()


def fit_sgd(train, validation, degree, epochs):
    X, y = factorial_features(train[0], degree), train[1]
    V, v = factorial_features(validation[0], degree), validation[1]
    net = nn.Linear(degree+1, 1, bias=False).to(dtype=DTYPE)
    nn.init.zeros_(net.weight)  # 单层线性回归可以从零开始；区别于全零多层ReLU。
    loader = DataLoader(TensorDataset(X, y), batch_size=10, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(4402))
    optimizer = torch.optim.SGD(net.parameters(), lr=.01)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(epochs+1):
        if epoch:
            net.train()
            for features, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss_fn(net(features), labels).backward()
                optimizer.step()
        if epoch in (0, 1) or epoch % 20 == 0 or epoch == epochs:
            net.eval()
            with torch.no_grad():
                history.append({"degree": degree, "epoch": epoch,
                                "updates": epoch*len(loader),
                                "train_mse": mse(net(X), y), "validation_mse": mse(net(V), v)})
    return {"degree": degree, "epochs": epochs, "updates": epochs*len(loader),
            "weights": net.weight.detach().flatten().tolist(), **history[-1]}, history


def least_squares(X, y):
    # CPU SVD：避免显式求逆及正规方程把条件数平方。
    result = torch.linalg.lstsq(X, y, rcond=1e-12, driver="gelsd")
    singular = result.singular_values
    return result.solution, int(result.rank), float(singular[0]/singular[-1])


def cross_validate(splits):
    x = torch.cat((splits['train'][0], splits['validation'][0]))
    y = torch.cat((splits['train'][1], splits['validation'][1]))
    permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(4403))
    folds = list(permutation.tensor_split(5))
    rows, summaries = [], []
    for degree in (1, 2, 3, 5, 10):
        X = factorial_features(x, degree)
        for fold in range(5):
            validation_ids = folds[fold]
            train_ids = torch.cat([ids for index, ids in enumerate(folds) if index != fold])
            w, rank, condition = least_squares(X[train_ids], y[train_ids])
            rows.append({"degree": degree, "fold": fold+1, "train_samples": len(train_ids),
                         "validation_samples": len(validation_ids), "rank": rank,
                         "train_mse": mse(X[train_ids]@w, y[train_ids]),
                         "validation_mse": mse(X[validation_ids]@w, y[validation_ids])})
        group = [row for row in rows if row['degree'] == degree]
        values = torch.tensor([row['validation_mse'] for row in group], dtype=DTYPE)
        summaries.append({"degree": degree, "mean_validation_mse": values.mean().item(),
                          "fold_std": values.std(unbiased=False).item()})
    best = min(summaries, key=lambda row: (row['mean_validation_mse'], row['degree']))
    degree = best['degree']
    w, rank, condition = least_squares(factorial_features(x, degree), y)
    # 选好次数，重新拟合全部200条开发数据后，才读取这100条独立测试数据。
    test_x, test_y = splits['test']
    test_mse = mse(factorial_features(test_x, degree)@w, test_y)
    return {'k': 5, 'seed': 4403, 'development_samples': 200, 'fold_sizes': [len(f) for f in folds],
            'validation_ids': [f.tolist() for f in folds], 'summaries': summaries, 'best': best,
            'refit_weights': w.flatten().tolist(), 'refit_rank': rank,
            'test_samples': 100, 'test_evaluations': 1, 'test_mse': test_mse}, rows


def interval_data(count):
    # 额外固定设计实验：等距输入，不声称这些输入是独立高斯抽样。
    x = torch.linspace(-1, 1, count, dtype=DTYPE).reshape(-1, 1)
    generator = torch.Generator().manual_seed(4405)
    y = truth(x) + SIGMA*torch.randn(count, 1, generator=generator, dtype=DTYPE)
    return x, y


def interval_study():
    generator = torch.Generator().manual_seed(4406)
    xv = 2*torch.rand(1000, 1, generator=generator, dtype=DTYPE)-1
    yv = truth(xv) + SIGMA*torch.randn(1000, 1, generator=generator, dtype=DTYPE)
    x, y = interval_data(20)
    complexity, amounts, fitted = [], [], {}
    for degree in range(20):
        X, V = legendre_features(x, degree), legendre_features(xv, degree)
        w, rank, condition = least_squares(X, y)
        complexity.append({'degree': degree, 'parameters': degree+1, 'train_samples': 20,
                           'rank': rank, 'condition': condition, 'train_mse': mse(X@w, y),
                           'validation_mse': mse(V@w, yv)})
        if degree in (1, 3, 19):
            fitted[degree] = w
    for count in (20, 40, 80, 160):
        xn, yn = interval_data(count)
        for degree in (3, 19):
            X, V = legendre_features(xn, degree), legendre_features(xv, degree)
            w, rank, condition = least_squares(X, yn)
            amounts.append({'train_samples': count, 'degree': degree, 'rank': rank,
                            'train_mse': mse(X@w, yn), 'validation_mse': mse(V@w, yv)})
    return complexity, amounts, (x, y, fitted)


def scaling_study(x, y):
    raw = torch.cat([x**i for i in range(20)], dim=1)
    factorial = factorial_features(x, 19)
    mean, std = raw[:, 1:].mean(0), raw[:, 1:].std(0, unbiased=False)
    standardized = torch.cat((raw[:, :1], (raw[:, 1:]-mean)/std), dim=1)
    rows = []
    for name, X in (('raw', raw), ('factorial', factorial), ('train_standardized', standardized)):
        singular = torch.linalg.svdvals(X)
        largest_eigenvalue = (singular[0]**2/len(X)).item()
        w = torch.zeros(20, 1, dtype=DTYPE)
        gradient = 2*X.T@(X@w-y)/len(X)
        updated = w-.01*gradient
        rows.append({'basis': name, 'condition': float(singular[0]/singular[-1]),
                     'max_abs_feature': X.abs().max().item(),
                     'full_batch_mse_lr_upper_bound': 1/largest_eigenvalue,
                     'initial_mse': mse(X@w, y), 'one_step_mse_lr_0_01': mse(X@updated, y)})
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def draw_figures(histories, complexity, amounts, curves, scales):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'font.size': 11,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    teal, orange, navy = '#147E83', '#D47F37', '#173448'
    fig, ax = plt.subplots(figsize=(10, 4)); ax.axis('off'); ax.set(xlim=(0, 10), ylim=(0, 5))
    for x, text, color in ((0, '训练：更新参数', teal), (3.45, '验证 / CV：选择方案', orange),
                           (6.9, '独立测试：最终评估', navy)):
        ax.add_patch(Rectangle((x, 3.9), 3.05, .85, facecolor=color))
        ax.text(x+1.525, 4.325, text, ha='center', va='center', color='white', fontsize=11)
    for fold in range(5):
        for part in range(5):
            ax.add_patch(Rectangle((2+part*1.4, 3.05-fold*.53), 1.29, .39,
                                   facecolor=orange if part == fold else teal, alpha=.85))
        ax.text(.65, 3.24-fold*.53, f'第{fold+1}折', va='center')
    ax.text(5, .27, '每折重新拟合；橙色验证一次，绿色训练四次；独立测试不进入折分。', ha='center', fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES/'4_4_data_roles.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, degree in zip(axes, (1, 3, 19)):
        rows = [row for row in histories if row['degree'] == degree]
        for key, label, color in (('train_mse', '训练', teal), ('validation_mse', '验证', orange)):
            ax.semilogy([r['epoch'] for r in rows], [r[key] for r in rows], label=label, color=color)
        ax.set(title=f'{degree}次 / {degree+1}参数', xlabel='epoch', ylabel='MSE'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('教材高斯数据：阶乘特征、SGD .01、batch10、每轮末重新评估', fontsize=12)
    fig.tight_layout(); fig.savefig(FIGURES/'4_4_sgd_training.png', dpi=170); plt.close(fig)
    x, y, fitted = curves
    grid = torch.linspace(-1, 1, 801, dtype=DTYPE).reshape(-1, 1)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, degree in zip(axes, (1, 3, 19)):
        prediction = legendre_features(grid, degree)@fitted[degree]
        ax.scatter(x.flatten(), y.flatten(), s=15, color=navy, label='20条训练样本')
        ax.plot(grid.flatten(), truth(grid).flatten(), '--', color=teal, label='真实三次函数')
        ax.plot(grid.flatten(), prediction.flatten(), color=orange, label='最小二乘拟合')
        ax.set(title=f'{degree}次多项式', xlabel='x', ylabel='y'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('额外区间实验：等距输入、噪声标准差 .1、Legendre等价基', fontsize=12)
    fig.tight_layout(); fig.savefig(FIGURES/'4_4_polynomial_fits.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, label, color in (('train_mse', '训练', teal), ('validation_mse', '验证', orange)):
        axes[0].semilogy([r['degree'] for r in complexity], [max(r[key], 1e-24) for r in complexity], 'o-', label=label, color=color)
    axes[0].set(xlabel='最高次数 d', ylabel='MSE', title='20条训练样本：求解到数值最小二乘'); axes[0].legend()
    for degree, color in ((3, teal), (19, orange)):
        rows = [r for r in amounts if r['degree'] == degree]
        for key, style, label in (('train_mse', '--', '训练'), ('validation_mse', '-', '验证')):
            axes[1].semilogy([r['train_samples'] for r in rows], [max(r[key], 1e-24) for r in rows],
                             style+'o', color=color, label=f'{degree}次 {label}')
    axes[1].set(xlabel='训练样本数（各规模重新等距设计）', ylabel='MSE', title='同一验证集：数据量与复杂度'); axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/'4_4_complexity_data.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6)); labels = ['原始幂', '除以阶乘', '训练集标准化']
    axes[0].bar(labels, [r['condition'] for r in scales], color=teal); axes[0].set_yscale('log')
    axes[0].set(title='同一高斯训练数据 / 20列', ylabel='设计矩阵条件数')
    axes[1].bar(labels, [r['one_step_mse_lr_0_01'] for r in scales], color=orange); axes[1].set_yscale('log')
    axes[1].set(title='零起点 / 全批量更新一次 / lr=.01', ylabel='更新后训练MSE')
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/'4_4_scaling.png', dpi=170); plt.close(fig)


def main():
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    splits = make_gaussian_data()
    sgd, histories = [], []
    for degree, epochs in ((1, 400), (3, 400), (19, 1500)):
        summary, rows = fit_sgd(splits['train'], splits['validation'], degree, epochs)
        X = factorial_features(splits['train'][0], degree)
        w, rank, condition = least_squares(X, splits['train'][1])
        summary['least_squares_train_mse'] = mse(X@w, splits['train'][1])
        summary['least_squares_rank'] = rank
        sgd.append(summary); histories.extend(rows); print('教材SGD:', summary, flush=True)
    cv, cv_rows = cross_validate(splits)
    complexity, amounts, curves = interval_study()
    scales = scaling_study(*splits['train'])
    draw_figures(histories, complexity, amounts, curves, scales)
    metrics = {'date': '2026-10-27', 'purpose': 'teacher demonstration, not learner completion',
               'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float64', 'download': False,
               'gaussian_seed': 4401, 'noise_std': SIGMA, 'true_factorial_weights': TRUE_W,
               'split_sizes': {name: len(data[0]) for name, data in splits.items()},
               'sgd': sgd, 'sgd_bias': False, 'sgd_initialization': 'zero single linear layer',
               'sgd_lr': .01, 'sgd_batch_size': 10, 'sgd_loader_seed': 4402,
               'cross_validation': cv, 'interval': {'scope': 'extra fixed equispaced design on [-1,1], not Gaussian iid sample',
               'basis': 'Legendre polynomial, same degree-limited function space', 'noise_seed': 4405,
               'validation_seed': 4406, 'validation_samples': 1000, 'complexity': complexity, 'sample_sizes': amounts},
               'scaling': scales, 'coin_example': {'labels': [0, 1, 1, 1, 0, 1], 'predict': 1,
               'training_error': 1/3, 'fair_coin_population_error': .5},
               'bayes_mse_for_noisy_labels': SIGMA**2}
    (REPORT/'4_4_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_csv(REPORT/'4_4_sgd_epochs.csv', histories)
    write_csv(REPORT/'4_4_complexity.csv', complexity)
    write_csv(REPORT/'4_4_sample_sizes.csv', amounts)
    write_csv(REPORT/'4_4_cross_validation.csv', cv_rows)
    write_csv(REPORT/'4_4_data.csv', [{'split': name, 'id': i, 'x': float(x[i]), 'y': float(y[i])}
              for name, (x, y) in splits.items() for i in range(len(x))])
    print('K折均值:', cv['summaries']); print('最终选择及独立测试:', cv['best'], cv['test_mse'])
    print('区间1/3/19次:', [r for r in complexity if r['degree'] in (1, 3, 19)])
    print('数据量:', amounts); print('尺度:', scales)
    print('教师演示不代表用户完成；下一节4.5权重衰减。')


if __name__ == '__main__':
    main()
