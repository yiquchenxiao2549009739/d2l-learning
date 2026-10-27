# -*- coding: utf-8 -*-
"""4.5 整节：L2惩罚、手写/框架SGD、验证选lambda与六道教材练习。

人工高维回归，CPU/float64，无下载、无安装。统一使用半MSE，权重衰减
只作用于weight；比较使用相同起点与批次，独立测试只评估选定方案一次。
"""
import csv
import hashlib
import json
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
from torch.utils.data import DataLoader, TensorDataset

REPORT = ROOT / "reports/ch04"
FIGURES = REPORT / "figures"
DTYPE = torch.float64
LR = .003
CANDIDATES = (0., .01, .1, 1., 3., 10., 30., 100.)


def make_data():
    generator = torch.Generator().manual_seed(4501)
    true_w = torch.full((200, 1), .01, dtype=DTYPE)
    splits = {}
    for name, count in (("train", 20), ("validation", 100), ("test", 100)):
        X = torch.randn(count, 200, generator=generator, dtype=DTYPE)
        y = X@true_w + .05 + .01*torch.randn(count, 1, generator=generator, dtype=DTYPE)
        splits[name] = (X, y)
    return splits


def initial_parameters():
    generator = torch.Generator().manual_seed(4502)
    return torch.randn(200, 1, generator=generator, dtype=DTYPE), torch.zeros(1, dtype=DTYPE)


def make_batches(count, epochs=100, batch_size=5, seed=4503):
    loader = DataLoader(TensorDataset(torch.arange(count)), batch_size=batch_size,
                        shuffle=True, drop_last=False, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    return [[ids.clone() for (ids,) in loader] for _ in range(epochs)]


def half_mse(prediction, target):
    assert prediction.shape == target.shape
    return ((prediction-target)**2).mean()/2


def l2_penalty(w):
    return w.square().sum()/2


def evaluate(w, b, dataset):
    with torch.no_grad():
        return half_mse(dataset[0]@w+b, dataset[1]).item()


def null_component_norm(X, w):
    # 只用于解释/核对：对训练X不可见的权重分量。
    _, _, vh = torch.linalg.svd(X, full_matrices=False)
    residual = w-vh.T@(vh@w)
    return residual.norm().item()


def fit(splits, lambd, method, epoch_batches, initial):
    w0, b0 = initial
    if method == "scratch":
        w = w0.clone().requires_grad_(); b = b0.clone().requires_grad_()
        net, optimizer = None, None
    elif method == "concise":
        net = nn.Linear(w0.numel(), 1).to(dtype=DTYPE)
        with torch.no_grad():
            net.weight.copy_(w0.T); net.bias.copy_(b0)
        optimizer = torch.optim.SGD([
            {"params": [net.weight], "weight_decay": lambd},
            {"params": [net.bias], "weight_decay": 0.}
        ], lr=LR, momentum=0, foreach=False)
        w, b = net.weight.T, net.bias
    else:
        raise ValueError("method must be scratch or concise")
    X, y = splits['train']
    history = []
    def record(epoch, updates):
        with torch.no_grad():
            train = evaluate(w, b, splits['train'])
            history.append({'method': method, 'lambda': lambd, 'epoch': epoch, 'updates': updates,
                            'train_half_mse': train, 'validation_half_mse': evaluate(w, b, splits['validation']),
                            'penalty': lambd*l2_penalty(w).item(),
                            'train_objective': train+lambd*l2_penalty(w).item(),
                            'weight_norm': w.norm().item(), 'bias': b.item()})
    record(0, 0); updates = 0
    for epoch, batches in enumerate(epoch_batches, 1):
        for ids in batches:
            if method == "scratch":
                w.grad = None; b.grad = None
                objective = half_mse(X[ids]@w+b, y[ids])+lambd*l2_penalty(w)
                objective.backward()
                with torch.no_grad():
                    w -= LR*w.grad; b -= LR*b.grad
            else:
                net.train(); optimizer.zero_grad(set_to_none=True)
                data_loss = half_mse(net(X[ids]), y[ids])
                data_loss.backward()  # 优化器已负责惩罚；这里不再手动叠加。
                optimizer.step()
            updates += 1
        record(epoch, updates)
    result = {**history[-1], 'weights': w.detach().flatten().tolist(),
              'training_null_component_norm': null_component_norm(X, w.detach())}
    return result, history


def ridge_solution(dataset, lambd):
    """只惩罚w、截距不惩罚；目标为半MSE + lambda*||w||²/2。"""
    X, y = dataset; Xc = X-X.mean(0); yc = y-y.mean(0)
    if lambd == 0:
        w = torch.linalg.lstsq(Xc, yc, rcond=1e-12, driver='gelsd').solution
    else:
        gram = Xc@Xc.T + len(X)*lambd*torch.eye(len(X), dtype=DTYPE)
        w = Xc.T@torch.linalg.solve(gram, yc)
    b = y.mean(0)-X.mean(0)@w
    return w, b


def bias_study():
    histories, results = [], []
    for bias_decay in (0., 3.):
        net = nn.Linear(2, 1).to(dtype=DTYPE)
        nn.init.zeros_(net.weight); nn.init.zeros_(net.bias)
        optimizer = torch.optim.SGD([
            {'params': [net.weight], 'weight_decay': 3.},
            {'params': [net.bias], 'weight_decay': bias_decay}
        ], lr=.1, foreach=False)
        X, y = torch.zeros(8, 2, dtype=DTYPE), torch.full((8, 1), 5., dtype=DTYPE)
        for step in range(201):
            if step:
                optimizer.zero_grad(set_to_none=True)
                half_mse(net(X), y).backward(); optimizer.step()
            histories.append({'bias_decay': bias_decay, 'step': step, 'bias': net.bias.item()})
        results.append({'bias_decay': bias_decay, 'bias': net.bias.item(),
                        'analytic_optimal_bias': 5/(1+bias_decay),
                        'half_mse': half_mse(net(X), y).item()})
    return results, histories


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def draw_figures(histories, sweep, biases):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'font.size': 11,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    teal, orange, navy = '#147E83', '#D47F37', '#173448'
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    xx, yy = np.meshgrid(np.linspace(-.4, 2.7, 200), np.linspace(-.7, 1.8, 200))
    data = ((xx-2)**2+2*(yy-.8)**2)/2
    axes[0].contour(xx, yy, data, levels=[.1, .4, 1., 2., 3.], colors=navy, alpha=.55)
    angle = np.linspace(0, 2*np.pi, 300)
    axes[0].plot(np.cos(angle), np.sin(angle), color=teal, label='L2范数约束边界')
    axes[0].plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], '--', color=orange, label='L1范数约束边界')
    axes[0].scatter([2], [.8], color=navy); axes[0].set(xlim=(-1.2, 2.7), ylim=(-1.2, 1.8),
                  xlabel='w1', ylabel='w2', title='惩罚控制系数，不改变参数个数')
    axes[0].set_aspect('equal'); axes[0].legend(fontsize=9)
    for lambd, color in ((0, navy), (1, teal), (3, orange)):
        optimum = (2/(1+lambd), 1.6/(2+lambd))
        axes[1].scatter(*optimum, color=color, s=55, label=f'λ={lambd}')
        axes[1].annotate(f'({optimum[0]:.2f}, {optimum[1]:.2f})', optimum, xytext=(5, 8), textcoords='offset points', fontsize=9)
    axes[1].set(xlim=(0, 2.6), ylim=(0, 1.2), xlabel='w1', ylabel='w2', title='示意目标：数据损失 + λ||w||²/2')
    axes[1].grid(alpha=.2); axes[1].legend(loc='upper left')
    fig.tight_layout(); fig.savefig(FIGURES/'4_5_penalty_geometry.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for lambd, color in ((0., teal), (3., orange)):
        rows = [r for r in histories if r['method']=='concise' and r['lambda']==lambd]
        for key, style, name in (('train_half_mse', '--', '训练'), ('validation_half_mse', '-', '验证')):
            axes[0].semilogy([r['epoch'] for r in rows], [max(r[key], 1e-24) for r in rows], style,
                             color=color, label=f'λ={lambd:g} {name}')
        axes[1].semilogy([r['epoch'] for r in rows], [r['weight_norm'] for r in rows], color=color, label=f'λ={lambd:g}')
    axes[0].set(xlabel='epoch', ylabel='半MSE（不含惩罚）', title='20训练 / 100验证 / 200特征')
    axes[1].set(xlabel='epoch', ylabel='权重L2范数', title='同一初始化与批次顺序')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES/'4_5_training.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    lambdas = [r['lambda'] for r in sweep]
    for key, name, color in (('train_half_mse', '训练数据损失', teal), ('validation_half_mse', '验证数据损失', orange)):
        axes[0].plot(lambdas, [max(r[key], 1e-24) for r in sweep], 'o-', label=name, color=color)
    axes[0].set_yscale('log'); axes[0].set(ylabel='半MSE', title='固定100轮，按验证数据损失比较'); axes[0].legend(fontsize=9)
    axes[1].plot(lambdas, [r['weight_norm'] for r in sweep], 'o-', color=navy)
    axes[1].set_yscale('log'); axes[1].set(ylabel='权重L2范数', title='正则强度与权重范数')
    for ax in axes:
        ax.set_xscale('symlog', linthresh=.01); ax.set_xlim(-.002, 120)
        ax.set_xticks(lambdas, [f'{value:g}' for value in lambdas])
        ax.set_xlabel('λ（包含0）'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/'4_5_lambda_sweep.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for wd, color in ((0., teal), (3., orange)):
        rows = [r for r in biases if r['bias_decay']==wd]
        ax.plot([r['step'] for r in rows], [r['bias'] for r in rows], color=color, label=f'bias weight_decay={wd:g}')
    ax.axhline(5, color=navy, ls='--', label='真实常数偏移5')
    ax.set(xlabel='更新次数', ylabel='偏置b', title='额外控制实验：输入全零、标签恒为5，weight均衰减')
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/'4_5_bias_groups.png', dpi=170); plt.close(fig)


def main():
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    splits = make_data(); initial = initial_parameters(); batches = make_batches(20)
    histories, results, states = [], [], {}
    for lambd in CANDIDATES:
        result, rows = fit(splits, lambd, 'concise', batches, initial)
        states[str(lambd)] = {'weights': result.pop('weights'), 'bias': result['bias']}
        results.append(result); histories.extend(rows)
        print('验证候选:', result, flush=True)
    scratch = []
    for lambd in (0., 3.):
        result, rows = fit(splits, lambd, 'scratch', batches, initial)
        scratch.append(result); histories.extend(rows)
    best = min(results, key=lambda row: (row['validation_half_mse'], row['lambda']))
    state = states[str(best['lambda'])]
    w = torch.tensor(state['weights'], dtype=DTYPE).reshape(-1, 1)
    b = torch.tensor([state['bias']], dtype=DTYPE)
    test = evaluate(w, b, splits['test'])
    ridges = []
    for lambd in CANDIDATES:
        wr, br = ridge_solution(splits['train'], lambd)
        ridges.append({'lambda': lambd, 'train_half_mse': evaluate(wr, br, splits['train']),
                       'validation_half_mse': evaluate(wr, br, splits['validation']),
                       'train_objective': evaluate(wr, br, splits['train'])+lambd*l2_penalty(wr).item(),
                       'weight_norm': wr.norm().item(), 'bias': br.item()})
    bias_results, bias_history = bias_study()
    draw_figures(histories, results, bias_history)
    batch_rows = [{'epoch': e+1, 'update': e*len(bs)+j+1, 'indices': ' '.join(map(str, ids.tolist()))}
                  for e, bs in enumerate(batches) for j, ids in enumerate(bs)]
    batch_hash = hashlib.sha256(torch.cat([ids for bs in batches for ids in bs]).numpy().tobytes()).hexdigest()
    metrics = {'date': '2026-10-27', 'purpose': 'teacher demonstration, not learner completion',
               'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float64', 'download': False,
               'data_seed': 4501, 'initial_seed': 4502, 'loader_seed': 4503,
               'split_sizes': {'train': 20, 'validation': 100, 'test': 100}, 'dimensions': 200,
               'true_weight': .01, 'true_bias': .05, 'noise_std': .01,
               'objective': 'mean squared error / 2 + lambda * sum(w squared) / 2; no bias penalty',
               'lr': LR, 'batch_size': 5, 'epochs': 100, 'updates': 400,
               'initial_weight_norm': initial[0].norm().item(),
               'initial_null_component_norm': null_component_norm(splits['train'][0], initial[0]),
               'batch_index_sha256': batch_hash, 'candidates': list(CANDIDATES), 'sweep': results,
               'scratch': scratch, 'concise_final_parameters': states,
               'best': best, 'selection': 'minimum final validation half-MSE, then smaller lambda',
               'test_half_mse': test, 'test_evaluations': 1, 'ridge_solutions': ridges,
               'ridge_scope': 'closed-form objective reference, not replacement of selected SGD scheme',
               'bias_control': bias_results}
    with (REPORT/'4_5_metrics.json').open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    write_csv(REPORT/'4_5_epochs.csv', histories)
    write_csv(REPORT/'4_5_lambda_sweep.csv', results)
    write_csv(REPORT/'4_5_batches.csv', batch_rows)
    write_csv(REPORT/'4_5_ridge.csv', ridges)
    print('正式候选最优:', best, '| 独立测试半MSE:', test)
    print('直接解参考:', ridges); print('偏置控制:', bias_results)
    print('手写与框架统一损失尺度；教师运行不代表用户掌握，下一节4.6暂退法。')


if __name__ == '__main__':
    main()
