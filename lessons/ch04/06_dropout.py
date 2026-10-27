# -*- coding: utf-8 -*-
"""4.6整节：暂退原理、手写/框架、模式切换和七道教材实验。

复用本地Fashion-MNIST，CPU/float32，无下载、无安装。正式默认训练后
测试一次；额外小子集比较只看验证集，不用候选结果替换正式测试方案。
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST

REPORT = ROOT/'reports/ch04'
FIGURES = REPORT/'figures'
CASES = [('A', 'native', (0., 0.), 0.),
         ('B', 'native', (.2, .5), 0.),
         ('C', 'native', (.5, .2), 0.),
         ('D', 'native', (.1, .1), 0.),
         ('E', 'native', (.5, .5), 0.),
         ('F', 'native', (0., 0.), .001),
         ('G', 'native', (.2, .5), .001),
         ('H', 'dropconnect', (.2, .5), 0.),
         ('I', 'gaussian', (.2, .5), 0.),
         ('J', 'manual', (.2, .5), 0.),
         ('K', 'gaussian', (.1, .1), 0.)]


def dropout_layer(X, p, training=True, generator=None):
    if not 0 <= p <= 1:
        raise ValueError('p must be between 0 and 1')
    if not training or p == 0:
        return X
    if p == 1:
        return X*0  # 保留求导路径，梯度为0。
    mask = torch.rand(X.shape, dtype=X.dtype, device=X.device, generator=generator) > p
    return X*mask.to(X.dtype)/(1-p)


class ManualDropout(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = p

    def forward(self, X):
        return dropout_layer(X, self.p, self.training)


class GaussianNoise(nn.Module):
    """乘性高斯噪声：均值1，方差p/(1-p)，不声称必然优于Dropout。"""
    def __init__(self, p):
        super().__init__(); self.p = p
        if not 0 <= p < 1:
            raise ValueError('Gaussian moment matching needs p < 1')

    def forward(self, X):
        if not self.training or self.p == 0:
            return X
        return X*(1+math.sqrt(self.p/(1-self.p))*torch.randn_like(X))


class DropConnectLinear(nn.Linear):
    """本节控制变体：按权重元素采样，同一前向批次共享权重掩码。"""
    def __init__(self, inputs, outputs, p):
        super().__init__(inputs, outputs); self.p = p

    def forward(self, X):
        weight = dropout_layer(self.weight, self.p, self.training)
        return F.linear(X, weight, self.bias)


def build_model(kind='native', probabilities=(.2, .5), widths=(256, 256),
                input_features=784, classes=10, seed=4600, dtype=torch.float32):
    p1, p2 = probabilities
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == 'dropconnect':
            first = DropConnectLinear(input_features, widths[0], p1)
            second = DropConnectLinear(widths[0], widths[1], p2)
            noises = (nn.Identity(), nn.Identity())
        else:
            first, second = nn.Linear(input_features, widths[0]), nn.Linear(widths[0], widths[1])
            cls = {'native': nn.Dropout, 'manual': ManualDropout, 'gaussian': GaussianNoise}[kind]
            noises = (cls(p1), cls(p2))
        net = nn.Sequential(nn.Flatten(), first, nn.ReLU(), noises[0],
                            second, nn.ReLU(), noises[1], nn.Linear(widths[1], classes)).to(dtype=dtype)
        for module in net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0., std=.01)
                nn.init.zeros_(module.bias)
    return net


def parameter_digest(net):
    digest = hashlib.sha256()
    for parameter in net.parameters(): digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def load_data():
    raw = FashionMNIST(ROOT/'data', train=True, download=False)
    base = TensorDataset(raw.data.unsqueeze(1).float()/255., raw.targets)
    generator = torch.Generator().manual_seed(3601)
    train_ids, val_ids = [], []
    for class_id in range(10):
        ids = torch.where(raw.targets == class_id)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        val_ids.append(ids[:600]); train_ids.append(ids[600:])
    train_ids, val_ids = torch.cat(train_ids), torch.cat(val_ids)
    hashes = {name: hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest()
              for name, ids in (('train', train_ids), ('validation', val_ids))}
    return Subset(base, train_ids), Subset(base, val_ids), hashes


def balanced_subset(dataset, per_class, seed):
    labels = dataset.dataset.tensors[1][dataset.indices]
    generator = torch.Generator().manual_seed(seed)
    parts = []
    for class_id in range(10):
        ids = torch.where(labels == class_id)[0]
        parts.append(ids[torch.randperm(len(ids), generator=generator)[:per_class]])
    ids = torch.cat(parts)
    return Subset(dataset, ids), hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest()


def make_loader(dataset, batch_size=256, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=False, generator=torch.Generator().manual_seed(3603))


def train_epoch(net, loader, optimizer):
    net.train()
    total_loss, correct, samples = 0., 0, 0
    for X, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = net(X)
        losses = F.cross_entropy(logits, y, reduction='none')
        losses.mean().backward(); optimizer.step()
        total_loss += losses.detach().sum().item()
        correct += (logits.detach().argmax(1) == y).sum().item(); samples += len(y)
    return {'online_train_ce': total_loss/samples, 'online_train_accuracy': correct/samples}


def evaluate(net, loader):
    net.eval()
    total_loss, correct, samples = 0., 0, 0
    with torch.no_grad():
        for X, y in loader:
            logits = net(X)
            total_loss += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(1) == y).sum().item(); samples += len(y)
    return {'ce': total_loss/samples, 'accuracy': correct/samples, 'correct': correct, 'samples': samples}


def activation_variance(net, probe, epoch, p1, draws=16):
    """固定参数、固定120验证图；eval数据方差与重复掩码方差分开。"""
    previous = net.training
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(4690+epoch)
        net.eval()
        h1 = net[2](net[1](net[0](probe)))
        h2 = net[5](net[4](h1))
        row = {'epoch': epoch, 'probe_samples': len(probe), 'mask_draws': draws,
               'eval_data_variance_h1': h1.var(0, unbiased=False).mean().item(),
               'eval_data_variance_h2': h2.var(0, unbiased=False).mean().item(),
               'conditional_mask_variance_h1_theory': h1.square().mean().item()*p1/(1-p1)}
        net.train(); noisy1, noisy2 = [], []
        for _ in range(draws):
            d1 = net[3](net[2](net[1](net[0](probe))))
            d2 = net[6](net[5](net[4](d1)))
            noisy1.append(d1); noisy2.append(d2)
        row['mask_variance_h1'] = torch.stack(noisy1).var(0, unbiased=False).mean().item()
        row['mask_variance_h2'] = torch.stack(noisy2).var(0, unbiased=False).mean().item()
    net.train(previous)
    return row


def fit(net, train, validation, epochs=10, lr=.5, wd=0., probe=None, p1=.2, verbose=False):
    torch.manual_seed(4603)
    train_loader = make_loader(train, shuffle=True)
    train_eval, val_loader = make_loader(train), make_loader(validation)
    linears = [module for module in net if isinstance(module, nn.Linear)]
    optimizer = torch.optim.SGD([
        {'params': [module.weight for module in linears], 'weight_decay': wd},
        {'params': [module.bias for module in linears], 'weight_decay': 0.}
    ], lr=lr, momentum=0, foreach=False)
    history, variances = [], []
    for epoch in range(epochs+1):
        online = train_epoch(net, train_loader, optimizer) if epoch else {
            'online_train_ce': None, 'online_train_accuracy': None}
        row = {'epoch': epoch, 'updates': epoch*len(train_loader), **online}
        for name, loader in (('train', train_eval), ('validation', val_loader)):
            result = evaluate(net, loader)
            row[name+'_ce'] = result['ce']; row[name+'_accuracy'] = result['accuracy']
        history.append(row)
        if probe is not None: variances.append(activation_variance(net, probe, epoch, p1))
        if verbose: print('正式训练:', row, flush=True)
    return history, variances


def moment_demo():
    X = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    generator = torch.Generator().manual_seed(4604)
    example = {str(p): dropout_layer(X, p, generator=generator).tolist() for p in (0., .5, 1.)}
    h = torch.tensor([1., 2., 4.], dtype=torch.float64)
    repeated = h.expand(100000, -1)
    Y = dropout_layer(repeated, .5, generator=torch.Generator().manual_seed(4605))
    return {'input': X.tolist(), 'outputs': example, 'p': .5, 'draws': 100000,
            'h': h.tolist(), 'sample_mean': Y.mean(0).tolist(), 'sample_variance': Y.var(0, unbiased=False).tolist(),
            'target_mean': h.tolist(), 'target_variance': h.square().tolist()}


def run_study(train, validation):
    small_train, train_hash = balanced_subset(train, 200, 4601)
    small_val, val_hash = balanced_subset(validation, 100, 4602)
    probe = torch.stack([small_val[c*100+i][0] for c in range(10) for i in range(12)])
    rows, summaries, variances, initial_hashes = [], [], [], {}
    for case, kind, probabilities, wd in CASES:
        net = build_model(kind, probabilities)
        initial_hashes[case] = parameter_digest(net)
        history, var_rows = fit(net, small_train, small_val, epochs=20, wd=wd,
                               probe=probe if case in ('A', 'B') else None, p1=probabilities[0])
        for row in history: rows.append({'case': case, **row})
        for row in var_rows: variances.append({'case': case, **row})
        summaries.append({'case': case, 'kind': kind, 'p1': probabilities[0], 'p2': probabilities[1],
                          'weight_decay': wd, **history[-1]})
        print('子集实验:', summaries[-1], flush=True)
    assert len(set(initial_hashes.values())) == 1
    best = min(summaries, key=lambda row: (row['validation_ce'], row['case']))
    return {'train_samples': 2000, 'validation_samples': 1000, 'subset_seeds': [4601, 4602],
            'subset_index_sha256': {'train': train_hash, 'validation': val_hash},
            'epochs': 20, 'updates_per_case': 160, 'initial_parameter_sha256': initial_hashes,
            'summaries': summaries, 'best': best, 'test_evaluations': 0,
            'scope': 'eleven fixed candidates, one initialization and noise seed; not global optimum',
            'variance_probe': {'balanced_samples': 120, 'mask_draws': 16, 'private_seed_base': 4690,
            'eval_variance': 'variance over images per unit, then mean over units',
            'mask_variance': 'variance across repeated masks at fixed parameters and image, then mean; denominator draws',
            'second_layer': 'includes upstream first-layer mask randomness'}}, rows, variances


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def draw_figures(history, moments, study=None, study_rows=None, variances=None):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'font.size': 11, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    teal, orange = '#147E83', '#D47F37'
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, data, title in zip(axes, (moments['input'], moments['outputs']['0.5']),
                             ('原激活值', 'p=.5：丢弃或乘2')):
        ax.imshow(data, cmap='YlGnBu', vmin=0, vmax=32)
        for i in range(2):
            for j in range(8): ax.text(j, i, f'{data[i][j]:g}', ha='center', va='center', fontsize=10,
                                      color='white' if data[i][j]>18 else '#173448')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    fig.suptitle('按元素随机掩码；保留的值除以1-p，参数本身不删除', fontsize=12)
    fig.tight_layout(); fig.savefig(FIGURES/'4_6_masks.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, actual, target, title in ((axes[0], moments['sample_mean'], moments['target_mean'], '条件均值'),
                                      (axes[1], moments['sample_variance'], moments['target_variance'], '条件方差')):
        ax.bar([i-.18 for i in range(3)], target, width=.35, label='理论', color=teal)
        ax.bar([i+.18 for i in range(3)], actual, width=.35, label='100000次抽样', color=orange)
        ax.set_xticks(range(3), ['h=1', 'h=2', 'h=4']); ax.set_title(title); ax.legend(fontsize=9)
    fig.suptitle('p=.5：E[h\u2032|h]=h；Var(h\u2032|h)=h²', fontsize=12)
    fig.tight_layout(); fig.savefig(FIGURES/'4_6_moments.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for name, label, color in (('train', '训练集eval重算', teal), ('validation', '验证集eval', orange)):
        for ax, key in zip(axes, ('ce', 'accuracy')):
            ax.plot([r['epoch'] for r in history], [r[name+'_'+key] for r in history], 'o-', label=label, color=color)
    axes[0].set(xlabel='epoch', ylabel='平均交叉熵', title='正式 (.2,.5) / lr=.5')
    axes[1].set(xlabel='epoch', ylabel='准确率', title='54,000训练 / 6,000验证', ylim=(0,1))
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES/'4_6_training.png', dpi=170); plt.close(fig)
    if study is None: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for case, color in (('A', teal), ('B', orange)):
        rows = [r for r in study_rows if r['case']==case]
        for key, style, label in (('train_ce', '--', '训练eval'), ('validation_ce', '-', '验证')):
            axes[0].plot([r['epoch'] for r in rows], [r[key] for r in rows], style, color=color, label=f'{case} {label}')
    axes[0].set(xlabel='epoch', ylabel='平均交叉熵', title='A无暂退 / B暂退 (.2,.5)'); axes[0].legend(fontsize=8)
    axes[1].bar([r['case'] for r in study['summaries']], [r['validation_ce'] for r in study['summaries']], color=teal)
    axes[1].set(xlabel='候选编号，见讲义表', ylabel='最终验证CE', title='十一方案：同初值 / 20轮 / 不用测试集')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/'4_6_study.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for layer in (1, 2):
        for case, color in (('A', teal), ('B', orange)):
            rows = [r for r in variances if r['case']==case]
            for ax, key in ((axes[0,layer-1], f'eval_data_variance_h{layer}'),
                            (axes[1,layer-1], f'mask_variance_h{layer}')):
                ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], color=color, label=case)
        axes[0,layer-1].set(title=f'隐藏层{layer}：eval数据方差', ylabel='逐单元样本方差的均值')
        axes[1,layer-1].set(title=f'隐藏层{layer}：固定输入/参数的掩码方差', ylabel='16次掩码的总体方差估计')
    for ax in axes.flatten(): ax.set_xlabel('epoch'); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('固定120张验证图片；A无暂退，B(.2,.5)；第二层包含上游噪声', fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES/'4_6_activation_variance.png', dpi=170); plt.close(fig)


def main(skip_experiments=False):
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    train, validation, hashes = load_data()
    moments = moment_demo(); net = build_model(); initial_hash = parameter_digest(net)
    history, _ = fit(net, train, validation, verbose=True)
    raw_test = FashionMNIST(ROOT/'data', train=False, download=False)
    test_set = TensorDataset(raw_test.data.unsqueeze(1).float()/255., raw_test.targets)
    test = evaluate(net, make_loader(test_set))
    artifact = ROOT/'artifacts/ch04/4_6_dropout.pt'; artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': net.state_dict(), 'kind': 'native', 'probabilities': [.2,.5],
                'widths': [256,256], 'epochs': 10, 'lr': .5, 'model_seed': 4600}, artifact)
    study, study_rows, variances = (None, None, None) if skip_experiments else run_study(train, validation)
    draw_figures(history, moments, study, study_rows, variances)
    metrics = {'date': '2026-10-27', 'purpose': 'teacher demonstration, not learner completion',
               'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float32', 'download': False,
               'split_sizes': {'train': len(train), 'validation': len(validation), 'test': len(test_set)},
               'split_seed': 3601, 'split_index_sha256': hashes, 'model_seed': 4600,
               'loader_seed': 3603, 'noise_seed': 4603, 'initial_parameter_sha256': initial_hash,
               'parameter_count': sum(p.numel() for p in net.parameters()), 'probabilities': [.2,.5],
               'initialization': 'normal std=.01; explicit zero bias', 'lr': .5, 'epochs': 10,
               'batch_size': 256, 'updates': 2110, 'training_tail': 240,
               'final': history[-1], 'test': test, 'test_evaluations': 1, 'moments': moments,
               'study': study, 'checkpoint_local_only': artifact.relative_to(ROOT).as_posix(),
               'checkpoint_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    with (REPORT/'4_6_metrics.json').open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    write_csv(REPORT/'4_6_epochs.csv', history)
    if study is not None:
        write_csv(REPORT/'4_6_study_epochs.csv', study_rows)
        write_csv(REPORT/'4_6_study_summary.csv', study['summaries'])
        write_csv(REPORT/'4_6_activation_variance.csv', variances)
    print('正式最终测试:', test)
    if study is not None: print('子集限定候选最低验证CE:', study['best'])
    print('数据与检查点仅本地；教师运行不记用户掌握。下一节4.7前向传播、反向传播和计算图。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-experiments', action='store_true')
    main(parser.parse_args().skip_experiments)
