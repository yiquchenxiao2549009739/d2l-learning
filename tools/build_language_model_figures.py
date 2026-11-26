"""用 8.3 的实测 JSON 重绘五幅讲义图。"""
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPORT = ROOT / 'reports/ch08'
FIGURES = REPORT / 'figures'
COLORS = ('#265c8d', '#147f75', '#b65d34', '#6e508d', '#aa3e57')


def save(fig, name):
    fig.savefig(FIGURES / f'8_3_{name}.png', dpi=180)
    plt.close(fig)


def rank_frequency(rows):
    fig, ax = plt.subplots(figsize=(8.7, 3.75), layout='constrained')
    for row, color in zip(rows, COLORS):
        f = row['frequencies']
        fit = row['fits'][0]
        ranks = range(fit['low'], fit['high'] + 1)
        ax.loglog(range(1, len(f) + 1), f, color=color, linewidth=1.8,
                  label=f"{row['order']} 元组  α={fit['alpha']:.3f}")
        ax.loglog(ranks, [math.exp(fit['intercept']) * r**-fit['alpha'] for r in ranks],
                  color=color, alpha=.8, linewidth=1.3, linestyle='--')
    ax.axvspan(10, 500, color='#bfcbd4', alpha=.23, label='拟合排名 10–500')
    ax.set(xlabel='频率排名', ylabel='出现次数')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(alpha=.22, which='major')
    save(fig, 'zipf')


def validation(rows):
    fig, ax = plt.subplots(figsize=(8.7, 3.75), layout='constrained')
    for row, color in zip(rows, COLORS):
        choices = row['validation']
        masses = [v['mass'] for v in choices]
        ppl = [v['perplexity'] for v in choices]
        prior = '低阶' if row['prior'] == 'lower' else '均匀'
        ax.loglog(masses, ppl, 'o-', color=color, label=f"{row['order']} 元 · {prior}")
        best = masses.index(row['best_mass'])
        ax.scatter(masses[best], ppl[best], s=113, linewidths=1.4, facecolors='none', edgecolors=color)
    ax.set_xticks([.1, 1, 10, 100], labels=['0.1', '1', '10', '100'])
    ax.set(xlabel='总伪计数 ε', ylabel='验证困惑度（对数刻度）')
    ax.legend(ncol=2, loc='upper right')
    ax.grid(alpha=.22)
    save(fig, 'smoothing')


def batches(rows):
    fig, axes = plt.subplots(2, 1, figsize=(8.7, 3.8), sharex=True, layout='constrained')
    for (mode, title), ax in zip([('random', '随机采样'), ('sequential', '顺序分区')], axes):
        row = rows[mode]
        for i, starts in enumerate(row['first_three_starts']):
            for lane, start in enumerate(starts):
                ax.add_patch(Rectangle((start-.45, lane-.33), 4.9, .66, facecolor=COLORS[i], alpha=.92))
                ax.text(start+2, lane, f'{start}–{start+4}', ha='center', va='center', color='white')
        ax.set(ylim=(1.6, -.65), xlim=(-.8,34), yticks=[0,1], yticklabels=['行 0', '行 1'])
        ax.set_title(f"{title}，偏移 {row['offset']}", loc='left')
        ax.grid(axis='x', alpha=.17)
    axes[1].set(xlabel='输入在原序列中的位置（蓝 / 绿 / 橙为第 1 / 2 / 3 批）')
    save(fig, 'batches')


def coverage(values):
    fig, ax = plt.subplots(figsize=(8.7, 3.6), layout='constrained')
    for mode, label, color in [('random','随机采样','#265c8d'), ('sequential','顺序分区','#b65d34')]:
        linestyle = '-' if mode == 'random' else '--'
        ax.plot(values['position'], values[mode], marker='.', linestyle=linestyle, color=color, label=label)
    ax.set(xlabel='输入在原序列中的位置', ylabel='每轮被用作输入的概率', ylim=(0,1.08))
    ax.legend(loc='lower center')
    ax.grid(alpha=.22)
    save(fig, 'coverage')


def sentences(row):
    fig, ax = plt.subplots(figsize=(8.7, 2.6), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 3.5))
    ax.axis('off')
    for block, offset in [('X', 1.05), ('Y', 5.45)]:
        lower = block.lower()
        ax.text(offset+1.5, 3.15, block + ('  输入' if block == 'X' else '  下一词元标签'), ha='center', weight='bold')
        for i, sequence in enumerate(row[lower]):
            for j, token in enumerate(sequence):
                color = '#e8edf0' if token == 0 else '#d6eee9'
                ax.add_patch(Rectangle((offset+j, 2-i), .94, .71, facecolor=color, edgecolor='#a0b7bc'))
                ax.text(offset+j+.47, 2.35-i, ['PAD','BOS','EOS','3','4'][token], ha='center', va='center')
    ax.annotate('', (5.25,1.84), (4.26,1.84), arrowprops=dict(arrowstyle='->', color='#546470'))
    ax.text(5, .44, '右下标签 PAD 不进入损失，EOS 仍参与预测', ha='center', color='#546470')
    save(fig, 'sentences')


if __name__ == '__main__':
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False})
    results = json.loads((REPORT / '8_3_metrics.json').read_text(encoding='utf-8'))
    FIGURES.mkdir(parents=True, exist_ok=True)
    rank_frequency(results['book'])
    validation(results['models']['settings'])
    batches(results['toy'])
    coverage(results['toy']['coverage'])
    sentences(results['toy']['sentences'])
