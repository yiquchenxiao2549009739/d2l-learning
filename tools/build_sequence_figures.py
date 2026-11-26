"""绘制 8.1 的可复现实验图。先运行教师入口 --record。"""
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPORT = ROOT/'reports/ch08'
FIGURES = REPORT/'figures'
BLUE, GREEN, ORANGE, GRAY = '#24688c','#197d77','#b26931','#526c75'


def main():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
    FIGURES.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(FIGURES/f'8_1_{name}.png',dpi=180)
        plt.close(fig)
    def panel():
        return plt.subplots(figsize=(8.5,3.15),layout='constrained')

    with (REPORT/'8_1_trajectories.csv').open(encoding='utf-8-sig',newline='') as file:
        rows = list(csv.DictReader(file))
    def column(key):
        return np.array([float(r[key]) if r[key] else np.nan for r in rows])
    t, clean, x = (column(k) for k in ('time','clean','observed'))
    result = json.loads((REPORT/'8_1_metrics.json').read_text(encoding='utf8'))
    formal = result['formal']
    fig,ax = panel()
    for lo, hi, color, label in [(1,604,BLUE,'训练'),(604,802,GREEN,'验证'),(802,1000,ORANGE,'测试')]:
        ax.axvspan(lo,hi,color=color,alpha=.1)
        ax.text((lo+hi)/2,1.55,label,ha='center',color=color,weight='bold')
    ax.plot(t,x,alpha=.5,color=BLUE,lw=.75,label='有噪观测')
    ax.plot(t,clean,color=ORANGE,lw=1.7,label='无噪真值')
    ax.set(xlim=(1,1000),ylim=(-1.65,1.75),xlabel='时刻 t',ylabel='序列值')
    ax.legend(loc='lower right')
    save(fig,'split')

    fig,ax = panel()
    ax.set(xlim=(0,8),ylim=(0,3.8))
    ax.axis('off')
    for j in range(3):
        for i in range(5):
            v = i+j+1
            ax.text(v,2.9-j*.85,f'x{v}',ha='center',va='center',
                    bbox=dict(boxstyle='round,pad=.4',fc='#dcebef' if i<4 else '#f5e1ce',
                              ec=BLUE if i<4 else ORANGE))
    ax.text(3.9,.4,'蓝：历史输入   /   橙：紧接着的标签',ha='center')
    save(fig,'windows')

    fig,ax = panel()
    epoch = formal['epoch_history']
    ax.plot([r['epoch'] for r in epoch],[r['online_mse'] for r in epoch],'o-',label='在线训练',color=BLUE)
    ax.plot([r['epoch'] for r in epoch],[r['frozen_mse'] for r in epoch],'o-',label='冻结训练评估',color=GREEN)
    ax.set(xlabel='轮次',ylabel='MSE',xticks=range(1,6))
    ax.grid(alpha=.2)
    ax.legend()
    save(fig,'training')

    fig,ax = panel()
    ax.axvspan(604,802,color=GREEN,alpha=.08)
    ax.axvspan(802,1000,color=ORANGE,alpha=.08)
    ax.plot(t,x,color=GRAY,lw=.7,alpha=.4,label='观测')
    ax.plot(t,column('rolling_one'),color=BLUE,lw=1.2,label='滚动单步')
    ax.plot(t,column('fixed_origin_604'),color=ORANGE,lw=1.8,label='固定604闭环')
    ax.set(xlim=(560,1000),xlabel='目标时刻',ylabel='序列值')
    ax.legend(ncol=3,loc='lower right')
    save(fig,'closed_loop')

    fig,ax = panel()
    labels = {'mlp':'闭环MLP','persistence':'末值保持','linear_ar4':'线性AR4','direct64':'直接输出64步'}
    for name,rs in formal['test_horizons'].items():
        ax.plot(range(4),[r['rmse'] for r in rs],'-o',label=labels[name])
    ax.set(xticks=range(4),xticklabels=['1','4','16','64'],xlabel='预测跨度 h（等距类别）',ylabel='测试RMSE')
    ax.legend(ncol=2)
    ax.grid(alpha=.2)
    save(fig,'horizons')

    fig,ax = panel()
    runs = result['validation_experiment']['results']
    ax.bar(list(runs),[rs['score'] for rs in runs.values()],
           color=[GREEN if k==result['validation_experiment']['best'] else BLUE for k in runs])
    ax.set(xlabel='验证候选',ylabel='四个跨度的平均MSE')
    save(fig,'candidates')

    fig,ax = panel()
    def recur(a,b):
        predicted = []
        for _ in range(64):
            a,b = b,2*np.cos(.01)*b-a
            predicted.append(b)
        return predicted
    h = np.arange(1,65)
    ax.plot(h,clean[604:668],label='无噪真值',lw=2,color=GREEN)
    ax.plot(h,recur(*clean[602:604]),'--',label='用无噪起点递推',lw=1.8,color=BLUE)
    ax.plot(h,recur(*x[602:604]),label='用有噪起点递推',lw=1.8,color=ORANGE)
    ax.set(xlabel='从604出发的跨度',ylabel='信号值')
    ax.legend()
    save(fig,'sine')


if __name__ == '__main__':
    main()
