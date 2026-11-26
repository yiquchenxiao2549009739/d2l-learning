"""根据本节记录绘制原理图和实测图，在项目根目录运行。"""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/ch07'
FIGURES = REPORT/'figures'
D = runpy.run_path(str(ROOT/'lessons/ch07/07_densenet.py'))


def read(name):
    return json.loads((REPORT/('7_7_'+name+'.json')).read_text(encoding='utf8'))


def main():
    plt = D['plot_setup']()
    blue, orange, green = '#256b8f', '#be7133', '#197d77'
    plt.rcParams.update({'font.size':11})
    def axes(columns=1):
        return plt.subplots(1,columns,figsize=(8.5,3.2),layout='constrained')
    def save(fig,name):
        fig.savefig(FIGURES/('7_7_'+name+'.png'),dpi=180)
        plt.close(fig)
    def schematic(labels, title, name):
        fig,ax = axes()
        ax.set(xlim=(0,len(labels)),ylim=(0,1))
        ax.axis('off')
        for j,label in enumerate(labels):
            ax.text(j+.5,.55,label,ha='center',va='center',fontsize=11,
                    bbox=dict(boxstyle='round,pad=.42',fc='#eef6f7',ec=green))
            if j:
                ax.annotate('',(j+.13,.55),(j-.16,.55),arrowprops=dict(arrowstyle='->',color=blue))
        ax.text(len(labels)/2,.13,title,ha='center',fontsize=11)
        save(fig,name)

    fig,ax = axes()
    ax.set(xlim=(-.6,5.2),ylim=(-.6,3.75))
    ax.axis('off')
    labels = ['X0','Y1','Y2','Y3']
    for i,(label,width,color) in enumerate(zip(labels,[4,3,3,3],[blue,green,orange,'#74498b'])):
        ax.text(.35,3-i,label,ha='center',va='center',color='white',fontsize=12,
                bbox=dict(boxstyle='round,pad=.4',fc=color,ec=color))
        ax.text(1.05,3-i,str(width)+' 通道',ha='left',va='center',fontsize=11)
    for i,inputs in enumerate(['原始输入','X0','X0、Y1','X0、Y1、Y2']):
        ax.text(2.68,3-i,inputs,ha='center',va='center',fontsize=11)
    for i,channels in enumerate(['4','4 + 3 = 7','7 + 3 = 10','10 + 3 = 13']):
        ax.text(4.48,3-i,channels,ha='center',va='center',fontsize=11)
    for x, label in [(.65,'特征'),(2.68,'本单元读取'),(4.48,'拼接后宽度')]:
        ax.text(x,3.57,label,ha='center',weight='bold')
    for y in [.5,1.5,2.5]:
        ax.axhline(y,color='#dbe6e8',lw=1)
    save(fig,'concat')
    schematic(['23×9×9','BN → ReLU','1×1 卷积\n10×9×9','平均池化\n10×4×4'],
              '卷积改通道，池化改空间；奇数边长 floor(9/2) = 4','transition')
    schematic(['64×24²','192→96\n24→12','224→112\n12→6','240→120\n6→3','248×3²','GAP\nLinear10'],
              '前三个稠密块后接过渡层；第四个直接进入分类头','architecture')
    fig,ax = axes()
    layer_counts = np.arange(1,17)
    for k,color in [(8,blue),(16,green),(32,orange)]:
        ax.plot(layer_counts,9*k*(64*layer_counts+k*layer_counts*(layer_counts-1)/2),
                marker='o',ms=3,label=f'增长率 {k}',color=color)
    ax.set(xlabel='块内单元数 L',ylabel='累计 3×3 权重数',title='固定输入64通道，后面的卷积读取更多特征')
    ax.grid(alpha=.2)
    ax.legend()
    save(fig,'growth')
    for key in ('metrics','experiments','benchmark','house'):
        if not (REPORT/('7_7_'+key+'.json')).exists():
            raise FileNotFoundError(f'缺少 {key} 实验报告')
    fig,aa = axes(2)
    h = read('metrics')['history']
    aa[0].plot(range(1,len(h)),[r['online_train_ce'] for r in h[1:]],'o-',label='在线训练')
    aa[0].plot(range(len(h)),[r['validation_ce'] for r in h],'o-',label='验证')
    aa[1].plot(range(len(h)),[r['validation_accuracy'] for r in h],'o-',color=green)
    aa[0].set(ylabel='交叉熵')
    aa[1].set(ylabel='验证准确率',ylim=(0,1))
    aa[0].legend()
    for ax in aa:
        ax.set(xlabel='轮次')
        ax.grid(alpha=.2)
    save(fig,'training')
    fig,aa = axes(2)
    runs = read('experiments')['results']
    for name,run in runs.items():
        aa[0].plot([r['epoch'] for r in run['history']],[r['validation_ce'] for r in run['history']],label=name)
    aa[0].set(xlabel='轮次',ylabel='验证交叉熵')
    aa[0].legend(ncol=2)
    aa[1].bar(list(runs),[r['history'][-1]['validation_accuracy'] for r in runs.values()],color=green)
    aa[1].set(ylabel='末轮验证准确率',ylim=(0,1))
    save(fig,'experiments')
    fig,aa = axes(2)
    train = [r for r in read('benchmark')['results'] if r['mode']=='train']
    labels = ['D/96','D/224','D/224\n重算','R18/224']
    aa[0].bar(labels,[r['peak_allocated_bytes']/2**20 for r in train],color=[blue,blue,green,orange])
    aa[0].set(ylabel='CUDA 分配峰值（MiB）')
    aa[1].bar(labels,[r['ms'] for r in train],color=[blue,blue,green,orange])
    aa[1].set(ylabel='训练毫秒/批')
    save(fig,'memory')
    fig,ax = axes()
    result = read('house')
    for name,color,offset,label in [('plain',blue,-.18,'普通MLP'),('dense',green,.18,'拼接MLP')]:
        rows = [r for r in result['folds'] if r['model']==name]
        ax.bar(np.arange(5)+offset,[r['validation_log_rmse'] for r in rows],.36,color=color,label=label)
    ax.set(xticks=range(5),xlabel='验证折',ylabel='对数 RMSE',ylim=(0,.18))
    ax.legend()
    save(fig,'house')
    paths = sorted((ROOT/'data/FashionMNIST/raw').glob('*ubyte'))
    files = [dict(filename=p.name,raw_bytes=p.stat().st_size,raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    D['write_json']('7_7_data_integrity.json',dict(strategy='reuse local read-only',files=files,house_train_sha256=result['train_sha256']))


if __name__ == '__main__':
    main()
