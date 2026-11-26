"""根据 8.2 的报告绘制结构图和覆盖率对照。"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPORT = ROOT/'reports/ch08'
FIG = REPORT/'figures'
BLUE, GREEN, ORANGE, GRAY = '#256b8f','#187f78','#b86331','#607c88'


def main():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False})
    result = json.loads((REPORT/'8_2_metrics.json').read_text(encoding='utf-8'))
    FIG.mkdir(parents=True,exist_ok=True)

    def save(fig,name):
        fig.savefig(FIG/f'8_2_{name}.png',dpi=180)
        plt.close(fig)

    fig,ax = plt.subplots(figsize=(8.5,2.7),layout='constrained')
    ax.set(xlim=(0,10),ylim=(0,4))
    ax.axis('off')
    labels = [('原始字符串','Time, time!'),('清洗文本','time time'),('单词词元','time  /  time'),
              ('查词表','4  /  4')]
    for i,(topic,sample) in enumerate(labels):
        x = 1.25+2.5*i
        ax.text(x,2.65,topic,ha='center',weight='bold',color=BLUE)
        ax.text(x,1.65,sample,ha='center',va='center',fontsize=12,
                bbox=dict(boxstyle='round,pad=.45',fc='#e9f3f4',ec=GREEN))
        if i:
            ax.annotate('',(x-.87,1.65),(x-1.67,1.65),arrowprops=dict(arrowstyle='->',color=GRAY))
    ax.text(5,.55,'同一个词表将相同词元映射到同一个 ID',ha='center',color=GRAY)
    save(fig,'pipeline')

    fig,ax = plt.subplots(figsize=(8.5,3.1),layout='constrained')
    top = result['book']['top_words']
    ax.bar([w for w,_ in top],[n for _,n in top],color=BLUE)
    ax.set(xlabel='高频单词',ylabel='全书出现次数')
    ax.grid(axis='y',alpha=.2)
    save(fig,'counts')

    rows = result['experiment']['word_thresholds']
    labels = [str(r['min_freq']) for r in rows]
    fig,ax = plt.subplots(figsize=(8.5,3.15),layout='constrained')
    ax.plot(labels,[r['full_vocab'] for r in rows],'o-',label='全书拟合',color=BLUE)
    ax.plot(labels,[r['train_vocab'] for r in rows],'o-',label='训练前缀拟合',color=GREEN)
    ax.set(xlabel='min_freq（等距列出候选值）',ylabel='词表大小（含未知词）',ylim=(0,5000))
    ax.legend()
    ax.grid(alpha=.2)
    save(fig,'thresholds')

    fig,ax = plt.subplots(figsize=(8.5,3.15),layout='constrained')
    unseen = [100*r['held_unseen']/r['held_tokens'] for r in rows]
    filtered = [100*r['held_filtered']/r['held_tokens'] for r in rows]
    ax.bar(labels,unseen,label='训练期从未出现',color=BLUE)
    ax.bar(labels,filtered,bottom=unseen,label='训练出现但被阈值过滤',color=ORANGE)
    ax.set(xlabel='min_freq（等距列出候选值）',ylabel='占留出段单词词元的百分比')
    ax.legend()
    save(fig,'unknown')

    fig,ax = plt.subplots(figsize=(8.5,3.15),layout='constrained')
    b = result['experiment']['bpe']
    counts = [b['held_words'],b['held_pieces'],b['held_characters']]
    bars = ax.bar(['单词','BPE 子词','字符'],counts,color=[BLUE,GREEN,ORANGE])
    ax.bar_label(bars,padding=3)
    ax.set(ylabel='同一留出段的序列长度',ylim=(0,31000))
    ax.text(.02,.94,'按词内容统计，边界另行保留',transform=ax.transAxes,va='top',color=GRAY)
    save(fig,'granularity')


if __name__ == '__main__':
    main()
