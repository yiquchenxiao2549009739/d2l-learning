"""7.3 原理图、确定性微型演示及只读资源计数；不训练、不安装。"""
import hashlib
from pathlib import Path
import runpy
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(ROOT/'lessons/ch07/03_nin.py'))


def demonstrations():
    gen=torch.Generator().manual_seed(7305)
    X=torch.randn(2,3,4,5,generator=gen,dtype=torch.float64)
    W=torch.randn(4,3,1,1,generator=gen,dtype=torch.float64)
    b=torch.randn(4,generator=gen,dtype=torch.float64)
    conv=F.conv2d(X,W,b)
    dense=(X.permute(0,2,3,1)@W[:,:,0,0].T+b).permute(0,3,1,2)
    maps=torch.tensor([[[[4.,0.]],[[0.,2.]]]],dtype=torch.float64,requires_grad=True)
    logits=F.adaptive_avg_pool2d(maps,1).flatten(1)
    loss=F.cross_entropy(logits,torch.tensor([0]));loss.backward()
    expected=(logits.softmax(1)-torch.tensor([[1.,0.]],dtype=torch.float64))[:,:,None,None]/2
    projection=torch.zeros(10,384,dtype=torch.float64);projection[:,:10]=torch.eye(10,dtype=torch.float64)
    a=torch.zeros(384,dtype=torch.float64);a[0]=1
    b=a.clone();b[10]=3
    result=dict(pointwise_vs_shared_linear_max_error=(conv-dense).abs().max().item(),
        gap_logits=logits.detach().tolist(),gap_ce=loss.item(),gap_gradient=maps.grad.tolist(),
        gap_gradient_max_error=(maps.grad-expected).abs().max().item(),
        softmax_after_gap=logits.softmax(1).detach().tolist(),
        mean_spatial_softmax=maps.softmax(1).mean((2,3)).detach().tolist(),
        projection_distinct_inputs_distance=(a-b).norm().item(),
        projection_outputs_distance=(projection@a-projection@b).norm().item(),
        projection_nullity=374)
    assert result['pointwise_vs_shared_linear_max_error']<1e-12
    assert result['gap_gradient_max_error']<1e-12
    assert result['projection_outputs_distance']==0
    return result


def main():
    H['COMMON']['configure']('cpu')
    structure=H['inspect_net']()
    H['write_json']('7_3_architecture.json',structure)
    H['write_csv']('7_3_layers.csv',structure['weighted_layers'])
    demo=demonstrations();H['write_json']('7_3_demonstrations.json',demo)
    comparisons={"NiN":structure,"AlexNet":H['PREVIOUS']['PREVIOUS']['inspect_net'](),
                 "VGG11":H['PREVIOUS']['inspect_net'](dict(H['PREVIOUS']['BASE'],ratio=1))}
    # 不调用旧课记录函数，不改变旧报告。
    resources={name:dict(parameters=data['parameters'],parameter_float32_bytes=data['parameters']*4,
        weight_multiplications=data.get('weight_multiplications',data.get('weight_multiplications_per_image')))
        for name,data in comparisons.items()}
    weighted=structure['weighted_layers']
    resources['NiN'].update(pointwise_parameters=sum(r['parameters'] for r in weighted if r['kernel']==[1,1]),
        pointwise_weight_multiplications=sum(r['weight_multiplications'] for r in weighted if r['kernel']==[1,1]),
        sum_conv_output_bytes_per_image=sum(r['output_float32_bytes'] for r in weighted),
        warning='逻辑输出字节之和不是训练/预测实际峰值；权重存储不等于全进程显存')
    H['write_json']('7_3_resource_comparison.json',resources)
    files=[]
    for name in ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte'):
        path=ROOT/'data/FashionMNIST/raw'/name
        files.append(dict(filename=name,raw_bytes=path.stat().st_size,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    H['write_json']('7_3_data_integrity.json',dict(strategy='reuse verified local Fashion-MNIST, read-only',files=files))
    plt=H['plot_setup']();folder=ROOT/'reports/ch07/figures'
    def save(fig,name):
        fig.savefig(folder/('7_3_'+name+'.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.6),constrained_layout=True)
    for i,text in enumerate(['空间卷积\n11/5/3/3 核','1×1卷积\n通道混合','1×1卷积\n再次非线性']):
        ax.text(i,1,text+'\n+ ReLU',ha='center',va='center',fontsize=14,
            bbox=dict(boxstyle='round,pad=.5',fc='#e8f4f3',ec='#147e83'))
        if i<2:ax.annotate('',xy=(i+.68,1),xytext=(i+.35,1),arrowprops=dict(arrowstyle='->',lw=2))
    ax.text(1,-.1,'同一组1×1参数用于所有位置；不同层使用不同参数',ha='center',fontsize=13)
    ax.set(xlim=(-.55,2.55),ylim=(-.5,2));ax.axis('off');save(fig,'block')
    fig,ax=plt.subplots(figsize=(8.5,3.4),constrained_layout=True)
    labels=['输入224×224','块1：96×54×54','汇聚：96×26×26','块2：256×26×26','汇聚：256×12×12',
            '块3：384×12×12','汇聚+Dropout：384×5×5','末块：10×5×5','GAP：10×1×1 → 10']
    for i,label in enumerate(labels):
        col,row=i//5,i%5
        ax.text(col*1.6,4-row,label,fontsize=13,va='center',
            bbox=dict(boxstyle='round,pad=.25',fc='#e8f4f3' if i<7 else '#fff2dc',ec='#147e83'))
    ax.set(xlim=(-.12,3.65),ylim=(-.6,4.6));ax.axis('off');save(fig,'architecture')
    fig,axes=plt.subplots(1,3,figsize=(8.5,3.0),constrained_layout=True)
    import numpy as np
    for ax,array,title in zip(axes,[np.array([[4,0]]),np.array([[0,2]]),np.array([demo['gap_logits'][0]])],
                             ['类别0响应图','类别1响应图','平均后的logits']):
        ax.imshow(array,cmap='YlGnBu',vmin=0,vmax=4,aspect='auto')
        for j,v in enumerate(array[0]):ax.text(j,0,str(int(v)),ha='center',va='center',fontsize=18,color='white' if v>=3 else '#193647')
        ax.set(xticks=[],yticks=[]);ax.set_title(title,fontsize=13)
    save(fig,'gap')
    fig,ax=plt.subplots(figsize=(8.5,3.2),constrained_layout=True)
    for y,widths,label in [(1,[384,10,10,10],'教材：直接压到10'),(0,[384,128,128,10],'对照F：先保留128')]:
        ax.text(-.25,y,label,ha='right',va='center',fontsize=12)
        for i,width in enumerate(widths):
            ax.text(i*.85,y,str(width),ha='center',va='center',fontsize=14,
                bbox=dict(boxstyle='round,pad=.3',fc='#fff2dc' if width==10 else '#e8f4f3',ec='#147e83'))
            if i<3:ax.annotate('',xy=(i*.85+.65,y),xytext=(i*.85+.2,y),arrowprops=dict(arrowstyle='->'))
    ax.text(1.2,-.7,'第一个箭头为3×3；后两个为1×1（各接ReLU）',ha='center',fontsize=12)
    ax.set(xlim=(-1.65,2.95),ylim=(-1,1.6));ax.axis('off');save(fig,'bottleneck')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
    for ax,key,scale,title in [(axes[0],'parameters',1e6,'参数/百万'),(axes[1],'weight_multiplications',1e9,'单图权重乘法/十亿')]:
        ax.bar(list(resources),[v[key]/scale for v in resources.values()],color=['#147e83','#75b4b5','#e6ab52'])
        ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
    save(fig,'cost')
    print('NiN structure, demonstrations, five figures and read-only data hashes recorded')


if __name__=='__main__':main()
