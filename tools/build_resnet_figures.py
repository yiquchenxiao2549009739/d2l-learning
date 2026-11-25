"""7.6原理、形状和研究变体演示及可重建图；不下载数据或训练正式模型。"""
import hashlib
from pathlib import Path
import runpy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(ROOT/'lessons/ch07/06_resnet.py'))


def demonstrations():
    H['COMMON']['configure']('cpu')
    torch.manual_seed(7605)
    gen=torch.Generator().manual_seed(7605)
    x=np.linspace(-1,1,9);y=np.sin(2*x)+.1*x*x;fits={}
    for degree in (1,3,5):
        design=np.stack([x**j for j in range(degree+1)],1);coef=np.linalg.lstsq(design,y,rcond=None)[0]
        fits[str(degree)]=dict(coefficients=coef.tolist(),training_mse=float(np.mean((design@coef-y)**2)))
    design=np.stack([np.ones_like(x),x*x],1);coef=np.linalg.lstsq(design,x,rcond=None)[0]
    nested=dict(x=x.tolist(),target=y.tolist(),fits=fits,nonnested_target='y=x',linear_mse=0.,even_basis_mse=float(np.mean((design@coef-x)**2)))
    inp=torch.tensor([[[[-2.,-1.,0.,1.,2.]]]],dtype=torch.float64,requires_grad=True);identity={}
    for name,cls in [('post',H['Residual']),('pre',H['PreActResidual'])]:
        block=cls(1,1).double().eval()
        nn.init.zeros_(block.conv2.weight);nn.init.zeros_(block.conv2.bias)
        out=block(inp);grad=torch.autograd.grad(out.sum(),inp)[0]
        identity[name]=dict(input=inp.detach().flatten().tolist(),output=out.detach().flatten().tolist(),input_gradient=grad.flatten().tolist())
    gradients={}
    for name,co,stride in [('identity',3,1),('projection',6,2)]:
        block=H['Residual'](3,co,stride).double().eval();X=torch.randn(2,3,7,7,generator=gen,dtype=torch.float64,requires_grad=True)
        main=block.bn2(block.conv2(F.relu(block.bn1(block.conv1(X)))));skip=block.shortcut(X)
        upstream=torch.randn(*main.shape,generator=gen,dtype=torch.float64);gate=(main+skip>0).to(main.dtype)
        full=torch.autograd.grad(block(X),X,upstream,retain_graph=True)[0]
        gm=torch.autograd.grad(main,X,upstream*gate,retain_graph=True)[0];gs=torch.autograd.grad(skip,X,upstream*gate)[0]
        gradients[name]=dict(input_shape=list(X.shape),output_shape=list(main.shape),max_sum_error=(full-gm-gs).abs().max().item(),
            main_gradient_norm=gm.norm().item(),skip_gradient_norm=gs.norm().item())
    block=H['Residual'](3,3).double().train();nn.init.zeros_(block.bn2.weight)
    X=torch.randn(3,3,5,5,generator=gen,dtype=torch.float64);out=block(X);out.square().mean().backward()
    zero=dict(output_vs_relu_input=(out-X.relu()).abs().max().item(),first_conv_gradient=block.conv1.weight.grad.abs().max().item(),
        last_bn_gamma_gradient=block.bn2.weight.grad.abs().max().item(),moving_stats_still_update=int(block.bn2.num_batches_tracked))
    shape_probes={}
    for name,co,stride in [('same',3,1),('project',6,2)]:
        m=H['Residual'](3,co,stride).eval()
        with torch.no_grad():shape_probes[name]=list(m(torch.zeros(4,3,6,6)).shape)
    architectures={}
    for depth in (18,34,50,101,152):
        for size in (96,224):architectures[str(depth)+'_'+str(size)]=H['inspect_net'](dict(H['BASE'],depth=depth,size=size))
    architectures['pre18_96']=H['inspect_net'](dict(H['BASE'],preactivation=True))
    minimum={str(s):H['inspect_net'](dict(H['BASE'],size=s))['outer'][-1]['shape'] for s in (1,2,3,28)}
    H['write_json']('7_6_architectures.json',architectures)
    H['write_json']('7_6_demonstrations.json',dict(nested=nested,zero_branch=identity,branch_gradients=gradients,zero_last_bn=zero,
        shape_probes=shape_probes,minimum_eval_shapes=minimum,bottleneck_conv_weights=dict(two_3x3_256=2*9*256*256,bottleneck_256_64=256*64+9*64*64+64*256),
        note='恒等论证限定匹配维数；post末ReLU仅对非负X保留值，pre无末门但投影支路非恒等；元数据与正式训练分开'))
    files=[]
    for p in sorted((ROOT/'data/FashionMNIST/raw').glob('*ubyte')):
        files.append(dict(filename=p.name,raw_bytes=p.stat().st_size,raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    H['write_json']('7_6_data_integrity.json',dict(strategy='existing Fashion-MNIST read-only',files=files))
    H['write_json']('7_6_sources.json',dict(checked_on='2026-11-25',sources=[
        dict(title='Deep Residual Learning for Image Recognition',url='https://arxiv.org/pdf/1512.03385',locator='PDF5 Table1; PDF6 Fig5',checked='原论文表1的基本/瓶颈结构与18/34/50/101/152重复数'),
        dict(title='Identity Mappings in Deep Residual Networks',url='https://arxiv.org/pdf/1603.05027',locator='PDF2 Fig1; Eq3-5',checked='预激活、加法后恒等映射与支路条件')],
        adaptation='灰度十分类、所有Conv保留bias、后激活投影无BN；瓶颈stride在首1×1；预激活变体统一stem/加末BNReLU，不冒称原论文完整训练配方'))
    return nested,identity,architectures


def figures(data):
    nested,identity,architectures=data;plt=H['plot_setup']();dest=ROOT/'reports/ch07/figures';dest.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.savefig(dest/('7_6_'+name+'.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.2),constrained_layout=True);xx=np.linspace(-1,1,200)
    ax.scatter(nested['x'],nested['target'],label='固定训练点')
    for degree,fit in nested['fits'].items():ax.plot(xx,np.polynomial.polynomial.polyval(xx,fit['coefficients']),label=degree+'次多项式')
    ax.set(xlabel='x',ylabel='拟合值');ax.legend();ax.grid(alpha=.2);save(fig,'function_classes')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
    for name,label in [('post','加法后ReLU'),('pre','加法后恒等')]:
        r=identity[name];axes[0].plot(r['input'],r['output'],'o-',label=label);axes[1].plot(r['input'],r['input_gradient'],'o-',label=label)
    axes[0].set(xlabel='输入X',ylabel='残差分支为0的输出');axes[1].set(xlabel='输入X',ylabel='输出和对输入的梯度')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,'identity')
    fig,ax=plt.subplots(figsize=(8.5,3.2),constrained_layout=True);layers=np.arange(1,31)
    ax.semilogy(layers,.5**layers,label='普通链：.5^L');ax.semilogy(layers,1.05**layers,label='残差例：(1+.05)^L');ax.semilogy(layers,3.**layers,label='残差也可爆炸：3^L')
    ax.set(xlabel='标量链深度L',ylabel='输入梯度绝对值');ax.legend();ax.grid(alpha=.2);save(fig,'gradients')
    def boxes(name,labels,caption):
        fig,ax=plt.subplots(figsize=(8.5,3.2),constrained_layout=True);ax.set(xlim=(0,len(labels)),ylim=(0,1));ax.axis('off')
        for i,t in enumerate(labels):
            ax.text(i+.5,.6,t,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.3',fc='#e4f2f1',ec='#147e83'))
            if i<len(labels)-1:ax.annotate('',xy=(i+1.1,.6),xytext=(i+.9,.6),arrowprops=dict(arrowstyle='->'))
        ax.text(len(labels)/2,.13,caption,ha='center',fontsize=11);save(fig,name)
    def branches(name, main, shortcut, shape, output, caption):
        fig, ax = plt.subplots(figsize=(8.5, 3.2), constrained_layout=True)
        ax.set(xlim=(0, 10), ylim=(0, 4))
        ax.axis('off')
        ax.text(.8, 2., shape, ha='center', va='center', fontsize=11)
        for y, label, color in [(3.15, main, '#147e83'), (1.05, shortcut, '#9a6816')]:
            ax.annotate('', xy=(2.15, y), xytext=(1.55, 2.), arrowprops={'arrowstyle':'->', 'color':color})
            ax.text(4.6, y, label, ha='center', va='center', fontsize=12,
                    bbox={'boxstyle':'round,pad=.4', 'fc':'#f5faf9', 'ec':color})
            ax.annotate('', xy=(7.45, 2.), xytext=(6.9, y), arrowprops={'arrowstyle':'->', 'color':color})
        ax.text(7.7, 2., '+', ha='center', va='center', fontsize=17,
                bbox={'boxstyle':'circle,pad=.22', 'fc':'white', 'ec':'#147e83'})
        ax.annotate('', xy=(8.55, 2.), xytext=(8., 2.), arrowprops={'arrowstyle':'->'})
        ax.text(9.25, 2., output, ha='center', va='center', fontsize=10)
        ax.text(5., .2, caption, ha='center', fontsize=11)
        save(fig, name)
    branches('block','3×3 Conv → BN → ReLU → 3×3 Conv → BN','捷径 S(X)：恒等或投影','输入 X','ReLU','两支在加法节点汇合，输出再经过 ReLU')
    branches('projection','3×3 /2 → BN → ReLU → 3×3 /1 → BN','1×1 /2 投影','C×9×9','2C×5×5\n再过 ReLU','主路和投影均到达 2C×5×5，9 经步幅2取整为5')
    boxes('architecture',['96输入','Stem\n64×24²','2块\n64×24²','2块\n128×12²','2块\n256×6²','2块\n512×3²','GAP\nLinear10'],'命名18层不计三条投影：实际20 Conv + 1 Linear，17 BN')
    boxes('bottleneck',['256输入','1×1\n256→64','3×3\n64→64','1×1\n64→256','+X ReLU'],'固定输出256时，三次卷积权重69632；两次256通道3×3为1179648')
    branches('preactivation','BN → ReLU → Conv3 → BN → ReLU → Conv3','同形捷径：原样传递 X','输入 X','输出 Y','加法直接输出 Y；若需投影，则从首次 BN 与 ReLU 之后取输入')
    bench=ROOT/'reports/ch07/7_6_benchmark.json'
    if bench.exists():
        import json
        r=json.loads(bench.read_text(encoding='utf8'));fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
        for mode,offset in [('train',-.18),('inference',.18)]:
            rows=[x for x in r['results'] if x['mode']==mode];xs=np.arange(4)+offset
            axes[0].bar(xs,[x['milliseconds_per_batch'] for x in rows],.36,label=mode)
            axes[1].bar(xs,[x['peak_allocated_bytes']/2**20 for x in rows],.36,label=mode)
        for ax in axes:ax.set_xticks(range(4));ax.set_xticklabels(['R18/96','R34/96','R50/96','R18/224']);ax.legend()
        axes[0].set_ylabel('每批毫秒');axes[1].set_ylabel('CUDA分配峰值MiB');save(fig,'benchmark')


if __name__=='__main__':figures(demonstrations());print('7.6结构、原理演示与图已生成')
