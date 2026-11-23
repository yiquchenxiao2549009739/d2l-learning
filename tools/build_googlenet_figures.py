"""7.4结构、降维成本和分支梯度图；只读原始数据，不启动训练。"""
import hashlib
from pathlib import Path
import runpy
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(ROOT/'lessons/ch07/04_googlenet.py'))


def demonstrations():
    torch.manual_seed(7405)
    block=H['Inception'](2,2,(2,3),(1,2),1).double()
    X=torch.randn(2,2,5,7,dtype=torch.float64,requires_grad=True)
    outputs=block.branches(X);Y=torch.cat(outputs,1);G=torch.randn_like(Y)
    full=torch.autograd.grad((Y*G).sum(),X,retain_graph=True)[0]
    pieces=G.split([2,3,2,1],dim=1)
    separate=[torch.autograd.grad((a*g).sum(),X,retain_graph=True)[0] for a,g in zip(outputs,pieces)]
    branch_error=(full-sum(separate)).abs().max().item()
    classifier=nn.Linear(8,3).double()
    maps=torch.randn(2,8,5,7,dtype=torch.float64)
    a=classifier(maps.mean((2,3)))
    b=torch.nn.functional.conv2d(maps,classifier.weight[:,:,None,None],classifier.bias).mean((2,3))
    logits=torch.tensor([[2.,-.5,1.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([0]);eps=.1
    ce=torch.nn.functional.cross_entropy(logits,y,label_smoothing=eps)
    ce.backward();target=torch.tensor([[1-eps+eps/3,eps/3,eps/3]],dtype=torch.float64)
    smoothing_error=(logits.grad-(logits.softmax(1)-target)).abs().max().item()
    minimum=[]
    for size in (1,2,3,31,32,33,96,224):
        info=H['inspect_net'](dict(H['BASE'],size=size))
        minimum.append(dict(size=size,stage_shapes=[r['shape'] for r in info['outer']]))
    # 最小尺寸用实际Tensor验证，不只依赖meta。
    tiny=H['build_net'](dict(H['BASE'],size=1,ratio=16)).eval()
    actual=tiny(torch.ones(1,1,1,1))
    assert actual.shape==(1,10) and torch.isfinite(actual).all()
    assert branch_error<1e-12 and (a-b).abs().max()<1e-12 and smoothing_error<1e-12
    return dict(branch_output_shapes=[list(t.shape) for t in outputs],input_gradient_sum_max_error=branch_error,
        gap_linear_vs_pointwise_gap_max_error=(a-b).abs().max().item(),label_smoothing_target=target.tolist(),
        label_smoothing_ce=ce.item(),label_smoothing_gradient_error=smoothing_error,
        minimum_shapes=minimum,actual_minimum_forward=list(actual.shape))


def main():
    H['COMMON']['configure']('cpu')
    info=H['inspect_net']();H['write_json']('7_4_architecture.json',info);H['write_csv']('7_4_layers.csv',info['weighted_layers'])
    demo=demonstrations();H['write_json']('7_4_demonstrations.json',demo)
    configs={'base96':dict(H['BASE']),'no_reduce96':dict(H['BASE'],reduction=False),
             'two3x3_96':dict(H['BASE'],factorized=True),'base224':dict(H['BASE'],size=224)}
    resources={name:H['inspect_net'](cfg) for name,cfg in configs.items()}
    summary={k:{n:v[n] for n in ('parameters','parameter_float32_bytes','weight_multiplications','conv_count')} for k,v in resources.items()}
    for name,path in [('AlexNet','7_1_architecture.json'),('NiN','7_3_architecture.json')]:
        import json
        data=json.loads((ROOT/'reports/ch07'/path).read_text(encoding='utf8'))
        summary[name]=dict(parameters=data['parameters'],parameter_float32_bytes=data['parameters']*4)
    summary['VGG11']=dict(parameters=128806154,parameter_float32_bytes=128806154*4)
    first=info['blocks'][0];first_rows=[r for r in info['weighted_layers'] if r['name'].startswith('b3.0.')]
    direct_rows=[r for r in resources['no_reduce96']['weighted_layers'] if r['name'].startswith('b3.0.')]
    summary['first_block']=dict(input_shape=first['input_shape'],output_shape=first['output_shape'],
        parameters=sum(r['parameters'] for r in first_rows),no_reduce_parameters=sum(r['parameters'] for r in direct_rows),
        weight_multiplications=sum(r['weight_multiplications'] for r in first_rows),
        no_reduce_weight_multiplications=sum(r['weight_multiplications'] for r in direct_rows))
    H['write_json']('7_4_resources.json',summary)
    files=[]
    for name in ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte'):
        p=ROOT/'data/FashionMNIST/raw'/name;files.append(dict(filename=name,raw_bytes=p.stat().st_size,raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    H['write_json']('7_4_data_integrity.json',dict(strategy='reuse verified local Fashion-MNIST, read-only',files=files))
    plt=H['plot_setup']();folder=ROOT/'reports/ch07/figures'
    from matplotlib.patches import FancyBboxPatch
    def save(fig,name):fig.savefig(folder/('7_4_'+name+'.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.0),constrained_layout=True)
    labels=['1×1 → ReLU','1×1 → 3×3\n各接ReLU','1×1 → 5×5\n各接ReLU','3×3最大汇聚\n→ 1×1 + ReLU']
    for y,label in zip([3,2,1,0],labels):
        ax.add_patch(FancyBboxPatch((.75,y-.28),1.3,.56,boxstyle='round,pad=.03',facecolor='#e8f4f3',edgecolor='#147e83'))
        ax.text(1.4,y,label,ha='center',va='center',fontsize=14)
        ax.annotate('',xy=(.76,y),xytext=(.12,1.5),arrowprops=dict(arrowstyle='->'))
        ax.annotate('',xy=(2.75,1.5),xytext=(2.05,y),arrowprops=dict(arrowstyle='->'))
    ax.text(-.08,1.5,'X',ha='center',fontsize=16);ax.text(2.92,1.5,'cat\ndim=1',ha='center',va='center',fontsize=15)
    ax.set(xlim=(-.3,3.3),ylim=(-.55,3.55));ax.axis('off');save(fig,'inception')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
    f=summary['first_block']
    axes[0].bar(['有降维','无降维'],[f['parameters'],f['no_reduce_parameters']],color=['#147e83','#e6ab52'])
    axes[0].set_ylabel('第一个块参数')
    axes[1].bar(['有降维','无降维'],[f['weight_multiplications']/1e6,f['no_reduce_weight_multiplications']/1e6],color=['#147e83','#e6ab52'])
    axes[1].set_ylabel('单图权重乘法/百万');save(fig,'reduction')
    fig,ax=plt.subplots(figsize=(8.5,3.4),constrained_layout=True)
    for i in range(4):
        ax.text(i,1.1,f'分支{i+1}\n梯度 G{i+1}',ha='center',fontsize=14,bbox=dict(boxstyle='round,pad=.3',fc='#e8f4f3',ec='#147e83'))
        ax.annotate('',xy=(1.5,.03),xytext=(i,.82),arrowprops=dict(arrowstyle='->'))
    ax.text(1.5,-.25,'对共同输入X的四项梯度相加',ha='center',fontsize=15)
    ax.text(1.5,-.75,'cat反向先按通道切片；各分支再用各自Jacobian反传',ha='center',fontsize=12)
    ax.set(xlim=(-.6,3.6),ylim=(-1,1.8));ax.axis('off');save(fig,'gradient')
    fig,ax=plt.subplots(figsize=(8.5,3.2),constrained_layout=True)
    labels=['b1\n64×24×24','b2\n192×12×12','b3：2块\n480×6×6','b4：5块\n832×3×3','b5：2块\nGAP→1024','Linear\n1024→10']
    for i,label in enumerate(labels):
        ax.text(i,1,label,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=.3',fc='#e8f4f3',ec='#147e83'))
        if i<5:ax.annotate('',xy=(i+.68,1),xytext=(i+.35,1),arrowprops=dict(arrowstyle='->'))
    ax.text(2.5,-.1,'同一批样本串行经过五段；每个Inception内部有四个分支',ha='center',fontsize=13)
    ax.set(xlim=(-.6,5.6),ylim=(-.5,1.9));ax.axis('off');save(fig,'architecture')
    fig,ax=plt.subplots(figsize=(8.5,3.0),constrained_layout=True)
    names=['AlexNet','VGG11','NiN','GoogLeNet'];values=[summary['AlexNet']['parameters'],summary['VGG11']['parameters'],summary['NiN']['parameters'],info['parameters']]
    ax.bar(names,[v/1e6 for v in values],color=['#75b4b5','#e6ab52','#147e83','#28536b'])
    ax.set_ylabel('单通道十类参数/百万');save(fig,'parameters')
    print('GoogLeNet: structure, five principle figures, demonstrations and read-only hashes recorded')


if __name__=='__main__':main()
