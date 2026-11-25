"""7.5统计/模式/研究思路演示与原理图；复用现有环境，无训练、下载或安装。"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import runpy
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(ROOT/'lessons/ch07/05_batch_norm.py'))


def demonstrations():
    gen=torch.Generator().manual_seed(7505)
    X=torch.tensor([[1.,10.],[3.,14.]],dtype=torch.float64)
    w,b=torch.ones(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64)
    rm,rv=torch.zeros_like(w),torch.ones_like(w)
    y,mean,var=H['batch_norm'](X,w,b,rm,rv,True)
    book,book_mean,book_var=H['textbook_batch_norm'](X,w,b,rm,rv)
    modes=[]
    for training in (True,False):
        for grad in (True,False):
            layer=H['ManualBatchNorm'](2,2).double();layer.train(training)
            with torch.set_grad_enabled(grad):
                out=layer(X)
                old,new_m,new_v=H['textbook_batch_norm'](X,w,b,rm,rv)
            modes.append(dict(training=training,grad_enabled=grad,manual_output=out.detach().tolist(),
                manual_batches=int(layer.num_batches_tracked),manual_requires_grad=out.requires_grad,
                book_output=old.tolist(),book_mean_update=(new_m-rm).abs().max().item()))
    # 批组成改变训练输出，eval固定状态时不依赖同批其他样本。
    batches=[torch.tensor([[1.,4.],[2.,5.],[3.,6.]],dtype=torch.float64),
             torch.tensor([[1.,4.],[-2.,-1.],[4.,9.]],dtype=torch.float64)]
    composition={}
    for training in (True,False):
        values=[]
        for batch in batches:
            layer=H['ManualBatchNorm'](2,2).double();layer.train(training)
            values.append(layer(batch)[0].detach().tolist())
        composition['train' if training else 'eval']=values
    # 训练偏置抵消；eval立即删除偏置需同步调整running_mean。
    a=torch.randn(8,2,generator=gen,dtype=torch.float64);bias=torch.tensor([.7,-1.2],dtype=torch.float64,requires_grad=True)
    one=H['ManualBatchNorm'](2,2).double();two=H['ManualBatchNorm'](2,2).double()
    p,q=one(a+bias),two(a);train_bias_error=(p-q).abs().max().item()
    g=torch.randn(8,2,generator=gen,dtype=torch.float64);(p*g).sum().backward()
    one.eval();two.eval();eval_naive_error=(one(a+bias)-two(a)).abs().max().item()
    two.load_state_dict(one.state_dict())
    with torch.no_grad():two.running_mean.sub_(bias)
    eval_adjusted_error=(one(a+bias)-two(a)).abs().max().item()
    linear=nn.Linear(3,2).double();norm=H['ManualBatchNorm'](2,2).double().eval()
    with torch.no_grad():
        norm.running_mean.copy_(torch.tensor([.3,-.2]));norm.running_var.copy_(torch.tensor([.8,1.7]));norm.weight.copy_(torch.tensor([2.,.5]));norm.bias.copy_(torch.tensor([.1,-.4]))
    inputs=torch.randn(7,3,generator=gen,dtype=torch.float64);scale=norm.weight/torch.sqrt(norm.running_var+norm.eps)
    fused=F.linear(inputs,scale[:,None]*linear.weight,norm.bias+scale*(linear.bias-norm.running_mean))
    fuse_error=(fused-norm(linear(inputs))).abs().max().item()
    assert max(train_bias_error,eval_adjusted_error,fuse_error)<1e-12
    return dict(example=dict(input=X.tolist(),mean=X.mean(0).tolist(),biased_var=X.var(0,unbiased=False).tolist(),
        normalized_output=y.tolist(),normalized_biased_var=y.var(0,unbiased=False).tolist(),running_mean=mean.tolist(),
        native_compatible_running_var=var.tolist(),book_running_var=book_var.tolist()),modes=modes,batch_composition=composition,
        bias=dict(training_error=train_bias_error,bias_gradient_max=bias.grad.abs().max().item(),eval_naive_deletion_error=eval_naive_error,
                  eval_adjusted_running_mean_error=eval_adjusted_error),eval_linear_bn_fusion_error=fuse_error)


def research():
    gen=torch.Generator().manual_seed(7506)
    X=torch.randn(2,4,2,3,generator=gen,dtype=torch.float64)
    def standard(a,axes):return (a-a.mean(axes,keepdim=True))/torch.sqrt(a.var(axes,unbiased=False,keepdim=True)+1e-5)
    values={'BatchNorm2d':(nn.BatchNorm2d(4,affine=False,track_running_stats=False).double(),standard(X,(0,2,3))),
        'LayerNorm':(nn.LayerNorm((4,2,3),elementwise_affine=False).double(),standard(X,(1,2,3))),
        'InstanceNorm2d':(nn.InstanceNorm2d(4,affine=False).double(),standard(X,(2,3))),
        'GroupNorm2':(nn.GroupNorm(2,4,affine=False).double(),standard(X.reshape(2,2,2,2,3),(2,3,4)).reshape_as(X))}
    errors={name:(m(X)-target).abs().max().item() for name,(m,target) in values.items()}
    U=torch.rand(2000,generator=gen,dtype=torch.float64);exp_samples=-torch.log1p(-U)
    transformed=-torch.expm1(-exp_samples)
    gaussian=torch.distributions.Normal(0.,1.).icdf(transformed.clamp(1e-6,1-1e-6))
    discrete=torch.tensor([0.,0.,1.,2.],dtype=torch.float64)
    empirical=torch.searchsorted(discrete,discrete,right=True).double()/len(discrete)
    raw=torch.randn(100,3,generator=gen,dtype=torch.float64)
    matrix=torch.tensor([[1.,.8,.4],[0.,.6,.1],[0.,0.,.4]],dtype=torch.float64)
    A=raw@matrix;center=A-A.mean(0);cov=center.T@center/len(A)
    vals,vecs=torch.linalg.eigh(cov);eps=1e-5
    W=(vecs*(vals+eps).rsqrt()[None,:])@vecs.T;Z=center@W;whitened=Z.T@Z/len(Z)
    diag=center/torch.sqrt(cov.diag()+eps);diagonal_cov=diag.T@diag/len(diag)
    deficient=torch.randn(3,4,generator=gen,dtype=torch.float64);deficient-=deficient.mean(0)
    deficient_cov=deficient.T@deficient/3
    assert max(errors.values())<1e-12 and (transformed-U).abs().max()<1e-12
    expected=(vecs*(vals/(vals+eps))[None,:])@vecs.T
    assert (whitened-expected).abs().max()<1e-12
    result=dict(normalization_axis_errors=errors,pit=dict(known_distribution='Exp(rate1), exact continuous CDF',
        identity_error=(transformed-U).abs().max().item(),uniform_mean=transformed.mean().item(),normal_mean=gaussian.mean().item(),
        normal_population_std=gaussian.std(unbiased=False).item(),discrete_values=discrete.tolist(),empirical_cdf=empirical.tolist()),
        whitening=dict(eps=eps,input_covariance=cov.tolist(),diagonal_standardization_covariance=diagonal_cov.tolist(),
        whitened_covariance=whitened.tolist(),input_eigenvalues=vals.tolist(),formula_error=(whitened-expected).abs().max().item(),
        deficient_m=3,deficient_C=4,deficient_rank=int(torch.linalg.matrix_rank(deficient_cov))))
    return result,exp_samples,transformed,cov,whitened


def api_reference():
    from html.parser import HTMLParser
    class Text(HTMLParser):
        def __init__(self):super().__init__();self.parts=[]
        def handle_data(self,data):self.parts.append(data)
    records=json.loads((ROOT/'tmp/pdfs/ch07_5/api_sources.json').read_text(encoding='utf8'))
    for i,record in enumerate(records):
        assert 'error' not in record,record
        parser=Text();parser.feed((ROOT/'tmp/pdfs/ch07_5'/('api_'+str(i)+'.html')).read_text(encoding='utf8'))
        joined=' '.join(parser.parts)
        record['checked_terms']={term:term in joined for term in ('affine','momentum','track_running_stats','unbiased')}
        assert all(record['checked_terms'].values())
    H['write_json']('7_5_api_sources.json',dict(checked_date='2026-11-25',installed_torch='2.4.1+cu121',sources=records))
    content='''# 7.5 官方API核对

2026-11-25，使用与本机torch2.4.1相应的PyTorch2.4文档。三个页面实际获取成功，标题、解析地址、SHA-256与关键词检查见7_5_api_sources.json；完整HTML只保留在本地tmp。

| API | 输入 | 每通道统计元素 |
| --- | --- | --- |
| [BatchNorm1d](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BatchNorm1d.html) | (N,C)或(N,C,L) | N或N×L |
| [BatchNorm2d](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BatchNorm2d.html) | (N,C,H,W) | N×H×W |
| [BatchNorm3d](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BatchNorm3d.html) | (N,C,D,H,W) | N×D×H×W |

训练前向使用有偏方差，移动方差更新使用无偏估计；eps在平方根内。momentum默认.1表示新统计的更新权重，与优化器动量不同。affine控制可学习gamma/beta；track_running_stats=False时训练和eval均使用当前批统计。momentum=None采用累计平均，本节手写类只实现固定更新权重，不冒称覆盖全部框架选项。

本节ManualBatchNorm仅接受2维/4维，参数形状(C,)，广播到输入。weight对应gamma，bias对应beta；running_mean、running_var、num_batches_tracked注册为buffer，随to/double迁移并进入state_dict。
'''
    (ROOT/'reports/ch07/7_5_api_reference.md').write_text(content,encoding='utf8')


def main():
    H['COMMON']['configure']('cpu');demo=demonstrations();res,raw,U,cov,white=research()
    H['write_json']('7_5_demonstrations.json',demo);H['write_json']('7_5_research.json',res)
    H['write_json']('7_5_architectures.json',{k:H['inspect_net'](v) for k,v in H['formal_configs']().items()})
    api_reference()
    files=[]
    for name in ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte'):
        path=ROOT/'data/FashionMNIST/raw'/name;files.append(dict(filename=name,raw_bytes=path.stat().st_size,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    H['write_json']('7_5_data_integrity.json',dict(strategy='verified local Fashion-MNIST, read-only',files=files))
    plt=H['plot_setup']();folder=ROOT/'reports/ch07/figures'
    def save(fig,name):fig.savefig(folder/('7_5_'+name+'.png'),dpi=180);plt.close(fig)
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.0),constrained_layout=True)
    for ax,rows,title in [(axes[0],4,'BN1d：沿N统计每列C'),(axes[1],8,'BN2d：沿N、H、W统计每个C')]:
        image=np.broadcast_to(np.arange(3)[None,:],(rows,3));ax.imshow(image,aspect='auto',cmap='YlGnBu',vmin=0,vmax=3)
        ax.set(xticks=[0,1,2],xticklabels=['C0','C1','C2'],yticks=[]);ax.set_title(title,fontsize=13)
    save(fig,'axes')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.0),constrained_layout=True)
    x=np.linspace(-3,3,120)
    for gamma,beta,label in [(1,0,'γ1 β0'),(2,0,'γ2 β0'),(.5,1,'γ.5 β1')]:axes[0].plot(x,gamma*x+beta,label=label)
    axes[0].set(xlabel='x_hat',ylabel='仿射输出');axes[0].legend(fontsize=10)
    values=[1/(1+e) for e in (1e-5,.1,1.)];axes[1].bar(['ε1e-5','ε.1','ε1'],values,color='#147e83');axes[1].set_ylabel('原方差1时，x_hat的有偏方差')
    save(fig,'affine_transform')
    fig,ax=plt.subplots(figsize=(8.5,3.0),constrained_layout=True)
    means=[0.]*5+[4.]*8;moving=0.;curve=[]
    for value in means:moving=.9*moving+.1*value;curve.append(moving)
    ax.plot(range(1,14),means,'o-',label='当前批均值');ax.plot(range(1,14),curve,'o-',label='移动均值m=.1')
    ax.set(xlabel='训练批次',ylabel='均值');ax.legend();save(fig,'moving_statistics')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.0),constrained_layout=True)
    labels=['train+grad','train+no_grad','eval+grad','eval+no_grad']
    axes[0].bar(range(4),[r['manual_batches'] for r in demo['modes']],color='#147e83');axes[0].set(xticks=range(4),xticklabels=labels,ylabel='手写层的批计数增加')
    axes[1].bar(range(4),[1 if r['book_mean_update'] else 0 for r in demo['modes']],color='#e6ab52');axes[1].set(xticks=range(4),xticklabels=labels,ylabel='教材函数是否更新统计')
    for ax in axes:ax.tick_params(axis='x',labelrotation=20,labelsize=9)
    save(fig,'modes')
    fig,ax=plt.subplots(figsize=(8.5,3.1),constrained_layout=True)
    labels=['28输入','Conv5+BN\n6×24×24','Pool→12\nConv5+BN\n16×8×8','Pool→4\nFlatten256','120+BN\n84+BN','Linear10']
    for i,text in enumerate(labels):
        ax.text(i,1,text,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=.3',fc='#e8f4f3',ec='#147e83'))
        if i<5:ax.annotate('',xy=(i+.68,1),xytext=(i+.35,1),arrowprops=dict(arrowstyle='->'))
    ax.text(2.5,-.1,'四个BN均在Conv/Linear后、Sigmoid前；最后logits不做BN',ha='center',fontsize=12)
    ax.set(xlim=(-.55,5.55),ylim=(-.5,2));ax.axis('off');save(fig,'architecture')
    fig,axes=plt.subplots(1,3,figsize=(8.5,3.0),constrained_layout=True)
    axes[0].hist(raw.numpy(),bins=24,color='#e6ab52');axes[0].set_title('已知指数分布样本',fontsize=12)
    axes[1].hist(U.numpy(),bins=20,color='#147e83');axes[1].set_title('精确CDF→近似均匀',fontsize=12)
    image=axes[2].imshow(white.numpy(),cmap='coolwarm',vmin=-1,vmax=1);axes[2].set_title('正则化白化的协方差',fontsize=12)
    axes[2].set(xticks=[0,1,2],yticks=[0,1,2]);fig.colorbar(image,ax=axes[2],shrink=.8)
    save(fig,'research')
    print('BN demonstrations, six principle figures, API sources and data hashes recorded')


if __name__=='__main__':main()
