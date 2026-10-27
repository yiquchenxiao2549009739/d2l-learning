# -*- coding: utf-8 -*-
"""4.8整节：连乘、梯度尺度、对称性、Xavier/Kaiming与初始化训练对照。

复用已有环境，CPU/float64人工数据；无安装、无下载。
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

REPORT=ROOT/'reports/ch04'
FIGURES=REPORT/'figures'
ARTIFACTS=ROOT/'artifacts/ch04'
DTYPE=torch.float64
METHODS=('small','large','xavier','kaiming','default_weight','zero')
LABELS={'small':'正态std=.01','large':'正态std=1','xavier':'Xavier gain=1',
        'kaiming':'Kaiming ReLU','default_weight':'默认权重尺度/零bias','zero':'全零'}


def finite_number(value):
    value=float(value)
    return value if math.isfinite(value) else None


def scalar_chains():
    rows=[]
    for activation,start in (('sigmoid',0.),('tanh',0.),('relu',1.)):
        for depth in (1,5,10,20,40):
            x=torch.tensor(start,dtype=DTYPE,requires_grad=True);h=x
            for _ in range(depth):h=getattr(torch,activation)(h)
            h.backward()
            rows.append({'activation':activation,'depth':depth,'value':h.item(),
                         'input_gradient':x.grad.item(),
                         'sigmoid_derivative_upper_bound':.25**depth if activation=='sigmoid' else None})
    precision=[]
    for dtype in (torch.float32,torch.float64):
        for factor in (.25,4.):
            x=torch.tensor(1.,dtype=dtype,requires_grad=True);h=x
            for _ in range(80):h=h*factor
            h.backward()
            precision.append({'dtype':str(dtype),'factor':factor,'depth':80,
                              'value':finite_number(h.item()),'gradient':finite_number(x.grad.item()),
                              'is_finite':bool(torch.isfinite(h)),'is_zero':h.item()==0})
    return rows,precision


def matrix_products():
    generator=torch.Generator().manual_seed(4801)
    bases=[torch.randn(4,4,generator=generator,dtype=DTYPE) for _ in range(100)]
    rows=[]
    for name,scale in (('std1',1.),('std_half',.5),('std_quarter',.25),('orthogonal',None)):
        product=torch.eye(4,dtype=DTYPE)
        for depth in range(101):
            if depth:
                base=bases[depth-1]
                matrix=torch.linalg.qr(base).Q if scale is None else base*scale
                product=product@matrix
            rows.append({'method':name,'depth':depth,'frobenius_norm':product.norm().item(),
                         'spectral_norm':torch.linalg.matrix_norm(product,ord=2).item()})
    return rows


def make_model(method,hidden=(32,32,32,32),seed=4806):
    dimensions=(2,)+tuple(hidden)+(2,)
    # 构造Linear的默认随机数不污染其他实验的随机流。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        modules=[]
        for i,(fan_in,fan_out) in enumerate(zip(dimensions[:-1],dimensions[1:])):
            modules.append(nn.Linear(fan_in,fan_out,dtype=DTYPE))
            if i<len(dimensions)-2:modules.append(nn.ReLU())
        net=nn.Sequential(*modules)
    normal_rng=torch.Generator().manual_seed(seed)
    uniform_rng=torch.Generator().manual_seed(seed+1)
    linears=[m for m in net if isinstance(m,nn.Linear)]
    with torch.no_grad():
        for i,layer in enumerate(linears):
            base=torch.randn(layer.weight.shape,generator=normal_rng,dtype=DTYPE)
            uniform=2*torch.rand(layer.weight.shape,generator=uniform_rng,dtype=DTYPE)-1
            fan_in,fan_out=layer.in_features,layer.out_features
            if method=='small':weight=.01*base
            elif method=='large':weight=base
            elif method=='xavier':weight=math.sqrt(2/(fan_in+fan_out))*base
            elif method=='kaiming':
                gain=math.sqrt(2) if i<len(linears)-1 else 1.
                weight=gain/math.sqrt(fan_in)*base
            elif method=='default_weight':weight=uniform/math.sqrt(fan_in)
            elif method=='zero':weight=torch.zeros_like(base)
            else:raise ValueError(method)
            layer.weight.copy_(weight);layer.bias.zero_()
    return net


def propagation_probe():
    generator=torch.Generator().manual_seed(4802)
    base_X=torch.randn(512,64,generator=generator,dtype=DTYPE)
    bases=[torch.randn(64,64,generator=generator,dtype=DTYPE) for _ in range(12)]
    direction=torch.randn(base_X.shape,generator=generator,dtype=DTYPE)
    direction/=direction.norm()
    rows=[]
    cases=[(s,'relu') for s in ('small','large','xavier','kaiming','default_weight')]+[('xavier','sigmoid')]
    for method,activation in cases:
        x=base_X.clone().requires_grad_();h=x;hidden=[x];pre=[]
        uniform_rng=torch.Generator().manual_seed(4803)
        for base in bases:
            if method=='small':w=.01*base
            elif method=='large':w=base
            elif method=='xavier':w=base/math.sqrt(64)
            elif method=='kaiming':w=base*math.sqrt(2/64)
            else:w=(2*torch.rand(base.shape,generator=uniform_rng,dtype=DTYPE)-1)/math.sqrt(64)
            z=h@w.T;h=getattr(torch,activation)(z);h.retain_grad();hidden.append(h);pre.append(z)
        # 统一末端单位范数方向，观察Jacobian传递，不混入任务损失的尺度。
        (h*direction).sum().backward()
        for depth,value in enumerate(hidden):
            rows.append({'method':method,'activation':activation,'depth':depth,
                         'mean':value.detach().mean().item(),'variance':value.detach().var(unbiased=False).item(),
                         'second_moment':value.detach().square().mean().item(),
                         'gradient_norm':value.grad.norm().item(),
                         'zero_fraction':(value.detach()==0).double().mean().item(),
                         'preactivation_variance':pre[depth-1].detach().var(unbiased=False).item() if depth else None})
    return rows


def initialization_api_probe():
    rows=[];fan_in,fan_out=128,64
    X=torch.randn(4096,fan_in,generator=torch.Generator().manual_seed(4804),dtype=DTYPE)
    for method in ('xavier_normal','xavier_uniform','kaiming_normal'):
        w=torch.empty(fan_out,fan_in,dtype=DTYPE)
        generator=torch.Generator().manual_seed(4810)
        if method=='xavier_normal':nn.init.xavier_normal_(w,gain=1.,generator=generator)
        elif method=='xavier_uniform':nn.init.xavier_uniform_(w,gain=1.,generator=generator)
        else:nn.init.kaiming_normal_(w,mode='fan_in',nonlinearity='relu',generator=generator)
        target=2/(fan_in+fan_out) if method.startswith('xavier') else 2/fan_in
        z=X@w.T;h=z.relu()
        rows.append({'method':method,'fan_in':fan_in,'fan_out':fan_out,
                     'target_weight_variance':target,'sample_weight_variance':w.var(unbiased=False).item(),
                     'sample_min':w.min().item(),'sample_max':w.max().item(),
                     'uniform_bound':math.sqrt(6/(fan_in+fan_out)) if method=='xavier_uniform' else None,
                     'input_second_moment':X.square().mean().item(),'preactivation_second_moment':z.square().mean().item(),
                     'relu_second_moment':h.square().mean().item(),'relu_variance':h.var(unbiased=False).item()})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4810);layer=nn.Linear(fan_in,fan_out,dtype=DTYPE)
    default={'weight_bound':1/math.sqrt(fan_in),'weight_min':layer.weight.min().item(),
             'weight_max':layer.weight.max().item(),'bias_min':layer.bias.min().item(),'bias_max':layer.bias.max().item(),
             'target_weight_variance':1/(3*fan_in),'bias_is_zero':bool((layer.bias==0).all())}
    return rows,default


def symmetry_probe():
    X=torch.tensor([[1.,-2.],[-1.,2.],[.5,1.],[-.5,-1.]],dtype=DTYPE)
    y=torch.tensor([[1.],[-1.],[.3],[-.3]],dtype=DTYPE)
    rows=[];states={}
    for mode in ('tied','different_output','dropout'):
        W1=torch.tensor([[.3,-.2],[.3,-.2]],dtype=DTYPE,requires_grad=True)
        b1=torch.full((2,),.1,dtype=DTYPE,requires_grad=True)
        W2=torch.tensor([[.4,.4 if mode!='different_output' else -.1]],dtype=DTYPE,requires_grad=True)
        b2=torch.zeros(1,dtype=DTYPE,requires_grad=True)
        params=[W1,b1,W2,b2];initial=[p.detach().clone() for p in params]
        rng=torch.Generator().manual_seed(4807)
        first_grads=None
        for step in range(21):
            h=torch.tanh(X@W1.T+b1)
            if mode=='dropout' and step<20:
                h=h*(torch.rand(h.shape,generator=rng,dtype=DTYPE)>=.5)/.5
            loss=((h@W2.T+b2-y)**2).mean()/2
            rows.append({'mode':mode,'step':step,'loss':loss.item(),
                         'hidden_row_difference':(W1[0]-W1[1]).abs().max().item(),
                         'bias_difference':(b1[0]-b1[1]).abs().item()})
            if step==20:break
            loss.backward()
            if step==0:first_grads=[p.grad.detach().clone() for p in params]
            with torch.no_grad():
                for p in params:p-=.1*p.grad;p.grad=None
        states[mode]={'initial':initial,'first_gradients':first_grads,'final':[p.detach().clone() for p in params]}
    W1=torch.tensor([[.3,-.2],[-.4,.5]],dtype=DTYPE);b1=torch.tensor([.1,-.2],dtype=DTYPE)
    W2=torch.tensor([[.4,-.1]],dtype=DTYPE);b2=torch.tensor([.2],dtype=DTYPE)
    reference=torch.relu(X@W1.T+b1)@W2.T+b2
    permutation=torch.tensor([1,0]);factor=3.
    permuted=torch.relu(X@W1[permutation].T+b1[permutation])@W2[:,permutation].T+b2
    rescaled=torch.relu(X@(factor*W1).T+factor*b1)@(W2/factor).T+b2
    odd_ref=torch.tanh(X@W1.T+b1)@W2.T+b2
    odd_changed=torch.tanh(X@(-W1).T-b1)@(-W2).T+b2
    linear_w=torch.zeros(2,1,dtype=DTYPE,requires_grad=True)
    ((X@linear_w-y)**2).mean().backward()
    softmax_w=torch.zeros(2,2,dtype=DTYPE,requires_grad=True)
    F.cross_entropy(X@softmax_w.T,torch.tensor([0,1,0,1])).backward()
    summary={'permutation_output_error':(reference-permuted).abs().max().item(),
             'positive_relu_rescaling_error':(reference-rescaled).abs().max().item(),
             'tanh_sign_symmetry_error':(odd_ref-odd_changed).abs().max().item(),
             'linear_zero_init_gradient':linear_w.grad.tolist(),'softmax_zero_init_gradient':softmax_w.grad.tolist(),
             'tied_first_hidden_gradient_difference':(states['tied']['first_gradients'][0][0]-states['tied']['first_gradients'][0][1]).abs().max().item(),
             'different_output_first_hidden_gradient_difference':(states['different_output']['first_gradients'][0][0]-states['different_output']['first_gradients'][0][1]).abs().max().item()}
    return rows,summary,states


def spectral_and_lars():
    A=torch.tensor([[1.,10.],[0.,1.]],dtype=DTYPE);B=A.T;P=A@B
    spectral={'A':A.tolist(),'B':B.tolist(),'rho_A':torch.linalg.eigvals(A).abs().max().item(),
              'rho_B':torch.linalg.eigvals(B).abs().max().item(),'rho_AB':torch.linalg.eigvals(P).abs().max().item(),
              'norm_AB':torch.linalg.matrix_norm(P,ord=2).item(),
              'product_norm_bound':(torch.linalg.matrix_norm(A,ord=2)*torch.linalg.matrix_norm(B,ord=2)).item(),
              'smallest_singular_AB':torch.linalg.svdvals(P).min().item(),
              'lower_singular_bound':(torch.linalg.svdvals(A).min()*torch.linalg.svdvals(B).min()).item()}
    lars=[]
    for name,w,g in (('layer_small',torch.tensor([1.,0.],dtype=DTYPE),torch.tensor([0.,100.],dtype=DTYPE)),
                     ('layer_large',torch.tensor([100.,0.],dtype=DTYPE),torch.tensor([0.,.01],dtype=DTYPE))):
        lr,eta,beta=.1,.001,0.
        trust=eta*w.norm()/(g.norm()+beta*w.norm())
        update=-lr*trust*(g+beta*w)
        lars.append({'layer':name,'weight_norm':w.norm().item(),'gradient_norm':g.norm().item(),
                     'lr':lr,'eta':eta,'beta':beta,'trust_ratio':trust.item(),
                     'sgd_relative_step':(lr*g.norm()/w.norm()).item(),
                     'lars_relative_step':(update.norm()/w.norm()).item()})
    return spectral,lars


def make_data():
    generator=torch.Generator().manual_seed(4805)
    signs=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=DTYPE)
    splits={}
    for name,per_quadrant in (('train',64),('validation',32),('test',32)):
        directions=signs.repeat_interleave(per_quadrant,dim=0)
        X=directions*(.4+1.2*torch.rand(directions.shape,generator=generator,dtype=DTYPE))
        y=(X[:,0]*X[:,1]<0).long();indices=torch.randperm(len(y),generator=generator)
        splits[name]=(X[indices],y[indices])
    return splits


def train_comparison(splits,epochs=400):
    X,y=splits['train'];VX,Vy=splits['validation'];history=[];summaries=[];states={}
    reference=None
    for method in METHODS:
        net=make_model(method);params=list(net.parameters())
        initial=[p.detach().clone() for p in params];status='completed';updates=0;last_gradient=None
        for epoch in range(epochs+1):
            logits=net(X);loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss):status='stopped_nonfinite_loss';break
            with torch.no_grad():
                val_logits=net(VX);val_loss=F.cross_entropy(val_logits,Vy)
            row={'method':method,'epoch':epoch,'updates':updates,'train_ce':loss.item(),
                 'validation_ce':finite_number(val_loss.item()),
                 'train_accuracy':(logits.argmax(1)==y).double().mean().item(),
                 'validation_accuracy':(val_logits.argmax(1)==Vy).double().mean().item(),
                 'first_weight_gradient_norm':None,'max_weight_gradient_abs':None}
            if epoch<epochs:
                net.zero_grad(set_to_none=True);loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in params):
                    status='stopped_nonfinite_gradient';history.append(row);break
                last_gradient=params[0].grad.norm().item()
                row['first_weight_gradient_norm']=finite_number(last_gradient)
                row['max_weight_gradient_abs']=max(p.grad.abs().max().item() for p in params)
            history.append(row)
            if epoch==epochs:break
            with torch.no_grad():
                for p in params:p-=.1*p.grad
            updates+=1
        final=[p.detach().clone() for p in params]
        states[method]={'initial':initial,'final':final,'updates':updates,'status':status}
        own=[r for r in history if r['method']==method]
        summaries.append({'method':method,'status':status,'updates':updates,'parameters':sum(p.numel() for p in params),
                          'initial_train_ce':own[0]['train_ce'],'initial_first_weight_gradient_norm':own[0]['first_weight_gradient_norm'],
                          'last_finite_record':own[-1]})
        if method=='kaiming':reference=net
    # 在任何结果产生之前固定Kaiming方案；测试不用于选择初始化。
    with torch.no_grad():
        TX,Ty=splits['test'];output=reference(TX)
        test={'method_fixed_before_training':'kaiming','ce':F.cross_entropy(output,Ty).item(),
              'correct':int((output.argmax(1)==Ty).sum()),'samples':len(Ty),'evaluations':1}
    return history,summaries,states,test


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def draw_figures(chains,products,propagation,apis,symmetry,history,lars):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'font.size':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    x=torch.linspace(-8,8,321,dtype=DTYPE);sig=x.sigmoid()
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].plot(x,sig,label='sigmoid');axes[0].plot(x,sig*(1-sig),label='局部导数，最大1/4')
    axes[0].set(xlabel='输入',title='饱和与局部导数');axes[0].legend(fontsize=9)
    for activation in ('sigmoid','tanh','relu'):
        rows=[r for r in chains if r['activation']==activation]
        axes[1].semilogy([r['depth'] for r in rows],[r['input_gradient'] for r in rows],'o-',label=activation)
    axes[1].set(xlabel='层数',ylabel='输入梯度绝对值',title='单位权重标量链：固定指定输入');axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_8_chain.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.7))
    for method,label in (('std1','高斯std=1'),('std_half','高斯std=.5'),('std_quarter','高斯std=.25'),('orthogonal','正交矩阵')):
        rows=[r for r in products if r['method']==method]
        ax.semilogy([r['depth'] for r in rows],[r['frobenius_norm'] for r in rows],label=label)
    ax.set(xlabel='4×4矩阵相乘次数',ylabel='乘积Frobenius范数',title='同一组高斯方向，不同尺度；正交对照');ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_8_matrix_products.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.9))
    for method,activation in [('small','relu'),('large','relu'),('xavier','relu'),('kaiming','relu'),('default_weight','relu'),('xavier','sigmoid')]:
        rows=[r for r in propagation if r['method']==method and r['activation']==activation]
        label=LABELS[method]+'/'+activation
        axes[0].semilogy([r['depth'] for r in rows],[r['second_moment'] for r in rows],label=label)
        axes[1].semilogy([r['depth'] for r in rows],[r['gradient_norm'] for r in rows],label=label)
    axes[0].set(xlabel='前向深度（0为输入）',ylabel='激活二阶矩 E[h²]',title='12层×64：初始化时的信号尺度')
    axes[1].set(xlabel='层位置（反传从右向左）',ylabel='对该层输出的梯度范数',title='固定末端单位方向的梯度')
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_8_layer_scales.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    positions=list(range(len(apis)))
    axes[0].bar([i-.18 for i in positions],[r['target_weight_variance'] for r in apis],width=.35,label='公式目标')
    axes[0].bar([i+.18 for i in positions],[r['sample_weight_variance'] for r in apis],width=.35,label='实际权重方差')
    axes[0].set_xticks(positions,['Xavier正态','Xavier均匀','Kaiming正态']);axes[0].set(title='128输入、64输出',ylabel='权重方差');axes[0].legend(fontsize=9)
    axes[1].bar([i-.18 for i in positions],[r['relu_second_moment'] for r in apis],width=.35,label='ReLU二阶矩')
    axes[1].bar([i+.18 for i in positions],[r['relu_variance'] for r in apis],width=.35,label='ReLU方差')
    axes[1].set_xticks(positions,['Xavier正态','Xavier均匀','Kaiming正态']);axes[1].set(title='二阶矩和方差不能互换',ylabel='激活统计');axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_8_initializers.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for mode,label in (('tied','两层对应权重相同'),('different_output','只把输出权重改成不同'),('dropout','相同初值 + 独立掩码')):
        rows=[r for r in symmetry if r['mode']==mode]
        axes[0].plot([r['step'] for r in rows],[r['hidden_row_difference'] for r in rows],label=label)
    axes[0].set(xlabel='更新次数',ylabel='两个隐藏权重行的最大差',title='打破相同隐藏单元的演化');axes[0].legend(fontsize=8)
    axes[1].set_yscale('log')
    axes[1].bar([-.18,.82],[r['sgd_relative_step'] for r in lars],width=.35,label='普通SGD')
    axes[1].bar([.18,1.18],[r['lars_relative_step'] for r in lars],width=.35,label='无动量LARS示意')
    axes[1].set_xticks([0,1],['参数小/梯度大','参数大/梯度小']);axes[1].set(ylabel='更新范数 / 权重范数',title='有限梯度的分层相对步长');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_8_symmetry_lars.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for method in METHODS:
        rows=[r for r in history if r['method']==method]
        if method!='large':
            axes[0].semilogy([r['epoch'] for r in rows],[r['train_ce'] for r in rows],label=LABELS[method])
        axes[1].plot([r['epoch'] for r in rows],[r['validation_accuracy'] for r in rows],label=LABELS[method])
    failed=[r for r in history if r['method']=='large']
    axes[0].text(.98,.96,'std=1：初始CE %.1f\n最后有限CE %.2e\n完整数值见报告，曲线另列说明' %
                 (failed[0]['train_ce'],failed[-1]['train_ce']),transform=axes[0].transAxes,
                 ha='right',va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'#D7E2E9','alpha':.9})
    axes[0].set(xlabel='全批量更新次数',ylabel='训练平均交叉熵（对数轴）',title='同数据/结构/学习率的初始化对照')
    axes[1].set(xlabel='全批量更新次数',ylabel='验证准确率',ylim=(0,1.05),title='人工XOR：2→32×4→2')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIGURES/'4_8_training.png',dpi=170);plt.close(fig)


def main(epochs=400):
    torch.set_num_threads(1)
    for folder in (REPORT,FIGURES,ARTIFACTS):folder.mkdir(parents=True,exist_ok=True)
    chains,precision=scalar_chains();products=matrix_products();propagation=propagation_probe()
    apis,default=initialization_api_probe();symmetry,symmetry_summary,symmetry_states=symmetry_probe()
    spectral,lars=spectral_and_lars();splits=make_data()
    history,training,states,test=train_comparison(splits,epochs)
    checkpoint=ARTIFACTS/'4_8_states.pt';torch.save({'training':states,'symmetry':symmetry_states},checkpoint)
    metrics={'date':'2026-10-27','purpose':'teacher demonstration, not learner completion',
             'torch':torch.__version__,'device':'cpu','dtype':'float64','download':False,
             'chains':chains,'precision':precision,'matrix_product_final':[r for r in products if r['depth']==100],
             'initialization_apis':apis,'linear_actual_default':default,'symmetry':symmetry_summary,
             'spectral_bounds':spectral,'lars':lars,'training_config':{'dimensions':[2,32,32,32,32,2],
              'epochs':epochs,'lr':.1,'batch_size':256,'data_seed':4805,'initialization_seed':4806,
              'all_biases_zero':True,'common_normal_base':True,'test_method_fixed_before_results':'kaiming'},
             'training':training,'test':test,'checkpoint_local_only':'artifacts/ch04/4_8_states.pt',
             'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (REPORT/'4_8_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,rows in (('chains',chains),('matrix_products',products),('layer_scales',propagation),
                      ('initializers',apis),('symmetry',symmetry),('epochs',history)):
        write_csv(REPORT/('4_8_'+name+'.csv'),rows)
    write_csv(REPORT/'4_8_data.csv',[{'split':name,'index':i,'x1':X[i,0].item(),'x2':X[i,1].item(),'label':int(y[i])}
              for name,(X,y) in splits.items() for i in range(len(y))])
    draw_figures(chains,products,propagation,apis,symmetry,history,lars)
    print('标量sigmoid链:',[r for r in chains if r['activation']=='sigmoid'])
    print('表示范围:',precision);print('矩阵连乘:',metrics['matrix_product_final'])
    print('真实Linear默认初始化:',default);print('对称性:',symmetry_summary)
    print('乘积谱界:',spectral);print('LARS相对步长:',lars)
    for result in training:print('训练:',result)
    print('预先固定Kaiming方案的最终测试:',test)
    print('教师运行不记为用户掌握；下一节4.9环境和分布偏移。')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=400)
    args=parser.parse_args()
    if args.epochs<1:parser.error('--epochs必须大于0')
    main(args.epochs)
