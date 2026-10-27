# -*- coding: utf-8 -*-
"""4.7整节：前向数值、计算图、手写反传、内存、二阶导数和分阶段执行。

小型人工数据，现有环境，无安装、无下载。CPU/float64完成数学核对；
设备演示按实际GPU数量选择，单GPU的CPU→GPU演示不记为多GPU实测。
"""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('CUDA_CACHE_PATH', str(ROOT/'.cache/cuda'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch
from torch import nn
from torch.nn import functional as F

REPORT = ROOT/'reports/ch04'
FIGURES = REPORT/'figures'
DTYPE = torch.float64


def fixture(bias=False):
    X = torch.tensor([[1., -2.]], dtype=DTYPE)
    y = torch.tensor([1], dtype=torch.long)
    W1 = torch.tensor([[.5, .2], [-.3, .4], [.1, -.2]], dtype=DTYPE)
    W2 = torch.tensor([[.2, -.1, .3], [-.4, .2, -.2]], dtype=DTYPE)
    b1 = torch.tensor([.05, -.1, .1], dtype=DTYPE) if bias else None
    return X, y, W1, W2, b1


def forward_values(X, y, W1, W2, lambd=.1, b1=None):
    Z = X@W1.T
    if b1 is not None: Z = Z+b1
    H = torch.relu(Z)
    O = H@W2.T
    L = F.cross_entropy(O, y)
    S = lambd*(W1.square().sum()+W2.square().sum())/2
    return {'Z': Z, 'H': H, 'O': O, 'L': L, 'S': S, 'J': L+S}


@torch.no_grad()
def manual_gradients(X, y, W1, W2, lambd=.1, b1=None):
    cache = forward_values(X, y, W1, W2, lambd, b1)
    GO = cache['O'].softmax(1)
    GO[torch.arange(len(y)), y] -= 1
    GO /= len(y)
    GH = GO@W2
    GZ = GH*(cache['Z']>0)
    data_W2, data_W1 = GO.T@cache['H'], GZ.T@X
    return {**cache, 'GO': GO, 'GH': GH, 'GZ': GZ,
            'W1': data_W1+lambd*W1, 'W2': data_W2+lambd*W2,
            'X': GZ@W1, 'b1': GZ.sum(0) if b1 is not None else None,
            'data_W1': data_W1, 'regularization_W1': lambd*W1,
            'data_W2': data_W2, 'regularization_W2': lambd*W2}


def single_example(bias=False):
    X, y, W1, W2, b1 = fixture(bias)
    X.requires_grad_(); W1.requires_grad_(); W2.requires_grad_()
    if b1 is not None: b1.requires_grad_()
    cache = forward_values(X, y, W1, W2, .1, b1)
    for tensor in cache.values(): tensor.retain_grad()
    cache['J'].backward()
    manual = manual_gradients(X.detach(), y, W1.detach(), W2.detach(), .1,
                              b1.detach() if b1 is not None else None)
    errors = {'W1': (W1.grad-manual['W1']).abs().max().item(),
              'W2': (W2.grad-manual['W2']).abs().max().item(),
              'X': (X.grad-manual['X']).abs().max().item()}
    if bias: errors['b1'] = (b1.grad-manual['b1']).abs().max().item()
    with torch.no_grad():
        updated = forward_values(X, y, W1-.1*W1.grad, W2-.1*W2.grad, .1,
                                 b1-.1*b1.grad if bias else None)
    return {'bias': bias, 'input': X.detach().tolist(), 'label': 1,
            'W1': W1.detach().tolist(), 'W2': W2.detach().tolist(),
            'b1': b1.detach().tolist() if bias else None,
            'forward': {k:v.detach().tolist() for k,v in cache.items()},
            'probabilities': cache['O'].detach().softmax(1).tolist(),
            'gradients': {k:v.detach().tolist() if v is not None else None for k,v in manual.items()
                          if k not in cache},
            'intermediate_autograd': {k:v.grad.detach().tolist() for k,v in cache.items()},
            'max_gradient_errors': errors, 'step_lr': .1,
            'after_one_step_J': updated['J'].item()}


def make_data():
    generator = torch.Generator().manual_seed(4701)
    signs = torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]], dtype=DTYPE)
    result = {}
    for name, per_quadrant in (('train',16),('validation',8),('test',8)):
        S = signs.repeat_interleave(per_quadrant, dim=0)
        X = S*(.4+1.2*torch.rand(S.shape, generator=generator, dtype=DTYPE))
        y = (X[:,0]*X[:,1]<0).long()
        ids = torch.randperm(len(X), generator=generator)
        result[name] = (X[ids], y[ids])
    return result


def train_three_paths(splits):
    generator = torch.Generator().manual_seed(4702)
    start = [.3*torch.randn(8,2,generator=generator,dtype=DTYPE),
             .3*torch.randn(2,8,generator=generator,dtype=DTYPE)]
    manual = [p.clone() for p in start]
    automatic = [p.clone().requires_grad_() for p in start]
    net = nn.Sequential(nn.Linear(2,8,bias=False), nn.ReLU(), nn.Linear(8,2,bias=False)).to(dtype=DTYPE)
    with torch.no_grad():net[0].weight.copy_(start[0]);net[2].weight.copy_(start[1])
    optimizer = torch.optim.SGD(net.parameters(),lr=.2,weight_decay=.01,foreach=False)
    X,y = splits['train'];history=[];max_error=0.
    for epoch in range(201):
        if epoch:
            grads = manual_gradients(X,y,*manual,lambd=.01)
            manual[0] -= .2*grads['W1'];manual[1] -= .2*grads['W2']
            for p in automatic:p.grad=None
            forward_values(X,y,*automatic,lambd=.01)['J'].backward()
            with torch.no_grad():
                for p in automatic:p -= .2*p.grad
            optimizer.zero_grad(set_to_none=True);F.cross_entropy(net(X),y).backward();optimizer.step()
        with torch.no_grad():
            for p,q,r in zip(manual,automatic,(net[0].weight,net[2].weight)):
                max_error=max(max_error,(p-q).abs().max().item(),(p-r).abs().max().item())
            values=forward_values(X,y,*manual,lambd=.01)
            validation=forward_values(*splits['validation'],*manual,lambd=.01)
            history.append({'epoch':epoch,'updates':epoch,'train_ce':values['L'].item(),
                            'penalty':values['S'].item(),'objective':values['J'].item(),
                            'validation_ce':validation['L'].item(),
                            'train_accuracy':(values['O'].argmax(1)==y).double().mean().item(),
                            'validation_accuracy':(validation['O'].argmax(1)==splits['validation'][1]).double().mean().item(),
                            'max_parameter_error_so_far':max_error})
    test=forward_values(*splits['test'],*manual,lambd=.01)
    return {'dimensions':[2,8,2],'bias':False,'parameters':32,'data_seed':4701,'model_seed':4702,
            'epochs':200,'lr':.2,'lambda':.01,'batch_size':64,'split_sizes':{k:len(v[0]) for k,v in splits.items()},
            'initial_weights':[p.tolist() for p in start],'final_weights':[p.tolist() for p in manual],
            'max_three_path_parameter_error':max_error,'final':history[-1],
            'test_ce':test['L'].item(),'test_correct':int((test['O'].argmax(1)==splits['test'][1]).sum()),
            'test_evaluations':1},history


def memory_probe(batch_size):
    generator=torch.Generator().manual_seed(4704)
    X=torch.randn(batch_size,784,generator=generator,dtype=torch.float32)
    y=torch.arange(batch_size,dtype=torch.long)%10
    W1=(.01*torch.randn(256,784,generator=generator)).requires_grad_()
    W2=(.01*torch.randn(10,256,generator=generator)).requires_grad_()
    def key(t):return (str(t.device),t.untyped_storage().data_ptr())
    parameter_keys={key(W1),key(W2)}
    records=[];storages={}
    def pack(t):
        storage_key=key(t)
        if storage_key not in storages:
            storages[storage_key]={'bytes':t.untyped_storage().nbytes(),'parameter':storage_key in parameter_keys}
        records.append({'shape':list(t.shape),'dtype':str(t.dtype),'logical_bytes':t.numel()*t.element_size(),
                        'parameter_storage':storage_key in parameter_keys})
        return t.detach()
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        values=forward_values(X,y,W1,W2,.01)
        values['J'].backward()
    parameter_bytes=sum(p.numel()*p.element_size() for p in (W1,W2))
    grad_bytes=sum(p.grad.numel()*p.grad.element_size() for p in (W1,W2))
    nonparameter=sum(v['bytes'] for v in storages.values() if not v['parameter'])
    inference_saves=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:inference_saves.append(t) or t,lambda t:t),torch.no_grad():
        forward_values(X,y,W1,W2,.01)
    row={'batch_size':batch_size,'parameters':W1.numel()+W2.numel(),'parameter_bytes':parameter_bytes,
         'parameter_gradient_bytes':grad_bytes,'saved_logical_bytes_with_repeats':sum(r['logical_bytes'] for r in records),
         'unique_saved_storage_bytes':sum(v['bytes'] for v in storages.values()),
         'saved_nonparameter_storage_bytes':nonparameter,
         'training_payload_lower_bound_bytes':parameter_bytes+grad_bytes+nonparameter,
         'inference_named_forward_budget_bytes':parameter_bytes+batch_size*(784+2*256+10)*4,
         'no_grad_autograd_save_count':len(inference_saves)}
    return row,records


def second_order():
    X,y,W1,W2,_=fixture()
    theta=torch.cat((W1.flatten(),W2.flatten())).requires_grad_()
    def scalar(t,lambd=.1):return forward_values(X,y,t[:6].reshape(3,2),t[6:].reshape(2,3),lambd)['J']
    gradient=torch.autograd.grad(scalar(theta),theta,create_graph=True)[0]
    direction=torch.randn(theta.shape,generator=torch.Generator().manual_seed(4703),dtype=DTYPE)
    direction/=direction.norm()
    hvp=torch.autograd.grad((gradient*direction).sum(),theta)[0]
    H=torch.autograd.functional.hessian(scalar,theta.detach())
    H0=torch.autograd.functional.hessian(lambda t:scalar(t,0.),theta.detach())
    def manual_flat(t):
        g=manual_gradients(X,y,t[:6].reshape(3,2),t[6:].reshape(2,3),.1)
        return torch.cat((g['W1'].flatten(),g['W2'].flatten()))
    eps=1e-5;finite=(manual_flat(theta.detach()+eps*direction)-manual_flat(theta.detach()-eps*direction))/(2*eps)
    return {'parameters':12,'hessian':H.tolist(),'hessian_shape':list(H.shape),'hessian_bytes':H.numel()*H.element_size(),
            'direction':direction.tolist(),'hvp':hvp.tolist(),'hvp_output_bytes':hvp.numel()*hvp.element_size(),
            'hvp_vs_dense_max_error':(hvp-H@direction).abs().max().item(),
            'hvp_central_difference_error':(hvp-finite).abs().max().item(),
            'hessian_symmetry_error':(H-H.T).abs().max().item(),
            'penalty_hessian_identity_error':(H-H0-.1*torch.eye(12,dtype=DTYPE)).abs().max().item(),
            'eigenvalue_min':torch.linalg.eigvalsh(H).min().item()}


def gradient_shape_demo():
    X=torch.arange(1,7,dtype=DTYPE).reshape(2,3).requires_grad_()
    scalar=X.square().sum()
    gradient=torch.autograd.grad(scalar,X)[0]
    vector=torch.tensor([[1.,0.,2.],[-1.,.5,1.]],dtype=DTYPE)
    vjp=torch.autograd.grad(X.square(),X,grad_outputs=vector)[0]
    jacobian=torch.autograd.functional.jacobian(lambda value:value.square(),X.detach())
    return {'input_shape':list(X.shape),'scalar_gradient_shape':list(gradient.shape),
            'scalar_gradient':gradient.tolist(),'tensor_output_shape':list(X.shape),
            'full_jacobian_shape':list(jacobian.shape),'vjp_shape':list(vjp.shape),
            'vjp_error':(vjp-2*X.detach()*vector).abs().max().item()}


def accumulated_gradient_check():
    generator=torch.Generator().manual_seed(4705)
    X=torch.randn(17,2,generator=generator,dtype=DTYPE);y=torch.arange(17,dtype=torch.long)%2
    _,_,first,second,_=fixture()
    full=[first.clone().requires_grad_(),second.clone().requires_grad_()]
    micro=[first.clone().requires_grad_(),second.clone().requires_grad_()]
    forward_values(X,y,*full,.1)['J'].backward()
    sizes=[]
    for start in range(0,17,5):
        xb,yb=X[start:start+5],y[start:start+5];sizes.append(len(yb))
        values=forward_values(xb,yb,*micro,0.)
        (values['L']*len(yb)/17).backward()
    (.1*(micro[0].square().sum()+micro[1].square().sum())/2).backward()
    error=max((a.grad-b.grad).abs().max().item() for a,b in zip(full,micro))
    torch.optim.SGD(full,lr=.1,foreach=False).step()
    torch.optim.SGD(micro,lr=.1,foreach=False).step()
    return {'samples':17,'microbatch_sizes':sizes,'optimizer_steps':1,
            'max_gradient_error':error,
            'max_parameter_error_after_step':max((a-b).abs().max().item() for a,b in zip(full,micro)),
            'penalty_added_once':True}


def split_device_demo(skip=False):
    count=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if skip:return {'status':'skipped_by_argument','available_gpus':count,'actual_multi_gpu':False}
    d0,d1=('cuda:0','cuda:1') if count>=2 else (('cpu','cuda:0') if count else ('cpu','cpu'))
    X,y,W1,W2,_=fixture()
    ref=[W1.clone().requires_grad_(),W2.clone().requires_grad_()]
    target=forward_values(X,y,*ref,.1)['J'];target.backward()
    a=W1.to(d0).detach().requires_grad_();b=W2.to(d1).detach().requires_grad_()
    h=torch.relu(X.to(d0)@a.T)
    o=h.to(d1)@b.T  # 设备复制保留图连接；这里不能detach。
    loss=F.cross_entropy(o,y.to(d1))+.1*(a.square().sum().to(d1)+b.square().sum())/2
    loss.backward()
    return {'status':'passed','available_gpus':count,'stage_devices':[d0,d1],'actual_multi_gpu':count>=2,
            'loss_error':abs(loss.item()-target.item()),
            'max_gradient_error':max((a.grad.cpu()-ref[0].grad).abs().max().item(),(b.grad.cpu()-ref[1].grad).abs().max().item())}


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def draw_graph(path,bias=False):
    fig,ax=plt.subplots(figsize=(11,4.2));ax.set(xlim=(-.8,10.6),ylim=(-1.6,3));ax.axis('off')
    teal,orange='#147E83','#D47F37'
    locations={'X':(0,1),'W1':(1.9,-.4),'Z':(1.9,1),'H':(3.8,1),
               'W2':(5.7,-.4),'O':(5.7,1),'y':(7.6,2.3),'L':(7.6,1),'S':(7.6,-.4),'J':(9.5,1)}
    labels={'X':'X (n,d)','W1':'W1 (h,d)','Z':'Z (n,h)','H':'H (n,h)','W2':'W2 (q,h)',
            'O':'O (n,q)','y':'标签 y','L':'数据损失 L','S':'L2惩罚 S','J':'目标 J=L+S'}
    for name,(x,y) in locations.items():
        ax.add_patch(FancyBboxPatch((x-.61,y-.27),1.22,.54,boxstyle='round,pad=.03',facecolor='#F1F7F8',edgecolor=teal))
        ax.text(x,y,labels[name],ha='center',va='center',fontsize=10)
    def edge(a,b,label='',color=teal,rad=0):
        x1,y1=locations[a];x2,y2=locations[b]
        dx,dy=x2-x1,y2-y1
        fraction=1/max(abs(dx)/.67,abs(dy)/.33)
        ax.add_patch(FancyArrowPatch((x1+fraction*dx,y1+fraction*dy),(x2-fraction*dx,y2-fraction*dy),
                                    arrowstyle='->',mutation_scale=12,shrinkA=0,shrinkB=0,color=color))
        if label:
            tx,ty=((x1+x2)/2,(y1+y2)/2+.12) if abs(dx)>.1 else (x1+.64,(y1+y2)/2)
            ax.text(tx,ty,label,ha='center',fontsize=9,color=color)
    edge('X','Z','@W1.T');edge('W1','Z');edge('Z','H','ReLU');edge('H','O','@W2.T')
    edge('W2','O');edge('O','L','CE');edge('y','L');edge('L','J',color=orange);edge('S','J',color=orange)
    # 正则分支走主链下方，避免穿过其他变量节点。
    ax.plot([1.9,1.9,7.6],[-.73,-1.05,-1.05],color=orange,lw=1.2)
    ax.add_patch(FancyArrowPatch((7.6,-1.05),(7.6,-.73),arrowstyle='->',mutation_scale=12,color=orange))
    edge('W2','S',color=orange)
    if bias:
        locations['b1']=(1.9,2.3)
        ax.add_patch(FancyBboxPatch((1.29,2.03),1.22,.54,boxstyle='round,pad=.03',facecolor='#FFF4E8',edgecolor=orange))
        ax.text(1.9,2.3,'偏置 b1 (h,)',ha='center',va='center',fontsize=10)
        edge('b1','Z','广播相加',orange)
    ax.text(4.7,-1.43,'同一轮的前向是有向无环图；反向沿相反依赖传梯度，参数处分支贡献相加。',ha='center',fontsize=10)
    ax.set_title('加入隐藏偏置：Z=X@W1.T+b1，偏置不进正则项' if bias else '无偏置模型：前向主链 + 权重惩罚分支',fontsize=12)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def draw_figures(example,history,memory,hessian):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'font.size':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    draw_graph(FIGURES/'4_7_computation_graph.png');draw_graph(FIGURES/'4_7_bias_graph.png',True)
    fig,ax=plt.subplots(figsize=(9,3.5));g=example['gradients']
    for offset,key,label,color in ((-.24,'data_W1','数据路径','#147E83'),(0,'regularization_W1','惩罚路径','#D47F37'),(.24,'W1','相加后的总梯度','#173448')):
        values=[v for row in g[key] for v in row];ax.bar([i+offset for i in range(6)],values,width=.23,label=label,color=color)
    ax.set_xticks(range(6),['W1[0,0]','W1[0,1]','W1[1,0]','W1[1,1]','W1[2,0]','W1[2,1]'],rotation=12)
    ax.axhline(0,color='#8094A0',lw=.7);ax.set(title='负预激活的隐藏单元：数据梯度为0，惩罚梯度仍可非零',ylabel='对J的梯度')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_7_gradient_paths.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for key,label,color in (('train_ce','训练CE','#147E83'),('validation_ce','验证CE','#D47F37'),('objective','训练J','#173448')):
        axes[0].plot([r['epoch'] for r in history],[r[key] for r in history],label=label,color=color)
    for key,label,color in (('train_accuracy','训练','#147E83'),('validation_accuracy','验证','#D47F37')):
        axes[1].plot([r['epoch'] for r in history],[r[key] for r in history],label=label,color=color)
    axes[0].set(xlabel='全批量更新次数',ylabel='平均CE / 正则目标',title='人工XOR：2→8→2、无偏置')
    axes[1].set(xlabel='全批量更新次数',ylabel='准确率',ylim=(0,1.05),title='手写、autograd与Linear逐步对齐')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(FIGURES/'4_7_training.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    axes[0].bar(['模型参数','参数梯度'],[memory[0]['parameter_bytes']/1024**2,memory[0]['parameter_gradient_bytes']/1024**2],color=['#147E83','#D47F37'])
    axes[0].set(ylabel='MiB',title='参数载荷：与当前批量大小无关')
    axes[1].plot([r['batch_size'] for r in memory],[r['saved_nonparameter_storage_bytes']/1024**2 for r in memory],'o-',color='#147E83',label='反传额外保存（去重）')
    axes[1].plot([r['batch_size'] for r in memory],[0 for r in memory],'--',color='#D47F37',label='no_grad的反传保存量')
    axes[1].set(xlabel='批量大小 n',ylabel='MiB',title='不含参数/梯度的保存storage');axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('784→256→10、无偏置、float32：按组成分开计数',fontsize=12)
    fig.text(.5,.015,'不是进程RAM或显存峰值；no_grad保存量为0不代表预测不需要内存。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(FIGURES/'4_7_memory.png',dpi=170);plt.close(fig)
    H=torch.tensor(hessian['hessian']);limit=H.abs().max().item()
    fig,ax=plt.subplots(figsize=(6,4.5));im=ax.imshow(H,cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax.axhline(5.5,color='black',lw=.7);ax.axvline(5.5,color='black',lw=.7)
    ax.set(xlabel='前6个W1参数 / 后6个W2参数',ylabel='参数索引',title='12参数Hessian：含跨层混合二阶导数')
    fig.colorbar(im,ax=ax,label='二阶导数');fig.tight_layout();fig.savefig(FIGURES/'4_7_hessian.png',dpi=170);plt.close(fig)


def main(skip_device_demo=False):
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    example=single_example();biased=single_example(True)
    splits=make_data();training,history=train_three_paths(splits)
    memory,records=[],{}
    for count in (1,8,32,256):
        row,items=memory_probe(count);memory.append(row);records[str(count)]=items
    hessian=second_order();shapes=gradient_shape_demo();accumulation=accumulated_gradient_check()
    devices=split_device_demo(skip_device_demo)
    draw_figures(example,history,memory,hessian)
    metrics={'date':'2026-10-27','purpose':'teacher demonstration, not learner completion',
             'torch':torch.__version__,'math_device':'cpu','dtype':'float64','download':False,
             'single_example':example,'hidden_bias_extension':biased,'training':training,
             'memory':memory,'memory_saved_tensor_records':records,
             'memory_scope':'logical tensor/storage payload accounting, not actual allocator/process/GPU peak',
             'second_order':hessian,'gradient_shapes':shapes,'gradient_accumulation':accumulation,'device_split':devices}
    with (REPORT/'4_7_metrics.json').open('w',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(REPORT/'4_7_epochs.csv',history);write_csv(REPORT/'4_7_memory.csv',memory)
    write_csv(REPORT/'4_7_data.csv',[{'split':name,'index':i,'x1':X[i,0].item(),'x2':X[i,1].item(),'label':int(y[i])}
              for name,(X,y) in splits.items() for i in range(len(y))])
    print('单样本前向:',example['forward']);print('单样本梯度:',example['gradients'])
    print('手写/autograd误差:',example['max_gradient_errors'],'| 一步后J:',example['after_one_step_J'])
    print('训练:',training);print('内存:',memory)
    print('二阶HVP/稠密误差:',hessian['hvp_vs_dense_max_error'],'| 差分误差:',hessian['hvp_central_difference_error'])
    print('梯度形状:',shapes);print('梯度累积:',accumulation);print('分阶段设备:',devices)
    print('教师准备不记为用户掌握；下一节4.8数值稳定性和模型初始化。')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-device-demo',action='store_true')
    main(parser.parse_args().skip_device_demo)
