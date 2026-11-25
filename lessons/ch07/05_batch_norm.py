"""7.5 批量规范化：从零/框架LeNet、模式与状态、学习率/位置/暂退对照。

默认运行预定四个完整十轮方案，第1/5/10轮测试仅画固定曲线、不调参。
--inspect只查结构，--experiments十五组验证子集不读测试，--guided中文导读。
--record才保存报告和本地模型。custom不读测试，孤立尾样本并入前批、不丢数据。
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

PREVIOUS=runpy.run_path(str(ROOT/'lessons/ch07/04_googlenet.py'))
COMMON=PREVIOUS['COMMON']
write_json,write_csv,plot_setup=(PREVIOUS[k] for k in ('write_json','write_csv','plot_setup'))
REPORT=ROOT/'reports/ch07'
BASE=dict(implementation='manual',placement='all',affine=True,bias=True,dropout=0.,
          lr=1.,batch_size=256,momentum=.1,eps=1e-5)


def batch_norm(X,weight,bias,running_mean,running_var,training,momentum=.1,eps=1e-5):
    """训练前向用有偏方差；移动方差用无偏修正，与本机PyTorch默认口径对齐。"""
    if X.ndim not in (2,4):raise ValueError('本节仅处理(N,C)或(N,C,H,W)')
    axes=(0,) if X.ndim==2 else (0,2,3)
    shape=(1,-1) if X.ndim==2 else (1,-1,1,1)
    if training:
        m=X.numel()//X.shape[1]
        if m<2:raise ValueError('训练时每通道统计元素须大于1')
        mean=X.mean(axes)
        centered=X-mean.reshape(shape)
        var=centered.square().mean(axes)
        with torch.no_grad():
            new_mean=(1-momentum)*running_mean+momentum*mean.detach()
            new_var=(1-momentum)*running_var+momentum*var.detach()*m/(m-1)
    else:
        mean,var=running_mean,running_var
        centered=X-mean.reshape(shape)
        new_mean,new_var=running_mean,running_var
    out=centered*torch.rsqrt(var.reshape(shape)+eps)
    if weight is not None:out=out*weight.reshape(shape)+bias.reshape(shape)
    return out,new_mean,new_var


class ManualBatchNorm(nn.Module):
    def __init__(self,features,dims,eps=1e-5,momentum=.1,affine=True,device='cpu'):
        super().__init__()
        if features<1 or dims not in (2,4):raise ValueError('检查通道数与维数')
        self.features,self.dims,self.eps,self.momentum=features,dims,eps,momentum
        if affine:
            self.weight=nn.Parameter(torch.ones(features,device=device))
            self.bias=nn.Parameter(torch.zeros(features,device=device))
        else:
            self.register_parameter('weight',None);self.register_parameter('bias',None)
        self.register_buffer('running_mean',torch.zeros(features,device=device))
        self.register_buffer('running_var',torch.ones(features,device=device))
        self.register_buffer('num_batches_tracked',torch.tensor(0,dtype=torch.long,device=device))

    def forward(self,X):
        if X.ndim!=self.dims or X.shape[1]!=self.features:raise ValueError('输入维数或通道不匹配')
        Y,mean,var=batch_norm(X,self.weight,self.bias,self.running_mean,self.running_var,self.training,self.momentum,self.eps)
        if self.training:
            with torch.no_grad():
                self.running_mean.copy_(mean);self.running_var.copy_(var);self.num_batches_tracked.add_(1)
        return Y


def textbook_batch_norm(X,weight,bias,running_mean,running_var,eps=1e-5,retention=.9):
    """仅用于展示教材原函数的模式/移动方差差异，不进入正式训练。"""
    axes=(0,) if X.ndim==2 else (0,2,3);shape=(1,-1) if X.ndim==2 else (1,-1,1,1)
    if torch.is_grad_enabled():
        mean=X.mean(axes);var=(X-mean.reshape(shape)).square().mean(axes)
        new_mean=retention*running_mean+(1-retention)*mean.detach()
        new_var=retention*running_var+(1-retention)*var.detach()
    else:mean,var=running_mean,running_var;new_mean,new_var=running_mean,running_var
    return (X-mean.reshape(shape))*torch.rsqrt(var.reshape(shape)+eps)*weight.reshape(shape)+bias.reshape(shape),new_mean,new_var


def checked(config=None):
    cfg=dict(BASE,**(config or {}))
    if cfg['implementation'] not in ('manual','native','none') or cfg['placement'] not in ('all','conv','dense','first'):
        raise ValueError('检查实现种类与放置位置')
    if cfg['batch_size']<2 or cfg['lr']<=0 or not 0<=cfg['dropout']<1:raise ValueError('本训练协议batch至少2，学习率为正，暂退概率在[0,1)')
    if cfg['eps']<=0 or not 0<cfg['momentum']<=1:raise ValueError('eps为正；momentum为(0,1]的新统计更新权重')
    return cfg


def norm_enabled(cfg,index):
    return cfg['implementation']!='none' and (cfg['placement']=='all' or cfg['placement']=='conv' and index<2
        or cfg['placement']=='dense' and index>=2 or cfg['placement']=='first' and index==0)


def build_net(config=None,device='cpu',seed=7500):
    cfg=checked(config)
    if torch.device(device).type not in ('cpu','meta'):raise ValueError('先CPU初始化，再to目标设备')
    def norm(features,dims,index):
        if not norm_enabled(cfg,index):return nn.Identity()
        if cfg['implementation']=='manual':return ManualBatchNorm(features,dims,cfg['eps'],cfg['momentum'],cfg['affine'],device)
        cls=nn.BatchNorm1d if dims==2 else nn.BatchNorm2d
        return cls(features,eps=cfg['eps'],momentum=cfg['momentum'],affine=cfg['affine'],device=device)
    bias=lambda index:cfg['bias'] or not norm_enabled(cfg,index)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        net=nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(1,6,5,bias=bias(0),device=device)),('bn1',norm(6,4,0)),('act1',nn.Sigmoid()),('pool1',nn.AvgPool2d(2,2)),
            ('conv2',nn.Conv2d(6,16,5,bias=bias(1),device=device)),('bn2',norm(16,4,1)),('act2',nn.Sigmoid()),('pool2',nn.AvgPool2d(2,2)),
            ('flatten',nn.Flatten()),('fc1',nn.Linear(256,120,bias=bias(2),device=device)),('bn3',norm(120,2,2)),('act3',nn.Sigmoid()),
            ('drop1',nn.Dropout(cfg['dropout'])),('fc2',nn.Linear(120,84,bias=bias(3),device=device)),('bn4',norm(84,2,3)),('act4',nn.Sigmoid()),
            ('drop2',nn.Dropout(cfg['dropout'])),('out',nn.Linear(84,10,device=device))]))
        if torch.device(device).type=='cpu':
            for j,m in enumerate(m for m in net.modules() if isinstance(m,(nn.Conv2d,nn.Linear))):
                torch.default_generator.manual_seed(seed+j);nn.init.xavier_uniform_(m.weight,gain=1.)
                if m.bias is not None:nn.init.zeros_(m.bias)
    return net


def state_hash(net,weights_only=False):
    digest=hashlib.sha256()
    pairs=[(n+'.weight',m.weight) for n,m in net.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))] if weights_only else net.state_dict().items()
    for name,p in pairs:
        digest.update(name.encode());digest.update(str(tuple(p.shape)).encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def inspect_net(config=None):
    cfg=checked(config);net=build_net(cfg,device='meta').eval();X=torch.zeros(2,1,28,28,device='meta');rows=[]
    for name,m in net.named_children():
        X=m(X);rows.append(dict(name=name,layer=type(m).__name__,shape=list(X.shape),parameters=sum(p.numel() for p in m.parameters())))
    return dict(config=cfg,layers=rows,parameters=sum(p.numel() for p in net.parameters()),
        normalization_parameters=sum(p.numel() for n,m in net.named_children() if n.startswith('bn') for p in m.parameters()),
        buffers=[dict(name=n,shape=list(b.shape),dtype=str(b.dtype),elements=b.numel()) for n,b in net.named_buffers()],
        note='本节首Conv无padding：28→24→12→8→4，Flatten256；不同于6.6首层padding2')


def train_batches(loader):
    """只把孤立尾样本并入前批；不丢数据，也不复制样本。batch_size至少2。"""
    pending=None
    for batch in loader:
        if pending is None:pending=batch;continue
        if len(batch[1])==1:
            yield tuple(torch.cat([a,b],0) for a,b in zip(pending,batch));pending=None
        else:yield pending;pending=batch
    if pending is not None:
        if len(pending[1])<2:raise ValueError('训练集至少需要2条；单元素批无法做当前BN1d统计')
        yield pending


def train_epoch(net,loader,optimizer,device):
    net.train();total,correct,count,updates,last=0.,0,0,0,0
    for X,y in train_batches(loader):
        X,y=X.to(device),y.to(device);optimizer.zero_grad(set_to_none=True);logits=net(X)
        loss=F.cross_entropy(logits,y)
        if not torch.isfinite(loss):return dict(status='stopped_nonfinite',updates=updates,samples=count)
        loss.backward();optimizer.step();total+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item()
        count+=len(y);updates+=1;last=len(y)
    return dict(status='complete',online_train_ce=total/count,online_train_accuracy=correct/count,updates=updates,samples=count,last_batch_size=last)


def evaluate(net,dataset,batch_size,device):
    return COMMON['evaluate'](net,COMMON['make_loader'](dataset,batch_size=batch_size),device)


def affine_rows(net):
    rows=[]
    for name,m in net.named_children():
        if isinstance(m,(ManualBatchNorm,nn.BatchNorm1d,nn.BatchNorm2d)):
            for i in range(len(m.running_mean)):
                rows.append(dict(layer=name,channel=i,gamma=m.weight[i].item() if m.weight is not None else 1.,
                    beta=m.bias[i].item() if m.bias is not None else 0.,running_mean=m.running_mean[i].item(),running_var=m.running_var[i].item(),
                    num_batches_tracked=m.num_batches_tracked.item()))
    return rows


def fit(cfg,splits,epochs,device,name,test_epochs=(),record=False,prefix=None):
    net=build_net(cfg).to(device);optimizer=torch.optim.SGD(net.parameters(),lr=cfg['lr'])
    loader=COMMON['make_loader'](splits['train'],shuffle=True,batch_size=cfg['batch_size'])
    torch.manual_seed(7503)
    if device.type=='cuda':torch.cuda.manual_seed_all(7503);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
    result=dict(config=cfg,model_seed_base=7500,initialization='CPU per-Conv/Linear seed7500+j, Xavier gain1, zero bias; BN gamma1/beta0',
        dropout_seed=7503,loader_seed=3603,initial_state_sha256=state_hash(net),initial_weights_sha256=state_hash(net,True),
        parameters=sum(p.numel() for p in net.parameters()),runtime=COMMON['runtime_metadata'](device),epochs=epochs,
        train_samples=len(splits['train']),validation_samples=len(splits['validation']),test_epochs=list(test_epochs),history=[],status='complete')
    started=time.perf_counter()
    for epoch in range(epochs+1):
        online={} if epoch==0 else train_epoch(net,loader,optimizer,device)
        if online.get('status')=='stopped_nonfinite':result['status']='stopped_nonfinite';result['stopped_epoch']=epoch;break
        val=evaluate(net,splits['validation'],cfg['batch_size'],device)
        row=dict(epoch=epoch,**online,**{'validation_'+k:v for k,v in val.items()})
        if epoch==epochs:row.update({'train_eval_'+k:v for k,v in evaluate(net,splits['train'],cfg['batch_size'],device).items()})
        if epoch in test_epochs:row.update({'test_'+k:v for k,v in evaluate(net,splits['test'],cfg['batch_size'],device).items()})
        result['history'].append(row)
        print('{} epoch {:2d}: online CE {}, val CE {:.6f}, acc {:.2%}'.format(name,epoch,
            '-' if not online else '{:.6f}'.format(online['online_train_ce']),val['ce'],val['accuracy']),flush=True)
    result.update(updates=sum(r.get('updates',0) for r in result['history']),last_batch_size=result['history'][-1].get('last_batch_size'),
        singleton_tail_merged=len(splits['train'])%cfg['batch_size']==1,final_state_sha256=state_hash(net),affine=affine_rows(net),
        test_evaluated=bool(test_epochs),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    if record:
        folder=ROOT/'artifacts/ch07';folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(config=cfg,epoch=result['history'][-1]['epoch'],model={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=optimizer.state_dict()),folder/(prefix+'.pt'))
    if device.type=='cuda':torch.cuda.synchronize()
    result['seconds_including_evaluation_and_save']=time.perf_counter()-started
    return result,net


def formal_configs():
    return {'manual':dict(BASE),'native':dict(BASE,implementation='native'),
            'plain01':dict(BASE,implementation='none',lr=.1),'plain1':dict(BASE,implementation='none')}


def candidates():
    base=dict(BASE,implementation='native',batch_size=128)
    return {'A':base,'B':dict(base,lr=.1),'C':dict(base,lr=3.),'D':dict(base,lr=10.),
        'E':dict(base,implementation='none',lr=.1),'F':dict(base,implementation='none'),
        'G':dict(base,implementation='none',lr=3.),'H':dict(base,implementation='none',lr=10.),
        'I':dict(base,placement='conv'),'J':dict(base,placement='dense'),'K':dict(base,placement='first'),
        'L':dict(base,affine=False),'M':dict(base,bias=False),
        'N':dict(base,implementation='none',dropout=.5),'O':dict(base,dropout=.5)}


def plot_formal(output):
    plt=plot_setup();fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
    for name,result in output['results'].items():
        h=result['history'];axes[0].plot([r['epoch'] for r in h[1:]],[r['online_train_accuracy'] for r in h[1:]],label=name)
        tests=[r for r in h if 'test_accuracy' in r];axes[1].plot([r['epoch'] for r in tests],[r['test_accuracy'] for r in tests],'o-',label=name)
    for ax in axes:ax.set(xlabel='epoch',ylim=(0,1));ax.legend(fontsize=9);ax.grid(alpha=.2)
    axes[0].set_ylabel('在线训练准确率');axes[1].set_ylabel('预定第1/5/10轮测试准确率')
    fig.savefig(REPORT/'figures/7_5_training.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
    for name,result in output['results'].items():
        if name not in ('manual','native'):continue
        rows=[r for r in result['affine'] if r['layer']=='bn1']
        axes[0].plot(range(6),[r['gamma'] for r in rows],'o-',label=name)
        axes[1].plot(range(6),[r['beta'] for r in rows],'o-',label=name)
    axes[0].set(xlabel='通道',ylabel='gamma');axes[1].set(xlabel='通道',ylabel='beta')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(REPORT/'figures/7_5_affine.png',dpi=180);plt.close(fig)


def run_formal(device,record):
    splits,hashes=COMMON['load_data'](include_test=True);results={};rows=[]
    for name,cfg in formal_configs().items():
        result,net=fit(cfg,splits,10,device,name,(1,5,10),record,'7_5_'+name)
        results[name]=result;rows.extend(dict(model=name,**r) for r in result['history']);del net
        if record:write_json('7_5_'+name+'_metrics.json',result);write_csv('7_5_'+name+'_epochs.csv',result['history'])
        if device.type=='cuda':torch.cuda.empty_cache()
    assert len({r['initial_weights_sha256'] for r in results.values()})==1
    output=dict(results=results,split_hashes=hashes,split_seed=3601,test_evaluated=True,test_epochs=[1,5,10],
        protocol='四个方案及测试轮次预先固定，仅画曲线；不使用测试选择学习率、轮次或变体；其余调参仅验证集')
    if record:
        write_json('7_5_metrics.json',output);write_csv('7_5_epochs.csv',rows)
        write_csv('7_5_affine.csv',[dict(model=k,**r) for k,v in results.items() for r in v['affine']]);plot_formal(output)
    return output


def run_experiments(device,record):
    splits,hashes=COMMON['load_data'](include_test=False)
    splits=dict(train=COMMON['balanced_subset'](splits['train'],2000,7501),validation=COMMON['balanced_subset'](splits['validation'],1000,7502))
    results,rows={},[]
    for name,cfg in candidates().items():
        result,net=fit(cfg,splits,20,device,name);results[name]=result
        rows.extend(dict(candidate=name,**r) for r in result['history']);del net
        if device.type=='cuda':torch.cuda.empty_cache()
    assert len({r['initial_weights_sha256'] for r in results.values()})==1
    eligible={k:v for k,v in results.items() if v['status']=='complete'}
    selected=min(eligible,key=lambda k:eligible[k]['history'][-1]['validation_ce'])
    output=dict(results=results,split_hashes=hashes,train_subset_seed=7501,validation_subset_seed=7502,test_evaluated=False,
        selected_by_final_validation_ce=selected,protocol='预定十五候选各20轮320更新，仅比较末轮验证CE；非有限会停止并记录，不读测试')
    if record:
        write_json('7_5_experiments.json',output);write_csv('7_5_experiments_epochs.csv',rows)
        plt=plot_setup();fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
        for names,label,ax in [('ABCD','BN',axes[0]),('EFGH','无BN',axes[1])]:
            for key in names:
                h=results[key]['history'];ax.plot([r['epoch'] for r in h],[r['validation_ce'] for r in h],label=key+':lr'+str(results[key]['config']['lr']))
            ax.set(xlabel='epoch',ylabel=label+'验证CE');ax.legend(fontsize=9)
        fig.savefig(REPORT/'figures/7_5_learning_rates.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8.5,3.2),constrained_layout=True)
        keys=['A','F','I','J','K','L','M','N','O'];ax.bar(keys,[results[k]['history'][-1]['validation_accuracy'] for k in keys],color='#147e83')
        ax.set(xlabel='A全BN F无BN I卷积 J全连接 K首层 L无仿射 M去偏置 N暂退 O两者',ylabel='末轮验证准确率',ylim=(0,1))
        fig.savefig(REPORT/'figures/7_5_ablations.png',dpi=180);plt.close(fig)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--inspect',action='store_true');modes.add_argument('--experiments',action='store_true')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--train-size',type=int);parser.add_argument('--validation-size',type=int)
    parser.add_argument('--guided',action='store_true');parser.add_argument('--record',action='store_true');args=parser.parse_args()
    if args.epochs<1 or args.batch_size<2:parser.error('轮数为正，当前BN训练batch至少2')
    if args.train_size is not None and args.train_size<2:parser.error('训练集至少2条')
    if args.validation_size is not None and args.validation_size<1:parser.error('验证集至少1条')
    custom=args.epochs!=10 or args.batch_size!=256 or args.train_size is not None or args.validation_size is not None
    if (args.inspect or args.experiments) and custom:parser.error('固定模式不接受额外训练参数')
    device=COMMON['configure'](args.device)
    if args.guided:print('7.5：训练用批统计，eval用移动状态；模式与梯度开关分开。孤立尾样本合并，不丢数据。',flush=True)
    if args.inspect:
        result={k:inspect_net(v) for k,v in formal_configs().items()}
        if args.record:write_json('7_5_architectures.json',result)
    elif args.experiments:result=run_experiments(device,args.record)
    elif custom:
        splits,hashes=COMMON['load_data'](include_test=False)
        for n,size,seed in (('train',args.train_size,7501),('validation',args.validation_size,7502)):
            if size is not None:splits[n]=COMMON['balanced_subset'](splits[n],size,seed)
        result,_=fit(dict(BASE,batch_size=args.batch_size),splits,args.epochs,device,'custom',(),args.record,'7_5_custom')
        result.update(custom=True,split_hashes=hashes)
        if args.record:write_json('7_5_custom_metrics.json',result);write_csv('7_5_custom_epochs.csv',result['history'])
    else:result=run_formal(device,args.record)
    if args.guided:
        if args.inspect:
            for k,r in result.items():print(k,'参数',r['parameters'],'BN参数',r['normalization_parameters'])
        elif 'results' in result:
            for k,r in result['results'].items():print(k,r['status'],'末轮验证',r['history'][-1]['validation_accuracy'])
        print('教材原函数的模式/移动方差差别另行演示；八处练习不代填。')
    else:print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
