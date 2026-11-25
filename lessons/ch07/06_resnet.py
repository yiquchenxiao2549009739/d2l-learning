"""7.6 ResNet：教材ResNet18、基本/瓶颈/预激活块、深度与残差受控对照。

默认96输入、完整64起始通道、SGD .05、batch256、十轮，末轮才测试。
--inspect结构；--experiments固定八候选只验证；--benchmark GPU随机短测。
--record覆盖本节相应报告和本地权重；孤立尾样本合并，不丢数据。
"""
import argparse
from collections import OrderedDict
from contextlib import nullcontext, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

PREVIOUS=runpy.run_path(str(ROOT/'lessons/ch07/05_batch_norm.py'))
COMMON=PREVIOUS['COMMON']
UTIL=PREVIOUS['PREVIOUS']
prepare,evaluate_base=(UTIL[k] for k in ('prepare','evaluate'))
write_json,write_csv,plot_setup=(UTIL[k] for k in ('write_json','write_csv','plot_setup'))
REPORT=ROOT/'reports/ch07'
LAYOUTS={18:('basic',[2,2,2,2]),34:('basic',[3,4,6,3]),50:('bottleneck',[3,4,6,3]),
         101:('bottleneck',[3,4,23,3]),152:('bottleneck',[3,8,36,3])}
BASE=dict(depth=18,width=64,size=96,residual=True,preactivation=False,zero_final_bn=False,lr=.05,batch_size=256)


class Residual(nn.Module):
    expansion=1

    def __init__(self,ci,width,stride=1,residual=True,device='cpu'):
        super().__init__()
        self.residual=residual
        self.conv1=nn.Conv2d(ci,width,3,stride,padding=1,bias=True,device=device)
        self.bn1=nn.BatchNorm2d(width,device=device)
        self.conv2=nn.Conv2d(width,width,3,padding=1,bias=True,device=device)
        self.bn2=nn.BatchNorm2d(width,device=device)
        self.shortcut=nn.Identity()
        if residual and (ci!=width or stride!=1):
            self.shortcut=nn.Conv2d(ci,width,1,stride,bias=True,device=device)

    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.residual:
            skip=self.shortcut(X)
            if Y.shape!=skip.shape:raise ValueError('相加需要全部维数相同，禁止广播')
            Y=Y+skip
        return F.relu(Y)


class Bottleneck(nn.Module):
    """1×1降维→3×3→1×1扩张，expansion4；降采样stride放首个1×1（v1）。"""
    expansion=4
    def __init__(self,ci,width,stride=1,residual=True,device='cpu'):
        super().__init__();co=4*width;self.residual=residual
        self.conv1=nn.Conv2d(ci,width,1,stride,bias=True,device=device);self.bn1=nn.BatchNorm2d(width,device=device)
        self.conv2=nn.Conv2d(width,width,3,padding=1,bias=True,device=device);self.bn2=nn.BatchNorm2d(width,device=device)
        self.conv3=nn.Conv2d(width,co,1,bias=True,device=device);self.bn3=nn.BatchNorm2d(co,device=device)
        self.shortcut=nn.Conv2d(ci,co,1,stride,bias=True,device=device) if residual and (ci!=co or stride!=1) else nn.Identity()

    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)));Y=F.relu(self.bn2(self.conv2(Y)));Y=self.bn3(self.conv3(Y))
        if self.residual:
            skip=self.shortcut(X)
            if Y.shape!=skip.shape:raise ValueError('瓶颈相加形状不一致')
            Y=Y+skip
        return F.relu(Y)


class PreActResidual(nn.Module):
    """受控基本块变体：BN-ReLU-Conv两次；恒等支路取原X，投影支路取预激活X，无加法后ReLU。"""
    expansion=1
    def __init__(self,ci,width,stride=1,residual=True,device='cpu'):
        super().__init__();self.residual=residual
        self.bn1=nn.BatchNorm2d(ci,device=device);self.conv1=nn.Conv2d(ci,width,3,stride,1,bias=True,device=device)
        self.bn2=nn.BatchNorm2d(width,device=device);self.conv2=nn.Conv2d(width,width,3,padding=1,bias=True,device=device)
        self.shortcut=nn.Conv2d(ci,width,1,stride,bias=True,device=device) if residual and (ci!=width or stride!=1) else nn.Identity()

    def forward(self,X):
        activated=F.relu(self.bn1(X));Y=self.conv1(activated);Y=self.conv2(F.relu(self.bn2(Y)))
        if self.residual:
            skip=X if isinstance(self.shortcut,nn.Identity) else self.shortcut(activated)
            if Y.shape!=skip.shape:raise ValueError('预激活相加形状不一致')
            Y=Y+skip
        return Y


def checked(config=None):
    cfg=dict(BASE,**(config or {}))
    if cfg['depth'] not in LAYOUTS or cfg['width']<1 or cfg['size']<1:raise ValueError('检查深度、宽度和尺寸')
    if cfg['preactivation'] and LAYOUTS[cfg['depth']][0]!='basic':raise ValueError('当前预激活变体仅支持基本块18/34')
    if cfg['preactivation'] and cfg['zero_final_bn']:raise ValueError('末BN置零仅用于后激活块对照')
    return cfg


def build_net(config=None,device='cpu',seed=7600):
    cfg=checked(config);kind,counts=LAYOUTS[cfg['depth']]
    cls=PreActResidual if cfg['preactivation'] else Residual if kind=='basic' else Bottleneck
    w=cfg['width'];layers=[('stem',nn.Sequential(OrderedDict([
        ('conv',nn.Conv2d(1,w,7,2,3,bias=True,device=device)),('bn',nn.BatchNorm2d(w,device=device)),
        ('relu',nn.ReLU()),('pool',nn.MaxPool2d(3,2,1))])))]
    ci=w
    for stage,count in enumerate(counts):
        blocks=[]
        for j in range(count):
            stride=2 if stage>0 and j==0 else 1
            blocks.append(('block'+str(j+1),cls(ci,w*2**stage,stride,cfg['residual'],device)))
            ci=w*2**stage*cls.expansion
        layers.append(('stage'+str(stage+1),nn.Sequential(OrderedDict(blocks))))
    if cfg['preactivation']:layers.append(('final',nn.Sequential(nn.BatchNorm2d(ci,device=device),nn.ReLU())))
    layers.extend([('gap',nn.AdaptiveAvgPool2d(1)),('flatten',nn.Flatten()),('classifier',nn.Linear(ci,10,device=device))])
    net=nn.Sequential(OrderedDict(layers))
    if device!='meta':
        for name,m in net.named_modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                value=seed+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
                torch.manual_seed(value);nn.init.xavier_uniform_(m.weight,gain=1);nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
        if cfg['zero_final_bn']:
            for m in net.modules():
                if isinstance(m,Residual):nn.init.zeros_(m.bn2.weight)
                elif isinstance(m,Bottleneck):nn.init.zeros_(m.bn3.weight)
    return net


def state_hash(net,main_only=False):
    digest=hashlib.sha256()
    main_keys=set()
    if main_only:
        for name,module in net.named_modules():
            if isinstance(module,(nn.Conv2d,nn.Linear)) and not name.endswith('.shortcut'):
                main_keys.update(name+'.'+key for key in ('weight','bias'))
    for name,value in net.state_dict().items():
        if main_only and name not in main_keys:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def inspect_net(config=None):
    cfg=checked(config);net=build_net(cfg,device='meta').eval();weighted=[];outer=[];handles=[]
    def hook(name):
        def record(m,inputs,out):
            n=out.numel()//out.shape[0];fan=m.in_channels*m.kernel_size[0]*m.kernel_size[1] if isinstance(m,nn.Conv2d) else m.in_features
            weighted.append(dict(name=name,kind=type(m).__name__,shape=list(out.shape),parameters=sum(p.numel() for p in m.parameters()),weight_multiplications=n*fan))
        return record
    for name,m in net.named_modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):handles.append(m.register_forward_hook(hook(name)))
    X=torch.zeros(1,1,cfg['size'],cfg['size'],device='meta')
    with torch.no_grad():
        for name,m in net.named_children():X=m(X);outer.append(dict(name=name,shape=list(X.shape)))
    for h in handles:h.remove()
    return dict(config=cfg,outer=outer,weighted_layers=weighted,parameters=sum(p.numel() for p in net.parameters()),
        block_count=sum(LAYOUTS[cfg['depth']][1]),named_depth=cfg['depth'],conv_count=sum(isinstance(m,nn.Conv2d) for m in net.modules()),
        bn_count=sum(isinstance(m,nn.BatchNorm2d) for m in net.modules()),linear_count=1,
        projection_count=sum(isinstance(m.shortcut,nn.Conv2d) for m in net.modules() if isinstance(m,(Residual,Bottleneck,PreActResidual))),
        parameter_float32_bytes=sum(p.numel() for p in net.parameters())*4,weight_multiplications=sum(r['weight_multiplications'] for r in weighted),
        note='灰度十分类/全部Conv含bias/投影无BN，瓶颈stride放首1×1；meta逐层计数，非ImageNet官方参数数、非实测FLOPs')


def evaluate(net,data,cfg,device):return evaluate_base(net,data,cfg,device)


def train_epoch(net,loader,optimizer,cfg,device):
    net.train();loss_sum=0.;correct=count=updates=0;tail=None
    for X,y in PREVIOUS['train_batches'](loader):
        X,y=prepare(X,cfg,device),y.to(device);optimizer.zero_grad(set_to_none=True);logits=net(X);loss=F.cross_entropy(logits,y)
        if not torch.isfinite(loss):raise FloatingPointError('损失非有限，检查学习率和输入')
        loss.backward();optimizer.step();loss_sum+=loss.item()*len(y)
        correct+=(logits.detach().argmax(1)==y).sum().item();count+=len(y);updates+=1;tail=len(y)
    return dict(online_train_ce=loss_sum/count,online_train_accuracy=correct/count,updates=updates,samples=count,last_batch_size=tail)


def fit(cfg,splits,epochs,device,name,use_test=False,record=False,prefix=None):
    net=build_net(cfg).to(device);optimizer=torch.optim.SGD(net.parameters(),lr=cfg['lr'])
    loader=COMMON['make_loader'](splits['train'],shuffle=True,batch_size=cfg['batch_size'])
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
    result=dict(config=cfg,model_seed_base=7600,initialization='按层名SHA-256角色种子7600+hash，Xavier gain1、零bias；BN1/0',
        loader_seed=3603,initial_state_sha256=state_hash(net),initial_main_sha256=state_hash(net,True),
        parameters=sum(p.numel() for p in net.parameters()),runtime=COMMON['runtime_metadata'](device),epochs=epochs,
        train_samples=len(splits['train']),validation_samples=len(splits['validation']),history=[])
    started=time.perf_counter()
    for epoch in range(epochs+1):
        online={} if epoch==0 else train_epoch(net,loader,optimizer,cfg,device)
        val=evaluate(net,splits['validation'],cfg,device);row=dict(epoch=epoch,**online,**{'validation_'+k:v for k,v in val.items()})
        if epoch==epochs:row.update({'train_eval_'+k:v for k,v in evaluate(net,splits['train'],cfg,device).items()})
        result['history'].append(row)
        print('{} epoch {:2d}: online CE {}, val CE {:.6f}, acc {:.2%}'.format(name,epoch,'-' if not online else '{:.6f}'.format(online['online_train_ce']),val['ce'],val['accuracy']),flush=True)
    if use_test:result['test']=evaluate(net,splits['test'],cfg,device)
    result.update(updates=sum(r.get('updates',0) for r in result['history']),last_batch_size=result['history'][-1].get('last_batch_size'),
        test_evaluated=use_test,singleton_tail_merged=len(splits['train'])%cfg['batch_size']==1,final_state_sha256=state_hash(net),
        bn_counters=sorted(set(int(m.num_batches_tracked) for m in net.modules() if isinstance(m,nn.BatchNorm2d))),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    if record:
        folder=ROOT/'artifacts/ch07';folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(config=cfg,epoch=epochs,model={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=optimizer.state_dict()),folder/(prefix+'.pt'))
    if device.type=='cuda':torch.cuda.synchronize()
    result['seconds_including_evaluation_and_save']=time.perf_counter()-started
    return result,net


def candidates():
    base=dict(BASE,width=16,batch_size=128)
    return {'A':dict(base),'B':dict(base,residual=False),'C':dict(base,depth=34),'D':dict(base,depth=34,residual=False),
            'E':dict(base,depth=50),'F':dict(base,preactivation=True),'G':dict(base,lr=.1),'H':dict(base,zero_final_bn=True)}


def plot_results(data,experiments=False):
    plt=plot_setup();fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
    if experiments:
        for name,result in data['results'].items():
            h=result['history'];axes[0].plot([r['epoch'] for r in h],[r['validation_ce'] for r in h],label=name)
        axes[1].bar(list(data['results']),[r['history'][-1]['validation_accuracy'] for r in data['results'].values()],color='#147e83')
        axes[1].set(ylabel='末轮验证准确率',ylim=(0,1));suffix='experiments'
    else:
        h=data['history'];axes[0].plot([r['epoch'] for r in h[1:]],[r['online_train_ce'] for r in h[1:]],'o-',label='在线训练')
        axes[0].plot([r['epoch'] for r in h],[r['validation_ce'] for r in h],'o-',label='验证')
        axes[1].plot([r['epoch'] for r in h],[r['validation_accuracy'] for r in h],'o-',label='验证');axes[1].set(ylabel='准确率',ylim=(0,1));suffix='training'
    for ax in axes:ax.set_xlabel('epoch');ax.grid(alpha=.2)
    axes[0].set_ylabel('交叉熵');axes[0].legend(fontsize=9)
    fig.savefig(REPORT/'figures'/('7_6_'+suffix+'.png'),dpi=180);plt.close(fig)


def run_experiments(device,record):
    splits,hashes=COMMON['load_data'](include_test=False)
    splits={n:COMMON['balanced_subset'](splits[n],size,seed) for n,size,seed in [('train',2000,7601),('validation',1000,7602)]}
    results,rows={},[]
    for name,cfg in candidates().items():
        r,net=fit(cfg,splits,20,device,name);results[name]=r;rows.extend(dict(candidate=name,**e) for e in r['history']);del net
        if device.type=='cuda':torch.cuda.empty_cache()
    assert len({results[k]['initial_main_sha256'] for k in ('A','B','F','G','H')})==1
    assert results['C']['initial_main_sha256']==results['D']['initial_main_sha256']
    selected=min(results,key=lambda k:results[k]['history'][-1]['validation_ce'])
    result=dict(results=results,split_hashes=hashes,train_subset_seed=7601,validation_subset_seed=7602,test_evaluated=False,
        selected_by_final_validation_ce=selected,protocol='固定八候选/起始宽16/96输入/20轮320更新，按末轮验证CE比较，不读测试；非完整原论文复现实验')
    if record:write_json('7_6_experiments.json',result);write_csv('7_6_experiments_epochs.csv',rows);plot_results(result,True)
    return result


def benchmark(device,record):
    if device.type!='cuda':raise RuntimeError('CUDA事件计时需要GPU')
    specs={'R18_96':dict(BASE),'R34_96':dict(BASE,depth=34),'R50_96':dict(BASE,depth=50),'R18_224':dict(BASE,size=224)};rows=[]
    for name,cfg in specs.items():
        for mode in ('train','inference'):
            torch.cuda.empty_cache();net=build_net(cfg).to(device);torch.manual_seed(7604)
            X=torch.rand(8,1,cfg['size'],cfg['size'],device=device);y=torch.arange(8,device=device)%10;optimizer=torch.optim.SGD(net.parameters(),lr=.05)
            if mode=='train':
                net.train()
                def step():optimizer.zero_grad(set_to_none=True);F.cross_entropy(net(X),y).backward();optimizer.step()
            else:
                net.eval()
                @torch.inference_mode()
                def step():net(X)
            for _ in range(2):step()
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device);baseline=torch.cuda.memory_allocated(device)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
            for _ in range(5):step()
            end.record();torch.cuda.synchronize()
            rows.append(dict(model=name,mode=mode,batch_size=8,size=cfg['size'],milliseconds_per_batch=start.elapsed_time(end)/5,
                baseline_allocated_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
            del net,optimizer,X,y
    result=dict(runtime=COMMON['runtime_metadata'](device),synthetic_seed=7604,warmup=2,repeats=5,results=rows,
        protocol='顺序GPU驻留输入短测，训练含前向/CE/反传/SGD；预测eval+inference_mode；不含加载/Resize，不等于全进程')
    if record:write_json('7_6_benchmark.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);modes=p.add_mutually_exclusive_group()
    for name in ('inspect','experiments','benchmark'):modes.add_argument('--'+name,action='store_true')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--train-size',type=int);p.add_argument('--validation-size',type=int)
    p.add_argument('--guided',action='store_true');p.add_argument('--record',action='store_true');args=p.parse_args()
    if args.epochs<1 or args.batch_size<2 or args.train_size is not None and args.train_size<2 or args.validation_size is not None and args.validation_size<1:p.error('epochs至少1、batch和训练子集至少2、验证至少1')
    custom=args.epochs!=10 or args.batch_size!=256 or args.train_size is not None or args.validation_size is not None
    if (args.inspect or args.experiments or args.benchmark) and custom:p.error('固定模式不接受额外训练参数')
    device=COMMON['configure'](args.device)
    if args.guided:print('7.6：F(X)+S(X)按位相加；恒等/投影分开，末ReLU的条件讲清。',flush=True)
    # 普通模式的标准输出仅包含JSON，逐轮进度写到标准错误。
    with nullcontext() if args.guided else redirect_stdout(sys.stderr):
        if args.inspect:
            result=inspect_net()
            if args.record:
                write_json('7_6_architecture.json',result)
                write_csv('7_6_layers.csv',result['weighted_layers'])
        elif args.experiments:
            result=run_experiments(device,args.record)
        elif args.benchmark:
            result=benchmark(device,args.record)
        else:
            splits,hashes=COMMON['load_data'](include_test=not custom)
            for name,size,seed in [('train',args.train_size,7601),('validation',args.validation_size,7602)]:
                if size is not None:
                    splits[name]=COMMON['balanced_subset'](splits[name],size,seed)
            prefix='7_6_custom' if custom else '7_6'
            cfg=dict(BASE,batch_size=args.batch_size)
            result,_=fit(cfg,splits,args.epochs,device,prefix,not custom,args.record,prefix)
            result.update(custom=custom,split_hashes=hashes,split_seed=3601)
            if args.record:
                write_json(prefix+'_metrics.json',result)
                write_csv(prefix+'_epochs.csv',result['history'])
                if not custom:
                    plot_results(result)
    if args.guided:
        if args.inspect:print('参数',result['parameters'],'命名深度',result['named_depth'],'实际Conv',result['conv_count'])
        elif 'results' in result:
            if isinstance(result['results'],dict):
                for k,r in result['results'].items():print(k,'验证',r['history'][-1]['validation_accuracy'])
            else:print('GPU短测',len(result['results']),'项')
        else:print('更新',result['updates'],'尾批',result['last_batch_size'],'合并',result['singleton_tail_merged'],'测试',result.get('test'))
        print('教师运行不记用户掌握；八处练习待填。')
    else:print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
