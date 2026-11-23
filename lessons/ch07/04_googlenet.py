"""7.4 教材GoogLeNet：四分支Inception、通道降维、GAP和小Linear。

默认96输入/完整通道/SGD .1/十轮。--inspect只查结构，--experiments预定六组
验证对照（BN、两3×3、标签平滑、残差、去降维），不冒称完整后继架构。
--benchmark独立测训练/预测；--record才保存。custom小训练不读测试。
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

PREVIOUS=runpy.run_path(str(ROOT/'lessons/ch07/03_nin.py'))
COMMON=PREVIOUS['COMMON']
prepare,evaluate=(PREVIOUS[k] for k in ('prepare','evaluate'))
write_json,write_csv,plot_setup=(PREVIOUS[k] for k in ('write_json','write_csv','plot_setup'))
REPORT=ROOT/'reports/ch07'
BASE=dict(size=96,ratio=1,reduction=True,batch_norm=False,factorized=False,residual=False,
          label_smoothing=0.,lr=.1,batch_size=128)
# c1, (3×3前宽度,3×3输出), (5×5前宽度,5×5输出), 汇聚分支输出。
ARCH=[(64,(96,128),(16,32),32),(128,(128,192),(32,96),64),
      (192,(96,208),(16,48),64),(160,(112,224),(24,64),64),
      (128,(128,256),(24,64),64),(112,(144,288),(32,64),64),
      (256,(160,320),(32,128),128),(256,(160,320),(32,128),128),
      (384,(192,384),(48,128),128)]


def conv_act(ci,co,k=1,stride=1,padding=0,batch_norm=False,device='cpu'):
    layers=[('conv',nn.Conv2d(ci,co,k,stride,padding,bias=True,device=device))]
    if batch_norm:layers.append(('bn',nn.BatchNorm2d(co,device=device)))
    layers.append(('relu',nn.ReLU()))
    return nn.Sequential(OrderedDict(layers))


class Inception(nn.Module):
    def __init__(self,ci,c1,c2,c3,c4,reduction=True,batch_norm=False,
                 factorized=False,residual=False,device='cpu'):
        super().__init__()
        if min(ci,c1,*c2,*c3,c4)<1:raise ValueError('通道数须为正')
        self.out_channels=c1+c2[1]+c3[1]+c4
        self.reduction,self.factorized,self.residual=reduction,factorized,residual
        self.p1=conv_act(ci,c1,batch_norm=batch_norm,device=device)
        self.p2_reduce=conv_act(ci,c2[0],batch_norm=batch_norm,device=device) if reduction else nn.Identity()
        self.p2_spatial=conv_act(c2[0] if reduction else ci,c2[1],3,padding=1,batch_norm=batch_norm,device=device)
        self.p3_reduce=conv_act(ci,c3[0],batch_norm=batch_norm,device=device) if reduction else nn.Identity()
        width=c3[0] if reduction else ci
        if factorized:
            self.p3_spatial=nn.Sequential(OrderedDict([
                ('first',conv_act(width,width,3,padding=1,batch_norm=batch_norm,device=device)),
                ('second',conv_act(width,c3[1],3,padding=1,batch_norm=batch_norm,device=device))]))
        else:self.p3_spatial=conv_act(width,c3[1],5,padding=2,batch_norm=batch_norm,device=device)
        self.p4_pool=nn.MaxPool2d(3,stride=1,padding=1)
        self.p4_project=conv_act(ci,c4,batch_norm=batch_norm,device=device)
        if residual:
            self.shortcut=nn.Identity() if ci==self.out_channels else nn.Conv2d(ci,self.out_channels,1,device=device)

    def branches(self,X):
        return [self.p1(X),self.p2_spatial(self.p2_reduce(X)),
                self.p3_spatial(self.p3_reduce(X)),self.p4_project(self.p4_pool(X))]

    def forward(self,X):
        Y=torch.cat(self.branches(X),dim=1)
        return F.relu(Y+self.shortcut(X)) if self.residual else Y


def checked(config=None):
    cfg=dict(BASE,**(config or {}))
    if cfg['size']<1 or cfg['ratio'] not in (1,2,4,8,16):raise ValueError('尺寸须为正，ratio须为1/2/4/8/16')
    if cfg['lr']<=0 or cfg['batch_size']<1 or not 0<=cfg['label_smoothing']<1:raise ValueError('检查学习率、批量与平滑概率')
    if cfg['factorized'] and not cfg['reduction']:raise ValueError('本节两3×3变体保留降维，单独比较')
    return cfg


def build_net(config=None,device='cpu',seed=7400):
    cfg=checked(config)
    if torch.device(device).type not in ('cpu','meta'):raise ValueError('先在CPU初始化，再to目标设备')
    scale=lambda n:max(1,n//cfg['ratio'])
    options={k:cfg[k] for k in ('reduction','batch_norm','factorized','residual')}
    options['device']=device
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        b1=nn.Sequential(conv_act(1,scale(64),7,2,3,cfg['batch_norm'],device),nn.MaxPool2d(3,2,1))
        b2=nn.Sequential(conv_act(scale(64),scale(64),batch_norm=cfg['batch_norm'],device=device),
            conv_act(scale(64),scale(192),3,padding=1,batch_norm=cfg['batch_norm'],device=device),nn.MaxPool2d(3,2,1))
        stages=[];ci=scale(192);index=0
        for count in (2,5,2):
            layers=[]
            for _ in range(count):
                c1,c2,c3,c4=ARCH[index];index+=1
                block=Inception(ci,scale(c1),tuple(scale(c) for c in c2),tuple(scale(c) for c in c3),scale(c4),**options)
                layers.append(block);ci=block.out_channels
            if index<9:layers.append(nn.MaxPool2d(3,2,1))
            else:layers.extend([nn.AdaptiveAvgPool2d((1,1)),nn.Flatten()])
            stages.append(nn.Sequential(*layers))
        net=nn.Sequential(OrderedDict([('b1',b1),('b2',b2),('b3',stages[0]),('b4',stages[1]),('b5',stages[2]),('fc',nn.Linear(ci,10,device=device))]))
        if torch.device(device).type=='cpu':
            for name,layer in net.named_modules():
                if isinstance(layer,(nn.Conv2d,nn.Linear)):
                    role_seed=(seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))%(2**31)
                    torch.default_generator.manual_seed(role_seed)
                    nn.init.xavier_uniform_(layer.weight,gain=1.);nn.init.zeros_(layer.bias)
    return net


def parameter_hash(net,weighted_only=False):
    digest=hashlib.sha256()
    if weighted_only:
        pairs=[(name+'.'+key,p) for name,m in net.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear)) for key,p in m.named_parameters(recurse=False)]
    else:pairs=net.state_dict().items()
    for name,value in pairs:
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def inspect_net(config=None):
    cfg=checked(config);net=build_net(cfg,device='meta').eval()
    weighted,blocks,outer,handles=[],[],[],[]
    def weight_hook(name):
        def hook(m,inputs,out):
            fan=m.in_channels*m.kernel_size[0]*m.kernel_size[1] if isinstance(m,nn.Conv2d) else m.in_features
            weighted.append(dict(name=name,layer=type(m).__name__,input_shape=list(inputs[0].shape),output_shape=list(out.shape),
                parameters=sum(p.numel() for p in m.parameters()),weight_multiplications=out.numel()*fan,output_float32_bytes=out.numel()*4))
        return hook
    def block_hook(name):
        def hook(m,inputs,out):
            shapes=[list(y.shape) for y in m.branches(inputs[0])]
            blocks.append(dict(name=name,input_shape=list(inputs[0].shape),branch_shapes=shapes,output_shape=list(out.shape)))
        return hook
    # 两次meta前向分开记录，避免块hook重走分支时重复计入权重乘法。
    handles=[m.register_forward_hook(weight_hook(name)) for name,m in net.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    X=torch.zeros(1,1,cfg['size'],cfg['size'],device='meta')
    with torch.no_grad():
        for name,m in net.named_children():
            X=m(X);outer.append(dict(name=name,shape=list(X.shape)))
    for h in handles:h.remove()
    handles=[m.register_forward_hook(block_hook(name)) for name,m in net.named_modules() if isinstance(m,Inception)]
    with torch.no_grad():net(torch.zeros(1,1,cfg['size'],cfg['size'],device='meta'))
    for h in handles:h.remove()
    params=sum(p.numel() for p in net.parameters())
    return dict(config=cfg,outer=outer,blocks=blocks,weighted_layers=weighted,parameters=params,
        parameter_float32_bytes=params*4,weight_multiplications=sum(r['weight_multiplications'] for r in weighted),
        conv_count=sum(isinstance(m,nn.Conv2d) for m in net.modules()),linear_count=1,
        batch_norm_count=sum(isinstance(m,nn.BatchNorm2d) for m in net.modules()),
        note='meta形状与理论权重乘法；不含加法、归一化、激活、汇聚、拼接或反传，不等于实测峰值')


def train_epoch(net,loader,optimizer,cfg,device):
    net.train();ce_total,objective_total,correct,count,updates=0.,0.,0,0,0
    for X,y in loader:
        X,y=prepare(X,cfg,device),y.to(device)
        optimizer.zero_grad(set_to_none=True);logits=net(X)
        hard=F.cross_entropy(logits,y)
        objective=hard if cfg['label_smoothing']==0 else F.cross_entropy(logits,y,label_smoothing=cfg['label_smoothing'])
        objective.backward();optimizer.step()
        ce_total+=hard.item()*len(y);objective_total+=objective.item()*len(y)
        correct+=(logits.argmax(1)==y).sum().item();count+=len(y);updates+=1
    return dict(online_train_ce=ce_total/count,online_train_objective=objective_total/count,
                online_train_accuracy=correct/count,updates=updates,samples=count)


def fit(cfg,splits,epochs,device,name,use_test=False,record=False,prefix=None):
    net=build_net(cfg).to(device);optimizer=torch.optim.SGD(net.parameters(),lr=cfg['lr'])
    loader=COMMON['make_loader'](splits['train'],shuffle=True,batch_size=cfg['batch_size'])
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
    result=dict(config=cfg,model_seed_base=7400,initialization='CPU Xavier gain1/zero bias; stable SHA256 layer-name seeds',
        loader_seed=3603,initial_parameter_sha256=parameter_hash(net),initial_weighted_sha256=parameter_hash(net,True),
        parameters=sum(p.numel() for p in net.parameters()),runtime=COMMON['runtime_metadata'](device),epochs=epochs,
        train_samples=len(splits['train']),validation_samples=len(splits['validation']),history=[])
    started=time.perf_counter()
    for epoch in range(epochs+1):
        online={} if epoch==0 else train_epoch(net,loader,optimizer,cfg,device)
        val=evaluate(net,splits['validation'],cfg,device)
        row=dict(epoch=epoch,**online,**{'validation_'+k:v for k,v in val.items()})
        if epoch==epochs:row.update({'train_eval_'+k:v for k,v in evaluate(net,splits['train'],cfg,device).items()})
        result['history'].append(row)
        print('{} epoch {:2d}: online CE {}, val CE {:.6f}, acc {:.2%}'.format(name,epoch,
            '-' if not online else '{:.6f}'.format(online['online_train_ce']),val['ce'],val['accuracy']),flush=True)
    if use_test:result['test']=evaluate(net,splits['test'],cfg,device)
    result.update(updates=sum(r.get('updates',0) for r in result['history']),last_batch_size=(len(splits['train'])-1)%cfg['batch_size']+1,
        test_evaluated=use_test,final_parameter_sha256=parameter_hash(net),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    if record:
        folder=ROOT/'artifacts/ch07';folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(config=cfg,epoch=epochs,model={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=optimizer.state_dict()),folder/(prefix+'.pt'))
    if device.type=='cuda':torch.cuda.synchronize()
    result['seconds_including_evaluation_and_save']=time.perf_counter()-started
    return result,net


def candidates():
    return {'A':dict(BASE),'B':dict(BASE,batch_norm=True),'C':dict(BASE,factorized=True),
            'D':dict(BASE,label_smoothing=.1),'E':dict(BASE,residual=True),'F':dict(BASE,reduction=False)}


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
        axes[1].plot([r['epoch'] for r in h],[r['validation_accuracy'] for r in h],'o-',label='验证')
        axes[1].set(xlabel='epoch',ylabel='准确率',ylim=(0,1));suffix='training'
    axes[0].set(xlabel='epoch',ylabel='硬标签交叉熵');axes[0].legend(fontsize=10)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(REPORT/'figures'/('7_4_'+suffix+'.png'),dpi=180);plt.close(fig)


def run_experiments(device,record):
    splits,hashes=COMMON['load_data'](include_test=False)
    splits=dict(train=COMMON['balanced_subset'](splits['train'],2000,7401),validation=COMMON['balanced_subset'](splits['validation'],1000,7402))
    results,rows={},[]
    for name,cfg in candidates().items():
        result,net=fit(cfg,splits,20,device,name);results[name]=result
        rows.extend(dict(candidate=name,**r) for r in result['history']);del net
        if device.type=='cuda':torch.cuda.empty_cache()
    assert results['A']['initial_weighted_sha256']==results['B']['initial_weighted_sha256']==results['D']['initial_weighted_sha256']
    selected=min(results,key=lambda k:results[k]['history'][-1]['validation_ce'])
    output=dict(split_hashes=hashes,train_subset_seed=7401,validation_subset_seed=7402,train_per_class=200,validation_per_class=100,
        test_evaluated=False,results=results,selected_by_final_validation_ce=selected,
        protocol='预定六候选20轮320更新，按硬标签末轮验证CE比较；非完整官方后继架构；不读测试')
    if record:
        write_json('7_4_experiments.json',output);write_csv('7_4_experiments_epochs.csv',rows);plot_results(output,True)
    return output


def benchmark(device,record):
    if device.type!='cuda':raise RuntimeError('CUDA事件和显存测量需要GPU')
    specs={'Base96':dict(BASE),'NoReduce96':dict(BASE,reduction=False),'Two3x3_96':dict(BASE,factorized=True),'Base224':dict(BASE,size=224)}
    rows=[]
    for name,cfg in specs.items():
        for mode in ('train','inference'):
            torch.cuda.empty_cache();torch.manual_seed(7404);torch.cuda.manual_seed_all(7404)
            net=build_net(cfg).to(device);X=torch.rand(8,1,cfg['size'],cfg['size'],device=device);y=torch.arange(8,device=device)%10
            optimizer=torch.optim.SGD(net.parameters(),lr=.1)
            if mode=='train':
                net.train()
                def step():
                    optimizer.zero_grad(set_to_none=True);F.cross_entropy(net(X),y).backward();optimizer.step()
            else:
                net.eval()
                @torch.inference_mode()
                def step():net(X)
            for _ in range(2):step()
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
            baseline=torch.cuda.memory_allocated(device)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
            for _ in range(5):step()
            end.record();torch.cuda.synchronize()
            rows.append(dict(model=name,mode=mode,batch_size=8,size=cfg['size'],milliseconds_per_batch=start.elapsed_time(end)/5,
                baseline_allocated_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
            del net,optimizer,X,y
    output=dict(runtime=COMMON['runtime_metadata'](device),synthetic_seed=7404,warmup=2,repeats=5,results=rows,
        protocol='单独顺序短测；GPU驻留输入；训练前向/CE/反传/SGD，预测eval+inference_mode；不含加载/Resize；分配不含全进程')
    if record:
        write_json('7_4_benchmark.json',output)
        plt=plot_setup();fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
        for mode,offset,color in [('train',-.18,'#147e83'),('inference',.18,'#e6ab52')]:
            data=[r for r in rows if r['mode']==mode];xs=[i+offset for i in range(4)]
            axes[0].bar(xs,[r['milliseconds_per_batch'] for r in data],.36,label=mode,color=color)
            axes[1].bar(xs,[r['peak_allocated_bytes']/2**20 for r in data],.36,label=mode,color=color)
        for ax in axes:ax.set_xticks(range(4));ax.set_xticklabels(['原96','无降维96','两3×3','原224']);ax.legend(fontsize=10)
        axes[0].set_ylabel('每批毫秒');axes[1].set_ylabel('分配峰值MiB')
        fig.savefig(REPORT/'figures/7_4_benchmark.png',dpi=180);plt.close(fig)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    for n in ('inspect','experiments','benchmark'):modes.add_argument('--'+n,action='store_true')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--train-size',type=int);parser.add_argument('--validation-size',type=int)
    parser.add_argument('--guided',action='store_true');parser.add_argument('--record',action='store_true');args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or any(v is not None and v<1 for v in (args.train_size,args.validation_size)):parser.error('轮数、批量、子集须为正')
    custom=args.epochs!=10 or args.batch_size!=128 or args.train_size is not None or args.validation_size is not None
    if (args.inspect or args.experiments or args.benchmark) and custom:parser.error('固定协议不接受额外训练参数')
    device=COMMON['configure'](args.device)
    if args.guided:print('7.4：四分支共享输入，1×1先降维，cat按通道拼接；GAP后仍有小Linear。',flush=True)
    if args.inspect:
        result=inspect_net()
        if args.record:write_json('7_4_architecture.json',result);write_csv('7_4_layers.csv',result['weighted_layers'])
    elif args.experiments:result=run_experiments(device,args.record)
    elif args.benchmark:result=benchmark(device,args.record)
    else:
        splits,hashes=COMMON['load_data'](include_test=not custom)
        for name,size,seed in (('train',args.train_size,7401),('validation',args.validation_size,7402)):
            if size is not None:splits[name]=COMMON['balanced_subset'](splits[name],size,seed)
        prefix='7_4_custom' if custom else '7_4';cfg=dict(BASE,batch_size=args.batch_size)
        result,_=fit(cfg,splits,args.epochs,device,'GoogLeNet',not custom,args.record,prefix)
        result.update(custom=custom,split_hashes=hashes,split_seed=3601,preprocessing='float32/255 then bilinear, align_corners=False, antialias=True')
        if args.record:
            write_json(prefix+'_metrics.json',result);write_csv(prefix+'_epochs.csv',result['history'])
            if not custom:plot_results(result)
    if args.guided:
        if args.inspect:
            for row in result['outer']:print(row['name'],row['shape'])
            print('参数',result['parameters'],'Conv',result['conv_count'],'单图权重乘法',result['weight_multiplications'])
        elif 'test' in result:print('固定末轮测试',result['test'])
        print('运行不代表掌握；练习八处TODO保留给自己填写。')
    else:print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
