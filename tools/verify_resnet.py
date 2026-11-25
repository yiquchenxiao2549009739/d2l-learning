"""7.6独立NumPy残差计算图、全缩通道网络/瓶颈/预激活梯度、练习与恢复核对。"""
import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(ROOT/'lessons/ch07/06_resnet.py'))
OLD=runpy.run_path(str(ROOT/'tools/verify_batch_norm.py'))


class ResGraph(OLD['BNGraph']):
    def module(self,m,x,name=''):
        if isinstance(m,(H['Residual'],H['Bottleneck'],H['PreActResidual'])):
            call=lambda key,v:self.module(getattr(m,key),v,name+'.'+key)
            if isinstance(m,H['PreActResidual']):
                activated=self.relu(call('bn1',x));y=call('conv1',activated);y=call('conv2',self.relu(call('bn2',y)))
                skip=x if isinstance(m.shortcut,nn.Identity) else call('shortcut',activated)
            else:
                y=self.relu(call('bn1',call('conv1',x)));y=call('bn2',call('conv2',y))
                if isinstance(m,H['Bottleneck']):y=call('bn3',call('conv3',self.relu(y)))
                skip=call('shortcut',x)
            if m.residual:y=self.operation(y.data+skip.data,[y,skip],lambda g:[g,g])
            return y if isinstance(m,H['PreActResidual']) else self.relu(y)
        return super().module(m,x,name)


def numpy_pass(net,params,X,y,backward=True):return ResGraph(params).run(net,X,y,backward)


def numerical():
    H['COMMON']['configure']('cpu');gen=torch.Generator().manual_seed(7607)
    cfg=dict(H['BASE'],width=2);net=H['build_net'](cfg).double().train()
    # 数值探针bias .05减少ReLU边界歧义，与正式零bias初始化分开。
    for m in net.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):nn.init.constant_(m.bias,.05)
    params={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
    X=torch.rand(3,1,96,96,generator=gen,dtype=torch.float64).requires_grad_();y=torch.tensor([0,4,9])
    logits=net(X);loss=F.cross_entropy(logits,y);loss.backward()
    ce,z,g,gx=numpy_pass(net,params,X.detach().numpy(),y.numpy())
    errors=dict(logits=float(np.max(np.abs(z-logits.detach().numpy()))),ce=abs(ce-loss.item()),
        parameter_gradient=max(float(np.max(np.abs(g[k]-p.grad.numpy()))) for k,p in net.named_parameters()),
        input_gradient=float(np.max(np.abs(gx-X.grad.numpy()))))
    optimizer=torch.optim.SGD(net.parameters(),lr=.01);optimizer.step()
    for k,p in net.named_parameters():params[k]-=.01*g[k]
    optimizer.zero_grad(set_to_none=True);F.cross_entropy(net(X.detach()[2:]),y[2:]).backward();optimizer.step()
    gradients=numpy_pass(net,params,X.detach().numpy()[2:],y.numpy()[2:])[2]
    for k,p in net.named_parameters():params[k]-=.01*gradients[k]
    errors['two_updates_parameters']=max(float(np.max(np.abs(params[k]-p.detach().numpy()))) for k,p in net.named_parameters())
    errors['two_updates_buffers']=max(float(np.max(np.abs(params[k]-p.detach().numpy()))) for k,p in net.named_buffers())
    assert max(errors.values())<1e-7,errors
    probes={}
    for name,cls,ci,width,stride in [('identity',H['Residual'],2,2,1),('projection',H['Residual'],2,3,2),
            ('bottleneck',H['Bottleneck'],2,2,2),('preactivation',H['PreActResidual'],2,3,2)]:
        block=nn.Sequential(cls(ci,width,stride),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(width*cls.expansion,3)).double().train()
        torch.manual_seed(7608)
        for m in block.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):nn.init.normal_(m.weight,0,.1);nn.init.constant_(m.bias,.05)
        xx=torch.rand(3,ci,5,5,generator=gen,dtype=torch.float64).requires_grad_();yy=torch.tensor([0,1,2])
        pp={k:v.detach().numpy().copy() for k,v in block.state_dict().items()};zz=block(xx);ll=F.cross_entropy(zz,yy);ll.backward()
        cc,zzn,gg,gxx=numpy_pass(block,{k:v.copy() for k,v in pp.items()},xx.detach().numpy(),yy.numpy())
        entry=dict(parameters=sum(p.numel() for p in block.parameters()),gradient_error=max(float(np.max(np.abs(gg[k]-p.grad.numpy()))) for k,p in block.named_parameters()),
            input_gradient_error=float(np.max(np.abs(gxx-xx.grad.numpy()))),logits_error=float(np.max(np.abs(zzn-zz.detach().numpy()))))
        finite=[]
        for key,p in block.named_parameters():
            coord=np.unravel_index(np.argmax(np.abs(gg[key])),pp[key].shape);old=pp[key][coord];eps=1e-6
            pp[key][coord]=old+eps;hi=numpy_pass(block,{k:v.copy() for k,v in pp.items()},xx.detach().numpy(),yy.numpy(),False)[0]
            pp[key][coord]=old-eps;lo=numpy_pass(block,{k:v.copy() for k,v in pp.items()},xx.detach().numpy(),yy.numpy(),False)[0]
            pp[key][coord]=old;finite.append(abs((hi-lo)/(2*eps)-gg[key][coord]))
        entry.update(finite_difference_coordinates=len(finite),finite_difference_step=1e-6,finite_difference_error=max(finite))
        assert max(entry[k] for k in ('gradient_error','input_gradient_error','logits_error','finite_difference_error'))<1e-7,entry
        probes[name]=entry
    return dict(parameters=sum(p.numel() for p in net.parameters()),residual_blocks=8,input_shape=list(X.shape),errors=errors,probes=probes,
        numerical_probe_bias=.05,scope='完整96空间拓扑/8块缩起始通道2；独立NumPy前反向和批3/1两次SGD，非1117万全参数梯度检查')


def practice():
    text=(ROOT/'docs/ch07/06_resnet.md').read_text(encoding='utf8');blocks=[]
    for page in text.split('<!-- pagebreak -->'):
        if re.search(r'## (19|20|21|22|23) ',page):blocks.extend(re.findall(r'```python\n(.*?)\n```',page.split('<!-- side -->')[0],re.S))
    assert len(blocks)==5
    path=ROOT/'.cache/7_6_follow.py';path.write_text('\n\n'.join(blocks)+'\n',encoding='utf8');out=io.StringIO()
    with contextlib.redirect_stdout(out):runpy.run_path(str(path),run_name='__main__')
    exercise=ROOT/'exercises/ch07/06_resnet.py';original=exercise.read_bytes();s=original.decode('utf8')
    fixes={'Y=None  # TODO 1':'Y=F.relu(self.bn1(self.conv1(X)))  # TODO 1','Y=None  # TODO 2':'Y=self.bn2(self.conv2(Y))  # TODO 2',
        'skip=None':'skip=self.shortcut(X)','combined=None':'combined=Y+skip','output=None':'output=F.relu(combined)',
        'loss=None':'loss=F.cross_entropy(logits,y)','pass  # TODO 7':'loss.backward()  # TODO 7','pass  # TODO 8':'optimizer.step()  # TODO 8'}
    assert len(re.findall(r'# TODO [1-8]：',s))==8
    for old,new in fixes.items():assert s.count(old)==1,old;s=s.replace(old,new)
    ref=io.StringIO()
    with contextlib.redirect_stdout(ref):exec(compile(s,str(exercise),'exec'),{'__name__':'__main__','__file__':str(exercise)})
    assert exercise.read_bytes()==original
    return dict(follow_blocks=5,follow_stdout=out.getvalue(),exercise_reference_stdout=ref.getvalue(),exercise_source_unchanged=True,unanswered_todos=8)


@torch.no_grad()
def checkpoint():
    path=ROOT/'artifacts/ch07/7_6.pt'
    if not path.exists():return dict(skipped='尚无本地正式模型')
    device=H['COMMON']['configure']('auto');state=torch.load(path,map_location='cpu',weights_only=True)
    net=H['build_net'](state['config']).eval();net.load_state_dict(state['model'])
    report=json.loads((ROOT/'reports/ch07/7_6_metrics.json').read_text(encoding='utf8'));assert H['state_hash'](net)==report['final_state_sha256']
    splits,_=H['COMMON']['load_data'](include_test=False);X=torch.stack([splits['validation'][i][0] for i in range(16)])
    X=H['prepare'](X,state['config'],'cpu');before={k:v.clone() for k,v in net.named_buffers()}
    cpu=net(X);ref=net.double()(X.double());net=net.float().to(device);gpu=net(X.to(device)).cpu()
    with torch.backends.cudnn.flags(enabled=False,benchmark=False,deterministic=True,allow_tf32=False):fallback=net(X.to(device)).cpu()
    errors=dict(cpu_cuda=(cpu-gpu).abs().max().item(),cpu_float64=(cpu.double()-ref).abs().max().item(),
        cudnn_float64=(gpu.double()-ref).abs().max().item(),fallback_float64=(fallback.double()-ref).abs().max().item())
    assert max(errors.values())<1e-3,errors
    assert all(torch.equal(cpu.argmax(1),a.argmax(1)) for a in (ref,gpu,fallback))
    assert all(torch.equal(v.cpu(),before[k]) for k,v in net.named_buffers())
    return dict(errors=errors,validation_positions=list(range(16)),predictions=cpu.argmax(1).tolist(),state_hash_matches=True,
        eval_buffers_unchanged=True,bn_counters=sorted(set(int(m.num_batches_tracked) for m in net.modules() if isinstance(m,nn.BatchNorm2d))))


def main():
    path=ROOT/'reports/ch07/7_6_verification.json';r=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
    r.update(numerical=numerical(),practice=practice(),checkpoint=checkpoint());H['write_json']('7_6_verification.json',r)
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
