"""7.4独立NumPy分支计算图、梯度、尾批SGD、文档跟做与检查点核对。

不重跑正式训练；练习参考仅在内存运行。NumPy图支持带padding的汇聚与cat梯度相加。
"""
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
H=runpy.run_path(str(ROOT/'lessons/ch07/04_googlenet.py'))


class Value:
    def __init__(self,data):self.data=data;self.grad=None


class NumpyGraph:
    """只用NumPy实现实际分支前向和反向，参数名对应待核对模块。"""
    def __init__(self,params):self.params=params;self.tape=[];self.gradients={}

    def operation(self,data,parents,backward):
        out=Value(data);self.tape.append((out,parents,backward));return out

    def relu(self,x):
        return self.operation(np.maximum(x.data,0),[x],lambda g:[g*(x.data>0)])

    def module(self,m,x,name=''):
        if isinstance(m,nn.Sequential):
            for key,child in m.named_children():x=self.module(child,x,name+'.'+key if name else key)
            return x
        if isinstance(m,H['Inception']):
            p1=self.module(m.p1,x,name+'.p1')
            p2=self.module(m.p2_spatial,self.module(m.p2_reduce,x,name+'.p2_reduce'),name+'.p2_spatial')
            p3=self.module(m.p3_spatial,self.module(m.p3_reduce,x,name+'.p3_reduce'),name+'.p3_spatial')
            p4=self.module(m.p4_project,self.module(m.p4_pool,x,name+'.p4_pool'),name+'.p4_project')
            branches=[p1,p2,p3,p4];cuts=np.cumsum([p.data.shape[1] for p in branches])[:-1]
            out=self.operation(np.concatenate([p.data for p in branches],axis=1),branches,lambda g:list(np.split(g,cuts,axis=1)))
            if m.residual:
                shortcut=self.module(m.shortcut,x,name+'.shortcut')
                out=self.relu(self.operation(out.data+shortcut.data,[out,shortcut],lambda g:[g,g]))
            return out
        if isinstance(m,nn.Identity):return x
        if isinstance(m,nn.ReLU):return self.relu(x)
        if isinstance(m,nn.Conv2d):
            kh,kw=m.kernel_size;sh,sw=m.stride;ph,pw=m.padding
            padded=np.pad(x.data,((0,0),(0,0),(ph,ph),(pw,pw)))
            patches=np.lib.stride_tricks.sliding_window_view(padded,(kh,kw),axis=(2,3))[:,:,::sh,::sw]
            w,b=self.params[name+'.weight'],self.params[name+'.bias']
            out=np.einsum('nchwkl,ockl->nohw',patches,w)+b[None,:,None,None]
            def backward(g):
                self.add_param(name+'.weight',np.einsum('nohw,nchwkl->ockl',g,patches))
                self.add_param(name+'.bias',g.sum((0,2,3)))
                gx=np.zeros_like(padded);oh,ow=g.shape[2:]
                for i in range(kh):
                    for j in range(kw):gx[:,:,i:i+oh*sh:sh,j:j+ow*sw:sw]+=np.einsum('nohw,oc->nchw',g,w[:,:,i,j])
                return [gx[:,:,ph:ph+x.data.shape[2],pw:pw+x.data.shape[3]]]
            return self.operation(out,[x],backward)
        if isinstance(m,nn.MaxPool2d):
            pair=lambda v:(v,v) if isinstance(v,int) else v
            kh,kw=pair(m.kernel_size);sh,sw=pair(m.stride or m.kernel_size);ph,pw=pair(m.padding)
            padded=np.pad(x.data,((0,0),(0,0),(ph,ph),(pw,pw)),constant_values=-np.inf)
            patches=np.lib.stride_tricks.sliding_window_view(padded,(kh,kw),axis=(2,3))[:,:,::sh,::sw]
            flat=patches.reshape(*patches.shape[:4],kh*kw);ids=flat.argmax(-1)
            def backward(g):
                gx=np.zeros_like(padded);oh,ow=g.shape[2:]
                for i in range(kh):
                    for j in range(kw):gx[:,:,i:i+oh*sh:sh,j:j+ow*sw:sw]+=g*(ids==i*kw+j)
                return [gx[:,:,ph:ph+x.data.shape[2],pw:pw+x.data.shape[3]]]
            return self.operation(flat.max(-1),[x],backward)
        if isinstance(m,nn.AdaptiveAvgPool2d):
            return self.operation(x.data.mean((2,3),keepdims=True),[x],
                lambda g:[np.broadcast_to(g,x.data.shape)/(x.data.shape[2]*x.data.shape[3])])
        if isinstance(m,nn.Flatten):
            return self.operation(x.data.reshape(len(x.data),-1),[x],lambda g:[g.reshape(x.data.shape)])
        if isinstance(m,nn.Linear):
            w,b=self.params[name+'.weight'],self.params[name+'.bias']
            def backward(g):
                self.add_param(name+'.weight',g.T@x.data);self.add_param(name+'.bias',g.sum(0))
                return [g@w]
            return self.operation(x.data@w.T+b,[x],backward)
        raise TypeError(type(m))

    def add_param(self,key,grad):
        self.gradients[key]=self.gradients.get(key,0)+grad

    def run(self,net,X,labels,backward=True):
        source=Value(X);out=self.module(net,source)
        shifted=out.data-out.data.max(1,keepdims=True)
        logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True))
        ce=-logp[np.arange(len(labels)),labels].mean()
        if not backward:return ce,out.data
        out.grad=np.exp(logp);out.grad[np.arange(len(labels)),labels]-=1;out.grad/=len(labels)
        for node,parents,fn in reversed(self.tape):
            if node.grad is None:continue
            for parent,g in zip(parents,fn(node.grad)):
                parent.grad=g.copy() if parent.grad is None else parent.grad+g
        return ce,out.data,self.gradients,source.grad


def numpy_pass(net,params,X,y,backward=True):return NumpyGraph(params).run(net,X,y,backward)


def numerical():
    H['COMMON']['configure']('cpu')
    cfg=dict(H['BASE'],ratio=16)
    net=H['build_net'](cfg).double().eval()
    for m in net.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):nn.init.constant_(m.bias,.05)
    params={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
    gen=torch.Generator().manual_seed(7407)
    X=torch.rand(3,1,96,96,generator=gen,dtype=torch.float64).requires_grad_();y=torch.tensor([1,4,7])
    logits=net(X);loss=F.cross_entropy(logits,y);loss.backward()
    ce,z,grads,gx=numpy_pass(net,params,X.detach().numpy(),y.numpy())
    errors=dict(logits=float(np.max(np.abs(z-logits.detach().numpy()))),ce=abs(ce-loss.item()),
        input_gradient=float(np.max(np.abs(gx-X.grad.numpy()))),
        parameter_gradient=max(float(np.max(np.abs(grads[k]-v.grad.numpy()))) for k,v in net.named_parameters()))
    optimizer=torch.optim.SGD(net.parameters(),lr=.1)
    for ids in ([0,1],[2]):
        optimizer.zero_grad(set_to_none=True);F.cross_entropy(net(X.detach()[ids]),y[ids]).backward();optimizer.step()
        g=numpy_pass(net,params,X.detach().numpy()[ids],y.numpy()[ids])[2]
        for key in params:params[key]-=.1*g[key]
    errors['sgd_two_updates_tail1']=max(float(np.max(np.abs(params[k]-v.detach().numpy()))) for k,v in net.named_parameters())
    probes={}
    for variant in ('baseline','factorized','residual','no_reduction'):
        options=dict(factorized=variant=='factorized',residual=variant=='residual',reduction=variant!='no_reduction')
        torch.manual_seed(7408)
        small=nn.Sequential(H['Inception'](2,2,(2,3),(1,2),1,**options),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(8,3)).double().eval()
        for m in small.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):nn.init.constant_(m.bias,.05)
        p={k:v.detach().numpy().copy() for k,v in small.state_dict().items()}
        x=torch.rand(3,2,5,7,generator=gen,dtype=torch.float64).requires_grad_();labels=torch.tensor([0,1,2])
        F.cross_entropy(small(x),labels).backward()
        ce,z,g,gx=numpy_pass(small,p,x.detach().numpy(),labels.numpy())
        diff=max(float(np.max(np.abs(g[k]-v.grad.numpy()))) for k,v in small.named_parameters())
        entry=dict(parameters=sum(v.numel() for v in small.parameters()),parameter_gradient_error=diff,
                   input_gradient_error=float(np.max(np.abs(gx-x.grad.numpy()))))
        if variant=='baseline':
            finite=[]
            for key in p:
                coordinate=np.unravel_index(np.argmax(np.abs(g[key])),p[key].shape);old,eps=p[key][coordinate],1e-6
                p[key][coordinate]=old+eps;high=numpy_pass(small,p,x.detach().numpy(),labels.numpy(),False)[0]
                p[key][coordinate]=old-eps;low=numpy_pass(small,p,x.detach().numpy(),labels.numpy(),False)[0]
                p[key][coordinate]=old;finite.append(abs((high-low)/(2*eps)-g[key][coordinate]))
            entry.update(central_difference_coordinates=len(finite),finite_difference_step=1e-6,central_difference_error=max(finite))
        probes[variant]=entry
    assert max(errors.values())<1e-7,errors
    for item in probes.values():
        assert item['parameter_gradient_error']<1e-7 and item['input_gradient_error']<1e-7,item
        assert item.get('central_difference_error',0)<1e-7,item
    return dict(parameters=sum(p.numel() for p in net.parameters()),conv_count=57,inception_blocks=9,input_shape=list(X.shape),
        errors=errors,probes=probes,numerical_probe_bias=.05,
        scope='独立NumPy四分支/带padding最大汇聚/cat/梯度相加/GAP/Linear；完整96空间拓扑缩通道，不是完整597万参数梯度')


def batch_norm_probe():
    torch.manual_seed(7409)
    layer=nn.BatchNorm2d(3,momentum=.1).double().train()
    X=torch.randn(2,3,4,5,dtype=torch.float64);a=X.numpy();mean=a.mean((0,2,3));var=a.var((0,2,3))
    out=layer(X);expected=(a-mean[None,:,None,None])/np.sqrt(var[None,:,None,None]+1e-5)
    train_error=float(np.max(np.abs(out.detach().numpy()-expected)))
    expected_mean=.1*mean;expected_var=.9+.1*a.var((0,2,3),ddof=1)
    state_error=max(float(np.max(np.abs(layer.running_mean.numpy()-expected_mean))),float(np.max(np.abs(layer.running_var.numpy()-expected_var))))
    layer.eval();actual=layer(X).detach().numpy()
    prediction=(a-expected_mean[None,:,None,None])/np.sqrt(expected_var[None,:,None,None]+1e-5)
    eval_error=float(np.max(np.abs(actual-prediction)))
    assert max(train_error,state_error,eval_error)<1e-12
    return dict(train_forward_error=train_error,running_statistics_error=state_error,eval_forward_error=eval_error,
        note='仅内置BatchNorm2d预览，训练前向用有偏方差、移动方差用无偏估计；完整教学留到7.5')


def practice():
    text=(ROOT/'docs/ch07/04_googlenet.md').read_text(encoding='utf8');blocks=[]
    for page in text.split('<!-- pagebreak -->'):
        if re.search(r'## (16|17|18|19|20) ',page):blocks.extend(re.findall(r'```python\n(.*?)\n```',page.split('<!-- side -->')[0],re.S))
    assert len(blocks)==5
    path=ROOT/'.cache/7_4_follow.py';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n\n'.join(blocks)+'\n',encoding='utf8')
    output=io.StringIO()
    with contextlib.redirect_stdout(output):runpy.run_path(str(path),run_name='__main__')
    exercise=ROOT/'exercises/ch07/04_googlenet.py';original=exercise.read_bytes();s=original.decode('utf8')
    assert len(re.findall(r'# TODO [1-8]：',s))==8
    fixes={'self.p1=None':'self.p1=nn.Conv2d(ci,c1,1)',
        'self.p2=None':'self.p2=nn.Sequential(nn.Conv2d(ci,c2[0],1),nn.ReLU(),nn.Conv2d(c2[0],c2[1],3,padding=1),nn.ReLU())',
        'self.p3=None':'self.p3=nn.Sequential(nn.Conv2d(ci,c3[0],1),nn.ReLU(),nn.Conv2d(c3[0],c3[1],5,padding=2),nn.ReLU())',
        'self.p4=None':'self.p4=nn.Sequential(nn.MaxPool2d(3,1,1),nn.Conv2d(ci,c4,1),nn.ReLU())',
        'output=None':'output=torch.cat(branches,dim=1)','loss=None':'loss=F.cross_entropy(logits,y)',
        'pass  # TODO 7':'loss.backward()  # TODO 7','pass  # TODO 8':'optimizer.step()  # TODO 8'}
    for old,new in fixes.items():assert s.count(old)==1,old;s=s.replace(old,new)
    reference=io.StringIO()
    with contextlib.redirect_stdout(reference):exec(compile(s,str(exercise),'exec'),{'__name__':'__main__','__file__':str(exercise)})
    assert exercise.read_bytes()==original
    return dict(follow_blocks=5,follow_stdout=output.getvalue(),exercise_reference_stdout=reference.getvalue(),
        exercise_source_unchanged=True,unanswered_todos=8)


@torch.no_grad()
def checkpoint():
    path=ROOT/'artifacts/ch07/7_4.pt'
    if not path.exists():return dict(skipped='尚无正式本地模型')
    device=H['COMMON']['configure']('auto');state=torch.load(path,map_location='cpu',weights_only=True)
    net=H['build_net'](state['config']).eval();net.load_state_dict(state['model'])
    report=json.loads((ROOT/'reports/ch07/7_4_metrics.json').read_text(encoding='utf8'))
    assert H['parameter_hash'](net)==report['final_parameter_sha256']
    splits,_=H['COMMON']['load_data'](include_test=False)
    X=torch.stack([splits['validation'][i][0] for i in range(2)])
    X=H['prepare'](X,state['config'],torch.device('cpu'));cpu=net(X)
    reference=net.double()(X.double());net=net.float().to(device);gpu=net(X.to(device)).cpu()
    with torch.backends.cudnn.flags(enabled=False,benchmark=False,deterministic=True,allow_tf32=False):
        fallback=net(X.to(device)).cpu()
    error=(cpu-gpu).abs().max().item()
    cpu_error=(cpu.double()-reference).abs().max().item()
    gpu_error=(gpu.double()-reference).abs().max().item()
    fallback_error=(fallback.double()-reference).abs().max().item()
    # 本机默认cuDNN路径较CPU/非cuDNN路径舍入差大；用double参照定位后分别核对。
    assert cpu_error<1e-4 and fallback_error<1e-4 and gpu_error<1e-3
    assert all(torch.equal(cpu.argmax(1),a.argmax(1)) for a in (gpu,reference,fallback))
    return dict(max_logit_difference=error,predictions=cpu.argmax(1).tolist(),validation_positions=[0,1],parameter_hash_matches=True,
        cpu_vs_float64=cpu_error,cudnn_vs_float64=gpu_error,cudnn_disabled_vs_float64=fallback_error,
        cudnn_error_limit=1e-3,note='两张验证图，默认cuDNN得分差超过原1e-4阈值；double及关闭cuDNN定位为后端路径舍入差，类别一致；正式训练协议未改')


def main():
    path=ROOT/'reports/ch07/7_4_verification.json';result=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
    result.update(numerical=numerical(),batch_norm=batch_norm_probe(),practice=practice(),checkpoint_recovery=checkpoint())
    H['write_json']('7_4_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
