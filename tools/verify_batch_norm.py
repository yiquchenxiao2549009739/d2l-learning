"""7.5独立NumPy BN前/反向、整网/SGD、框架对齐、文档与练习核对。"""
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
H=runpy.run_path(str(ROOT/'lessons/ch07/05_batch_norm.py'))
OLD=runpy.run_path(str(ROOT/'tools/verify_googlenet.py'))


class BNGraph(OLD['NumpyGraph']):
    def module(self,m,x,name=''):
        if isinstance(m,nn.Sigmoid):
            a=1/(1+np.exp(-x.data))
            return self.operation(a,[x],lambda g:[g*a*(1-a)])
        if isinstance(m,nn.Dropout):
            assert m.p==0 or not m.training
            return x
        if isinstance(m,nn.AvgPool2d):
            k=m.kernel_size;s=m.stride or k
            windows=np.lib.stride_tricks.sliding_window_view(x.data,(k,k),axis=(2,3))[:,:,::s,::s]
            def backward(g):
                gx=np.zeros_like(x.data);oh,ow=g.shape[2:]
                for i in range(k):
                    for j in range(k):gx[:,:,i:i+oh*s:s,j:j+ow*s:s]+=g/(k*k)
                return [gx]
            return self.operation(windows.mean((-2,-1)),[x],backward)
        if isinstance(m,(H['ManualBatchNorm'],nn.BatchNorm1d,nn.BatchNorm2d)):
            axes=(0,) if x.data.ndim==2 else (0,2,3);shape=(1,-1) if x.data.ndim==2 else (1,-1,1,1)
            count=x.data.size//x.data.shape[1]
            if m.training:
                mean=x.data.mean(axes);var=x.data.var(axes)
                self.params[name+'.running_mean']=(1-m.momentum)*self.params[name+'.running_mean']+m.momentum*mean
                self.params[name+'.running_var']=(1-m.momentum)*self.params[name+'.running_var']+m.momentum*var*count/(count-1)
                self.params[name+'.num_batches_tracked']+=1
            else:mean,var=self.params[name+'.running_mean'],self.params[name+'.running_var']
            inv=1/np.sqrt(var.reshape(shape)+m.eps);xhat=(x.data-mean.reshape(shape))*inv
            affine=m.weight is not None;gamma=self.params[name+'.weight'].reshape(shape) if affine else 1.
            y=xhat*gamma+(self.params[name+'.bias'].reshape(shape) if affine else 0.)
            def backward(g):
                if affine:
                    self.add_param(name+'.weight',(g*xhat).sum(axes));self.add_param(name+'.bias',g.sum(axes))
                dx=gamma*inv*(g-g.mean(axes,keepdims=True)-xhat*(g*xhat).mean(axes,keepdims=True)) if m.training else g*gamma*inv
                return [dx]
            return self.operation(y,[x],backward)
        return super().module(m,x,name)


def numpy_pass(net,params,X,y,backward=True):return BNGraph(params).run(net,X,y,backward)


def numerical():
    H['COMMON']['configure']('cpu');cfg=dict(H['BASE']);net=H['build_net'](cfg).double().train()
    params={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
    gen=torch.Generator().manual_seed(7507)
    X=torch.rand(5,1,28,28,generator=gen,dtype=torch.float64).requires_grad_();labels=torch.tensor([0,1,4,7,9])
    logits=net(X);loss=F.cross_entropy(logits,labels);loss.backward()
    ce,z,g,gx=numpy_pass(net,params,X.detach().numpy(),labels.numpy())
    errors=dict(logits=float(np.max(np.abs(z-logits.detach().numpy()))),ce=abs(ce-loss.item()),input_gradient=float(np.max(np.abs(gx-X.grad.numpy()))),
        parameter_gradient=max(float(np.max(np.abs(g[k]-v.grad.numpy()))) for k,v in net.named_parameters()))
    optimizer=torch.optim.SGD(net.parameters(),lr=.1);optimizer.step()
    for k,v in net.named_parameters():params[k]-=.1*g[k]
    for ids in ([0,1],[2,3,4]):
        optimizer.zero_grad(set_to_none=True);F.cross_entropy(net(X.detach()[ids]),labels[ids]).backward();optimizer.step()
        gradients=numpy_pass(net,params,X.detach().numpy()[ids],labels.numpy()[ids])[2]
        for k,v in net.named_parameters():params[k]-=.1*gradients[k]
    errors['three_sgd_updates_parameters']=max(float(np.max(np.abs(params[k]-v.detach().numpy()))) for k,v in net.named_parameters())
    errors['three_updates_running_states']=max(float(np.max(np.abs(params[k]-v.detach().numpy()))) for k,v in net.named_buffers())
    assert max(errors.values())<1e-8,errors
    # 单层的中心差分使用独立公式，不运行PyTorchautograd求数值差分。
    finite=[]
    for shape in ((5,3),(2,3,2,3)):
        a=torch.randn(*shape,generator=gen,dtype=torch.float64);gamma=torch.randn(3,generator=gen,dtype=torch.float64);beta=torch.randn(3,generator=gen,dtype=torch.float64)
        upstream=torch.randn(*shape,generator=gen,dtype=torch.float64).numpy();axes=(0,) if len(shape)==2 else (0,2,3);view=(1,-1) if len(shape)==2 else (1,-1,1,1)
        data=a.numpy().copy();w=gamma.numpy().copy();b=beta.numpy().copy()
        def value():
            mu=data.mean(axes,keepdims=True);var=data.var(axes,keepdims=True)
            return (((data-mu)/np.sqrt(var+1e-5)*w.reshape(view)+b.reshape(view))*upstream).sum()
        mu=data.mean(axes,keepdims=True);var=data.var(axes,keepdims=True);hat=(data-mu)/np.sqrt(var+1e-5)
        dx=w.reshape(view)/np.sqrt(var+1e-5)*(upstream-upstream.mean(axes,keepdims=True)-hat*(upstream*hat).mean(axes,keepdims=True))
        dw=(upstream*hat).sum(axes);db=upstream.sum(axes)
        selected=[(data,dx,c) for c in [tuple(0 for _ in shape),tuple(s-1 for s in shape)]]
        selected += [(w,dw,(i,)) for i in range(3)]+[(b,db,(i,)) for i in range(3)]
        for array,grad,coord in selected:
            old=array[coord];eps=1e-6;array[coord]=old+eps;high=value();array[coord]=old-eps;low=value();array[coord]=old
            finite.append(abs((high-low)/(2*eps)-grad[coord]))
    assert max(finite)<1e-7,finite
    return dict(parameters=sum(p.numel() for p in net.parameters()),input_shape=list(X.shape),errors=errors,
        finite_difference_coordinates=len(finite),finite_difference_step=1e-6,finite_difference_error=max(finite),
        scope='完整本节LeNet44878参数、独立NumPyConv/AvgPool/Sigmoid/BN/CE与移动状态；三次SGD批量5/2/3')


def paired():
    H['COMMON']['configure']('cpu')
    def run_pair(count,parts):
        manual=H['build_net']().double().train()
        native=H['build_net'](dict(H['BASE'],implementation='native')).double().train();native.load_state_dict(manual.state_dict())
        gen=torch.Generator().manual_seed(7508);X=torch.randn(count,1,28,28,generator=gen,dtype=torch.float64)
        y=torch.arange(count)%10
        if count==5:y=torch.tensor([0,1,4,7,9])
        opt1=torch.optim.SGD(manual.parameters(),lr=.1);opt2=torch.optim.SGD(native.parameters(),lr=.1)
        max_output,max_grad=0.,0.;steps=[]
        for ids in parts*3:
            opt1.zero_grad(set_to_none=True);opt2.zero_grad(set_to_none=True)
            a,b=manual(X[ids]),native(X[ids]);out_error=(a-b).abs().max().item();max_output=max(max_output,out_error)
            F.cross_entropy(a,y[ids]).backward();F.cross_entropy(b,y[ids]).backward()
            grad_error=max((p.grad-q.grad).abs().max().item() for p,q in zip(manual.parameters(),native.parameters()))
            max_grad=max(max_grad,grad_error);opt1.step();opt2.step()
            steps.append(dict(batch=len(ids),output_error=out_error,gradient_error=grad_error))
        state_error=max((manual.state_dict()[k]-v).abs().max().item() for k,v in native.state_dict().items())
        manual.eval();native.eval();eval_error=(manual(X)-native(X)).abs().max().item()
        return dict(updates=6,batches=[len(ids) for ids in parts]*3,max_training_output_error=max_output,max_gradient_error=max_grad,
                    max_state_error=state_error,eval_error=eval_error,steps=steps,dtype='float64 CPU, lr.1, same initial full state')
    regular=run_pair(9,([0,1,2,3],[4,5,6,7,8]))
    assert max(regular[k] for k in ('max_training_output_error','max_gradient_error','max_state_error','eval_error'))<1e-9,regular
    small=run_pair(5,([0,1],[2,3,4]))
    assert max(small[k] for k in ('max_training_output_error','max_gradient_error','max_state_error','eval_error'))<1e-4,small
    return dict(batch4_5=regular,batch2_3_sensitivity=small,
                note='小批量下相同数学公式的舍入差可经更新放大；记录原2/3实验，不声称长期逐位相同。4/5对齐保留1e-9严格界限')


def tails():
    batches=[(torch.tensor([i,i+1]),torch.tensor([i,i+1])) for i in range(0,10,2)]+[(torch.tensor([10]),torch.tensor([10]))]
    merged=list(H['train_batches'](batches));ids=torch.cat([p[1] for p in merged]).tolist()
    assert ids==list(range(11)) and [len(p[1]) for p in merged]==[2,2,2,2,3]
    return dict(samples=ids,batch_sizes=[2,2,2,2,3],no_drop_or_duplicate=True)


def practice():
    text=(ROOT/'docs/ch07/05_batch_norm.md').read_text(encoding='utf8');blocks=[]
    for page in text.split('<!-- pagebreak -->'):
        if re.search(r'## (20|21|22|23|24|25) ',page):blocks.extend(re.findall(r'```python\n(.*?)\n```',page.split('<!-- side -->')[0],re.S))
    assert len(blocks)==6
    path=ROOT/'.cache/7_5_follow.py';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n\n'.join(blocks)+'\n',encoding='utf8')
    output=io.StringIO()
    with contextlib.redirect_stdout(output):runpy.run_path(str(path),run_name='__main__')
    exercise=ROOT/'exercises/ch07/05_batch_norm.py';original=exercise.read_bytes();s=original.decode('utf8')
    assert len(re.findall(r'# TODO [1-8]：',s))==8
    fixes={'axes=None':'axes=(0,) if X.ndim==2 else (0,2,3)','mean=None':'mean=X.mean(axes,keepdim=True)',
        'var=None':'var=(X-mean).square().mean(axes,keepdim=True)','normalized=None':'normalized=(X-mean)/torch.sqrt(var+1e-5)',
        'out=None':'out=normalized*self.weight.reshape(shape)+self.bias.reshape(shape)',
        'loss=None':'loss=F.cross_entropy(logits,y)','pass  # TODO 7':'loss.backward()  # TODO 7','pass  # TODO 8':'optimizer.step()  # TODO 8'}
    for old,new in fixes.items():assert s.count(old)==1,old;s=s.replace(old,new)
    reference=io.StringIO()
    with contextlib.redirect_stdout(reference):exec(compile(s,str(exercise),'exec'),{'__name__':'__main__','__file__':str(exercise)})
    assert exercise.read_bytes()==original
    return dict(follow_blocks=6,follow_stdout=output.getvalue(),exercise_reference_stdout=reference.getvalue(),exercise_source_unchanged=True,unanswered_todos=8)


@torch.no_grad()
def checkpoints():
    device=H['COMMON']['configure']('auto');splits,_=H['COMMON']['load_data'](include_test=False)
    X=torch.stack([splits['validation'][i][0] for i in range(16)]);output={}
    for name in ('manual','native'):
        path=ROOT/'artifacts/ch07'/('7_5_'+name+'.pt')
        if not path.exists():output[name]=dict(skipped='尚无本地正式模型');continue
        state=torch.load(path,map_location='cpu',weights_only=True);net=H['build_net'](state['config']).eval();net.load_state_dict(state['model'])
        report=json.loads((ROOT/'reports/ch07'/('7_5_'+name+'_metrics.json')).read_text(encoding='utf8'));assert H['state_hash'](net)==report['final_state_sha256']
        cpu=net(X);before={k:v.clone() for k,v in net.named_buffers()}
        reference=net.double()(X.double());net=net.float().to(device);gpu=net(X.to(device)).cpu()
        with torch.backends.cudnn.flags(enabled=False,benchmark=False,deterministic=True,allow_tf32=False):
            fallback=net(X.to(device)).cpu()
        error=(cpu-gpu).abs().max().item()
        cpu_error=(cpu.double()-reference).abs().max().item();gpu_error=(gpu.double()-reference).abs().max().item()
        fallback_error=(fallback.double()-reference).abs().max().item()
        assert cpu_error<1e-4 and fallback_error<1e-4 and gpu_error<1e-3
        assert all(torch.equal(cpu.argmax(1),a.argmax(1)) for a in (gpu,reference,fallback))
        assert all(torch.equal(v.cpu(),before[k]) for k,v in net.named_buffers())
        output[name]=dict(max_logit_difference=error,predictions=cpu.argmax(1).tolist(),validation_positions=list(range(16)),
            all_state_hash_matches=True,eval_buffers_unchanged=True,bn_batches_tracked=int(net.bn1.num_batches_tracked),
            cpu_vs_float64=cpu_error,cudnn_vs_float64=gpu_error,cudnn_disabled_vs_float64=fallback_error,
            note='默认cuDNN差超过原1e-4后，以double及非cuDNN路径定位；默认路径界限1e-3，其余1e-4；16张验证图类别一致')
    return output


def main():
    path=ROOT/'reports/ch07/7_5_verification.json';r=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
    r.update(numerical=numerical(),paired=paired(),tail_merge=tails(),practice=practice(),checkpoint_recovery=checkpoints())
    H['write_json']('7_5_verification.json',r);print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
