"""7.3 独立NumPy小通道核对、逐段跟做、内存练习参考与检查点验证。

不重跑正式训练；不改变练习源文件。GAP在独立NumPy引擎中用固定平均矩阵实现。
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
H=runpy.run_path(str(ROOT/'lessons/ch07/03_nin.py'))
ENGINE=runpy.run_path(str(ROOT/'tools/verify_alexnet.py'))['numpy_pass']


def numerical():
    H['COMMON']['configure']('cpu')
    cfg=dict(H['BASE'],channels=[2,3,4],dropout=0.)
    net=H['build_net'](cfg).double().eval()
    # 仅数值探针使用正bias，减少ReLU/最大汇聚切换对中心差分的干扰。
    for m in net.modules():
        if isinstance(m,nn.Conv2d):nn.init.constant_(m.bias,.05)
    leaves=[m for m in net.modules() if not list(m.children())]
    before_gap=leaves[:next(i for i,m in enumerate(leaves) if isinstance(m,nn.AdaptiveAvgPool2d))]
    average=nn.Linear(250,10).double()
    with torch.no_grad():
        average.weight.zero_();average.bias.zero_()
        for k in range(10):average.weight[k,k*25:(k+1)*25]=1/25
    average.requires_grad_(False)
    surrogate=nn.Sequential(*before_gap,nn.Flatten(),average)
    params={k:v.detach().numpy().copy() for k,v in surrogate.state_dict().items()}
    gen=torch.Generator().manual_seed(7307)
    X=torch.rand(3,1,224,224,generator=gen,dtype=torch.float64).requires_grad_()
    labels=torch.tensor([1,4,7])
    logits=net(X);ce=F.cross_entropy(logits,labels);ce.backward()
    loss_np,logits_np,gradients,gx=ENGINE(surrogate,params,X.detach().numpy(),labels.numpy())
    trainable={k:v for k,v in surrogate.named_parameters() if v.requires_grad}
    errors=dict(logits=float(np.max(np.abs(logits_np-logits.detach().numpy()))),ce=abs(loss_np-ce.item()),
        input_gradient=float(np.max(np.abs(gx-X.grad.numpy()))),
        parameter_gradient=max(float(np.max(np.abs(gradients[k]-v.grad.numpy()))) for k,v in trainable.items()))
    finite=[]
    for key in trainable:
        coord=np.unravel_index(np.argmax(np.abs(gradients[key])),params[key].shape)
        old,eps=params[key][coord],1e-6
        params[key][coord]=old+eps
        high=ENGINE(surrogate,params,X.detach().numpy(),labels.numpy(),False)[0]
        params[key][coord]=old-eps
        low=ENGINE(surrogate,params,X.detach().numpy(),labels.numpy(),False)[0]
        params[key][coord]=old
        finite.append(abs((high-low)/(2*eps)-gradients[key][coord]))
    errors['central_difference_24_coordinates']=max(finite)
    optimizer=torch.optim.SGD(net.parameters(),lr=.1)
    for ids in ([0,1],[2]):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(net(X.detach()[ids]),labels[ids]).backward();optimizer.step()
        grads=ENGINE(surrogate,params,X.detach().numpy()[ids],labels.numpy()[ids])[2]
        for key in trainable:params[key]-=.1*grads[key]
    errors['sgd_two_updates_tail1']=max(float(np.max(np.abs(params[k]-v.detach().numpy()))) for k,v in trainable.items())
    assert max(errors.values())<1e-7,errors
    return dict(parameters=sum(p.numel() for p in net.parameters()),conv_count=12,input_shape=list(X.shape),errors=errors,
        finite_difference_step=1e-6,numerical_probe_bias=.05,
        gap_numpy='fixed block-diagonal 1/25 averaging matrix, no trainable averaging parameters',
        scope='224 geometry; channels2/3/4 and ten-class maps; independent NumPy engine from7.1; not all full-model gradients')


def practice():
    text=(ROOT/'docs/ch07/03_nin.md').read_text(encoding='utf8')
    blocks=[]
    for page in text.split('<!-- pagebreak -->'):
        if re.search(r'## (13|14|15|16|17) ',page):
            blocks.extend(re.findall(r'```python\n(.*?)\n```',page.split('<!-- side -->')[0],re.S))
    assert len(blocks)==5
    path=ROOT/'.cache/7_3_follow.py';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n\n'.join(blocks)+'\n',encoding='utf8')
    output=io.StringIO()
    with contextlib.redirect_stdout(output):runpy.run_path(str(path),run_name='__main__')
    exercise=ROOT/'exercises/ch07/03_nin.py';original=exercise.read_bytes();source=original.decode('utf8')
    assert len(re.findall(r'# TODO [1-8]：',source))==8
    substitutions={'spatial=None':'spatial=nn.Conv2d(in_channels,out_channels,3,padding=1)',
        'point1=None':'point1=nn.Conv2d(out_channels,out_channels,1)',
        'point2=None':'point2=nn.Conv2d(out_channels,out_channels,1)',
        'gap=None':'gap=nn.AdaptiveAvgPool2d((1,1))','flatten=None':'flatten=nn.Flatten()',
        'loss=None':'loss=F.cross_entropy(logits,y)',
        'pass  # TODO 7':'loss.backward()  # TODO 7','pass  # TODO 8':'optimizer.step()  # TODO 8'}
    for old,new in substitutions.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    reference=io.StringIO()
    with contextlib.redirect_stdout(reference):
        exec(compile(source,str(exercise),'exec'),{'__name__':'__main__','__file__':str(exercise)})
    assert exercise.read_bytes()==original
    return dict(follow_blocks=5,follow_stdout=output.getvalue(),exercise_reference_stdout=reference.getvalue(),
        exercise_source_unchanged=True,exercise_unanswered_todos=8)


@torch.no_grad()
def checkpoints():
    path=ROOT/'artifacts/ch07/7_3.pt'
    if not path.exists():return dict(skipped='尚无本地正式检查点')
    device=H['COMMON']['configure']('auto')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    net=H['build_net'](checkpoint['config']).eval();net.load_state_dict(checkpoint['model'])
    report=json.loads((ROOT/'reports/ch07/7_3_metrics.json').read_text(encoding='utf8'))
    assert H['parameter_hash'](net)==report['final_parameter_sha256']
    splits,_=H['COMMON']['load_data'](include_test=False)
    X=torch.stack([splits['validation'][i][0] for i in range(2)])
    X=H['prepare'](X,checkpoint['config'],torch.device('cpu'))
    cpu=net(X);net=net.to(device);gpu=net(X.to(device)).cpu()
    difference=(cpu-gpu).abs().max().item()
    assert difference<1e-4 and torch.equal(cpu.argmax(1),gpu.argmax(1))
    current={}
    handle=net.block4.register_forward_hook(lambda m,a,o:current.update(maps=o))
    confusion=torch.zeros(10,10,dtype=torch.int64)
    positive=torch.zeros(10,dtype=torch.int64);zero=torch.zeros(10,dtype=torch.int64)
    maxima=torch.full((10,),-float('inf'));total_positions=0
    for X,y in H['COMMON']['make_loader'](splits['validation'],batch_size=128):
        logits=net(H['prepare'](X,checkpoint['config'],device)).cpu()
        maps=current['maps']
        positive+=(maps>0).sum((0,2,3)).cpu();total_positions+=len(y)*maps.shape[-1]*maps.shape[-2]
        zero+=(logits==0).sum(0);maxima=torch.maximum(maxima,logits.max(0).values)
        confusion+=torch.bincount(y*10+logits.argmax(1),minlength=100).reshape(10,10)
    handle.remove()
    diagnostics=dict(dataset='validation only, no test reread',samples=6000,confusion=confusion.tolist(),
        map_positive_fraction=(positive/total_positions).tolist(),zero_logits_samples=zero.tolist(),
        maximum_logit_by_class=maxima.tolist(),all_zero_response_classes=torch.where(positive==0)[0].tolist())
    assert confusion.trace().item()==report['history'][-1]['validation_correct']
    H['write_json']('7_3_diagnostics.json',diagnostics)
    return dict(max_logit_difference=difference,predictions=cpu.argmax(1).tolist(),
        parameter_hash_matches=True,validation_positions=[0,1],diagnostics=diagnostics)


def main():
    path=ROOT/'reports/ch07/7_3_verification.json'
    result=json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
    result.update(numerical=numerical(),practice=practice(),checkpoint_recovery=checkpoints())
    H['write_json']('7_3_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
