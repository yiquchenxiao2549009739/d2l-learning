"""逐字段核对 DenseNet 前反向、重算、教材跟做和检查点。"""
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

ROOT = Path(__file__).resolve().parents[1]
D = runpy.run_path(str(ROOT/'lessons/ch07/07_densenet.py'))
OLD = runpy.run_path(str(ROOT/'tools/verify_batch_norm.py'))


class DenseGraph(OLD['BNGraph']):
    def module(self,m,x,name=''):
        if isinstance(m,D['DenseBlock']):
            for i,unit in enumerate(m.units):
                y = self.module(unit,x,f'{name}.units.{i}')
                split = x.data.shape[1]
                x = self.operation(np.concatenate((x.data,y.data),axis=1),[x,y],
                                   lambda grad, c=split: [grad[:,:c],grad[:,c:]])
            return x
        return super().module(m,x,name)


def examples():
    x = torch.tensor([[1.,2.]],requires_grad=True)
    y1 = 2*x
    y2 = x+3*y1
    joined = torch.cat((x,y1,y2),dim=1)
    joined.sum().backward()
    assert torch.equal(x.grad,torch.full_like(x,10))
    with torch.no_grad():
        out = D['DenseBlock'](2,3,10).eval()(torch.zeros(4,3,8,8))
        reduced = D['transition'](23,10).eval()(out)
    assert list(out.shape)==[4,23,8,8] and list(reduced.shape)==[4,10,4,4]
    # 先用meta验证最小尺寸及前一个尺寸的差别。
    result = D['inspect_net'](dict(D['BASE'],size=29))
    assert result['outer'][-1]['shape']==[1,10]
    try:
        D['inspect_net'](dict(D['BASE'],size=28))
    except RuntimeError as e:
        assert 'Output size is too small' in str(e)
    else:
        raise AssertionError('28应在第三次过渡池化失败')
    return dict(joined=joined.detach().tolist(),grad=x.grad.tolist(),book_block=list(out.shape),
                book_transition=list(reduced.shape),min_square_eval_input=29)


def numeric():
    D['COMMON']['configure']('cpu')
    cfg = dict(D['BASE'],stem=2,growth=2,blocks=[2,2,2,2],size=32)
    net = D['build_net'](cfg).double().train()
    X = torch.rand(3,1,32,32,dtype=torch.float64,generator=torch.Generator().manual_seed(7706))
    X.requires_grad_()
    y = torch.tensor([1,4,8])
    params = {k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
    logits = net(X)
    ce = F.cross_entropy(logits,y)
    ce.backward()
    nce,nlogits,g,gx = DenseGraph(params).run(net,X.detach().numpy(),y.numpy())
    errors = dict(logits=float(np.abs(nlogits-logits.detach().numpy()).max()),ce=abs(nce-ce.item()),
                  parameter_grad=max(float(np.abs(g[k]-v.grad.numpy()).max()) for k,v in net.named_parameters()),
                  input_grad=float(np.abs(gx-X.grad.numpy()).max()))
    opt = torch.optim.SGD(net.parameters(),lr=.01)
    opt.step()
    for k,v in net.named_parameters():
        params[k] -= .01*g[k]
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(net(X.detach()[1:]),y[1:]).backward()
    opt.step()
    _,_,g,_ = DenseGraph(params).run(net,X.detach().numpy()[1:],y.numpy()[1:])
    for k,v in net.named_parameters():
        params[k] -= .01*g[k]
    errors['parameters_after_two_sgd'] = max(float(np.abs(params[k]-v.detach().numpy()).max()) for k,v in net.named_parameters())
    errors['buffers_after_two_sgd'] = max(float(np.abs(params[k]-v.detach().numpy()).max()) for k,v in net.named_buffers())
    net.eval()
    nce,nlogits = DenseGraph(params).run(net,X.detach().numpy(),y.numpy(),False)
    errors['eval_logits'] = float(np.abs(nlogits-net(X).detach().numpy()).max())
    assert max(errors.values()) < 1e-8,errors
    # 单独对两类单元做中心差分，避开整网ReLU/MaxPool折点。
    probes = {}
    for bottleneck in (False,True):
        block = nn.Sequential(D['DenseBlock'](2,2,2,bottleneck),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(6,3)).double().eval()
        gen = torch.Generator().manual_seed(7707)
        with torch.no_grad():
            for m in block.modules():
                if isinstance(m,(nn.Conv2d,nn.Linear)):
                    m.weight.copy_(torch.randn(m.weight.shape,generator=gen,dtype=torch.float64)*.1)
                    if m.bias is not None:
                        m.bias.fill_(.2)
        x = torch.rand(2,2,4,4,generator=gen,dtype=torch.float64).requires_grad_()
        t = torch.tensor([0,2])
        p = {k:v.detach().numpy().copy() for k,v in block.state_dict().items()}
        F.cross_entropy(block(x),t).backward()
        _,_,gg,ggx = DenseGraph(p).run(block,x.detach().numpy(),t.numpy())
        grad_diff = max(float(np.abs(gg[k]-v.grad.numpy()).max()) for k,v in block.named_parameters())
        finite = []
        for k,_ in block.named_parameters():
            index = np.unravel_index(np.abs(gg[k]).argmax(),p[k].shape)
            before = p[k][index]
            p[k][index] = before+1e-6
            high,_ = DenseGraph(p).run(block,x.detach().numpy(),t.numpy(),False)
            p[k][index] = before-1e-6
            low,_ = DenseGraph(p).run(block,x.detach().numpy(),t.numpy(),False)
            p[k][index] = before
            finite.append(abs((high-low)/2e-6-gg[k][index]))
        assert max(grad_diff,max(finite)) < 1e-7
        probes[str(bottleneck)] = dict(grad_error=grad_diff,input_error=float(np.abs(ggx-x.grad.numpy()).max()),
                                      finite_coordinates=len(finite),finite_max_error=max(finite))
    return dict(parameters=sum(p.numel() for p in net.parameters()),input_shape=list(X.shape),errors=errors,probes=probes)


def recompute():
    cfg = dict(D['BASE'],stem=4,growth=3,blocks=[2,2],size=24,bottleneck=True)
    a = D['build_net'](cfg).double()
    b = D['build_net'](dict(cfg,recompute=True)).double()
    b.load_state_dict(a.state_dict())
    snapshots = {k:v for k,v in b.named_buffers()}
    opts = [torch.optim.SGD(net.parameters(),lr=.02) for net in (a,b)]
    gen = torch.Generator().manual_seed(7708)
    cases = []
    for size in (3,2,3):
        X = torch.rand(size,1,24,24,generator=gen,dtype=torch.float64)
        y = torch.arange(size)
        outputs, inputs = [], []
        for net,opt in zip((a,b),opts):
            opt.zero_grad(set_to_none=True)
            x = X.clone().requires_grad_()
            out = net(x)
            F.cross_entropy(out,y).backward()
            opt.step()
            outputs.append(out.detach())
            inputs.append(x.grad)
        cases.append(dict(logits=(outputs[0]-outputs[1]).abs().max().item(),
                          input_grad=(inputs[0]-inputs[1]).abs().max().item(),
                          parameter_grad=max((p.grad-q.grad).abs().max().item() for p,q in zip(a.parameters(),b.parameters())),
                          state=max((p-q).abs().max().item() for p,q in zip(a.state_dict().values(),b.state_dict().values()))))
    assert max(v for case in cases for v in case.values()) < 1e-9
    assert all(snapshots[k] is v for k,v in b.named_buffers())
    counters = sorted({int(m.num_batches_tracked) for m in b.modules() if isinstance(m,nn.BatchNorm2d)})
    assert counters == [3]
    return dict(batches=[3,2,3],cases=cases,bn_counters=counters,buffer_objects_restored=True)


def practice():
    text = (ROOT/'docs/ch07/07_densenet.md').read_text(encoding='utf8')
    blocks = []
    for page in text.split('<!-- pagebreak -->'):
        if re.search(r'## (17|18|19|20|21) ',page):
            blocks.extend(re.findall(r'```python\n(.*?)\n```',page.split('<!-- side -->')[0],re.S))
    assert len(blocks) == 5
    path = ROOT/'.cache/7_7_follow.py'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n\n'.join(blocks)+'\n',encoding='utf8')
    followed = io.StringIO()
    with contextlib.redirect_stdout(followed):
        runpy.run_path(str(path),run_name='__main__')
    exercise = ROOT/'exercises/ch07/07_densenet.py'
    old = exercise.read_bytes()
    source = old.decode('utf8')
    assert len(re.findall(r'# TODO [1-8]：',source)) == 8
    updates = {'width = None':'width = ci + i * k','Y = None':'Y = unit(X)',
               'joined = None':'joined = torch.cat((X, Y), dim=1)',
               'projection = None':'projection = nn.Conv2d(ci,co,1)','pooling = None':'pooling = nn.AvgPool2d(2,2)',
               'loss = None':'loss = F.cross_entropy(logits,y)','pass  # TODO 7':'loss.backward()  # TODO 7',
               'pass  # TODO 8':'opt.step()  # TODO 8'}
    for key,value in updates.items():
        assert source.count(key) == 1,key
        source = source.replace(key,value)
    ref = io.StringIO()
    with contextlib.redirect_stdout(ref):
        exec(compile(source,str(exercise),'exec'),{'__file__':str(exercise),'__name__':'__main__'})
    assert exercise.read_bytes() == old
    return dict(follow=followed.getvalue(),exercise_reference=ref.getvalue(),unanswered=8)


@torch.no_grad()
def recovery():
    path = ROOT/'artifacts/ch07/7_7.pt'
    if not path.is_file():
        return dict(skipped='未找到本地正式权重')
    state = torch.load(path,map_location='cpu',weights_only=True)
    model = D['build_net'](state['config']).eval()
    model.load_state_dict(state['model'])
    report = json.loads((ROOT/'reports/ch07/7_7_metrics.json').read_text(encoding='utf8'))
    assert D['RES']['state_hash'](model) == report['final_state_sha256']
    data,_ = D['COMMON']['load_data'](include_test=False)
    device = D['COMMON']['configure']('auto')
    X = torch.stack([data['validation'][i][0] for i in range(16)])
    X = D['RES']['prepare'](X,state['config'],'cpu')
    logits = model(X)
    ref = model.double()(X.double())
    model.float().to(device)
    result = model(X.to(device)).cpu()
    error = (logits-result).abs().max().item()
    assert error < 1e-3 and torch.equal(logits.argmax(1),result.argmax(1))
    assert D['RES']['state_hash'](model) == report['final_state_sha256']
    return dict(images=16,device=str(device),cpu_device_max_error=error,
                cpu_float64_error=(logits-ref).abs().max().item(),
                device_float64_error=(result-ref).abs().max().item(),state_hash_matches=True,class_predictions_match=True)


if __name__ == '__main__':
    result = dict(numeric=numeric(),examples=examples(),recompute=recompute(),practice=practice(),recovery=recovery())
    D['write_json']('7_7_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
