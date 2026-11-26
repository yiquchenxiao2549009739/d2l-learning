"""独立 NumPy 重放 Adam 并核对序列索引、可见信息与跟做代码。"""
import contextlib
import csv
import io
import json
from pathlib import Path
import re
import runpy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
H = runpy.run_path(str(ROOT/'lessons/ch08/01_sequence_models.py'))


def numeric():
    torch.set_num_threads(1)
    _,_,observed = H['make_series']()
    # NumPy的索引和前反向独立于教师实现。仅复用同一个批顺序和初始参数。
    s = observed.numpy()
    X = np.stack([s[i:i+4] for i in range(600)])
    y = s[4:604,None]
    model = H['build_net'](4)
    params = [p.detach().numpy().copy() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters(),lr=.01,foreach=False)
    loader = DataLoader(TensorDataset(torch.arange(600)),batch_size=16,shuffle=True,
                        generator=torch.Generator().manual_seed(8103))

    def forward(a):
        w,b,v,c = params
        pre = a @ w.T+b
        hid = np.maximum(pre,0)
        return pre,hid,hid @ v.T+c

    first_moment = [np.zeros_like(p) for p in params]
    second_moment = [np.zeros_like(p) for p in params]
    changes = dict(pred=0.,gradient=0.,parameters=0.)
    crossed = 0
    mse = []
    for _ in range(5):
        seen = []
        for (idx,) in loader:
            seen.extend(idx.tolist())
            crossed += 1
            x,t = X[idx.numpy()],y[idx.numpy()]
            pre,hid,guess = forward(x)
            delta = 2*(guess-t)
            dh = (delta @ params[2])*(pre>0)
            grads = [dh.T@x, dh.sum(0), delta.T@hid, delta.sum(0)]
            opt.zero_grad(set_to_none=True)
            actual = model(torch.from_numpy(x))
            (actual-torch.from_numpy(t)).square().sum().backward()
            changes['pred'] = max(changes['pred'],float(np.abs(actual.detach().numpy()-guess).max()))
            for grad,p in zip(grads,model.parameters()):
                changes['gradient'] = max(changes['gradient'],float(np.abs(grad-p.grad.numpy()).max()))
            opt.step()
            for j,(arr,g,p) in enumerate(zip(params,grads,model.parameters())):
                m = first_moment[j] = .9*first_moment[j]+.1*g
                v = second_moment[j] = .999*second_moment[j]+.001*g*g
                arr -= .01*(m/(1-.9**crossed))/(np.sqrt(v/(1-.999**crossed))+1e-8)
                changes['parameters'] = max(changes['parameters'],float(np.abs(arr-p.detach().numpy()).max()))
        assert sorted(seen) == list(range(600)) and len(idx)==8
        mse.append(float(((forward(X)[2]-y)**2).mean()))
    assert crossed==190 and max(changes.values())<1e-9
    report = json.loads((ROOT/'reports/ch08/8_1_metrics.json').read_text(encoding='utf8'))
    assert np.allclose(mse,[r['frozen_mse'] for r in report['formal']['epoch_history']],atol=1e-12,rtol=0)
    # 对训练后的首十个坐标做中心差分，核对MSE梯度。
    pre,hid,z = forward(X[:11])
    delta = 2*(z-y[:11])/11
    dh = (delta @ params[2])*(pre>0)
    g = (dh.T@X[:11]).reshape(-1)
    w = params[0].reshape(-1)
    errors = []
    for i in range(10):
        val = w[i]
        w[i] = val+1e-6
        high = ((forward(X[:11])[2]-y[:11])**2).mean()
        w[i] = val-1e-6
        low = ((forward(X[:11])[2]-y[:11])**2).mean()
        w[i] = val
        errors.append(abs((high-low)/2e-6-g[i]))
    assert max(errors)<1e-8
    return dict(updates=crossed,tail_batch=8,maximum_differences=changes,frozen_epoch_mse=mse,
                finite_difference_coordinates=10,finite_difference_max_error=max(errors))


def alignment():
    t,clean,x = H['make_series']()
    W,Y,labels = H['windows'](x,4)
    assert W.shape==(996,4) and Y.shape==(996,1)
    assert torch.equal(W[0],x[:4]) and torch.equal(W[599],x[599:603])
    assert labels[599:601].tolist()==[604,605] and Y[599,0]==x[603] and Y[600,0]==x[604]
    for previous,end in [(604,802),(802,1000)]:
        origins = H['split_origins'](previous,end)
        target = H['targets'](t,origins,64)
        assert target.shape==(135,64)
        assert int(target[0,0])==previous+1 and int(target[-1,-1])==end
        assert torch.equal(H['gather'](t,origins,4)[:,-1].long(),origins)
    forecast = lambda z: z[:,-1:]+.1*z[:,-2:-1]
    origin = torch.tensor([604])
    altered = x.clone()
    altered[604:] += 100
    a = H['rollout'](forecast,H['gather'](x,origin,4),64)
    b = H['rollout'](forecast,H['gather'](altered,origin,4),64)
    assert torch.equal(a,b)
    a1 = forecast(H['gather'](x,origin+1,4))
    b1 = forecast(H['gather'](altered,origin+1,4))
    assert not torch.equal(a1,b1)
    seq = x.numpy()
    manual = seq[600:604].tolist()
    for _ in range(64):
        manual.append(manual[-1]+.1*manual[-2])
    assert np.array_equal(manual[4:],a.numpy()[0])
    probes = H['theory_probes'](x[:802],clean[:802])
    assert probes['noiseless_sine_max_abs_error']<1e-10
    assert np.allclose(probes['two_steps'],[[.7,.3],[.45,.55]])
    assert np.allclose(probes['reverse_conditional'],[[.56/.65,.09/.65],[.4,.6]])
    return dict(last_training_label=604,validation_origin_range=[604,738],test_origin_range=[802,936],
                future_tamper_changes_rolling_but_not_closed=True,numpy_forecast_matches=True,theory_matches=True)


def practice():
    text = (ROOT/'docs/ch08/01_sequence_models.md').read_text(encoding='utf8')
    blocks = []
    for page in text.split('<!-- pagebreak -->'):
        if re.search(r'## (16|17|18|19|20) ',page):
            blocks.extend(re.findall(r'```python\n(.*?)\n```',page.split('<!-- side -->')[0],re.S))
    assert len(blocks)==5
    path = ROOT/'.cache/8_1_follow.py'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n\n'.join(blocks)+'\n',encoding='utf8')
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        runpy.run_path(str(path),run_name='__main__')
    exercise = ROOT/'exercises/ch08/01_sequence_models.py'
    prior = exercise.read_bytes()
    code = prior.decode('utf8')
    refs = {'features = None':'features = observed.unfold(0,tau+1,1)[:,:tau]',
            'labels = None':'labels = observed[tau:].reshape(-1,1)',
            'train_count = None':'train_count = cutoff-tau','loss = None':'loss = (net(X)-y).square().sum()',
            'pass  # TODO 5':'loss.backward()  # TODO 5','pass  # TODO 6':'optimizer.step()  # TODO 6',
            'context = None':'context = observed[cutoff-tau:cutoff].reshape(1,-1)',
            'updated = None':'updated = torch.cat((context[:,1:],next_value),dim=1)'}
    for key,value in refs.items():
        assert code.count(key)==1
        code = code.replace(key,value)
    answers = io.StringIO()
    with contextlib.redirect_stdout(answers):
        exec(compile(code,str(exercise),'exec'),{'__file__':str(exercise),'__name__':'__main__'})
    assert exercise.read_bytes()==prior and len(re.findall(r'# TODO [1-8]：',prior.decode('utf8')))==8
    return dict(follow=stream.getvalue(),reference=answers.getvalue(),remaining_todos=8)


if __name__ == '__main__':
    result = dict(numeric=numeric(),alignment=alignment(),practice=practice())
    H['save_json']('8_1_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
