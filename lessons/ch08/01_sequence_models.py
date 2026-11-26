"""8.1 序列模型。固定教材实验、按时间划分、滚动及闭环多步预测。"""
import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/ch08'
SEED = 8101
TRAIN_END = 604
VAL_END = 802
TOTAL = 1000
HORIZONS = (1, 4, 16, 64)


def make_series(total=TOTAL, noise=.2, seed=SEED):
    time = torch.arange(1, total+1, dtype=torch.float64)
    clean = (time * .01).sin()
    generator = torch.Generator().manual_seed(seed)
    return time, clean, clean + noise * torch.randn(total, generator=generator, dtype=clean.dtype)


def windows(x, history):
    """输入为一维，返回按从远到近排列的X、列向量y、从1计的标签时刻。"""
    if x.ndim != 1 or not 1 <= history < len(x):
        raise ValueError('需要一维序列和合法窗口')
    rows = x.unfold(0, history+1, 1)
    return rows[:, :-1], rows[:, -1:], torch.arange(history+1, len(x)+1)


def encode(X, aggregation=False):
    if aggregation:
        # 16个时间位置压成4列。原始窗口仍随预测逐点滚动。
        return torch.cat((X[:, :-3].mean(dim=1, keepdim=True), X[:, -3:]), dim=1)
    return X


def build_net(width, hidden=(10,), outputs=1, seed=8102):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        sizes = (width, *hidden, outputs)
        layers = []
        for i, (ci, co) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = nn.Linear(ci, co, dtype=torch.float64)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i+1 < len(sizes)-1:
                layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def fit(net, X, y, epochs=5, batch=16, lr=.01):
    generator = torch.Generator().manual_seed(8103)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, foreach=False)
    rows = []
    updates = 0
    for epoch in range(1, epochs+1):
        net.train()
        total = 0.
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (net(features)-labels).square()
            total += loss.detach().sum().item()
            loss.sum().backward()
            optimizer.step()
            updates += 1
        net.eval()
        with torch.no_grad():
            frozen = (net(X)-y).square().mean().item()
        rows.append(dict(epoch=epoch, updates=updates, online_mse=total/y.numel(), frozen_mse=frozen))
    return rows


def gather(x, origins, history):
    """origin是已见最后时刻(从1计)，索引末列为origin-1。"""
    if x.ndim != 1 or origins.ndim != 1 or origins.dtype != torch.long or not len(origins):
        raise ValueError('需要一维序列和非空长整型起点')
    if not 1 <= history <= int(origins.min()) <= int(origins.max()) <= len(x):
        raise ValueError('起点必须拥有足够历史')
    return x[origins[:, None] - history + torch.arange(history)]


@torch.no_grad()
def rollout(predict, seed_windows, horizon):
    if seed_windows.ndim != 2 or seed_windows.shape[1] < 1 or horizon < 1:
        raise ValueError('需要二维窗口和正预测步数')
    current = seed_windows
    out = []
    for _ in range(horizon):
        Y = predict(current)
        if Y.shape != current[:, -1:].shape:
            raise ValueError('每个起点每次只应输出一个预测值')
        out.append(Y)
        current = torch.cat((current[:, 1:], Y), dim=1)
    return torch.cat(out, dim=1)


def targets(x, origins, horizon):
    """列0对应h=1，索引origin恰为下一时刻。"""
    if origins.ndim != 1 or origins.dtype != torch.long or not len(origins):
        raise ValueError('评价需要非空的一维长整型起点')
    if horizon < 1 or int(origins.min()) < 1 or int(origins.max())+horizon > len(x):
        raise ValueError('评价目标超出观测范围')
    return x[origins[:, None] + torch.arange(horizon)]


def metrics(predicted, observed, clean=None):
    assert predicted.shape == observed.shape
    error = predicted-observed
    result = dict(mse=error.square().mean().item(), mae=error.abs().mean().item(),
                  rmse=error.square().mean().sqrt().item(), count=error.numel())
    if clean is not None:
        result['clean_rmse'] = (predicted-clean).square().mean().sqrt().item()
    return result


def horizon_rows(pred, x, clean, origins):
    observed = targets(x, origins, pred.shape[1])
    underlying = targets(clean, origins, pred.shape[1])
    return [dict(horizon=h, **metrics(pred[:,h-1], observed[:,h-1], underlying[:,h-1]))
            for h in HORIZONS if h <= pred.shape[1]]


def split_origins(end_previous, end_current, horizon=64):
    return torch.arange(end_previous, end_current-horizon+1)


def ordinary_fit(x, history, hidden, earliest, aggregation=False):
    X, y, time = windows(x, history)
    take = (time >= earliest) & (time <= TRAIN_END)
    features = encode(X[take], aggregation)
    model = build_net(features.shape[1], hidden)
    trace = fit(model, features, y[take])
    return model, trace, int(take.sum())


def direct_fit(x, history=4, horizon=64):
    origins = torch.arange(history, TRAIN_END-horizon+1)
    net = build_net(history, outputs=horizon)
    # 此处输出64列，用元素平均的MSE作为训练目标。
    X, y = gather(x, origins, history), targets(x, origins, horizon)
    opt = torch.optim.Adam(net.parameters(), lr=.01, foreach=False)
    loader = DataLoader(TensorDataset(X,y), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(8103))
    for _ in range(5):
        for features, labels in loader:
            opt.zero_grad(set_to_none=True)
            (net(features)-labels).square().mean().backward()
            opt.step()
    return net.eval(), len(origins)


def fixed_models(x, clean):
    X, y, t = windows(x, 4)
    net, trace, count = ordinary_fit(x, 4, (10,), 5)
    with torch.no_grad():
        single = net(X).squeeze(1)
    test_mask = t > VAL_END
    val_mask = (t > TRAIN_END) & (t <= VAL_END)
    test_origins = split_origins(VAL_END, TOTAL)
    val_origins = split_origins(TRAIN_END, VAL_END)
    base = rollout(net, gather(x, test_origins, 4), 64)
    persistent = gather(x,test_origins,1).expand(-1,64)
    fit_X = torch.cat((X[t <= TRAIN_END],torch.ones(count,1,dtype=x.dtype)),dim=1)
    w = torch.linalg.lstsq(fit_X, y[t <= TRAIN_END]).solution
    def linear(z):
        return z @ w[:-1] + w[-1]
    linear_test = rollout(linear, gather(x,test_origins,4),64)
    fixed_start = rollout(net, gather(x,torch.tensor([TRAIN_END]),4), TOTAL-TRAIN_END).flatten()
    direct, direct_rows = direct_fit(x)
    with torch.no_grad():
        direct_val = direct(gather(x,val_origins,4))
        direct_test = direct(gather(x,test_origins,4))
    baseline_val = rollout(net, gather(x,val_origins,4),64)
    # 同一组起点上，一个模型的滚动单步h1与闭环第一步应一致。
    torch.testing.assert_close(base[:,0],single[test_origins-4])
    results = dict(config=dict(history=4,hidden=[10],epochs=5,batch_size=16,lr=.01,reduction='sum',dtype=str(x.dtype),
                               train_rows=count,validation_rows=int(val_mask.sum()),test_rows=int(test_mask.sum()),
                               parameters=sum(p.numel() for p in net.parameters()),
                               model_seed=8102,loader_seed=8103,series_seed=SEED,noise_std=.2),
                   epoch_history=trace,
                   one_step=dict(validation=metrics(single[val_mask],y[val_mask,0],clean[t[val_mask]-1]),
                                 test=metrics(single[test_mask],y[test_mask,0],clean[t[test_mask]-1])),
                   origins=dict(first_validation=int(val_origins[0]),last_validation=int(val_origins[-1]),
                                first_test=int(test_origins[0]),last_test=int(test_origins[-1]),count=len(test_origins)),
                   test_horizons=dict(mlp=horizon_rows(base,x,clean,test_origins),
                                      persistence=horizon_rows(persistent,x,clean,test_origins),
                                      linear_ar4=horizon_rows(linear_test,x,clean,test_origins),
                                      direct64=horizon_rows(direct_test,x,clean,test_origins)),
                   val_horizons=dict(mlp=horizon_rows(baseline_val,x,clean,val_origins),
                                     direct64=horizon_rows(direct_val,x,clean,val_origins)),
                   direct=dict(training_rows=direct_rows,first_origin=4,last_origin=540,
                               updates=5*((direct_rows+15)//16),loss='元素平均MSE',parameters=sum(p.numel() for p in direct.parameters())),
                   linear=dict(lag_weights=w[:-1,0].tolist(),bias=w[-1].item()),
                   retrospective_604=dict(rows=len(fixed_start),**metrics(fixed_start,x[TRAIN_END:],clean[TRAIN_END:])))
    predictions = dict(single=single, times=t, fixed_start=fixed_start, origin=test_origins,
                       mlp=base, persistence=persistent, linear_ar4=linear_test, direct64=direct_test)
    return results, predictions


def validation_experiment(x, clean):
    specs = [('A',4,(10,),False),('B',1,(10,),False),('C',2,(10,),False),
             ('D',8,(10,),False),('E',16,(10,),False),('F',16,(10,),True),
             ('G',4,(16,8),False),('H',4,(),False)]
    origins = split_origins(TRAIN_END, VAL_END)
    results = {}
    for name, history, hidden, agg in specs:
        net, trace, count = ordinary_fit(x[:TRAIN_END], history, hidden, 17, agg)
        result = rollout(lambda z: net(encode(z,agg)),gather(x[:VAL_END],origins,history),64)
        horizon = horizon_rows(result,x[:VAL_END],clean[:VAL_END],origins)
        results[name] = dict(history=history,hidden=list(hidden),aggregate_older=agg,features=4 if agg else history,
                             parameters=sum(p.numel() for p in net.parameters()),train_rows=count,
                             train_label_range=[17,TRAIN_END],epochs=trace,validation=horizon,
                             score=sum(r['mse'] for r in horizon)/len(horizon))
    best = min(results,key=lambda key:results[key]['score'])
    return dict(results=results,best=best,origin_range=[int(origins[0]),int(origins[-1])],
                selection='仅验证四个horizon的MSE等权平均',test_used=False)


def theory_probes(noisy, clean):
    c = 2*torch.cos(torch.tensor(.01,dtype=torch.float64))
    def recurrence(X):
        return c*X[:,-1:]-X[:,-2:-1]
    origin = torch.tensor([TRAIN_END])
    exact = rollout(recurrence,gather(clean,origin,2),64)
    perturbed = rollout(recurrence,gather(noisy,origin,2),64)
    actual = targets(clean,origin,64)
    transition = torch.tensor([[.8,.2],[.3,.7]],dtype=torch.float64)
    initial = torch.tensor([.7,.3],dtype=torch.float64)
    joint = initial[:,None]*transition
    posterior = (joint/joint.sum(0)).T
    return dict(noiseless_sine_max_abs_error=(exact-actual).abs().max().item(),
                noisy_seed_analytic_rmse=metrics(perturbed,actual)['rmse'],
                noisy_seed_analytic_h64_abs_error=abs(perturbed[0,-1].item()-actual[0,-1].item()),
                coefficient=c.item(),initial_state_probability=initial.tolist(),
                transition=transition.tolist(),two_steps=(transition@transition).tolist(),
                reverse_conditional=posterior.tolist(),
                scalar_error_10_steps={str(gain):.1*sum(gain**i for i in range(10)) for gain in (.5,1.,1.2)})


def save_json(name, data):
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def csv_file(path, header, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def record(result, predictions, series):
    t,clean,x = series
    formal = result['formal']
    save_json('8_1_metrics.json',result)
    csv_file(ROOT/'datasets/ch08/sequence_sine.csv',['time','clean','observed'],
             zip(t.to(torch.long).tolist(),clean.tolist(),x.tolist()))
    at = predictions['times'].tolist()
    all_preds = dict(zip(at,predictions['single'].tolist()))
    csv_file(REPORT/'8_1_trajectories.csv',['time','phase','clean','observed','rolling_one','fixed_origin_604'],
             [(i,'train' if i<=TRAIN_END else 'validation' if i<=VAL_END else 'test',clean[i-1].item(),x[i-1].item(),
               all_preds.get(i),predictions['fixed_start'][i-TRAIN_END-1].item() if i>TRAIN_END else None)
              for i in range(1,TOTAL+1)])
    csv_file(REPORT/'8_1_test_horizons.csv',['model','horizon','count','mse','mae','rmse','clean_rmse'],
             [(m,row['horizon'],row['count'],row['mse'],row['mae'],row['rmse'],row['clean_rmse'])
              for m,rows in formal['test_horizons'].items() for row in rows])
    csv_file(REPORT/'8_1_train.csv',['epoch','updates','online_mse','frozen_mse'],
             [(d['epoch'],d['updates'],d['online_mse'],d['frozen_mse']) for d in formal['epoch_history']])


def run():
    torch.set_num_threads(1)
    series = make_series()
    _, clean, x = series
    experiment = validation_experiment(x[:VAL_END],clean[:VAL_END])
    formal, predictions = fixed_models(x,clean)
    result = dict(formal=formal,validation_experiment=experiment,probes=theory_probes(x[:VAL_END],clean[:VAL_END]),
                  split=dict(train_end=TRAIN_END,validation_end=VAL_END,last_time=TOTAL))
    return result,predictions,series


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--guided',action='store_true',help='打印中文导读和关键结果')
    p.add_argument('--record',action='store_true',help='写出人工数据和实验报告')
    args = p.parse_args()
    result, predictions, series = run()
    if args.record:
        record(result,predictions,series)
    if args.guided:
        print('训练标签5..604，验证605..802，测试803..1000。')
        print('固定5轮更新',result['formal']['epoch_history'][-1]['updates'],'次。')
        print('滚动单步测试RMSE',f"{result['formal']['one_step']['test']['rmse']:.6f}")
        print('共同起点上的闭环RMSE',[(r['horizon'],round(r['rmse'],6)) for r in result['formal']['test_horizons']['mlp']])
        print('验证四跨度平均MSE最低候选',result['validation_experiment']['best'])
    else:
        json.dump(result,sys.stdout,ensure_ascii=False,indent=2,allow_nan=False)
        print()


if __name__ == '__main__':
    main()
