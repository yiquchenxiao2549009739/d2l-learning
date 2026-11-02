# -*- coding: utf-8 -*-
"""5.3整节：LazyLinear的形状推断、首次前向、初始化及训练。

复用现有openmmlab，CPU/float64、人工数据，无安装下载。
"""
import copy
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import UninitializedParameter

REPORT = ROOT / 'reports/ch05'
FIGURES = REPORT / 'figures'
DTYPE = torch.float64


def unresolved(net):
    return sum(isinstance(p, UninitializedParameter) for p in net.parameters())


def inspect_parameters(net, model, stage):
    rows = []
    for name, p in net.named_parameters():
        unknown = isinstance(p, UninitializedParameter)
        rows.append(dict(model=model, stage=stage, name=name, type=type(p).__name__,
                         shape='unknown' if unknown else str(list(p.shape)),
                         elements=None if unknown else p.numel(), dtype=str(p.dtype),
                         device=str(p.device), requires_grad=p.requires_grad))
    return rows


def warm_up(net, example):
    # 本例无BN/Dropout；显式eval并恢复原模式，不把no_grad当成eval。
    previous_mode = net.training
    net.eval()
    with torch.no_grad():
        output = net(example)
    net.train(previous_mode)
    return output


def materialization_demo():
    torch.manual_seed(5301)
    configurations = {
        'eager': nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10)),
        'mixed': nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.Linear(256, 10)),
        'all_lazy': nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))}
    rows, summaries, states = [], [], {}
    for name, net in configurations.items():
        net.double()
        before = inspect_parameters(net, name, 'constructed')
        rows.extend(before)
        identity = [id(p) for p in net.parameters()]
        handles = []
        if name == 'all_lazy':
            for index in (0, 2):
                handles.append(net[index].register_forward_hook(
                    lambda module, args, output, i=index:
                    rows.extend(inspect_parameters(net, name, 'after_layer_' + str(i)))))
        output = warm_up(net, torch.ones(2, 20, dtype=DTYPE))
        for handle in handles:handle.remove()
        rows.extend(inspect_parameters(net, name, 'after_forward'))
        states[name] = copy.deepcopy(net.state_dict())
        summaries.append(dict(model=name, unknown_before=sum(r['elements'] is None for r in before),
                              known_elements_before=sum(r['elements'] or 0 for r in before),
                              unknown_after=unresolved(net), elements_after=sum(p.numel() for p in net.parameters()),
                              output_shape=list(output.shape),
                              classes_after=[type(net[i]).__name__ for i in (0, 2)],
                              parameter_identity_preserved=identity == [id(p) for p in net.parameters()],
                              no_gradients_after_warm_up=all(p.grad is None for p in net.parameters())))
    return summaries, rows, states


def expected_error(label, operation):
    try:
        operation()
    except (RuntimeError, ValueError) as error:
        return dict(case=label, error_type=type(error).__name__, message=str(error).splitlines()[0][:220])
    raise AssertionError('预期错误未发生：' + label)


def shapes_and_order_demo():
    fresh = nn.LazyLinear(8).double()
    errors = [expected_error('shape_before_materialization', lambda: fresh.weight.shape),
              expected_error('numel_before_materialization', lambda: fresh.weight.numel()),
              expected_error('normal_init_too_early', lambda: nn.init.normal_(fresh.weight))]
    warm_up(fresh, torch.zeros(2, 20, dtype=DTYPE))
    before = copy.deepcopy(fresh.state_dict())
    batch_shape = list(fresh(torch.zeros(7, 20, dtype=DTYPE)).shape)
    leading_shape = list(fresh(torch.zeros(2, 3, 20, dtype=DTYPE)).shape)
    unchanged = all(torch.equal(value, fresh.state_dict()[name]) for name, value in before.items())
    errors.append(expected_error('changed_feature_width', lambda: fresh(torch.zeros(2, 21, dtype=DTYPE))))
    bad = nn.Sequential(nn.Linear(20, 8), nn.ReLU(), nn.Linear(7, 1)).double()
    errors.append(expected_error('explicit_layer_mismatch', lambda: bad(torch.zeros(2, 20, dtype=DTYPE))))
    torch.manual_seed(5302)
    eager = nn.Sequential(nn.Linear(20, 8, dtype=DTYPE), nn.ReLU(), nn.Linear(8, 1, dtype=DTYPE))
    torch.manual_seed(5302)
    mixed = nn.Sequential(nn.LazyLinear(8, dtype=DTYPE), nn.ReLU(), nn.Linear(8, 1, dtype=DTYPE))
    warm_up(mixed, torch.zeros(2, 20, dtype=DTYPE))
    random_difference = max((a-b).abs().max().item() for a,b in zip(eager.parameters(), mixed.parameters()))
    mixed.load_state_dict(eager.state_dict())
    X = torch.linspace(-1, 1, 40, dtype=DTYPE).reshape(2, 20)
    copied_error = (mixed(X)-eager(X)).abs().max().item()
    return dict(changed_batch_output_shape=batch_shape, leading_dimensions_output_shape=leading_shape,
                repeated_forward_preserves_values=unchanged, fixed_in_features=fresh.in_features,
                same_seed_mixed_vs_eager_parameter_difference=random_difference,
                copied_state_output_max_error=copied_error), errors


class TwoWidthModel(nn.Module):
    """输入宽度不同时用各自投影，宽度统一后共享输出层。"""
    def __init__(self):
        super().__init__()
        self.from_two = nn.Linear(2, 8)
        self.from_three = nn.Linear(3, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, X):
        if X.shape[-1] == 2: H = self.from_two(X)
        elif X.shape[-1] == 3: H = self.from_three(X)
        else: raise ValueError('本例只定义2维或3维输入的投影。')
        return self.out(F.relu(H))


def adapter_demo():
    torch.manual_seed(5303)
    net = TwoWidthModel().double()
    left = net(torch.ones(4, 2, dtype=DTYPE))
    right = net(torch.ones(4, 3, dtype=DTYPE))
    left.square().mean().backward()
    first = net.out.weight.grad.clone();net.zero_grad(set_to_none=True)
    right.square().mean().backward()
    second = net.out.weight.grad.clone();net.zero_grad(set_to_none=True)
    (net(torch.ones(4, 2, dtype=DTYPE)).square().mean() +
     net(torch.ones(4, 3, dtype=DTYPE)).square().mean()).backward()
    return dict(parameter_count=sum(p.numel() for p in net.parameters()), left_shape=list(left.shape),
                right_shape=list(right.shape), shared_head_gradient_sum_max_error=
                (net.out.weight.grad-first-second).abs().max().item())


def make_data():
    rng = torch.Generator().manual_seed(5310)
    data = {}
    for name, count in [('train', 64), ('validation', 32), ('test', 32)]:
        X = 4 * torch.rand(count, 2, generator=rng, dtype=DTYPE) - 2
        data[name] = (X, X[:, :1].square() + .5 * X[:, 1:])
    return data


def training_demo():
    torch.manual_seed(5311)
    lazy = nn.Sequential(nn.LazyLinear(16), nn.ReLU(), nn.LazyLinear(1)).double()
    data = make_data();X, y = data['train']
    warm_up(lazy, torch.zeros(2, 2, dtype=DTYPE))
    nn.init.kaiming_normal_(lazy[0].weight, nonlinearity='relu')
    nn.init.zeros_(lazy[0].bias)
    nn.init.xavier_normal_(lazy[2].weight, gain=1)
    nn.init.zeros_(lazy[2].bias)
    initial = copy.deepcopy(lazy.state_dict())
    eager = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)).double()
    eager.load_state_dict(initial)
    models = [lazy, eager]
    optimizers = [torch.optim.SGD(net.parameters(), lr=.03) for net in models]
    history, max_gradient_error, max_parameter_error = [], 0., 0.
    for step in range(201):
        with torch.no_grad():
            train_losses = [F.mse_loss(net(X), y).item() for net in models]
            vx, vy = data['validation']
            validation = F.mse_loss(lazy(vx), vy).item()
        history.append(dict(step=step, lazy_train_mse=train_losses[0], eager_train_mse=train_losses[1],
                            validation_mse=validation))
        if step == 200:break
        for net, optimizer in zip(models, optimizers):
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(net(X), y).backward()
        max_gradient_error = max(max_gradient_error, max((a.grad-b.grad).abs().max().item()
                                  for a,b in zip(lazy.parameters(), eager.parameters())))
        for optimizer in optimizers:optimizer.step()
        max_parameter_error = max(max_parameter_error, max((a-b).abs().max().item()
                                   for a,b in zip(lazy.parameters(), eager.parameters())))
    lazy.eval()
    with torch.no_grad():
        tx, ty = data['test'];prediction = lazy(tx);test_mse = F.mse_loss(prediction, ty).item()
    info = dict(parameter_count=sum(p.numel() for p in lazy.parameters()), updates_per_model=200,
                learning_rate=.03, train_count=64, validation_count=32, test_count=32,
                initial_train_mse=history[0]['lazy_train_mse'], final_train_mse=history[-1]['lazy_train_mse'],
                final_validation_mse=history[-1]['validation_mse'], test_mse=test_mse,
                lazy_vs_eager_gradient_max_error=max_gradient_error,
                lazy_vs_eager_parameter_max_error=max_parameter_error,
                lazy_vs_eager_loss_max_error=max(abs(r['lazy_train_mse']-r['eager_train_mse']) for r in history))
    return info, history, dict(initial=initial, final=copy.deepcopy(lazy.state_dict()), data=data, prediction=prediction)


def draw_figures(summary, history):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False,
                         'font.size': 11, 'figure.dpi': 145})
    teal, gold, navy = '#147e83', '#d58a30', '#173448'
    fig,ax=plt.subplots(figsize=(9,3.8));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,5))
    for x,text in [(1.5,'定义LazyLinear(256)\n输入宽度未知\n参数是占位对象'),
                   (5,'首次输入(n,20)\n确定weight(256,20)\n产生默认初值'),
                   (8.5,'以后输入(m,20)\n复用同一组参数\n反传后优化器更新')]:
        ax.text(x,2.6,text,ha='center',va='center',linespacing=1.7,bbox=dict(boxstyle='round,pad=.6',fc='#eff7f8',ec=teal))
    for x in (3,6.5):ax.annotate('',xy=(x+.6,2.6),xytext=(x,2.6),arrowprops=dict(arrowstyle='->',color=navy,lw=1.7))
    ax.set_title('确定形状、填充初值、训练更新是不同阶段')
    fig.tight_layout();fig.savefig(FIGURES/'5_3_lifecycle.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    labels=['全部Linear','首层Lazy','两层Lazy']
    axes[0].bar(labels,[r['known_elements_before'] for r in summary],color=teal)
    axes[0].set(title='首次前向之前：已确定的元素数',ylabel='已确定元素数')
    axes[1].bar(labels,[r['elements_after'] for r in summary],color=gold)
    axes[1].set(title='首次前向之后：均为7946参数',ylabel='参数元素数')
    for ax,values in zip(axes,[[r['known_elements_before'] for r in summary],[7946]*3]):
        for i,value in enumerate(values):ax.text(i,value+120,str(value),ha='center')
        ax.set_ylim(0,9000)
    fig.tight_layout();fig.savefig(FIGURES/'5_3_materialization.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.3))
    ax.semilogy([r['step'] for r in history],[r['lazy_train_mse'] for r in history],label='Lazy构建后训练',color=teal)
    ax.semilogy([r['step'] for r in history],[r['eager_train_mse'] for r in history],ls='--',label='Linear同起点训练（重合）',color=navy)
    ax.semilogy([r['step'] for r in history],[r['validation_mse'] for r in history],label='验证32条',color=gold)
    ax.set(xlabel='全批量SGD更新次数',ylabel='MSE（对数坐标）',title='2→16→1，65参数；延后初始化完成后按普通模型训练')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIGURES/'5_3_training.png');plt.close(fig)


def write_csv(name, rows):
    with (REPORT/name).open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    torch.set_num_threads(1);REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    summary,rows,states=materialization_demo()
    shapes,errors=shapes_and_order_demo();adapters=adapter_demo()
    training,history,train_state=training_demo()
    metrics=dict(section='5.3',date='2026-11-02',torch_version=torch.__version__,device='cpu',dtype=str(DTYPE),
                 configurations=summary,shapes_and_order=shapes,adapters=adapters,training=training)
    write_csv('5_3_parameters.csv',rows);write_csv('5_3_expected_errors.csv',errors);write_csv('5_3_epochs.csv',history)
    path=ROOT/'artifacts/ch05/5_3_states.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(configurations=states,training=train_state),path)
    draw_figures(summary,history)
    (REPORT/'5_3_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    output='\n'.join(['5.3 延后初始化（CPU/float64，无安装下载）',
        '首次前向前后：'+json.dumps(summary,ensure_ascii=False),
        '形状与随机顺序：'+json.dumps(shapes,ensure_ascii=False),
        '不同输入的投影/共享：'+json.dumps(adapters,ensure_ascii=False),
        '训练与同起点核对：'+json.dumps(training,ensure_ascii=False),
        '五种预期错误已记录；综合练习保持留空。'])
    (REPORT/'5_3_demo_output.txt').write_text(output+'\n',encoding='utf8');print(output)


if __name__=='__main__':main()
