# -*- coding: utf-8 -*-
"""5.2整节：参数访问、梯度/快照、初始化及共享参数的真实训练。

沿用openmmlab，CPU/float64、人工数据，无安装和下载。
报告、图与本地核对状态均保存在本项目；不会填写用户练习。
"""
import copy
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

REPORT = ROOT / 'reports/ch05'
FIGURES = REPORT / 'figures'
DTYPE = torch.float64


def basic_net():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1)).double()


def unique_state(net):
    return {name: parameter.detach().clone() for name, parameter in net.named_parameters()}


def parameter_rows(net, model):
    return [dict(model=model, name=name, shape=str(list(p.shape)), elements=p.numel(),
                 requires_grad=p.requires_grad, is_leaf=p.is_leaf, grad_is_none=p.grad is None)
            for name, p in net.named_parameters()]


def access_demo():
    torch.manual_seed(5201)
    net = basic_net()
    X = torch.arange(8, dtype=DTYPE).reshape(2, 4) / 4
    y = torch.tensor([[1.], [-1.]], dtype=DTYPE)
    rows = parameter_rows(net, '4_to_8_to_1')
    initial = unique_state(net)
    before_none = net[2].weight.grad is None
    F.mse_loss(net(X), y).backward()
    first = net[2].weight.grad.clone()
    F.mse_loss(net(X), y).backward()  # 新建前向图，梯度仍累加到同一叶参数。
    accumulation_error = (net[2].weight.grad - 2 * first).abs().max().item()
    net.zero_grad(set_to_none=True)
    detached = net[0].weight.detach()  # 与参数共享存储，只是脱离求导。
    cloned = net[0].weight.detach().clone()
    live = net.state_dict()  # 映射浅拷贝，张量仍引用模型存储。
    snapshot = copy.deepcopy(net.state_dict())
    identity = id(net[0].weight)
    with torch.no_grad():
        net[0].weight.add_(1)
        net[0].weight[0, 0] = 42
    info = dict(parameter_count=sum(p.numel() for p in net.parameters()),
                output_shape=list(net(X).shape), parameter_type=type(net[2].bias).__name__,
                before_backward_grad_is_none=before_none,
                accumulated_gradient_max_error=accumulation_error,
                after_zero_grad_is_none=net[2].weight.grad is None,
                detach_tracks_changed_value=torch.equal(detached, net[0].weight),
                clone_preserves_old_value=torch.equal(cloned, initial['0.weight']),
                state_dict_tracks_changed_value=torch.equal(live['0.weight'], net[0].weight),
                deep_snapshot_preserves_old_value=torch.equal(snapshot['0.weight'], initial['0.weight']),
                no_grad_edit_preserves_parameter_identity=id(net[0].weight) == identity,
                edited_first_value=net[0].weight[0, 0].item())
    def block():
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
    stack = nn.Sequential()
    for i in range(4):stack.add_module('block' + str(i), block())
    nested = nn.Sequential(stack, nn.Linear(4, 1)).double()
    info['nested_parameter_count'] = sum(p.numel() for p in nested.parameters())
    info['nested_bias_path'] = '0.block1.0.bias'
    info['nested_path_matches_index'] = nested.get_parameter('0.block1.0.bias') is nested[0][1][0].bias
    rows += parameter_rows(nested, 'nested_four_blocks')
    previous = importlib.import_module('lessons.ch05.01_layers_and_blocks')
    fixed = previous.FixedHiddenMLP().double()
    info['previous_fixed_parameter_names'] = list(dict(fixed.named_parameters()))
    info['previous_fixed_buffer_names'] = list(dict(fixed.named_buffers()))
    info['previous_fixed_state_names'] = list(fixed.state_dict())
    return info, rows, dict(initial=initial, X=X, y=y, first_output_gradient=first)


def init_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=.01)
        if module.bias is not None:nn.init.zeros_(module.bias)


def init_constant(module):
    if isinstance(module, nn.Linear):
        nn.init.ones_(module.weight)
        if module.bias is not None:nn.init.zeros_(module.bias)


@torch.no_grad()
def init_custom(module):
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.mul_(module.weight.abs() >= 5)
        if module.bias is not None:nn.init.zeros_(module.bias)


def initialization_demo():
    states, rows = {}, []
    for index, kind in enumerate(('default', 'normal_0.01', 'ones', 'mixed', 'custom')):
        torch.manual_seed(5202 + index)
        net = basic_net()
        if kind == 'normal_0.01':net.apply(init_normal)
        elif kind == 'ones':net.apply(init_constant)
        elif kind == 'mixed':
            nn.init.xavier_uniform_(net[0].weight, gain=1)
            nn.init.constant_(net[2].weight, 42)
            nn.init.zeros_(net[0].bias); nn.init.zeros_(net[2].bias)
        elif kind == 'custom':net.apply(init_custom)
        states[kind] = unique_state(net)
        for name, p in net.named_parameters():
            values = p.detach()
            rows.append(dict(method=kind, name=name, count=p.numel(),
                             mean=values.mean().item(), population_std=values.std(unbiased=False).item(),
                             minimum=values.min().item(), maximum=values.max().item(),
                             zeros=int((values == 0).sum())))
    torch.manual_seed(5208)
    wide = nn.Linear(200, 100, bias=False).double()
    init_custom(wide)
    sample = wide.weight.detach().flatten().clone()
    probabilities = dict(negative=(sample <= -5).double().mean().item(),
                         zero=(sample == 0).double().mean().item(),
                         positive=(sample >= 5).double().mean().item())
    info = dict(methods=list(states), default_hidden_bound=1 / math.sqrt(4),
                default_output_bound=1 / math.sqrt(8), custom_sample_count=len(sample),
                custom_empirical_probabilities=probabilities,
                custom_theoretical_probabilities=dict(negative=.25, zero=.5, positive=.25),
                custom_sample_variance=sample.var(unbiased=False).item(),
                custom_theoretical_variance=175 / 6)
    return info, rows, dict(states=states, custom_sample=sample)


def make_shared_net():
    shared = nn.Linear(8, 8)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(),
                         shared, nn.ReLU(), nn.Linear(8, 1)).double()


def init_for_relu(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:nn.init.zeros_(module.bias)


def make_data():
    rng = torch.Generator().manual_seed(5210)
    data = {}
    for name, count in [('train', 64), ('validation', 32), ('test', 32)]:
        X = torch.randn(count, 4, generator=rng, dtype=DTYPE)
        y = .8 * X[:, :1] - .5 * X[:, 1:2] + .4 * X[:, 2:3] * X[:, 3:4]
        data[name] = (X, y)
    return data


def train_shared():
    torch.manual_seed(5211)
    net = make_shared_net()
    net.apply(init_for_relu)
    # 输出接线性回归，gain1；不是把输出也按ReLU处理。
    nn.init.xavier_normal_(net[6].weight, gain=1)
    initial = unique_state(net)
    data = make_data();X, y = data['train']
    # 复制两套独立中间层，初始数值与共享模型相同，专门核对梯度贡献。
    untied = nn.Sequential(copy.deepcopy(net[0]), nn.ReLU(), copy.deepcopy(net[2]),
                          nn.ReLU(), copy.deepcopy(net[4]), nn.ReLU(), copy.deepcopy(net[6]))
    F.mse_loss(net(X), y).backward();F.mse_loss(untied(X), y).backward()
    gradient_parts = {name: untied[position].weight.grad.detach().clone()
                      for name, position in [('first_use', 2), ('second_use', 4)]}
    gradient_parts['shared_total'] = net[2].weight.grad.detach().clone()
    summed = untied[2].weight.grad + untied[4].weight.grad
    info = dict(shared_parameter_count=sum(p.numel() for p in net.parameters()),
                untied_parameter_count=sum(p.numel() for p in untied.parameters()),
                unique_parameter_tensors=len(list(net.parameters())),
                named_paths_without_dedup=len(list(net.named_parameters(remove_duplicate=False))),
                state_dict_key_count=len(net.state_dict()),
                parameter_identity=net[2].weight is net[4].weight,
                initial_output_max_error=(net(X) - untied(X)).abs().max().item(),
                shared_weight_gradient_sum_max_error=(net[2].weight.grad - summed).abs().max().item(),
                shared_bias_gradient_sum_max_error=(net[2].bias.grad - untied[2].bias.grad - untied[4].bias.grad).abs().max().item(),
                state_alias_storage=net.state_dict()['2.weight'].data_ptr() == net.state_dict()['4.weight'].data_ptr())
    net.zero_grad(set_to_none=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=.02)
    history, traces = [], []
    for step in range(301):
        net.eval()
        with torch.no_grad():
            train_mse = F.mse_loss(net(X), y).item()
            vx, vy = data['validation']
            validation_mse = F.mse_loss(net(vx), vy).item()
        history.append(dict(step=step, train_mse=train_mse, validation_mse=validation_mse,
                            shared_weight_norm=net[2].weight.norm().item(),
                            shared_weight_00=net[2].weight[0, 0].item()))
        if step == 300:break
        net.train();optimizer.zero_grad(set_to_none=True)
        F.mse_loss(net(X), y).backward()
        for name, parameter in net.named_parameters():
            traces.append(dict(update=step+1, name=name, parameter_norm=parameter.norm().item(),
                               gradient_norm=parameter.grad.norm().item(),
                               shared_same_object=net[2].weight is net[4].weight))
        optimizer.step()
    net.eval()
    with torch.no_grad():
        tx, ty = data['test'];prediction = net(tx);test_mse = F.mse_loss(prediction, ty).item()
    info.update(dict(updates=300, learning_rate=.02, train_count=64, validation_count=32, test_count=32,
                     initial_train_mse=history[0]['train_mse'], final_train_mse=history[-1]['train_mse'],
                     final_validation_mse=history[-1]['validation_mse'], test_mse=test_mse,
                     final_parameter_identity=net[2].weight is net[4].weight,
                     shared_parameter_change=(net[2].weight - initial['2.weight']).norm().item()))
    return info, history, traces, dict(initial=initial, final=unique_state(net), data=data,
                                      gradient_parts=gradient_parts, prediction=prediction)


def draw_figures(initialization, training, history, traces):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False,
                         'font.size': 11, 'figure.dpi': 145})
    teal, gold, navy = '#147e83', '#d58a30', '#173448'
    fig, ax = plt.subplots(figsize=(9, 4.3));ax.axis('off')
    items = [(1.2, 3, '输入\n(n,4)'), (4.3, 3, '0: Linear(4,8)\nweight(8,4), bias(8)\n40个参数'),
             (7.7, 3, '2: Linear(8,1)\nweight(1,8), bias(1)\n9个参数')]
    ax.set(xlim=(0, 10), ylim=(0, 5))
    for x,y,text in items:ax.text(x,y,text,ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#eff7f8',ec=teal))
    for start,end in [((2,3),(2.8,3)),((5.8,3),(6.2,3))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=navy,lw=1.6))
    ax.text(6,3.6,'ReLU',ha='center',color=teal)
    ax.text(5,.8,'named_parameters：0.weight / 0.bias / 2.weight / 2.bias\nLinear的weight按(输出维, 输入维)保存；共49个参数元素',ha='center',linespacing=1.8)
    fig.tight_layout();fig.savefig(FIGURES/'5_2_parameter_map.png');plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax,(name,state) in zip(axes.flat,initialization['states'].items()):
        values=torch.cat([p.flatten() for key,p in state.items() if key.endswith('weight')]).numpy()
        ax.hist(values,bins=16,color=teal);ax.set_title(name);ax.set_ylabel('参数元素数')
    axes.flat[-1].hist(initialization['custom_sample'].numpy(),bins=40,color=gold)
    axes.flat[-1].set(title='自定义分布：独立20000个值',xlabel='权重值',ylabel='参数元素数')
    fig.suptitle('前五图是同一4→8→1网络的40个权重，不要求小样本频率等于理论概率')
    fig.tight_layout();fig.savefig(FIGURES/'5_2_initialization.png');plt.close(fig)
    parts=training['gradient_parts'];limit=max(part.abs().max().item() for part in parts.values())
    fig,axes=plt.subplots(1,3,figsize=(10,3.8))
    for ax,(name,value) in zip(axes,parts.items()):
        im=ax.imshow(value.numpy(),cmap='coolwarm',vmin=-limit,vmax=limit)
        ax.set(title={'first_use':'第一次使用的贡献','second_use':'第二次使用的贡献','shared_total':'共享权重的总梯度'}[name],xlabel='输入列',ylabel='输出行')
    fig.subplots_adjust(wspace=.38,right=.89,top=.78,bottom=.17)
    cax=fig.add_axes([.92,.2,.018,.5]);fig.colorbar(im,cax=cax)
    fig.suptitle('共享参数：两个使用位置的贡献相加，不取平均')
    fig.savefig(FIGURES/'5_2_shared_gradients.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.4))
    ax.semilogy([r['step'] for r in history],[r['train_mse'] for r in history],label='训练64条',color=teal)
    ax.semilogy([r['step'] for r in history],[r['validation_mse'] for r in history],label='验证32条',color=gold)
    ax.set(xlabel='全批量SGD更新次数',ylabel='MSE（对数坐标）',title='共享8→8层调用两次，121个不同参数')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIGURES/'5_2_training.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,label in [('0.weight','输入层W'),('2.weight','共享层W'),('6.weight','输出层W')]:
        part=[r for r in traces if r['name']==name]
        axes[0].plot([r['update'] for r in part],[r['parameter_norm'] for r in part],label=label)
        axes[1].semilogy([r['update'] for r in part],[max(r['gradient_norm'],1e-16) for r in part],label=label)
    axes[0].set(title='每次更新前的参数范数',xlabel='更新序号',ylabel='L2/Frobenius范数')
    axes[1].set(title='每次更新前的梯度范数',xlabel='更新序号',ylabel='梯度范数（对数坐标）')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'5_2_parameter_traces.png');plt.close(fig)


def write_csv(name, rows):
    with (REPORT/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    access,parameters,access_state=access_demo()
    initialization,initializers,init_state=initialization_demo()
    sharing,history,traces,train_state=train_shared()
    metrics=dict(section='5.2',date='2026-10-29',torch_version=torch.__version__,dtype=str(DTYPE),
                 device='cpu',access=access,initialization=initialization,sharing_and_training=sharing)
    write_csv('5_2_parameters.csv',parameters);write_csv('5_2_initializers.csv',initializers)
    write_csv('5_2_epochs.csv',history);write_csv('5_2_parameter_traces.csv',traces)
    artifact=ROOT/'artifacts/ch05/5_2_states.pt';artifact.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(access=access_state,initialization=init_state,training=train_state),artifact)
    draw_figures(init_state,train_state,history,traces)
    (REPORT/'5_2_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    output='\n'.join(['5.2 参数管理（CPU/float64，无下载/安装）',
        '参数访问/梯度/快照：'+json.dumps(access,ensure_ascii=False),
        '初始化：'+json.dumps(initialization,ensure_ascii=False),
        '共享梯度与300次真实更新：'+json.dumps(sharing,ensure_ascii=False),
        '继续完成exercises/ch05/02_parameter_management.py，八处TODO留空。'])
    (REPORT/'5_2_demo_output.txt').write_text(output+'\n',encoding='utf8');print(output)


if __name__=='__main__':main()
