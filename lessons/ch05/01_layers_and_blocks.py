# -*- coding: utf-8 -*-
"""5.1整节：自定义块、注册、顺序/并行/嵌套、共享和动态控制流。

CPU/float64，小型人工数据，无下载和安装；所有输出留在本项目。
运行本文件即可贯通整节，用户作业另放exercises，不自动填写。
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

REPORT = ROOT / 'reports/ch05'
FIGURES = REPORT / 'figures'
DTYPE = torch.float64


class MLP(nn.Module):
    """与教材一致：20→256→10；构造一次，多次复用。"""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    """只实现单输入单输出顺序执行，不复制Sequential的完整接口。"""
    def __init__(self, *blocks):
        super().__init__()
        self.block_count = len(blocks)
        for index, block in enumerate(blocks):
            # 公开API完成注册；教材直接写_modules，此处避免依赖内部属性。
            self.add_module(str(index), block)

    def forward(self, X):
        # 按位置执行；children()会去重，同一对象重复传入时不能用于执行顺序。
        for index in range(self.block_count):
            X = self.get_submodule(str(index))(X)
        return X


class ListChain(nn.Module):
    def __init__(self, *blocks, registered=False):
        super().__init__()
        self.blocks = nn.ModuleList(blocks) if registered else list(blocks)

    def forward(self, X):
        for block in self.blocks:
            X = block(X)
        return X


class Parallel(nn.Module):
    """两分支接收相同输入，沿最后一维拼接，不是先后串行执行。"""
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, X):
        return torch.cat((self.left(X), self.right(X)), dim=-1)


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # buffer不交给优化器，仍随.to迁移并写入state_dict。
        self.register_buffer('rand_weight', torch.rand(20, 20))
        self.linear = nn.Linear(20, 20)
        self.last_halvings = 0

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        X = self.linear(X)  # 同一个对象，两处使用，梯度贡献相加。
        if not bool(torch.isfinite(X).all()):
            raise ValueError('控制流演示要求有限输入和中间结果。')
        count = 0
        while bool(X.detach().abs().sum() > 1):
            X = X / 2  # 不原地改写自动求导可能保存的张量。
            count += 1
        self.last_halvings = count
        return X.sum()  # 整个批量的一个标量，演示用，不是逐样本预测。


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


def repeat_blocks(factory, count):
    """每次调用factory创建独立对象；相邻块的形状必须能衔接。"""
    return MySequential(*(factory() for _ in range(count)))


def count_parameters(net):
    return sum(parameter.numel() for parameter in net.parameters())


def make_parallel_regressor():
    return nn.Sequential(
        Parallel(nn.Sequential(nn.Linear(2, 8), nn.ReLU()),
                 nn.Sequential(nn.Linear(2, 4), nn.Tanh())),
        nn.Linear(12, 1)).double()


def make_regression_data():
    rng = torch.Generator().manual_seed(5106)
    data = {}
    for name, n in [('train', 64), ('validation', 32), ('test', 32)]:
        X = 4 * torch.rand(n, 2, generator=rng, dtype=DTYPE) - 2
        y = X[:, :1].square() + .5 * X[:, 1:] + .1 * torch.sin(2 * X[:, 1:])
        data[name] = (X, y)
    return data


def equivalent_models():
    torch.manual_seed(5101)
    standard = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10)).double()
    custom = MLP().double()
    custom.hidden.load_state_dict(standard[0].state_dict())
    custom.out.load_state_dict(standard[2].state_dict())
    sequential = MySequential(copy.deepcopy(standard[0]), nn.ReLU(),
                              copy.deepcopy(standard[2]))
    models = [standard, custom, sequential]
    rng = torch.Generator().manual_seed(5102)
    X = torch.randn(13, 20, generator=rng, dtype=DTYPE)
    y = torch.randn(13, 10, generator=rng, dtype=DTYPE)
    initial = copy.deepcopy(standard.state_dict())
    input_grads = []
    for net in models:
        sample = X.clone().requires_grad_()
        F.mse_loss(net(sample), y).backward()
        input_grads.append(sample.grad.clone())
        net.zero_grad(set_to_none=True)
    input_error = max((g - input_grads[0]).abs().max().item() for g in input_grads)
    optimizers = [torch.optim.SGD(net.parameters(), lr=.03) for net in models]
    rows = []
    for epoch in range(3):
        for start in range(0, len(X), 5):
            bx, by = X[start:start + 5], y[start:start + 5]
            predictions, grads = [], []
            for net, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=True)
                prediction = net(bx)
                F.mse_loss(prediction, by).backward()
                predictions.append(prediction.detach())
                grads.append([p.grad.clone() for p in net.parameters()])
                optimizer.step()
            output_error = max((p - predictions[0]).abs().max().item() for p in predictions)
            grad_error = max((g - grads[0][j]).abs().max().item()
                             for group in grads for j, g in enumerate(group))
            base_parameters = list(standard.parameters())
            parameter_error = max((p - base_parameters[j]).abs().max().item()
                                  for net in models for j, p in enumerate(net.parameters()))
            rows.append(dict(step=len(rows) + 1, epoch=epoch + 1, batch_size=len(bx),
                             output_max_error=output_error, gradient_max_error=grad_error,
                             parameter_max_error=parameter_error))
    calls = []
    handle = custom.register_forward_hook(lambda module, args, result: calls.append(1))
    custom(X); custom.forward(X)  # 第二次不会触发custom本身的forward hook。
    handle.remove()
    return dict(parameter_count=count_parameters(standard), input_gradient_max_error=input_error,
                parent_hook_calls=len(calls)), rows, dict(initial=initial, final=standard.state_dict(), X=X, y=y)


def registration_demo():
    results = []
    for registered in (False, True):
        torch.manual_seed(5103)
        net = ListChain(nn.Linear(2, 3), nn.ReLU(), nn.Dropout(.5),
                        nn.Linear(3, 1), registered=registered)
        loss = net(torch.ones(4, 2)).square().mean()
        loss.backward()
        try:
            torch.optim.SGD(net.parameters(), lr=.1)
            optimizer_error = None
        except ValueError as error:
            optimizer_error = str(error)
        net.eval()
        net.double()
        results.append(dict(container='ModuleList' if registered else 'list',
                            parameter_count=count_parameters(net),
                            state_keys=list(net.state_dict()),
                            hidden_weight_has_gradient=net.blocks[0].weight.grad is not None,
                            child_dropout_training=net.blocks[2].training,
                            child_weight_dtype=str(net.blocks[0].weight.dtype),
                            optimizer_error=optimizer_error))
    return results


def composition_demo():
    torch.manual_seed(5104)
    fixed = FixedHiddenMLP().double()
    X = torch.linspace(-1, 1, 40, dtype=DTYPE).reshape(2, 20)
    state = copy.deepcopy(fixed.state_dict())
    value = fixed(X)
    value.backward()
    gradients = {name: p.grad.clone() for name, p in fixed.named_parameters()}
    optimizer = torch.optim.SGD(fixed.parameters(), lr=.01)
    optimizer.step()
    info = dict(output=value.item(), output_shape=list(value.shape), halvings=fixed.last_halvings,
                parameter_count=count_parameters(fixed), buffer_count=fixed.rand_weight.numel(),
                buffer_unchanged=torch.equal(state['rand_weight'], fixed.rand_weight),
                buffer_has_gradient=fixed.rand_weight.grad is not None,
                weight_update_norm=(fixed.linear.weight - state['linear.weight']).norm().item())
    torch.manual_seed(5105)
    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP()).double()
    output = chimera(X)
    output.backward()
    info['nested_parameter_count'] = count_parameters(chimera)
    info['nested_output_shape'] = list(output.shape)
    info['nested_all_parameter_gradients_finite'] = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in chimera.parameters())
    factory = lambda: nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    independent = repeat_blocks(factory, 3)
    shared_block = factory()
    shared = nn.Sequential(shared_block, shared_block, shared_block)
    info['independent_repeat_parameters'] = count_parameters(independent)
    info['shared_repeat_parameters'] = count_parameters(shared)
    info['independent_objects'] = len({id(block) for block in independent.children()})
    info['shared_positions_same_object'] = shared[0] is shared[1] is shared[2]
    return info, dict(state=state, X=X, gradients=gradients,
                      output=value.detach(), halvings=info['halvings'])


def train_parallel():
    torch.manual_seed(5107)
    net = make_parallel_regressor()
    initial = copy.deepcopy(net.state_dict())
    data = make_regression_data()
    X, y = data['train']
    optimizer = torch.optim.SGD(net.parameters(), lr=.03)
    rows = []
    for step in range(301):
        net.eval()
        with torch.no_grad():
            train_mse = F.mse_loss(net(X), y).item()
            vx, vy = data['validation']
            validation_mse = F.mse_loss(net(vx), vy).item()
        rows.append(dict(step=step, train_mse=train_mse, validation_mse=validation_mse))
        if step == 300:
            break
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(net(X), y)
        loss.backward()
        optimizer.step()
    net.eval()
    with torch.no_grad():
        tx, ty = data['test']
        prediction = net(tx)
        test_mse = F.mse_loss(prediction, ty).item()
    info = dict(parameter_count=count_parameters(net), input_shape=list(X.shape),
                branch_widths=[8, 4], concatenated_shape=[64, 12], output_shape=[64, 1],
                updates=300, learning_rate=.03, train_count=64, validation_count=32, test_count=32,
                initial_train_mse=rows[0]['train_mse'], final_train_mse=rows[-1]['train_mse'],
                final_validation_mse=rows[-1]['validation_mse'], test_mse=test_mse)
    return info, rows, dict(initial=initial, final=net.state_dict(), data=data, prediction=prediction)


def draw_figures(rows, metrics):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False,
                         'font.size': 11, 'figure.dpi': 145})
    navy, teal, gold = '#173448', '#147e83', '#d58a30'
    def boxes(ax, items, arrows):
        ax.set(xlim=(0, 10), ylim=(0, 6)); ax.axis('off')
        for x, y, text in items:
            ax.text(x, y, text, ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=.65', fc='#eff7f8', ec=teal))
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=navy, lw=1.7))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    boxes(ax, [(5, 5.2, '模型也是一个块：Parallel 回归器'),
               (2.6, 3.5, '左块：Linear 2→8 + ReLU'), (7.4, 3.5, '右块：Linear 2→4 + tanh'),
               (5, 1.8, 'cat(dim=-1)：8+4=12维'), (5, .4, '输出层：Linear 12→1')],
          [((4, 4.8), (2.6, 4)), ((6, 4.8), (7.4, 4)),
           ((2.6, 3), (4, 2.2)), ((7.4, 3), (6, 2.2)), ((5, 1.3), (5, .85))])
    fig.tight_layout(); fig.savefig(FIGURES / '5_1_blocks.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.axis('off')
    cell = [['普通 list', '可计算/可求导', '父模块发现0参数', '子Dropout仍训练', '仍float32'],
            ['ModuleList', '可计算/可求导', '发现13参数', '子Dropout已评估', '已float64']]
    tab = ax.table(cellText=cell, colLabels=['容器', 'forward/backward', 'parameters()', 'parent.eval()', 'parent.double()'],
                   loc='center', cellLoc='center', colWidths=[.15, .2, .23, .23, .19])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#d7e2e9')
        if r == 0: cell.set_facecolor(navy); cell.set_text_props(color='white')
    ax.set_title('能算出结果，不代表父模块能管理这些层', pad=15)
    fig.tight_layout(); fig.savefig(FIGURES / '5_1_registration.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    boxes(ax, [(1, 3, 'X\n(2,20)'), (3.4, 3, '共享 Linear\nW,b 第一次'),
               (6.3, 3, '固定buffer R\nReLU(XR+1)'), (9, 3, '同一 Linear\nW,b 第二次'),
               (6.2, .8, 'while：除2直到L1≤1；最后整批sum')],
          [((1.55, 3), (2.45, 3)), ((4.35, 3), (5.15, 3)),
           ((7.45, 3), (8.05, 3)), ((9, 2.35), (7.3, 1.15))])
    ax.set_title('只有420个可训练参数；R的400个值不进优化器')
    fig.tight_layout(); fig.savefig(FIGURES / '5_1_control_flow.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].bar(['工厂新建3次', '同一对象用3次'], [216, 72], color=[teal, gold])
    axes[0].set(ylabel='不同参数的元素总数', title='8→8全连接块 × 3')
    axes[1].axis('off')
    axes[1].text(.05, .8, '独立实例：W1、W2、W3', fontsize=14, color=teal)
    axes[1].text(.05, .56, '复用实例：同一个W', fontsize=14, color=gold)
    axes[1].text(.05, .26, '共享时：各使用位置的梯度相加\n结构相同 ≠ 参数是同一对象', fontsize=12, linespacing=1.8)
    fig.tight_layout(); fig.savefig(FIGURES / '5_1_instances.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy([r['step'] for r in rows], [r['train_mse'] for r in rows], color=teal, label='训练64条')
    ax.semilogy([r['step'] for r in rows], [r['validation_mse'] for r in rows], color=gold, label='验证32条')
    ax.set(xlabel='全批量SGD更新次数', ylabel='MSE（对数坐标）', title='双分支 2→(8+4)→1，49参数')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / '5_1_training.png'); plt.close(fig)


def write_csv(name, rows):
    with (REPORT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    equivalence, steps, eq_state = equivalent_models()
    registration = registration_demo()
    composition, fixed_state = composition_demo()
    training, epochs, train_state = train_parallel()
    metrics = dict(section='5.1', date='2026-10-29', torch_version=torch.__version__,
                   dtype=str(DTYPE), device='cpu', equivalence=equivalence,
                   registration=registration, composition=composition, training=training)
    write_csv('5_1_equivalence.csv', steps); write_csv('5_1_epochs.csv', epochs)
    artifact = ROOT / 'artifacts/ch05/5_1_states.pt'; artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(equivalence=eq_state, fixed=fixed_state, training=train_state), artifact)
    draw_figures(epochs, metrics)
    (REPORT / '5_1_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    output = '\n'.join([
        '5.1 层和块：完整演示（CPU/float64，无下载/安装）',
        '三种20→256→10写法：7946参数、9次更新，输出/梯度/参数最大差：',
        str([max(row[key] for row in steps) for key in
             ['output_max_error', 'gradient_max_error', 'parameter_max_error']]),
        '普通list与ModuleList：' + json.dumps(registration, ensure_ascii=False),
        '固定/共享/嵌套：' + json.dumps(composition, ensure_ascii=False),
        '双分支训练：' + json.dumps(training, ensure_ascii=False),
        '接下来完成exercises/ch05/01_layers_and_blocks.py，八处TODO仍留空。'])
    (REPORT / '5_1_demo_output.txt').write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
