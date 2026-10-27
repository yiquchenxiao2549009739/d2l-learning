# -*- coding: utf-8 -*-
"""4.5综合实践：17训练/60验证/60测试、40特征、按验证选lambda，八处TODO。"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DTYPE = torch.float64


def half_mse(prediction, target):
    assert prediction.shape == target.shape
    return ((prediction-target)**2).mean()/2


def l2_penalty(w):
    return None  # TODO 1：权重平方和的一半。


def main():
    torch.set_num_threads(1)
    if l2_penalty(torch.ones(2, 1, dtype=DTYPE)) is None:
        print('请集中补齐八处TODO；惩罚只用于报告，反向传播只用数据损失。'); return
    generator = torch.Generator().manual_seed(4507)
    X = torch.randn(137, 40, generator=generator, dtype=DTYPE)
    y = X@torch.full((40, 1), .01, dtype=DTYPE)+.05+.01*torch.randn(137, 1, generator=generator, dtype=DTYPE)
    initial_w = torch.randn(1, 40, generator=torch.Generator().manual_seed(4508), dtype=DTYPE)
    candidates = []
    for lambd in (0., 3., 30.):
        net = None  # TODO 2：40输入、1输出的Linear，转换为DTYPE。
        if net is None:
            print('请构造线性回归模型。'); return
        with torch.no_grad():
            net.weight.copy_(initial_w); net.bias.zero_()
        groups = None  # TODO 3：weight衰减lambd，bias衰减0，分为两组。
        if groups is None:
            print('请分组设置weight_decay。'); return
        optimizer = None  # TODO 4：SGD(groups)，lr=.003，无动量。
        if optimizer is None:
            print('请连接优化器。'); return
        loader = DataLoader(TensorDataset(X[:17], y[:17]), batch_size=5, shuffle=True,
                            num_workers=0, drop_last=False,
                            generator=torch.Generator().manual_seed(4509))
        sizes, updates = [], 0
        for epoch in range(80):
            net.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                data_loss = None  # TODO 5：平均半平方损失；这里不再加L2。
                if data_loss is None:
                    print('请计算数据损失。'); return
                pass  # TODO 6：对数据损失反向传播。
                if any(p.grad is None for p in net.parameters()):
                    print('请先计算梯度。'); return
                pass  # TODO 7：优化器更新。
                updates += 1
                if epoch == 0: sizes.append(len(yb))
        net.eval()
        with torch.no_grad():
            train_loss = half_mse(net(X[:17]), y[:17]).item()
            val_loss = half_mse(net(X[17:77]), y[17:77]).item()
            penalty = lambd*l2_penalty(net.weight).item()
        candidates.append({'lambda': lambd, 'train_half_mse': train_loss, 'validation_half_mse': val_loss,
                           'penalty': penalty, 'weight_norm': net.weight.norm().item(), 'model': net})
    best = None  # TODO 8：按validation_half_mse最小选择候选。
    if best is None:
        print('请按验证数据损失选lambda，不能看测试集选择。'); return
    with torch.no_grad(): test_loss = half_mse(best['model'](X[77:]), y[77:]).item()
    print('首轮批量:', sizes, '| 每个候选更新:', updates, '| 参数数:', sum(p.numel() for p in best['model'].parameters()))
    print('候选:', [{k:v for k,v in row.items() if k!='model'} for row in candidates])
    print('选择lambda:', best['lambda'], '| 最终独立测试半MSE:', test_loss)
    print('解释：为什么训练损失变高，验证可能改善？为什么bias单独一组？')
    print('解释：mean之后是否再除批量？手动惩罚与weight_decay同时使用会怎样？')


if __name__ == '__main__':
    main()
