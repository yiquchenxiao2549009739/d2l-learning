# -*- coding: utf-8 -*-
"""5.1综合练习：创建和注册双分支块，拼接后训练；八处TODO留空。"""
import torch
from torch import nn
from torch.nn import functional as F


class ParallelRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = None  # TODO 1：注册Linear(2,4)+ReLU的顺序块。
        self.right = None  # TODO 2：注册Linear(2,3)+Tanh的顺序块。
        self.out = None  # TODO 3：拼接后7维到1维的输出层。

    def forward(self, X):
        left = self.left(X)
        right = self.right(X)
        hidden = None  # TODO 4：沿最后一维拼接两分支输出。
        return self.out(hidden)


def make_data():
    rng = torch.Generator().manual_seed(5181)
    X = 4 * torch.rand(48, 2, generator=rng, dtype=torch.float64) - 2
    y = X[:, :1].square() + .5 * X[:, 1:]
    return X[:32], y[:32], X[32:], y[32:]


def main():
    torch.set_num_threads(1); torch.manual_seed(5182)
    net = ParallelRegressor().double()
    if any(layer is None for layer in (net.left, net.right, net.out)):
        print('请集中填写八处TODO：双分支模块、拼接、优化器、MSE、反向传播和更新。')
        return
    optimizer = None  # TODO 5：SGD，学习率.03，管理net.parameters()。
    if optimizer is None:return
    X, y, VX, Vy = make_data()
    with torch.no_grad():initial = F.mse_loss(net(X), y).item()
    for _ in range(250):
        net.train(); optimizer.zero_grad(set_to_none=True)
        prediction = net(X)
        loss = None  # TODO 6：prediction与y之间的平均平方误差。
        if loss is None:return
        pass  # TODO 7：反向传播。
        if not any(p.grad is not None for p in net.parameters()):
            print('请先计算梯度。');return
        pass  # TODO 8：优化器更新。
    net.eval()
    with torch.no_grad():
        train_mse = F.mse_loss(net(X), y).item()
        validation_mse = F.mse_loss(net(VX), Vy).item()
        print('分支形状/拼接形状/输出形状:', net.left(X).shape, net.right(X).shape,
              torch.cat((net.left(X), net.right(X)), dim=-1).shape, net(X).shape)
    print('参数总数:', sum(p.numel() for p in net.parameters()))
    print('初始/最终训练MSE与验证MSE:', initial, train_mse, validation_mse)
    print('解释1：把分支层只放普通list，会失去哪几种自动管理？能不能仍有梯度？')
    print('解释2：为什么dim=-1而不是dim=0？为什么输出层接收7维？')
    print('解释3：同一块用三次，与新建三个块，参数和梯度有何区别？')


if __name__ == '__main__':main()
