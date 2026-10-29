# -*- coding: utf-8 -*-
"""5.2综合练习：检查、初始化、绑定参数并训练；八处TODO留空。"""
import torch
from torch import nn
from torch.nn import functional as F


def make_net():
    shared = None  # TODO 1：创建一个Linear(4,4)，供两个位置共同使用。
    if shared is None:return None
    return nn.Sequential(nn.Linear(2,4),nn.ReLU(),shared,nn.ReLU(),
                         shared,nn.ReLU(),nn.Linear(4,1)).double()


def initialize(module):
    if isinstance(module,nn.Linear):
        pass  # TODO 2：对weight作Xavier均匀初始化，gain=1。
        nn.init.zeros_(module.bias)


def main():
    torch.set_num_threads(1);torch.manual_seed(5281)
    net=make_net()
    if net is None:
        print('请集中填写八处TODO：共享层、初始化、参数名与快照、SGD、损失、反传及更新。');return
    net.apply(initialize)
    names = None  # TODO 3：取得net.named_parameters()提供的参数名列表。
    snapshot = None  # TODO 4：保存共享层weight的detach().clone()独立快照。
    optimizer = None  # TODO 5：SGD，lr=.03，管理net.parameters()。
    if names is None or snapshot is None or optimizer is None:return
    rng=torch.Generator().manual_seed(5282)
    X=2*torch.rand(48,2,generator=rng,dtype=torch.float64)-1
    y=X[:,:1].square()+.5*X[:,1:]
    with torch.no_grad():initial=F.mse_loss(net(X[:32]),y[:32]).item()
    for _ in range(250):
        net.train();optimizer.zero_grad(set_to_none=True)
        prediction=net(X[:32])
        loss = None  # TODO 6：prediction与训练标签的平均MSE。
        if loss is None:return
        pass  # TODO 7：反向传播。
        if net[2].weight.grad is None:
            print('请先计算梯度。');return
        pass  # TODO 8：优化器更新。
    net.eval()
    with torch.no_grad():
        train_mse=F.mse_loss(net(X[:32]),y[:32]).item()
        valid_mse=F.mse_loss(net(X[32:]),y[32:]).item()
    print('去重参数名称:',names)
    print('不同参数总数:',sum(p.numel() for p in net.parameters()))
    print('两个位置是否同一Parameter:',net[2].weight is net[4].weight)
    print('共享层相对独立初始快照的变化:',(net[2].weight-snapshot).norm().item())
    print('初始训练/最终训练/验证MSE:',initial,train_mse,valid_mse)
    print('解释1：Parameter、detach视图和detach().clone()有什么区别？')
    print('解释2：为什么state_dict有两条共享权重路径，却只计一组参数？')
    print('解释3：共享梯度为什么相加？共享会怎样影响容量与适用任务？')


if __name__=='__main__':main()
