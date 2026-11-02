# -*- coding: utf-8 -*-
"""5.3综合练习：先确定形状，再自定义初始化并训练，八处TODO留空。"""
import torch
from torch import nn
from torch.nn import functional as F


def main():
    torch.set_num_threads(1);torch.manual_seed(5381)
    net = None  # TODO 1：LazyLinear(8)、ReLU、LazyLinear(1)组成的float64顺序模型。
    if net is None:
        print('请集中补齐八处TODO：Lazy网络、首次试运行、初始化和训练循环。');return
    net.eval()
    with torch.no_grad():
        pass  # TODO 2：用形状(2,3)、float64的零张量完成首次前向。
    if isinstance(net[0].weight,torch.nn.parameter.UninitializedParameter):
        print('请先完成首次前向，确定参数形状。');return
    pass  # TODO 3：第一层weight作Kaiming正态初始化，nonlinearity=relu。
    pass  # TODO 4：输出层weight作Xavier正态初始化，gain=1。
    nn.init.zeros_(net[0].bias);nn.init.zeros_(net[2].bias)
    optimizer = None  # TODO 5：SGD，lr=.03，管理net.parameters()。
    if optimizer is None:return
    rng=torch.Generator().manual_seed(5382)
    X=2*torch.rand(48,3,generator=rng,dtype=torch.float64)-1
    y=X[:,:1].square()+.5*X[:,1:2]-.3*X[:,2:3]
    with torch.no_grad():initial=F.mse_loss(net(X[:32]),y[:32]).item()
    for _ in range(200):
        net.train();optimizer.zero_grad(set_to_none=True)
        prediction=net(X[:32])
        loss = None  # TODO 6：训练预测与y[:32]的平均MSE。
        if loss is None:return
        pass  # TODO 7：反向传播。
        if net[0].weight.grad is None:
            print('请先计算梯度。');return
        pass  # TODO 8：优化器更新。
    net.eval()
    with torch.no_grad():
        train_mse=F.mse_loss(net(X[:32]),y[:32]).item()
        valid_mse=F.mse_loss(net(X[32:]),y[32:]).item()
    print('两层weight形状:',tuple(net[0].weight.shape),tuple(net[2].weight.shape))
    print('参数总数:',sum(p.numel() for p in net.parameters()))
    print('改变批量大小后的输出:',tuple(net(torch.zeros(7,3,dtype=torch.float64)).shape))
    print('初始训练/最终训练/验证MSE:',initial,train_mse,valid_mse)
    print('解释1：为什么首次前向之前不能直接统计全部参数元素？')
    print('解释2：no_grad试运行为什么不等于完成训练？')
    print('解释3：输入改成4维会发生什么？怎样保留可复用的后续层？')


if __name__=='__main__':main()
