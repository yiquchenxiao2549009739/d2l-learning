# -*- coding: utf-8 -*-
"""4.2 综合实践：37张已有服装图，784→32→10，八处TODO。"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import FashionMNIST

ROOT=Path(__file__).resolve().parents[2]


def relu(X):
    result=None  # TODO 1：X与zeros_like(X)逐元素取最大值。
    return result


def main():
    torch.set_num_threads(1)
    raw=FashionMNIST(ROOT/'data',train=True,download=False)
    images=raw.data[:37].unsqueeze(1).float()/255.
    labels=raw.targets[:37]
    loader=DataLoader(TensorDataset(images,labels),batch_size=8,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(4208))
    g=torch.Generator().manual_seed(4207)
    W1=None  # TODO 2：nn.Parameter包装784×32的.01倍正态随机权重，使用g。
    b1=nn.Parameter(torch.zeros(32))
    W2=None  # TODO 3：nn.Parameter包装32×10的.01倍正态随机权重，使用g。
    b2=nn.Parameter(torch.zeros(10))
    if W1 is None or W2 is None:
        print('请集中填写八处TODO，再运行完整MLP实践。');return
    parameters=[W1,b1,W2,b2]
    optimizer=torch.optim.SGD(parameters,lr=.1)
    loss_fn=nn.CrossEntropyLoss(reduction='none')

    def forward(X):
        X=X.reshape(len(X),-1)
        H=None  # TODO 4：第一层仿射后调用relu。
        if H is None:return None,None
        logits=None  # TODO 5：第二层仿射，输出原始得分。
        return H,logits

    sizes,updates=[],0
    for epoch in range(30):
        for X,y in loader:
            optimizer.zero_grad(set_to_none=True)
            H,logits=forward(X)
            if H is None or logits is None:
                print('请完成手写ReLU与两层前向。');return
            losses=loss_fn(logits,y)
            pass  # TODO 6：平均损失反向传播。
            if any(p.grad is None for p in parameters):
                print('请让四组参数都参与反向传播。');return
            pass  # TODO 7：优化器更新，不重复除批量大小。
            updates+=1
            if epoch==0:sizes.append(len(y))
    with torch.no_grad():
        H,logits=forward(images)
        if H is None or logits is None:return
        predicted=logits.argmax(dim=1)
        accuracy=None  # TODO 8：与labels比较，计算正确比例。
        final_ce=loss_fn(logits,labels).mean().item()
    if accuracy is None:
        print('请完成训练准确率。');return
    print('X/H/logits:',images.reshape(len(images),-1).shape,H.shape,logits.shape)
    print('参数形状:',[tuple(p.shape) for p in parameters],'| 参数总数:',sum(p.numel() for p in parameters))
    print('首轮批量:',sizes,'| 更新次数:',updates)
    print('训练CE:',final_ce,'| 训练准确率:',float(accuracy))
    print('集中解释：Parameter与普通requires_grad张量有什么关系，为什么显式传params？')
    print('mean反传后还除批量吗？手写max-ReLU在0处和torch.relu的梯度一样吗？')
    print('如何公平比较隐藏宽度/层数/学习率，为什么不能用测试集选最佳配置？')


if __name__=='__main__':
    main()
