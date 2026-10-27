# -*- coding: utf-8 -*-
"""4.3 综合练习：37张真实图像，784→32→10，六个None和两个pass。"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import FashionMNIST

ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1);torch.manual_seed(4307)
    raw=FashionMNIST(ROOT/'data',train=True,download=False)
    images=raw.data[:37].unsqueeze(1).float()/255.;labels=raw.targets[:37]
    loader=DataLoader(TensorDataset(images,labels),batch_size=8,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(4308))
    flatten=None  # TODO 1：保留样本维的Flatten。
    hidden_layer=None  # TODO 2：Linear，784输入、32输出。
    activation=None  # TODO 3：ReLU模块。
    output_layer=None  # TODO 4：Linear，32输入、10输出。
    if flatten is None or hidden_layer is None or activation is None or output_layer is None:
        print('请集中填写八处TODO，再运行完整框架MLP。');return
    net=nn.Sequential(flatten,hidden_layer,activation,output_layer)

    def init_weights(module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,std=.01);nn.init.zeros_(module.bias)
    net.apply(init_weights)
    loss_fn=None  # TODO 5：CrossEntropyLoss，reduction='none'。
    optimizer=None  # TODO 6：SGD，net.parameters()，lr=.1。
    if loss_fn is None or optimizer is None:
        print('请完成交叉熵与优化器。');return
    sizes,updates=[],0
    for epoch in range(30):
        net.train()
        for X,y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=net(X)
            losses=loss_fn(logits,y)
            pass  # TODO 7：平均损失反向传播。
            if any(p.grad is None for p in net.parameters()):
                print('请让四组参数得到梯度。');return
            pass  # TODO 8：优化器更新。
            updates+=1
            if epoch==0:sizes.append(len(y))
    net.eval()
    with torch.no_grad():
        logits=net(images)
        accuracy=(logits.argmax(1)==labels).float().mean().item()
        ce=loss_fn(logits,labels).mean().item()
    print('输入/得分:',images.shape,logits.shape,'| 标签:',labels.shape,labels.dtype)
    print('参数形状:',{name:tuple(p.shape) for name,p in net.named_parameters()})
    print('参数数目:',sum(p.numel() for p in net.parameters()),'| 首轮批量:',sizes,'| 更新次数:',updates)
    print('最终训练CE:',ce,'| 训练准确率:',accuracy)
    print('集中解释：Sequential怎样对应手写两层？weight方向为什么要转置？')
    print('apply和parameters分别做什么？为什么模型末尾不放Softmax？')
    print('eval与no_grad有什么区别？比较激活/初始化时如何控制其余条件？')


if __name__=='__main__':main()
