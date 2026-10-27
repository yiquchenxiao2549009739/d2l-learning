# -*- coding: utf-8 -*-
"""4.6综合练习：37张图片、784→32→16→10，集中填写八处TODO。"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[2]


def dropout_layer(X, p, training=True):
    if not 0 <= p <= 1: raise ValueError('p必须在0到1之间')
    if not training or p == 0: return X
    if p == 1: return X*0
    mask = None  # TODO 1：同形状均匀随机数大于p，得到保留掩码。
    if mask is None: return None
    output = None  # TODO 2：X乘掩码，除以保留概率。
    return output


class ManualDropout(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = p

    def forward(self, X):
        return dropout_layer(X, self.p, self.training)


def main():
    torch.set_num_threads(1); torch.manual_seed(4607)
    if dropout_layer(torch.ones(2, 4), .5) is None:
        print('请集中补齐八处TODO，再运行整节暂退练习。'); return
    raw = FashionMNIST(ROOT/'data', train=True, download=False)
    images = raw.data[:37].unsqueeze(1).float()/255.; labels = raw.targets[:37]
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True, num_workers=0,
                        drop_last=False, generator=torch.Generator().manual_seed(4608))
    net = None  # TODO 3：Flatten，Linear784→32，ReLU，手写暂退.2；Linear32→16，ReLU，手写暂退.5；Linear16→10。
    if net is None:
        print('请完成两个隐藏层及暂退模块。'); return
    linears = [module for module in net if isinstance(module, nn.Linear)]
    for index, module in enumerate(linears):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu' if index<2 else 'linear')
        nn.init.zeros_(module.bias)
    loss_fn = None  # TODO 4：CrossEntropyLoss。
    optimizer = None  # TODO 5：SGD，lr=.1。
    if loss_fn is None or optimizer is None:
        print('请连接损失和优化器。'); return
    sizes, updates = [], 0
    for epoch in range(30):
        pass  # TODO 6：切回训练模式，激活Dropout。
        if not net.training:
            print('每轮评估后，需要切回训练模式。'); return
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(X), y)
            pass  # TODO 7：反向传播。
            if any(p.grad is None for p in net.parameters()):
                print('请让参数得到梯度。'); return
            pass  # TODO 8：更新参数。
            updates += 1
            if epoch == 0: sizes.append(len(y))
        net.eval()
        with torch.no_grad():
            logits = net(images)
            ce = loss_fn(logits, labels).item()
            accuracy = (logits.argmax(1)==labels).float().mean().item()
    with torch.no_grad(): repeated = net(images)
    print('输入/得分/标签:', images.shape, logits.shape, labels.shape)
    print('参数数目:', sum(p.numel() for p in net.parameters()), '| 首轮批量:', sizes, '| 更新次数:', updates)
    print('训练集eval重算CE/准确率:', ce, accuracy, '| eval重复一致:', torch.equal(logits,repeated))
    print('解释：为什么除以1-p？p是丢弃还是保留概率？')
    print('解释：train/eval与no_grad分别控制什么？为什么不能只用训练在线准确率比较两模型？')


if __name__ == '__main__':
    main()
