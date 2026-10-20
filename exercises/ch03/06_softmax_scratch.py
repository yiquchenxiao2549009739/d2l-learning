# -*- coding: utf-8 -*-
"""3.6 综合实践：37 张真实训练图像，从零训练 10 轮。八处 TODO。"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import FashionMNIST

ROOT=Path(__file__).resolve().parents[2]


def softmax(logits):
    shifted=logits-logits.max(dim=1,keepdim=True).values
    numerator=None  # TODO 1：对平移后的得分取指数。
    denominator=None  # TODO 2：沿类别维求和，保留维度。
    if numerator is None or denominator is None:
        return None
    return numerator/denominator


def losses_from_logits(logits,labels):
    shifted=logits-logits.max(dim=1,keepdim=True).values
    log_probs=shifted-shifted.exp().sum(dim=1,keepdim=True).log()
    losses=None  # TODO 3：选各样本真实类别的 log_probs 并取负号。
    return losses


def main():
    torch.set_num_threads(1)
    raw=FashionMNIST(ROOT/'data',train=True,download=False)
    images=raw.data[:37].unsqueeze(1).float()/255.
    labels=raw.targets[:37]
    loader=DataLoader(TensorDataset(images,labels),batch_size=8,shuffle=True,
                      num_workers=0,generator=torch.Generator().manual_seed(3605))
    weight=(.01*torch.randn(784,10,generator=torch.Generator().manual_seed(3604))).requires_grad_()
    bias=torch.zeros(10,requires_grad=True)
    preview=softmax(torch.zeros(2,10))
    if preview is None:
        print('请集中填写八处 TODO，再运行整个训练实践。');return
    sizes,updates=[],0
    for epoch in range(10):
        for x,y in loader:
            logits=None  # TODO 4：保留样本维展平后，计算 XW+b。
            if logits is None:
                print('请完成模型得分。');return
            losses=losses_from_logits(logits,y)
            if losses is None:
                print('请完成逐样本交叉熵。');return
            pass  # TODO 5：对损失求和，再反向传播。
            if weight.grad is None or bias.grad is None:
                print('请完成反向传播。');return
            with torch.no_grad():
                for parameter in (weight,bias):
                    assert parameter.grad is not None
                    pass  # TODO 6：lr=0.1，除以当前批实际样本数，原地更新。
                    parameter.grad.zero_()
            updates+=1
            if epoch==0:sizes.append(len(y))
    with torch.no_grad():
        final_logits=images.flatten(start_dim=1)@weight+bias
        predicted=None  # TODO 7：每张图取最大得分的类别编号。
        accuracy=None  # TODO 8：得到正确比例。
        final_losses=losses_from_logits(final_logits,labels)
        if final_losses is None:
            print('请完成逐样本交叉熵。');return
        final_loss=final_losses.mean()
    if predicted is None or accuracy is None:
        print('请完成预测和准确率。');return
    print('首轮批量:',sizes,'| 更新次数:',updates)
    print('最终交叉熵:',final_loss.item(),'| 训练准确率:',float(accuracy))
    final_probabilities=softmax(final_logits)
    if final_probabilities is None:
        print('请完成 softmax。');return
    print('概率行和:',final_probabilities[:3].sum(dim=1))
    print('权重形状:',weight.shape,'| 偏置形状:',bias.shape)
    print('集中解释：手写 W 为何是 (784,10)？accuracy 函数的正确数量和比例怎样区分？')
    print('sum 反传后为何除实际批量？为什么不能再把批准确率直接平均？')
    print('概率取 log 会遇到什么问题？小训练集表现好能否说明新图也分得好？')


if __name__=='__main__':
    main()
