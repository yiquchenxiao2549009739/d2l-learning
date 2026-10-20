# -*- coding: utf-8 -*-
"""3.7 综合实践：37 张已有图像，八处 TODO 接通框架训练。"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(3707)
    raw = FashionMNIST(ROOT/"data", train=True, download=False)
    images = raw.data[:37].unsqueeze(1).float()/255.
    labels = raw.targets[:37]
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(3708))
    flatten = None  # TODO 1：构造保留批次维的展平层。
    linear = None  # TODO 2：构造 784 输入、10 输出的线性层。
    if flatten is None or linear is None:
        print("请集中填写八处 TODO，再运行完整训练。");return
    net = nn.Sequential(flatten, linear)

    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0., std=.01)
            nn.init.zeros_(module.bias)

    net.apply(init_weights)
    loss_fn = None  # TODO 3：交叉熵，reduction='none'。
    optimizer = None  # TODO 4：SGD，net.parameters()，学习率0.1。
    if loss_fn is None or optimizer is None:
        print("请完成损失函数和优化器。");return
    sizes, updates = [], 0
    for epoch in range(10):
        net.train()
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = None  # TODO 5：图片送入模型，保留原始得分。
            if logits is None:
                print("请完成模型前向计算。");return
            losses = loss_fn(logits, y)
            pass  # TODO 6：逐样本损失取平均，然后反向传播。
            if linear.weight.grad is None:
                print("请完成反向传播。");return
            pass  # TODO 7：让优化器执行一次更新。
            updates += 1
            if epoch == 0:
                sizes.append(len(y))
    net.eval()
    with torch.no_grad():
        final_logits = net(images)
        accuracy = None  # TODO 8：按行 argmax，与标签比较，求正确比例。
        final_ce = loss_fn(final_logits, labels).mean()
        probabilities = final_logits.softmax(dim=1)
    if accuracy is None:
        print("请完成准确率。");return
    print("首轮批量:", sizes, "| 总更新次数:", updates)
    print("最终交叉熵:", final_ce.item(), "| 训练准确率:", float(accuracy))
    print("得分形状:", final_logits.shape, "| 标签形状/类型:", labels.shape, labels.dtype)
    print("weight/bias:", linear.weight.shape, linear.bias.shape)
    print("前三行概率和:", probabilities[:3].sum(dim=1))
    print("集中解释：为何模型末尾没有 Softmax？weight 为何是 (10,784)？")
    print("none/mean/sum 的区别？mean 反传后是否再除8，尾批5怎样处理？")
    print("zero_grad/backward/step 各做什么？eval 与 no_grad 能否互相替代？")
    print("小训练集表现好，能否说明新图片表现好？")


if __name__ == "__main__":
    main()
