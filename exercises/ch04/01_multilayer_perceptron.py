# -*- coding: utf-8 -*-
"""4.1 综合实践：24条XOR训练样本，8隐藏单元，八处TODO。"""
from pathlib import Path
import csv
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    path=ROOT/"datasets/ch04/01_xor_train.csv"
    with path.open(encoding="utf-8",newline="") as handle:
        rows=list(csv.DictReader(handle))[:24]
    X=torch.tensor([[float(r['x1']),float(r['x2'])] for r in rows],dtype=torch.float64)
    y=torch.tensor([int(r['label']) for r in rows],dtype=torch.long)
    g=torch.Generator().manual_seed(4107)
    W1=(.4*torch.randn(2,8,generator=g,dtype=torch.float64)).requires_grad_()
    b1=torch.zeros(8,dtype=torch.float64,requires_grad=True)
    W2=(.4*torch.randn(8,2,generator=g,dtype=torch.float64)).requires_grad_()
    b2=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.SGD([W1,b1,W2,b2],lr=.1)

    def forward():
        Z=None  # TODO 1：第一层仿射变换，X、W1、b1。
        if Z is None:return None,None
        H=None  # TODO 2：逐元素ReLU。
        if H is None:return None,None
        logits=None  # TODO 3：第二层仿射变换，H、W2、b2。
        return H,logits

    for epoch in range(300):
        optimizer.zero_grad(set_to_none=True)
        H,logits=forward()
        if H is None or logits is None:
            print("请集中填写八处 TODO，再运行整个MLP实践。");return
        loss=None  # TODO 4：原始得分与y，默认平均交叉熵。
        if loss is None:
            print("请完成交叉熵。");return
        pass  # TODO 5：反向传播。
        if W1.grad is None or W2.grad is None:
            print("请完成反向传播。");return
        pass  # TODO 6：优化器更新。
    with torch.no_grad():
        H,logits=forward()
        if H is None or logits is None:return
        predicted=None  # TODO 7：按类别维取最大得分索引。
        accuracy=None  # TODO 8：与y比较并求正确比例。
        final_ce=nn.CrossEntropyLoss()(logits,y).item()
    if predicted is None or accuracy is None:
        print("请完成预测与准确率。");return
    print("X/H/logits:",X.shape,H.shape,logits.shape)
    print("参数数目:",sum(p.numel() for p in (W1,b1,W2,b2)),"| 更新次数:",300)
    print("最终训练CE:",final_ce,"| 训练准确率:",float(accuracy))
    print("集中解释：去掉ReLU后，两层为什么能合并？输入层计入参数层数吗？")
    print("ReLU负半轴的梯度是什么？原点可导吗？sigmoid饱和时怎样影响梯度？")
    print("24条训练样本全对，是否证明泛化或通用近似定理？跨batch归一化有什么问题？")


if __name__=="__main__":
    main()
