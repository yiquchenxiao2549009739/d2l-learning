# -*- coding: utf-8 -*-
"""4.7综合练习：5条人工样本、隐藏偏置、不惩罚偏置，八处TODO。"""
import torch
from torch.nn import functional as F


def main():
    torch.set_num_threads(1)
    X=torch.tensor([[1.,-2.],[-1.,2.],[.5,1.],[-.5,-1.],[1.,1.3]],dtype=torch.float64)
    y=torch.tensor([1,0,1,0,0],dtype=torch.long)
    W1=torch.tensor([[.5,.2],[-.3,.4],[.1,-.2]],dtype=torch.float64)
    W2=torch.tensor([[.2,-.1,.3],[-.4,.2,-.2]],dtype=torch.float64)
    b1=torch.tensor([.05,-.1,.1],dtype=torch.float64)
    lambd=.1
    Z=None  # TODO 1：第一层仿射变换，带隐藏偏置。
    H=None  # TODO 2：ReLU。
    O=None  # TODO 3：输出原始得分。
    if Z is None or H is None or O is None:
        print('请集中填写八处TODO，接通完整前向与反向。');return
    L=F.cross_entropy(O,y)
    S=lambd*(W1.square().sum()+W2.square().sum())/2
    P=O.softmax(1);Y=F.one_hot(y,num_classes=2).to(X.dtype)
    GO=None  # TODO 4：平均交叉熵对O的梯度。
    if GO is None:
        print('请先计算输出梯度。');return
    GW2=None  # TODO 5：输出权重的数据梯度加正则梯度。
    GZ=None  # TODO 6：传回隐藏层，乘ReLU局部导数。
    if GW2 is None or GZ is None:
        print('请沿局部链式法则继续反传。');return
    GW1=None  # TODO 7：第一层权重的总梯度。
    Gb1=None  # TODO 8：隐藏偏置梯度，不加lambda*b1。
    if GW1 is None or Gb1 is None:
        print('请接通第一层权重与偏置梯度。');return
    a=W1.clone().requires_grad_();b=W2.clone().requires_grad_();c=b1.clone().requires_grad_()
    logits=torch.relu(X@a.T+c)@b.T
    J=F.cross_entropy(logits,y)+lambd*(a.square().sum()+b.square().sum())/2
    J.backward()
    error=max((GW1-a.grad).abs().max().item(),(GW2-b.grad).abs().max().item(),(Gb1-c.grad).abs().max().item())
    updated_W1=W1-.1*GW1;updated_W2=W2-.1*GW2;updated_b1=b1-.1*Gb1
    updated=F.cross_entropy(torch.relu(X@updated_W1.T+updated_b1)@updated_W2.T,y)
    updated+=lambd*(updated_W1.square().sum()+updated_W2.square().sum())/2
    print('Z/H/O形状:',Z.shape,H.shape,O.shape,'| 参数数:',W1.numel()+W2.numel()+b1.numel())
    print('L/S/J:',L.item(),S.item(),(L+S).item())
    print('W1/W2/b1梯度形状:',GW1.shape,GW2.shape,Gb1.shape,'| 与autograd最大误差:',error)
    print('一步更新后J:',updated.item())
    print('解释：权重梯度为什么有两条路径？ReLU关闭时，权重总梯度一定为0吗？')
    print('解释：backward与step分别做什么？为什么训练通常需要更多内存？')


if __name__=='__main__':main()
