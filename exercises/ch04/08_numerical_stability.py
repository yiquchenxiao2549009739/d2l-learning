# -*- coding: utf-8 -*-
"""4.8综合练习：三层隐藏网络，比较初始化；八处TODO保持留空。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def build_model(mode):
    dimensions=[2,16,16,16,2]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4809)
        parts=[]
        for i,(fan_in,fan_out) in enumerate(zip(dimensions[:-1],dimensions[1:])):
            parts.append(nn.Linear(fan_in,fan_out,dtype=torch.float64))
            if i<3:parts.append(nn.ReLU())
        net=nn.Sequential(*parts)
    generator=torch.Generator().manual_seed(4809)
    with torch.no_grad():
        layers=[layer for layer in net if isinstance(layer,nn.Linear)]
        for i,layer in enumerate(layers):
            fan_in,fan_out=layer.in_features,layer.out_features
            xavier_std=None  # TODO 1：gain=1的Xavier正态标准差。
            kaiming_std=None  # TODO 2：ReLU、fan_in模式的标准差。
            output_std=None  # TODO 3：统一的线性输出层，方差取1/fan_in。
            if any(value is None for value in (xavier_std,kaiming_std,output_std)):
                print('请集中填写八处TODO，再运行完整初始化对照。');return None
            sigma={'small':.01,'xavier':xavier_std,'kaiming':kaiming_std}[mode] if i<3 else output_std
            base=torch.randn(layer.weight.shape,generator=generator,dtype=torch.float64)
            weight=None  # TODO 4：把同一标准正态基底缩放为目标标准差。
            if weight is None:
                print('请先构造权重。');return None
            layer.weight.copy_(weight)
            pass  # TODO 5：把当前层的bias清零。
    return net


def make_data():
    generator=torch.Generator().manual_seed(4808)
    signs=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=torch.float64)
    result=[]
    for count in (24,12):
        directions=signs.repeat_interleave(count,dim=0)
        X=directions*(.4+1.2*torch.rand(directions.shape,generator=generator,dtype=torch.float64))
        y=(X[:,0]*X[:,1]<0).long()
        result.append((X,y))
    return result


def main():
    torch.set_num_threads(1)
    (X,y),(VX,Vy)=make_data()
    for mode in ('small','xavier','kaiming'):
        net=build_model(mode)
        if net is None:return
        optimizer=torch.optim.SGD(net.parameters(),lr=.1)
        print('方案/参数数/初始最大bias:',mode,sum(p.numel() for p in net.parameters()),
              max(m.bias.abs().max().item() for m in net if isinstance(m,nn.Linear)))
        first_gradient=None
        for step in range(200):
            optimizer.zero_grad(set_to_none=True)
            logits=net(X)
            loss=None  # TODO 6：批量平均交叉熵，接原始logits。
            if loss is None:
                print('请先接通损失。');return
            pass  # TODO 7：反向传播。
            if net[0].weight.grad is None:
                print('请先计算梯度，再更新参数。');return
            if step==0:first_gradient=net[0].weight.grad.norm().item()
            pass  # TODO 8：执行一次优化器更新。
        with torch.no_grad():
            train=net(X);validation=net(VX)
            print('初始第一层梯度范数:',first_gradient)
            print('最终训练CE/验证CE/验证正确数:',F.cross_entropy(train,y).item(),
                  F.cross_entropy(validation,Vy).item(),int((validation.argmax(1)==Vy).sum()),'/',len(Vy))
    print('解释：小初始化的准确率与CE是否给出同样的信息？ReLU为什么仍需要控制权重尺度？')
    print('解释：标准差和方差有什么区别？为什么允许bias为0，却不把这个深层网络所有权重都置0？')


if __name__=='__main__':main()
