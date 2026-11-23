"""7.4综合练习：两个Inception小网络、原始28输入、37/20图、三轮15更新。

八处TODO按分支→拼接→训练填写，不冒称完整九块GoogLeNet。
"""
from pathlib import Path
import runpy
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]


class Inception(nn.Module):
    def __init__(self,ci,c1,c2,c3,c4):
        super().__init__()
        self.p1=None  # TODO 1：ci→c1的1×1卷积；ReLU已在forward中。
        self.p2=None  # TODO 2：1×1降维到c2[0]再3×3到c2[1]，各接ReLU。
        self.p3=None  # TODO 3：1×1降维到c3[0]再5×5到c3[1]，各接ReLU。
        self.p4=None  # TODO 4：3×3/stride1/padding1最大汇聚，再1×1到c4和ReLU。
        self.ready=all(m is not None for m in (self.p1,self.p2,self.p3,self.p4))

    def forward(self,X):
        branches=[F.relu(self.p1(X)),self.p2(X),self.p3(X),self.p4(X)]
        output=None  # TODO 5：把四个分支按通道维拼接。
        return output


def main():
    torch.set_num_threads(4);torch.manual_seed(7400)
    a=Inception(8,4,(4,6),(2,2),4)
    b=Inception(16,6,(4,8),(2,2),4)
    if not a.ready or not b.ready:
        print('先完成TODO 1-4：让四分支保持相同高宽。');return
    if a(torch.zeros(2,8,14,14)) is None:
        print('请完成TODO 5：按通道拼接，而不是逐元素相加。');return
    net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        a,nn.MaxPool2d(2),b,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(20,10))
    for j,m in enumerate(m for m in net.modules() if isinstance(m,(nn.Conv2d,nn.Linear))):
        torch.default_generator.manual_seed(7400+j);nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    print('参数',sum(p.numel() for p in net.parameters()),'输出',tuple(net(torch.zeros(8,1,28,28)).shape))
    helper=runpy.run_path(str(ROOT/'lessons/ch06/06_lenet.py'))
    splits,_=helper['load_data'](include_test=False)
    train=helper['balanced_subset'](splits['train'],37,7401)
    val=helper['balanced_subset'](splits['validation'],20,7402)
    loader=helper['make_loader'](train,shuffle=True,batch_size=8)
    optimizer=torch.optim.SGD(net.parameters(),lr=.1);updates=0
    for epoch in range(3):
        net.train();total=0.
        for X,y in loader:
            optimizer.zero_grad(set_to_none=True);logits=net(X)
            loss=None  # TODO 6：logits与long标签的批平均交叉熵。
            if loss is None:print('请接通TODO 6-8：损失、反传与更新。');return
            pass  # TODO 7：反向传播。
            pass  # TODO 8：执行一次SGD更新。
            updates+=1;total+=loss.item()*len(y)
        print('epoch',epoch+1,'在线CE',total/len(train))
    net.eval()
    with torch.no_grad():
        X,y=next(iter(helper['make_loader'](val,batch_size=20)))
        prediction=net(X).argmax(1)
    assert updates==15
    print('更新',updates,'验证正确数',int((prediction==y).sum()),'/',len(y))
    print('解释cat为何保留四分支，以及输入梯度为何相加；短预算不用于证明容量。')


if __name__=='__main__':main()
