"""7.5综合练习：补齐批量统计与仿射变换，接小CNN训练；八处TODO留给自己。

37/20图，batch8、三轮15更新，无孤立尾样本。正式模型和练习模型分开。
"""
from pathlib import Path
import runpy
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]


class ExerciseBN(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(features));self.bias=nn.Parameter(torch.zeros(features))
        self.register_buffer('running_mean',torch.zeros(features));self.register_buffer('running_var',torch.ones(features))
        self.register_buffer('num_batches_tracked',torch.tensor(0,dtype=torch.long))

    def forward(self,X):
        shape=(1,-1) if X.ndim==2 else (1,-1,1,1)
        if self.training:
            axes=None  # TODO 1：(N,C)与(N,C,H,W)分别应在哪些轴上统计？
            if axes is None:print('先完成TODO 1：保留通道轴。');return None
            mean=None  # TODO 2：按axes求均值，keepdim=True。
            var=None  # TODO 3：按axes求有偏方差，keepdim=True。
            if mean is None or var is None:print('请完成TODO 2-3的均值与方差。');return None
            normalized=None  # TODO 4：用sqrt(var+1e-5)规范化X。
            if normalized is None:print('请完成TODO 4，注意eps在平方根内部。');return None
        else:
            normalized=(X-self.running_mean.reshape(shape))/torch.sqrt(self.running_var.reshape(shape)+1e-5)
        out=None  # TODO 5：weight/bias广播到shape，完成缩放和平移。
        if out is None:print('请完成TODO 5的仿射变换。');return None
        if self.training:
            m=X.numel()//X.shape[1]
            if m<2:raise ValueError('每通道统计元素至少2个')
            with torch.no_grad():
                self.running_mean.mul_(.9).add_(mean.reshape(-1),alpha=.1)
                self.running_var.mul_(.9).add_(var.reshape(-1)*m/(m-1),alpha=.1)
                self.num_batches_tracked.add_(1)
        return out


def main():
    torch.set_num_threads(4);torch.manual_seed(7500)
    probe=ExerciseBN(4)
    if probe(torch.arange(12.).reshape(3,4)) is None:return
    net=nn.Sequential(nn.Conv2d(1,4,3,padding=1),ExerciseBN(4),nn.Sigmoid(),nn.AvgPool2d(2),
        nn.Conv2d(4,8,3,padding=1),ExerciseBN(8),nn.Sigmoid(),nn.AvgPool2d(2),nn.Flatten(),
        nn.Linear(392,16),ExerciseBN(16),nn.Sigmoid(),nn.Linear(16,10))
    for j,m in enumerate(m for m in net.modules() if isinstance(m,(nn.Conv2d,nn.Linear))):
        torch.default_generator.manual_seed(7500+j);nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    net.eval();print('参数',sum(p.numel() for p in net.parameters()),'输出',tuple(net(torch.zeros(8,1,28,28)).shape))
    common=runpy.run_path(str(ROOT/'lessons/ch06/06_lenet.py'))
    splits,_=common['load_data'](include_test=False)
    train=common['balanced_subset'](splits['train'],37,7501);val=common['balanced_subset'](splits['validation'],20,7502)
    loader=common['make_loader'](train,shuffle=True,batch_size=8);optimizer=torch.optim.SGD(net.parameters(),lr=.1);updates=0
    for epoch in range(3):
        net.train();total=0.
        for X,y in loader:
            optimizer.zero_grad(set_to_none=True);logits=net(X)
            loss=None  # TODO 6：logits与long标签的批平均交叉熵。
            if loss is None:print('请接通TODO 6-8：损失、反传和更新。');return
            pass  # TODO 7：反向传播。
            pass  # TODO 8：执行SGD更新。
            total+=loss.item()*len(y);updates+=1
        print('epoch',epoch+1,'在线CE',total/len(train))
    net.eval()
    with torch.no_grad():
        X,y=next(iter(common['make_loader'](val,batch_size=20)));pred=net(X).argmax(1)
    assert updates==15
    print('更新',updates,'验证正确数',int((pred==y).sum()),'/',len(y),'BN批计数',net[1].num_batches_tracked.item())
    print('解释参数/缓冲区、统计轴、有偏/无偏方差与eval/no_grad的区别。')


if __name__=='__main__':main()
