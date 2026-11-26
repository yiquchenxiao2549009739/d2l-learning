"""7.7 综合练习。接通稠密块、过渡层和小网络训练，填写八处 TODO。"""
from pathlib import Path
import runpy
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


class ExerciseDense(nn.Module):
    def __init__(self, ci, k, layers):
        super().__init__()
        self.units = nn.ModuleList()
        for i in range(layers):
            width = None  # TODO 1：第 i 个单元需要接收的通道数。
            if width is None:
                print('先完成 TODO 1，计算各单元输入通道。')
                return
            self.units.append(nn.Sequential(nn.BatchNorm2d(width), nn.ReLU(),
                                            nn.Conv2d(width, k, 3, padding=1)))
        self.ready = True

    def forward(self, X):
        for unit in self.units:
            Y = None  # TODO 2：在目前的全部特征上计算新的 k 个通道。
            if Y is None:
                print('接通 TODO 2-3 的卷积和拼接。')
                return None
            joined = None  # TODO 3：把旧特征和新特征拼到通道轴。
            if joined is None:
                print('完成 TODO 3。')
                return None
            X = joined
        return X


def make_transition(ci, co):
    projection = None  # TODO 4：用1×1卷积把通道从 ci 改成 co。
    pooling = None  # TODO 5：用窗口2、步幅2的平均汇聚。
    if projection is None or pooling is None:
        print('完成 TODO 4-5 的过渡层。')
        return None
    return nn.Sequential(nn.BatchNorm2d(ci), nn.ReLU(), projection, pooling)


def main():
    torch.set_num_threads(4)
    first = ExerciseDense(4,3,2)
    if not getattr(first,'ready',False):
        return
    with torch.no_grad():
        probe = first.eval()(torch.zeros(2,4,9,9))
    if probe is None:
        return
    assert tuple(probe.shape) == (2,10,9,9)
    shrink = make_transition(10,5)
    if shrink is None:
        return
    net = nn.Sequential(nn.Conv2d(1,4,3,padding=1),first,shrink,ExerciseDense(5,3,2),
                        nn.BatchNorm2d(11),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(11,10))
    for i,m in enumerate(m for m in net.modules() if isinstance(m,(nn.Conv2d,nn.Linear))):
        torch.manual_seed(7700+i)
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    common = runpy.run_path(str(ROOT/'lessons/ch06/06_lenet.py'))
    splits,_ = common['load_data'](include_test=False)
    train = common['balanced_subset'](splits['train'],37,7701)
    val = common['balanced_subset'](splits['validation'],20,7702)
    loader = common['make_loader'](train,shuffle=True,batch_size=8)
    opt = torch.optim.SGD(net.parameters(),lr=.05)
    updates = 0
    for epoch in range(3):
        net.train()
        total = 0.
        for X,y in loader:
            opt.zero_grad(set_to_none=True)
            logits = net(X)
            loss = None  # TODO 6：批平均交叉熵。
            if loss is None:
                print('完成 TODO 6-8 的训练更新。')
                return
            pass  # TODO 7：从损失做反向传播。
            pass  # TODO 8：按梯度做一次SGD更新。
            updates += 1
            total += loss.item()*len(y)
        print('epoch',epoch+1,'在线CE',total/len(train))
    result = common['evaluate'](net,common['make_loader'](val),'cpu')
    assert updates == 15
    print('参数',sum(p.numel() for p in net.parameters()),'更新',updates,'验证',result)


if __name__ == '__main__':
    main()
