"""7.6综合练习：把残差支路、投影、相加、损失和更新串起来；八处TODO待填。"""
from pathlib import Path
import runpy
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]


class ExerciseResidual(nn.Module):
    def __init__(self,ci,co,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(ci,co,3,stride,1);self.bn1=nn.BatchNorm2d(co)
        self.conv2=nn.Conv2d(co,co,3,padding=1);self.bn2=nn.BatchNorm2d(co)
        self.shortcut=nn.Conv2d(ci,co,1,stride) if ci!=co or stride!=1 else nn.Identity()

    def forward(self,X):
        Y=None  # TODO 1：第一层Conv→BN→ReLU。
        if Y is None:print('先完成TODO 1-5，接通残差块。');return None
        Y=None  # TODO 2：第二层Conv→BN，此处先不做ReLU。
        if Y is None:print('完成TODO 2。');return None
        skip=None  # TODO 3：让X通过shortcut；Identity或投影已在构造时选定。
        if skip is None:print('完成TODO 3。');return None
        assert Y.shape==skip.shape, '逐元素相加禁止广播'
        combined=None  # TODO 4：逐元素相加Y和skip。
        if combined is None:print('完成TODO 4。');return None
        output=None  # TODO 5：加法后ReLU。
        if output is None:print('完成TODO 5。');return None
        return output


def main():
    torch.set_num_threads(4)
    probe=ExerciseResidual(4,8,2).eval()
    if probe(torch.zeros(2,4,9,9)) is None:return
    net=nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.BatchNorm2d(4),nn.ReLU(),
        ExerciseResidual(4,4),ExerciseResidual(4,8,2),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(8,10))
    for j,m in enumerate(m for m in net.modules() if isinstance(m,(nn.Conv2d,nn.Linear))):
        torch.manual_seed(7600+j);nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    net.eval()
    with torch.no_grad():print('参数',sum(p.numel() for p in net.parameters()),'输出',tuple(net(torch.zeros(8,1,28,28)).shape))
    h=runpy.run_path(str(ROOT/'lessons/ch06/06_lenet.py'));splits,_=h['load_data'](include_test=False)
    train=h['balanced_subset'](splits['train'],37,7601);val=h['balanced_subset'](splits['validation'],20,7602)
    loader=h['make_loader'](train,shuffle=True,batch_size=8);optimizer=torch.optim.SGD(net.parameters(),lr=.05);updates=0
    for epoch in range(3):
        net.train();total=0.
        for X,y in loader:
            optimizer.zero_grad(set_to_none=True);logits=net(X)
            loss=None  # TODO 6：logits和long标签的批平均交叉熵。
            if loss is None:print('完成TODO 6-8的损失、反传和更新。');return
            pass  # TODO 7：反向传播。
            pass  # TODO 8：SGD更新。
            updates+=1;total+=loss.item()*len(y)
        print('epoch',epoch+1,'在线CE',total/len(train))
    result=h['evaluate'](net,h['make_loader'](val),'cpu');assert updates==15
    print('更新',updates,'验证',result,'首BN计数',net[1].num_batches_tracked.item())
    print('解释恒等支路、投影、加法与cat、末ReLU和BN状态。')


if __name__=='__main__':main()
