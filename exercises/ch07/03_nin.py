"""7.3 综合练习：三块小NiN，原始28输入、37/20样本、三轮15次更新。

九Conv、无Linear，不冒称教材完整224模型。八处TODO分阶段填写。
"""
from pathlib import Path
import runpy
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]


def nin_block(in_channels,out_channels):
    spatial=None  # TODO 1：3×3空间卷积，padding=1。
    point1=None  # TODO 2：out_channels到自身的1×1卷积。
    point2=None  # TODO 3：再构造一个独立的1×1卷积。
    if any(x is None for x in (spatial,point1,point2)):return None
    return nn.Sequential(spatial,nn.ReLU(),point1,nn.ReLU(),point2,nn.ReLU())


def main():
    torch.set_num_threads(4);torch.manual_seed(7300)
    blocks=[nin_block(1,8),nin_block(8,16),nin_block(16,10)]
    if any(x is None for x in blocks):
        print("先完成TODO 1-3：空间卷积与两个1×1卷积。")
        return
    gap=None  # TODO 4：全局平均汇聚到1×1，保留每个通道。
    flatten=None  # TODO 5：只展平通道和空间维，保留批量。
    if gap is None or flatten is None:
        print("请完成TODO 4-5：把(批量,10,7,7)变为(批量,10)。")
        return
    net=nn.Sequential(blocks[0],nn.MaxPool2d(2,2),blocks[1],nn.MaxPool2d(2,2),
        nn.Dropout(.5),blocks[2],gap,flatten)
    for index,layer in enumerate(m for m in net.modules() if isinstance(m,nn.Conv2d)):
        torch.default_generator.manual_seed(7300+index)
        nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
    print("参数",sum(p.numel() for p in net.parameters()),"输出",tuple(net(torch.zeros(8,1,28,28)).shape))
    helper=runpy.run_path(str(ROOT/'lessons/ch06/06_lenet.py'))
    splits,_=helper['load_data'](include_test=False)
    train=helper['balanced_subset'](splits['train'],37,7301)
    validation=helper['balanced_subset'](splits['validation'],20,7302)
    loader=helper['make_loader'](train,shuffle=True,batch_size=8)
    optimizer=torch.optim.SGD(net.parameters(),lr=.1)
    torch.manual_seed(7303);updates=0
    for epoch in range(3):
        net.train();total=0.
        for X,y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=net(X)
            loss=None  # TODO 6：logits与long标签的批平均交叉熵。
            if loss is None:
                print("请接通TODO 6-8：损失、反传和更新。")
                return
            pass  # TODO 7：反向传播。
            pass  # TODO 8：执行一次SGD更新。
            total+=loss.item()*len(y);updates+=1
        print("epoch",epoch+1,"在线CE",total/len(train))
    net.eval()
    with torch.no_grad():
        X,y=next(iter(helper['make_loader'](validation,batch_size=20)))
        prediction=net(X).argmax(1)
    assert updates==15
    print("更新",updates,"验证正确数",int((prediction==y).sum()),"/",len(y))
    print("解释空间共享、1×1的作用、全局平均与最后ReLU；不要用三轮小样本分数判断模型容量。")


if __name__=='__main__':main()
