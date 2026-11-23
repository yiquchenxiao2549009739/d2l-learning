"""7.2综合练习：用块搭建三块小网络，原始28图、37/20样本、三轮15更新。

这是五卷积加两Linear的小实践，七个含权重层，不冒称完整五块VGG-11。
八处TODO按块→分类头→训练三阶段填写，源文件不由教师代填。
"""
import runpy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        conv = None  # TODO 1：当前输入通道到out_channels的3×3卷积，padding=1。
        next_in = None  # TODO 2：下一卷积的输入通道应取什么值？
        if conv is None or next_in is None:
            return None
        layers.extend([conv, nn.ReLU()])
        in_channels = next_in
    pool = None  # TODO 3：2×2、stride=2的最大汇聚。
    if pool is None:
        return None
    layers.append(pool)
    block = None  # TODO 4：把layers解包成一个nn.Sequential。
    return block


def main():
    torch.set_num_threads(4)
    torch.manual_seed(7200)
    blocks, in_channels = [], 1
    for count, out_channels in ((1, 8), (2, 16), (2, 32)):
        block = vgg_block(count, in_channels, out_channels)
        if block is None:
            print("先完成TODO 1-4，构造并注册块中的所有层。")
            return
        blocks.append(block)
        in_channels = out_channels
    in_features = None  # TODO 5：原始28图经三次汇聚后的展平宽度。
    if in_features is None:
        print("请完成TODO 5：先算28→14→7→3，再接分类头。")
        return
    net = nn.Sequential(*blocks, nn.Flatten(), nn.Linear(in_features, 32),
                        nn.ReLU(), nn.Dropout(.5), nn.Linear(32, 10))
    for index, layer in enumerate(m for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))):
        torch.default_generator.manual_seed(7200 + index)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    print("参数", sum(p.numel() for p in net.parameters()), "输出", tuple(net(torch.zeros(8, 1, 28, 28)).shape))
    helper = runpy.run_path(str(ROOT / "lessons/ch06/06_lenet.py"))
    splits, _ = helper["load_data"](include_test=False)
    train = helper["balanced_subset"](splits["train"], 37, 7201)
    validation = helper["balanced_subset"](splits["validation"], 20, 7202)
    loader = helper["make_loader"](train, shuffle=True, batch_size=8)
    optimizer = torch.optim.SGD(net.parameters(), lr=.05)
    torch.manual_seed(7203)
    updates = 0
    for epoch in range(3):
        net.train()
        total = 0.
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = net(X)
            loss = None  # TODO 6：由logits和long标签求批平均交叉熵。
            if loss is None:
                print("请接通TODO 6-8的损失、反传与更新。")
                return
            pass  # TODO 7：对loss反向传播。
            pass  # TODO 8：执行一次优化器更新。
            total += loss.item() * len(y)
            updates += 1
        print("epoch", epoch + 1, "在线CE", total / len(train))
    net.eval()
    with torch.no_grad():
        X, y = next(iter(helper["make_loader"](validation, batch_size=20)))
        prediction = net(X).argmax(1)
    assert updates == 15
    print("更新", updates, "验证正确数", int((prediction == y).sum()), "/", len(y))
    print("解释代码复用为何没有共享权重、288从哪里来，以及train/eval怎样改变Dropout。")


if __name__ == "__main__":
    main()
