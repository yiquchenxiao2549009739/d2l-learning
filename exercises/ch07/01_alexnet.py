"""7.1 综合练习：直接读取28×28的小CNN，三阶段填写八处TODO。

37张训练/20张验证，CPU三轮、batch8。先预测形状，再接通前向、反传与评估。
"""
import runpy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def build_model():
    conv1 = None  # TODO 1：1→16通道，3×3核、padding=1。
    conv2 = None  # TODO 2：16→32通道，3×3核、padding=1。
    conv3 = None  # TODO 3：32→64通道，3×3核、padding=1。
    flatten = None  # TODO 4：保留批量轴，展平每张图的其他轴。
    if any(layer is None for layer in (conv1, conv2, conv3, flatten)):
        return None
    return nn.Sequential(conv1, nn.ReLU(), nn.MaxPool2d(2, 2),
                         conv2, nn.ReLU(), nn.MaxPool2d(2, 2),
                         conv3, nn.ReLU(), nn.MaxPool2d(2, 2), flatten,
                         nn.Linear(576, 64), nn.ReLU(), nn.Dropout(.5), nn.Linear(64, 10))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(7100)
    net = build_model()
    if net is None:
        print("请先完成TODO 1-4，预测三次汇聚后的高宽14/7/3。")
        return
    for layer in net:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    print("参数", sum(p.numel() for p in net.parameters()))
    helper = runpy.run_path(str(ROOT / "lessons/ch06/06_lenet.py"))
    splits, _ = helper["load_data"](include_test=False)
    train = helper["balanced_subset"](splits["train"], 37, 7101)
    validation = helper["balanced_subset"](splits["validation"], 20, 7102)
    loader = helper["make_loader"](train, shuffle=True, batch_size=8)
    optimizer = torch.optim.SGD(net.parameters(), lr=.05)
    updates = 0
    for epoch in range(3):
        net.train()
        total = 0.
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = net(X)
            loss = None  # TODO 5：logits和long标签的批平均交叉熵。
            if loss is None:
                print("请完成TODO 5-7，接通训练循环。")
                return
            pass  # TODO 6：对loss反向传播。
            pass  # TODO 7：执行一次SGD更新。
            total += loss.item() * len(y)
            updates += 1
        print("epoch", epoch + 1, "在线训练CE", total / len(train))
    net.eval()
    with torch.no_grad():
        X, y = next(iter(helper["make_loader"](validation, batch_size=20)))
        logits = net(X)
        prediction = None  # TODO 8：逐行找最大logits对应的类别索引。
    if prediction is None:
        print("请完成TODO 8，再解释eval怎样改变Dropout。")
        return
    assert updates == 15
    print("总更新", updates, "验证正确数", int((prediction == y).sum()), "/", len(y))
    print("解释576的来源、尾批5、训练与评估Dropout行为，以及为何不需要放大到224。")


if __name__ == "__main__":
    main()
