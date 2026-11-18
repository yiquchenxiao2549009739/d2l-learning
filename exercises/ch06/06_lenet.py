"""6.6 综合练习：37 张训练图、20 张验证图，CPU 上贯通小型 LeNet。

先预测各层形状，再分模型/训练/评估三阶段填写八处 TODO。
这是运行流程练习，小样本三轮不要求获得高准确率，不访问测试集。
"""
import runpy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def build_model():
    conv1 = None  # TODO 1：1→6 通道，5×5 核，padding=2 的二维卷积。
    conv2 = None  # TODO 2：6→16 通道，5×5 核，无填充的二维卷积。
    if conv1 is None or conv2 is None:
        return None
    flatten = None  # TODO 3：保留批量轴，展平每张图的其余维度。
    if flatten is None:
        return None
    return nn.Sequential(conv1, nn.ReLU(), nn.AvgPool2d(2, 2),
                         conv2, nn.ReLU(), nn.AvgPool2d(2, 2), flatten,
                         nn.Linear(400, 32), nn.ReLU(), nn.Linear(32, 10))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(6600)
    net = build_model()
    if net is None:
        print("请先完成 TODO 1-3，并预测两个汇聚层后的形状。")
        return
    shape = net(torch.zeros(8, 1, 28, 28)).shape
    print("输出形状", tuple(shape), "参数", sum(p.numel() for p in net.parameters()))
    helper = runpy.run_path(str(ROOT / "lessons/ch06/06_lenet.py"))
    splits, _ = helper["load_data"](include_test=False)
    train = helper["balanced_subset"](splits["train"], 37, 6601)
    validation = helper["balanced_subset"](splits["validation"], 20, 6602)
    loader = helper["make_loader"](train, shuffle=True, batch_size=8)
    optimizer = None  # TODO 4：SGD 管理 net.parameters()，学习率为 .1。
    if optimizer is None:
        print("请完成 TODO 4-7，再运行三轮小批量训练。")
        return
    updates = 0
    for epoch in range(3):
        net.train()
        loss_sum = 0.
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = net(X)
            loss = None  # TODO 5：直接由 logits 和 long 标签计算批平均交叉熵。
            if loss is None:
                print("请继续填写 TODO 5-7。")
                return
            pass  # TODO 6：反向传播平均损失。
            pass  # TODO 7：执行一次优化器更新。
            loss_sum += loss.item() * len(y)
            updates += 1
        print("epoch", epoch + 1, "在线训练 CE", loss_sum / len(train))
    prediction = None
    net.eval()
    with torch.no_grad():
        X, y = next(iter(helper["make_loader"](validation, batch_size=20)))
        logits = net(X)
        prediction = None  # TODO 8：每行取最大 logits 的类别索引。
    if prediction is None:
        print("请完成 TODO 8，并解释 eval 与 no_grad 的不同职责。")
        return
    print("更新次数", updates, "验证正确数", int((prediction == y).sum()), "/", len(y))
    assert updates == 15
    print("每轮批次为 8/8/8/8/5。解释尾批为什么不用再除以 8。")


if __name__ == "__main__":
    main()
