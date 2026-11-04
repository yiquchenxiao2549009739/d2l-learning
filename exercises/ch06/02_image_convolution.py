"""6.2 综合练习，串起滑窗、目标响应和卷积核训练。

先按讲义复现教师的左右差分，再在本文件处理上下亮度变化。
TODO 1–4 实现完整窗口，TODO 5 构造目标核，TODO 6–8 接通训练。
先预测输出形状和第一列，运行后记录初始/最终损失及核的变化。
本练习使用平均平方误差，教师演示使用平方误差和，比较时留意尺度。
"""
import torch
from torch import nn


def corr2d(X, K):
    h, w = K.shape
    rows = None  # TODO 1：完整窗口能放的行数。
    cols = None  # TODO 2：完整窗口能放的列数。
    if rows is None or cols is None:
        return None
    outputs = []
    for i in range(rows):
        line = []
        for j in range(cols):
            patch = None  # TODO 3：切出以 (i, j) 为左上角的窗口。
            if patch is None:
                return None
            value = None  # TODO 4：窗口与核逐元素相乘再求和。
            if value is None:
                return None
            line.append(value)
        outputs.append(torch.stack(line))
    return torch.stack(outputs)


def main():
    print("6.2 综合练习。先判断上减下会在哪里产生非零响应。")
    print("按滑窗、目标、训练三个阶段填写八处 TODO，并记录运行结果。")
    torch.manual_seed(6281)
    X = torch.ones(8, 6)
    X[2:6, :] = 0
    K = None  # TODO 5：构造形状 (2, 1) 的核，计算上减下。
    if K is None:
        print("请先填写滑窗的 TODO 1–4，再构造竖向差分核。")
        return
    target = corr2d(X, K)
    if target is None:
        print("请检查 corr2d 中的 TODO 1–4。")
        return
    print("输出形状和第一列", tuple(target.shape), target[:, 0].tolist())

    model = nn.Conv2d(1, 1, (2, 1), bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.9)
    x4, y4 = X[None, None], target[None, None]
    for step in range(24):
        prediction = model(x4)
        loss = None  # TODO 6：计算预测与目标的均方误差（取 mean）。
        if loss is None:
            print("请补上 TODO 6–8，完成训练。")
            return
        optimizer.zero_grad(set_to_none=True)  # 每次反传前清除上次留下的梯度。
        pass  # TODO 7：从损失反向传播。
        pass  # TODO 8：用优化器更新参数。
        if step == 0:
            print("初始 MSE", loss.item())
    with torch.no_grad():
        print("更新后的 MSE", (model(x4) - y4).square().mean().item())
        print("学到的核", model.weight[0, 0].tolist())


if __name__ == "__main__":
    main()
