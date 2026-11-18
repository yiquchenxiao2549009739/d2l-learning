"""6.5 综合练习。

先完成无填充矩形汇聚，再用内置层与固定平均核核对。
最后对最大汇聚反传，观察输入梯度并实现最小汇聚。
TODO 1-4 为窗口扫描，TODO 5-6 为内置和卷积核对，TODO 7-8 为梯度与最小值。
"""
import torch
from torch.nn import functional as F


def pool2d(X, pool_size, stride, mode):
    kh, kw = pool_size
    sh, sw = stride
    rows = None  # TODO 1：计算能放下多少行完整窗口。
    cols = None  # TODO 2：计算能放下多少列完整窗口。
    if rows is None or cols is None:
        return None
    outputs = []
    for i in range(rows):
        line = []
        for j in range(cols):
            patch = None  # TODO 3：按 i*sh、j*sw 找到并切出窗口。
            if patch is None:
                return None
            if mode == "max":
                flat = patch.reshape(-1)
                value = None  # TODO 4：通过 argmax 索引选择第一个最大值。
                if value is None:
                    return None
            elif mode == "avg":
                value = patch.mean()
            else:
                raise ValueError("mode 只能是 max 或 avg")
            line.append(value)
        outputs.append(torch.stack(line))
    return torch.stack(outputs)


def main():
    print("6.5 先预测形状与首行，再按窗口、核对、反传三个阶段填写。")
    X = torch.arange(1, 21, dtype=torch.float64).reshape(4, 5).requires_grad_()
    size, stride = (2, 2), (1, 2)
    Y = pool2d(X, size, stride, "max")
    if Y is None:
        print("请先完成 TODO 1-4。")
        return
    print("最大汇聚形状、首行", tuple(Y.shape), Y[0].tolist())
    expected = None  # TODO 5：用 F.max_pool2d 计算同窗口、同步幅的二维参考。
    if expected is None:
        print("请完成 TODO 5-6，核对最大与平均汇聚。")
        return
    torch.testing.assert_close(Y, expected)
    average = pool2d(X, size, stride, "avg")
    print("平均汇聚首行", average[0].tolist())
    fixed_kernel = None  # TODO 6：构造每个权重为 1/4 的 (1,1,2,2) 核，dtype 对齐 X。
    if fixed_kernel is None:
        return
    average_conv = F.conv2d(X[None, None], fixed_kernel, stride=stride)[0, 0]
    torch.testing.assert_close(average, average_conv)
    pass  # TODO 7：对全部最大汇聚输出之和反向传播。
    print("最大汇聚的输入梯度", None if X.grad is None else X.grad.tolist())
    minimum = None  # TODO 8：用已知最大汇聚函数计算最小汇聚，窗口和步幅保持一致。
    if minimum is None:
        print("请完成 TODO 7-8，再解释梯度为何只到达部分像素。")
        return
    print("最小汇聚首行", minimum[0].tolist())
    print("再解释汇聚没有可训练参数时，前面卷积的参数怎样收到梯度。")


if __name__ == "__main__":
    main()
