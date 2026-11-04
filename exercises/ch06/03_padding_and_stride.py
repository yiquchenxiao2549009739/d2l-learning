"""6.3 综合练习，将填充、输出尺寸和窗口起点放进同一段程序。

先按讲义复现对称填充与步幅 (2,3)，再处理本题四侧不同的填充。
TODO 1–3 准备填充网格并计算输出尺寸，TODO 4–7 定位并计算窗口，
TODO 8 用内置运算核对。运行前在纸上画出第一格和最后一格。
反馈形状、第一行、核对误差，以及边界补零为何出现在这些窗口中。
"""
import torch
from torch.nn import functional as F


def corr2d(X, K, padding, stride):
    # padding 顺序为左、右、上、下，stride 顺序为高、宽。
    padded = None  # TODO 1：用 F.pad 给 X 零填充。
    if padded is None:
        return None
    kh, kw = K.shape
    sh, sw = stride
    rows = None  # TODO 2：用整除计算输出行数。
    cols = None  # TODO 3：用整除计算输出列数。
    if rows is None or cols is None:
        return None
    outputs = []
    for i in range(rows):
        line = []
        for j in range(cols):
            row = None  # TODO 4：第 i 行窗口在 padded 中的起始行。
            col = None  # TODO 5：第 j 列窗口在 padded 中的起始列。
            if row is None or col is None:
                return None
            patch = None  # TODO 6：切出 kh×kw 窗口。
            if patch is None:
                return None
            value = None  # TODO 7：窗口乘核再求和。
            if value is None:
                return None
            line.append(value)
        outputs.append(torch.stack(line))
    return torch.stack(outputs)


def main():
    print("6.3 综合练习。先写出补零后高宽，再列出所有合法窗口起点。")
    print("按准备网格、扫描窗口、内置核对三个阶段填写八处 TODO。")
    X = torch.arange(1, 31, dtype=torch.float64).reshape(5, 6)
    K = torch.tensor([[1., 0., -1.], [1., 0., -1.]], dtype=X.dtype)
    padding, stride = (1, 2, 0, 1), (2, 3)
    Y = corr2d(X, K, padding, stride)
    if Y is None:
        print("先预测形状和第一行，再填写 TODO 1–7。")
        return
    reference = None  # TODO 8：对填充后的四维输入用 F.conv2d，并取回二维输出。
    if reference is None:
        print("请补上 TODO 8，用内置运算核对。")
        return
    torch.testing.assert_close(Y, reference)
    print("输出形状", tuple(Y.shape))
    print("第一行", Y[0].tolist())
    print("完整响应", Y.tolist())
    print("最大核对误差", (Y - reference).abs().max().item())
    print("再解释每个窗口在原始输入中的起点，以及最后一行用了哪些补零。")


if __name__ == "__main__":
    main()
