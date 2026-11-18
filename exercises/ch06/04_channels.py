"""6.4 综合练习。

先按讲义复现教材双通道例，再在本题完成新的双输入、双输出卷积。
TODO 1-3 连接输入通道求和，TODO 4-6 连接多输出、偏置及内置核对，
TODO 7-8 用矩阵乘法计算 1×1 通道映射。先预测形状，再填写和运行。
"""
import runpy
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
corr2d = runpy.run_path(str(ROOT / "lessons/ch06/03_padding_and_stride.py"))["corr2d"]


def multi_in(X, K):
    parts = []
    for channel in range(X.shape[0]):
        image = None  # TODO 1：取出一个输入通道的二维图。
        kernel = None  # TODO 2：取出与它对应的二维核。
        if image is None or kernel is None:
            return None
        parts.append(corr2d(image, kernel))
    response = None  # TODO 3：将 parts 叠放，再沿输入通道轴求和。
    return response


def multi_out(X, K, bias):
    outputs = []
    for channel in range(K.shape[0]):
        response = None  # TODO 4：用 multi_in 计算这一组核的响应。
        if response is None:
            return None
        outputs.append(response)
    outputs = torch.stack(outputs)
    Y = None  # TODO 5：给每个输出通道加对应偏置，并广播到全部空间位置。
    return Y


def main():
    print("6.4 先预测输出高宽、通道数和含偏置参数量，再按三个阶段填写。")
    first = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    X = torch.stack((first, first + 1))
    base = torch.tensor([[[1., 0.], [0., -1.]], [[0., 1.], [-1., 0.]]], dtype=X.dtype)
    K = torch.stack((base, base + 1))
    bias = torch.tensor([.5, -1.], dtype=X.dtype)
    single = multi_in(X, base)
    if single is None:
        print("请先完成 TODO 1-3，核对单输出的形状和首格。")
        return
    print("第一阶段单输出形状、首格", tuple(single.shape), single[0, 0].item())
    Y = multi_out(X, K, bias)
    if Y is None:
        print("请完成 TODO 4-5，组织多输出并添加偏置。")
        return
    print("输出形状、各通道左上值", tuple(Y.shape), Y[:, 0, 0].tolist())
    builtin = None  # TODO 6：添加批量轴，用 F.conv2d 核对并取回 CHW。
    if builtin is None:
        print("请补上 TODO 6。")
        return
    torch.testing.assert_close(Y, builtin)
    print("含偏置参数量", K.numel() + bias.numel())
    point_weight = torch.tensor([[1., 0.], [0., 1.], [1., -1.]], dtype=X.dtype)
    point_bias = torch.tensor([0., .25, -.5], dtype=X.dtype)
    columns = None  # TODO 7：把 X 重排成 (2,16)，每列为一个像素的通道向量。
    if columns is None:
        print("请补上 TODO 7-8，完成 1×1 映射。")
        return
    point_output = None  # TODO 8：矩阵乘法加偏置，再恢复 (3,4,4)。
    if point_output is None:
        return
    expected = F.conv2d(X[None], point_weight[:, :, None, None], point_bias)[0]
    torch.testing.assert_close(point_output, expected)
    print("1×1 输出形状及左上通道向量", tuple(point_output.shape), point_output[:, 0, 0].tolist())
    print("再解释输入通道为什么相加、输出通道为什么叠放、偏置怎样共享。")


if __name__ == "__main__":
    main()
