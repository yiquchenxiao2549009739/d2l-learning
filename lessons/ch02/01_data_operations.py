"""教材 2.1 数据操作的完整演示：一次运行全部内容。

已有张量与 reshape 示例整合在这里。重复旧文件已清理。
对应 PDF 第 58-65 页，正文第 40-47 页。
"""

import numpy as np
import torch

torch.manual_seed(42)


def show(name, value):
    """只负责显示结果；本次不要求自己编写这个辅助函数。"""
    print(f"\n{name}:\n{value}")


print("\n===== 1. 创建、属性和形状 =====")

# 沿用你已经完成的三条样本、每条两个特征的写法。
data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
X = torch.tensor(data, dtype=torch.float32)
show("X", X)
print("shape / ndim / numel:", X.shape, X.ndim, X.numel())
print("dtype / device:", X.dtype, X.device)
show("arange：包括起点，不包括终点", torch.arange(1, 7))
show("全 0", torch.zeros(3, 2))
show("全 1", torch.ones(3, 2))
show("标准正态随机数", torch.randn(3, 2))
show("reshape 为两行三列", X.reshape(2, 3))
show("-1 推断行数", X.reshape(-1, 3))
show("二维转置：交换行列", X.T)
print("原 X 的形状仍是:", X.shape)


print("\n===== 2. 按元素运算、比较、求和和拼接 =====")

Y = torch.ones_like(X) * 2
show("Y", Y)
show("X + Y", X + Y)
show("X - Y", X - Y)
show("X * Y：按位置相乘", X * Y)
show("X / Y", X / Y)
show("X ** 2：平方", X ** 2)
show("exp：以 e 为底的指数", torch.exp(X))
show("X == Y", X == Y)
show("X > Y", X > Y)
show("X < Y", X < Y)
show("全部元素求和", X.sum())
show("沿轴 0 求和：每列的和", X.sum(dim=0))
show("沿轴 1 求和：每行的和", X.sum(dim=1))
show("每行求和，保留长度为 1 的轴", X.sum(dim=1, keepdim=True))
show("沿轴 0 拼接：增加行", torch.cat((X, Y), dim=0))
show("沿轴 1 拼接：增加列", torch.cat((X, Y), dim=1))


print("\n===== 3. 广播：不同形状怎样一起计算 =====")

offset = torch.tensor([10.0, 100.0])
show("X + 每个特征的偏移量", X + offset)
show("X + 标量", X + 10)
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
show("a，形状 (3,1)", a)
show("b，形状 (1,2)", b)
show("a + b，形状 (3,2)", a + b)
a3 = torch.arange(6).reshape(2, 1, 3)
b3 = torch.arange(4).reshape(1, 4, 1)
show("三维广播，结果形状 (2,4,3)", a3 + b3)

# 故意演示不能广播的形状；捕获这个预期错误后程序继续。
try:
    incompatible = torch.ones(2, 3) + torch.ones(2)
except RuntimeError as error:
    print("\n预期的广播错误：", error)


print("\n===== 4. 索引、切片、筛选和赋值 =====")

show("第一行 X[0]", X[0])
show("最后一行 X[-1]", X[-1])
show("第 2、3 行 X[1:3]", X[1:3])
show("第 2 行第 1 列 X[1,0]", X[1, 0])
show("第 2 列，得到一维 X[:,1]", X[:, 1])
show("第 2 列，保留二维 X[:,1:2]", X[:, 1:2])
show("前两行、所有列 X[:2,:]", X[:2, :])
show("条件筛选 X[X > 3]", X[X > 3])
modified = X.clone()  # 独立复制后修改，原 X 保留用于后续演示。
modified[1, 0] = 99
modified[:2, :] = 12
show("单个和一片元素赋值后的复制", modified)
show("原 X 未改变", X)


print("\n===== 5. 重新绑定、原地修改、切片和复制 =====")

rebinding = torch.tensor([1.0, 2.0])
old_reference = rebinding
old_id = id(rebinding)
rebinding = rebinding + 10
print("重新绑定后对象身份相同:", id(rebinding) == old_id)
show("新对象", rebinding)
show("原引用仍指向旧对象", old_reference)

inplace = torch.tensor([1.0, 2.0])
inplace_alias = inplace
old_id = id(inplace)
inplace += 10
print("原地更新后对象身份相同:", id(inplace) == old_id)
show("原引用也能看到更新", inplace_alias)

source = torch.tensor([1.0, 2.0, 3.0])
alias = source
view = source[1:]  # 本例的基本切片与 source 共享底层存储。
snapshot = source.clone()  # 独立复制。
view[0] = 99
show("修改切片后 source", source)
show("同一个对象的 alias", alias)
show("独立 snapshot", snapshot)

Z = torch.zeros_like(X)
old_id = id(Z)
Z[:] = X + Y
print("切片赋值保留目标对象:", id(Z) == old_id)
show("Z[:] = X + Y", Z)


print("\n===== 6. NumPy 与 Python 数值的转换 =====")

numpy_source = torch.tensor([1.0, 2.0, 3.0])
array = numpy_source.numpy()  # 本例 CPU 张量与这个数组共享存储。
shared = torch.from_numpy(array)  # 共享数组存储。
copied = torch.tensor(array)  # 复制数据，独立存储。
array[0] = 88
show("NumPy 数组修改后", array)
show("原张量也改变", numpy_source)
show("from_numpy 的共享张量也改变", shared)
show("torch.tensor 的复制仍是原值", copied)
independent_array = numpy_source.numpy().copy()
independent_array[1] = 77
show("显式复制的 NumPy 数组", independent_array)
show("原张量第 2 个数仍不变", numpy_source)
scalar = torch.tensor([3.5])
print("item / float / int:", scalar.item(), float(scalar), int(scalar))
print("NumPy 数组类型:", type(array), "元素类型:", array.dtype)
standalone_array = np.array([10.0, 20.0], dtype=np.float32)
show("直接从 NumPy 创建的数组转换", torch.from_numpy(standalone_array))


print("\n===== 7. 综合应用：同一批数据的计算 =====")

feature_mean = X.mean(dim=0)
centered = X - feature_mean  # (3,2) - (2,)，广播到每条样本。
row_total = X.sum(dim=1, keepdim=True)
augmented = torch.cat((X, row_total), dim=1)
show("每个特征的均值", feature_mean)
show("每列减去自己的均值", centered)
show("每条样本添加总和特征", augmented)

# 连接前面学过的损失：数值相同也需要正确的形状。
prediction = X[:, 1:2]  # (3,1)
target = X[:, 1]  # (3,)
wrong_error = prediction - target  # 意外广播成 (3,3)。
correct_error = prediction - target.reshape(-1, 1)
print("错误形状的误差:", wrong_error.shape, "MSE:", wrong_error.square().mean().item())
print("对齐形状的误差:", correct_error.shape, "MSE:", correct_error.square().mean().item())
print("\n完整演示结束。完成综合练习后，一次反馈代码结果和概念题。")
