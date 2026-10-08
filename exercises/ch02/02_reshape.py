# 教材 2.1.1 练习：先预测，再自己填写三个表达式。

import torch

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

reshaped = None  # TODO：将 x 重新组织为两行三列
transposed = None  # TODO：对二维 x 交换行列
inferred = None  # TODO：指定每行三个数，让程序推断行数

if reshaped is None or transposed is None or inferred is None:
    print("请先填写三个表达式，保存后运行。")
else:
    print("重新组织：", reshaped)
    print("交换行列：", transposed)
    print("推断行数：", inferred)
    print("三个形状：", reshaped.shape, transposed.shape, inferred.shape)
    print("原 x 的形状：", x.shape)

# 请解释：reshape 与转置为什么形状相同，排列却不同？
