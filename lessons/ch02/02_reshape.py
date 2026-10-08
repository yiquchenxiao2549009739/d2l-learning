# 教材 2.1.1：arange、reshape 与推断形状
# 用二维转置对照：形状相同，不代表内容排列相同。

import torch

flat = torch.arange(1, 7)  # 从 1 开始，不包括终点 7
x = flat.reshape(3, 2)

print("一维数据：", flat)
print("原矩阵：", x)
print("重新组织为两行三列：", x.reshape(2, 3))
print("交换行列：", x.T)
print("自动推断行数：", x.reshape(-1, 3))
print("原矩阵形状仍然是：", x.shape)
