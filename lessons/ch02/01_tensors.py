# 教材 2.1.1：张量、形状和元素数量

import torch

# 当前约定：每个内层列表是一条样本，每个数字是一个特征。
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print("数据：", x)
print("形状：", x.shape)
print("元素总数：", x.numel())
print("第一条样本：", x[0])  # Python 的索引从 0 开始
