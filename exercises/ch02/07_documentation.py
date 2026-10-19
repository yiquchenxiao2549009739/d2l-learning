"""2.7 综合练习：查 torch.max 文档，找出每条样本分数最高的类别。

先在本机帮助或 reports/ch02/2_7_local_help.txt 中找到带 dim 的签名。
三处 TODO 自己填写；不硬编码最大值或类别编号。
"""

import torch


# 三条样本，四个类别；每行是一组人为设定的类别概率。
probabilities = torch.tensor([
    [0.10, 0.60, 0.20, 0.10],
    [0.15, 0.10, 0.55, 0.20],
    [0.70, 0.05, 0.15, 0.10],
], dtype=torch.float64)

# 目标：每条样本分别取最大值，且保留被汇总的轴。
result = None  # TODO 1：查 torch.max，选择合适的签名和参数。
best_scores = None  # TODO 2：从 result 读取最大值张量，目标形状 (3,1)。
best_indices = None  # TODO 3：从 result 读取类别索引张量，目标形状 (3,1)。

print("PyTorch:", torch.__version__)
print("输入概率:\n", probabilities)
for name, value in (("函数返回", result), ("最大概率", best_scores),
                    ("类别索引（从 0 开始）", best_indices)):
    print("\n" + name + ":")
    print("待填写" if value is None else value)
    if isinstance(value, torch.Tensor):
        print("shape:", tuple(value.shape), "dtype:", value.dtype)

print("\n集中反馈：运行输出，以及下面三个解释。")
print("A. 你选择了 torch.max 的哪条签名？为什么不能只写 torch.max(probabilities)？")
print("B. 返回的最大值与索引分别表示什么？keepdim=False 后形状怎样变化？")
print("C. help(torch.max) 与 help(torch.max(probabilities)) 查的是同一个对象吗？")
