# 教材 2.1.1 练习：亲手创建三条样本、每条两个特征的数据。

import torch

# TODO：把 None 替换为 torch.tensor(...)。
# 里面放三个列表，每个列表两个数值，数值自己选择。
x = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
]
x = torch.tensor(x)

if x is None:
    print("请先创建张量，保存后运行。")
else:
    print("数据：", x)
    print("形状：", x.shape)
    print("元素总数：", x.numel())
    print("第一条样本：", x[0])

# 在聊天中报告结果：和两条样本、每条三个特征的示例相比，
# 哪些数量发生变化？哪些没变？
