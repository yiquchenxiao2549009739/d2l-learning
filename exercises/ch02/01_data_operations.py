"""教材 2.1 综合练习：一次完成，再集中反馈。

前面单独的 tensor 与 reshape 练习已整合，重复旧文件已清理。
本次只需要填写这里的表达式，并解释最后的观察题。
"""

import torch

# 沿用你已完成的三条样本、每条两个特征。
data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
X = torch.tensor(data, dtype=torch.float32)

# 集中填写下面八个表达式。不会的查完整演示，不用另开小课。
reshaped = None  # 1. 将 X 重新组织为两行三列。
transposed = None  # 2. 对二维 X 交换行列。
centered = None  # 3. 每个元素减去它所在列的均值，结果仍是 (3,2)。
row_total = None  # 4. 每条样本的两个特征求和，保留 (3,1) 形状。
augmented = None  # 5. 把总和作为新特征拼到 X 后面，得到 (3,3)。
last_two = None  # 6. 取出 X 的最后两条样本，保留全部特征。
mask = None  # 7. 产生与 X 同形状的布尔张量，标记大于 3 的位置。

a = torch.arange(6).reshape(2, 1, 3)
b = torch.arange(4).reshape(1, 4, 1)
broadcasted = None  # 8. 计算 a 与 b 的按元素和，先预测结果形状。

# 以下是显示区，不要求自己编写。它会指出哪些表达式还没填写。
results = {
    "重新组织": reshaped,
    "交换行列": transposed,
    "按列中心化": centered,
    "每行总和": row_total,
    "添加总和特征": augmented,
    "最后两条样本": last_two,
    "大于 3 的位置": mask,
    "三维广播": broadcasted,
}
for name, value in results.items():
    print("\n" + name + ":")
    if isinstance(value, torch.Tensor):
        print(value)
        print("形状：", value.shape)
    elif value is None:
        print("待填写")
    else:
        print("请让表达式返回张量，而不是把结果写成普通数字或列表。")

print("\n===== 内存与转换观察题：先预测，再看输出 =====")
original = torch.tensor([1.0, 2.0, 3.0])
alias = original
view = original[1:]
snapshot = original.clone()
array = original.numpy()
shared = torch.from_numpy(array)
copied = torch.tensor(array)
array[0] = 99
view[0] = 77
print("original：", original)
print("alias：", alias)
print("snapshot：", snapshot)
print("from_numpy：", shared)
print("torch.tensor(array)：", copied)

print("\n一次反馈：八个表达式的结果，以及下面三个概念问题。")
print("A. reshape 与转置的形状一样，为什么排列不同？")
print("B. (3,1) 的预测减去 (3,) 的标签会产生什么形状？如何对齐？")
print("C. 上面的哪些对象跟着 original 改变，哪些保持独立？为什么？")
