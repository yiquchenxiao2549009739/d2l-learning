"""教材 2.3 综合实践：四条样本、三个特征、两个输出。

集中把 None 换成表达式，不硬编码结果；下方显示区不用修改。
示例值是人为设置的权重，只做前向计算，不代表已经训练。
"""

import torch

torch.set_printoptions(precision=4, sci_mode=False)

X = torch.tensor([[1.0, 2.0, 1.0],
                  [2.0, 1.0, 3.0],
                  [3.0, 2.0, 2.0],
                  [4.0, 3.0, 1.0]])
W = torch.tensor([[2.0, -1.0], [0.5, 1.0], [-1.0, 2.0]])
b = torch.tensor([0.5, -0.5])
y = torch.tensor([[3.0, 2.0], [1.0, 5.0], [4.0, 4.0], [8.0, 1.0]])

# 同一条流程的十个表达式；可以集中全部填写后运行。
feature_mean = None  # 1. 每个特征在四条样本中的均值，形状 (3,)。
row_sums = None  # 2. 每条样本的特征和，保留轴，形状 (4,1)。
row_normalized = None  # 3. 每行除以自己的和；此数据行和都非零。
first_score = None  # 4. 只用点积算第 0 条样本的第 0 个输出，含偏置。
pred = None  # 5. 一次算完四条样本的两个输出，形状 (4,2)。
error = None  # 6. 预测减去真实值，形状 (4,2)。
mae = None  # 7. 所有误差元素绝对值的平均。
mse = None  # 8. 所有误差元素平方的平均。
weight_fro = None  # 9. 整个权重矩阵的 Frobenius 范数。
sample_distance = None  # 10. 第 0、1 条样本之间的欧氏距离。

# 显示区：模板原样运行也不会报错。填写后逐项显示结果与形状。
results = {
    "每列均值": feature_mean,
    "每行和": row_sums,
    "按行归一化": row_normalized,
    "单样本单输出点积": first_score,
    "完整预测": pred,
    "误差": error,
    "MAE": mae,
    "MSE": mse,
    "权重 Frobenius 范数": weight_fro,
    "两条样本的距离": sample_distance,
}
for name, value in results.items():
    print("\n" + name + ":")
    print("待填写" if value is None else value)
    if isinstance(value, torch.Tensor):
        print("形状:", tuple(value.shape))

print("\n集中反馈三个问题：")
print("A. X*W 与 X@W 有何不同？为什么 (4,3)@(3,2) 得到 (4,2)？")
print("B. 为什么逐行归一化保留 (4,1)？若方阵不保留轴为何可能静默算错？")
print("C. W 有 3 行 2 列，它的 Frobenius 范数为什么不是 3 或 2？")
print("再用自己的话说：这十步中哪些在处理样本，哪些在度量误差？")
