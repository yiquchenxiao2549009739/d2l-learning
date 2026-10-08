"""教材 2.2 综合练习：集中完成整个预处理流程。

先读演示，再填写表达式。CSV 已准备，不需要下载数据或安装软件。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
data = pd.read_csv(ROOT / "datasets" / "ch02" / "house_practice.csv")
print("原始表：\n", data)
print("缺失数量：\n", data.isna().sum())

# 下面集中填写，不要改原始房价，不要硬编码每列的均值。
inputs = None  # 1. 选择除 Price 以外的输入特征，并创建独立副本。
targets = None  # 2. 取出 Price 作为标签。
most_missing_column = None  # 3. 找出输入特征中缺失值最多的列名。
reduced = None  # 4. 从输入特征中删除这列。
filled = None  # 5. 数值列用各自均值填补；类别缺失留到下一步。
encoded = None  # 6. 类别独热编码，并将缺失作为一种类别状态。
X = None  # 7. 转为 float32 特征张量。
y = None  # 8. 转为 float32 标签张量，形状为 (样本数,1)。

# 以下是显示区。初始状态会提示待填写，完成后展示结果。
results = {
    "输入特征": inputs,
    "真实标签": targets,
    "缺失最多的列名": most_missing_column,
    "删列后": reduced,
    "填补后": filled,
    "编码后": encoded,
    "特征张量": X,
    "标签张量": y,
}
for name, value in results.items():
    print("\n" + name + ":")
    if value is None:
        print("待填写")
    else:
        print(value)
        if isinstance(value, (pd.DataFrame, pd.Series, torch.Tensor)):
            print("形状：", value.shape)
        if isinstance(value, torch.Tensor):
            print("类型：", value.dtype, "含 NaN：", torch.isnan(value).any().item())

print("\n一次反馈预处理结果及三个问题：")
print("A. 房间数量的缺失值为什么不能默认填 0？均值填补在做什么假设？")
print("B. 无序类别直接编码为 0、1、2，会给数值模型引入什么关系？独热编码怎样表示？")
print("C. 以后有训练集和测试集时，填补均值应该从哪里计算？为什么？")
