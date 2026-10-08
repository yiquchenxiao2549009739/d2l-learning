"""教材完整 2.2：CSV -> 缺失值处理 -> 类别编码 -> PyTorch 张量。

一次运行整块。演示使用书中的四条房屋样本，综合练习使用更大的表。
对应 PDF 第 65-68 页，正文第 47-50 页。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "datasets" / "ch02" / "house_tiny.csv"
REPORT_DIR = ROOT / "reports" / "ch02"


def show(name, value):
    """显示辅助函数；本次不要求自己实现它。"""
    print(f"\n{name}:\n{value}")


print("\n===== 1. 从 CSV 读取原始数据 =====")
data = pd.read_csv(DATA_FILE)
show("原始数据", data)
show("列的数据类型", data.dtypes)
show("每列缺失数量", data.isna().sum())
print("pandas 版本:", pd.__version__)
print("原始表形状:", data.shape)


print("\n===== 2. 先区分输入特征与真实标签 =====")
inputs = data.iloc[:, :2].copy()  # 前两列：房间数量、巷子类型。
targets = data.iloc[:, 2].copy()  # 第三列：房价。
show("输入特征", inputs)
show("真实标签", targets)


print("\n===== 3. 数值缺失：用已知数据的均值填补 =====")
room_mean = inputs["NumRooms"].mean()  # 默认跳过 NaN：(2+4)/2=3。
inputs["NumRooms"] = inputs["NumRooms"].fillna(room_mean)
print("已知房间数量的均值:", room_mean)
show("填补后的输入", inputs)
show("剩余缺失数量", inputs.isna().sum())


print("\n===== 4. 类别特征：独热编码，缺失作为一种状态 =====")
encoded = pd.get_dummies(
    inputs,
    columns=["Alley"],
    dummy_na=True,
    dtype=np.float32,
)
show("编码后的特征", encoded)
print("特征列顺序:", list(encoded.columns))


print("\n===== 5. 转成数值数组，再转成张量 =====")
feature_array = encoded.to_numpy(dtype=np.float32)
target_array = targets.to_numpy(dtype=np.float32)
if not np.isfinite(feature_array).all() or not np.isfinite(target_array).all():
    raise ValueError("转换前仍有缺失值或无穷值，请检查预处理。")
X = torch.tensor(feature_array, dtype=torch.float32)
y = torch.tensor(target_array, dtype=torch.float32).reshape(-1, 1)
show("特征张量 X", X)
show("标签张量 y", y)
print("X / y 形状:", X.shape, y.shape)
print("X / y 数据类型:", X.dtype, y.dtype)
print("X / y 还有 NaN:", torch.isnan(X).any().item(), torch.isnan(y).any().item())

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = REPORT_DIR / "2_2_processed_features.csv"
encoded.to_csv(OUTPUT_FILE, index=False)
print("已保存清洗后的特征表:", OUTPUT_FILE.relative_to(ROOT))


print("\n===== 6. 对照删除法：删除行与删除列不同 =====")
show("直接删除所有含缺失值的行，当前表会清空", data.dropna())
show("只按房间数量是否缺失筛选行", data.dropna(subset=["NumRooms"]))

raw_inputs = data.iloc[:, :2].copy()
missing_counts = raw_inputs.isna().sum()
most_missing_column = missing_counts.idxmax()
print("输入特征中缺失最多的列:", most_missing_column)
show("删除该列后的另一种方案", raw_inputs.drop(columns=[most_missing_column]))
print("删除法是另一种选择，本演示的主流程采用填补与编码，仍保留四条样本。")


print("\n===== 7. 综合练习的较大数据表 =====")
practice_data = pd.read_csv(ROOT / "datasets" / "ch02" / "house_practice.csv")
show("综合练习原始数据", practice_data)
show("综合练习每列缺失数量", practice_data.isna().sum())
print("\n整节演示结束。完成综合练习后，一次反馈结果和三个原理问题。")
