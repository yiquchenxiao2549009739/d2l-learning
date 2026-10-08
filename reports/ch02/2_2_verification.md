# 完整 2.2 数据预处理：教师运行记录

日期：2026-10-08（Asia/Shanghai）。使用已有 openmmlab 环境，pandas 2.0.3；无新增安装。

运行 `lessons/ch02/02_data_preprocessing.py`：

- 原始示例四行三列，NumRooms 缺失 2 项，Alley 缺失 3 项，Price 无缺失。
- 已知房间数为 2、4，均值填补为 3。
- 独热编码的列顺序为 NumRooms、Alley_Pave、Alley_nan。
- X 为 (4,3)，y 为 (4,1)，都为 float32 且不含 NaN。
- 特征值为 [[3,1,0],[2,0,1],[4,0,1],[3,0,1]]，标签与源 CSV 按行一致。
- 对原始表直接 dropna 得到空表，说明删除所有含缺失的行会损失全部样本。

完整输出：`2_2_demo_output.txt`。处理后的特征 CSV：`2_2_processed_features.csv`。

综合练习使用另一个六行数据表，教师没有填写用户的作答。尚未据此标记用户已掌握本节。
