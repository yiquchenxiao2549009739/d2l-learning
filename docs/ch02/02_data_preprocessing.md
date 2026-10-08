# 2.2 数据预处理：完整一块讲解与实践

对应教材 PDF 第 65–68 页，正文第 47–50 页，覆盖 2.2.1–2.2.3 和教材末尾扩展练习。

本节一次完成：CSV -> 特征与标签 -> 数值缺失处理 -> 类别编码 -> 数值数组 -> 张量。使用本机已有 pandas 2.0.3，不安装或升级软件。

- 完整演示：`lessons/ch02/02_data_preprocessing.py`
- 综合练习：`exercises/ch02/02_data_preprocessing.py`
- 原始示例：`datasets/ch02/house_tiny.csv`
- 扩展练习数据：`datasets/ch02/house_practice.csv`

CSV 是小型人工教学数据，放在可同步的 datasets 目录；以后下载的大型数据仍使用本地 data 目录。

## 为什么需要预处理

上一节的数据已经是整齐的数值张量。真实输入常包含文件、字符串、缺失项和不同数据类型，需要先转换成模型能计算的数值结构。

DataFrame 是 pandas 的表格对象，有行、列名和各列的数据类型。CSV 用逗号分隔字段，一行是一条记录。

沿用教材四条房屋样本：

| NumRooms | Alley | Price |
| --- | --- | --- |
| 缺失 | Pave | 127500 |
| 2 | 缺失 | 106000 |
| 4 | 缺失 | 178100 |
| 缺失 | 缺失 | 140000 |

NumRooms 是房间数，Alley 是巷子类型，Price 是房价标签。这是教学数据，不代表实际市场。

## 2.2.1 读取并分开特征与标签

```python
import pandas as pd

data = pd.read_csv(data_file)
inputs = data.iloc[:, :2].copy()
targets = data.iloc[:, 2].copy()
```

read_csv 根据表头建立列名，识别数值和默认的缺失标记。CSV 中的 NA 在本例被读取为 NaN。

iloc 按位置取数据：第一项选行，第二项选列；冒号表示全部，切片不包含终点。前两列是输入，第三列是待预测的真实标签。copy 创建独立表格或序列，后面的赋值不修改原始表。

按名字选列也可以：`data['Price']` 取标签，`data.drop(columns=['Price'])` 取其他列作为输入。

观察缺失：

```python
data.isna()       # 每个位置是否缺失
data.isna().sum() # DataFrame 默认按列汇总，得到各列缺失数量
```

这里 pandas DataFrame.sum 默认按列；上一节 torch.Tensor.sum 默认对全部元素求和，不要混淆不同库的默认行为。判断缺失用 isna，不用与 NaN 做相等比较。

## 2.2.2 数值缺失：填补与删除

缺失代表没有记录，不能自动理解为 0。房间数量未知不等于房间数量为零。

教材采用已知数值的均值作为替代：

```python
room_mean = inputs['NumRooms'].mean()
inputs['NumRooms'] = inputs['NumRooms'].fillna(room_mean)
```

mean 默认忽略缺失项，因此只使用 2 和 4，均值为 3。填补后房间数为 `[3,2,4,3]`。

均值填补假设已有记录的平均值可作为未知值的基准替代。它不是恢复了真实房间数，可能偏离某个样本的真实值；中位数、其他固定值或专门模型也是可能的方案，需结合任务选择。

处理多个数值列时，可以按各自均值填补：

```python
numeric_means = inputs.mean(numeric_only=True)
filled = inputs.fillna(numeric_means)
```

numeric_only=True 只汇总数值列；fillna 按列名对齐这些均值，类别缺失暂时不动。不要照搬旧版教材对混合类型整个表直接 mean 的写法，在现有 pandas 2.0.3 中可能尝试对字符串计算均值。

另一条路线是删除：

```python
data.dropna()                       # 删除任一列缺失的整行
data.dropna(subset=['NumRooms'])     # 只根据房间数缺失筛选行
inputs.drop(columns=['Alley'])      # 删除指定列
```

本例四行都含有至少一项缺失，直接 dropna 会得到空表。删除行会减少样本，删除列会减少特征；填补则保留它们但引入替代值假设。

教材的扩展练习要求删除缺失最多的列，可计算：

```python
missing_counts = inputs.isna().sum()
column_name = missing_counts.idxmax()
reduced = inputs.drop(columns=[column_name])
```

idxmax 返回最大值所在的索引标签，在这个按列汇总的结果中就是列名。本练习有唯一的最大缺失列。真实任务选择删列还需结合特征作用，不由缺失数量单独决定。

如果删行，特征和标签必须按相同的行一起筛选，保持一条样本对应一个正确标签。本例主流程只填补和编码，不删行。

## 类别特征：独热编码

Alley 中 Pave 是文字。对无序类别直接指定 0、1、2，并当作数值输入线性模型，会引入人为大小和距离关系，例如认为 2 比 1 大、1 位于 0 与 2 中间。这些关系未必符合类别本身。

独热编码为不同状态建立不同列：

```python
encoded = pd.get_dummies(
    inputs,
    columns=['Alley'],
    dummy_na=True,
    dtype=np.float32,
)
```

dummy_na=True 将缺失也编码为一个状态；它表示记录缺失，不自动等于“没有巷子”。

| Alley 原值 | Alley_Pave | Alley_nan |
| --- | --- | --- |
| Pave | 1 | 0 |
| 缺失 | 0 | 1 |

最终的特征列是 NumRooms、Alley_Pave、Alley_nan，共三个数值特征。指定 float32 让编码列直接成为浮点数。

get_dummies 也可以不指定 columns，默认编码字符串等类别列。综合练习在删除 Garage 后，对剩余 Alley 的 Pave、Gravel、缺失三个状态编码。

## 2.2.3 转换成张量

先将整表统一转换成数值 NumPy 数组，再创建张量：

```python
X = torch.tensor(encoded.to_numpy(dtype=np.float32), dtype=torch.float32)
y = torch.tensor(targets.to_numpy(dtype=np.float32), dtype=torch.float32)
y = y.reshape(-1, 1)
```

本例结果：

```text
X = [[3,1,0],
     [2,0,1],
     [4,0,1],
     [3,0,1]]

y = [[127500],
     [106000],
     [178100],
     [140000]]
```

X.shape=(4,3)：四条样本，每条三个特征。y.shape=(4,1)：四条样本，每条一个房价。按这一约定，后续模型输出也用 (N,1)，避免上一节的意外广播。

书中使用 float64 和一维标签，本项目显式使用 float32 与列形式标签，方便衔接常见 PyTorch 模型；数据值与书中相同。

转换成张量不是清洗本身：NaN 可以进入浮点张量并影响后续计算。因此程序在转换前检查有限数值，转换后打印形状、类型和是否仍有 NaN。

程序将清洗后的特征表导出到 `reports/ch02/2_2_processed_features.csv`，并保持原样本行顺序。

## 后续训练时，预处理参数从哪里来

当前四行只是教学示例，还没有划分训练与测试集。以后进行真实训练评估时，填补均值等统计参数只从训练集计算，验证和测试使用这些已确定的参数。

原因：未见数据不应参与确定训练流程，也应按同一套转换规则处理。独热编码的列名与顺序同样需要保持一致，这样模型的每个输入位置才有固定含义。

## 一次完成综合练习

`datasets/ch02/house_practice.csv` 有六行、更多特征，包含不同的数值缺失和巷子类别。

在一个练习文件中集中填写：

1. 输入特征与房价标签。
2. 输入里缺失最多的列名及删除后的表。
3. 各数值列按自身均值填补。
4. 类别独热编码，包括缺失状态。
5. 转成 float32 的 X、y，标签形状为 (N,1)。

不要改原始房价，也不要硬编码均值；通过表格操作计算。

一次反馈：删除的列、数值列填补均值、编码后的列名、X/y 的形状和是否含 NaN，以及三个原理问题：

- A：为什么未知房间数不能默认当作 0，均值填补做了什么假设？
- B：无序类别直接用 0、1、2 作数值，会引入什么关系？独热编码怎样表示？
- C：以后训练集和测试集各自计算填补均值是否合适？参数应从哪里来？

后续下一整块为教材 2.3 线性代数。之前 2.1 的未提交综合练习保留，不因用户要求前进而虚构已掌握记录。
