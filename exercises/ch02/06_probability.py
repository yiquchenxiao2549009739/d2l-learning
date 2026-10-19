"""2.6 综合实践：虚构零件联合抽样 -> 经验后验 -> 理论概率与方差。

九处 TODO 集中填写，不硬编码结果；不需要下载数据。
H=1 表示缺陷，D=1 表示报警；联合表行 H=0/1，列 D=0/1。
"""

import torch
from torch.distributions import Multinomial

torch.set_printoptions(precision=6, sci_mode=False)
torch.manual_seed(20261019)
prior = 0.02  # P(H=1)
sensitivity = 0.9  # P(D=1|H=1)
false_positive = 0.05  # P(D=1|H=0)
m, n = 200, 100

joint = None  # TODO 1：构造 float64 的 2x2 联合概率表，四格合计为 1。
counts = None  # TODO 2：展平联合表，生成 m 组、每组 n 个零件的四类计数。
cum_counts = None  # TODO 3：沿组轴计算累积计数，形状 (m,4)。
estimates = None  # TODO 4：每行除以累积总数，保留广播所需的轴。
estimated_joint = None  # TODO 5：最后一行经验概率恢复为 (2,2)。
estimated_posterior = None  # TODO 6：由经验表求 P(H=1|D=1)。
exact_posterior = None  # TODO 7：由理论表求同一后验概率。
alarm_mean = None  # TODO 8：D 取 0/1，理论 E[D] 是多少？
alarm_variance = None  # TODO 9：从 E[(D-E[D])^2] 或二阶矩公式求理论方差。

results = {"理论联合表": joint, "前 3 组计数": None if counts is None else counts[:3],
           "累计计数末行": None if cum_counts is None else cum_counts[-1],
           "最后经验联合表": estimated_joint, "经验后验": estimated_posterior,
           "理论后验": exact_posterior, "报警变量期望": alarm_mean, "报警变量方差": alarm_variance}
for name, value in results.items():
    print("\n" + name + ":")
    print("待填写" if value is None else value)
    if isinstance(value, torch.Tensor):
        print("形状:", tuple(value.shape))

print("\n集中反馈运行结果与三个解释：")
print("A. P(D=1|H=1) 与 P(H=1|D=1) 为什么不同？各自的分母是什么？")
print("B. 改 m、n 时，总样本数怎样变化？更多样本是否保证每一步误差都变小？")
print("C. 期望、方差各描述什么？直接重复同一确定性报警结果能否当独立新证据？")
