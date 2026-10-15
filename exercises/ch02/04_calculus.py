"""2.4 综合练习：手工算梯度，为同一批数据做一次参数更新。

将八个 None 换成表达式；不用 backward，不硬编码计算结果。
先用 lr=0.01，完成后只改成 0.2 再运行。每次均从 w=0、b=0 开始。
"""

import torch

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
y = torch.tensor([3., 5., 7., 9.], dtype=torch.float64)
w = torch.tensor(0., dtype=torch.float64)
b = torch.tensor(0., dtype=torch.float64)
lr = 0.01

# 只填写这一组表达式，后面的显示区不用修改。
pred = None  # 1. 四条样本的预测 w*x+b。
error = None  # 2. 预测减去真实值。
loss = None  # 3. 所有样本的误差平方平均。
grad_w = None  # 4. 对 w 的偏导：每条样本经链式法则再取平均。
grad_b = None  # 5. 对 b 的偏导；x、w、y 暂时视为不变。
new_w = None  # 6. 使用旧 w 和本次 grad_w 更新。
new_b = None  # 7. 使用旧 b 和本次 grad_b 更新。
new_loss = None  # 8. 用新 w、新 b 重新预测并计算损失。

results = {"旧预测": pred, "误差": error, "旧损失": loss,
           "w 的偏导": grad_w, "b 的偏导": grad_b,
           "新 w": new_w, "新 b": new_b, "新损失": new_loss}
for name, value in results.items():
    print("\n" + name + ":")
    print("待填写" if value is None else value)
    if isinstance(value, torch.Tensor):
        print("shape =", tuple(value.shape))

print("\n本次学习率:", lr)
print("集中反馈两次运行结果和三个解释：")
print("A. 求 grad_w 时，为什么要把误差乘以 x？求 grad_b 时为什么不乘 x？")
print("B. lr 从 0.01 改为 0.2，起点相同，初始梯度会不会改变？什么改变了？")
print("C. loss 是一个数，为什么需要两个偏导？梯度为 0 是否一定是最低点？")
