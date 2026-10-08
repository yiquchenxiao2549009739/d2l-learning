# 教材 1.2.4 的小实验：一次反向传播与一次梯度下降更新。
# 暂时只学习 w，把输入、真实标签和 b 固定下来。

import torch

x = 3.0
y_true = 10.0
b = 1.0
learning_rate = 0.01  # 学习率：控制每次更新的步长

w = torch.tensor(2.0, requires_grad=True)  # 记录后续计算，以便求梯度

y_pred = w * x + b
loss = (y_pred - y_true) ** 2

print("更新前 w：", w.item())
print("更新前预测：", y_pred.item())
print("更新前损失：", loss.item())

loss.backward()  # 反向传播：计算损失对 w 的梯度，存入 w.grad
print("梯度：", w.grad.item())

with torch.no_grad():  # 参数更新这一步不记录到求导的计算图中
    w -= learning_rate * w.grad  # 一次梯度下降更新

new_y_pred = w * x + b
new_loss = (new_y_pred - y_true) ** 2

print("更新后 w：", w.item())
print("更新后预测：", new_y_pred.item())
print("更新后损失：", new_loss.item())

# 先原样运行，再把 learning_rate 改成 0.2，保存并从头运行。
# 观察：方向由梯度确定，但步长会怎样影响最终损失？
