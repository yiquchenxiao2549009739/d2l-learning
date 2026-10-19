"""2.5 综合实践：用自动微分替代上一节的手写求导。

集中完成八处 TODO；不用手写 grad_w / grad_b 的数学公式。
复制梯度供显示；参数在 no_grad 中原地更新。
"""

import torch

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
y = torch.tensor([3., 5., 7., 9.], dtype=torch.float64)
w = torch.tensor(0., dtype=torch.float64, requires_grad=True)
b = torch.tensor(0., dtype=torch.float64, requires_grad=True)
lr = 0.01

pred = None  # TODO 1：四条样本的预测。
loss = None  # TODO 2：误差平方平均，必须保留张量计算。
if isinstance(loss, torch.Tensor):
    pass  # TODO 3：把这一行换成反向传播调用。

grad_w = None  # TODO 4：从 w.grad 复制本次梯度供显示。
grad_b = None  # TODO 5：从 b.grad 复制本次梯度供显示。
if isinstance(grad_w, torch.Tensor) and isinstance(grad_b, torch.Tensor):
    with torch.no_grad():
        pass  # TODO 6：使用本次梯度，原地更新 w。
        pass  # TODO 7：使用本次梯度，原地更新 b。

new_loss = None  # TODO 8：使用新参数重新计算 MSE。

results = {"旧预测": pred, "旧损失": loss, "自动梯度 w": grad_w,
           "自动梯度 b": grad_b, "新损失": new_loss}
for name, value in results.items():
    print("\n" + name + ":")
    print("待填写" if value is None else value)
print("\n当前 w / b:", w.detach(), b.detach())
print("w 仍为叶张量 / 仍跟踪梯度:", w.is_leaf, w.requires_grad)
w.grad = None
b.grad = None
print("为下一轮清理后 w.grad / b.grad:", w.grad, b.grad)

print("\n集中反馈运行结果与三个解释：")
print("A. backward 后参数会自动改变吗？为什么更新放在 no_grad 中？")
print("B. 新前向不清梯度会怎样？把 mean 改为 sum，本例梯度会变为几倍？")
print("C. y=x*x，z=y.detach()*x 与 z=y*x 的梯度为何不同？")
