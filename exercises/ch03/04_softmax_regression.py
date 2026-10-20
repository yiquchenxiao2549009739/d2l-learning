# -*- coding: utf-8 -*-
"""3.4 综合实践：一批六条人工样本，接通得分、概率、交叉熵和梯度。

一次填写八处 TODO；更新与对照代码已提供，不下载数据。
"""
import torch
from torch.nn import functional as F


def main():
    dtype = torch.float64
    features = torch.tensor([[-2., -1.], [-1.5, -.8], [2., -1.],
                             [1.5, -.5], [0., 2.], [.3, 1.5]], dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    weight = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    bias = torch.zeros(3, dtype=dtype, requires_grad=True)
    one_hot = None  # TODO 1：独热标签，三个类别，转成 dtype。
    logits = None  # TODO 2：批量仿射计算，形状为 (6,3)。
    if one_hot is None or logits is None:
        print("请集中填写八处 TODO，再运行整个实践。")
        return
    logits.retain_grad()  # 留下中间得分的梯度，便于核对。
    probabilities = None  # TODO 3：沿类别维度计算 softmax。
    log_probabilities = F.log_softmax(logits, dim=1)
    indices = torch.arange(len(labels))
    per_sample_loss = None  # TODO 4：从 log_probabilities 取真实类别并取负号。
    if probabilities is None or per_sample_loss is None:
        print("请完成概率与逐样本交叉熵。")
        return
    loss = None  # TODO 5：把逐样本损失取平均。
    if loss is None:
        print("请完成标量损失。")
        return
    pass  # TODO 6：反向传播。
    if weight.grad is None or logits.grad is None:
        print("请完成反向传播。")
        return
    predicted = None  # TODO 7：每条样本取最大得分的类别编号。
    accuracy = None  # TODO 8：预测编号与标签比较，转浮点后取平均。
    if predicted is None or accuracy is None:
        print("请完成分类预测和准确率。")
        return
    expected_gradient = (probabilities.detach() - one_hot) / len(labels)
    print("概率行和:", probabilities.sum(dim=1).detach())
    print("逐样本交叉熵:", per_sample_loss.detach())
    print("更新前损失、准确率:", loss.item(), float(accuracy))
    print("logits 梯度符合 (p-one_hot)/批量:",
          torch.allclose(logits.grad, expected_gradient))
    with torch.no_grad():
        weight -= .1 * weight.grad
        bias -= .1 * bias.grad
        after_logits = features @ weight + bias
        after_loss = F.cross_entropy(after_logits, labels)
        after_accuracy = (after_logits.argmax(dim=1) == labels).double().mean()
    print("一次更新后损失、准确率:", after_loss.item(), after_accuracy.item())
    print("集中解释：类别编号为何不用 MSE？softmax 的 dim=1 在归一化什么？")
    print("CrossEntropyLoss 应接收得分还是概率？最大项没变时损失还能下降吗？")
    print("单样本梯度 p-y 与平均批损失梯度差在哪里？概率大就保证校准吗？")


if __name__ == "__main__":
    main()
