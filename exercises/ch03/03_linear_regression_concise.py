# -*- coding: utf-8 -*-
"""3.3 综合实践：用框架部件训练 37 条人工样本。

八处 TODO 集中填写。批量 8、默认 20 轮；使用 MSELoss(mean)。
不要再手动除批量大小。已有环境即可运行，无下载。
"""

import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def main(lr=0.03, epochs=20, batch_size=8):
    generator = torch.Generator().manual_seed(3304)
    features = torch.randn(37, 2, generator=generator, dtype=torch.float64)
    true_weight = torch.tensor([[1.5], [-2.]], dtype=torch.float64)
    labels = features @ true_weight + 0.7
    labels += 0.01 * torch.randn(37, 1, generator=generator, dtype=torch.float64)
    dataset = None  # TODO 1：用 TensorDataset 绑定特征和标签。
    loader = None  # TODO 2：DataLoader，打乱、保留尾批，num_workers=0。
    net = None  # TODO 3：Sequential 包含 Linear，2 个输入、1 个输出，float64。
    if dataset is None or loader is None or net is None:
        print("先集中填写八处 TODO，再运行训练。")
        return
    nn.init.normal_(net[0].weight, mean=0., std=.01,
                    generator=torch.Generator().manual_seed(3305))
    nn.init.zeros_(net[0].bias)
    loss_fn = None  # TODO 4：平均均方误差损失。
    trainer = None  # TODO 5：用模型参数构造 SGD，学习率为 lr。
    if loss_fn is None or trainer is None:
        print("请完成损失与优化器构造。")
        return
    updates, first_sizes = 0, []
    for epoch in range(epochs):
        for x, y in loader:
            trainer.zero_grad(set_to_none=True)
            loss = None  # TODO 6：计算当前批模型预测及损失，标签保持列形状。
            if loss is None:
                print("请完成当前批损失。")
                return
            pass  # TODO 7：反向传播计算梯度。
            if net[0].weight.grad is None:
                print("请完成反向传播。")
                return
            pass  # TODO 8：调用优化器更新参数。
            updates += 1
            if epoch == 0: first_sizes.append(len(x))
    with torch.no_grad():
        final_mse = nn.MSELoss()(net(features), labels).item()
    print("首轮批量大小:", first_sizes, "| 更新总次数:", updates)
    print("权重形状:", tuple(net[0].weight.shape))
    print("学到权重、偏置:", net[0].weight.detach(), net[0].bias.detach())
    print("最终 MSE:", final_mse)
    print("最后一批的权重梯度:", net[0].weight.grad)
    print("集中解释：Linear 为什么用 (1,2) 权重？TensorDataset 和 DataLoader 各负责什么？")
    print("MSELoss 与上一节半平方损失差在哪里？zero_grad、backward、step 各做什么？")
    print("尾批有几条、共更新几次？教材 Huber 公式应对应哪个 API？")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.lr <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("lr、epochs、batch-size 都应为正数")
    main(args.lr, args.epochs, args.batch_size)
