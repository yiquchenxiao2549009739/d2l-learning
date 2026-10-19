"""3.2 综合实践：23 条样本、批量 10，自己接好从零训练。

六处 TODO 集中完成。这里使用 sum().backward()，SGD 内除实际批量大小。
不使用 nn.Linear、DataLoader 或 optim.SGD。已有环境即可运行。
"""

import argparse
import random
import torch


def data_iter(batch_size, features, labels, rng):
    indices = list(range(len(features)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = None  # TODO 1：将本批索引切片转成 long 张量，保留尾批。
        if batch_indices is None:
            print("请先集中填写六处 TODO。")
            return
        yield features[batch_indices], labels[batch_indices]


def linreg(x, w, b):
    return x @ w + b


def squared_loss(y_hat, y):
    result = None  # TODO 2：每条样本的半平方误差，先对齐标签形状。
    return result


def sgd(params, lr, num_samples):
    with torch.no_grad():
        for param in params:
            assert param.grad is not None
            pass  # TODO 3：按 sum 求导约定，除实际样本数并原地更新。
            param.grad.zero_()


def main(lr=0.05, epochs=20, batch_size=10):
    g = torch.Generator().manual_seed(3204)
    features = torch.randn(23, 2, generator=g, dtype=torch.float64)
    true_w = torch.tensor([[1.5], [-2.]], dtype=torch.float64)
    labels = features @ true_w + 0.7
    labels += 0.01 * torch.randn(23, 1, generator=g, dtype=torch.float64)
    w = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    rng = random.Random(3205)
    updates, sizes = 0, []
    for epoch in range(epochs):
        received = False
        for x, y in data_iter(batch_size, features, labels, rng):
            received = True
            prediction = None  # TODO 4：调用模型计算当前批预测。
            if prediction is None:
                print("请完成预测、求导和更新的 TODO。")
                return
            per_sample = squared_loss(prediction, y)
            if per_sample is None:
                print("请完成损失函数 TODO。")
                return
            pass  # TODO 5：损失先求和，再执行反向传播。
            if w.grad is None or b.grad is None:
                print("请完成反向传播 TODO。")
                return
            pass  # TODO 6：调用 sgd，传本批实际样本数。
            updates += 1
            if epoch == 0: sizes.append(len(x))
        if not received:
            return
    with torch.no_grad():
        final = squared_loss(linreg(features, w, b), labels)
    print("第一轮批量大小:", sizes, "| 更新总次数:", updates)
    print("学到 w,b:", w.detach().ravel(), b.item())
    print("真实-估计:", (true_w-w.detach()).ravel(), 0.7-b.item())
    print("最终半均方误差:", None if final is None else final.mean().item())
    print("集中解释：yield 怎样把批次交给外层循环？为什么 X、y 共用索引？")
    print("sum+SGD 除批量，与 mean+不再除批量，为什么等价？尾批分母是多少？")
    print("本次共多少次更新？零初始化在这个线性模型里是否有效？")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    if args.lr <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("lr、epochs、batch-size 都需要为正数")
    main(args.lr, args.epochs, args.batch_size)
