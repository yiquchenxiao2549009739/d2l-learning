"""3.1 综合练习：把预测、损失、梯度和更新接成完整训练。

六处 TODO 留给自己填写；使用半均方误差，不硬编码训练结果。
默认参数 lr=0.05、steps=200。可改变学习率观察，比较时从零重新开始。
"""

import argparse
import torch


def main(lr=0.05, steps=200):
    x = torch.tensor([[-2., 1.], [-1., -1.], [0., 2.],
                      [1., 0.], [2., -2.], [0.5, 1.5]], dtype=torch.float64)
    y = torch.tensor([[-4.3], [1.2], [-3.3], [2.2], [7.7], [-1.55]],
                     dtype=torch.float64)
    w = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    for step in range(steps):
        prediction = None  # TODO 1：整个小数据集的线性预测，形状 (6,1)。
        loss = None  # TODO 2：所有样本的平均平方误差再除以 2，标量。
        if prediction is None or loss is None:
            print("先集中填写六处 TODO，再运行完整训练。")
            return
        pass  # TODO 3：让自动微分计算这次损失的梯度。
        if w.grad is None or b.grad is None:
            print("尚未得到参数梯度，请检查 TODO 3。")
            return
        with torch.no_grad():
            pass  # TODO 4：用 lr 和两个梯度，原地更新 w、b。
        w.grad = None
        b.grad = None
        if step in (0, 9, 49, steps - 1):
            print("第", step + 1, "次更新前的损失:", loss.item())

    with torch.no_grad():
        final_prediction = None  # TODO 5：使用更新后的参数重新预测。
        final_loss = None  # TODO 6：重新计算半均方误差。
    print("\n最终 w、b:", w.detach().ravel(), b.detach())
    print("最终预测:\n", final_prediction)
    print("最终损失:", final_loss)
    print("\n集中解释：")
    print("A. x、w、b 各扮演什么角色？哪些是参数，哪些设置是超参数？")
    print("B. backward、参数更新、清理梯度各做什么？为什么 mean 后不用再除样本数？")
    print("C. 换成带噪声数据，训练损失能否要求为零？怎样检查对新样本的预测能力？")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.lr <= 0 or args.steps <= 0:
        parser.error("lr 和 steps 需要是正数")
    main(args.lr, args.steps)
