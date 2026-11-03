"""5.5 综合练习。依次填写八处 TODO，核对保存、恢复和部分复用。"""
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/ch05/5_5_exercise"


def build_model(out_features=1):
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, out_features)).double()


def train_steps(model, optimizer, X, y, steps):
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(X) - y).square().mean()
        loss.backward()
        optimizer.step()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(5581)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_path = OUTPUT / "tensors.pt"
    X = torch.rand(24, 2, dtype=torch.float64)
    y = X[:, :1] - .5 * X[:, 1:]
    pass  # TODO 1：将含 X、y 的字典保存到 data_path。
    loaded = None  # TODO 2：从 data_path 加载该字典到 CPU。
    if loaded is None:
        print("请先完成张量字典的保存与读取，再继续检查点练习。")
        return
    print("训练张量往返相同", torch.equal(loaded["X"], X), torch.equal(loaded["y"], y))

    model = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=.08, momentum=.9)
    train_steps(model, optimizer, X, y, 30)
    checkpoint_path = OUTPUT / "checkpoint_step30.pt"
    pass  # TODO 3：保存 model_state、optimizer_state 和 step=30。
    checkpoint = None  # TODO 4：加载 checkpoint_path 到 CPU。
    if checkpoint is None:
        return

    restored = build_model()
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=.08, momentum=.9)
    pass  # TODO 5：给 restored 加载 checkpoint 中的模型状态。
    pass  # TODO 6：给 restored_optimizer 加载其中的优化器状态。
    train_steps(model, optimizer, X, y, 20)
    train_steps(restored, restored_optimizer, X, y, 50 - checkpoint["step"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        error = None  # TODO 7：计算两个模型在 X 上的最大预测绝对差。
    print("第50步的最大预测差", error)

    wider = build_model(out_features=2)
    pass  # TODO 8：只把 restored 的第一层状态加载给 wider 的第一层。
    print("第一层权重相同", torch.equal(wider[0].weight, restored[0].weight))
    print("新模型输出形状", tuple(wider(X).shape))


if __name__ == "__main__":
    main()
