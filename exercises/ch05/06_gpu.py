"""5.6 综合练习。填写 TODO，将一次训练、模型恢复和日志搬到选定设备。"""
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/ch05/5_6_exercise"


def main():
    torch.manual_seed(5681)
    device = None  # TODO 1：CUDA 可用时选择 cuda:0，否则用 CPU。
    if device is None:
        print("先完成 TODO 1，再按顺序填写设备迁移、日志和读档。")
        return

    X_cpu = torch.randn(32, 2)
    y_cpu = X_cpu[:, :1] - .5 * X_cpu[:, 1:] + .2
    model = nn.Linear(2, 1)
    X = None  # TODO 2：把 X_cpu 移到 device，保留返回值。
    y = None  # TODO 3：把 y_cpu 移到 device，保留返回值。
    pass  # TODO 4：将 model 移到 device。
    if X is None or y is None or any(t.device != device for t in (X, y, model.weight)):
        print("核对 X、y、model.weight 的 device，三者都应等于 device。")
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    history = []
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(X) - y).square().mean()
        loss.backward()
        optimizer.step()
        value = None  # TODO 5：拿到脱离计算图但仍在原设备上的损失张量。
        if value is not None:
            history.append(value)

    host_history = None  # TODO 6：把 history 堆叠成一个张量，再整批搬到 CPU。
    if host_history is None:
        print("请先完成 TODO 5 和 6。")
        return
    print("模型、输入、标签和最后一步梯度", model.weight.device, X.device, y.device, model.weight.grad.device)
    print("日志条数和设备", len(host_history), host_history.device)
    print("第一次、最后一次更新前的 MSE", host_history[0].item(), host_history[-1].item())

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"linear_{device.type}.pt"
    torch.save(model.state_dict(), path)
    state = None  # TODO 7：把自己保存的权重加载到 CPU，显式设置 weights_only=True。
    if state is None:
        return
    restored = nn.Linear(2, 1).eval()
    pass  # TODO 8：把 state 装入 CPU 模型。
    with torch.no_grad():
        error = (restored(X_cpu) - model(X).cpu()).abs().max().item()
    print("文件加载后的权重设备", state["weight"].device)
    print("CPU 恢复预测的最大绝对差", error)


if __name__ == "__main__":
    main()
