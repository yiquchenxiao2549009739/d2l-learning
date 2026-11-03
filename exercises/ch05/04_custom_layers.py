"""5.4 综合练习。按顺序填写八处 TODO。"""
import torch
from torch import nn


class CenteredLayer(nn.Module):
    def forward(self, X):
        return None  # TODO 1：减去整个输入张量的均值。


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = None  # TODO 2：注册 (in_features, out_features) 的随机权重。
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        return None  # TODO 3：计算仿射结果，再应用 ReLU。


class QuadraticLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, 3, 2))

    def forward(self, X):
        return None  # TODO 4：沿 i、j 求和，返回两个二次特征。


class FourierLayer(nn.Module):
    def forward(self, X):
        return None  # TODO 5：沿最后一维返回实信号的单边频谱。


def main():
    torch.set_num_threads(1)
    torch.manual_seed(5481)
    dtype = torch.float64
    centered = CenteredLayer()(torch.arange(1, 6, dtype=dtype))
    if centered is None:
        print("请依次完成八处 TODO。先观察中心化，再核对参数、频谱并训练二次层。")
        return
    print("中心化结果及均值:", centered.tolist(), centered.mean().item())

    linear = MyLinear(3, 2).double()
    if linear.weight is None:
        return
    Y = linear(torch.ones(4, 3, dtype=dtype))
    if Y is None:
        return
    print("线性层输出形状和参数量:", tuple(Y.shape), sum(p.numel() for p in linear.parameters()))

    layer = QuadraticLayer().double()
    X = 2 * torch.rand(72, 3, dtype=dtype) - 1
    a, b, c = X.unbind(dim=1)
    y = torch.stack((a.square() + .5 * a * b, b * c - .25 * c.square()), dim=1)
    if layer(X[:2]) is None:
        return
    signal = torch.arange(7, dtype=dtype)
    spectrum = FourierLayer()(signal)
    if spectrum is None:
        return
    print("7点信号频谱形状和重建误差:", tuple(spectrum.shape),
          (torch.fft.irfft(spectrum, n=7) - signal).abs().max().item())

    optimizer = torch.optim.SGD(layer.parameters(), lr=.2)
    with torch.no_grad():
        initial = (layer(X[:48]) - y[:48]).square().mean().item()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        prediction = layer(X[:48])
        loss = None  # TODO 6：计算预测与前48条训练标签的平均MSE。
        if loss is None:
            return
        pass  # TODO 7：反向传播。
        if layer.weight.grad is None:
            print("请先完成反向传播。")
            return
        pass  # TODO 8：更新参数。
    with torch.no_grad():
        train = (layer(X[:48]) - y[:48]).square().mean().item()
        valid = (layer(X[48:]) - y[48:]).square().mean().item()
    print("二次层权重/输出形状:", tuple(layer.weight.shape), tuple(layer(X[:2]).shape))
    print("初始训练/最终训练/验证MSE:", initial, train, valid)


if __name__ == "__main__":
    main()
