"""教材完整 2.3：对象与形状 -> 求和与点积 -> 矩阵乘法 -> 范数。

使用已有 openmmlab 环境，一次运行整节；不下载、不安装、不训练模型。
对应原教材 PDF 第 68-81 页，正文第 50-63 页（含章末练习）。
"""

import torch

torch.set_printoptions(precision=4, sci_mode=False)


def show(name, value):
    """辅助显示；本节重点是下面的数学计算。"""
    print("\n" + name + ":")
    print(value)
    if isinstance(value, torch.Tensor):
        print("shape =", tuple(value.shape), "; ndim =", value.ndim)


print("===== 1. 标量、向量、矩阵、高阶张量 =====")
s = torch.tensor(3.0)
x = torch.tensor([1.0, 2.0, 3.0])
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
T = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
for name, value in [("标量 s", s), ("向量 x", x), ("矩阵 A", A), ("张量 T", T)]:
    show(name, value)
print("三维向量 x 的轴数:", x.ndim, "; 元素数:", x.numel())
print("len(T) 是第 0 轴长度:", len(T), "; T 的元素数:", T.numel())
show("显式行向量", x.reshape(1, -1))
show("显式列向量", x.reshape(-1, 1))

print("\n===== 2. 转置与对称：交换行列，不是重新分组 =====")
show("A.T", A.T)
show("A.reshape(3, 2)，注意它与 A.T 的元素排列不同", A.reshape(3, 2))
S = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
print("S 是对称矩阵:", torch.equal(S, S.T))
print("转置两次恢复 A:", torch.equal(A.T.T, A))
show("高阶张量只交换后两个轴", T.transpose(1, 2).shape)

print("\n===== 3. 按元素运算与沿轴汇总 =====")
B = A.clone()
show("A+B", A + B)
show("A*B：Hadamard 积，仍为 (2,3)", A * B)
show("2*A+1：缩放与平移每个元素", 2 * A + 1)
show("所有元素和", A.sum())
show("dim=0：跨行相加，每列一个结果", A.sum(dim=0))
show("dim=1：跨列相加，每行一个结果", A.sum(dim=1))
show("列均值", A.mean(dim=0))
print("均值 = 总和 / 元素数:", torch.allclose(A.mean(), A.sum() / A.numel()))
print("同时汇总两个轴:", A.sum(dim=(0, 1)).item())
for axis in range(T.ndim):
    print("T.sum(dim=%d).shape =" % axis, tuple(T.sum(dim=axis).shape))

print("\n===== 4. keepdim、广播陷阱和累积和 =====")
row_sums = A.sum(dim=1, keepdim=True)
show("保留轴的每行和", row_sums)
normalized = A / row_sums  # 当前行和均非零，才可直接相除。
show("每行除以自己的行和", normalized)
show("检查归一化后的每行和", normalized.sum(dim=1))
try:
    A / A.sum(dim=1)  # (2,3) 与 (2,) 的末轴不匹配。
except RuntimeError:
    print("预期的形状错误：(2,3) 不能直接除以 (2,)。")
Q = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
show("方阵的错误写法不报错，却沿列除以 [4,8]", Q / Q.sum(dim=1))
show("正确的逐行归一化", Q / Q.sum(dim=1, keepdim=True))
show("沿 dim=0 累积和：保留每一个前缀和", A.cumsum(dim=0))
show("沿 dim=1 累积和", A.cumsum(dim=1))

print("\n===== 5. 点积：把对应位置的乘积加起来 =====")
w = torch.tensor([2.0, -1.0, 0.5])
show("x*w：保留三个贡献", x * w)
show("dot(x,w)：汇总成一个分数", torch.dot(x, w))
print("点积 = 按元素乘后求和:", torch.allclose(torch.dot(x, w), (x * w).sum()))
weights = torch.tensor([0.2, 0.3, 0.5])
show("非负且和为 1 的权重给出加权平均", torch.dot(x, weights))
u = torch.tensor([3.0, 4.0])
v = torch.tensor([4.0, 3.0])
cosine = torch.dot(u, v) / (torch.linalg.vector_norm(u) * torch.linalg.vector_norm(v))
show("两个非零向量的余弦相似度", cosine)

print("\n===== 6. 矩阵-向量积：每一行都与同一向量做点积 =====")
show("A @ w", A @ w)
manual_mv = torch.stack([torch.dot(row, w) for row in A])
print("逐行点积与矩阵-向量积一致:", torch.allclose(A @ w, manual_mv))
print("torch.mv 与 @ 一致:", torch.allclose(torch.mv(A, w), A @ w))
rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
show("逆时针旋转 90 度：[1,0] -> [0,1]", rotation @ torch.tensor([1.0, 0.0]))

print("\n===== 7. 矩阵-矩阵积：左行与右列做点积 =====")
W = torch.tensor([[1.0, 2.0], [0.0, -1.0], [1.0, 0.5]])
C = A @ W
show("W：每一列是一组输出权重", W)
show("A(2,3) @ W(3,2) -> C(2,2)", C)
print("C[0,1] 来自左第 0 行与右第 1 列:", torch.dot(A[0], W[:, 1]).item())
manual_mm = torch.stack([A @ W[:, j] for j in range(W.shape[1])], dim=1)
print("逐列矩阵-向量积与矩阵乘法一致:", torch.allclose(C, manual_mm))
print("torch.mm 与 @ 一致:", torch.allclose(torch.mm(A, W), C))
P = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
Q = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
show("P*Q：按元素乘", P * Q)
show("P@Q：矩阵乘法", P @ Q)
show("Q@P：交换次序一般改变结果", Q @ P)
print("(A@W).T = W.T@A.T:", torch.allclose((A @ W).T, W.T @ A.T))

print("\n===== 8. 范数：用一个数度量整体大小 =====")
u = torch.tensor([3.0, -4.0])
show("L1：绝对值之和", u.abs().sum())
show("L2：平方和再开方", torch.linalg.vector_norm(u))
show("L2 的平方：不是另一个范数", (u * u).sum())
F = torch.ones(4, 9)
show("4x9 个 1 的 Frobenius 范数", torch.linalg.matrix_norm(F, ord="fro"))
show("三轴 T 不指定 dim 的 linalg.norm：整体展开求 L2", torch.linalg.norm(T))
show("只沿 T 最后一个轴求向量 L2", torch.linalg.vector_norm(T, dim=2))
print("三轴整体范数 = 展开后的 L2:",
      torch.allclose(torch.linalg.norm(T), torch.linalg.vector_norm(T.reshape(-1))))
print("范数缩放规律 ||-2u|| = 2||u||:",
      torch.allclose(torch.linalg.vector_norm(-2 * u), 2 * torch.linalg.vector_norm(u)))
print("注意：矩阵 ord=2 表示谱范数；本节矩阵整体大小用 ord='fro'。")

print("\n===== 9. 串成一次小批量预测：目前只前向计算，不训练 =====")
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
w_col = torch.tensor([[2.0], [-1.0]])
b = torch.tensor(0.5)
y = torch.tensor([[1.0], [2.0], [6.0]])
y_hat = X @ w_col + b
error = y_hat - y
mse = error.square().mean()
show("X：三条样本、每条两个特征", X)
show("y_hat = X @ w_col + b", y_hat)
show("误差向量", error)
show("MSE", mse)
rmse = mse.sqrt()
show("RMSE = L2(误差) / sqrt(样本数)", rmse)
print("MSE = L2(误差)^2 / 误差元素数:",
      torch.allclose(mse, torch.linalg.vector_norm(error).square() / error.numel()))
print("预测与标签形状一致:", tuple(y_hat.shape), tuple(y.shape))
bad_error = y_hat - y.reshape(-1)
show("故意去掉标签轴后，广播生成错误的 3x3 误差", bad_error)
show("错误广播得到的 MSE", bad_error.square().mean())
print("\n整节结束：去综合练习文件集中填写，再一次反馈输出和三个解释。")
