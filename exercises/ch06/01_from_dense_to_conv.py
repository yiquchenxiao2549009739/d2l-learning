"""6.1 综合练习，让同一个局部规则寻找平移前后的十字形。

先完成讲义的 L 形跟做，再运行本文件，按三个阶段填写八处 TODO。
阶段一用切片和乘加手算一格，阶段二组织 NCHW 并扫描两张图，
阶段三对齐重叠区域、比较峰值和参数量。每阶段先预测再运行。
集中反馈响应形状、峰值坐标、重叠差和局部性/共享的解释。
"""
import torch
from torch.nn import functional as F


def place_motif(motif, row, col):
    image = torch.zeros(8, 8)
    image[row:row + 3, col:col + 3] = motif
    return image


def main():
    print("6.1 综合练习。先预测十字形平移后，响应最高点会怎样移动。")
    print("先填写 TODO 1–2，再填写 3–5，最后完成 6–8。")
    motif = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    kernel = 2 * motif - 1
    row, col = 1, 2
    dr, dc = 2, 1
    first = place_motif(motif, row, col)
    moved = place_motif(motif, row + dr, col + dc)
    patch = None  # TODO 1：从 first 切出左上角为 (row, col) 的 3×3 窗口。
    if patch is None:
        print("先完成 TODO 1，从图像中切出一个窗口。")
        return
    local_score = None  # TODO 2：patch 与 kernel 逐元素乘积后求和。
    images = torch.stack([first, moved])  # 两张二维图先沿批量轴叠起来。
    inputs = None  # TODO 3：给 images 添加长度为 1 的通道轴，得到 NCHW。
    weights = None  # TODO 4：给 kernel 添加输出和输入通道轴，形状为 (1, 1, 3, 3)。
    if local_score is None or inputs is None or weights is None:
        print("先完成 TODO 2 到 4，并检查形状。")
        return
    maps = None  # TODO 5：调用 F.conv2d 扫描两张图，再移除单个输出通道轴。
    if maps is None:
        print("先完成 TODO 5，让同一个核扫描两张图。")
        return
    peak_positions = [divmod(int(response.argmax()), response.shape[1]) for response in maps]
    print("响应图形状", tuple(maps.shape))
    print("手算式得分、同位置扫描得分", local_score.item(), maps[0, row, col].item())
    print("两个峰值坐标（行, 列）", peak_positions)

    before = None  # TODO 6：在 maps[0] 里取平移后仍可比较的重叠区域。
    after = None  # TODO 7：在 maps[1] 里取与 before 对应的平移后区域。
    if before is None or after is None:
        print("先完成 TODO 6 和 7。对应坐标从 (i,j) 变为 (i+dr,j+dc)。")
        return
    print("重叠区域的最大绝对差", (after - before).abs().max().item())
    params = None  # TODO 8：计算一个 3×3 核及一个共享偏置的参数量。
    print("若加一个共享偏置，卷积参数量", params)


if __name__ == "__main__":
    main()
