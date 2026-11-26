"""8.3 综合练习：用玩具词元完成双词计数、评分和两种取批。"""
from collections import Counter
import math
import random


def main():
    corpus = [0, 1, 2, 0, 1, 3, 0, 1, 2, 3]
    vocab_size, mass = 4, 2.0
    pairs = None  # TODO 1：用相差一位的切片组成相邻词元对。
    totals = None  # TODO 2：用 Counter 统计每个有后继的上下文。
    if pairs is None or totals is None:
        print('先完成 TODO 1–2，统计转移次数和分母。')
        return
    counts = Counter(pairs)
    assert len(pairs) == 9 and totals[1] == 3 and counts[1, 2] == 2

    probs = None  # TODO 3：为候选 0..3 计算给定 1 的均匀先验平滑概率。
    loss = None  # TODO 4：目标依次为 2、3，两次负对数概率取平均。
    if probs is None or loss is None:
        print('继续 TODO 3–4，求分布和平均负对数似然。')
        return
    assert all(math.isclose(p, q) for p, q in zip(probs, [.1, .1, .5, .3]))
    assert len(probs) == 4 and math.isclose(sum(probs), 1)
    assert math.isclose(math.exp(loss), 1 / math.sqrt(.15))
    print('概率', probs, '困惑度', round(math.exp(loss), 6))

    seq = list(range(35))
    batch_size, steps = 2, 5
    offset = 3
    r_starts = None  # TODO 5：先预留标签，再求随机采样的所有起点。
    if r_starts is None:
        print('完成 TODO 5，列出随机采样的起点。')
        return
    assert r_starts == [3, 8, 13, 18, 23, 28]
    random.Random(8301).shuffle(r_starts)
    starts = r_starts[:batch_size]
    rx = None  # TODO 6：按首批起点截取输入。
    ry = None  # TODO 7：按同一起点加一的位置截取标签。
    if rx is None or ry is None:
        print('完成 TODO 6–7，检查每个标签对应下一位置。')
        return
    assert rx == [[13, 14, 15, 16, 17], [28, 29, 30, 31, 32]]
    assert all([x+1 for x in row] == target for row, target in zip(rx, ry))

    seq_offset = 0
    lane = (len(seq) - seq_offset - 1) // batch_size
    second_starts = None  # TODO 8：算出顺序分区第二批的两条行起点。
    if second_starts is None:
        print('完成 TODO 8，检查相邻批同行是否接上。')
        return
    assert second_starts == [5, 22]
    print('随机首批', starts, '顺序第二批', second_starts)


if __name__ == '__main__':
    main()
