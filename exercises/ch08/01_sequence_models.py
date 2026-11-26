"""将时间切分、训练和两种预测串起来。八处 TODO 等你填写。"""
from pathlib import Path
import runpy

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
H = runpy.run_path(str(ROOT/'lessons/ch08/01_sequence_models.py'))


def main():
    torch.set_num_threads(1)
    time, clean, observed = H['make_series'](120)
    tau = 4
    cutoff = 84
    features = None  # TODO 1：行i使用observed[i:i+tau]，共120-tau行。
    labels = None  # TODO 2：下一步标签，形状(116,1)。
    if features is None or labels is None:
        print('先填写 TODO 1-2，核对首行4个输入和第5个观测标签。')
        return
    assert features.shape == (116,4) and labels.shape == (116,1)
    assert torch.equal(features[0],observed[:4]) and labels[0,0]==observed[4]
    train_count = None  # TODO 3：只使用标签时刻不晚于cutoff的行数。
    if train_count is None:
        print('填写 TODO 3，训练边界用目标时刻检查。')
        return
    assert train_count == 80
    net = H['build_net'](tau,(6,))
    order = torch.Generator().manual_seed(8103)
    loader = DataLoader(TensorDataset(features[:train_count],labels[:train_count]),
                        batch_size=8,shuffle=True,generator=order)
    optimizer = torch.optim.Adam(net.parameters(),lr=.01,foreach=False)
    updates = 0
    for epoch in range(3):
        net.train()
        for X,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = None  # TODO 4：模型得分与y的批内平方误差之和。
            if loss is None:
                print('填写 TODO 4-6，完成训练更新。')
                return
            pass  # TODO 5：反向传播。
            pass  # TODO 6：执行一次Adam更新。
            updates += 1
    net.eval()
    with torch.no_grad():
        single = net(features[train_count:]).flatten()
        context = None  # TODO 7：只取到cutoff为止的最后tau个真值，整理成(1,4)。
        if context is None:
            print('填写 TODO 7，从已见历史取起始窗口。')
            return
        assert torch.equal(context,observed[cutoff-tau:cutoff].reshape(1,-1))
        preds = []
        for _ in range(36):
            next_value = net(context)
            preds.append(next_value)
            updated = None  # TODO 8：丢最旧一列，把本次预测接在最后。
            if updated is None:
                print('填写 TODO 8，闭环窗口只补回自己的预测。')
                return
            context = updated
        closed = torch.cat(preds,dim=1).flatten()
    assert updates==30 and single.shape==closed.shape==(36,)
    torch.testing.assert_close(single[0],closed[0])
    print('训练行',train_count,'更新',updates,'输出长度',len(closed))
    print('单步 RMSE',H['metrics'](single,observed[cutoff:])['rmse'],
          '闭环 RMSE',H['metrics'](closed,observed[cutoff:])['rmse'])


if __name__ == '__main__':
    main()
