# -*- coding: utf-8 -*-
"""4.4 综合实践：60条开发数据做5折选次数，最后40条测试只评估一次。"""
import torch

DTYPE = torch.float64


def features(x, degree):
    columns = [torch.ones_like(x)]
    for power in range(1, degree+1):
        columns.append(columns[-1]*x/power)
    return torch.cat(columns, dim=1)


def main():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(4407)
    x = 2*torch.rand(100, 1, generator=generator, dtype=DTYPE)-1
    true_w = torch.tensor([5., 1.2, -3.4, 5.6], dtype=DTYPE).reshape(-1, 1)
    y = features(x, 3)@true_w + .1*torch.randn(100, 1, generator=generator, dtype=DTYPE)
    dev_x, dev_y, test_x, test_y = x[:60], y[:60], x[60:], y[60:]
    k = None  # TODO 1：使用5折。
    if k is None:
        print('请集中完成八处TODO；先只用60条开发数据选择模型。'); return
    permutation = torch.randperm(60, generator=torch.Generator().manual_seed(4408))
    folds = list(permutation.tensor_split(k))
    summaries = []
    for degree in (1, 3, 10):
        X = features(dev_x, degree)
        errors = []
        for fold in range(k):
            valid_ids = None  # TODO 2：本次留下的那一折。
            train_ids = None  # TODO 3：拼接其余各折，不能含valid_ids。
            if valid_ids is None or train_ids is None:
                print('请先完成互斥的训练/验证索引。'); return
            w = None  # TODO 4：lstsq(X[train_ids],dev_y[train_ids],rcond=1e-12,driver='gelsd').solution。
            if w is None:
                print('请独立拟合每一折，不能沿用上一折参数。'); return
            valid_mse = None  # TODO 5：这一折预测与标签的平均平方误差，转为Python数值。
            if valid_mse is None:
                print('请计算验证MSE。'); return
            errors.append(valid_mse)
        summaries.append({'degree': degree, 'mean_mse': sum(errors)/k, 'fold_errors': errors})
    best_degree = None  # TODO 6：按mean_mse最小选择次数；不能看test_y。
    if best_degree is None:
        print('请先根据交叉验证选择次数。'); return
    final_w = None  # TODO 7：选好次数后，用全部60条开发数据重新拟合。
    if final_w is None:
        print('请重新拟合最终模型。'); return
    test_mse = None  # TODO 8：最后在40条测试数据上评估一次MSE。
    if test_mse is None:
        print('请完成最终测试评估。'); return
    print('折大小:', [len(ids) for ids in folds], '| 每折训练/验证:', 60-len(folds[0]), len(folds[0]))
    print('各候选:', summaries)
    print('选择次数:', best_degree, '| 最终参数形状:', final_w.shape, '| 测试MSE:', test_mse)
    print('解释：为什么不能按测试MSE选次数？每折为什么必须重新拟合？')
    print('解释：训练误差高是否一定是模型太简单？带噪声的泛化MSE为什么不一定能到0？')


if __name__ == '__main__':
    main()
