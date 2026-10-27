# -*- coding: utf-8 -*-
"""4.10综合练习：137条真实训练行、五折、23条预测，八处TODO留空。"""
import importlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
lesson=importlib.import_module('lessons.ch04.10_kaggle_house_prices')


def fit_predict(raw_fit,log_y,raw_valid):
    preprocessor=None  # TODO 3：新建并只在raw_fit上拟合预处理器。
    if preprocessor is None:
        print('请先补齐折内预处理。');return None
    X=preprocessor.transform(raw_fit);VX=preprocessor.transform(raw_valid)
    offset=log_y.mean().item()
    net=None  # TODO 4：输入维度为X.shape[1]、输出1的float64线性模型。
    if net is None:return None
    with torch.no_grad():net.weight.zero_();net.bias.zero_()
    optimizer=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=0.,foreach=False)
    generator=torch.Generator().manual_seed(41082)
    for _ in range(50):
        order=torch.randperm(len(X),generator=generator)
        for start in range(0,len(X),32):
            index=order[start:start+32]
            optimizer.zero_grad(set_to_none=True)
            prediction=net(X[index]).reshape(-1)
            loss=None  # TODO 5：预测与中心化log标签之间的平均平方误差。
            if loss is None:return None
            pass  # TODO 6：反向传播。
            if net.weight.grad is None:
                print('请先反向传播。');return None
            pass  # TODO 7：优化器更新。
    net.eval()
    with torch.no_grad():
        validation_log_prediction=None  # TODO 8：预测展平，并加回该训练集的log均值。
    if validation_log_prediction is None:return None
    return validation_log_prediction.numpy(),len(preprocessor.columns)


def main():
    torch.set_num_threads(1)
    train,test,_=lesson.read_data();train=train.iloc[:137].copy();test=test.iloc[:23].copy()
    features=None  # TODO 1：只保留训练输入，排除Id与SalePrice。
    log_y=None  # TODO 2：将真实房价转为float64张量，并取自然对数。
    if features is None or log_y is None:
        print('请集中填写八处TODO，完成真实数据的小型五折流程。');return
    folds=lesson.fold_indices(137,5,41081);oof=np.full(137,np.nan)
    for fold,valid in enumerate(folds):
        fit=np.concatenate([part for i,part in enumerate(folds) if i!=fold])
        result=fit_predict(features.iloc[fit],log_y[fit],features.iloc[valid])
        if result is None:return
        prediction,width=result;oof[valid]=prediction
        print('折/训练数/验证数/编码维度/log-RMSE:',fold,len(fit),len(valid),width,
              float(np.sqrt(np.mean((prediction-log_y[valid].numpy())**2))))
    print('五折覆盖数/OOF log-RMSE:',int(np.isfinite(oof).sum()),float(np.sqrt(np.mean((oof-log_y.numpy())**2))))
    result=fit_predict(features,log_y,test.drop(columns=['Id']))
    if result is None:return
    prediction,_=result
    path=ROOT/'artifacts/ch04/4_10_practice_submission.csv';path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'Id':test.Id.to_numpy(),'SalePrice':np.exp(prediction)}).to_csv(path,index=False)
    print('23行练习预测:',path,'；正式比赛使用演示程序的1459行文件。')
    print('解释：为什么每折重新fit预处理？为什么Id要保留到导出却不作输入？')
    print('解释：OOF分数和Kaggle线上分数有什么区别？为什么生成预测后要exp还原？')


if __name__=='__main__':main()
