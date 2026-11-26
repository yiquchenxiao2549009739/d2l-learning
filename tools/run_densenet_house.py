"""7.7教材练习5。对同一房价训练CSV做五折特征复用实验。"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
HOUSE = runpy.run_path(str(ROOT / 'lessons/ch04/10_kaggle_house_prices.py'))
DENSE = runpy.run_path(str(ROOT / 'lessons/ch07/07_densenet.py'))

import numpy as np
import torch
from torch import nn


class PriceMLP(nn.Module):
    def __init__(self, inputs, dense, seed):
        super().__init__()
        self.dense = dense
        self.stem = nn.Linear(inputs, 32)
        self.first = nn.Linear(32, 16 if dense else 29)
        self.second = nn.Linear(48 if dense else 29, 16 if dense else 13)
        self.head = nn.Linear(64 if dense else 13, 1)
        with torch.random.fork_rng(devices=[]):
            for i, layer in enumerate((self.stem, self.first, self.second, self.head)):
                torch.manual_seed(seed + i)
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu' if i < 3 else 'linear')
                nn.init.zeros_(layer.bias)

    def forward(self, X):
        X = self.stem(X).relu()
        H = self.first(X).relu()
        X = torch.cat((X,H), dim=1) if self.dense else H
        H = self.second(X).relu()
        X = torch.cat((X,H), dim=1) if self.dense else H
        return self.head(X).squeeze(1)


def run():
    torch.set_num_threads(4)
    raw = HOUSE['DATA'] / 'kaggle_house_pred_train.csv'
    data = raw.read_bytes()
    if hashlib.sha1(data).hexdigest() != HOUSE['EXPECTED']['train']:
        raise ValueError('房价训练CSV哈希不符')
    train = HOUSE['pd'].read_csv(raw, keep_default_na=False, na_values=['NA',''])
    assert train.shape == (1460,81) and train.SalePrice.gt(0).all() and train.Id.is_unique
    src = train.drop(columns=['Id','SalePrice'])
    log_y = torch.tensor(np.log(train.SalePrice.to_numpy(dtype=np.float64)), dtype=torch.float64)
    folds = HOUSE['fold_indices'](len(src))
    oof = {name: torch.empty_like(log_y) for name in ('plain','dense')}
    rows, histories = [], []
    for fold, val_ids in enumerate(folds):
        train_ids = np.concatenate([ids for j,ids in enumerate(folds) if j != fold])
        processor = HOUSE['Preprocessor']().fit(src.iloc[train_ids])
        X = processor.transform(src.iloc[train_ids])
        V = processor.transform(src.iloc[val_ids])
        y = log_y[train_ids]
        offset = y.mean()
        hashes = []
        for name in ('plain','dense'):
            model = PriceMLP(X.shape[1], name == 'dense', 77100+10*fold).double()
            hashes.append(hashlib.sha256(model.stem.weight.detach().numpy().tobytes()).hexdigest())
            weights = [p for n,p in model.named_parameters() if n.endswith('weight')]
            biases = [p for n,p in model.named_parameters() if n.endswith('bias')]
            opt = torch.optim.Adam([dict(params=weights,weight_decay=.01),dict(params=biases,weight_decay=0.)],
                                   lr=.003, foreach=False)
            order_rng = torch.Generator().manual_seed(77200+fold)
            for epoch in range(1,101):
                for batch in torch.randperm(len(y), generator=order_rng).split(64):
                    opt.zero_grad(set_to_none=True)
                    loss = (model(X[batch])+offset-y[batch]).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'{name} fold {fold}')
                    loss.backward()
                    opt.step()
                if epoch in (1,20,50,100):
                    with torch.no_grad():
                        prediction = model(V)+offset
                        tr_rmse = (model(X)+offset-y).square().mean().sqrt().item()
                        val_rmse = (prediction-log_y[val_ids]).square().mean().sqrt().item()
                    histories.append(dict(model=name, fold=fold, epoch=epoch, train_log_rmse=tr_rmse,
                                          validation_log_rmse=val_rmse))
            oof[name][val_ids] = prediction
            rows.append(dict(model=name, fold=fold, train_n=len(y), valid_n=len(val_ids), features=X.shape[1],
                             parameters=sum(p.numel() for p in model.parameters()), updates=1900,
                             stem_weight_sha256=hashes[-1], validation_log_rmse=val_rmse))
            print(name,'fold',fold,'log RMSE',f'{val_rmse:.6f}',flush=True)
        assert hashes[0] == hashes[1]
    result = dict(folds=rows, oof_log_rmse={name:(values-log_y).square().mean().sqrt().item() for name,values in oof.items()},
                  history=histories, rows=1460, fold_seed=41001, epochs=100, batch_size=64, lr=.003,
                  train_sha256=hashlib.sha256(data).hexdigest(), test_read=False,
                  protocol='同折预处理与标签对数中心化；近参数量对照，验证折只用于评估；固定100轮，不拟合提交版')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', action='store_true')
    args = p.parse_args()
    result = run()
    if args.record:
        DENSE['write_json']('7_7_house.json', result)
        DENSE['write_csv']('7_7_house_cv.csv', result['folds'])
    print(json.dumps(result['oof_log_rmse'],ensure_ascii=False,indent=2))
