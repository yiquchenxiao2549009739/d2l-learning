# -*- coding: utf-8 -*-
"""4.10整节：真实房价CSV、折内预处理、对数指标、Adam/K折和本地提交文件。

默认只读本项目缓存，无安装、无自动下载、无Kaggle账户操作。
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import torch
from torch import nn

REPORT=ROOT/'reports/ch04'
FIGURES=REPORT/'figures'
ARTIFACTS=ROOT/'artifacts/ch04'
DATA=ROOT/'data/house_prices'
DTYPE=torch.float64
EXPECTED={'train':'585e9cc93e70b39160e7921475f9bcd7d31219ce',
          'test':'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'}
CONFIGS=[
    dict(name='A_raw_linear',target='price',hidden=0,lr=5.,decay=0.,dropout=0.,standardize=True,missing=False),
    dict(name='B_log_linear',target='log',hidden=0,lr=.01,decay=0.,dropout=0.,standardize=True,missing=False),
    dict(name='C_log_decay',target='log',hidden=0,lr=.01,decay=.01,dropout=0.,standardize=True,missing=False),
    dict(name='D_log_lr03',target='log',hidden=0,lr=.03,decay=.01,dropout=0.,standardize=True,missing=False),
    dict(name='E_log_mlp64',target='log',hidden=64,lr=.01,decay=.01,dropout=0.,standardize=True,missing=False),
    dict(name='F_log_dropout',target='log',hidden=64,lr=.01,decay=.01,dropout=.2,standardize=True,missing=False),
    dict(name='G_no_scale',target='log',hidden=0,lr=.01,decay=.01,dropout=0.,standardize=False,missing=False),
    dict(name='H_missing_flags',target='log',hidden=0,lr=.01,decay=.01,dropout=0.,standardize=True,missing=True),
]


class Preprocessor:
    """只在训练行fit，保存数值统计及类别词表；未知类别有专门列。"""
    def __init__(self,standardize=True,missing=False):
        self.standardize=standardize;self.missing=missing

    def fit(self,frame):
        self.numeric=frame.select_dtypes(exclude='object').columns.tolist()
        self.categorical=[c for c in frame.columns if c not in self.numeric]
        self.means=frame[self.numeric].mean().fillna(0.)
        self.scales=frame[self.numeric].std(ddof=1).fillna(1.)
        self.scales=self.scales.where(self.scales.abs()>1e-12,1.)
        self.levels={c:sorted(frame[c].dropna().astype(str).unique().tolist()) for c in self.categorical}
        self.flags=[c for c in self.numeric if frame[c].isna().any()] if self.missing else []
        self.columns=self.numeric.copy()
        for c in self.categorical:
            assert '__MISSING__' not in self.levels[c] and '__UNKNOWN__' not in self.levels[c]
            self.columns += [c+'='+s for s in self.levels[c]+['__MISSING__','__UNKNOWN__']]
        self.columns += [c+'__is_missing' for c in self.flags]
        assert len(self.columns)==len(set(self.columns))
        return self

    def transform(self,frame):
        observed=frame[self.numeric]
        numeric=observed.fillna(self.means)
        if self.standardize:numeric=(numeric-self.means)/self.scales
        arrays=[numeric.to_numpy(dtype=np.float64)]
        for c in self.categorical:
            values=frame[c].astype('object').where(frame[c].notna(),'__MISSING__').astype(str)
            levels=self.levels[c]+['__MISSING__','__UNKNOWN__']
            values=values.where(values.isin(levels),'__UNKNOWN__')
            arrays.append(np.column_stack([(values==s).to_numpy(dtype=np.float64) for s in levels]))
        if self.flags:arrays.append(frame[self.flags].isna().to_numpy(dtype=np.float64))
        result=np.concatenate(arrays,axis=1)
        if result.shape[1]!=len(self.columns) or not np.isfinite(result).all():raise ValueError('预处理结果异常')
        return torch.from_numpy(result)

    def state(self):
        return {'standardize':self.standardize,'missing':self.missing,'numeric':self.numeric,
                'categorical':self.categorical,'means':self.means.to_dict(),'scales':self.scales.to_dict(),
                'levels':self.levels,'flags':self.flags,'columns':self.columns}


def read_data():
    frames={};records=[]
    for kind in ('train','test'):
        path=DATA/('kaggle_house_pred_'+kind+'.csv')
        if not path.is_file():raise FileNotFoundError('缺少已校验本地CSV，请先阅读tools/prepare_house_prices.py')
        raw=path.read_bytes()
        if hashlib.sha1(raw).hexdigest()!=EXPECTED[kind]:raise ValueError('比赛数据哈希不符，保留文件并检查')
        # 保留MasVnrType等字段的字面类别"None"；NA/空字段才按缺失解析。
        frames[kind]=pd.read_csv(path,keep_default_na=False,na_values=['NA',''])
        records.append({'role':kind,'filename':path.name,'bytes':len(raw),'sha1':hashlib.sha1(raw).hexdigest(),
                        'sha256':hashlib.sha256(raw).hexdigest()})
    assert frames['train'].shape==(1460,81) and frames['test'].shape==(1459,80)
    assert 'SalePrice' not in frames['test'] and frames['train'].SalePrice.gt(0).all()
    assert frames['train'].Id.is_unique and frames['test'].Id.is_unique
    return frames['train'],frames['test'],records


def fold_indices(n,k=5,seed=41001):
    if not 1<k<=n:raise ValueError('折数不合法')
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).numpy()
    return [part.copy() for part in np.array_split(order,k)]


def get_model(width,config,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if config['hidden']:
            net=nn.Sequential(nn.Linear(width,config['hidden']),nn.ReLU(),nn.Dropout(config['dropout']),nn.Linear(config['hidden'],1)).to(dtype=DTYPE)
            with torch.no_grad():
                nn.init.kaiming_normal_(net[0].weight,mode='fan_in',nonlinearity='relu')
                nn.init.normal_(net[3].weight,std=.01)
                net[0].bias.zero_();net[3].bias.zero_()
        else:
            net=nn.Linear(width,1).to(dtype=DTYPE)
            with torch.no_grad():net.weight.zero_();net.bias.zero_()
    return net


def state_hash(net):
    h=hashlib.sha256()
    for key,value in net.state_dict().items():h.update(key.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def predicted_log(net,X,mode,offset):
    raw=net(X).reshape(-1)
    return raw+offset if mode=='log' else raw.clamp(min=1.).log()


def fit_model(X,y,VX,Vy,config,epochs=100,fold=0):
    net=get_model(X.shape[1],config,41010+fold)
    initial_hash=state_hash(net)
    offset=y.log().mean().item() if config['target']=='log' else 0.
    target=y.log()-offset if config['target']=='log' else y
    weights=[p for name,p in net.named_parameters() if name.endswith('weight')]
    biases=[p for name,p in net.named_parameters() if name.endswith('bias')]
    optimizer=torch.optim.Adam([{'params':weights,'weight_decay':config['decay']},
                               {'params':biases,'weight_decay':0.}],lr=config['lr'],foreach=False)
    generator=torch.Generator().manual_seed(41020+fold);history=[];updates=0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41030+fold)
        for epoch in range(1,epochs+1):
            net.train();order=torch.randperm(len(y),generator=generator)
            for start in range(0,len(y),64):
                batch=order[start:start+64]
                optimizer.zero_grad(set_to_none=True)
                loss=(net(X[batch]).reshape(-1)-target[batch]).square().mean()
                if not torch.isfinite(loss):raise ValueError('损失非有限：'+config['name'])
                loss.backward();optimizer.step();updates+=1
            net.eval()
            with torch.no_grad():
                train_rmse=(predicted_log(net,X,config['target'],offset)-y.log()).square().mean().sqrt().item()
                valid_rmse=(predicted_log(net,VX,config['target'],offset)-Vy.log()).square().mean().sqrt().item() if VX is not None else None
            history.append({'candidate':config['name'],'fold':fold,'epoch':epoch,'updates':updates,
                            'train_log_rmse':train_rmse,'validation_log_rmse':valid_rmse})
    net.eval()
    return net,offset,history,initial_hash


def missingness_example():
    true=np.array([80.,90.,100.,110.,200.,220.]);observed=true.copy();observed[-2:]=np.nan
    imputed=np.where(np.isnan(observed),np.nanmean(observed),observed)
    return {'latent_values':true.tolist(),'observed_values':[None if np.isnan(v) else float(v) for v in observed],
            'mean_of_observed':float(np.nanmean(observed)),'mean_when_missing':float(true[-2:].mean()),
            'imputed_values':imputed.tolist(),'meaning':'constructed missingness depends on latent value; observed mean need not match missing cases'}


def run_experiment(train,test,epochs=100):
    features=train.drop(columns=['Id','SalePrice']);test_features=test.drop(columns=['Id'])
    y=torch.tensor(train.SalePrice.to_numpy(dtype=np.float64),dtype=DTYPE)
    folds=fold_indices(len(train));oof_rows=[];scores=[];histories=[];fold_states={};fold_summary=[]
    constant=np.empty(len(train));fold_assignment=np.empty(len(train),dtype=int)
    for fold,valid in enumerate(folds):
        fit=np.concatenate([part for i,part in enumerate(folds) if i!=fold])
        constant[valid]=float(y[fit].log().mean());fold_assignment[valid]=fold
    for config in CONFIGS:
        oof=np.full(len(train),np.nan);fold_scores=[]
        for fold,valid in enumerate(folds):
            fit=np.concatenate([part for i,part in enumerate(folds) if i!=fold])
            pp=Preprocessor(config['standardize'],config['missing']).fit(features.iloc[fit])
            X=pp.transform(features.iloc[fit]);VX=pp.transform(features.iloc[valid])
            net,offset,history,initial_hash=fit_model(X,y[fit],VX,y[valid],config,epochs,fold)
            with torch.no_grad():prediction=predicted_log(net,VX,config['target'],offset).numpy()
            oof[valid]=prediction;fold_rmse=float(np.sqrt(np.mean((prediction-np.log(train.SalePrice.to_numpy()[valid]))**2)))
            fold_scores.append(fold_rmse);histories.extend(history)
            key=config['name']+':'+str(fold)
            fold_states[key]={'config':config.copy(),'preprocessor':pp.state(),'offset':offset,
                              'state_dict':{k:v.detach().clone() for k,v in net.state_dict().items()}}
            fold_summary.append({'candidate':config['name'],'fold':fold,'train_n':len(fit),'valid_n':len(valid),
                                 'feature_n':X.shape[1],'parameters':sum(p.numel() for p in net.parameters()),
                                 'updates':history[-1]['updates'],'last_batch_n':len(fit)%64 or 64,
                                 'initial_state_sha256':initial_hash,'train_log_rmse':history[-1]['train_log_rmse'],
                                 'validation_log_rmse':fold_rmse})
        assert np.isfinite(oof).all()
        residual=oof-np.log(train.SalePrice.to_numpy())
        score=float(np.sqrt(np.mean(residual**2)))
        scores.append({**config,'epochs':epochs,'oof_log_rmse':score,'mean_fold_log_rmse':float(np.mean(fold_scores)),
                       'std_fold_log_rmse':float(np.std(fold_scores,ddof=1))})
        oof_rows.extend({'candidate':config['name'],'Id':int(train.Id.iloc[i]),'fold':int(fold_assignment[i]),
                         'actual_log_price':float(y[i].log()),'predicted_log_price':float(oof[i])} for i in range(len(train)))
        print('五折:',config['name'],'OOF=',score,'各折=',fold_scores,flush=True)
    selected=min(scores,key=lambda row:row['oof_log_rmse'])
    config=next(c for c in CONFIGS if c['name']==selected['name'])
    pp=Preprocessor(config['standardize'],config['missing']).fit(features)
    X=pp.transform(features);TX=pp.transform(test_features)
    net,offset,full_history,_=fit_model(X,y,None,None,config,epochs,99)
    with torch.no_grad():log_prices=predicted_log(net,TX,config['target'],offset);prices=log_prices.exp().numpy()
    if not np.isfinite(prices).all() or not (prices>0).all():raise ValueError('最终预测不是有限正价格')
    submission=pd.DataFrame({'Id':test.Id.to_numpy(),'SalePrice':prices})
    assert submission.shape==(1459,2) and submission.Id.is_unique
    submission.to_csv(REPORT/'4_10_submission.csv',index=False,float_format='%.10f')
    full_state={'config':config,'preprocessor':pp.state(),'offset':offset,
                'state_dict':{k:v.detach().clone() for k,v in net.state_dict().items()}}
    checkpoint=ARTIFACTS/'4_10_states.pt';torch.save({'folds':fold_states,'full':full_state},checkpoint)
    label_log=np.log(train.SalePrice.to_numpy(dtype=np.float64))
    baseline=float(np.sqrt(np.mean((constant-label_log)**2)))
    summary={'fold_seed':41001,'folds':5,'epochs':epochs,'batch_size':64,
             'selection_metric':'pooled out-of-fold natural-log RMSE on all 1460 labelled rows',
             'constant_log_mean_oof_rmse':baseline,'selected':selected,
             'final_features':X.shape[1],'final_parameters':sum(p.numel() for p in net.parameters()),
             'final_train_log_rmse':full_history[-1]['train_log_rmse'],'final_updates':full_history[-1]['updates'],
             'final_last_batch_n':len(y)%64 or 64,'test_rows':len(test),
             'prediction_min':float(prices.min()),'prediction_max':float(prices.max()),
             'submission':'reports/ch04/4_10_submission.csv','kaggle_uploaded':False,'kaggle_score':None,
             'checkpoint_local_only':'artifacts/ch04/4_10_states.pt',
             'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    assignments=[{'Id':int(train.Id.iloc[i]),'row_index':i,'validation_fold':int(fold_assignment[i])} for i in range(len(train))]
    return summary,scores,fold_summary,histories,full_history,oof_rows,assignments


def draw_figures(train,test,summary,scores,fold_summary,histories,full_history,oof_rows,diagnostics):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'font.size':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,2.8));ax.set(xlim=(0,11),ylim=(0,3));ax.axis('off')
    labels=['划分五折','折内拟合预处理','训练并收集OOF','选固定方案','全量重训/导出']
    for i,label in enumerate(labels):
        x=.1+i*2.2;ax.add_patch(FancyBboxPatch((x,1.35),1.9,.65,boxstyle='round,pad=.03',facecolor='#F1F7F8',edgecolor='#147E83'))
        ax.text(x+.95,1.675,label,ha='center',va='center',fontsize=11)
        if i<4:ax.annotate('',xy=(x+2.18,1.675),xytext=(x+1.95,1.675),arrowprops={'arrowstyle':'->','color':'#147E83'})
    ax.text(5.5,.7,'验证行不参与该折的填补值、标准差、类别词表或标签均值估计。',ha='center',fontsize=11)
    fig.tight_layout();fig.savefig(FIGURES/'4_10_workflow.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].hist(train.SalePrice/10000,bins=35,color='#147E83');axes[0].set(xlabel='SalePrice（万美元）',ylabel='房屋数',title='价格尺度和长尾')
    axes[1].hist(np.log(train.SalePrice),bins=35,color='#D47F37');axes[1].set(xlabel='自然对数价格',ylabel='房屋数',title='模型预测log价格的目标')
    fig.tight_layout();fig.savefig(FIGURES/'4_10_prices.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    missing=train.drop(columns=['Id','SalePrice']).isna().sum().sort_values(ascending=False).head(8)
    axes[0].barh(missing.index[::-1],missing.values[::-1],color='#147E83');axes[0].set(xlabel='训练集缺失数',title='缺失比例高不等于直接删除')
    axes[1].bar(['数值列','类别列','全量编码列'],[36,43,summary['final_features']],color=['#8094A0','#D47F37','#147E83']);axes[1].set(ylabel='列数',title='原始79个输入与最终特征')
    fig.tight_layout();fig.savefig(FIGURES/'4_10_preprocessing.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8));values=[r['oof_log_rmse'] for r in scores]
    ax.bar([r['name'][0] for r in scores],values,color=['#D47F37' if r['name']==summary['selected']['name'] else '#147E83' for r in scores])
    ax.set_yscale('log');ax.set(xlabel='固定候选A–H，定义见讲义表格',ylabel='OOF log-RMSE（对数轴显示）',title='同一五折下的八方案比较')
    for i,value in enumerate(values):ax.text(i,value*1.08,'%.4f'%value,ha='center',fontsize=9)
    ax.axhline(summary['constant_log_mean_oof_rmse'],color='#8094A0',ls='--',label='折内常数基线');ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(FIGURES/'4_10_cv.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));chosen=summary['selected']['name']
    first=[r for r in histories if r['candidate']==chosen and r['fold']==0]
    for key,label in [('train_log_rmse','训练'),('validation_log_rmse','验证')]:axes[0].plot([r['epoch'] for r in first],[r[key] for r in first],label=label)
    axes[0].set(xlabel='轮数',ylabel='log-RMSE',title='选定方案第一折');axes[0].legend(fontsize=9)
    for name,label in [('A_raw_linear','训练原价MSE'),('B_log_linear','训练log价格MSE')]:
        rows=[r for r in histories if r['candidate']==name]
        mean=[np.mean([r['validation_log_rmse'] for r in rows if r['epoch']==epoch]) for epoch in range(1,len(first)+1)]
        axes[1].semilogy(range(1,len(mean)+1),mean,label=label)
    axes[1].set(xlabel='轮数',ylabel='五折平均验证log-RMSE',title='共同评估指标，训练目标不同');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_10_training.png',dpi=170);plt.close(fig)
    selected=[r for r in oof_rows if r['candidate']==chosen]
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));actual=[r['actual_log_price'] for r in selected];pred=[r['predicted_log_price'] for r in selected]
    axes[0].scatter(actual,pred,s=9,alpha=.4,color='#147E83');lo=min(actual+pred);hi=max(actual+pred)
    axes[0].plot([lo,hi],[lo,hi],color='#D47F37');axes[0].set(xlabel='真实log价格',ylabel='OOF预测log价格',title='每条训练样本只用未见过它的模型预测')
    submission=pd.read_csv(REPORT/'4_10_submission.csv')
    axes[1].hist(submission.SalePrice/10000,bins=35,color='#D47F37');axes[1].set(xlabel='预测价格（万美元）',ylabel='测试房屋数',title='1459条预测，真实测试标签未知')
    fig.tight_layout();fig.savefig(FIGURES/'4_10_predictions.png',dpi=170);plt.close(fig)
    example=diagnostics['missingness_example'];fig,ax=plt.subplots(figsize=(8.5,3.4))
    ax.bar(np.arange(6)-.17,example['latent_values'],width=.34,label='人工真实值')
    ax.bar(np.arange(6)+.17,example['imputed_values'],width=.34,label='观测均值填补')
    ax.set_xticks(range(6),['已观测1','已观测2','已观测3','已观测4','缺失1','缺失2']);ax.set(ylabel='人工面积',title='非随机缺失：已观测均值95，缺失组实际均值210');ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(FIGURES/'4_10_missingness.png',dpi=170);plt.close(fig)


def main(epochs=100):
    torch.set_num_threads(1)
    for folder in (REPORT,FIGURES,ARTIFACTS):folder.mkdir(parents=True,exist_ok=True)
    train,test,sources=read_data()
    # 仅报告教材全量合并的编码维度；不把这个矩阵用于严格CV。
    combined=pd.concat((train.drop(columns=['Id','SalePrice']),test.drop(columns=['Id'])),ignore_index=True)
    legacy_width=pd.get_dummies(combined,dummy_na=True).shape[1]
    diagnostics={'raw_train_shape':list(train.shape),'raw_test_shape':list(test.shape),
                 'numeric_inputs':36,'categorical_inputs':43,'textbook_combined_encoding_columns':legacy_width,
                 'csv_na_policy':'keep_default_na=False; na_values=[NA, empty]; preserve literal None category',
                 'train_MasVnrType_literal_None':int((train.MasVnrType=='None').sum()),
                 'train_MasVnrType_missing':int(train.MasVnrType.isna().sum()),
                 'missingness_example':missingness_example(),'strict_fold_local_preprocessing':True,
                 'target_transform':'natural log, not log1p; log models centered by fold training label mean',
                 'adam':'coupled L2 on weight matrices only; biases excluded; not AdamW'}
    summary,scores,folds,histories,full_history,oof,assignments=run_experiment(train,test,epochs)
    metrics={'date':'2026-10-27','purpose':'teacher demonstration, not learner completion','device':'cpu','dtype':'float64',
             'torch':torch.__version__,'pandas':pd.__version__,'data_sources':sources,'diagnostics':diagnostics,
             'experiment':summary,'candidate_scores':scores}
    with (REPORT/'4_10_metrics.json').open('w',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for name,rows in [('scores',scores),('folds',folds),('epochs',histories),('full_epochs',full_history),('oof',oof),('fold_assignments',assignments)]:
        table=pd.DataFrame(rows)
        if name=='oof':table=table.drop(columns=['actual_log_price'])
        table.to_csv(REPORT/('4_10_'+name+'.csv'),index=False)
    draw_figures(train,test,summary,scores,folds,histories,full_history,oof,diagnostics)
    print('原始形状/教材合并编码维度:',diagnostics)
    print('最终方案与本地提交文件:',summary)
    print('没有登录或上传Kaggle，线上成绩未知。教师运行不记作用户掌握；下一节5.1层和块。')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args()
    if args.epochs<1:parser.error('--epochs必须为正整数')
    main(args.epochs)
