# -*- coding: utf-8 -*-
"""4.9整节：协变量/标签/概念偏移、加权训练、在线预测与反馈。

纯人工数据，CPU/float64，复用已有环境，无安装、无下载。
目标适应特征与最终评估集分开；目标标签不参与密度比或先验估计。
"""
import csv
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

REPORT=ROOT/'reports/ch04'
FIGURES=REPORT/'figures'
DTYPE=torch.float64
CAP=20.


def make_covariate_data():
    rng=torch.Generator().manual_seed(4901)
    groups={}
    for name,n,mean,labelled in (
        ('source_domain',1200,-.8,False),('target_adapt',600,.8,False),
        ('source_domain_holdout',300,-.8,False),('target_domain_holdout',300,.8,False),
        ('source_train',2000,-.8,True),('source_validation',600,-.8,True),('target_test',2000,.8,True)):
        x=torch.randn(n,generator=rng,dtype=DTYPE)+mean
        y=x.square()+.1*torch.randn(n,generator=rng,dtype=DTYPE) if labelled else None
        groups[name]=(x,y)
    return groups


def fit_domain(source_x,target_x):
    x=torch.cat((source_x,target_x))
    z=torch.cat((torch.zeros_like(source_x),torch.ones_like(target_x)))
    a=torch.zeros((),dtype=DTYPE,requires_grad=True)
    b=torch.zeros((),dtype=DTYPE,requires_grad=True)
    history=[]
    for step in range(801):
        loss=F.binary_cross_entropy_with_logits(a*x+b,z)
        history.append({'step':step,'bce':loss.item(),'a':a.item(),'b':b.item()})
        if step==800:break
        loss.backward()
        with torch.no_grad():a-=.1*a.grad;b-=.1*b.grad
        a.grad=None;b.grad=None
    return a.detach(),b.detach(),history


def density_weights(x,a,b,n_source,n_target,cap=None):
    # g=P(domain=target|x)按域训练样本比例定义，赔率需乘πS/πT。
    log_ratio=a*x+b+math.log(n_source/n_target)
    if cap is not None:log_ratio=log_ratio.clamp(max=math.log(cap))
    return log_ratio.exp().detach()


def effective_sample_size(weight):
    return (weight.sum().square()/weight.square().sum()).item()


def fit_regression(x,y,weight,method):
    theta=torch.zeros(2,dtype=DTYPE,requires_grad=True)
    design=torch.stack((x,torch.ones_like(x)),dim=1)
    history=[]
    for step in range(401):
        errors=design@theta-y
        loss=(weight*errors.square()/2).mean()
        history.append({'method':method,'step':step,'weighted_half_mse':loss.item(),
                        'slope':theta[0].item(),'intercept':theta[1].item()})
        if step==400:break
        loss.backward()
        with torch.no_grad():theta-=.03*theta.grad
        theta.grad=None
    return theta.detach(),history


def half_mse(theta,x,y):
    return ((theta[0]*x+theta[1]-y).square().mean()/2).item()


def covariate_experiment(groups):
    source=groups['source_domain'][0];adapt=groups['target_adapt'][0]
    a,b,domain_history=fit_domain(source,adapt)
    n_source,n_target=len(source),len(adapt)
    # 域分类留出集各300个，评价时把训练域先验调整到1/2。
    sx=groups['source_domain_holdout'][0];tx=groups['target_domain_holdout'][0]
    balanced_offset=math.log(n_source/n_target)
    correct=int((a*sx+b+balanced_offset<0).sum()+(a*tx+b+balanced_offset>=0).sum())
    X,y=groups['source_train']
    oracle=(1.6*X).exp()
    weights={'unweighted':torch.ones_like(X),'oracle_raw':oracle,
             'oracle_cap20':oracle.clamp(max=CAP),
             'estimated_cap20':density_weights(X,a,b,n_source,n_target,CAP)}
    models={};history=[];risk_rows=[];weight_rows=[]
    for name,weight in weights.items():
        theta,rows=fit_regression(X,y,weight,name);models[name]=theta;history.extend(rows)
        risk_rows.append({'method':name,'slope':theta[0].item(),'intercept':theta[1].item(),
                          'source_train_half_mse':half_mse(theta,X,y),
                          'source_validation_half_mse':half_mse(theta,*groups['source_validation']),
                          'weighted_source_train_half_mse':rows[-1]['weighted_half_mse'],
                          'target_test_half_mse':half_mse(theta,*groups['target_test']),
                          'weight_mean':weight.mean().item(),'weight_max':weight.max().item(),
                          'ess':effective_sample_size(weight)})
    for i in range(len(X)):
        weight_rows.append({'index':i,'x':X[i].item(),'y':y[i].item(),
                            **{name:weight[i].item() for name,weight in weights.items()}})
    raw_est=density_weights(X,a,b,n_source,n_target)
    summary={'data_seed':4901,'source_mean':-.8,'target_mean':.8,'std':1.,'noise_std':.1,
             'conditional':'y=x^2+N(0,.1^2) in both domains','regressor':'slope*x+intercept (misspecified)',
             'domain_source_samples':n_source,'domain_target_samples':n_target,
             'domain_a':a.item(),'domain_b':b.item(),'bayes_domain_a':1.6,'bayes_domain_b':math.log(n_target/n_source),
             'ratio_prior_multiplier':n_source/n_target,'domain_balanced_holdout_correct':correct,
             'domain_balanced_holdout_samples':len(sx)+len(tx),'cap_fixed_before_results':CAP,
             'raw_estimated_weight_mean':raw_est.mean().item(),
             'wrong_uncorrected_odds_mean':(raw_est*n_target/n_source).mean().item(),
             'weighted_training_steps':400,'weighted_training_lr':.03,'domain_steps':800,'domain_lr':.1,
             'population_target_linear_optimum':[1.6,.36],'population_target_min_half_mse':1.005,
             'target_test_used_only_for_final_comparison':True,'models':risk_rows}
    return summary,domain_history,history,risk_rows,weight_rows


def binary_channel(n,prior_one,seed):
    n1=int(round(n*prior_one));n0=n-n1
    y=torch.cat((torch.zeros(n0,dtype=torch.long),torch.ones(n1,dtype=torch.long)))
    x=torch.cat((torch.zeros(n0*4//5),torch.ones(n0//5),torch.zeros(n1//5),torch.ones(n1*4//5))).to(DTYPE)
    assert len(x)==n
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed))
    return x[order],y[order]


def confusion_conditional(prediction,labels,k=2):
    counts=torch.zeros(k,k,dtype=DTYPE)
    for j in range(k):
        for i in range(k):counts[i,j]=((prediction==i)&(labels==j)).sum()
    return counts/counts.sum(dim=0,keepdim=True),counts


def binary_metrics(probability_one,labels):
    p=probability_one.clamp(1e-15,1-1e-15)
    prediction=(p>.5).long()
    loss=-(labels*p.log()+(1-labels)*(1-p).log()).mean().item()
    return {'ce':loss,'correct':int((prediction==labels).sum()),'samples':len(labels),
            'class0_recall':((prediction[labels==0]==0).double().mean().item()),
            'class1_recall':((prediction[labels==1]==1).double().mean().item())}


def label_shift_experiment():
    source_x,source_y=binary_channel(1000,.5,4904)
    adapt_x,adapt_audit_y=binary_channel(1000,.9,4905)
    test_x,test_y=binary_channel(1000,.9,4906)
    C,counts=confusion_conditional(source_x.long(),source_y)
    mu=torch.bincount(adapt_x.long(),minlength=2).to(DTYPE)/len(adapt_x)
    prior=torch.linalg.solve(C,mu)  # 只接校准C与无标签目标域预测比例。
    source_prior=torch.bincount(source_y,minlength=2).to(DTYPE)/len(source_y)
    if (prior<0).any() or not torch.isclose(prior.sum(),torch.tensor(1.,dtype=DTYPE)):
        raise ValueError('先验解不合法，需要检查假设/误差与约束估计。')
    weight=prior/source_prior
    p_source=.2+.6*test_x
    corrected=(p_source*weight[1])/(p_source*weight[1]+(1-p_source)*weight[0])
    joint=counts/counts.sum()
    joint_solution=torch.linalg.solve(joint,mu)
    singular=torch.tensor([[1.,1.],[0.,0.]],dtype=DTYPE)
    summary={'source_prior':source_prior.tolist(),'true_target_prior_audit_only':[.1,.9],
             'estimated_target_prior':prior.tolist(),'class_weights':weight.tolist(),'conditional_C':C.tolist(),
             'counts_rows_pred_columns_true':counts.tolist(),'joint_C':joint.tolist(),'joint_solution_is_weights':joint_solution.tolist(),
             'target_prediction_proportions':mu.tolist(),'matrix_condition_number':torch.linalg.cond(C).item(),
             'constant_classifier_matrix_rank':int(torch.linalg.matrix_rank(singular)),
             'source_posterior_is_known_for_controlled_channel':True,
             'target_adaptation_labels_used':False,'before':binary_metrics(p_source,test_y),
             'after':binary_metrics(corrected,test_y),
             'corrected_p_y1_at_x0':(.2*weight[1]/(.2*weight[1]+.8*weight[0])).item(),
             'corrected_p_y1_at_x1':(.8*weight[1]/(.8*weight[1]+.2*weight[0])).item()}
    rows=[]
    for role,x,y in (('source_calibration',source_x,source_y),('target_adapt_unlabelled',adapt_x,None),('target_test',test_x,test_y)):
        rows.extend({'role':role,'index':i,'x':int(x[i]),'label':int(y[i]) if y is not None else ''} for i in range(len(x)))
    # adapt_audit_y只用于验证人工生成器，不传给估计器，也不写入无标签CSV。
    assert int(adapt_audit_y.sum())==900
    return summary,rows


def concept_and_online_experiment():
    x=torch.linspace(-2,2,400,dtype=DTYPE)
    source_y=(x>0).long();target_y=1-source_y
    prediction=(x>0).long()
    da,db,_=fit_domain(x,x.clone())
    domain_prediction=(da*torch.cat((x,x))+db>=0).long()
    domain_labels=torch.cat((torch.zeros(400),torch.ones(400))).long()
    summary={'identical_feature_arrays':True,'source_accuracy':(prediction==source_y).double().mean().item(),
             'target_after_label_flip_accuracy':(prediction==target_y).double().mean().item(),
             'best_domain_accuracy_for_exactly_paired_features':.5,'concept_grid_points':400,
             'paired_feature_domain_parameters':[da.item(),db.item()],
             'paired_feature_domain_accuracy':(domain_prediction==domain_labels).double().mean().item()}
    rng=torch.Generator().manual_seed(4907)
    stream_x=torch.randn(800,generator=rng,dtype=DTYPE)
    w=torch.tensor(3.,dtype=DTYPE,requires_grad=True);b=torch.zeros((),dtype=DTYPE,requires_grad=True)
    rows=[]
    for t,xi in enumerate(stream_x):
        yi=(xi>0).to(DTYPE) if t<400 else (xi<0).to(DTYPE)
        # 时序：先给出预测，之后得到标签和损失，再做更新。
        logit=w*xi+b;frozen=3*xi
        loss=F.binary_cross_entropy_with_logits(logit,yi)
        row={'time':t,'x':xi.item(),'y':int(yi),'phase':'before' if t<400 else 'after',
             'weight_before':w.item(),'bias_before':b.item(),'online_prediction':int(logit.detach()>0),
             'frozen_prediction':int(frozen>0),'online_ce_before_update':loss.item(),
             'frozen_ce':F.binary_cross_entropy_with_logits(frozen,yi).item()}
        loss.backward()
        with torch.no_grad():w-=.2*w.grad;b-=.2*b.grad
        w.grad=None;b.grad=None
        row.update(weight_after=w.item(),bias_after=b.item());rows.append(row)
    summary['online']={'seed':4907,'lr':.2,'initial_parameters':[3.,0.],'switch_time':400,
                       'final_parameters':[w.item(),b.item()],'samples':800,'evaluation':'predict, reveal label, score, update',
                       'windows':[]}
    for start,end in ((0,400),(400,500),(600,800)):
        section=rows[start:end]
        summary['online']['windows'].append({'start':start,'end_exclusive':end,
             'online_ce':sum(r['online_ce_before_update'] for r in section)/len(section),
             'frozen_ce':sum(r['frozen_ce'] for r in section)/len(section),
             'online_accuracy':sum(r['online_prediction']==r['y'] for r in section)/len(section),
             'frozen_accuracy':sum(r['frozen_prediction']==r['y'] for r in section)/len(section)})
    return summary,rows


def feedback_and_group_metrics():
    rows=[]
    for strategy in ('raw_count_squared','rate_equal'):
        exposure=torch.tensor([600.,400.],dtype=DTYPE)
        for step in range(7):
            expected_clicks=.1*exposure
            rows.append({'strategy':strategy,'step':step,'exposure_A':exposure[0].item(),
                         'exposure_B':exposure[1].item(),'expected_clicks_A':expected_clicks[0].item(),
                         'expected_clicks_B':expected_clicks[1].item(),'true_rate_A':.1,'true_rate_B':.1})
            score=expected_clicks.square() if strategy=='raw_count_squared' else expected_clicks/exposure
            exposure=1000*score/score.sum()
    group=[]
    for name,tp,tn,fp,fn in (('A',90,90,10,10),('B',0,10,0,10)):
        n=tp+tn+fp+fn
        group.append({'group':name,'n':n,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
                      'accuracy':(tp+tn)/n,'tpr':tp/(tp+fn),'fpr':fp/(fp+tn),
                      'cost_fp1_fn5_per_sample':(fp+5*fn)/n})
    summary={'simulation':'deterministic expected clicks; allocation rule explicitly specified, not real deployment evidence',
             'same_true_click_rate':.1,'group_metrics':group,
             'overall_accuracy':sum(g['tp']+g['tn'] for g in group)/sum(g['n'] for g in group),
             'overall_cost_fp1_fn5_per_sample':sum(g['fp']+5*g['fn'] for g in group)/sum(g['n'] for g in group)}
    return summary,rows


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def draw_figures(groups,covariate,risks,weight_rows,label_shift,concept,online,feedback,feedback_rows):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'font.size':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    grid=torch.linspace(-3.5,3.5,300,dtype=DTYPE)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for role,name,color in (('source_domain','源域','#147E83'),('target_adapt','目标适应域','#D47F37')):
        axes[0].hist(groups[role][0].numpy(),bins=25,density=True,alpha=.5,label=name,color=color)
    axes[0].set(title='输入分布变了',xlabel='x',ylabel='密度');axes[0].legend(fontsize=9)
    axes[1].plot(grid,grid.square(),color='#173448',label='同一真实条件均值 x²')
    for name,color in (('unweighted','#D47F37'),('estimated_cap20','#147E83')):
        row=next(r for r in risks if r['method']==name)
        axes[1].plot(grid,row['slope']*grid+row['intercept'],color=color,label=name)
    axes[1].set(title='有限容量线性模型：重点区域不同',xlabel='x',ylabel='y / 预测');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_9_covariate.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    a,b=covariate['domain_a'],covariate['domain_b']
    axes[0].plot(grid,(a*grid+b).sigmoid(),label='域分类器估计')
    axes[0].plot(grid,(1.6*grid+math.log(.5)).sigmoid(),'--',label='已知分布的理论后验')
    axes[0].set(title='域标签：源0、目标1；训练比例2:1',xlabel='x',ylabel='P(目标域 | x)');axes[0].legend(fontsize=8)
    raw=(1.6*grid).exp();estimated=(a*grid+b+math.log(2)).exp()
    axes[1].semilogy(grid,raw,label='真实密度比 exp(1.6x)')
    axes[1].semilogy(grid,estimated,'--',label='估计赔率 × 2')
    axes[1].semilogy(grid,estimated.clamp(max=CAP),label='固定上限20')
    axes[1].set(title='概率、赔率和密度比要区分',xlabel='x',ylabel='目标密度 / 源密度');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_9_density_ratio.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));positions=range(4)
    axes[0].bar([i-.18 for i in positions],[r['source_validation_half_mse'] for r in risks],width=.35,label='源域验证')
    axes[0].bar([i+.18 for i in positions],[r['target_test_half_mse'] for r in risks],width=.35,label='目标域最终评估')
    labels=['不加权','真比率','真比率截断','估计比率截断']
    axes[0].set_xticks(list(positions),labels,rotation=12);axes[0].set(ylabel='半MSE',title='固定方案的跨域误差');axes[0].legend(fontsize=8)
    axes[1].bar(list(positions),[r['ess'] for r in risks],color=['#8094A0','#D47F37','#147E83','#173448'])
    axes[1].set_xticks(list(positions),labels,rotation=12);axes[1].set(ylabel='ESS（权重诊断）',title='原始训练样本数2000')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_9_risks.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for offset,key,name in ((-.22,'source_prior','源域'),(0,'true_target_prior_audit_only','目标真实（仅核对）'),(.22,'estimated_target_prior','无标签估计')):
        axes[0].bar([i+offset for i in range(2)],label_shift[key],width=.21,label=name)
    axes[0].set_xticks([0,1],['类别0','类别1']);axes[0].set(ylabel='类别比例',ylim=(0,1),title='标签偏移：条件通道保持不变');axes[0].legend(fontsize=8)
    axes[1].bar([-.16,.84],[.2,.8],width=.32,label='源域后验')
    axes[1].bar([.16,1.16],[label_shift['corrected_p_y1_at_x0'],label_shift['corrected_p_y1_at_x1']],width=.32,label='先验调整后')
    axes[1].axhline(.5,color='#8094A0',ls='--');axes[1].set_xticks([0,1],['x=0','x=1'])
    axes[1].set(ylabel='P(y=1 | x)',ylim=(0,1),title='后验变化，不能只数预测类别');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_9_label_shift.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    axes[0].plot(grid,(grid>0).double(),label='原概念');axes[0].plot(grid,(grid<0).double(),'--',label='翻转后的概念')
    axes[0].set(title='输入分布相同，标签规则改变',xlabel='x',ylabel='P(y=1 | x)');axes[0].legend(fontsize=8)
    for key,name in (('frozen_ce','冻结模型'),('online_ce_before_update','先预测再更新')):
        means=[sum(r[key] for r in online[start:start+25])/len(online[start:start+25]) for start in range(0,800,25)]
        axes[1].plot([start+12 for start in range(0,800,25)],means,label=name)
    axes[1].axvline(400,color='#8094A0',ls='--');axes[1].set(title='概念变化后：25步窗口平均',xlabel='时间',ylabel='更新前交叉熵');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_9_online.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for method,name in (('raw_count_squared','按总点击平方分配'),('rate_equal','按点击率分配')):
        rows=[r for r in feedback_rows if r['strategy']==method]
        axes[0].plot([r['step'] for r in rows],[r['exposure_A']/1000 for r in rows],'o-',label=name)
    axes[0].set(title='确定性期望点击模拟：真实点击率相同',xlabel='分配轮次',ylabel='A曝光份额',ylim=(0,1.05));axes[0].legend(fontsize=8)
    g=feedback['group_metrics'];axes[1].bar(['总体 n=220','A n=200','B n=20'],[feedback['overall_accuracy'],g[0]['accuracy'],g[1]['accuracy']],color=['#173448','#147E83','#D47F37'])
    axes[1].set(title='总体准确率会掩盖小群体误差',ylabel='准确率',ylim=(0,1))
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_9_feedback.png',dpi=170);plt.close(fig)


def main():
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    groups=make_covariate_data()
    covariate,domain_history,regression_history,risks,weights=covariate_experiment(groups)
    label_shift,label_rows=label_shift_experiment()
    concept,online=concept_and_online_experiment();feedback,feedback_rows=feedback_and_group_metrics()
    metrics={'date':'2026-10-27','purpose':'teacher demonstration, not learner completion','torch':torch.__version__,
             'device':'cpu','dtype':'float64','download':False,'covariate':covariate,
             'label_shift':label_shift,'concept':concept,'feedback_and_groups':feedback}
    with (REPORT/'4_9_metrics.json').open('w',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for name,rows in (('domain_epochs',domain_history),('regression_epochs',regression_history),('risks',risks),
                      ('weights',weights),('label_data',label_rows),('online',online),('feedback',feedback_rows)):
        write_csv(REPORT/('4_9_'+name+'.csv'),rows)
    cov_rows=[]
    for role,(x,y) in groups.items():
        cov_rows.extend({'role':role,'index':i,'x':x[i].item(),'label':y[i].item() if y is not None else ''} for i in range(len(x)))
    write_csv(REPORT/'4_9_covariate_data.csv',cov_rows)
    draw_figures(groups,covariate,risks,weights,label_shift,concept,online,feedback,feedback_rows)
    print('域分类器:',{k:v for k,v in covariate.items() if k!='models'})
    for row in risks:print('协变量纠正:',row)
    print('标签偏移:',label_shift);print('概念与在线:',concept);print('反馈与分组:',feedback)
    print('教师准备不等于用户掌握；下一节4.10实战Kaggle比赛：预测房价。')


if __name__=='__main__':main()
