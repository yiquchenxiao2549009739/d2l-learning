# -*- coding: utf-8 -*-
"""4.3：用 Sequential/Linear/ReLU 管理 MLP，训练与三类对照一次完成。

已有 Fashion-MNIST，CPU/float32，无下载或安装。先读 build_model、train_epoch、
evaluate、fit；额外固定子集比较深度/学习率、激活及初始化，不用测试集调参。
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST

REPORT = ROOT / "reports/ch04"
FIGURES = REPORT / "figures"


def make_loader(dataset, batch_size=256, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=False, generator=torch.Generator().manual_seed(3603))


def load_data():
    raw = FashionMNIST(ROOT / "data", train=True, download=False)
    base = TensorDataset(raw.data.unsqueeze(1).float()/255., raw.targets)
    generator = torch.Generator().manual_seed(3601)
    train_ids, validation_ids = [], []
    for class_id in range(10):
        ids = torch.where(raw.targets == class_id)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        validation_ids.append(ids[:600]); train_ids.append(ids[600:])
    train_ids, validation_ids = torch.cat(train_ids), torch.cat(validation_ids)
    hashes = {name: hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()
              for name,ids in (("train",train_ids),("validation",validation_ids))}
    return Subset(base,train_ids), Subset(base,validation_ids), hashes


def build_model(hidden_sizes=(256,), activation="relu", initialization="normal", seed=4300, dtype=torch.float32):
    activation_classes = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
    if activation not in activation_classes: raise ValueError("Unknown activation")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dimensions = [784] + list(hidden_sizes) + [10]
        layers = [nn.Flatten()]
        for index,(inputs,outputs) in enumerate(zip(dimensions,dimensions[1:])):
            layers.append(nn.Linear(inputs,outputs))
            if index < len(dimensions)-2:
                layers.append(activation_classes[activation]())
        net = nn.Sequential(*layers).to(dtype=dtype)
        linears = [module for module in net if isinstance(module,nn.Linear)]
        last_linear = linears[-1]

        def init_weights(module):
            if not isinstance(module,nn.Linear): return
            nonlinearity = activation if module is not last_linear else "linear"
            if initialization == "normal":
                nn.init.normal_(module.weight,mean=0.,std=.01)
            elif initialization == "xavier":
                nn.init.xavier_normal_(module.weight,gain=nn.init.calculate_gain(nonlinearity))
            elif initialization == "kaiming":
                if nonlinearity not in ("relu","linear"):
                    raise ValueError("This Kaiming comparison only covers ReLU and linear output")
                nn.init.kaiming_normal_(module.weight,mode="fan_in",nonlinearity=nonlinearity)
            elif initialization == "zeros":
                nn.init.zeros_(module.weight)
            else:
                raise ValueError("Unknown initialization")
            nn.init.zeros_(module.bias)

        net.apply(init_weights)
    return net


def parameter_digest(net):
    digest = hashlib.sha256()
    for parameter in net.parameters(): digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def initialization_stats(net,mode):
    rows=[]
    linears=[module for module in net if isinstance(module,nn.Linear)]
    for index,module in enumerate(linears):
        gain = 2.**.5 if index < len(linears)-1 else 1.
        if mode=="normal": target=.01
        elif mode=="zeros": target=0.
        elif mode=="xavier": target=gain*(2/(module.in_features+module.out_features))**.5
        else: target=gain/module.in_features**.5
        rows.append({"layer":index+1,"weight_shape":list(module.weight.shape),
                     "measured_std":module.weight.detach().std(unbiased=False).item(),"target_std":target})
    return rows


def train_epoch(net,loader,loss_fn,optimizer):
    net.train()
    total_loss,correct,samples=0.,0,0
    for images,labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits=net(images)
        losses=loss_fn(logits,labels)
        losses.mean().backward()
        optimizer.step()
        total_loss+=losses.detach().sum().item()
        correct+=(logits.detach().argmax(1)==labels).sum().item()
        samples+=len(labels)
    return {"online_train_ce":total_loss/samples,"online_train_accuracy":correct/samples}


def evaluate(net,loader):
    net.eval()
    loss_fn=nn.CrossEntropyLoss(reduction="none")
    total_loss,correct,samples=0.,0,0
    with torch.no_grad():
        for images,labels in loader:
            logits=net(images)
            total_loss+=loss_fn(logits,labels).sum().item()
            correct+=(logits.argmax(1)==labels).sum().item();samples+=len(labels)
    return {"ce":total_loss/samples,"accuracy":correct/samples,"correct":int(correct),"samples":samples}


def fit(net,train,validation,epochs=10,lr=.1,batch_size=256,verbose=False):
    train_loader=make_loader(train,batch_size,True)
    train_eval=make_loader(train,batch_size);validation_loader=make_loader(validation,batch_size)
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    loss_fn=nn.CrossEntropyLoss(reduction="none")
    history=[]
    for epoch in range(epochs+1):
        online=train_epoch(net,train_loader,loss_fn,optimizer) if epoch else {
            "online_train_ce":None,"online_train_accuracy":None}
        row={"epoch":epoch,"updates":epoch*len(train_loader),**online}
        for name,loader in (("train",train_eval),("validation",validation_loader)):
            result=evaluate(net,loader)
            row[name+"_ce"]=result["ce"];row[name+"_accuracy"]=result["accuracy"]
        history.append(row)
        if verbose: print(row,flush=True)
    return history


def balanced_subset(dataset,per_class,seed):
    labels=dataset.dataset.tensors[1][dataset.indices]
    generator=torch.Generator().manual_seed(seed)
    parts=[]
    for class_id in range(10):
        ids=torch.where(labels==class_id)[0]
        parts.append(ids[torch.randperm(len(ids),generator=generator)[:per_class]])
    ids=torch.cat(parts)
    return Subset(dataset,ids),hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()


def experiments(train,validation):
    small_train,th=balanced_subset(train,200,4301)
    small_val,vh=balanced_subset(validation,100,4302)
    cases=[("A",(256,),.1,"relu","normal"),
           ("B",(256,64),.1,"relu","normal"),
           ("C",(256,128,64),.1,"relu","normal"),
           ("D",(256,64),.3,"relu","normal"),
           ("E",(256,),.1,"sigmoid","normal"),
           ("F",(256,),.1,"tanh","normal"),
           ("G",(256,),.1,"relu","xavier"),
           ("H",(256,),.1,"relu","kaiming"),
           ("I",(256,),.1,"relu","zeros")]
    summaries,all_rows,initial_hashes,scales=[],[],{},{}
    for case,hidden,lr,activation,initialization in cases:
        net=build_model(hidden,activation,initialization)
        initial_hashes[case]=parameter_digest(net)
        if case in ("A","G","H","I"):scales[case]=initialization_stats(net,initialization)
        history=fit(net,small_train,small_val,10,lr)
        for row in history:
            all_rows.append({"case":case,"hidden_sizes":"-".join(map(str,hidden)),
                             "activation":activation,"initialization":initialization,"lr":lr,**row})
        summary={"case":case,"hidden_sizes":list(hidden),"activation":activation,"initialization":initialization,
                 "lr":lr,"parameters":sum(p.numel() for p in net.parameters()),**history[-1]}
        if case=="I":
            summary["zero_diagnostics"]={"first_weight_max_abs":net[1].weight.detach().abs().max().item(),
                "first_bias_max_abs":net[1].bias.detach().abs().max().item(),
                "output_weight_max_abs":net[3].weight.detach().abs().max().item(),
                "output_bias_norm":net[3].bias.detach().norm().item()}
        summaries.append(summary)
        print("子集对照:",summary,flush=True)
    assert initial_hashes['A']==initial_hashes['E']==initial_hashes['F']
    best=min(summaries,key=lambda row:(row['validation_ce'],-row['validation_accuracy'],row['parameters']))
    return {"train_samples":2000,"validation_samples":1000,"per_class":[200,100],
            "subset_seeds":[4301,4302],"subset_index_sha256":{"train":th,"validation":vh},
            "model_seed":4300,"loader_seed":3603,"epochs":10,"batch_size":256,"updates_per_case":80,
            "test_evaluations":0,"initial_parameter_sha256":initial_hashes,
            "activation_comparison_identical_initial_parameters":True,"initialization_scales":scales,
            "summaries":summaries,"selection_metric":"minimum validation CE, then accuracy, then parameters",
            "best":best,"scope":"fixed small subset and nine candidates only"},all_rows


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def draw_figures(history,study):
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei"],"font.size":11,
                         "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
    teal,orange="#147E83","#D47F37"
    fig,ax=plt.subplots(figsize=(10,3.2));ax.set(xlim=(0,10),ylim=(0,3.2));ax.axis('off')
    nodes=[(.1,'Flatten','(n,784)'),(2.6,'Linear(784,256)','(n,256)'),
           (5.1,'ReLU','(n,256)'),(7.6,'Linear(256,10)','logits (n,10)')]
    for x,title,shape in nodes:
        ax.add_patch(FancyBboxPatch((x,1.5),2.15,1.05,boxstyle='round,pad=.04',facecolor='#F1F7F8',edgecolor=teal))
        ax.text(x+1.075,2.22,title,ha='center',fontsize=11,weight='bold')
        ax.text(x+1.075,1.82,shape,ha='center',fontsize=10)
        if x<7:ax.annotate('',xy=(x+2.45,2.02),xytext=(x+2.19,2.02),arrowprops={'arrowstyle':'->','color':teal})
    ax.text(5,2.91,'Sequential 按顺序执行；Linear 自动登记 weight 和 bias',ha='center',fontsize=11)
    ax.text(5,.85,'net.parameters() → SGD；原始得分 → CrossEntropyLoss',ha='center',color=teal)
    ax.text(5,.3,'Flatten 和 ReLU 无可训练参数；总参数仍为 203530',ha='center',fontsize=11)
    fig.tight_layout();fig.savefig(FIGURES/'4_3_components.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for name,label,color in (('train','训练集重算',teal),('validation','验证集',orange)):
        for ax,key in zip(axes,('ce','accuracy')):
            ax.plot([r['epoch'] for r in history],[r[name+'_'+key] for r in history],'o-',label=label,color=color)
    axes[0].set(xlabel='epoch',ylabel='平均交叉熵',title='轮末固定参数重新评估')
    axes[1].set(xlabel='epoch',ylabel='准确率',ylim=(0,1),title='54,000训练 / 6,000验证')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(FIGURES/'4_3_training.png',dpi=170);plt.close(fig)
    if study is None:return
    lookup={r['case']:r for r in study['summaries']}
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for ax,ids,title in zip(axes,('ABCD','AEF','AGHI'),('层数与学习率','激活函数','初始化方式')):
        values=[lookup[k]['validation_ce'] for k in ids]
        bars=ax.bar(range(len(ids)),values,color=teal)
        for bar,key in zip(bars,ids):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.06,f"{lookup[key]['validation_accuracy']:.1%}",ha='center',fontsize=9)
        ax.set(xticks=range(len(ids)),xticklabels=list(ids),ylabel='验证平均交叉熵',title=title,ylim=(0,2.8))
        ax.grid(axis='y',alpha=.2)
    fig.text(.5,.015,'方案编号对应讲义表；柱高为CE，柱上数字为准确率。固定2000训练/1000验证、十轮，不使用测试集。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(FIGURES/'4_3_experiments.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    labels=['A: .01正态','G: Xavier','H: Kaiming','I: 全零']
    for layer,ax in enumerate(axes):
        rows=[study['initialization_scales'][key][layer] for key in 'AGHI']
        ax.bar([i-.18 for i in range(4)],[r['target_std'] for r in rows],width=.35,label='理论目标',color=teal)
        ax.bar([i+.18 for i in range(4)],[r['measured_std'] for r in rows],width=.35,label='实测标准差',color=orange)
        ax.set_xticks(range(4),labels,rotation=12);ax.set(ylabel='weight标准差',title='第一隐藏层' if layer==0 else '输出层：线性gain=1')
        ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(FIGURES/'4_3_initialization.png',dpi=170);plt.close(fig)


def main(epochs=10,lr=.1,skip_experiments=False):
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    train,validation,hashes=load_data()
    net=build_model()
    initial_hash=parameter_digest(net)
    start=time.perf_counter();history=fit(net,train,validation,epochs,lr,verbose=True)
    training_seconds=time.perf_counter()-start
    raw_test=FashionMNIST(ROOT/'data',train=False,download=False)
    test_set=TensorDataset(raw_test.data.unsqueeze(1).float()/255.,raw_test.targets)
    test=evaluate(net,make_loader(test_set))
    artifact=ROOT/'artifacts/ch04/4_3_mlp_concise.pt';artifact.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':net.state_dict(),'hidden_sizes':[256],'activation':'relu',
                'initialization':'normal','epochs':epochs,'lr':lr,'model_seed':4300,'split_seed':3601},artifact)
    study,study_rows=(None,None) if skip_experiments else experiments(train,validation)
    draw_figures(history,study)
    metrics={'date':'2026-10-27','purpose':'teacher demonstration, not learner completion',
             'torch':torch.__version__,'device':'cpu','dtype':'float32','download':False,
             'split_sizes':{'train':len(train),'validation':len(validation),'test':len(test_set)},
             'split_seed':3601,'loader_seed':3603,'model_seed':4300,'split_index_sha256':hashes,
             'hidden_sizes':[256],'activation':'relu','initialization':'normal std=.01; explicit zero bias',
             'bias_vs_textbook':'textbook reinitializes weights only; this project zeros bias for control',
             'initial_parameters_sha256':initial_hash,'epochs':epochs,'learning_rate':lr,'batch_size':256,
             'updates':history[-1]['updates'],'training_tail':240,
             'named_parameter_shapes':{name:list(p.shape) for name,p in net.named_parameters()},
             'parameter_count':sum(p.numel() for p in net.parameters()),'final':history[-1],
             'test':test,'test_evaluations':1,'training_and_epoch_eval_seconds':training_seconds,
             'experiments':study,'checkpoint_local_only':artifact.relative_to(ROOT).as_posix(),
             'checkpoint_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
    with (REPORT/'4_3_metrics.json').open('w',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(REPORT/'4_3_epochs.csv',history)
    if study_rows is not None:write_csv(REPORT/'4_3_experiments.csv',study_rows)
    print('正式最终测试:',test,'| 正式训练与逐轮评估秒:',training_seconds)
    if study is not None:print('本次九候选最低验证CE:',study['best'])
    print('检查点与原始数据仅本地；教师完成不记为用户掌握。')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--lr',type=float,default=.1)
    parser.add_argument('--skip-experiments',action='store_true')
    args=parser.parse_args()
    if args.epochs<=0 or args.lr<=0:parser.error('epochs、lr必须为正')
    main(args.epochs,args.lr,args.skip_experiments)
