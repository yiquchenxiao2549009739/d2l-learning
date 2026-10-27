# -*- coding: utf-8 -*-
"""4.2：Parameter + 手写 ReLU + 张量前向，在已有 Fashion-MNIST 上训练。

默认784→256→10，CPU/float32、十轮。模型不用Linear/Sequential，自动求导、
稳定交叉熵和SGD复用框架。另用固定小子集完成六道教材调参练习，无下载/安装。
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
NAMES = ["T恤/上衣", "长裤", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]


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
              for name, ids in (("train", train_ids), ("validation", validation_ids))}
    return Subset(base, train_ids), Subset(base, validation_ids), hashes


def initialize_parameters(hidden_sizes=(256,), seed=4200, dtype=torch.float32):
    dimensions = [784] + list(hidden_sizes) + [10]
    generator = torch.Generator().manual_seed(seed)
    parameters = []
    for inputs, outputs in zip(dimensions, dimensions[1:]):
        weight = nn.Parameter(.01 * torch.randn(inputs, outputs, generator=generator, dtype=dtype))
        bias = nn.Parameter(torch.zeros(outputs, dtype=dtype))
        parameters.extend([weight, bias])
    return parameters


def relu(X):
    # 教材的逐元素max：在恰好等于0处梯度为.5，与torch.relu的0不同。
    return torch.max(X, torch.zeros_like(X))


def net(images, parameters):
    H = images.reshape(len(images), -1)
    for index in range(0, len(parameters), 2):
        H = H @ parameters[index] + parameters[index+1]
        if index < len(parameters)-2:
            H = relu(H)
    return H  # 原始logits；输出层不再加ReLU或softmax。


def train_epoch(loader, parameters, loss_fn, optimizer, record_probe=False):
    loss_total, correct, samples = 0., 0, 0
    probe = None
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = net(images, parameters)
        losses = loss_fn(logits, labels)
        losses.mean().backward()
        if record_probe and probe is None:
            gradient_norms = []
            for parameter in parameters:
                assert parameter.grad is not None
                gradient_norms.append(parameter.grad.norm().item())
            probe = {"batch_size": len(labels),
                     "parameter_gradient_norms": gradient_norms}
        optimizer.step()
        loss_total += losses.detach().sum().item()
        correct += (logits.detach().argmax(1) == labels).sum().item()
        samples += len(labels)
    return {"online_train_ce": loss_total/samples, "online_train_accuracy": correct/samples}, probe


def evaluate(loader, parameters, keep_predictions=False):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss_total, correct, samples = 0., 0, 0
    predictions, labels_all, confidence = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = net(images, parameters)
            loss_total += loss_fn(logits, labels).sum().item()
            correct += (logits.argmax(1) == labels).sum().item()
            samples += len(labels)
            if keep_predictions:
                values, indices = logits.softmax(1).max(1)
                predictions.extend(indices.tolist()); labels_all.extend(labels.tolist())
                confidence.extend(values.tolist())
    result = {"ce":loss_total/samples, "accuracy":correct/samples, "correct":int(correct), "samples":samples}
    return (result, predictions, labels_all, confidence) if keep_predictions else result


def fit(train_set, validation_set, hidden_sizes=(256,), lr=.1, epochs=10, batch_size=256, verbose=False):
    parameters = initialize_parameters(hidden_sizes)
    train_loader = make_loader(train_set, batch_size, True)
    train_eval = make_loader(train_set, batch_size)
    validation_loader = make_loader(validation_set, batch_size)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(parameters, lr=lr)
    history, first_probe = [], None
    for epoch in range(epochs+1):
        if epoch:
            online, probe = train_epoch(train_loader, parameters, loss_fn, optimizer, epoch==1)
            if probe is not None: first_probe = probe
        else:
            online = {"online_train_ce":None, "online_train_accuracy":None}
        row = {"epoch":epoch, "updates":epoch*len(train_loader), **online}
        for name,loader in (("train",train_eval), ("validation",validation_loader)):
            result = evaluate(loader, parameters)
            row[name+"_ce"] = result["ce"]; row[name+"_accuracy"] = result["accuracy"]
        history.append(row)
        if verbose: print(row, flush=True)
    return parameters, history, first_probe


def balanced_subset(dataset, per_class, seed):
    labels = dataset.dataset.tensors[1][dataset.indices]
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for class_id in range(10):
        ids = torch.where(labels==class_id)[0]
        indices.append(ids[torch.randperm(len(ids), generator=generator)[:per_class]])
    indices = torch.cat(indices)
    digest = hashlib.sha256(indices.numpy().astype("<i8").tobytes()).hexdigest()
    return Subset(dataset, indices), digest


def search_experiments(train_set, validation_set):
    train, train_hash = balanced_subset(train_set, 200, 4201)
    validation, validation_hash = balanced_subset(validation_set, 100, 4202)
    cases = [("A",(256,),.1), ("B",(64,),.1), ("C",(128,),.1), ("D",(512,),.1),
             ("E",(256,64),.1), ("F",(256,),.03), ("G",(256,),.3),
             ("H",(128,),.3), ("I",(128,64),.3)]
    summaries, all_rows, candidates = [], [], []
    for case,hidden,lr in cases:
        parameters, history, _ = fit(train,validation,hidden,lr,10,256)
        count = sum(p.numel() for p in parameters)
        for row in history:
            all_rows.append({"case":case,"hidden_sizes":"-".join(map(str,hidden)),"lr":lr,**row})
        for epoch in (5,10):
            candidates.append({"case":case,"hidden_sizes":list(hidden),"lr":lr,"parameters":count,
                               **{k:v for k,v in history[epoch].items() if not k.startswith("online_")}})
        summaries.append(candidates[-1])
        print("子集调参:",candidates[-1],flush=True)
    best = min(candidates, key=lambda row:(row["validation_ce"],-row["validation_accuracy"],row["parameters"]))
    return {"train_samples":2000,"validation_samples":1000,"batch_size":256,
            "subset_seeds":[4201,4202],"subset_index_sha256":{"train":train_hash,"validation":validation_hash},
            "model_seed":4200,"loader_seed":3603,"test_evaluations":0,
            "cases":len(cases),"candidate_checkpoints":len(candidates),
            "selection_metric":"minimum validation CE, then accuracy, then parameter count",
            "scope":"best only among these checkpoints on this fixed small subset",
            "summaries_epoch10":summaries,"best":best},all_rows,candidates


def write_csv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def draw_figures(history, test_set, predictions, labels, confidence, hidden, search):
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei"],"font.size":11,
                         "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
    teal,orange="#147E83","#D47F37"
    fig,ax=plt.subplots(figsize=(10,3.3));ax.set(xlim=(0,10),ylim=(0,3.3));ax.axis("off")
    nodes=[(.1,"展平图片","(n,784)"),(2.6,"第一层 + ReLU",f"H (n,{hidden})"),
           (5.1,"第二层得分","logits (n,10)"),(7.6,"稳定交叉熵","losses (n,)")]
    for x,title,shape in nodes:
        ax.add_patch(FancyBboxPatch((x,1.55),2.15,1.05,boxstyle="round,pad=.04",facecolor="#F1F7F8",edgecolor=teal))
        ax.text(x+1.075,2.25,title,ha="center",weight="bold",fontsize=11)
        ax.text(x+1.075,1.85,shape,ha="center",fontsize=10)
        if x<7:ax.annotate("",xy=(x+2.45,2.05),xytext=(x+2.19,2.05),arrowprops={"arrowstyle":"->","color":teal})
    ax.text(2.5,2.95,f"W1 (784,{hidden})、b1 ({hidden},)",ha="center",fontsize=10)
    ax.text(6.1,2.95,f"W2 ({hidden},10)、b2 (10,)",ha="center",fontsize=10)
    ax.text(5,.9,"losses.mean().backward()：四组参数都得到梯度",ha="center",color=teal)
    ax.text(5,.32,"SGD(params, lr=.1)：显式传入参数；step 不再除批量",ha="center",color=teal)
    fig.tight_layout();fig.savefig(FIGURES/"4_2_forward.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for name,label,color in (("train","训练集重算",teal),("validation","验证集",orange)):
        axes[0].plot([r['epoch'] for r in history],[r[name+'_ce'] for r in history],"o-",label=label,color=color)
        axes[1].plot([r['epoch'] for r in history],[r[name+'_accuracy'] for r in history],"o-",label=label,color=color)
    axes[0].set(xlabel="epoch",ylabel="平均交叉熵",title="轮末固定参数重新评估")
    axes[1].set(xlabel="epoch",ylabel="准确率",ylim=(0,1),title="54,000训练 / 6,000验证")
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(FIGURES/"4_2_training.png",dpi=170);plt.close(fig)
    selected=([i for i,(a,b) in enumerate(zip(labels,predictions)) if a==b][:6]+
              [i for i,(a,b) in enumerate(zip(labels,predictions)) if a!=b][:6])
    fig,axes=plt.subplots(2,6,figsize=(12,6.3))
    for ax in axes.ravel():ax.axis("off")
    for ax,index in zip(axes.ravel(),selected):
        image,true=test_set[index];predicted=predictions[index]
        ax.imshow(image[0].numpy(),cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        ax.set_title(f"真：{NAMES[true]}\n预测：{NAMES[predicted]}\np={confidence[index]:.2f}",
                     color=teal if true==predicted else "#B34C46",fontsize=11)
    fig.text(.5,.015,"上排前六正确例，下排前六错误例；总体成绩使用全部测试图像。© 2017 Zalando SE · MIT。",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1),h_pad=2.5)
    fig.savefig(FIGURES/"4_2_predictions.png",dpi=170);plt.close(fig)
    if search is not None:
        rows=search['summaries_epoch10'];fig,axes=plt.subplots(1,2,figsize=(10,4.2))
        xs=list(range(len(rows)));labels_case=[r['case'] for r in rows]
        axes[0].bar(xs,[r['validation_ce'] for r in rows],color=teal)
        axes[1].bar(xs,[r['validation_accuracy'] for r in rows],color=orange)
        for ax in axes:ax.set_xticks(xs,labels_case);ax.set_xlabel("方案编号（对应讲义表）");ax.grid(axis='y',alpha=.2)
        axes[0].set(ylabel="验证平均交叉熵",title="固定2000训练 / 1000验证，十轮")
        axes[1].set(ylabel="验证准确率",ylim=(0,1),title="九方案；未访问官方测试集")
        fig.tight_layout();fig.savefig(FIGURES/"4_2_search.png",dpi=170);plt.close(fig)
    return selected


def main(epochs=10,lr=.1,hidden=256,skip_search=False):
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    train,validation,hashes=load_data()
    start=time.perf_counter()
    parameters,history,probe=fit(train,validation,(hidden,),lr,epochs,256,True)
    training_seconds=time.perf_counter()-start
    raw_test=FashionMNIST(ROOT/"data",train=False,download=False)
    test_set=TensorDataset(raw_test.data.unsqueeze(1).float()/255.,raw_test.targets)
    test,predictions,labels,confidence=evaluate(make_loader(test_set),parameters,True)
    checkpoint=ROOT/"artifacts/ch04/4_2_mlp_scratch.pt"
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"parameters":[p.detach().clone() for p in parameters],"hidden_sizes":[hidden],
                "epochs":epochs,"lr":lr,"model_seed":4200,"split_seed":3601},checkpoint)
    search,search_rows,candidates=(None,None,None) if skip_search else search_experiments(train,validation)
    selected=draw_figures(history,test_set,predictions,labels,confidence,hidden,search)
    audit={}
    for name,function in (("book_max",relu),("torch_relu",torch.relu)):
        x=torch.tensor([-1.,0.,1.],requires_grad=True)
        audit[name+"_gradient"]=torch.autograd.grad(function(x).sum(),x)[0].tolist()
    metrics={"date":"2026-10-27","purpose":"teacher demonstration, not learner completion",
             "torch":torch.__version__,"device":"cpu","dtype":"float32","download":False,
             "split_sizes":{"train":len(train),"validation":len(validation),"test":len(test_set)},
             "split_seed":3601,"loader_seed":3603,"model_seed":4200,"split_index_sha256":hashes,
             "hidden_sizes":[hidden],"parameter_shapes":[list(p.shape) for p in parameters],
             "parameter_count":sum(p.numel() for p in parameters),"all_parameters_leaf":all(p.is_leaf for p in parameters),
             "epochs":epochs,"learning_rate":lr,"batch_size":256,"updates":history[-1]['updates'],"training_tail":240,
             "loss_reduction":"none then mean backward","first_batch_probe":probe,"relu_audit":audit,
             "final":history[-1],"test":test,"test_evaluations":1,"training_and_epoch_eval_seconds":training_seconds,
             "search":search,"prediction_example_indices":selected,
             "checkpoint_local_only":checkpoint.relative_to(ROOT).as_posix(),
             "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    with (REPORT/"4_2_metrics.json").open('w',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(REPORT/"4_2_epochs.csv",history)
    write_csv(REPORT/"4_2_predictions.csv",[{"test_index":i,"true":labels[i],"predicted":predictions[i],"confidence":confidence[i]} for i in selected])
    if search_rows is not None:
        write_csv(REPORT/"4_2_search.csv",search_rows)
        write_csv(REPORT/"4_2_candidates.csv",[{**r,"hidden_sizes":"-".join(map(str,r['hidden_sizes']))} for r in candidates])
    print("正式最终测试:",test,"| 正式训练和逐轮评估秒:",training_seconds)
    print("手写ReLU与内置原点约定:",audit)
    if search is not None:print("有限子集候选中最低验证CE:",search['best'])
    print("数据和检查点只留本地；子集搜索不访问官方测试，不据教师运行标记用户掌握。")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr',type=float,default=.1)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--skip-search',action='store_true',help='跳过额外小子集搜索，正式训练照常运行')
    args=parser.parse_args()
    if args.epochs<=0 or args.lr<=0 or args.hidden<=0:parser.error('epochs、lr、hidden必须为正')
    main(args.epochs,args.lr,args.hidden,args.skip_search)
