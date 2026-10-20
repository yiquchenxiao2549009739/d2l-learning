# -*- coding: utf-8 -*-
"""3.6：真实 Fashion-MNIST 上从零实现 softmax 分类器。

核心只用张量、自动求导和手写 SGD；框架损失仅用于独立核对。
已有数据，CPU/float32，无下载。54,000 训练 / 6,000 验证 / 10,000 测试。
先读 linear_logits、softmax、cross_entropy_logits、sgd 和 train_epoch。
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
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST

REPORT = ROOT / "reports/ch03"
FIGURES = REPORT / "figures"
NAMES = ["T恤/上衣", "长裤", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]


def stratified_indices(labels):
    generator = torch.Generator().manual_seed(3601)
    train_indices, validation_indices = [], []
    for class_id in range(10):
        ids = torch.where(labels == class_id)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        validation_indices.append(ids[:600])
        train_indices.append(ids[600:])
    return torch.cat(train_indices), torch.cat(validation_indices)


def make_loader(dataset, batch_size=256, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=False, generator=torch.Generator().manual_seed(3603))


def load_data(batch_size=256):
    raw_train = FashionMNIST(ROOT / "data", train=True, download=False)
    raw_test = FashionMNIST(ROOT / "data", train=False, download=False)
    # 3.5 已验证该固定向量化变换与逐张 ToTensor 相同，不改写 uint8 原始数据。
    features = raw_train.data.unsqueeze(1).float().div_(255.)
    test_features = raw_test.data.unsqueeze(1).float().div_(255.)
    base = TensorDataset(features, raw_train.targets)
    train_ids, validation_ids = stratified_indices(raw_train.targets)
    train = Subset(base, train_ids)
    validation = Subset(base, validation_ids)
    test = TensorDataset(test_features, raw_test.targets)
    splits = {"train": train, "validation": validation, "test": test}
    loaders = {name: make_loader(dataset, batch_size, shuffle=(name == "train"))
               for name, dataset in splits.items()}
    hashes = {name: hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()
              for name, ids in (("train", train_ids), ("validation", validation_ids))}
    return splits, loaders, hashes


def initialize_parameters():
    generator = torch.Generator().manual_seed(3602)
    weight = (.01 * torch.randn(784, 10, generator=generator)).requires_grad_()
    bias = torch.zeros(10, requires_grad=True)
    return weight, bias


def linear_logits(images, weight, bias):
    return images.reshape(len(images), -1) @ weight + bias


def softmax(logits):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    numerator = shifted.exp()
    return numerator / numerator.sum(dim=1, keepdim=True)


def log_softmax(logits):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    return shifted - shifted.exp().sum(dim=1, keepdim=True).log()


def net(images, weight, bias):
    return softmax(linear_logits(images, weight, bias))


def cross_entropy_probabilities(probabilities, labels):
    """教材直观形式；概率已下溢成 0 时会有 log(0)。"""
    return -probabilities[torch.arange(len(labels)), labels].log()


def cross_entropy_logits(logits, labels):
    return -log_softmax(logits)[torch.arange(len(labels)), labels]


def accuracy(scores, labels):
    """返回正确数量；除以总样本数后才是准确率。"""
    return int((scores.argmax(dim=1) == labels).sum().item())


class Accumulator:
    def __init__(self, count):
        self.data = [0.] * count

    def add(self, *values):
        assert len(values) == len(self.data)
        self.data = [a + float(b) for a, b in zip(self.data, values)]

    def reset(self):
        self.data = [0.] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def sgd(parameters, lr, batch_size):
    with torch.no_grad():
        for parameter in parameters:
            assert parameter.grad is not None
            parameter -= lr * parameter.grad / batch_size
            parameter.grad.zero_()


def train_epoch(loader, weight, bias, lr):
    metric = Accumulator(3)
    updates = 0
    for images, labels in loader:
        logits = linear_logits(images, weight, bias)
        losses = cross_entropy_logits(logits, labels)
        losses.sum().backward()
        sgd((weight, bias), lr, len(labels))
        metric.add(losses.detach().sum().item(), accuracy(logits.detach(), labels), len(labels))
        updates += 1
    return {"online_train_ce": metric[0] / metric[2],
            "online_train_accuracy": metric[1] / metric[2], "updates": updates}


def evaluate(loader, weight, bias, keep_predictions=False):
    metric = Accumulator(3)
    all_predictions, all_labels, all_confidence = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = linear_logits(images, weight, bias)
            losses = cross_entropy_logits(logits, labels)
            metric.add(losses.sum().item(), accuracy(logits, labels), len(labels))
            if keep_predictions:
                probabilities = softmax(logits)
                confidence, prediction = probabilities.max(dim=1)
                all_predictions.extend(prediction.tolist())
                all_labels.extend(labels.tolist())
                all_confidence.extend(confidence.tolist())
    result = {"ce": metric[0] / metric[2], "accuracy": metric[1] / metric[2],
              "correct": int(metric[1]), "samples": int(metric[2])}
    if keep_predictions:
        return result, all_predictions, all_labels, all_confidence
    return result


def audit_core():
    logits = torch.tensor([[2., 1., 0.], [0., -1., 2.]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 2])
    manual = cross_entropy_logits(logits, labels)
    reference = F.cross_entropy(logits, labels, reduction="none")
    assert torch.allclose(manual, reference, atol=1e-14)
    manual.mean().backward()
    expected = (softmax(logits.detach()) - F.one_hot(labels, 3)) / 2
    assert torch.allclose(logits.grad, expected, atol=1e-14)
    moderate = torch.tensor([[.1, .3, .6], [.3, .2, .5]], dtype=torch.float64)
    assert accuracy(moderate, labels) == 1
    extreme = torch.tensor([[1000., 0., -1000.]], dtype=torch.float64)
    p = softmax(extreme)
    direct = cross_entropy_probabilities(p, torch.tensor([2]))
    stable = cross_entropy_logits(extreme, torch.tensor([2]))
    assert not torch.isfinite(direct).all() and stable.item() == 2000.
    # 尾批加权：一批 3 条全对，另一批 1 条全错，总准确率为 3/4。
    weighted_accuracy = (3 + 0) / (3 + 1)
    unweighted_accuracy = ((3 / 3) + (0 / 1)) / 2
    costs = torch.tensor([[0., 1.], [10., 0.]], dtype=torch.float64)
    posterior = torch.tensor([.6, .4], dtype=torch.float64)
    risks = posterior @ costs
    return {"book_example_losses": cross_entropy_probabilities(moderate, labels).tolist(),
            "book_example_correct_count": 1, "book_example_accuracy": .5,
            "gradient_matches_p_minus_y_over_batch": True,
            "mean_ce": manual.mean().item(), "stable_extreme_ce": stable.item(),
            "direct_log_of_zero_is_nonfinite": True,
            "exp50_float32": torch.exp(torch.tensor(50.)).item(),
            "exp50_float32_finite": bool(torch.isfinite(torch.exp(torch.tensor(50.)))),
            "exp50_float16_finite": bool(torch.isfinite(torch.exp(torch.tensor(50., dtype=torch.float16)))),
            "exp100_float32_finite": bool(torch.isfinite(torch.exp(torch.tensor(100.)))),
            "weighted_accuracy": weighted_accuracy, "wrong_batch_average_accuracy": unweighted_accuracy,
            "cost_matrix_true_rows_action_columns": costs.tolist(), "posterior": posterior.tolist(),
            "action_risks": risks.tolist(), "argmax_class": int(posterior.argmax()),
            "minimum_risk_action": int(risks.argmin()),
            "large_vocabulary_example": {"classes": 100000, "weight_float32_mib": 784*100000*4/2**20,
                                          "batch256_logits_float32_mib": 256*100000*4/2**20}}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def draw_figures(history, test_set, predictions, labels, confidence, weight):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = ["#147E83", "#D47F37"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, title, color in (("train", "训练集重算", colors[0]), ("validation", "验证集", colors[1])):
        axes[0].plot([r["epoch"] for r in history], [r[name+"_ce"] for r in history], "o-", label=title, color=color)
        axes[1].plot([r["epoch"] for r in history], [r[name+"_accuracy"] for r in history], "o-", label=title, color=color)
    axes[0].set(xlabel="epoch", ylabel="平均交叉熵", title="每轮后用当前参数重新评估")
    axes[1].set(xlabel="epoch", ylabel="准确率", ylim=(0, 1), title="训练与验证；测试只在最后评估")
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES/"3_6_training.png", dpi=170); plt.close(fig)

    selected = ([i for i,(a,b) in enumerate(zip(labels,predictions)) if a==b][:6] +
                [i for i,(a,b) in enumerate(zip(labels,predictions)) if a!=b][:6])
    fig, axes = plt.subplots(2, 6, figsize=(12, 5.3))
    for ax, index in zip(axes.ravel(), selected):
        image, true = test_set[index]
        predicted = predictions[index]
        ax.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"真：{NAMES[true]}\n预测：{NAMES[predicted]}\np={confidence[index]:.2f}",
                     color=colors[0] if predicted==true else "#B34C46", fontsize=10)
        ax.axis("off")
    fig.text(.5,.015,"上排：测试集中前六个正确例；下排：前六个错误例。图像 © 2017 Zalando SE · MIT。",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(FIGURES/"3_6_predictions.png",dpi=170,bbox_inches="tight",pad_inches=.12); plt.close(fig)

    matrix = np.zeros((10,10), dtype=int)
    for true,predicted in zip(labels,predictions): matrix[true,predicted]+=1
    fig,ax=plt.subplots(figsize=(5.8,5.4))
    ax.imshow(matrix,cmap="Blues",vmin=0,vmax=1000)
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=8,
                    color="white" if matrix[i,j]>500 else "#173448")
    ax.set(xticks=range(10),yticks=range(10),xlabel="预测类别编号",ylabel="真实类别编号",title="全部 10,000 张测试图像的计数")
    fig.tight_layout();fig.savefig(FIGURES/"3_6_confusion.png",dpi=170);plt.close(fig)

    weights=weight.detach().numpy().reshape(28,28,10)
    limit=np.abs(weights).max()
    fig,axes=plt.subplots(2,5,figsize=(10,4.7))
    for j,ax in enumerate(axes.ravel()):
        ax.imshow(weights[:,:,j],cmap="RdBu_r",vmin=-limit,vmax=limit,interpolation="nearest")
        ax.set_title(f"{j}: {NAMES[j]}");ax.axis("off")
    fig.text(.5,.015,"红为正权重，蓝为负权重；共同色标。预测取决于各类别完整得分的比较。",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(FIGURES/"3_6_weights.png",dpi=170);plt.close(fig)
    return selected, matrix.tolist()


def main(epochs=10, lr=.1, batch_size=256):
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    splits,loaders,hashes=load_data(batch_size)
    train_eval_loader=make_loader(splits["train"],batch_size)
    weight,bias=initialize_parameters()
    history=[]
    start=time.perf_counter()
    def record(epoch,online=None):
        row={"epoch":epoch,"updates":epoch*len(loaders["train"])}
        if online is not None: row.update({k:v for k,v in online.items() if k!="updates"})
        else: row.update({"online_train_ce":None,"online_train_accuracy":None})
        for name,loader in (("train",train_eval_loader),("validation",loaders["validation"])):
            result=evaluate(loader,weight,bias)
            row[name+"_ce"]=result["ce"];row[name+"_accuracy"]=result["accuracy"]
        history.append(row)
        print(row,flush=True)
    record(0)
    for epoch in range(epochs):
        online=train_epoch(loaders["train"],weight,bias,lr)
        record(epoch+1,online)
    training_seconds=time.perf_counter()-start
    test,predictions,labels,confidence=evaluate(loaders["test"],weight,bias,True)
    selected,matrix=draw_figures(history,splits["test"],predictions,labels,confidence,weight)
    audit=audit_core()
    artifact=ROOT/"artifacts/ch03/3_6_softmax_scratch.pt"
    artifact.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"weight":weight.detach(),"bias":bias.detach(),"epochs":epochs,"lr":lr,
                "batch_size":batch_size,"split_seed":3601,"init_seed":3602,"loader_seed":3603},artifact)
    metrics={"purpose":"teacher demonstration, not learner completion","date":"2026-10-20",
             "torch":torch.__version__,"device":"cpu","dtype":"float32","download":False,
             "split_sizes":{k:len(v) for k,v in splits.items()},"split_seed":3601,"init_seed":3602,"loader_seed":3603,
             "split_index_sha256":hashes,"validation_per_class":600,"training_per_class":5400,
             "batch_size":batch_size,"epochs":epochs,"learning_rate":lr,
             "updates":epochs*len(loaders["train"]),"training_tail":len(splits["train"])%batch_size or batch_size,
             "weight_shape":list(weight.shape),"bias_shape":list(bias.shape),"parameter_count":weight.numel()+bias.numel(),
             "train_ce":history[-1]["train_ce"],"train_accuracy":history[-1]["train_accuracy"],
             "validation_ce":history[-1]["validation_ce"],"validation_accuracy":history[-1]["validation_accuracy"],
             "test":test,"test_evaluations":1,"training_and_epoch_eval_seconds":training_seconds,
             "weight_norm":weight.detach().norm().item(),"bias_norm":bias.detach().norm().item(),
             "prediction_example_indices":selected,"confusion_true_rows":matrix,"audit":audit,
             "checkpoint_local_only":"artifacts/ch03/3_6_softmax_scratch.pt",
             "checkpoint_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest()}
    with (REPORT/"3_6_metrics.json").open("w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    write_csv(REPORT/"3_6_epochs.csv",history)
    write_csv(REPORT/"3_6_predictions.csv",[{"test_index":i,"true":labels[i],"predicted":predictions[i],
               "confidence":confidence[i]} for i in selected])
    print("最终测试:",test,"| 更新次数:",metrics["updates"],"| 训练和逐轮评估秒:",training_seconds)
    print("数值与决策核对:",audit)
    print("参数只保存本地 artifacts；报告、四张图供学习复盘。")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--lr",type=float,default=.1)
    parser.add_argument("--batch-size",type=int,default=256)
    args=parser.parse_args()
    if args.epochs<=0 or args.lr<=0 or args.batch_size<=0:
        parser.error("epochs、lr、batch-size 必须为正")
    main(args.epochs,args.lr,args.batch_size)
