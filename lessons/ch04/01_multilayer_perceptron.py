# -*- coding: utf-8 -*-
"""4.1 多层感知机：隐藏层、非线性、激活梯度与 XOR 综合实验。

CPU/float64，复用 openmmlab，无下载。4.2 再从零训练 Fashion-MNIST。
本节用小型二维分类对照，重点理解 Linear → 激活 → Linear。
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

DATA = ROOT / "datasets/ch04"
REPORT = ROOT / "reports/ch04"
FIGURES = REPORT / "figures"
DTYPE = torch.float64


def xor_points(groups, seed):
    """每个正幅值对配四种符号，确保类别平衡及镜像对称。"""
    generator = torch.Generator().manual_seed(seed)
    magnitudes = .2 + .8 * torch.rand(groups, 2, generator=generator, dtype=DTYPE)
    signs = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=DTYPE)
    X = (magnitudes[:, None, :] * signs[None, :, :]).reshape(-1, 2)
    y = (X[:, 0] * X[:, 1] < 0).long()
    return X, y


def prepare_data():
    DATA.mkdir(parents=True, exist_ok=True)
    sets, integrity = {}, []
    for name, groups, seed in (("train", 120, 4101), ("validation", 40, 4102), ("test", 40, 4103)):
        path = DATA / ("01_xor_" + name + ".csv")
        expected_X, expected_y = xor_points(groups, seed)
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["x1", "x2", "label"])
                for row, label in zip(expected_X.tolist(), expected_y.tolist()):
                    writer.writerow([format(row[0], ".17g"), format(row[1], ".17g"), label])
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        X = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=DTYPE)
        y = torch.tensor([int(r["label"]) for r in rows])
        if not torch.equal(X, expected_X) or not torch.equal(y, expected_y):
            raise ValueError("数据与预先固定生成规则不同，请先检查，不自动覆盖：" + str(path))
        sets[name] = (X, y)
        integrity.append({"file": path.relative_to(ROOT).as_posix(), "seed": seed,
                          "samples": len(y), "class_counts": torch.bincount(y).tolist(),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return sets, integrity


def build_models():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4100)
        relu_mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).double()
        for module in relu_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.4)
                nn.init.zeros_(module.bias)
        affine_stack = nn.Sequential(nn.Linear(2, 8), nn.Identity(), nn.Linear(8, 2)).double()
        affine_stack.load_state_dict(relu_mlp.state_dict())
        linear = nn.Linear(2, 2).double()
        nn.init.normal_(linear.weight, std=.4)
        nn.init.zeros_(linear.bias)
    return {"linear": linear, "affine_stack": affine_stack, "relu_mlp": relu_mlp}


def evaluate(net, X, y):
    net.eval()
    with torch.no_grad():
        logits = net(X)
        loss = nn.CrossEntropyLoss()(logits, y).item()
        correct = int((logits.argmax(dim=1) == y).sum())
    return {"ce": loss, "correct": correct, "samples": len(y), "accuracy": correct / len(y)}


def fit(net, train, validation, epochs=600, lr=.1):
    X, y = train
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs + 1):
        if epoch:
            net.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(X), y)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0 or epoch == epochs:
            row = {"epoch": epoch, "updates": epoch}
            for name, dataset in (("train", train), ("validation", validation)):
                result = evaluate(net, *dataset)
                row[name + "_ce"] = result["ce"]
                row[name + "_accuracy"] = result["accuracy"]
            history.append(row)
    return history


def constructed_xor_logits(X):
    # d = |x1-x2| - |x1+x2|：异号为正，同号为负，坐标轴为边界。
    W1 = torch.tensor([[1., -1., 1., -1.], [-1., 1., 1., -1.]], dtype=X.dtype)
    W2 = torch.tensor([[-1., 1.], [-1., 1.], [1., -1.], [1., -1.]], dtype=X.dtype)
    return torch.relu(X @ W1) @ W2


def theory_audit():
    generator = torch.Generator().manual_seed(4110)
    X, W1, W2 = [torch.randn(*shape, generator=generator, dtype=DTYPE)
                  for shape in ((7, 4), (4, 5), (5, 3))]
    b1 = torch.randn(5, generator=generator, dtype=DTYPE)
    b2 = torch.randn(3, generator=generator, dtype=DTYPE)
    composed = (X @ W1 + b1) @ W2 + b2
    merged = X @ (W1 @ W2) + b1 @ W2 + b2
    merge_error = (composed - merged).abs().max().item()
    assert merge_error < 1e-12
    points = torch.tensor([-8., -1., 0., 1., 8.], dtype=DTYPE)
    activations = {}
    for name, function in (("relu", torch.relu), ("sigmoid", torch.sigmoid), ("tanh", torch.tanh)):
        x = points.clone().requires_grad_()
        values = function(x)
        derivative, = torch.autograd.grad(values.sum(), x)
        expected = (x > 0).double() if name == "relu" else (
            values * (1-values) if name == "sigmoid" else 1-values.square())
        assert torch.allclose(derivative, expected, atol=1e-15)
        activations[name] = {"x": points.tolist(), "values": values.detach().tolist(), "dx": derivative.tolist()}
    x = torch.tensor([-1., 0., 1.], dtype=DTYPE, requires_grad=True)
    alpha = torch.tensor([.1], dtype=DTYPE, requires_grad=True)
    out = torch.prelu(x, alpha)
    dx, da = torch.autograd.grad(out.sum(), (x, alpha))
    identity_error = (torch.tanh(points)+1-2*torch.sigmoid(2*points)).abs().max().item()
    batch = torch.tensor([[1., 2.], [3., 4.]], dtype=DTYPE)
    changed = batch.clone(); changed[1] = torch.tensor([30., 40.], dtype=DTYPE)
    batch_audit = {"relu_first_unchanged": bool(torch.equal(batch.relu()[0], changed.relu()[0])),
                   "class_softmax_first_unchanged": bool(torch.equal(batch.softmax(1)[0], changed.softmax(1)[0])),
                   "wrong_dim0_before": batch.softmax(0)[0].tolist(),
                   "wrong_dim0_after": changed.softmax(0)[0].tolist()}
    assert batch_audit["relu_first_unchanged"] and batch_audit["class_softmax_first_unchanged"]
    shape_net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    with torch.no_grad():
        shape_output = shape_net(torch.zeros(3, 1, 28, 28))
    return {"affine_merge_max_error": merge_error, "activation_samples": activations,
            "prelu": {"alpha": .1, "x": x.detach().tolist(), "dx": dx.tolist(), "d_alpha": da.item(),
                      "at_zero": "classically not differentiable when alpha != 1; torch chooses alpha"},
            "tanh_sigmoid_identity_max_error": identity_error, "batch_axis": batch_audit,
            "fashion_forward_only": {"input_shape": [3, 1, 28, 28], "output_shape": list(shape_output.shape),
                                     "parameter_count": sum(p.numel() for p in shape_net.parameters()),
                                     "training_performed": False}}


def draw_figures(models, histories):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    teal, orange, blue = "#147E83", "#D47F37", "#5A6FA8"
    fig, ax = plt.subplots(figsize=(9, 3.8))
    positions = [(0, [-.7, .7]), (2, [-1.2, -.4, .4, 1.2]), (4, [-.7, .7])]
    for (x1, ys1), (x2, ys2) in zip(positions, positions[1:]):
        for a in ys1:
            for b in ys2:
                ax.plot([x1, x2], [a, b], color="#BCCDD3", lw=.8, zorder=0)
    for (x, ys), color in zip(positions, (blue, teal, orange)):
        ax.scatter([x]*len(ys), ys, s=650, color=color, edgecolors="white", zorder=2)
    for x, label in ((0,"输入：2维"),(2,"隐藏表示：4维"),(4,"输出得分：2类")):
        ax.text(x,1.75,label,ha="center")
    ax.text(1,-1.65,"仿射变换 → ReLU",ha="center",color=teal)
    ax.text(3,-1.65,"仿射变换 → logits",ha="center",color=teal)
    ax.set(xlim=(-.6,4.6),ylim=(-2,2));ax.axis("off")
    fig.tight_layout();fig.savefig(FIGURES/"4_1_network.png",dpi=170);plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for j,(name,function) in enumerate((("ReLU",torch.relu),("sigmoid",torch.sigmoid),("tanh",torch.tanh))):
        x = torch.linspace(-8,8,801,dtype=DTYPE,requires_grad=True)
        y = function(x); grad, = torch.autograd.grad(y.sum(),x)
        axes[0,j].plot(x.detach().numpy(),y.detach().numpy(),color=teal)
        axes[1,j].plot(x.detach().numpy(),grad.numpy(),color=orange)
        axes[0,j].set_title(name);axes[0,j].set_ylabel("输出")
        axes[1,j].set(xlabel="输入 x",ylabel="局部导数")
        for ax in axes[:,j]: ax.grid(alpha=.2);ax.axvline(0,color="#BAC9CF",lw=.6)
    fig.tight_layout();fig.savefig(FIGURES/"4_1_activations.png",dpi=170);plt.close(fig)

    names={"linear":"单层线性，6参数","affine_stack":"无激活两层，42参数","relu_mlp":"ReLU 两层，42参数"}
    colors={"linear":blue,"affine_stack":orange,"relu_mlp":teal}
    fig, axes=plt.subplots(1,2,figsize=(10,3.8))
    for name,rows in histories.items():
        axes[0].plot([r["epoch"] for r in rows],[r["train_ce"] for r in rows],label=names[name],color=colors[name])
        axes[1].plot([r["epoch"] for r in rows],[r["validation_accuracy"] for r in rows],label=names[name],color=colors[name])
    axes[0].set(xlabel="整批更新轮数",ylabel="训练平均交叉熵")
    axes[1].set(xlabel="整批更新轮数",ylabel="验证准确率",ylim=(0,1.04))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(FIGURES/"4_1_training.png",dpi=170);plt.close(fig)

    axis=torch.linspace(-1.1,1.1,171,dtype=DTYPE)
    a,b=torch.meshgrid(axis,axis,indexing="xy")
    grid=torch.stack((a.reshape(-1),b.reshape(-1)),1)
    corners=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=DTYPE)
    labels=(corners[:,0]*corners[:,1]<0).long()
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    for ax,(name,net) in zip(axes.ravel(),list(models.items())+[("constructed",constructed_xor_logits)]):
        with torch.no_grad():pred=net(grid).argmax(1).reshape(171,171).numpy()
        ax.contourf(a.numpy(),b.numpy(),pred,levels=[-.5,.5,1.5],colors=["#DDECEF","#FAE5CE"],alpha=.95)
        for label,marker,color in ((0,"o",teal),(1,"^",orange)):
            points=corners[labels==label]
            ax.scatter(points[:,0],points[:,1],marker=marker,c=color,s=65,edgecolors="white",label="真类"+str(label))
        ax.set(xlabel="x1",ylabel="x2",title=names.get(name,"手工构造，4个 ReLU 单元"),aspect="equal")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(FIGURES/"4_1_regions.png",dpi=170);plt.close(fig)

    x=torch.linspace(-2,2,801,dtype=DTYPE,requires_grad=True)
    y=torch.relu(x+1)-2*torch.relu(x)+torch.relu(x-1)
    grad,=torch.autograd.grad(y.sum(),x)
    grad_plot=grad.numpy().copy()
    for knot in (-1.,0.,1.):grad_plot[np.isclose(x.detach().numpy(),knot)]=np.nan
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    axes[0].plot(x.detach().numpy(),y.detach().numpy(),color=teal)
    axes[1].plot(x.detach().numpy(),grad_plot,color=orange)
    axes[0].set(xlabel="x",ylabel="f(x)",title="三个 ReLU 拼出连续折线")
    axes[1].set(xlabel="x",ylabel="分段导数",title="转折点的经典导数不存在")
    for ax in axes:
        ax.grid(alpha=.2)
        for knot in (-1,0,1):ax.axvline(knot,color="#BBCBD1",lw=.6,ls="--")
    fig.tight_layout();fig.savefig(FIGURES/"4_1_piecewise.png",dpi=170);plt.close(fig)


def main(epochs=600, lr=.1):
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    sets, integrity=prepare_data()
    models=build_models()
    assert all(torch.equal(a,b) for a,b in zip(models["affine_stack"].parameters(),models["relu_mlp"].parameters()))
    histories, results, rows = {}, {}, []
    for name,net in models.items():
        history=fit(net,sets["train"],sets["validation"],epochs,lr)
        histories[name]=history
        results[name]={split:evaluate(net,*dataset) for split,dataset in sets.items()}
        results[name]["parameters"]=sum(p.numel() for p in net.parameters())
        for row in history:rows.append({"model":name,**row})
        print(name,results[name],flush=True)
    audit=theory_audit()
    construction={}
    for name,(X,y) in sets.items():
        correct=int((constructed_xor_logits(X).argmax(1)==y).sum())
        construction[name]={"correct":correct,"samples":len(y),"accuracy":correct/len(y)}
        assert correct==len(y)
    draw_figures(models,histories)
    checkpoint=ROOT/"artifacts/ch04/4_1_xor_models.pt"
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"models":{k:v.state_dict() for k,v in models.items()},"epochs":epochs,"lr":lr},checkpoint)
    metrics={"date":"2026-10-27","purpose":"teacher demonstration, not learner completion",
             "torch":torch.__version__,"device":"cpu","dtype":"float64","download":False,
             "dataset":"synthetic symmetric XOR; not Fashion-MNIST",
             "epochs":epochs,"learning_rate":lr,"batch":"full 480 samples","updates_per_model":epochs,
             "model_seed":4100,"test_evaluations_per_trained_model":1,"results":results,
             "construction":construction,"theory":audit,
             "checkpoint_local_only":checkpoint.relative_to(ROOT).as_posix(),
             "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    for path,payload in ((REPORT/"4_1_metrics.json",metrics),(REPORT/"4_1_data_integrity.json",integrity)):
        with path.open("w",encoding="utf-8",newline="\n") as handle:
            handle.write(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    with (REPORT/"4_1_epochs.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    print("手工构造 XOR:",construction)
    print("仿射合并最大误差:",audit["affine_merge_max_error"])
    print("PReLU 导数:",audit["prelu"])
    print("Fashion-MNIST 形状示意:",audit["fashion_forward_only"],"| 4.2 再训练真实图像")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",type=int,default=600)
    parser.add_argument("--lr",type=float,default=.1)
    args=parser.parse_args()
    if args.epochs<=0 or args.lr<=0:parser.error("epochs、lr 必须为正")
    main(args.epochs,args.lr)
