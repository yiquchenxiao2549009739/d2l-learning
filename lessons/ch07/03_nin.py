"""7.3 NiN：空间卷积、逐位置MLP、十通道与全局平均汇聚。

默认复现教材224输入/完整通道/SGD .1/十轮，保留最后ReLU。
--inspect只查结构；--experiments比较预定七方案验证集；--benchmark测训练/预测。
--record才保存报告、曲线和本地权重。自定义短训练不读取测试集、不覆盖正式记录。
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

PREVIOUS = runpy.run_path(str(ROOT / "lessons/ch07/02_vgg.py"))
COMMON = PREVIOUS["COMMON"]
prepare, evaluate, train_epoch = (PREVIOUS[n] for n in ("prepare", "evaluate", "train_epoch"))
write_json, write_csv, plot_setup = (PREVIOUS[n] for n in ("write_json", "write_csv", "plot_setup"))
REPORT = ROOT / "reports/ch07"
BASE = dict(size=224, channels=[96, 256, 384], pointwise=2, last_width=10,
            final_relu=True, dropout=.5, lr=.1, batch_size=128)


def checked(config=None):
    cfg = dict(BASE, **(config or {}))
    if cfg["size"] < 67 or len(cfg["channels"]) != 3 or min(cfg["channels"]) < 1:
        raise ValueError("三次3×3汇聚需要至少67输入，channels含三个正整数")
    if cfg["pointwise"] not in (0, 1, 2) or cfg["last_width"] < 1:
        raise ValueError("pointwise须为0/1/2，last_width须为正")
    if cfg["pointwise"] == 0 and cfg["last_width"] != 10:
        raise ValueError("无1×1层时末块空间卷积须直接输出10通道")
    if not 0 <= cfg["dropout"] < 1 or cfg["lr"] <= 0 or cfg["batch_size"] < 1:
        raise ValueError("检查暂退概率、学习率与批量大小")
    return cfg


def nin_block(in_channels, out_channels, kernel_size, stride=1, padding=0,
              pointwise=2, first_width=None, final_relu=True, device="cpu"):
    """空间卷积后接0/1/2个1×1层；默认三个Conv各接ReLU。"""
    width = out_channels if first_width is None else first_width
    if pointwise not in (0, 1, 2) or (pointwise == 0 and width != out_channels):
        raise ValueError("检查逐位置层数及末端通道")
    layers = [("spatial", nn.Conv2d(in_channels, width, kernel_size, stride, padding, device=device))]
    if pointwise or final_relu:
        layers.append(("relu0", nn.ReLU()))
    for index in range(1, pointwise + 1):
        next_width = out_channels if index == pointwise else width
        layers.append(("point" + str(index), nn.Conv2d(width, next_width, 1, device=device)))
        if index < pointwise or final_relu:
            layers.append(("relu" + str(index), nn.ReLU()))
        width = next_width
    return nn.Sequential(OrderedDict(layers))


def build_net(config=None, device="cpu", seed=7300):
    cfg = checked(config)
    if torch.device(device).type not in ("cpu", "meta"):
        raise ValueError("先CPU初始化，再to目标设备；meta仅核对结构")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        c1, c2, c3 = cfg["channels"]
        layers = []
        for index, (ci, co, kernel, stride, padding) in enumerate(
                [(1,c1,11,4,0), (c1,c2,5,1,2), (c2,c3,3,1,1), (c3,10,3,1,1)]):
            layers.append(("block"+str(index+1), nin_block(ci,co,kernel,stride,padding,
                pointwise=cfg["pointwise"], first_width=cfg["last_width"] if index==3 else co,
                final_relu=cfg["final_relu"] if index==3 else True,device=device)))
            if index < 3:
                layers.append(("pool"+str(index+1),nn.MaxPool2d(3,2)))
            if index == 2:
                layers.append(("dropout",nn.Dropout(cfg["dropout"])))
        layers.extend([("gap",nn.AdaptiveAvgPool2d((1,1))), ("flatten",nn.Flatten())])
        net = nn.Sequential(OrderedDict(layers))
        if torch.device(device).type == "cpu":
            for block_index in range(1,5):
                for role, layer in getattr(net,"block"+str(block_index)).named_children():
                    if isinstance(layer,nn.Conv2d):
                        role_index = 0 if role=="spatial" else int(role[-1])
                        torch.default_generator.manual_seed(seed+10*(block_index-1)+role_index)
                        nn.init.xavier_uniform_(layer.weight,gain=1.)
                        nn.init.zeros_(layer.bias)
    return net


def parameter_hash(net, trunk_only=False):
    digest = hashlib.sha256()
    for name,value in net.state_dict().items():
        if trunk_only and not name.startswith(("block1.","block2.","block3.")):
            continue
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def inspect_net(config=None):
    cfg=checked(config)
    net=build_net(cfg,device="meta").eval()
    weighted,outer,handles=[],[],[]
    def hook(name):
        def record(layer,inputs,output):
            weighted.append(dict(name=name,input_shape=list(inputs[0].shape),output_shape=list(output.shape),
                kernel=list(layer.kernel_size),parameters=sum(p.numel() for p in layer.parameters()),
                weight_multiplications=output.numel()*layer.in_channels*layer.kernel_size[0]*layer.kernel_size[1],
                output_float32_bytes=output.numel()*4))
        return record
    for name,layer in net.named_modules():
        if isinstance(layer,nn.Conv2d):handles.append(layer.register_forward_hook(hook(name)))
    X=torch.zeros(1,1,cfg["size"],cfg["size"],device="meta")
    with torch.no_grad():
        for name,layer in net.named_children():
            X=layer(X)
            outer.append(dict(name=name,layer=type(layer).__name__,shape=list(X.shape)))
    for h in handles:h.remove()
    count=sum(p.numel() for p in net.parameters())
    last_map=outer[-3]["shape"]
    return dict(config=cfg,outer=outer,weighted_layers=weighted,parameters=count,
        parameter_float32_bytes=count*4,weight_multiplications=sum(r["weight_multiplications"] for r in weighted),
        conv_count=len(weighted),linear_count=0,global_average_additions_per_image=10*(last_map[-1]*last_map[-2]-1),
        note="meta结构计数；权重乘法不含bias/ReLU/汇聚/反传，不是实测FLOPs或全部显存")


def fit(cfg,splits,epochs,device,name,use_test=False,record=False,prefix=None):
    net=build_net(cfg).to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr=cfg["lr"])
    loader=COMMON["make_loader"](splits["train"],shuffle=True,batch_size=cfg["batch_size"])
    torch.manual_seed(7303)
    if device.type=="cuda":
        torch.cuda.manual_seed_all(7303)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
    result=dict(config=cfg,model_seed_base=7300,initialization="CPU Xavier gain1, zero bias; seed7300+10*block_index+role_index",
        dropout_seed=7303,loader_seed=3603,initial_parameter_sha256=parameter_hash(net),
        initial_trunk_sha256=parameter_hash(net,True),parameters=sum(p.numel() for p in net.parameters()),
        runtime=COMMON["runtime_metadata"](device),epochs=epochs,train_samples=len(splits["train"]),
        validation_samples=len(splits["validation"]),history=[])
    started=time.perf_counter()
    for epoch in range(epochs+1):
        online={} if epoch==0 else train_epoch(net,loader,optimizer,cfg,device)
        val=evaluate(net,splits["validation"],cfg,device)
        row=dict(epoch=epoch,**online,**{"validation_"+k:v for k,v in val.items()})
        if epoch==epochs:row.update({"train_eval_"+k:v for k,v in evaluate(net,splits["train"],cfg,device).items()})
        result["history"].append(row)
        print("{} epoch {:2d}: online CE {}, val CE {:.6f}, acc {:.2%}".format(name,epoch,
            "-" if not online else "{:.6f}".format(online["online_train_ce"]),val["ce"],val["accuracy"]),flush=True)
    if use_test:result["test"]=evaluate(net,splits["test"],cfg,device)
    result.update(updates=sum(r.get("updates",0) for r in result["history"]),
        last_batch_size=(len(splits["train"])-1)%cfg["batch_size"]+1,test_evaluated=use_test,
        final_parameter_sha256=parameter_hash(net),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
    if record:
        folder=ROOT/"artifacts/ch07";folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(config=cfg,epoch=epochs,model={k:v.detach().cpu() for k,v in net.state_dict().items()},
            optimizer=optimizer.state_dict()),folder/(prefix+".pt"))
    if device.type=="cuda":torch.cuda.synchronize()
    result["seconds_including_evaluation_and_save"]=time.perf_counter()-started
    return result,net


def candidates():
    return {"A":dict(BASE),"B":dict(BASE,lr=.05),"C":dict(BASE,lr=.2),
            "D":dict(BASE,pointwise=1),"E":dict(BASE,pointwise=0),
            "F":dict(BASE,last_width=128),"G":dict(BASE,final_relu=False)}


def run_experiments(device,record):
    splits,hashes=COMMON["load_data"](include_test=False)
    splits=dict(train=COMMON["balanced_subset"](splits["train"],2000,7301),
                validation=COMMON["balanced_subset"](splits["validation"],1000,7302))
    results,rows={},[]
    for name,cfg in candidates().items():
        result,net=fit(cfg,splits,20,device,name)
        results[name]=result;rows.extend(dict(candidate=name,**r) for r in result["history"])
        del net
        if device.type=="cuda":torch.cuda.empty_cache()
    assert len({results[k]["initial_parameter_sha256"] for k in ("A","B","C","G")})==1
    assert results["A"]["initial_trunk_sha256"]==results["F"]["initial_trunk_sha256"]
    selected=min(results,key=lambda k:results[k]["history"][-1]["validation_ce"])
    output=dict(split_hashes=hashes,train_subset_seed=7301,validation_subset_seed=7302,
        train_per_class=200,validation_per_class=100,test_evaluated=False,results=results,selected_by_final_validation_ce=selected,
        protocol="预定七候选，各20轮320更新；固定末轮验证CE比较，不读测试，不重新挑早期最佳轮次")
    if record:
        write_json("7_3_experiments.json",output);write_csv("7_3_experiments_epochs.csv",rows)
        plot_results(output,experiments=True)
    return output


def plot_results(data,experiments=False):
    plt=plot_setup();fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
    if experiments:
        for name,result in data["results"].items():
            h=result["history"]
            axes[0].plot([r["epoch"] for r in h],[r["validation_ce"] for r in h],label=name)
        axes[1].bar(list(data["results"]),[r["history"][-1]["validation_accuracy"] for r in data["results"].values()],color="#147e83")
        axes[1].set(ylabel="末轮验证准确率",ylim=(0,1));suffix="experiments"
    else:
        h=data["history"]
        axes[0].plot([r["epoch"] for r in h[1:]],[r["online_train_ce"] for r in h[1:]],"o-",label="在线训练")
        axes[0].plot([r["epoch"] for r in h],[r["validation_ce"] for r in h],"o-",label="验证")
        axes[1].plot([r["epoch"] for r in h],[r["validation_accuracy"] for r in h],"o-",label="验证")
        axes[1].set(xlabel="epoch",ylabel="准确率",ylim=(0,1));suffix="training"
    axes[0].set(xlabel="epoch",ylabel="交叉熵");axes[0].legend(fontsize=10)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(REPORT/"figures"/("7_3_"+suffix+".png"),dpi=180);plt.close(fig)


def benchmark(device,record):
    if device.type!="cuda":raise RuntimeError("显存与CUDA事件测量需要GPU")
    results=[]
    for batch in (1,32,128):
        for mode in ("train","inference"):
            torch.cuda.empty_cache()
            torch.manual_seed(7304);torch.cuda.manual_seed_all(7304)
            net=build_net().to(device)
            X=torch.rand(batch,1,224,224,device=device)
            y=torch.arange(batch,device=device)%10
            optimizer=torch.optim.SGD(net.parameters(),lr=.1)
            if mode=="train":
                net.train()
                def step():
                    optimizer.zero_grad(set_to_none=True)
                    torch.nn.functional.cross_entropy(net(X),y).backward();optimizer.step()
            else:
                net.eval()
                @torch.inference_mode()
                def step():net(X)
            for _ in range(2):step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
            baseline=torch.cuda.memory_allocated(device)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):step()
            end.record();torch.cuda.synchronize()
            results.append(dict(mode=mode,batch_size=batch,milliseconds_per_batch=start.elapsed_time(end)/5,
                baseline_allocated_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
            del net,optimizer,X,y
    output=dict(runtime=COMMON["runtime_metadata"](device),synthetic_seed=7304,warmup=2,repeats=5,results=results,
        protocol="单独顺序短测，随机GPU驻留float32；训练前向+CE+反传+SGD，预测eval+inference_mode；不含数据/Resize；分配值含权重输入而非全进程")
    if record:
        write_json("7_3_benchmark.json",output)
        plt=plot_setup();fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
        for mode in ("train","inference"):
            rows=[r for r in results if r["mode"]==mode]
            axes[0].plot([r["batch_size"] for r in rows],[r["milliseconds_per_batch"] for r in rows],"o-",label=mode)
            axes[1].plot([r["batch_size"] for r in rows],[r["peak_allocated_bytes"]/2**20 for r in rows],"o-",label=mode)
        axes[0].set(xlabel="batch",ylabel="每批毫秒");axes[1].set(xlabel="batch",ylabel="分配峰值MiB")
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.savefig(REPORT/"figures/7_3_benchmark.png",dpi=180);plt.close(fig)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    for name in ("inspect","experiments","benchmark"):modes.add_argument("--"+name,action="store_true")
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--epochs",type=int,default=10);parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--train-size",type=int);parser.add_argument("--validation-size",type=int)
    parser.add_argument("--guided",action="store_true");parser.add_argument("--record",action="store_true")
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1:parser.error("轮数和批量须为正整数")
    if any(v is not None and v<1 for v in (args.train_size,args.validation_size)):parser.error("子集大小须为正整数")
    custom=args.epochs!=10 or args.batch_size!=128 or args.train_size is not None or args.validation_size is not None
    if (args.inspect or args.experiments or args.benchmark) and custom:parser.error("固定协议不接受额外训练参数")
    device=COMMON["configure"](args.device)
    if args.guided:print("7.3：空间卷积→共享逐位置MLP→十通道→全局平均。默认保留教材末端ReLU。",flush=True)
    if args.inspect:
        result=inspect_net()
        if args.record:
            write_json("7_3_architecture.json",result);write_csv("7_3_layers.csv",result["weighted_layers"])
    elif args.experiments:result=run_experiments(device,args.record)
    elif args.benchmark:result=benchmark(device,args.record)
    else:
        splits,hashes=COMMON["load_data"](include_test=not custom)
        for name,size,seed in (("train",args.train_size,7301),("validation",args.validation_size,7302)):
            if size is not None:splits[name]=COMMON["balanced_subset"](splits[name],size,seed)
        cfg=dict(BASE,batch_size=args.batch_size);prefix="7_3_custom" if custom else "7_3"
        result,net=fit(cfg,splits,args.epochs,device,"NiN",use_test=not custom,record=args.record,prefix=prefix)
        result.update(custom=custom,split_hashes=hashes,split_seed=3601,
            preprocessing="float32/255 then bilinear, align_corners=False, antialias=True")
        if args.record:
            write_json(prefix+"_metrics.json",result);write_csv(prefix+"_epochs.csv",result["history"])
            if not custom:plot_results(result)
    if args.guided:
        if args.inspect:
            for row in result["outer"]:print(row["name"],row["shape"])
            print("参数",result["parameters"],"单图权重乘法",result["weight_multiplications"])
        elif "test" in result:print("固定末轮测试",result["test"])
        print("运行验证不代表掌握；综合练习八处TODO留给自己填写。")
    else:print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
