# -*- coding: utf-8 -*-
"""3.5 Fashion-MNIST：样本、变换、可视化、批次与读取性能。

只读取项目 data 内的已有数据；不下载、不训练模型。
先读 load_data_fashion_mnist，再看 main 中的一批数据与 Resize。
"""
import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

DATA_ROOT = ROOT / "data"
REPORT = ROOT / "reports/ch03"
FIGURES = REPORT / "figures"
NAMES = ["T恤/上衣", "长裤", "套衫", "连衣裙", "外套",
         "凉鞋", "衬衫", "运动鞋", "包", "短靴"]


def make_transform(resize=None):
    operations = []
    if resize is not None:
        operations.append(transforms.Resize((resize, resize), antialias=True))
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def load_data_fashion_mnist(batch_size=256, resize=None, num_workers=0):
    transform = make_transform(resize)
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True,
                                      transform=transform, download=False)
    test_set = datasets.FashionMNIST(DATA_ROOT, train=False,
                                     transform=transform, download=False)
    generator = torch.Generator().manual_seed(3501)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False, generator=generator)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, test_loader


def scan_loader(loader):
    sizes, counts = [], torch.zeros(10, dtype=torch.long)
    start = time.perf_counter()
    for images, labels in loader:
        assert images.shape[0] == labels.shape[0]
        assert images.dtype == torch.float32 and labels.dtype == torch.int64
        sizes.append(len(labels))
        counts += torch.bincount(labels, minlength=10)
    return {"samples": sum(sizes), "batches": len(sizes), "last_batch": sizes[-1],
            "class_counts": counts.tolist(), "seconds_with_count_checks": time.perf_counter()-start}


def benchmark(train_set, workers=True, sample_count=4096, repeats=2):
    """相同子集、顺序和 ToTensor，计时包含启动、迭代和本轮结束。"""
    subset = Subset(train_set, range(sample_count))
    settings = [("PIL_ToTensor", b, 0) for b in (1, 32, 256)]
    if workers:
        settings += [("PIL_ToTensor", 256, n) for n in (2, 4)]
    settings.append(("vectorized_fixed_transform", 256, 0))
    rows = []
    for method, batch_size, num_workers in settings:
        print("性能对照:", method, "batch", batch_size, "workers", num_workers, flush=True)
        for repeat in range(repeats):
            start = time.perf_counter()
            current = subset
            if method == "vectorized_fixed_transform":
                x = train_set.data[:sample_count].unsqueeze(1).float() / 255.
                current = TensorDataset(x, train_set.targets[:sample_count])
            loader = DataLoader(current, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=False)
            setup = time.perf_counter() - start
            count, batches = 0, 0
            for _, labels in loader:
                count += len(labels); batches += 1
            total = time.perf_counter() - start
            assert count == sample_count
            rows.append({"method": method, "batch_size": batch_size, "num_workers": num_workers,
                         "repeat": repeat+1, "samples": count, "batches": batches,
                         "setup_seconds": setup, "iteration_seconds": total-setup,
                         "total_seconds": total, "samples_per_second": count/total,
                         "worker_startup_included": True})
    return rows


def summarize_benchmark(rows):
    result = []
    keys = list(dict.fromkeys((r["method"], r["batch_size"], r["num_workers"]) for r in rows))
    for method, batch, workers in keys:
        selected = [r for r in rows if (r["method"], r["batch_size"], r["num_workers"]) == (method,batch,workers)]
        result.append({"method": method, "batch_size": batch, "num_workers": workers,
                       "samples": selected[0]["samples"], "repeats": len(selected),
                       "median_total_seconds": statistics.median(r["total_seconds"] for r in selected)})
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def draw_figures(train_set, resized_set, benchmarks):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    indices = [int(torch.where(train_set.targets == j)[0][0]) for j in range(10)]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.8))
    for j, (ax, index) in enumerate(zip(axes.ravel(), indices)):
        image, label = train_set[index]
        ax.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{label}: {NAMES[label]}"); ax.axis("off")
    fig.text(.5, .012, "Fashion-MNIST · © 2017 Zalando SE · MIT；每类取训练集首个样本", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1)); fig.savefig(FIGURES/"3_5_classes.png", dpi=170); plt.close(fig)

    image, label = train_set[0]
    resized, same_label = resized_set[0]
    assert label == same_label
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(train_set.data[0].numpy(), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("原始 uint8：28×28，0～255")
    axes[1].imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("ToTensor：1×28×28，0～1")
    axes[2].imshow(resized[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Resize 后：1×64×64")
    for ax in axes: ax.axis("off")
    fig.text(.5,.02,"同一训练样本，标签 9：短靴；放大像素网格不会恢复新细节。© 2017 Zalando SE · MIT",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94))
    fig.savefig(FIGURES/"3_5_transforms.png",dpi=170,bbox_inches="tight",pad_inches=.14)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    texts = [("单张图片", "(1, 28, 28)", "通道 C · 高 H · 宽 W"),
             ("堆成一批", "(256, 1, 28, 28)", "样本 N · 通道 C · 高 H · 宽 W"),
             ("准备线性模型", "(256, 784)", "保留样本维，只展平后面三维")]
    for i,(title,shape,note) in enumerate(texts):
        x = .02 + i*.335
        ax.add_patch(plt.Rectangle((x,.25),.29,.58,facecolor="#F1F7F8",edgecolor="#147E83",transform=ax.transAxes))
        ax.text(x+.145,.69,title,ha="center",transform=ax.transAxes,color="#173448",fontsize=13)
        ax.text(x+.145,.50,shape,ha="center",transform=ax.transAxes,color="#147E83",fontsize=15)
        ax.text(x+.145,.33,note,ha="center",transform=ax.transAxes,fontsize=9)
        if i<2: ax.text(x+.312,.53,"→",ha="center",transform=ax.transAxes,fontsize=19)
    ax.text(.5,.08,"对应标签始终为 (256,) 的整数类别编号；本节不训练模型。",ha="center",transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(FIGURES/"3_5_shapes.png",dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10,4))
    selected = [r for r in benchmarks if r["method"]=="PIL_ToTensor" and r["num_workers"]==0]
    axes[0].bar([str(r["batch_size"]) for r in selected], [r["median_total_seconds"] for r in selected],color="#147E83")
    axes[0].set(xlabel="batch_size（workers=0）",ylabel="秒，中位数",title="同样读取前 4096 条")
    choices = [r for r in benchmarks if r["batch_size"]==256]
    labels = ["预转张量" if r["method"]!="PIL_ToTensor" else f"进程数 {r['num_workers']}" for r in choices]
    values = [r["median_total_seconds"] for r in choices]
    axes[1].bar(labels,values,color=["#147E83"]*(len(values)-1)+["#D47F37"])
    axes[1].set(yscale="log",ylabel="秒，中位数（对数轴）",title="含启动/退出；预转含转换时间")
    for i,value in enumerate(values): axes[1].text(i,value*1.12,f"{value:.3f}",ha="center",fontsize=10)
    axes[1].set_ylim(min(values)*.5,max(values)*2)
    for ax in axes: ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/"3_5_benchmark.png",dpi=170); plt.close(fig)
    return indices


def main(skip_worker_benchmark=False):
    torch.set_num_threads(1)
    REPORT.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    train_loader,test_loader = load_data_fashion_mnist()
    train_set,test_set = train_loader.dataset,test_loader.dataset
    image,label = train_set[0]
    images,labels = next(iter(train_loader))
    assert torch.equal(image, train_set.data[0].unsqueeze(0).float()/255.)
    resized_set = datasets.FashionMNIST(DATA_ROOT,train=True,transform=make_transform(64),download=False)
    resized_batch,_ = next(iter(DataLoader(resized_set,batch_size=32)))
    scans = {"train":scan_loader(train_loader),"test":scan_loader(test_loader)}
    assert scans["train"]["samples"]==60000 and scans["train"]["last_batch"]==96
    assert scans["test"]["samples"]==10000 and scans["test"]["last_batch"]==16
    assert scans["train"]["class_counts"]==[6000]*10
    assert scans["test"]["class_counts"]==[1000]*10
    timings = benchmark(train_set,workers=not skip_worker_benchmark)
    summary = summarize_benchmark(timings)
    indices = draw_figures(train_set,resized_set,summary)
    metrics = {"purpose":"teacher demonstration, not learner completion", "date":"2026-10-20",
               "torch":torch.__version__,"torchvision":torchvision.__version__,"device":"cpu",
               "dataset_root":"data/FashionMNIST/raw", "download":False,
               "source":"verified existing local data, hard links; source content untouched",
               "class_names":NAMES,"train_size":len(train_set),"test_size":len(test_set),
               "raw_image_shape":list(train_set.data[0].shape),"raw_image_dtype":str(train_set.data.dtype),
               "first_image_shape":list(image.shape),"first_image_dtype":str(image.dtype),
               "first_label":label,"first_image_min":image.min().item(),"first_image_max":image.max().item(),
               "batch_shape":list(images.shape),"batch_dtype":str(images.dtype),
               "labels_shape":list(labels.shape),"labels_dtype":str(labels.dtype),
               "resized_batch_shape":list(resized_batch.shape),"loader_seed":3501,
               "full_passes":scans,"example_indices_by_class":indices,
               "benchmark_subset":"first 4096 training samples, ordered, no model/GPU",
               "benchmark_repeats":2,"worker_startup_included":True,
               "benchmark_summary":summary,"no_model_training":True}
    with (REPORT/"3_5_metrics.json").open("w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    write_csv(REPORT/"3_5_benchmark.csv",timings)
    write_csv(REPORT/"3_5_class_counts.csv",[{"class_id":i,"name":NAMES[i],"train":6000,"test":1000} for i in range(10)])
    print("3.5 图像分类数据集 |",torch.__version__,torchvision.__version__)
    print("已有数据，无下载；训练/测试数量:",len(train_set),len(test_set))
    print("首张:",image.shape,image.dtype,"标签:",label,NAMES[label],"值域:",(image.min().item(),image.max().item()))
    print("一批:",images.shape,images.dtype,"标签:",labels.shape,labels.dtype)
    print("Resize=64:",resized_batch.shape)
    print("完整遍历:",scans)
    for row in summary: print(row)
    print("本节完成读取与检查，没有训练模型。报告和四张图在 reports/ch03。")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-worker-benchmark",action="store_true",help="重复演示时略过多进程计时")
    main(parser.parse_args().skip_worker_benchmark)
