# -*- coding: utf-8 -*-
"""3.5 综合实践：读取真实图像并组织批次，一次填写八处 TODO。"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

NAMES=["T恤/上衣","长裤","套衫","连衣裙","外套","凉鞋","衬衫","运动鞋","包","短靴"]


def main():
    transform=None  # TODO 1：Resize 到 32×32，再 ToTensor，用 Compose 组织。
    train_set=None  # TODO 2：FashionMNIST 训练集，root 为 ROOT/'data'，禁止重复下载。
    if transform is None or train_set is None:
        print("集中填写八处 TODO；已有数据，无需下载或安装。")
        return
    first_image=None  # TODO 3：取数据集第 0 条样本中的图片。
    first_label=None  # TODO 4：取第 0 条样本中的标签。
    subset=Subset(train_set,range(37))
    loader=None  # TODO 5：每批 8 条，打乱、保留尾批，num_workers=0。
    if first_image is None or first_label is None or loader is None:
        print("请完成样本读取和批次组织。")
        return
    sizes,seen=[],0
    for images,labels in loader:
        sizes.append(len(labels));seen+=len(labels)
        flat=None  # TODO 6：只保留样本维，展平通道与高宽。
        readable_labels=None  # TODO 7：把本批整数标签转成中文名称列表。
        if flat is None or readable_labels is None:
            print("请完成展平和名称映射。")
            return
        assert flat.shape[0]==len(labels)
    first_for_plot=None  # TODO 8：仅取灰度通道，得到二维图供 imshow 使用。
    if first_for_plot is None:
        print("请完成二维可视化数据。")
        return
    figure,axis=plt.subplots(figsize=(3.2,3.2))
    axis.imshow(first_for_plot.numpy(),cmap="gray",vmin=0,vmax=1)
    axis.set_title(f"class {int(first_label)}");axis.axis("off")
    output=ROOT/"artifacts/ch03/3_5_exercise_preview.png"
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.tight_layout();figure.savefig(output,dpi=130);plt.close(figure)
    print("首图:",first_image.shape,first_image.dtype,"标签:",first_label,NAMES[int(first_label)])
    print("批量大小:",sizes,"覆盖数:",seen)
    print("最后一批展平:",flat.shape,"标签名称:",readable_labels)
    print("预览已保存:",output)
    print("集中解释：(N,C,H,W) 各代表什么？ToTensor 后还应再除一次 255 吗？")
    print("Resize 和 reshape 有何不同？37 条每批 8 条的尾批有几条？")
    print("train=False 是验证集还是官方测试集？num_workers 越大一定越快吗？")


if __name__=="__main__":
    main()
