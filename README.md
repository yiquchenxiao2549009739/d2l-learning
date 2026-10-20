# 李沐《动手学深度学习》学习项目

从“看懂简单代码”开始，逐步做到自己实现、训练和解释模型。

当前一次学习 **完整的 3.6 softmax 回归的从零开始实现（3.6.1–3.6.7）**。先看 [整节讲义](docs/ch03/06_softmax_scratch.md)，运行一个完整演示，再完成一个综合练习并集中反馈。

## 阅读版 PDF

| 小节 | 页数 | 文件 |
| --- | --- | --- |
| 2.1 数据操作 | 8 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.1-数据操作-学习讲义.pdf) |
| 2.2 数据预处理 | 7 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.2-数据预处理-学习讲义.pdf) |
| 2.3 线性代数 | 13 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.3-线性代数-学习讲义.pdf) |
| 2.4 微积分 | 13 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.4-微积分-学习讲义.pdf) |
| 2.5 自动微分 | 13 页 | [学习讲义 PDF](output/pdf/2.5-自动微分-学习讲义.pdf) |
| 2.6 概率 | 15 页 | [学习讲义 PDF](output/pdf/2.6-概率-学习讲义.pdf) |
| 2.7 查阅文档 | 6 页 | [学习讲义 PDF](output/pdf/2.7-查阅文档-学习讲义.pdf) |
| 3.1 线性回归 | 14 页 | [学习讲义 PDF](output/pdf/3.1-线性回归-学习讲义.pdf) |
| 3.2 从零实现线性回归 | 11 页 | [学习讲义 PDF](output/pdf/3.2-线性回归的从零开始实现-学习讲义.pdf) |
| 3.3 简洁实现线性回归 | 11 页 | [学习讲义 PDF](output/pdf/3.3-线性回归的简洁实现-学习讲义.pdf) |
| 3.4 softmax 回归 | 15 页 | [学习讲义 PDF](output/pdf/3.4-softmax回归-学习讲义.pdf) |
| 3.5 图像分类数据集 | 11 页 | [学习讲义 PDF](output/pdf/3.5-图像分类数据集-学习讲义.pdf) |
| 3.6 softmax 从零实现 | 15 页 | [学习讲义 PDF](output/pdf/3.6-softmax回归的从零开始实现-学习讲义.pdf) |

十三份包含原理、代码示例、结果解释与综合练习，嵌入中文字体并提供章节书签和页码。2.2 附有预处理数据；2.3–2.5 配有原理图及模型计算；2.6 配有频率收敛、贝叶斯更新、条件独立和均值方差四张图，贯通抽样与条件判断。2.7 通过查阅工具、签名解释和最小实验贯通整节。3.1 配有训练、正态分布、损失与网络四张图，并对照最小二乘和梯度下降。3.2 配有尾批、训练与学习率三张图，逐步组织生成器和手写 SGD。3.3 配有训练、实现对照与离群值三张图，贯通 DataLoader、Linear、MSELoss、SGD；同批对照核对损失尺度，并说明教材 Huber 公式与 SmoothL1Loss 的关系。3.4 配有得分与概率、交叉熵与温度、训练、分类区域与测试计数四张图，覆盖标签、梯度、信息论及全部节末拓展。3.5 配有十类样本、像素与 Resize、批次形状及性能对照四张图，使用本机已有 Fashion-MNIST 完成真实读取。3.6 配有训练、正确/错误预测、混淆矩阵与权重四张图，手写稳定交叉熵、SGD、累加统计和完整真实图像训练。3.6 整理日期：2026-10-20，旧资料日期保留。

PDF 由 `tools/build_study_pdfs.py` 使用现有文档工具环境生成，不需要修改 openmmlab。可用 `--section 3.6` 只重建当前讲义。原始教材仍不纳入 Git；只允许上述学习 PDF 同步。

2.1–2.4 PDF 保持本地删除状态；2.4 的删除在本轮开始时检测到。GitHub 均保留，因此表中提供存档链接。本轮不恢复文件，也不把这些本地删除提交到远端。

## 在 VS Code 中开始

1. 用 VS Code 打开本目录中的 `d2l-learning.code-workspace`。
2. 打开 `lessons/ch03/06_softmax_scratch.py`。
3. 按 **F5**，选择“3.6 softmax 从零实现：完整演示”，一次运行整节。也可以点编辑器右上角“运行 Python 文件”。
4. 在 `exercises/ch03/06_softmax_scratch.py` 集中填写八处 TODO，在 37 张真实图片上完成 softmax、交叉熵、模型、反向传播、手写更新、预测与准确率，再统一解释尾批、统计与泛化。

这块使用已有 Fashion-MNIST，手写 784→10 分类器、稳定损失、累加器、SGD 与完整训练/评估。官方训练集按类固定划分为 54,000 训练与 6,000 验证；10,000 测试只在固定十轮方案结束后评估。默认 CPU/float32、lr=0.1、batch256，2110 次更新。

前面各节的代码、讲义和练习继续保留，2.1–2.4 PDF 的本地状态见上文；单独的旧 tensor、reshape 文件已删除且未重建。问答与已完成实践的记录在 `learning_log.md`。用户主动要求进入 3.6，未据此将未提交的旧综合练习标记完成。

解释器已经指定为已有的 `openmmlab` 环境：`D:/Anaconda/envs/openmmlab/python.exe`。第一次打开若提示选择解释器，选择 `openmmlab`。

本机 Fashion-MNIST 原始文件已核对官方归档 MD5、解压内容 SHA-256 和 IDX 头。项目 `data/` 中的硬链接与已有文件共享存储，不额外复制约 52.4 MiB 原始像素文件；这些文件只用于读取。机器相关来源保存在被忽略的本地记录中，数据不会提交 GitHub。其他电脑需要先准备自己的数据；`tools/prepare_fashion_mnist.py --source <已有数据根目录>` 可校验并在同一文件系统建立硬链接。

默认读取使用 `num_workers=0`。3.5 演示另测 0/2/4 个加载进程与批量大小，Windows 入口已加 main 保护；重复运行 3.5 可用 `--skip-worker-benchmark`。3.6 在内存中复用固定像素变换，原始文件不改写；训练与逐轮评估本次约 78.68 秒。图像示例来自 Fashion-MNIST，许可保存在 `docs/references/fashion_mnist_LICENSE.txt`。

## 文件各自做什么

| 文件或目录 | 用途 |
| --- | --- |
| `docs/ch03/06_softmax_scratch.md` | 完整的 3.6 讲义 |
| `lessons/ch03/06_softmax_scratch.py` | 当前统一演示入口 |
| `exercises/ch03/06_softmax_scratch.py` | 当前综合练习入口 |
| `datasets/ch02/`、`datasets/ch03/` | 预处理样例、回归与分类人工训练/验证/测试 CSV |
| `data/FashionMNIST/raw/` | 本地官方图像数据，已排除 Git；四个硬链接复用现有存储 |
| `reports/ch02/figures/`、`reports/ch03/figures/` | 已讲章节的原理图和实验曲线 |
| `docs/roadmap.md` | 全书路线和 PDF 页码 |
| `lessons/` | 讲解示例 |
| `exercises/` | 自己动手写的练习 |
| `notes/` | 自己的答案和理解 |
| `progress.md` | 当前进度与下一步 |
| `learning_log.md` | 每回合的讲解位置、真实问答和反馈 |
| `reports/` | 同步 GitHub 的实验指标、曲线与观察 |
| `tools/` | 环境检查与训练验证，暂时不需要读懂 |
| `scripts/` | 选择已有环境和 Git 同步 |
| `artifacts/` | 训练生成的曲线、指标、模型，只保存在本地 |

后续每次以完整小节或较大的主题块推进，集中讲原理、整合程序、统一实践和反馈，不再将单个函数拆成多轮。下一块为 3.7 softmax 回归的简洁实现，将手写部件与框架层、损失和优化器对应。旧练习的未验证状态保留，教师运行不记作用户掌握。

## 环境和验证

项目复用本机已有的 Conda `openmmlab` 环境：Python 3.8.20、PyTorch 2.4.1+cu121、NumPy 1.23.5、pandas 2.0.3、Matplotlib 3.7.2。GPU 为 RTX 4060 Laptop GPU。按照用户要求，不额外安装或升级软件。

已经建立环境的电脑可在项目终端运行：

```powershell
& 'D:\Anaconda\envs\openmmlab\python.exe' tools/check_environment.py
& 'D:\Anaconda\envs\openmmlab\python.exe' tools/smoke_train.py --device cpu
& 'D:\Anaconda\envs\openmmlab\python.exe' tools/smoke_train.py --device cuda
```

最后两条是安装验证用的小型线性回归训练，不要求现在读懂，也不算完成教材第 3 章。训练结果在 `artifacts/smoke/`。

换电脑或选择别的已有环境时，提供解释器路径：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/use_environment.ps1 -PythonPath 'D:\Anaconda\envs\openmmlab\python.exe'
```

该脚本只检查已有环境、设置 VS Code 和保存本机路径，不安装任何软件。`requirements-observed.txt` 记录当前用到的库版本，供排查和比较，暂时不需要执行安装命令。

## 保存和同步

代码和笔记保存到 [GitHub 私有仓库 d2l-learning](https://github.com/yiquchenxiao2549009739/d2l-learning)；工具缓存、原 PDF、数据集、模型和个人配置不会提交。

每个教学回合，我会记录你的回答、讲解、反馈、代码练习、实际实验结果和下一步，完成 Git 提交与推送。已生成的讲解文件记录本回合内容；后续回答在下一回合补入日志。不将尚未作答的内容写成已掌握。

`reports/` 保存用于学习复盘的指标、曲线和实验观察；大型原始数据、完整模型与缓存留在本地。

你在 VS Code 自己修改后，也可以在项目终端执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/sync.ps1 -Message "learn: finish Python variable practice"
```

该脚本先检查当前分支和远端，再提交修改并推送；分支有未合并的远端更新时会停下。发生冲突时请把提示发给我。

## 教材与参考

教材为用户提供的《动手学深度学习》Release 2.0.0（2023-08-18，797 个 PDF 页面）。原教材保留在原位置，本项目保存本地路径配置与章节索引，不复制全文。

- [教材在线版](https://zh.d2l.ai/)
- [PyTorch 官方版本安装命令](https://pytorch.org/get-started/previous-versions/)

跟随纸书时请区分印刷页码与 PDF 页码，详见路线表。
