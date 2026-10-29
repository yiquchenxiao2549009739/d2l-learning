# 李沐《动手学深度学习》学习项目

从“看懂简单代码”开始，逐步做到自己实现、训练和解释模型。

当前一次学习 **完整的5.2参数管理（5.2.1–5.2.3及四道节末练习）**。先看 [整节讲义](docs/ch05/02_parameter_management.md)，运行统一实验，再完成综合练习并集中反馈。

## 阅读版 PDF

| 小节 | 页数 | 文件 |
| --- | --- | --- |
| 2.1 数据操作 | 8 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.1-数据操作-学习讲义.pdf) |
| 2.2 数据预处理 | 7 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.2-数据预处理-学习讲义.pdf) |
| 2.3 线性代数 | 13 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.3-线性代数-学习讲义.pdf) |
| 2.4 微积分 | 13 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.4-微积分-学习讲义.pdf) |
| 2.5 自动微分 | 13 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.5-自动微分-学习讲义.pdf) |
| 2.6 概率 | 15 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.6-概率-学习讲义.pdf) |
| 2.7 查阅文档 | 6 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.7-查阅文档-学习讲义.pdf) |
| 3.1 线性回归 | 14 页 | [学习讲义 PDF](output/pdf/3.1-线性回归-学习讲义.pdf) |
| 3.2 从零实现线性回归 | 11 页 | [学习讲义 PDF](output/pdf/3.2-线性回归的从零开始实现-学习讲义.pdf) |
| 3.3 简洁实现线性回归 | 11 页 | [学习讲义 PDF](output/pdf/3.3-线性回归的简洁实现-学习讲义.pdf) |
| 3.4 softmax 回归 | 15 页 | [学习讲义 PDF](output/pdf/3.4-softmax回归-学习讲义.pdf) |
| 3.5 图像分类数据集 | 11 页 | [学习讲义 PDF](output/pdf/3.5-图像分类数据集-学习讲义.pdf) |
| 3.6 softmax 从零实现 | 15 页 | [学习讲义 PDF](output/pdf/3.6-softmax回归的从零开始实现-学习讲义.pdf) |
| 3.7 softmax 简洁实现 | 12 页 | [学习讲义 PDF](output/pdf/3.7-softmax回归的简洁实现-学习讲义.pdf) |
| 4.1 多层感知机 | 14 页 | [学习讲义 PDF](output/pdf/4.1-多层感知机-学习讲义.pdf) |
| 4.2 MLP从零实现 | 13 页 | [学习讲义 PDF](output/pdf/4.2-多层感知机的从零开始实现-学习讲义.pdf) |
| 4.3 MLP简洁实现 | 11 页 | [学习讲义 PDF](output/pdf/4.3-多层感知机的简洁实现-学习讲义.pdf) |
| 4.4 模型选择、欠拟合和过拟合 | 14 页 | [学习讲义 PDF](output/pdf/4.4-模型选择、欠拟合和过拟合-学习讲义.pdf) |
| 4.5 权重衰减 | 15 页 | [学习讲义 PDF](output/pdf/4.5-权重衰减-学习讲义.pdf) |
| 4.6 暂退法（Dropout） | 15 页 | [学习讲义 PDF](output/pdf/4.6-暂退法（Dropout）-学习讲义.pdf) |
| 4.7 前向传播、反向传播和计算图 | 16 页 | [学习讲义 PDF](output/pdf/4.7-前向传播、反向传播和计算图-学习讲义.pdf) |
| 4.8 数值稳定性和模型初始化 | 16 页 | [学习讲义 PDF](output/pdf/4.8-数值稳定性和模型初始化-学习讲义.pdf) |
| 4.9 环境和分布偏移 | 18 页 | [学习讲义 PDF](output/pdf/4.9-环境和分布偏移-学习讲义.pdf) |
| 4.10 实战Kaggle比赛：预测房价 | 19 页 | [学习讲义 PDF](output/pdf/4.10-实战Kaggle比赛：预测房价-学习讲义.pdf) |
| 5.1 层和块 | 14 页 | [学习讲义 PDF](output/pdf/5.1-层和块-学习讲义.pdf) |
| 5.2 参数管理 | 14 页 | [学习讲义 PDF](output/pdf/5.2-参数管理-学习讲义.pdf) |

二十六份包含原理、代码示例、结果解释与综合练习，嵌入中文字体并提供章节书签和页码。2.2 附有预处理数据；2.3–2.5 配有原理图及模型计算；2.6 配有频率收敛、贝叶斯更新、条件独立和均值方差四张图，贯通抽样与条件判断。2.7 通过查阅工具、签名解释和最小实验贯通整节。3.1 配有训练、正态分布、损失与网络四张图，并对照最小二乘和梯度下降。3.2 配有尾批、训练与学习率三张图，逐步组织生成器和手写 SGD。3.3 配有训练、实现对照与离群值三张图，贯通 DataLoader、Linear、MSELoss、SGD；同批对照核对损失尺度，并说明教材 Huber 公式与 SmoothL1Loss 的关系。3.4 配有得分与概率、交叉熵与温度、训练、分类区域与测试计数四张图，覆盖标签、梯度、信息论及全部节末拓展。3.5 配有十类样本、像素与 Resize、批次形状及性能对照四张图，使用本机已有 Fashion-MNIST 完成真实读取。3.6 配有训练、正确/错误预测、混淆矩阵与权重四张图，手写稳定交叉熵、SGD、累加统计和完整真实图像训练。3.7 配有组件、正式训练、调参与延长轮数四图，对应 Flatten、Linear、CrossEntropyLoss、SGD；附同起点逐步等价核对及教材两道练习。4.1 配有网络、激活及梯度、训练、分类区域、分段仿射五图，含三模型XOR对照和全部教材练习，区分通用近似、优化和泛化。4.2配有前向、真实训练、预测及子集搜索四图，手写Parameter模型与max-ReLU，核对原点约定并完成九组/18检查点有限搜索。4.3配有模块、正式训练、三类实验与初始化尺度四图；激活对照共用完全相同初值，九方案限定子集比较深度/学习率、激活和初始化。4.4配有数据职责与五折、SGD、拟合曲线、次数/数据量、缩放五图；区分容量、优化与数值问题，覆盖全部四道教材练习。4.5配有惩罚几何、正式训练、λ扫描和偏置分组四图；对齐半MSE/完整MSE、核对四种SGD写法，补充L1、Frobenius、MAP先验与全部六题。4.6配有掩码、条件矩、正式训练、子集比较与激活方差五图；覆盖模式切换、十一方案、两种替代噪声及全部七题。4.7配有无偏置/隐藏偏置计算图、梯度分支、三路径训练、保存内存与Hessian六图；手写反传、autograd和框架SGD同起点对齐，覆盖全部五题。4.8配有标量链、矩阵连乘、逐层尺度、初始化API、对称/LARS、真实训练六图；区分方差与二阶矩、Xavier折中与稳定保证，完整覆盖四题。4.9配有协变量模型、域密度比、风险/ESS、标签比例、概念/在线、反馈/分组六图；涵盖三类偏移、带域先验修正的加权训练、标签矩阵归一化及全部四题。4.10配有流程、价格、预处理、五折比较、训练、预测和非随机缺失七图；真实数据折内预处理、八候选/40折训练、完整预测导出与六题一并准备。5.1配有块结构、注册、动态计算、独立/共享实例与训练五图；覆盖自定义、顺序、平行、嵌套、buffer和三道教材练习。5.2配有参数路径、初始化分布、共享梯度、训练及参数/梯度范数五图；覆盖快照、嵌套参数、内置/自定义初始化、绑定训练与全部四题。5.2整理日期：2026-10-29，旧资料日期保留。

PDF 由 `tools/build_study_pdfs.py` 使用现有文档工具环境生成，不需要修改 openmmlab。可用 `--section 5.2` 只重建当前讲义。原始教材仍不纳入 Git；只允许上述学习 PDF 同步。

2.1–2.7 PDF原位置的本地删除均保留、不提交，GitHub存档仍在。2.6/2.7在4.5制作期间移入既有output/pdf/D归档，字节与移动前一致；旧删除的检测历史见AGENTS.md。已有4.3代码修改、2.7帮助记录修改和未跟踪02_whole_process.py均保留，不纳入本节提交。

## 在 VS Code 中开始

1. 用 VS Code 打开本目录中的 `d2l-learning.code-workspace`。
2. 打开 `lessons/ch05/02_parameter_management.py`。
3. 按 **F5**，选择“5.2 参数管理：完整演示”，一次运行整节。也可以点编辑器右上角“运行 Python 文件”。
4. 在 `exercises/ch05/02_parameter_management.py` 集中补齐八处TODO：共享层、Xavier初始化、读取参数名、独立权重快照、SGD、MSE、反向传播和参数更新。32训练/16验证、250次全批量更新，反馈参数与损失，再集中解释三个原理问题。

本节CPU/float64、小型人工数据，无下载或安装。基础4→8→1共49参数，嵌套四块309参数；检查Parameter、grad累积/清理、detach视图和独立快照。普通state_dict变量仍引用模型状态，独立快照需复制。教材.data改写使用no_grad保留对象身份。

保存默认、正态.01、全1、局部Xavier/42和自定义五种初始化状态。自定义[-10,10]阈值置零，20000值的负/零/正频率.2481/.5012/.2507。常量42与大方差自定义用于说明接口，不用于正式训练。

共享MLP 4→8→8→8→1，中间同一层使用两次，121不同参数/6个Parameter张量、8条状态路径；独立两层则193参数。同点共享梯度等于两次贡献之和。64/32/32人工回归、SGD .02、300整批更新，训练MSE.872987→.080923、验证.216725、最后测试.184329。保留1800条逐参数梯度/范数记录，独立NumPy重放和121坐标差分通过。详见[验证记录](reports/ch05/5_2_verification.md)。

4.10的1459行[房价预测CSV](reports/ch04/4_10_submission.csv)继续保留，未上传Kaggle，线上分数未知。旧讲义、代码和练习继续保留；单独旧tensor/reshape文件不重建。问答与实际学习状态在learning_log.md，未提交的综合作答不标记完成。

解释器已经指定为已有的 `openmmlab` 环境：`D:/Anaconda/envs/openmmlab/python.exe`。第一次打开若提示选择解释器，选择 `openmmlab`。

本机 Fashion-MNIST 原始文件已核对官方归档 MD5、解压内容 SHA-256 和 IDX 头。项目 `data/` 中的硬链接与已有文件共享存储，不额外复制约 52.4 MiB 原始像素文件；这些文件只用于读取。机器相关来源保存在被忽略的本地记录中，数据不会提交 GitHub。其他电脑需要先准备自己的数据；`tools/prepare_fashion_mnist.py --source <已有数据根目录>` 可校验并在同一文件系统建立硬链接。

默认读取使用 `num_workers=0`。3.5 演示另测 0/2/4 个加载进程与批量大小，Windows 入口已加 main 保护；重复运行 3.5 可用 `--skip-worker-benchmark`。4.1先用XOR解释原理，4.2/4.3使用真实图像；像素在内存做固定转换，原始文件不改写。4.3当时的正式训练与逐轮评估约15.74秒，额外实验另计；4.4/4.5使用人工回归数据，4.6再回到已有图像分类，4.7用人工分类逐项检查反向传播，4.8继续用人工分类比较初始化，4.9使用人工跨域回归、分类和在线数据，4.10使用真实房价表格。图像示例来自 Fashion-MNIST，许可保存在 `docs/references/fashion_mnist_LICENSE.txt`。

## 文件各自做什么

| 文件或目录 | 用途 |
| --- | --- |
| `docs/ch05/02_parameter_management.md` | 完整的5.2讲义 |
| `lessons/ch05/02_parameter_management.py` | 当前统一演示入口 |
| `exercises/ch05/02_parameter_management.py` | 当前综合练习入口 |
| `datasets/ch02/`、`datasets/ch03/`、`datasets/ch04/` | 预处理、回归、分类及XOR人工训练/验证/测试CSV |
| `data/FashionMNIST/raw/` | 本地官方图像数据，已排除 Git；四个硬链接复用现有存储 |
| `reports/ch02/figures/`、`reports/ch03/figures/`、`reports/ch04/figures/`、`reports/ch05/figures/` | 已讲章节的原理图和实验曲线 |
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

后续每次以完整小节或较大的主题块推进，集中讲原理、整合程序、统一实践和反馈，不再将单个函数拆成多轮。下一块为5.3延后初始化，理解参数形状推断与首次前向的作用。旧练习的未验证状态保留，教师运行不记作用户掌握。

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
