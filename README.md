# 李沐《动手学深度学习》学习项目

从“看懂简单代码”开始，逐步做到自己实现、训练和解释模型。

当前一次学习 **完整的 3.2 线性回归的从零开始实现（3.2.1–3.2.7）**。先看 [整节讲义](docs/ch03/02_linear_regression_scratch.md)，运行一个完整演示，再完成一个综合练习并集中反馈。

## 阅读版 PDF

| 小节 | 页数 | 文件 |
| --- | --- | --- |
| 2.1 数据操作 | 8 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.1-数据操作-学习讲义.pdf) |
| 2.2 数据预处理 | 7 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.2-数据预处理-学习讲义.pdf) |
| 2.3 线性代数 | 13 页 | [学习讲义 PDF](output/pdf/2.3-线性代数-学习讲义.pdf) |
| 2.4 微积分 | 13 页 | [学习讲义 PDF](output/pdf/2.4-微积分-学习讲义.pdf) |
| 2.5 自动微分 | 13 页 | [学习讲义 PDF](output/pdf/2.5-自动微分-学习讲义.pdf) |
| 2.6 概率 | 15 页 | [学习讲义 PDF](output/pdf/2.6-概率-学习讲义.pdf) |
| 2.7 查阅文档 | 6 页 | [学习讲义 PDF](output/pdf/2.7-查阅文档-学习讲义.pdf) |
| 3.1 线性回归 | 14 页 | [学习讲义 PDF](output/pdf/3.1-线性回归-学习讲义.pdf) |
| 3.2 从零实现线性回归 | 11 页 | [学习讲义 PDF](output/pdf/3.2-线性回归的从零开始实现-学习讲义.pdf) |

九份包含原理、代码示例、结果解释与综合练习，嵌入中文字体并提供章节书签和页码。2.2 附有预处理数据；2.3–2.5 配有原理图及模型计算；2.6 配有频率收敛、贝叶斯更新、条件独立和均值方差四张图，贯通抽样与条件判断。2.7 通过查阅工具、签名解释和最小实验贯通整节。3.1 配有训练、正态分布、损失与网络四张图，并对照最小二乘和梯度下降。3.2 配有尾批、训练与学习率三张图，逐步组织生成器和手写 SGD。3.2 整理日期：2026-10-19，旧资料日期保留。

PDF 由 `tools/build_study_pdfs.py` 使用现有文档工具环境生成，不需要修改 openmmlab。可用 `--section 3.2` 只重建当前讲义。原始教材仍不纳入 Git；只允许上述学习 PDF 同步。

2.1 PDF 保持既有本地删除状态；本轮制作期间又检测到 2.2 PDF 本地删除。GitHub 均保留，因此表中提供存档链接。本轮不恢复文件，也不把这些本地删除提交到远端。

## 在 VS Code 中开始

1. 用 VS Code 打开本目录中的 `d2l-learning.code-workspace`。
2. 打开 `lessons/ch03/02_linear_regression_scratch.py`。
3. 按 **F5**，选择“3.2 从零实现线性回归：完整演示”，一次运行整节。也可以点编辑器右上角“运行 Python 文件”。
4. 在 `exercises/ch03/02_linear_regression_scratch.py` 集中填写六处 TODO，在 23 条样本上完成分批索引、逐样本损失、手写 SGD、预测、求导与调用优化器，再统一解释 yield、尾批和梯度尺度。

这块覆盖人工数据、打乱与配对、生成器和 yield、叶参数初始化、模型与逐样本损失、sum/mean 的配套归一化、实际尾批大小、手写 SGD、完整 epoch/step 训练、学习率与零初始化对照。核心仅使用张量和自动求导。

前面各节的代码、讲义和练习继续保留，2.1、2.2 PDF 的本地状态见上文；单独的旧 tensor、reshape 文件已删除且未重建。问答与已完成实践的记录在 `learning_log.md`。用户主动要求进入 3.2，未据此将未提交的旧综合练习标记完成。

解释器已经指定为已有的 `openmmlab` 环境：`D:/Anaconda/envs/openmmlab/python.exe`。第一次打开若提示选择解释器，选择 `openmmlab`。

## 文件各自做什么

| 文件或目录 | 用途 |
| --- | --- |
| `docs/ch03/02_linear_regression_scratch.md` | 完整的 3.2 讲义 |
| `lessons/ch03/02_linear_regression_scratch.py` | 当前统一演示入口 |
| `exercises/ch03/02_linear_regression_scratch.py` | 当前综合练习入口 |
| `datasets/ch02/`、`datasets/ch03/` | 预处理样例与线性回归人工训练/验证/测试 CSV |
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

后续每次以完整小节或较大的主题块推进，集中讲原理、整合程序、统一实践和反馈，不再将单个函数拆成多轮。下一块为 3.3 线性回归的简洁实现，使用框架提供的数据工具、层、损失与优化器组织同一流程。

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
