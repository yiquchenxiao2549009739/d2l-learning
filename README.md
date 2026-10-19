# 李沐《动手学深度学习》学习项目

从“看懂简单代码”开始，逐步做到自己实现、训练和解释模型。

当前一次学习 **完整的 2.5 自动微分（2.5.1–2.5.4）**。先看 [整节讲义](docs/ch02/05_autograd.md)，运行一个完整演示，再完成一个综合练习并集中反馈。

## 阅读版 PDF

| 小节 | 页数 | 文件 |
| --- | --- | --- |
| 2.1 数据操作 | 8 页 | [GitHub 存档 PDF](https://github.com/yiquchenxiao2549009739/d2l-learning/blob/main/output/pdf/2.1-数据操作-学习讲义.pdf) |
| 2.2 数据预处理 | 7 页 | [学习讲义 PDF](output/pdf/2.2-数据预处理-学习讲义.pdf) |
| 2.3 线性代数 | 13 页 | [学习讲义 PDF](output/pdf/2.3-线性代数-学习讲义.pdf) |
| 2.4 微积分 | 13 页 | [学习讲义 PDF](output/pdf/2.4-微积分-学习讲义.pdf) |
| 2.5 自动微分 | 13 页 | [学习讲义 PDF](output/pdf/2.5-自动微分-学习讲义.pdf) |

五份均包含原理、代码示例、结果解释、综合练习和速查表，嵌入中文字体并提供章节书签和页码。2.2 附有预处理数据；2.3、2.4 配有线性代数与微积分原理图；2.5 展示自动求导、梯度累积、计算图寿命、detach、控制流和一次完整更新。2.5 整理日期：2026-10-19，旧资料日期保留。

PDF 由 `tools/build_study_pdfs.py` 使用现有文档工具环境生成，不需要修改 openmmlab。可用 `--section 2.5` 只重建当前讲义。原始教材仍不纳入 Git；只允许上述学习 PDF 同步。

本轮开始时，2.1 PDF 已有本地删除状态，GitHub 仍保留，因此表中提供存档链接。本轮未恢复文件，也未把这个既有删除提交到远端。

## 在 VS Code 中开始

1. 用 VS Code 打开本目录中的 `d2l-learning.code-workspace`。
2. 打开 `lessons/ch02/05_autograd.py`。
3. 按 **F5**，选择“2.5 自动微分：完整演示”，一次运行整节。也可以点编辑器右上角“运行 Python 文件”。
4. 在 `exercises/ch02/05_autograd.py` 集中填写八处 TODO，使用自动求导完成一次参数更新，再一次反馈结果和三个原理问题。

这块覆盖 requires_grad、叶张量、backward、梯度累积与清理、计算图缓存、多元素输出的汇总、detach 与 no_grad、Python 控制流、二阶导数，并用自动求导重现 2.4 的小批量更新。继续使用 float64，梯度与手算和数值差分交叉核对。

前面各节的代码、讲义和练习继续保留，2.1 PDF 的本地状态见上文；单独的旧 tensor、reshape 文件已删除且未重建。问答与已完成实践的记录在 `learning_log.md`。用户主动要求进入 2.5，未据此将未提交的旧综合练习标记完成。

解释器已经指定为已有的 `openmmlab` 环境：`D:/Anaconda/envs/openmmlab/python.exe`。第一次打开若提示选择解释器，选择 `openmmlab`。

## 文件各自做什么

| 文件或目录 | 用途 |
| --- | --- |
| `docs/ch02/05_autograd.md` | 完整的 2.5 讲义 |
| `lessons/ch02/05_autograd.py` | 当前统一演示入口 |
| `exercises/ch02/05_autograd.py` | 当前综合练习入口 |
| `datasets/ch02/` | 2.2 的小型人工 CSV 数据 |
| `reports/ch02/figures/` | 2.3–2.5 原理配图 |
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

后续每次以完整小节或较大的主题块推进，集中讲原理、整合程序、统一实践和反馈，不再将单个函数拆成多轮。下一块为教材 2.6 概率。

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
