# 李沐《动手学深度学习》学习项目

从“看懂简单代码”开始，逐步做到自己实现、训练和解释模型。

当前一次学习 **完整的 2.3 线性代数（2.3.1–2.3.11）**。先看 [整节讲义](docs/ch02/03_linear_algebra.md)，运行一个完整演示，再完成一个综合练习并集中反馈。

## 阅读版 PDF

| 小节 | 页数 | 文件 |
| --- | --- | --- |
| 2.1 数据操作 | 8 页 | [学习讲义 PDF](output/pdf/2.1-数据操作-学习讲义.pdf) |
| 2.2 数据预处理 | 7 页 | [学习讲义 PDF](output/pdf/2.2-数据预处理-学习讲义.pdf) |
| 2.3 线性代数 | 13 页 | [学习讲义 PDF](output/pdf/2.3-线性代数-学习讲义.pdf) |

三份均包含原理、代码示例、结果解释、综合练习和速查表，嵌入中文字体并提供章节书签和页码。2.2 附有完整流程代码及六行扩展练习数据；2.3 配有点积、矩阵乘法、范数三张图，以及转置证明和小批量预测完整代码，可离线阅读。整理日期：2026-10-15。

PDF 由 `tools/build_study_pdfs.py` 使用现有文档工具环境生成，不需要修改 openmmlab。可用 `--section 2.3` 只重建当前讲义。原始教材仍不纳入 Git；只允许上述三份学习 PDF 同步。

## 在 VS Code 中开始

1. 用 VS Code 打开本目录中的 `d2l-learning.code-workspace`。
2. 打开 `lessons/ch02/03_linear_algebra.py`。
3. 按 **F5**，选择“2.3 线性代数：完整演示”，一次运行整节。也可以点编辑器右上角“运行 Python 文件”。
4. 在 `exercises/ch02/03_linear_algebra.py` 集中填写十个表达式，运行后一次反馈结果和三个概念问题。

这块覆盖标量、向量、矩阵与高阶张量，转置与对称、按元素运算、沿轴求和、keepdim 与累积和、点积、矩阵乘法、L1/L2/Frobenius 范数，以及完整的小批量预测和误差计算。沿用 float32，重点核对各轴的含义。

完整 2.1、2.2 的资料继续保留，单独的旧 tensor、reshape 示例与旧讲义已经删除。问答与已完成实践的记录在 `learning_log.md`。用户主动要求进入 2.3，未据此将未提交的 2.1、2.2 综合练习标记完成。

解释器已经指定为已有的 `openmmlab` 环境：`D:/Anaconda/envs/openmmlab/python.exe`。第一次打开若提示选择解释器，选择 `openmmlab`。

## 文件各自做什么

| 文件或目录 | 用途 |
| --- | --- |
| `docs/ch02/03_linear_algebra.md` | 完整的 2.3 讲义 |
| `lessons/ch02/03_linear_algebra.py` | 当前统一演示入口 |
| `exercises/ch02/03_linear_algebra.py` | 当前综合练习入口 |
| `datasets/ch02/` | 2.2 的小型人工 CSV 数据 |
| `reports/ch02/figures/` | 2.3 原理配图 |
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

后续每次以完整小节或较大的主题块推进，集中讲原理、整合程序、统一实践和反馈，不再将单个函数拆成多轮。下一块为教材 2.4 微积分。

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
