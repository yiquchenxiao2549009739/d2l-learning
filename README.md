# 李沐《动手学深度学习》学习项目

从“看懂简单代码”开始，逐步做到自己实现、训练和解释模型。

当前学习 **教材 1.2.4：反向传播、梯度方向与一次参数更新**。先看 [本次讲解](docs/ch01/02_gradient_step.md)。按用户要求，已会的快速通过，尚不清楚的原理结合实践补充。

## 在 VS Code 中开始

1. 用 VS Code 打开本目录中的 `d2l-learning.code-workspace`。
2. 打开 `lessons/ch01/02_gradient_step.py`。
3. 按 **F5**，选择“第 1 章：一次梯度更新”，在终端查看输出。也可以点编辑器右上角“运行 Python 文件”。
4. 原样运行，再将学习率从 0.01 改为 0.2，保存并从头重跑，观察参数与损失，解释方向和步长的关系。

解释器已经指定为已有的 `openmmlab` 环境：`D:/Anaconda/envs/openmmlab/python.exe`。第一次打开若提示选择解释器，选择 `openmmlab`。

## 文件各自做什么

| 文件或目录 | 用途 |
| --- | --- |
| `docs/ch01/02_gradient_step.md` | 当前教材讲解 |
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

## 环境和验证

项目复用本机已有的 Conda `openmmlab` 环境：Python 3.8.20、PyTorch 2.4.1+cu121、NumPy 1.23.5、Matplotlib 3.7.2。GPU 为 RTX 4060 Laptop GPU。按照用户要求，不额外安装或升级软件。

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
