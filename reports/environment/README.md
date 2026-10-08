# 已有环境的小型训练验证

2026-10-08，在本机已有 Conda openmmlab 环境运行 `tools/smoke_train.py`，分别验证 CPU 和 RTX 4060 GPU。

这是工程验证，用户尚未学习或亲手实现该训练过程。环境与训练结果的解释见 [验证报告](../../docs/environment_check.md)。

- `cpu_metrics.json`、`cuda_metrics.json`：参数与损失。
- `cpu_loss.csv`、`cuda_loss.csv`：每次参数更新前的损失记录。
- `cpu_loss.png`、`cuda_loss.png`：训练曲线。

两个设备均从约 30.053 的 MSE 收敛到约 0.000109，参数符合预设的合成数据关系。这些小型结果用于复盘，完整模型位于本地 `artifacts/`。
