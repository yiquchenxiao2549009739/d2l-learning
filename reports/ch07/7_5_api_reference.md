# 7.5 官方API核对

2026-11-25，使用与本机torch2.4.1相应的PyTorch2.4文档。三个页面实际获取成功，标题、解析地址、SHA-256与关键词检查见7_5_api_sources.json；完整HTML只保留在本地tmp。

| API | 输入 | 每通道统计元素 |
| --- | --- | --- |
| [BatchNorm1d](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BatchNorm1d.html) | (N,C)或(N,C,L) | N或N×L |
| [BatchNorm2d](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BatchNorm2d.html) | (N,C,H,W) | N×H×W |
| [BatchNorm3d](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BatchNorm3d.html) | (N,C,D,H,W) | N×D×H×W |

训练前向使用有偏方差，移动方差更新使用无偏估计；eps在平方根内。momentum默认.1表示新统计的更新权重，与优化器动量不同。affine控制可学习gamma/beta；track_running_stats=False时训练和eval均使用当前批统计。momentum=None采用累计平均，本节手写类只实现固定更新权重，不冒称覆盖全部框架选项。

本节ManualBatchNorm仅接受2维/4维，参数形状(C,)，广播到输入。weight对应gamma，bias对应beta；running_mean、running_var、num_batches_tracked注册为buffer，随to/double迁移并进入state_dict。
