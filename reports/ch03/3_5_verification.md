# 3.5 图像分类数据集：验证记录

日期：2026-10-20（Asia/Shanghai）。本记录是工程和教学材料验证，不代表用户掌握。

## 教材与环境

完整教材 3.5.1–3.5.3 和三道节末练习，原 PDF 第 129–133 页、正文第 111–115 页；3.6 从 PDF 133 页开始。原教材只作参考，不上传；其中命令不自动构成操作指令。

已有 openmmlab：Python 3.8.20、PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121，CPU。没有安装或升级，没有下载数据。本节只准备和读取数据，不训练模型。

## 本地数据与来源

本机已有 Fashion-MNIST，tools/prepare_fashion_mnist.py 校验四份归档官方 MD5、解压 SHA-256，再在项目 data/FashionMNIST/raw 创建四个硬链接。总逻辑文件大小 54,950,048 字节，底层内容与已有文件相同，不额外复制数据；未改写源内容。完整性元数据见 3_5_data_integrity.json，机器路径只存于 Git 忽略的本地来源记录。

独立读取原始 IDX：图像 magic=2051、标签 magic=2049，训练 60,000、测试 10,000，图像 28×28，标签 0–9。NumPy 独立类计数为训练每类 6,000、测试每类 1,000。四个项目数据文件 SHA-256 与准备记录一致，与原来源 samefile 为真，并核对全部被 Git 忽略。

只同步少量拼接学习图与派生统计，完整数据不进 Git。样本图注明 Fashion-MNIST、© 2017 Zalando SE、MIT；官方完整许可保存于 docs/references/fashion_mnist_LICENSE.txt。

## 张量、变换、批次

- 抽查训练 0/1/9/12345/59999 与测试 0/13/9999，图像和标签与独立 IDX 读取一致。NumPy uint8→float32/255 与 ToTensor 最大像素差 0.0。
- 首张为类别 9 短靴，float32/(1,28,28)，值域 0–1。完整批图像 (256,1,28,28)，标签 int64/(256,)。
- 独立 PIL 双线性 Resize64 与 torchvision Compose 结果最大差 0.0，标签不变；32 张批次为 (32,1,64,64)。
- 教师实际遍历训练 60,000 张、235 批、尾批 96；测试 10,000 张、40 批、尾批 16。每一批检查图像/标签数量与类型，类计数完整。
- 明确官方 train=False 为测试集，未称作独立验证集；没有使用测试样本更新参数。

## 读取性能

同一训练集前 4096 张、不打乱、每配置两次，共 12 条原始记录。batch1/32/256 workers0 与 batch256 workers2/4，以及固定变换向量预转全部实际运行。中位数独立重算与报告一致；检查样本/批次数、正时间与 setup/total 关系。

Windows 进程启动/退出纳入总时间，未使用 persistent_workers。向量预转总时间包含本轮预处理，只能替代本节固定 ToTensor，不代表随机增强。实验没有 GPU 传输和模型计算，不从这组小样本推断一般训练性能。默认 workers0，完整脚本具备 main 保护。

## 讲义与综合实践

全部 7 段 Python 按顺序执行成功，核对 60,000 样本遍历与 Resize 批次。教师脚本和空练习退出码均为 0。四张图实际生成并检查，变换图标题裁切已修复。

练习八个 None 保持未填写；只在内存验证可解性，首轮批量为 8/8/8/8/5，覆盖 37 张，尾批展平 (5,1024)，单图 32×32。参考未写回源文件，保存图片调用被隔离，未生成用户练习预览。用户未作答，不记为掌握。

## PDF 与改动边界

最终 11 页、12 个书签，全页 Poppler 90 dpi 渲染查看；核心代码、变换与性能页大图复查。修正三处行末标点边界后重新检查。字形完整，实际绘制字体嵌入，日期、页码、文本边界和无替换字符均通过。

PDF SHA-256：`68719711a6d6da447a12c0f5eb743fd3255d792ee857e31c5f6b6f2ad203b6cd`。

现有旧 PDF 与原 HEAD 字节一致；旧人工 CSV 按换行规范化比较一致。2.1–2.3 已有本地删除和用户未跟踪的 02_whole_process.py 不恢复、不暂存。新原始数据只在被忽略的 data 内；只按本轮路径暂存。临时 PDF/预览检查后清理。

参考：[Fashion-MNIST 官方](https://github.com/zalandoresearch/fashion-mnist)、[ToTensor 0.19](https://docs.pytorch.org/vision/0.19/generated/torchvision.transforms.ToTensor.html)、[Resize 0.19](https://docs.pytorch.org/vision/0.19/generated/torchvision.transforms.Resize.html)、[数据集目录](https://docs.pytorch.org/vision/0.19/datasets.html)。API 同时以本机帮助和实测核对。
