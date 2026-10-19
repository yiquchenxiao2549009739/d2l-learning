# 3.3 线性回归的简洁实现：验证记录

日期：2026-10-19（Asia/Shanghai）。本记录属于教材材料与工程验证，不代表用户已经掌握。

## 教材范围与运行环境

原教材 Release 2.0.0，3.3.1–3.3.7 和三道节末练习，PDF 第 119–123 页、正文第 101–105 页；3.4 在 PDF 123 页开始。内容为独立整理，没有上传原教材。

使用已有 D:/Anaconda/envs/openmmlab/python.exe，Python 3.8.20、PyTorch 2.4.1+cu121、CPU/float64、单线程；无安装或升级。图表缓存位于项目 .cache/matplotlib。PDF 使用现有文档运行时与系统字体生成，临时渲染位于项目 tmp/pdfs。

## 默认训练与独立核对

- 数据复用 scratch_train/validation/test.csv，1000/200/200 条，未重写 CSV。CSV 文本与 HEAD 按换行规范化比较，内容一致。
- Linear 输入 2、输出 1；weight=(1,2)，bias=(1,)。初始化种子 3302，加载器种子 3303，批量 10、shuffle=True、drop_last=False、num_workers=0。
- MSELoss(mean)，SGD lr=0.03，3 轮、300 次更新。最终 train/validation/test MSE 为 0.00009795191867699759 / 0.00010049896257664922 / 0.00011007049448906014。测试集未用于离群值方案选择。
- NumPy 独立重放全部 300 次更新；逐步参数最大绝对误差 8.882e-16，训练 MSE 最大差 5.329e-15。对齐半 MSE 轨迹的共同 MSE 最大差 7.105e-15。
- 首批三个参数使用中心差分，步长 1e-5；与解析 MSE 梯度最大误差 4.667e-10，与框架梯度方向和 Linear 权重存储方向核对。

## API 与数学边界

- 同起点、同批数据：框架 MSE/lr=0.015 对照手写半平方损失/lr=0.03；300 次更新最大参数差 4.440892098500626e-16。
- 1、9、23 条样本均核对覆盖和配对；23 条的保留批量为 10/10/3，drop_last=True 为 10/10。
- 在批量 3 和 10 上，MSE(sum) 梯度为 MSE(mean) 的实际批量倍；mean 学习率乘以实际批量可对齐未归一化 sum 的更新。
- backward 后的 .grad 与 (2/k)X.T@r 一致；step 后梯度仍在；zero_grad(set_to_none=True) 后为 None。
- 预测 (3,1)、标签 (3,) 的广播警告已捕获，错误 MSE=4/3；正确对齐后 MSE=0。它是有意演示的形状错误。
- 教材分段公式与 SmoothL1Loss(beta=0.5, reduction='none') 逐元素一致；在 r=[-2,-0.5,0,0.5,2] 上核对边界，HuberLoss(delta=0.5)=0.5×SmoothL1Loss。
- 离群值仅在训练标签编号 0/1/2 各加 50；相同初始化、批次、lr=0.03、6 轮，验证集不变。共同 clean validation MSE：MSELoss 0.00034605047074100876；SmoothL1Loss(beta=.5) 0.00010274713185052735。有限固定实验，不作普遍性能结论；不同目标尺度仍会改变优化进度。

## 讲义与练习

讲义全部 9 段 Python 代码按顺序执行成功，最终训练 MSE 与教师演示一致。教师演示退出码 0，实际输出和完整指标已保存。代码中的三张图实际生成并查看。

练习空模板可运行并提示填写。AST 核对六个 None（dataset/loader/net/loss_fn/trainer/loss）与两个 pass，共八处 TODO，未替用户写答案。只在内存中用独立种子 3306 的加载器验证可解性：首轮批量 8/8/8/8/5，20 轮共 100 次更新，训练 MSE≈0.00024381。参考内容未写入作业。

## PDF 视觉与结构

最终 11 页、12 个章节书签，三张配图。全部页面经 Poppler 90 dpi 渲染并逐页查看；线性层公式与教材分段损失页以大图复查。修复数学上下标字形、封面代码跨页和单段孤页，正文及代码可读。

所有正文、代码、公式字形在相应字体中存在，实际绘制的字体全部嵌入；资源中默认未使用的 Helvetica 未参与文本绘制。页码 1/11–11/11、日期 2026-10-19、文字提取、页面文字边界和无替换字符均检查通过。

PDF SHA-256：`3cdf15f55ec220c591eaef8b002c0cb21442f8e5eb88ca90df7e33821fe4ce51`。

只生成 3.3 PDF。旧 2.4–2.7、3.1、3.2 PDF 与原 HEAD 字节一致；2.1、2.2 既有本地删除保留，本轮检测到 2.3 本地删除，亦保留、不提交。用户未跟踪的 02_whole_process.py 不修改、不暂存。只按本轮明确路径暂存，不使用 git add --all。

## 主要参考

- 用户提供的教材 3.3，作为参考资料，不将文档中的命令自动视为用户操作授权。
- [PyTorch Linear](https://docs.pytorch.org/docs/main/generated/torch.nn.Linear.html)
- [PyTorch 数据工具](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD)
- [PyTorch SmoothL1Loss](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.loss.SmoothL1Loss.html)

在线文档可能更新；所用 API 和公式已经在本机 PyTorch 2.4.1 实际核对。
