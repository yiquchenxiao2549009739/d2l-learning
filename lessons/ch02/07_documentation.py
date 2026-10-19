"""2.7 查阅文档：发现名字 -> 读帮助 -> 拆参数 -> 用小实验验证。

复用现有环境，离线可运行；完整本机帮助保存到 reports/ch02。
"""

import inspect
import io
from contextlib import redirect_stdout
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "ch02"


def local_help(obj):
    """捕获 help 的打印内容，让脚本运行时不会进入交互式分页器。"""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        returned = help(obj)
    text = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines())
    return text.rstrip() + "\n", returned


def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    print("2.7 查阅文档 | 本机 PyTorch:", torch.__version__)

    print("\n1. dir 找名字：这里只筛选本节关心的候选项")
    names = dir(torch.distributions)
    public_names = [name for name in names if not name.startswith("_")]
    candidates = [name for name in public_names if name in
                  ("Categorical", "Multinomial", "Normal", "Uniform")]
    print("候选名字:", candidates)
    print("dir 返回名字列表，不能代替参数和返回值说明。")

    print("\n2. help 读本机文档：传函数本身，不先调用函数")
    help_parts = []
    for obj in (torch.ones, torch.sum, torch.max):
        text, returned = local_help(obj)
        help_parts.append(text)
        print("帮助对象:", obj.__name__, "| help 返回:", returned)
        assert returned is None
    print("ones 文档第一条签名:", torch.ones.__doc__.strip().splitlines()[0])

    print("\n3. 把签名转成一次真实调用")
    ones = torch.ones(2, 3, dtype=torch.float64,
                      device="cpu", requires_grad=True)
    print(ones)
    print("shape:", tuple(ones.shape), "dtype:", ones.dtype,
          "device:", ones.device, "requires_grad:", ones.requires_grad)
    print("元组形状写法一致:", torch.equal(ones, torch.ones((2, 3), dtype=torch.float64)))

    print("\n4. 最小实验：验证 dim 和 keepdim")
    x = torch.tensor([[2., 4., 6.], [3., 6., 9.]], dtype=torch.float64)
    total = torch.sum(x)
    row_sums = torch.sum(x, dim=1)
    kept_sums = torch.sum(x, dim=1, keepdim=True)
    normalized = x / kept_sums
    for name, value in (("输入", x), ("全部求和", total),
                        ("每行求和", row_sums), ("保留轴", kept_sums),
                        ("按行归一化", normalized)):
        print(name, "=", value, "| shape:", tuple(value.shape))
    print("归一化后每行和:", normalized.sum(dim=1))
    try:
        x / row_sums
    except RuntimeError as error:
        print("预期捕获的形状错误:", str(error).splitlines()[0])
    else:
        raise AssertionError("这个 (2,3)/(2,) 对照应触发形状错误")

    print("\n5. 读报错：回到签名检查参数名")
    try:
        torch.sum(x, dim=1, keep_dim=True)
    except TypeError as error:
        print("预期捕获:", type(error).__name__, str(error).splitlines()[0])
    else:
        raise AssertionError("拼错的 keep_dim 应触发参数错误")
    print("本机文档的参数名是 keepdim。程序已用正确写法继续运行。")

    print("\n6. 来源边界：不是每个函数都能显示 Python 源码")
    try:
        inspect.getsource(torch.ones)
    except (TypeError, OSError) as error:
        print("torch.ones 源码读取:", type(error).__name__)
    print("本机 help 与最小实验可离线使用；网页 stable 版本会更新。")

    REPORTS.mkdir(parents=True, exist_ok=True)
    help_path = REPORTS / "2_7_local_help.txt"
    help_path.write_text("\n\n".join(help_parts).rstrip() + "\n", encoding="utf-8")
    print("完整本机帮助:", help_path.relative_to(ROOT).as_posix())
    print("\n接下来打开 exercises/ch02/07_documentation.py，集中完成三处 TODO。")
    return {"candidates": candidates, "ones": ones, "x": x,
            "total": total, "row_sums": row_sums,
            "kept_sums": kept_sums, "normalized": normalized}


if __name__ == "__main__":
    main()
