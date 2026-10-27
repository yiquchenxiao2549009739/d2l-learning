"""Build Chinese study booklets from the section notes.

Uses the existing PDF runtime (ReportLab / pypdf) and Windows fonts.
All generated output remains inside this project. No package installation.
"""

import argparse
import csv
import re
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, Image, KeepTogether,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf"
TEMP = ROOT / "tmp" / "pdfs"
DATE = "2026-10-15"
PAGE_W, PAGE_H = A4
LEFT = RIGHT = 46
TOP = 63
BOTTOM = 49
WIDTH = PAGE_W - LEFT - RIGHT
NAVY = colors.HexColor("#173448")
TEAL = colors.HexColor("#147E83")
INK = colors.HexColor("#233746")
MUTED = colors.HexColor("#657987")
LINE = colors.HexColor("#D7E2E9")
PALE = colors.HexColor("#F1F7F8")

pdfmetrics.registerFont(TTFont("CN", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("CN-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Mono", r"C:\Windows\Fonts\consola.ttf"))
pdfmetrics.registerFont(TTFont("MathSymbol", r"C:\Windows\Fonts\seguisym.ttf"))
pdfmetrics.registerFontFamily("CN", normal="CN", bold="CN-Bold", italic="CN", boldItalic="CN-Bold")

STYLES = {}
for name, settings in {
    "Body": dict(fontSize=10.6, leading=17.8, spaceAfter=8),
    "H2": dict(fontName="CN-Bold", fontSize=14, leading=21, spaceBefore=14, spaceAfter=8, keepWithNext=True, textColor=NAVY),
    "H3": dict(fontName="CN-Bold", fontSize=11.4, leading=18, spaceBefore=10, spaceAfter=6, keepWithNext=True),
    "Title": dict(fontName="CN-Bold", fontSize=28, leading=39, spaceAfter=8, textColor=NAVY),
    "Eyebrow": dict(fontName="CN-Bold", fontSize=10, leading=16, spaceAfter=9, textColor=TEAL),
    "Subtitle": dict(fontSize=12.2, leading=20, spaceAfter=14, textColor=MUTED),
    "Caption": dict(fontSize=8.5, leading=13.5, spaceAfter=8, textColor=MUTED),
    "Bullet": dict(fontSize=10.6, leading=17.8, spaceAfter=5, leftIndent=13, firstLineIndent=-10),
    "Lead": dict(fontSize=10.6, leading=17.8, spaceAfter=8, keepWithNext=True),
    "Cell": dict(fontSize=9.1, leading=14.5, spaceAfter=0),
    "CellHead": dict(fontName="CN-Bold", fontSize=9.1, leading=14.5, spaceAfter=0, textColor=colors.white),
    "Equation": dict(fontSize=12, leading=23, spaceBefore=5, spaceAfter=10, textColor=TEAL),
}.items():
    options = dict(fontName="CN", textColor=INK, alignment=TA_LEFT, wordWrap="CJK", allowWidows=0, allowOrphans=0)
    options.update(settings)
    STYLES[name] = ParagraphStyle(name, **options)


def normalize(text):
    return text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")


def code_runs(text):
    """Keep ASCII code monospaced while embedding Chinese comment glyphs."""
    runs = []
    for character in text:
        font = "MathSymbol" if character == "∇" else ("Mono" if ord(character) < 128 else "CN")
        if runs and runs[-1][0] == font:
            runs[-1] = (font, runs[-1][1] + character)
        else:
            runs.append((font, character))
    return runs


def inline(text):
    pieces = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", normalize(text))
    formatted = []
    for piece in pieces:
        if piece.startswith("`") and piece.endswith("`"):
            formatted.append("".join(
                f'<font name="{font}" size="9.1" color="#185A67">{escape(value)}</font>'
                for font, value in code_runs(piece[1:-1])
            ))
        elif piece.startswith("**") and piece.endswith("**"):
            formatted.append("<b>" + inline(piece[2:-2]) + "</b>")
        else:
            formatted.append(escape(piece).replace("∇", '<font name="MathSymbol">∇</font>'))
    return "".join(formatted)


def p(text, style="Body"):
    return Paragraph(inline(text), STYLES[style])


class CodeBlock(Flowable):
    def __init__(self, text, language="python", continued=False):
        super().__init__()
        self.lines = normalize(text).expandtabs(4).splitlines() or [""]
        self.language = language
        self.continued = continued
        self.spaceBefore = 3
        self.spaceAfter = 11
        self.font_size = 8.6
        self.line_height = 13.1

    def wrap(self, available_width, available_height):
        self.width = available_width
        max_width = available_width - 24
        for line in self.lines:
            line_width = sum(pdfmetrics.stringWidth(value, font, self.font_size) for font, value in code_runs(line))
            if line_width > max_width + 0.1:
                raise ValueError(f"Code line too wide ({line_width:.1f} > {max_width:.1f}): {line}")
        self.height = 35 + len(self.lines) * self.line_height
        return self.width, self.height

    def split(self, available_width, available_height):
        count = int((available_height - 35) // self.line_height)
        if count < 3 or len(self.lines) - count < 2:
            return []
        first = CodeBlock("\n".join(self.lines[:count]), self.language, self.continued)
        second = CodeBlock("\n".join(self.lines[count:]), self.language, True)
        return [first, second]

    def draw(self):
        canvas = self.canv
        canvas.setFillColor(colors.HexColor("#F5F7FA"))
        canvas.setStrokeColor(LINE)
        canvas.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=1)
        canvas.setFillColor(MUTED)
        canvas.setFont("CN", 7.6)
        label = "PYTHON" if self.language == "python" else "结果 / 结构"
        canvas.drawString(12, self.height - 15, label + ("  /  续" if self.continued else ""))
        y = self.height - 31
        for line in self.lines:
            x = 12
            for font, value in code_runs(line):
                canvas.setFont(font, self.font_size)
                canvas.setFillColor(colors.HexColor("#396C5B") if line.lstrip().startswith("#") else INK)
                canvas.drawString(x, y, value)
                x += pdfmetrics.stringWidth(value, font, self.font_size)
            y -= self.line_height


class Route(Flowable):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.spaceBefore = 5
        self.spaceAfter = 15

    def wrap(self, width, height):
        self.width = width
        self.height = 49
        return width, self.height

    def draw(self):
        canvas = self.canv
        count = len(self.labels)
        gap = 11
        box_w = (self.width - gap * (count - 1)) / count
        for index, label in enumerate(self.labels):
            x = index * (box_w + gap)
            canvas.setFillColor(PALE)
            canvas.setStrokeColor(LINE)
            canvas.roundRect(x, 4, box_w, 39, 5, stroke=1, fill=1)
            canvas.setFillColor(TEAL)
            canvas.setFont("CN-Bold", 9)
            canvas.drawCentredString(x + box_w / 2, 19, label)
            if index < count - 1:
                canvas.setStrokeColor(TEAL)
                canvas.line(x + box_w + 2, 23, x + box_w + gap - 2, 23)
                canvas.line(x + box_w + gap - 5, 26, x + box_w + gap - 2, 23)
                canvas.line(x + box_w + gap - 5, 20, x + box_w + gap - 2, 23)


def table(rows):
    count = len(rows[0])
    widths = [WIDTH * 0.40, WIDTH * 0.60] if count == 2 else [WIDTH / count] * count
    cells = [[p(cell, "CellHead" if row_i == 0 else "Cell") for cell in row] for row_i, row in enumerate(rows)]
    result = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
    ]))
    result.spaceBefore = 4
    result.spaceAfter = 12
    return result


def markdown_blocks(text, source_dir=None, figure_height=178):
    lines = text.splitlines()
    result = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- pagebreak -->":
            result.append(PageBreak())
            index += 1
            continue
        if line.startswith('<div class="equation">'):
            formula = line[len('<div class="equation">'):-len('</div>')]
            formula = formula.replace("∇", '<font name="MathSymbol">∇</font>')
            result.append(Paragraph(formula, STYLES["Equation"]))
            index += 1
            continue
        figure = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", line)
        if figure:
            if source_dir is None:
                raise ValueError("Image requires a document source directory")
            path = (source_dir / figure.group(2)).resolve()
            if ROOT.resolve() not in path.parents:
                raise ValueError("Figure must remain inside the project")
            item = Image(str(path))
            scale = min(WIDTH / item.imageWidth, figure_height / item.imageHeight)
            item.drawWidth = item.imageWidth * scale
            item.drawHeight = item.imageHeight * scale
            item.hAlign = "CENTER"
            result.append(KeepTogether([item, Spacer(1, 5), p(figure.group(1), "Caption")]))
            index += 1
            continue
        if line.startswith("```"):
            language = line[3:].strip()
            index += 1
            code = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            result.append(CodeBlock("\n".join(code), language))
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in row):
                    rows.append(row)
                index += 1
            grouped = []
            cursor = len(result) - 1
            while cursor >= 0 and len(grouped) < 3:
                previous = result[cursor]
                if not isinstance(previous, Paragraph):
                    break
                grouped.insert(0, previous)
                if previous.style.name == "H2":
                    del result[cursor:]
                    break
                cursor -= 1
            else:
                grouped = []
            if not grouped or grouped[0].style.name != "H2":
                grouped = []
            result.append(KeepTogether(grouped + [table(rows)]))
            continue
        if line.startswith("## "):
            result.append(p(line[3:], "H2"))
        elif line.startswith("### "):
            result.append(p(line[4:], "H3"))
        elif line.startswith("- ") or re.match(r"\d+\. ", line):
            items = []
            while index < len(lines):
                item = lines[index].strip()
                if not (item.startswith("- ") or re.match(r"\d+\. ", item)):
                    break
                items.append(p(item, "Bullet"))
                index += 1
            result.append(KeepTogether(items))
            continue
        else:
            paragraph = [line]
            while index + 1 < len(lines):
                following = lines[index + 1].strip()
                if not following or following.startswith(("#", "```", "|", "- ")) or re.match(r"\d+\. ", following):
                    break
                index += 1
                paragraph.append(following)
            text = " ".join(paragraph)
            result.append(p(text, "Lead" if text.endswith((":", "：")) else "Body"))
        index += 1
    return result


class StudyDoc(BaseDocTemplate):
    def __init__(self, path, spec, total_pages=None):
        super().__init__(str(path), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
                         topMargin=TOP, bottomMargin=BOTTOM, pageCompression=1,
                         title=f"{spec['section']} {spec['title']} - 学习讲义",
                         author="深度学习学习项目")
        self.spec = spec
        self.total_pages = total_pages
        self.outline_index = 0
        frame = Frame(LEFT, BOTTOM, WIDTH, PAGE_H - TOP - BOTTOM,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="study", frames=[frame], onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(TEAL)
        canvas.rect(LEFT, PAGE_H - 35, 20, 3, fill=1, stroke=0)
        canvas.setFont("CN-Bold", 8.4)
        canvas.setFillColor(NAVY)
        canvas.drawString(LEFT + 28, PAGE_H - 36, "动手学深度学习 / 学习讲义")
        canvas.setFont("CN", 8.2)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 36, self.spec['section'] + " " + self.spec['title'])
        canvas.setStrokeColor(LINE)
        canvas.line(LEFT, 37, PAGE_W - RIGHT, 37)
        canvas.setFont("CN", 7.8)
        canvas.drawString(LEFT, 23, "阅读 · 运行 · 解释    |    " + self.spec.get('date', DATE))
        page_label = f"{doc.page}" + (f" / {self.total_pages}" if self.total_pages else "")
        canvas.drawRightString(PAGE_W - RIGHT, 23, page_label)
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "H2":
            self.outline_index += 1
            key = f"section-{self.outline_index}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)


EXERCISE_21 = """
## 综合练习：把操作放在同一个任务中

使用本讲义开头的 X（三条样本、每条两个特征），集中完成以下操作。先预测形状，再写表达式和运行。

| 任务 | 目标与约束 |
| --- | --- |
| 1. 重新组织 | 将 X 变为两行三列 |
| 2. 转置 | 交换二维 X 的行与列 |
| 3. 按列中心化 | 每个元素减去所在列的均值，保留 (3,2) |
| 4. 每行总和 | 将两个特征求和，保留 (3,1) |
| 5. 添加新特征 | 将每行总和拼到 X 后，得到 (3,3) |
| 6. 索引与切片 | 取最后两条样本的全部特征 |
| 7. 条件比较 | 标记 X 中大于 3 的位置，保留布尔矩阵 |
| 8. 三维广播 | 计算下面 a+b，解释结果形状 |

```python
a = torch.arange(6).reshape(2, 1, 3)
b = torch.arange(4).reshape(1, 4, 1)
```

集中解释三个问题：

1. reshape 与转置形状相同，为什么元素排列不同？
2. (3,1) 的预测减去 (3,) 的标签会发生什么？如何逐样本对齐？
3. alias、基本切片、clone、from_numpy 与 torch.tensor(array)，哪些共享原数据，哪些独立？用修改一个元素的实验说明。

配套练习文件：`exercises/ch02/01_data_operations.py`。记录操作结果、预期与实际的差异，以及自己的解释；无需背诵全部 API。

## 阅读后的速查表

| 需要做什么 | 首先想到的工具 |
| --- | --- |
| 检查结构 | shape、ndim、numel、dtype、device |
| 调整排列 | reshape；二维交换轴用 .T |
| 批量计算 | 按元素运算和广播 |
| 汇总或增加数据 | sum/mean；cat |
| 取样本或特征 | 索引、切片、布尔筛选 |
| 保留独立副本 | clone；NumPy 数组用 copy |
| 导出普通对象 | numpy、from_numpy、item |

配套完整演示：`lessons/ch02/01_data_operations.py`。程序包含故意捕获的广播错误，用来说明不兼容形状；它不是需要修复的运行失败。
"""


END_TO_END_22 = """
## 完整流程：对照原理再运行一次

下面程序在项目根目录运行，读取本节配套 CSV。它将整个主流程放在一起，方便离线阅读时查对。

```python
import numpy as np
import pandas as pd
import torch

data = pd.read_csv("datasets/ch02/house_tiny.csv")
inputs = data.iloc[:, :2].copy()
targets = data.iloc[:, 2].copy()

mean_rooms = inputs["NumRooms"].mean()
inputs["NumRooms"] = inputs["NumRooms"].fillna(mean_rooms)
encoded = pd.get_dummies(
    inputs, columns=["Alley"],
    dummy_na=True, dtype=np.float32,
)

X = torch.tensor(
    encoded.to_numpy(dtype=np.float32),
    dtype=torch.float32,
)
y = torch.tensor(
    targets.to_numpy(dtype=np.float32),
    dtype=torch.float32,
).reshape(-1, 1)

print(encoded)
print(X.shape, y.shape)
print(torch.isnan(X).any(), torch.isnan(y).any())
```

自检：X 为四行三列，y 为四行一列，均为 float32，不含 NaN。输入和标签保持同一行顺序。程序能运行只是第一步，还需解释每个转换的原因。
"""


def exercise_22():
    with (ROOT / "datasets/ch02/house_practice.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    md = "\n## 综合练习：更多行和列的原始数据\n\n"
    md += "下面是配套的完整人工数据，可以直接在纸上分析缺失数量，也可以运行 CSV 练习。NA 表示缺失。\n\n"
    md += "| " + " | ".join(rows[0]) + " |\n"
    md += "| " + " | ".join(["---"] * len(rows[0])) + " |\n"
    md += "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
    md += """

一次完成以下处理：

1. 取出 Price 作为标签，其余列作为输入。
2. 计算输入各列的缺失数量，找出并删除缺失最多的列。
3. 数值列按各自已知数值的均值填补，不硬编码均值。
4. 对无序类别独热编码，同时表示缺失状态。
5. 创建 float32 的 X 与 (N,1) 形式的 y，检查行数和 NaN。

注意练习表的输入列数多于示例，不能只复制“取前两列”而忽略其他输入。配套文件：`exercises/ch02/02_data_preprocessing.py`。

集中记录：删除了哪列、填补均值、编码后的列名、X/y 的形状，以及下面三个解释。

- A：未知房间数为什么不能默认填 0？均值填补做了什么假设？
- B：把无序类别直接作为 0、1、2 的数值输入，会引入什么关系？独热编码如何表达？
- C：训练集与测试集应各自计算填补均值吗？同一特征的编码列顺序为什么要一致？

## 阅读后的速查表

| 环节 | 方法与关注点 |
| --- | --- |
| 读取文件 | read_csv；确认列名、行数和 dtype |
| 分开输入与目标 | iloc 或按列名选择；Price 不放入输入 |
| 识别缺失 | isna().sum()；缺失不等于零 |
| 数值填补 | mean + fillna；按列计算 |
| 删除数据 | dropna 删除行；drop(columns=...) 删除列 |
| 编码类别 | get_dummies；保留缺失状态和特征列名 |
| 转成张量 | to_numpy(dtype=...) + torch.tensor |
| 检查计算约定 | X 行数与 y 行数一致；标签形状对齐 |

配套完整演示：`lessons/ch02/02_data_preprocessing.py`。下一节为 2.3 线性代数。
"""
    return md


SPECS = [
    dict(section="2.1", title="数据操作", subtitle="用形状、运算和存储关系，把一组数值组织成正确的计算。",
         source="docs/ch02/01_data_operations.md", start="## 统一的数据",
         end="## 一次完成综合练习", pages="PDF 第 58-65 页 / 正文第 40-47 页",
         route=["创建与形状", "运算与广播", "索引与修改", "共享与转换"],
         goals="读完后，能够判断张量形状和广播结果，按样本与特征取数据，区分共享存储与独立复制，并发现损失计算中的形状错配。",
         filename="2.1-数据操作-学习讲义.pdf"),
    dict(section="2.2", title="数据预处理", subtitle="从含有缺失值与文字的原始表格，得到模型可计算的张量。",
         source="docs/ch02/02_data_preprocessing.md", start="## 为什么需要预处理",
         end="## 一次完成综合练习", pages="PDF 第 65-68 页 / 正文第 47-50 页",
         route=["读取表格", "处理缺失", "编码类别", "转换与核对"],
         goals="读完后，能够分开特征与标签，解释填补或删除的选择，将类别特征编码成数值，并按同一套规则转换成无缺失的张量。",
         filename="2.2-数据预处理-学习讲义.pdf"),
    dict(section="2.3", title="线性代数", subtitle="从一条样本的加权和，到一批预测与误差的大小。",
         source="docs/ch02/03_linear_algebra.md", start="## 这节到底解决什么问题",
         end=None, pages="PDF 第 68-81 页 / 正文第 50-63 页（含节末练习）",
         route=["对象与形状", "求和与点积", "矩阵乘法", "范数与误差"],
         goals="能够解释按元素乘、点积和矩阵乘法的区别，判断汇总与乘法的输出形状，理解范数，并独立写出一次小批量预测和误差计算。",
         filename="2.3-线性代数-学习讲义.pdf"),
    dict(section="2.4", title="微积分", subtitle="从参数的小变化，理解导数、梯度和链式法则。",
         source="docs/ch02/04_calculus.md", start="## 本节要回答的核心问题",
         end=None, pages="PDF 第 81-87 页 / 正文第 63-69 页（含节末练习）",
         route=["导数与微分", "偏导与梯度", "链式法则", "一步参数更新"],
         figure_height=235,
         goals="能够解释局部变化率，手工求出简单损失对权重和偏置的梯度，理解多路径链式法则，并用两种学习率比较一次参数更新。",
         filename="2.4-微积分-学习讲义.pdf"),
    dict(section="2.5", title="自动微分", subtitle="沿计算图求梯度，把手写求导接成一轮参数更新。",
         source="docs/ch02/05_autograd.md", start="## 本节的主线：记录、求导、清理、更新",
         end=None, pages="PDF 第 87-91 页 / 正文第 69-73 页（含节末练习）",
         route=["记录与求导", "梯度与图", "分离与控制流", "一轮参数更新"],
         date="2026-10-19", figure_height=210,
         goals="能够使用 backward 获取梯度，分清梯度累积和图缓存寿命，解释多元素输出、detach 和控制流的求导，并完成一次自动微分更新。",
         filename="2.5-自动微分-学习讲义.pdf"),
    dict(section="2.6", title="概率", subtitle="从随机抽样到条件判断，用分布、证据与波动理解预测。",
         source="docs/ch02/06_probability.md", start="## 概率为什么出现在深度学习里",
         end=None, pages="PDF 第 91-99 页 / 正文第 73-81 页（含节末练习）",
         route=["分布与抽样", "联合与条件", "贝叶斯与独立", "期望与方差"],
         date="2026-10-19", figure_height=185,
         goals="能够区分概率与频率，从联合表计算边际和条件概率，说明贝叶斯公式及独立性假设，并通过模拟计算期望和方差。",
         filename="2.6-概率-学习讲义.pdf"),
    dict(section="2.7", title="查阅文档", subtitle="把陌生函数的说明，变成能够验证和解释的调用。",
         source="docs/ch02/07_documentation.md", start="## 这节要建立的习惯",
         end=None, pages="PDF 第 99-101 页 / 正文第 81-83 页",
         route=["发现名字", "读懂签名", "最小验证", "查阅与实践"],
         date="2026-10-19",
         goals="能够用 dir 和 help 查找工具，读懂基本签名、参数和返回值，核对版本，并通过最小实验完成一个陌生函数的调用。",
         filename="2.7-查阅文档-学习讲义.pdf"),
    dict(section="3.1", title="线性回归", subtitle="把数据、模型、损失与优化连接成一个完整学习问题。",
         chapter="第 3 章 · 线性神经网络",
         source="docs/ch03/01_linear_regression.md", start="## 先说清楚：我们要预测什么",
         end=None, pages="PDF 第 103-112 页 / 正文第 85-94 页（含节末练习）",
         route=["模型与损失", "最小二乘与优化", "矢量化与概率", "网络与实践"],
         date="2026-10-19", figure_height=195,
         goals="能够解释线性回归的四个要素，区分参数与超参数，对照最小二乘与梯度下降，并理解平方损失的高斯噪声解释及单层网络表示。",
         filename="3.1-线性回归-学习讲义.pdf"),
    dict(section="3.2", title="线性回归的从零开始实现", subtitle="自己组织批次、前向、损失与 SGD，读懂每一次参数更新。",
         chapter="第 3 章 · 线性神经网络",
         source="docs/ch03/02_linear_regression_scratch.md", start="## 把一轮学习组织成可运行的程序",
         end=None, pages="PDF 第 113-118 页 / 正文第 95-100 页（含节末练习）",
         route=["生成与读取", "模型与损失", "手写小批量 SGD", "训练与核对"],
         date="2026-10-19", figure_height=200,
         goals="能够使用生成器保持特征与标签配对，正确处理尾批，区分求和与平均的梯度尺度，并用张量和自动求导独立接成小批量训练循环。",
         filename="3.2-线性回归的从零开始实现-学习讲义.pdf"),
    dict(section="3.3", title="线性回归的简洁实现", subtitle="把手写部件对应到框架组件，保留清晰的训练逻辑。",
         chapter="第 3 章 · 线性神经网络",
         source="docs/ch03/03_linear_regression_concise.md", start="## 从五个手写部件到框架组件",
         end=None, pages="PDF 第 119-123 页 / 正文第 101-105 页（含节末练习）",
         route=["数据加载", "层与参数", "损失与优化", "对照与实践"],
         date="2026-10-19", figure_height=200,
         goals="能够用 TensorDataset、DataLoader、Linear、MSELoss 和 SGD 完成训练，正确读取参数与梯度，并核对损失尺度、尾批及教材 Huber 公式。",
         filename="3.3-线性回归的简洁实现-学习讲义.pdf"),
    dict(section="3.4", title="softmax 回归", subtitle="从类别得分到概率，用交叉熵连接预测、信息与梯度。",
         chapter="第 3 章 · 线性神经网络",
         source="docs/ch03/04_softmax_regression.md", start="## 从预测数值到选择类别",
         end=None, pages="PDF 第 123-129 页 / 正文第 105-111 页（含节末练习）",
         route=["标签与得分", "概率与损失", "梯度与信息", "分类与实践"],
         date="2026-10-20", figure_height=195,
         goals="能够区分类别编号、独热标签、得分与概率，按行计算 softmax，解释交叉熵及其梯度，并用准确率评估一次完整分类训练。",
         filename="3.4-softmax回归-学习讲义.pdf"),
    dict(section="3.5", title="图像分类数据集", subtitle="把真实图像读成张量，检查标签、批次与数据流水线。",
         chapter="第 3 章 · 线性神经网络",
         source="docs/ch03/05_image_classification_dataset.md", start="## 这节把真实图像接入学习流程",
         end=None, pages="PDF 第 129-133 页 / 正文第 111-115 页（含节末练习）",
         route=["样本与标签", "图像与张量", "批次与变换", "性能与实践"],
         date="2026-10-20", figure_height=210,
         goals="能够读取 Fashion-MNIST，解释通道和批量形状、像素缩放与 Resize，正确组织训练和测试加载器，并用实际计时判断数据读取方案。",
         filename="3.5-图像分类数据集-学习讲义.pdf"),
    dict(section="3.6", title="softmax 回归的从零开始实现", subtitle="用张量写出真实图像分类的训练、评估与预测。",
         chapter="第 3 章 · 线性神经网络",
         source="docs/ch03/06_softmax_scratch.md", start="## 把整条分类链亲手接起来",
         end=None, pages="PDF 第 133-141 页 / 正文第 115-123 页（含节末练习）",
         route=["参数与概率", "损失与统计", "手写更新", "训练与预测"],
         date="2026-10-20", figure_height=320,
         goals="能够用张量和自动求导训练十类图像分类器，解释逐样本交叉熵、正确数量与加权统计，正确处理尾批，并核对数值稳定性与决策代价。",
         filename="3.6-softmax回归的从零开始实现-学习讲义.pdf"),
    dict(section="3.7", title="softmax 回归的简洁实现", subtitle="用框架部件接通分类训练，并核对它与手写实现的关系。",
         chapter="第 3 章 · 线性神经网络",
         source="docs/ch03/07_softmax_concise.md", start="## 一张图看清本节替换了什么",
         end=None, pages="PDF 第 141-144 页 / 正文第 123-126 页（含节末练习）",
         route=["模型与初始化", "稳定损失", "优化与训练", "调参与实践"],
         date="2026-10-20", figure_height=240, start_on_new_page=True,
         goals="能够对应手写与框架实现，正确使用原始得分、平均交叉熵和SGD，并通过训练/验证实验理解超参数和训练轮数的影响。",
         filename="3.7-softmax回归的简洁实现-学习讲义.pdf"),
    dict(section="4.1", title="多层感知机", subtitle="从隐藏表示与非线性出发，理解激活、梯度与模型表达能力。",
         chapter="第 4 章 · 多层感知机",
         source="docs/ch04/01_multilayer_perceptron.md", start="## 为什么这一章需要隐藏层",
         end=None, pages="PDF 第 145-152 页 / 正文第 127-134 页（含四道节末练习）",
         route=["隐藏层与形状", "非线性与激活", "梯度与表达", "XOR与实践"],
         date="2026-10-27", figure_height=390, start_on_new_page=True,
         goals="能够解释线性层合并与非线性激活的作用，核对ReLU、sigmoid、tanh的梯度，通过XOR对照理解表示能力，并正确区分通用近似、优化与泛化。",
         filename="4.1-多层感知机-学习讲义.pdf"),
    dict(section="4.2", title="多层感知机的从零开始实现", subtitle="手写两层参数与ReLU，训练真实图像分类器并进行有限调参。",
         chapter="第 4 章 · 多层感知机",
         source="docs/ch04/02_mlp_scratch.md", start="## 从隐藏层公式到真实图像训练",
         end=None, pages="PDF 第 153-156 页 / 正文第 135-138 页（含六道节末练习）",
         route=["参数与形状", "手写前向", "训练与预测", "调参与实践"],
         date="2026-10-27", figure_height=270, start_on_new_page=True,
         goals="能够管理两层参数、手写ReLU和前向，连接平均交叉熵与SGD完成图像训练，并在固定预算内用验证集比较超参数。",
         filename="4.2-多层感知机的从零开始实现-学习讲义.pdf"),
    dict(section="4.3", title="多层感知机的简洁实现", subtitle="用模块组织同一条训练链，比较深度、激活与初始化。",
         chapter="第 4 章 · 多层感知机",
         source="docs/ch04/03_mlp_concise.md", start="## 同一个模型，交给模块管理",
         end=None, pages="PDF 第 156-157 页 / 正文第 138-139 页（含三道节末练习）",
         route=["模块与参数", "训练与核对", "三类实验", "综合实践"],
         date="2026-10-27", figure_height=260, start_on_new_page=True,
         goals="能够对应手写与模块化MLP，正确初始化和训练网络，并在固定条件下比较隐藏层、激活及初始化，说明结果的适用范围。",
         filename="4.3-多层感知机的简洁实现-学习讲义.pdf"),
    dict(section="4.4", title="模型选择、欠拟合和过拟合", subtitle="区分容量、优化与泛化，用验证数据选择合适的模型。",
         chapter="第 4 章 · 多层感知机",
         source="docs/ch04/04_model_selection.md", start="## 从拟合已有样本，到预测未见样本",
         end=None, pages="PDF 第 157-167 页 / 正文第 139-149 页（含四道节末练习）",
         route=["泛化与数据划分", "拟合与直接解", "模型选择实验", "数值与实践"],
         date="2026-10-27", figure_height=245, start_on_new_page=True,
         goals="能够区分训练误差与泛化误差，正确使用验证及K折选择模型，通过多项式实验判断容量、优化和数值问题，并解释噪声导致的误差下限。",
         filename="4.4-模型选择、欠拟合和过拟合-学习讲义.pdf"),
    dict(section="4.5", title="权重衰减", subtitle="在拟合数据与限制权重之间作权衡，理解正则化的作用。",
         chapter="第 4 章 · 多层感知机",
         source="docs/ch04/05_weight_decay.md", start="## 用惩罚限制权重的大小",
         end=None, pages="PDF 第 167-174 页 / 正文第 149-156 页（含六道节末练习）",
         route=["目标与梯度", "两种实现", "验证选择", "范数与先验"],
         date="2026-10-27", figure_height=235, start_on_new_page=True,
         goals="能够推导L2惩罚的衰减更新，对齐手写与框架实现的损失尺度，通过验证选择强度，并解释偏置分组、L1/Frobenius及贝叶斯先验。",
         filename="4.5-权重衰减-学习讲义.pdf"),
    dict(section="4.6", title="暂退法（Dropout）", subtitle="用随机掩码扰动隐藏激活，区分训练、预测与统计期望。",
         chapter="第 4 章 · 多层感知机",
         source="docs/ch04/06_dropout.md", start="## 从惩罚权重到扰动隐藏表示",
         end=None, pages="PDF 第 174-180 页 / 正文第 156-162 页（含七道节末练习）",
         route=["掩码与期望", "手写与框架", "训练与预测", "受控实验"],
         date="2026-10-27", figure_height=310, start_on_new_page=True,
         goals="能够解释暂退概率和保留值缩放，正确切换训练/评估模式，核对手写与内置实现，并通过概率、轮数、方差和组合噪声实验理解正则化的条件。",
         filename="4.6-暂退法（Dropout）-学习讲义.pdf"),
    dict(section="4.7", title="前向传播、反向传播和计算图", subtitle="沿依赖计算数值，沿链式法则传递梯度，再更新参数。",
         chapter="第 4 章 · 多层感知机",
         source="docs/ch04/07_backpropagation.md", start="## 把一次训练分成三件事",
         end=None, pages="PDF 第 180-183 页 / 正文第 162-165 页（含五道节末练习）",
         route=["前向与计算图", "链式法则", "三种实现核对", "内存与二阶"],
         date="2026-10-27", figure_height=260, start_on_new_page=True,
         goals="能够对应列向量与批量公式，推导含L2分支和偏置的梯度，区分求导与更新，并说明保存中间量、二阶求导及模型分阶段执行的代价。",
         filename="4.7-前向传播、反向传播和计算图-学习讲义.pdf"),
    dict(section="4.8", title="数值稳定性和模型初始化", subtitle="理解多层连乘的尺度，选择合适的起点，再观察真实训练。",
         chapter="第 4 章 · 多层感知机", date="2026-10-27",
         route=["梯度连乘", "对称与方差", "初始化诊断", "训练与拓展"],
         goals="理解梯度消失与爆炸、隐藏单元对称性，推导Xavier并区分ReLU的二阶矩；用逐层诊断和同任务训练检查初始化，覆盖四道教材练习。",
         pages="PDF 第 184-188 页 / 正文第 166-170 页（含全部节末练习）",
         source="docs/ch04/08_numerical_stability.md", start="## 从上一节的链式法则继续", end=None,
         figure_height=250, start_on_new_page=True,
         filename="4.8-数值稳定性和模型初始化-学习讲义.pdf"),
    dict(section="4.9", title="环境和分布偏移", subtitle="识别数据来源的变化，在明确假设下纠正，并考虑决策的反馈。",
         chapter="第 4 章 · 多层感知机", date="2026-10-27",
         route=["三类偏移", "加权与域检测", "标签与在线", "反馈与实践"],
         goals="区分协变量、标签和概念偏移，推导带域先验修正的密度比与标签比例估计，完成加权实验，并解释在线反馈、学习情景与分组评价。",
         pages="PDF 第 188-198 页 / 正文第 170-180 页（含四道节末练习）",
         source="docs/ch04/09_distribution_shift.md", start="## 从模型是否学好，到环境是否改变", end=None,
         figure_height=245, start_on_new_page=True,
         filename="4.9-环境和分布偏移-学习讲义.pdf"),
    dict(section="4.10", title="实战Kaggle比赛：预测房价", subtitle="从真实表格到严格验证，再生成可检查的房价预测文件。",
         chapter="第 4 章 · 多层感知机", date="2026-10-27",
         route=["数据与预处理", "对数目标", "五折选模型", "重训与导出"],
         goals="完成真实房价数据的折内预处理与训练，区分原价和log目标、OOF和线上成绩，比较八个固定方案并生成1459行预测，覆盖六道教材练习。",
         pages="PDF 第 198-208 页 / 正文第 180-190 页（含六道节末练习）",
         source="docs/ch04/10_kaggle_house_prices.md", start="## 用一个真实项目串起第4章", end=None,
         figure_height=235, start_on_new_page=True,
         filename="4.10-实战Kaggle比赛：预测房价-学习讲义.pdf"),
]


def story(spec):
    result = [
        p(spec.get('chapter', "第 2 章 · 预备知识") + " / " + spec['section'], "Eyebrow"),
        p(spec['section'] + " " + spec['title'], "Title"),
        p(spec['subtitle'], "Subtitle"),
        Route(spec['route']),
        p("本节目标", "H2"),
        p(spec['goals']),
        p("建议先连续阅读完整知识链，再运行示例，最后集中完成章末练习。已会的内容可快速通过，重点追踪自己无法解释的结果。"),
        p("教材定位：" + spec['pages'] + "。", "Caption"),
        p("依据用户提供的《动手学深度学习》Release 2.0.0 独立整理，配套程序与练习在项目内。", "Caption"),
    ]
    raw = (ROOT / spec['source']).read_text(encoding="utf-8")
    body = raw[raw.index(spec['start']):]
    if spec['end'] is not None:
        body = body[:body.index(spec['end'])]
    body = body.replace("沿用你已经成功创建的三条样本、每条两个特征：", "本节统一使用三条样本、每条两个特征：")
    if spec['section'] == "2.2":
        body = body.replace("data = pd.read_csv(data_file)", 'data_file = "datasets/ch02/house_tiny.csv"\ndata = pd.read_csv(data_file)')
    if spec.get('start_on_new_page'):
        result.append(PageBreak())
    result.extend(markdown_blocks(body, (ROOT / spec['source']).parent,
                                  spec.get('figure_height', 178)))
    if spec['section'] == "2.1":
        result.extend(markdown_blocks(EXERCISE_21))
    elif spec['section'] == "2.2":
        result.extend(markdown_blocks(END_TO_END_22))
        result.extend(markdown_blocks(exercise_22()))
    result.extend([
        Spacer(1, 9),
        p("资料说明：教材作者为 Aston Zhang、Zachary C. Lipton、Mu Li、Alexander J. Smola；页码基于用户提供的中文 PDF。文中相对路径均从“李沐-深度学习”项目根目录计算。已验证环境：Python 3.8.20、PyTorch 2.4.1+cu121、NumPy 1.23.5、pandas 2.0.3。整理日期：" + spec.get('date', DATE) + "。", "Caption"),
    ])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=[spec['section'] for spec in SPECS],
                        help="Build only one booklet; leave older PDFs unchanged")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        if args.section and spec['section'] != args.section:
            continue
        draft = TEMP / (spec['section'].replace('.', '_') + "_layout.pdf")
        StudyDoc(draft, spec).build(story(spec))
        total = len(PdfReader(draft).pages)
        target = OUTPUT / spec['filename']
        StudyDoc(target, spec, total_pages=total).build(story(spec))
        reader = PdfReader(target)
        assert len(reader.pages) == total
        assert reader.outline, "Section bookmarks missing"
        print(f"{target.name}: {total} pages, {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
