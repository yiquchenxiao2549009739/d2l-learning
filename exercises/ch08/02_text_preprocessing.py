"""8.2 综合练习。八处 TODO 将读取、清洗、计数与索引串成一条链。"""
from collections import Counter
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    raw = (ROOT/'datasets/ch08/text_demo.txt').read_text(encoding='utf-8').splitlines()
    clean = None  # TODO 1：按行连续替换非 ASCII 字母，再 strip 和 lower。
    if clean is None:
        print('先完成 TODO 1，把原始文本清洗成行。')
        return
    words = None  # TODO 2：逐行按空白分词，保留空行对应的空列表。
    chars = None  # TODO 3：逐行按字符切分，保留行内空格。
    if words is None or chars is None:
        print('完成 TODO 2-3，检查单词与字符两种粒度。')
        return
    assert len(words)==len(chars)==5 and words[3]==chars[3]==[]
    counts = None  # TODO 4：遍历训练前四行词元，用 Counter 计数。
    if counts is None:
        print('完成 TODO 4，只统计训练行。')
        return
    assert counts['time']==3 and 'wells' not in counts
    reserved = ['<unk>','<pad>','<bos>','<eos>']
    sorted_counts = sorted(counts.items(),key=lambda item:item[1],reverse=True)
    idx_to_token = None  # TODO 5：保留词元在前，然后加入频率至少2的词。
    if idx_to_token is None:
        print('完成 TODO 5，检查词表顺序。')
        return
    token_to_idx = None  # TODO 6：用 enumerate 构建词元到索引的字典。
    if token_to_idx is None:
        print('完成 TODO 6，生成正向映射。')
        return
    sample = ['<bos>', *words[0], *words[4], '<eos>', '<pad>']
    ids = None  # TODO 7：查每个词的 ID，缺失时回退到未知 ID 0。
    decoded = None  # TODO 8：按索引从 idx_to_token 查回词元。
    if ids is None or decoded is None:
        print('完成 TODO 7-8，运行编码和解码。')
        return
    assert len(idx_to_token)==5 and ids==[2,4,4,0,0,0,3,1]
    assert decoded == ['<bos>','time','time','<unk>','<unk>','<unk>','<eos>','<pad>']
    print('词表',idx_to_token)
    print('索引',ids,'解码',decoded)
    print('字符数',sum(map(len,chars)))


if __name__ == '__main__':
    main()
