"""8.3 语言模型和数据集：计数模型、困惑度与两种序列读取器。"""
import argparse
from collections import Counter
from importlib import import_module
import json
import math
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import torch
from torch.nn.functional import cross_entropy


preprocess = import_module('.02_text_preprocessing', __package__) if __package__ else import_module(
    '02_text_preprocessing'
)
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/ch08/8_3_metrics.json'
SEED = 8301
MASS_GRID = (0.1, 1.0, 10.0, 100.0)


def ngrams(tokens, order):
    """同一段内的连续 n 元组，按出现顺序生成。"""
    return zip(*(tokens[i:] for i in range(order)))


def rank_fit(frequencies, low, high):
    """在闭区间 [low, high] 对 ln 次数与 ln 排名做普通最小二乘。"""
    values = [(math.log(i), math.log(frequencies[i - 1])) for i in range(low, high + 1)]
    mean_x = math.fsum(x for x, _ in values) / len(values)
    mean_y = math.fsum(y for _, y in values) / len(values)
    xx = math.fsum((x - mean_x)**2 for x, _ in values)
    xy = math.fsum((x - mean_x)*(y - mean_y) for x, y in values)
    slope = xy / xx
    intercept = mean_y - slope * mean_x
    residual = math.fsum((y - slope*x - intercept)**2 for x, y in values)
    total = math.fsum((y - mean_y)**2 for _, y in values)
    return dict(low=low, high=high, alpha=-slope, intercept=intercept, r_squared=1-residual/total)


def book_statistics(words):
    results = []
    for order in (1, 2, 3):
        counts = Counter(ngrams(words, order))
        ranked = counts.most_common()
        frequencies = [count for _, count in ranked]
        results.append(dict(order=order, total=sum(frequencies), types=len(ranked),
                            singletons=frequencies.count(1), top10=ranked[:10],
                            frequencies=frequencies, fits=[rank_fit(frequencies, a, b)
                                                          for a, b in [(10, 500), (20, 1000)]]))
    return results


class NGramCounts:
    """统计 1–max_order 元组及其有后继的上下文出现次数。"""

    def __init__(self, tokens, vocab_size, max_order=3):
        self.vocab_size = vocab_size
        self.counts = {n: Counter(ngrams(tokens, n)) for n in range(1, max_order + 1)}
        self.totals = {n: Counter() for n in self.counts}
        for n, table in self.counts.items():
            for event, count in table.items():
                self.totals[n][event[:-1]] += count

    def probability(self, history, token, order, mass, prior='lower'):
        """mass 是分摊给整个词表的总伪计数。"""
        context = tuple(history[-(order-1):]) if order > 1 else ()
        n = len(context) + 1
        count = self.counts[n][context + (token,)]
        support = self.totals[n][context]
        if mass == 0:
            # 无后继历史在未平滑模型中未定义。调用方单独统计。
            return None if support == 0 else count / support
        base = (self.probability(context, token, n-1, mass, prior)
                if n > 1 and prior == 'lower' else 1 / self.vocab_size)
        return (count + mass * base) / (support + mass)


def evaluate(model, ids, order, mass, prior='lower'):
    """各模型使用同一标签集合。每个标签只读取所在区间中的真实前缀。"""
    losses = []
    undefined = zero = 0
    for i in range(2, len(ids)):
        p = model.probability(ids[i-2:i], ids[i], order, mass, prior)
        if p is None:
            undefined += 1
        elif p == 0:
            zero += 1
        else:
            losses.append(-math.log(p))
    examples = max(0, len(ids) - 2)
    nll = math.fsum(losses) / examples if examples and len(losses) == examples else None
    return dict(tokens=examples, undefined_context=undefined, zero_probability=zero,
                nll=nll, perplexity=None if nll is None else math.exp(nll))


def compare_models(rows):
    """按原始行切分。词表和频次仅在训练段拟合，平滑量仅用验证段选择。"""
    train_end = len(rows) * 8 // 10
    val_end = len(rows) * 9 // 10
    train, val, test = [preprocess.flatten(part) for part in (rows[:train_end], rows[train_end:val_end], rows[val_end:])]
    vocab = preprocess.Vocab(train, min_freq=2)
    train_ids, val_ids, test_ids = vocab[train], vocab[val], vocab[test]
    model = NGramCounts(train_ids, len(vocab))
    configs = []
    # n=1 时 lower 与 uniform 相同，只比较一次。
    for order, prior in [(1, 'uniform'), (2, 'uniform'), (2, 'lower'), (3, 'uniform'), (3, 'lower')]:
        validation = [dict(mass=mass, **evaluate(model, val_ids, order, mass, prior)) for mass in MASS_GRID]
        best = min(validation, key=lambda row: row['nll'])
        configs.append(dict(order=order, prior=prior, validation=validation, best_mass=best['mass'],
                            validation_best=best['perplexity'], test=evaluate(model, test_ids, order, best['mass'], prior)))
    return dict(split_lines=[train_end, val_end-train_end, len(rows)-val_end],
                split_tokens=[len(train_ids), len(val_ids), len(test_ids)], vocab_size=len(vocab),
                unknown=[ids.count(0) for ids in (train_ids, val_ids, test_ids)],
                uniform_perplexity=len(vocab), settings=configs,
                raw_validation=[evaluate(model, val_ids, n, 0) for n in (1, 2, 3)])


def batch_starts(length, batch_size, num_steps, mode, rng, offset=None):
    """生成绝对位置，便于区分词元值与它在语料中的位置。"""
    if batch_size <= 0 or num_steps <= 0:
        raise ValueError('batch_size 和 num_steps 必须是正整数')
    if mode not in ('random', 'sequential'):
        raise ValueError('mode 必须是 random 或 sequential')
    max_offset = num_steps - 1 if mode == 'random' else num_steps
    if offset is None:
        offset = rng.randint(0, max_offset)
    elif not 0 <= offset <= max_offset:
        raise ValueError(f'{mode} 的 offset 应在 0 到 {max_offset} 之间')
    usable = max(0, length - offset - 1)
    if mode == 'random':
        starts = list(range(offset, offset + (usable // num_steps) * num_steps, num_steps))
        rng.shuffle(starts)
        for i in range(0, len(starts) // batch_size * batch_size, batch_size):
            yield starts[i:i+batch_size]
    else:
        lane = usable // batch_size
        for t in range(0, lane // num_steps * num_steps, num_steps):
            yield [offset + row*lane + t for row in range(batch_size)]


def tensors_from_starts(corpus, starts, num_steps):
    x = [corpus[s:s+num_steps] for s in starts]
    y = [corpus[s+1:s+num_steps+1] for s in starts]
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SeqDataLoader:
    """每次 __iter__ 重新选偏移，保留同样的语料与词表。"""

    def __init__(self, corpus, vocab, batch_size, num_steps, use_random_iter=False, seed=None):
        self.corpus = corpus
        self.vocab = vocab
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.mode = 'random' if use_random_iter else 'sequential'
        self.rng = random.Random(seed)

    def __iter__(self):
        for starts in batch_starts(len(self.corpus), self.batch_size, self.num_steps, self.mode, self.rng):
            yield tensors_from_starts(self.corpus, starts, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000, *, seed=None, download=False):
    raw = preprocess.checked_bytes(download)
    corpus, vocab = preprocess.load_corpus_time_machine(max_tokens, raw)
    return SeqDataLoader(corpus, vocab, batch_size, num_steps, use_random_iter, seed), vocab


def describe_batches(corpus, batch_size, num_steps, mode, offset):
    starts = list(batch_starts(len(corpus), batch_size, num_steps, mode, random.Random(SEED), offset))
    if starts:
        x, y = tensors_from_starts(corpus, starts[0], num_steps)
        first = dict(x=x.tolist(), y=y.tolist())
    else:
        first = None
    return dict(offset=offset, batches=len(starts), input_positions=len(starts)*batch_size*num_steps,
                max_pairs=max(0, len(corpus)-1), first_starts=starts[0] if starts else [],
                first=first, first_three_starts=starts[:3],
                row_continuity=(sum(b == a + num_steps for left, right in zip(starts, starts[1:])
                                    for a, b in zip(left, right))),
                compared_rows=max(0, len(starts)-1)*batch_size)


def toy_coverage(length=35, batch_size=2, num_steps=5):
    """精确计算各位置被用作 X 的期望概率，无需蒙特卡洛。"""
    random_cover = [0.0] * (length - 1)
    seq_cover = [0.0] * (length - 1)
    for offset in range(num_steps):
        q = max(0, length - offset - 1) // num_steps
        if q:
            p = (q // batch_size * batch_size) / q / num_steps
            for position in range(offset, offset + q*num_steps):
                random_cover[position] += p
    for offset in range(num_steps + 1):
        for batch in batch_starts(length, batch_size, num_steps, 'sequential', random.Random(0), offset):
            for start in batch:
                for position in range(start, start + num_steps):
                    seq_cover[position] += 1 / (num_steps + 1)
    return dict(position=list(range(length-1)), random=random_cover, sequential=seq_cover)


def sentence_minibatch():
    """两句各自移位，再补齐，在损失上忽略 PAD 标签。"""
    pad = 0
    # BOS=1，EOS=2，词元=3/4。句子之间无跨接标签。
    x = torch.tensor([[1, 3, 4], [1, 3, pad]])
    y = torch.tensor([[3, 4, 2], [3, 2, pad]])
    logits = torch.zeros(2, 3, 5, dtype=torch.float64)
    # 两个词的正确类分数加 log(2)，用于核对掩码分母。
    logits[:, 0, 3] = math.log(2)
    flat_logits, flat_labels = logits.reshape(-1, 5), y.reshape(-1)
    masked = cross_entropy(flat_logits, flat_labels, ignore_index=pad)
    loss = float(masked)
    return dict(x=x.tolist(), y=y.tolist(), mask=y.ne(pad).tolist(), valid=int(y.ne(pad).sum()),
                nll=loss, perplexity=math.exp(loss), unmasked=float(cross_entropy(flat_logits, flat_labels)))


def experiment(raw, batch_size, num_steps, max_tokens):
    rows = preprocess.tokenize(preprocess.read_lines(raw))
    corpus, vocab = preprocess.load_corpus_time_machine(max_tokens, raw)
    toy = list(range(35))
    return dict(
        book=book_statistics(preprocess.flatten(rows)),
        models=compare_models(rows),
        character=dict(length=len(corpus), vocab_size=len(vocab), batch_size=batch_size, num_steps=num_steps,
                       random=describe_batches(corpus, batch_size, num_steps, 'random',
                                               random.Random(SEED).randint(0, num_steps-1)),
                       sequential=describe_batches(corpus, batch_size, num_steps, 'sequential',
                                                   random.Random(SEED).randint(0, num_steps))),
        toy=dict(random=describe_batches(toy, 2, 5, 'random', 3),
                 sequential=describe_batches(toy, 2, 5, 'sequential', 0),
                 coverage=toy_coverage(), sentences=sentence_minibatch()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guided', action='store_true', help='按步骤显示主要结果')
    parser.add_argument('--record', action='store_true', help='记录完整默认对照')
    parser.add_argument('--download', action='store_true', help='缺少本地语料时取得教材版本')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=35)
    parser.add_argument('--max-tokens', type=int, default=10000)
    args = parser.parse_args()
    shape = (args.batch_size, args.num_steps, args.max_tokens)
    if min(shape) <= 0:
        parser.error('batch-size、num-steps、max-tokens 必须为正整数')
    if args.record and shape != (32, 35, 10000):
        parser.error('比较参数时请省略 --record')
    results = experiment(preprocess.checked_bytes(args.download), *shape)
    output = json.dumps(results, ensure_ascii=False, indent=2)
    if args.record:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with NamedTemporaryFile(mode='w', encoding='utf-8', dir=REPORT.parent, delete=False) as temp:
                temp_path = Path(temp.name)
                temp.write(output + '\n')
            temp_path.replace(REPORT)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    if args.guided:
        for row in results['book']:
            print(f"{row['order']} 元组：{row['total']} 次、{row['types']} 种，指数 {row['fits'][0]['alpha']:.4f}")
        print('训练 / 验证 / 测试行数', results['models']['split_lines'])
        for row in results['models']['settings']:
            print(f"{row['order']} 元 {row['prior']:7s} 伪计数 {row['best_mass']:5.1f}  "
                  f"验证 PPL {row['validation_best']:.3f}  测试 PPL {row['test']['perplexity']:.3f}")
        c = results['character']
        print('字符长度', c['length'], '词表', c['vocab_size'], '批形状', (c['batch_size'], c['num_steps']))
        for mode in ('random', 'sequential'):
            row = c[mode]
            print(mode, '偏移', row['offset'], '批数', row['batches'], '输入位置', row['input_positions'],
                  '相邻批同行续接', (row['row_continuity'], row['compared_rows']))
    else:
        print(output)


if __name__ == '__main__':
    main()
