"""独立核对 8.3 的频次、概率、取批和讲义代码。"""
from collections import defaultdict
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
from itertools import permutations, product
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
lesson = import_module('lessons.ch08.03_language_models')
REPORT = ROOT / 'reports/ch08'


def independent_tables(tokens):
    tables = [defaultdict(int) for _ in range(3)]
    totals = [defaultdict(int) for _ in range(3)]
    for order, (table, prefixes) in enumerate(zip(tables, totals), 1):
        for end in range(order, len(tokens) + 1):
            event = tuple(tokens[end-order:end])
            table[event] += 1
            prefixes[event[:-1]] += 1
    return tables, totals


def verify_frequencies(words, book):
    expected_counts = independent_tables(words)[0]
    for expected, record in zip(expected_counts, book):
        ranked = sorted(expected.items(), key=lambda row: -row[1])
        freq = np.array([count for _, count in ranked])
        assert record['top10'] == [[list(event), count] for event, count in ranked[:10]]
        assert record['frequencies'] == freq.tolist()
        assert record['total'] == int(freq.sum()) and record['types'] == freq.size
        assert record['singletons'] == int((freq == 1).sum())
        for fit in record['fits']:
            lo, hi = fit['low'], fit['high']
            x, y = np.log(np.arange(lo, hi+1)), np.log(freq[lo-1:hi])
            slope, intercept = np.polyfit(x, y, 1)
            assert math.isclose(-slope, fit['alpha'], rel_tol=1e-12)
            assert math.isclose(intercept, fit['intercept'], rel_tol=1e-12)
            assert math.isclose(np.corrcoef(x, y)[0, 1]**2, fit['r_squared'], rel_tol=1e-12)


def log_score(tables, totals, ids, order, mass, prior, size):
    undefined = zero = 0
    losses = []
    for end in range(2, len(ids)):
        probability = 1 / size
        for n in range(1 if prior == 'lower' else order, order + 1):
            event = tuple(ids[end-n+1:end+1])
            count = tables[n-1].get(event, 0)
            support = totals[n-1].get(event[:-1], 0)
            probability = (count + mass*probability) / (support + mass) if support + mass else None
        if probability is None:
            undefined += 1
        elif probability == 0:
            zero += 1
        else:
            losses.append(-math.log(probability))
    loss = sum(losses) / len(losses) if not (undefined or zero) else None
    return len(ids)-2, undefined, zero, loss


def compare_score(actual, computed):
    assert (actual['tokens'], actual['undefined_context'], actual['zero_probability']) == computed[:3]
    loss = computed[3]
    if loss is None:
        assert actual['nll'] is None and actual['perplexity'] is None
    else:
        assert math.isclose(actual['nll'], loss, rel_tol=1e-12)
        assert math.isclose(actual['perplexity'], math.exp(loss), rel_tol=1e-12)


def verify_models(rows, record):
    first, second, third = record['split_lines']
    train = lesson.preprocess.flatten(rows[:first])
    val = lesson.preprocess.flatten(rows[first:first+second])
    test = lesson.preprocess.flatten(rows[-third:])
    assert first + second + third == len(rows) == 3221
    counts = defaultdict(int)
    for word in train:
        counts[word] += 1
    keep = sorted((word for word in counts if counts[word] >= 2), key=lambda word: -counts[word])
    ids = {word: index for index, word in enumerate(keep, 1)}
    a, b, c = [[ids.get(word, 0) for word in section] for section in (train, val, test)]
    assert record['split_tokens'] == list(map(len, (a, b, c)))
    assert record['unknown'] == [a.count(0), b.count(0), c.count(0)]
    size = len(keep) + 1
    assert size == record['vocab_size'] == record['uniform_perplexity'] == 1869
    tables, totals = independent_tables(a)
    for order, raw in enumerate(record['raw_validation'], 1):
        compare_score(raw, log_score(tables, totals, b, order, 0, 'uniform', size))
    for model in record['settings']:
        n, prior = model['order'], model['prior']
        for row in model['validation']:
            compare_score(row, log_score(tables, totals, b, n, row['mass'], prior, size))
        best = min(model['validation'], key=lambda row: row['nll'])
        assert model['best_mass'] == best['mass'] and model['validation_best'] == best['perplexity']
        compare_score(model['test'], log_score(tables, totals, c, n, best['mass'], prior, size))
    toy = lesson.NGramCounts([0, 1, 2, 0, 1, 3, 0, 1, 2, 3], 4)
    checks = 0
    for order, prior, mass in product((1, 2, 3), ('uniform', 'lower'), (0.1, 2, 100)):
        for history in product(range(4), repeat=order-1):
            p = [toy.probability(history, y, order, mass, prior) for y in range(4)]
            assert min(p) > 0 and math.isclose(sum(p), 1, abs_tol=1e-12)
            checks += 1
    assert toy.probability((1,), 3, 2, 0) == 1/3
    assert toy.probability((3, 3), 1, 3, 0) is None
    assert lesson.evaluate(toy, [0, 1], 3, 1)['nll'] is None
    return checks


def reference_starts(length, b, t, offset, random_mode, seed):
    sequence = list(range(offset, length))
    if not sequence:
        return []
    paired = sequence[:-1]
    if random_mode:
        rows = [paired[j:j+t] for j in range(0, len(paired), t)]
        complete = [row[0] for row in rows if len(row) == t]
        random.Random(seed).shuffle(complete)
        return [complete[j:j+b] for j in range(0, len(complete)-b+1, b)]
    offset_matrix = np.array(paired[:len(paired) - len(paired)%b]).reshape(b, -1)
    return [offset_matrix[:, j].tolist() for j in range(0, offset_matrix.shape[1]-t+1, t)]


def verify_readers(character, toy, raw):
    cases = 0
    for length, b, t, random_mode in product((0, 1, 4, 5, 7, 19, 33, 35, 68), (1, 2, 3, 8), (1, 2, 5, 9), (False, True)):
        mode = 'random' if random_mode else 'sequential'
        for offset in range(t if random_mode else t+1):
            expected = reference_starts(length, b, t, offset, random_mode, lesson.SEED)
            result = list(lesson.batch_starts(length, b, t, mode, random.Random(lesson.SEED), offset))
            assert expected == result
            cells = [s+j for group in result for s in group for j in range(t)]
            assert len(cells) == len(set(cells))
            assert all(p+1 < length for p in cells)
            cases += 1
    corpus, vocab = lesson.preprocess.load_corpus_time_machine(character['length'], raw)
    assert len(vocab) == character['vocab_size'] == 28
    for source, record, b, t in [
        (corpus, character, character['batch_size'], character['num_steps']),
        (list(range(35)), toy, 2, 5),
    ]:
        for random_mode in (False, True):
            key = 'random' if random_mode else 'sequential'
            row = record[key]
            groups = reference_starts(len(source), b, t, row['offset'], random_mode, lesson.SEED)
            assert groups[:3] == row['first_three_starts'] and len(groups) == row['batches']
            for j, group in enumerate(groups):
                expected_x = [[source[s+k] for k in range(t)] for s in group]
                expected_y = [[source[s+k+1] for k in range(t)] for s in group]
                x, y = lesson.tensors_from_starts(source, group, t)
                assert x.dtype == y.dtype == torch.long and tuple(x.shape) == tuple(y.shape) == (b, t)
                assert x.tolist() == expected_x and y.tolist() == expected_y
                if j == 0:
                    assert row['first'] == dict(x=expected_x, y=expected_y)
    # 连续两轮取批可由同一个种子重放。
    for random_mode in (False, True):
        a = lesson.SeqDataLoader(corpus, vocab, 8, 12, random_mode, seed=8301)
        b = lesson.SeqDataLoader(corpus, vocab, 8, 12, random_mode, seed=8301)
        for _ in range(2):
            expected, result = list(a), list(b)
            assert expected and len(expected) == len(result)
            assert all(torch.equal(x, xx) and torch.equal(y, yy)
                       for (x,y), (xx,yy) in zip(expected, result))
    return cases


def verify_coverage(coverage):
    t, b, length = 5, 2, 35
    random_prob = np.zeros(length-1)
    for offset in range(t):
        starts = list(range(offset, length-t, t))
        counts = np.zeros(length-1)
        n = 0
        for order in permutations(starts):
            n += 1
            for start in order[:len(order)//b*b]:
                counts[start:start+t] += 1
        random_prob += counts / n / t
    seq_prob = np.zeros(length-1)
    for offset in range(t+1):
        groups = reference_starts(length, b, t, offset, False, 0)
        for group in groups:
            for start in group:
                seq_prob[start:start+t] += 1/(t+1)
    assert np.allclose(random_prob, coverage['random'], rtol=0, atol=1e-14)
    assert np.allclose(seq_prob, coverage['sequential'], rtol=0, atol=1e-14)
    # 增加一个候选数不是 B 倍数的试验，核对随机舍弃对中间位置的影响。
    short = lesson.toy_coverage(length=34)
    assert math.isclose(short['random'][10], .96)


def capture(source, path):
    out = StringIO()
    with redirect_stdout(out):
        exec(compile(source, str(path), 'exec'), {'__file__': str(path), '__name__': '__main__'})
    return out.getvalue()


def verify_practice():
    doc = ROOT / 'docs/ch08/03_language_models.md'
    pages = doc.read_text(encoding='utf-8').split('<!-- pagebreak -->')
    blocks = []
    for page in pages:
        if re.search(r'^## (15|16|17|18|19) ', page, re.M):
            blocks.extend(re.findall(r'```python\n(.*?)\n```', page.split('<!-- side -->')[0], re.S))
    assert len(blocks) == 5
    code = '\n\n'.join(blocks)
    path = ROOT / '.cache/8_3_follow.py'
    output = capture(code, path)
    for expected in ('loss / PPL: 0.948560 2.581989', 'random: [13, 28] (2, 5)',
                     'sequential: [[0, 17], [5, 22], [10, 27]] lane: 17',
                     'masked loss / PPL: 1.405108 4.075966'):
        assert expected in output
    alternatives = [
        ('4, 2.0', '4, 4.0', '2.857738'),
        ('steps, offset = 2, 5, 3', 'steps, offset = 3, 5, 3', '(3, 5)'),
        ('seq_offset = 0', 'seq_offset = 1', '[[1, 17], [6, 22], [11, 27]] lane: 16'),
        (', ignore_index=0', '', 'masked loss / PPL: 1.439163'),
    ]
    for old, new, expected in alternatives:
        assert code.count(old) == 1 and expected in capture(code.replace(old, new), path)
    exercise = ROOT / 'exercises/ch08/03_language_models.py'
    original = exercise.read_bytes()
    source = original.decode('utf-8')
    refs = [
        ('pairs = None', 'pairs = list(zip(corpus[:-1], corpus[1:]))'),
        ('totals = None', 'totals = Counter(x for x, _ in pairs)'),
        ('probs = None', 'probs = [(counts[1, y] + mass/vocab_size)/(totals[1] + mass) for y in range(vocab_size)]'),
        ('loss = None', 'loss = sum(-math.log(probs[y]) for y in (2, 3))/2'),
        ('r_starts = None', 'r_starts = list(range(offset, len(seq)-steps, steps))'),
        ('rx = None', 'rx = [seq[i:i+steps] for i in starts]'),
        ('ry = None', 'ry = [seq[i+1:i+steps+1] for i in starts]'),
        ('second_starts = None', 'second_starts = [seq_offset + lane*i + steps for i in range(batch_size)]'),
    ]
    assert len(re.findall(r'# TODO [1-8]：', source)) == 8
    assert '先完成 TODO 1–2' in capture(source, exercise)
    for old, new in refs:
        assert source.count(old) == 1
        source = source.replace(old, new)
    done = capture(source, exercise)
    assert '2.581989' in done and '顺序第二批 [5, 22]' in done
    assert exercise.read_bytes() == original
    return dict(follow=output, parameter_checks=4, exercise_reference=done, remaining_todos=8)


def main():
    raw = lesson.preprocess.checked_bytes()
    rows = lesson.preprocess.tokenize(lesson.preprocess.read_lines(raw))
    record = json.loads((REPORT / '8_3_metrics.json').read_text(encoding='utf-8'))
    verify_frequencies(lesson.preprocess.flatten(rows), record['book'])
    distributions = verify_models(rows, record['models'])
    cases = verify_readers(record['character'], record['toy'], raw)
    verify_coverage(record['toy']['coverage'])
    manual = -(2*math.log(2/6)+3*math.log(1/5))/5
    assert math.isclose(manual, record['toy']['sentences']['nll'], rel_tol=1e-12)
    output = dict(zipf_fits=6, validation_candidates=20, test_scores=5, normalized_distributions=distributions,
                  sampler_cases=cases, coverage_positions=34, masked_loss=manual, practice=verify_practice())
    path = REPORT / '8_3_verification.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'已核对 {cases} 组取批、20 个验证候选、5 段跟做和练习参考')


if __name__ == '__main__':
    main()
