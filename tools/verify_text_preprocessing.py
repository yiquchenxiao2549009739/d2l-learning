"""独立重算语料清洗、频次、保留词元、BPE，以及练习输出。"""
from collections import defaultdict
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import string

ROOT = Path(__file__).resolve().parents[1]
H = runpy.run_path(str(ROOT/'lessons/ch08/02_text_preprocessing.py'))


def independent_clean(line):
    segments, current = [], []
    for char in line:
        if char in string.ascii_letters:
            current.append(char.lower())
        elif current:
            segments.append(''.join(current))
            current = []
    if current:
        segments.append(''.join(current))
    return ' '.join(segments)


def independent_counts(tokens):
    result = {}
    for t in tokens:
        result[t] = result.get(t,0)+1
    return result


def verify_corpus(raw, metrics):
    lines = [independent_clean(x) for x in raw.decode('utf-8').splitlines()]
    assert H['read_lines'](raw) == lines and len(lines)==3221
    book = metrics['book']
    words = ' '.join(lines).split()
    count = independent_counts(words)
    ranked = sorted(count,key=lambda word:-count[word])
    word_vocab = H['Vocab'](H['tokenize'](lines))
    assert word_vocab.idx_to_token==['<unk>',*ranked]
    assert len(words)==book['word_count']==32775 and len(count)==4579
    assert [ranked.index(w)+1 for w in lines[0].split()]==book['first_word_ids']
    assert [ranked.index(w)+1 for w in lines[10].split()]==book['line10_word_ids']
    flat = ''.join(lines)
    chars = independent_counts(flat)
    names = ['<unk>',*sorted(chars,key=lambda ch:-chars[ch])]
    ids,vocab = H['load_corpus_time_machine'](raw=raw)
    assert len(flat)==len(ids)==170580 and names==vocab.idx_to_token==book['char_tokens']
    assert ids==[names.index(ch) for ch in flat]
    assert ''.join(vocab.to_tokens(ids))==flat
    prefix,cv = H['load_corpus_time_machine'](67,raw)
    assert prefix==ids[:67] and cv.idx_to_token==names
    assert len(' '.join(filter(None,lines)))==173427
    train,held = ' '.join(lines[:2576]).split(), ' '.join(lines[2576:]).split()
    train_count = independent_counts(train)
    held_count = independent_counts(held)
    for row in metrics['experiment']['word_thresholds']:
        n = row['min_freq']
        unseen = sum(freq for token,freq in held_count.items() if token not in train_count)
        cut = sum(freq for token,freq in held_count.items() if token in train_count and train_count[token]<n)
        assert row['full_vocab']==1+sum(v>=n for v in count.values())
        assert row['train_vocab']==1+sum(v>=n for v in train_count.values())
        assert row['held_tokens']==len(held)==6392 and row['held_unk']==unseen+cut
        assert row['held_unseen']==unseen and row['held_filtered']==cut
        assert row['full_unk']==sum(v for v in count.values() if v<n)
        assert row['held_unk_rate']==(unseen+cut)/len(held)
    return lines,train,held


def verify_bpe(train,held,record):
    counts = independent_counts(train)
    forms = {w:list(w) for w in counts}
    # 逐轮独立算频次并找胜者，按左到右互不重叠的位置改写。
    def substitute(pieces,pair):
        selected = set()
        for i in range(len(pieces)-1):
            if i-1 not in selected and pieces[i:i+2]==pair:
                selected.add(i)
        return [pieces[j]+pieces[j+1] if j in selected else pieces[j]
                for j in range(len(pieces)) if j-1 not in selected]
    for pair in record['merges']:
        pressure = defaultdict(int)
        for word,pieces in forms.items():
            for i in range(len(pieces)-1):
                pressure[tuple(pieces[i:i+2])] += counts[word]
        largest = max(pressure.values())
        assert tuple(pair)==min(k for k,v in pressure.items() if v==largest)
        forms = {w:substitute(seq,pair) for w,seq in forms.items()}
    encoded_held = {w:list(w) for w in set(held)}
    for pair in record['merges']:
        encoded_held = {w:substitute(seq,pair) for w,seq in encoded_held.items()}
    for key,expected in encoded_held.items():
        assert ''.join(expected)==key and tuple(expected)==H['bpe_word'](key,[tuple(p) for p in record['merges']])
    alphabet = {p for w in forms for p in forms[w]}
    segments = [p for w in held for p in encoded_held[w]]
    assert record['vocab_size']==len(alphabet)+1==107
    assert len(segments)==record['held_pieces']==16026
    assert sum(p not in alphabet for p in segments)==record['held_unk']==0
    assert sum(len(seq)*counts[w] for w,seq in forms.items())==record['train_pieces']
    return dict(merges_checked=len(record['merges']),held_words_checked=len(held),held_pieces=len(segments))


def boundaries():
    v = H['Vocab']([['a','b'],['b','a'],['rare']],2,H['RESERVED'])
    assert list(v.token_to_idx)==['<unk>','<pad>','<bos>','<eos>','a','b']
    assert v[['rare','new','a']]==[0,0,4]
    assert v.to_tokens([0,0,4])==['<unk>','<unk>','a']
    assert H['Vocab']().idx_to_token==['<unk>']
    assert H['count_corpus'](['a','ab'])=={'a':1,'ab':1}
    failures = 0
    for call in (lambda:v.to_tokens(-1),lambda:v.to_tokens(len(v)),lambda:v.to_tokens(1.5),
                 lambda:H['Vocab'](min_freq=-1),lambda:H['Vocab'](reserved=['<pad>','<pad>']),
                 lambda:H['Vocab'](reserved=['<unk>']),lambda:H['tokenize'](['x'],'missing')):
        try:
            call()
        except ValueError:
            failures += 1
    assert failures == 7
    return dict(stable_tie_order=True,unknown_collisions_expected=True,invalid_cases=failures)


def practice():
    text = (ROOT/'docs/ch08/02_text_preprocessing.md').read_text(encoding='utf-8')
    blocks = []
    for page in text.split('<!-- pagebreak -->'):
        if re.search(r'## (13|14|15|16|17) ',page):
            blocks.extend(re.findall(r'```python\n(.*?)\n```',page.split('<!-- side -->')[0],re.S))
    assert len(blocks)==5
    # The example resolves data relative to __file__. Give it a path in .cache
    # without touching the learner's file.
    target = str(ROOT / '.cache/8_2_follow.py')
    follow_code = compile('\n\n'.join(blocks), target, 'exec')
    follow = io.StringIO()
    with contextlib.redirect_stdout(follow):
        exec(follow_code, {'__file__': target, '__name__': '__main__'})
    ex = ROOT/'exercises/ch08/02_text_preprocessing.py'
    source = ex.read_bytes()
    refs = {'clean = None':"clean = [re.sub('[^A-Za-z]+',' ',s).strip().lower() for s in raw]",
            'words = None':'words = [s.split() for s in clean]',
            'chars = None':'chars = [list(s) for s in clean]',
            'counts = None':'counts = Counter(w for row in words[:4] for w in row)',
            'idx_to_token = None':'idx_to_token = reserved+[w for w,n in sorted_counts if n>=2]',
            'token_to_idx = None':'token_to_idx = {w:i for i,w in enumerate(idx_to_token)}',
            'ids = None':'ids = [token_to_idx.get(w,0) for w in sample]',
            'decoded = None':'decoded = [idx_to_token[i] for i in ids]'}
    code = source.decode('utf-8')
    for old,new in refs.items():
        assert code.count(old)==1
        code = code.replace(old,new)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code,str(ex),'exec'),{'__file__':str(ex),'__name__':'__main__'})
    assert ex.read_bytes()==source and len(re.findall(r'# TODO [1-8]：',source.decode('utf-8')))==8
    return dict(follow=follow.getvalue(),exercise_reference=captured.getvalue(),remaining_todos=8)


if __name__ == '__main__':
    raw = H['checked_bytes']()
    report = ROOT/'reports/ch08'
    metrics = json.loads((report/'8_2_metrics.json').read_text(encoding='utf-8'))
    assert metrics['source']['sha1']==hashlib.sha1(raw).hexdigest()==H['SHA1']
    assert metrics['source']['sha256']==hashlib.sha256(raw).hexdigest()
    lines,train,held = verify_corpus(raw,metrics)
    result = dict(lines=len(lines),characters_roundtrip=170580,stable_word_ids=True,
                  frequency_rows_checked=8,bpe=verify_bpe(train,held,metrics['experiment']['bpe']),
                  boundaries=boundaries(),practice=practice())
    (report/'8_2_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(report/'8_2_verification.json')
