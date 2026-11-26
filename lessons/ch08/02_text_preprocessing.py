"""8.2 文本预处理：读取、词元化、词表、编码和留出区间比较。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/timemachine.txt'
REPORT = ROOT / 'reports/ch08'
SHA1 = '090b5e7e70c295757f55df93cb0a180b9691891a'
URL = 'https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
MIRROR = 'https://d2l-data.s3.amazonaws.com/timemachine.txt'
RESERVED = ('<pad>','<bos>','<eos>')
THRESHOLDS = (0,1,2,3,5,10,20,50)


def checked_bytes(download=False):
    """已有文件只读且先验哈希匹配。显式下载时写入忽略的 data 目录。"""
    if not DATA.is_file():
        if not download:
            raise FileNotFoundError('缺少 data/timemachine.txt。加 --download 获取并校验教材语料。')
        error = None
        for url in (URL,MIRROR):
            try:
                with urlopen(url, timeout=15) as response:
                    raw = response.read(300000)
            except OSError as exc:
                error = exc
                continue
            if hashlib.sha1(raw).hexdigest() != SHA1:
                raise ValueError('下载文件与教材 SHA-1 不匹配，未写入')
            DATA.parent.mkdir(parents=True,exist_ok=True)
            with DATA.open('xb') as file:
                file.write(raw)
            break
        else:
            raise OSError('两处教材存储地址均未成功') from error
    raw = DATA.read_bytes()
    if hashlib.sha1(raw).hexdigest() != SHA1:
        raise ValueError('本地教材语料 SHA-1 不匹配，请检查现有文件')
    return raw


def normalize(line):
    """教材口径：连续非 ASCII 字母合为一个空格，去首尾空格，转小写。"""
    return re.sub(r'[^A-Za-z]+', ' ', line).strip().lower()


def read_lines(raw):
    return [normalize(line) for line in raw.decode('utf-8').splitlines()]


def tokenize(lines, mode='word'):
    if mode == 'word':
        return [line.split() for line in lines]
    if mode == 'char':
        return [list(line) for line in lines]
    raise ValueError('教材词元模式仅接受 word 或 char')


def count_corpus(tokens):
    counter = Counter()
    for row in tokens:
        if isinstance(row,str):
            counter[row] += 1
        else:
            counter.update(row)
    return counter


class Vocab:
    def __init__(self, tokens=(), min_freq=0, reserved=()):
        if not isinstance(min_freq, Integral) or min_freq < 0:
            raise ValueError('min_freq 应为非负整数')
        self.idx_to_token = ['<unk>', *reserved]
        if len(set(self.idx_to_token)) != len(self.idx_to_token):
            raise ValueError('保留词元须互异，并避开 <unk>')
        self.token_to_idx = {token:i for i,token in enumerate(self.idx_to_token)}
        # Python 排序稳定。并列频次沿 Counter 中的首次出现次序。
        self.token_freqs = sorted(count_corpus(tokens).items(),key=lambda entry: entry[1],reverse=True)
        for token,freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)

    @property
    def unk(self):
        return 0

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if isinstance(tokens,(list,tuple)):
            return [self[token] for token in tokens]
        return self.token_to_idx.get(tokens,self.unk)

    def to_tokens(self, ids):
        if isinstance(ids,(list,tuple)):
            return [self.to_tokens(i) for i in ids]
        if isinstance(ids,bool) or not isinstance(ids,Integral) or not 0 <= ids < len(self):
            raise ValueError('解码索引须在 [0, 词表大小) 内')
        return self.idx_to_token[ids]


def flatten(rows):
    return [token for row in rows for token in row]


def load_corpus_time_machine(max_tokens=-1, raw=None):
    rows = tokenize(read_lines(checked_bytes() if raw is None else raw),'char')
    vocab = Vocab(rows)
    ids = vocab[flatten(rows)]
    return (ids[:max_tokens] if max_tokens > 0 else ids), vocab


def regex_tokens(text):
    """在未做英文清洗的字符串上切分，保留数字、标点与英文缩约。"""
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+(?:\.\d+)?|[^\w\s]|_",text)


def dictionary_tokens(text, lexicon):
    """前向最大匹配。未知片段回退单字符。使用自建小词典。"""
    widest = max(map(len,lexicon),default=1)
    position, out = 0, []
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        longest = min(widest,len(text)-position)
        match = next((text[position:position+n] for n in range(longest,0,-1)
                      if text[position:position+n] in lexicon),text[position])
        out.append(match)
        position += len(match)
    return out


def merge_pair(pieces, pair):
    out = []
    i = 0
    while i < len(pieces):
        if i+1 < len(pieces) and (pieces[i],pieces[i+1]) == pair:
            out.append(pieces[i]+pieces[i+1])
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return tuple(out)


def train_bpe(words, limit):
    """词内字符 BPE。用训练词频计权，同频按词对字典序。"""
    counts = Counter(words)
    forms = {word:tuple(word) for word in counts}
    rules = []
    for _ in range(limit):
        pairs = Counter()
        for word,pieces in forms.items():
            for pair in zip(pieces[:-1],pieces[1:]):
                pairs[pair] += counts[word]
        if not pairs:
            break
        pair = min(pairs,key=lambda p:(-pairs[p],p))
        rules.append(pair)
        forms = {word:merge_pair(pieces,pair) for word,pieces in forms.items()}
    return rules


def bpe_word(word, rules):
    pieces = tuple(word)
    for pair in rules:
        pieces = merge_pair(pieces,pair)
    return pieces


def unknown_counts(tokens, vocab, fitted_counts):
    counts = Counter(tokens)
    unseen = sum(n for word,n in counts.items() if word not in fitted_counts)
    filtered = sum(n for word,n in counts.items() if word in fitted_counts and word not in vocab.token_to_idx)
    unknown = sum(i==vocab.unk for i in vocab[tokens])
    assert unknown == unseen+filtered
    return dict(tokens=len(tokens),types=len(counts),unknown=unknown,
                unseen=unseen,filtered=filtered,unknown_rate=unknown/max(1,len(tokens)))


def frequency_sweep(word_rows, train_line_count):
    """全书敏感性与训练前缀的留出诊断使用不同词表。"""
    train = word_rows[:train_line_count]
    tail = flatten(word_rows[train_line_count:])
    full_words, train_words = flatten(word_rows), flatten(train)
    full_count, train_count = Counter(full_words), Counter(train_words)
    results = []
    for threshold in THRESHOLDS:
        whole = Vocab(full_words,threshold)
        prefix = Vocab(train_words,threshold)
        reference = unknown_counts(full_words,whole,full_count)
        held_out = unknown_counts(tail,prefix,train_count)
        results.append(dict(min_freq=threshold,full_vocab=len(whole),full_unk=reference['unknown'],
                            full_unk_rate=reference['unknown_rate'],train_vocab=len(prefix),held_tokens=len(tail),
                            held_unk=held_out['unknown'],held_unseen=held_out['unseen'],
                            held_filtered=held_out['filtered'],held_unk_rate=held_out['unknown_rate']))
    return results


def subword_experiment(word_rows, train_line_count):
    train, held = flatten(word_rows[:train_line_count]), flatten(word_rows[train_line_count:])
    rules = train_bpe(train,80)
    pieces_for = {word:bpe_word(word,rules) for word in set(train+held)}
    train_pieces = [p for word in train for p in pieces_for[word]]
    held_pieces = [p for word in held for p in pieces_for[word]]
    # 空格通过单独的词边界保留。比较按词产生的piece，故对齐不含空格/换行。
    vocab = Vocab(train_pieces)
    return dict(merges=[list(p) for p in rules],train_pieces=len(train_pieces),vocab_size=len(vocab),
                held_words=len(held),held_pieces=len(held_pieces),
                held_characters=sum(map(len,held)),held_unk=sum(i==0 for i in vocab[held_pieces]),
                all_words_roundtrip=all(''.join(pieces)==word for word,pieces in pieces_for.items()),
                example_pieces={w:list(bpe_word(w,rules)) for w in ('time','traveller','travelling','space','moonlight')})


def run(raw,min_freq=0,max_tokens=-1):
    lines = read_lines(raw)
    word_rows = tokenize(lines)
    word_vocab = Vocab(word_rows,min_freq)
    corpus, char_vocab = load_corpus_time_machine(max_tokens,raw)
    full_chars = sum(map(len,lines))
    train_end = 8*len(lines)//10
    demo_words = tokenize([normalize("Time-travel isn't 8.2!")])[0]
    regex_demo = regex_tokens("Time-travel isn't 8.2!")
    chinese = '研究生命起源'
    lexicon = {'研究生','研究','生命','起源'}
    special = Vocab([['time','time','walk'],['time','train']],min_freq=2,reserved=RESERVED)
    demo_ids = special[['<bos>','time','walk','never','<eos>','<pad>']]
    return dict(
        source=dict(local='data/timemachine.txt',url=URL,sha1=hashlib.sha1(raw).hexdigest(),
                    sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)),
        book=dict(lines=len(lines),empty_lines=sum(not x for x in lines),
                  first_lines={str(i):lines[i] for i in (0,10)},
                  word_count=sum(map(len,word_rows)),word_types=len(count_corpus(word_rows)),
                  word_vocab=len(word_vocab),min_freq=min_freq,top_words=word_vocab.token_freqs[:12],
                  first_words=word_rows[0],first_word_ids=word_vocab[word_rows[0]],
                  line10_words=word_rows[10],line10_word_ids=word_vocab[word_rows[10]],
                  characters=full_chars,characters_with_line_space=len(' '.join(filter(None,lines))),
                  char_vocab=len(char_vocab),char_tokens=char_vocab.idx_to_token,
                  corpus_length=len(corpus),max_tokens=max_tokens,corpus_head=corpus[:24]),
        experiment=dict(train_lines=train_end,held_out_lines=len(lines)-train_end,
                        word_thresholds=frequency_sweep(word_rows,train_end),
                        bpe=subword_experiment(word_rows,train_end)),
        examples=dict(english_cleaned=demo_words,regex=regex_demo,
                      chinese_longest=dictionary_tokens(chinese,lexicon),
                      chinese_alternative=['研究','生命','起源'],
                      utf8_bytes=list('a中'.encode('utf-8')),
                      special_ids=demo_ids,decoded=special.to_tokens(demo_ids),special_vocab=special.idx_to_token))


def save(result):
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'8_2_metrics.json').open('w',encoding='utf-8') as file:
        json.dump(result,file,ensure_ascii=False,indent=2)
        file.write('\n')
    rows = result['experiment']['word_thresholds']
    with (REPORT/'8_2_frequency_sweep.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='缺少本地语料时，从教材存储地址读取并校验')
    parser.add_argument('--guided',action='store_true',help='按步骤打印结果')
    parser.add_argument('--record',action='store_true',help='写出完整默认实验的指标和阈值表')
    parser.add_argument('--min-freq',type=int,default=0,help='教材版单词词表的最低频次')
    parser.add_argument('--max-tokens',type=int,default=-1,help='输出字符索引上限，-1 表示全部')
    args = parser.parse_args()
    if args.min_freq<0 or args.max_tokens==0 or args.max_tokens < -1:
        parser.error('min-freq 应为非负整数，max-tokens 应为正整数或 -1')
    if args.record and (args.min_freq!=0 or args.max_tokens!=-1):
        parser.error('--record 写入完整默认基线。试验参数请省略 --record')
    result = run(checked_bytes(args.download),args.min_freq,args.max_tokens)
    if args.record:
        save(result)
    if args.guided:
        book = result['book']
        print('行数',book['lines'],'词元',book['word_count'],'单词词表',book['word_vocab'])
        print('首行词元',book['first_words'])
        print('首行索引',book['first_word_ids'])
        print('字符序列',book['corpus_length'],'字符词表',book['char_vocab'])
        print('阈值比较')
        for row in result['experiment']['word_thresholds']:
            print(row['min_freq'],row['full_vocab'],f"{row['held_unk_rate']:.2%}")
        print('三个扩展例子',result['examples']['regex'],result['examples']['chinese_longest'],
              result['experiment']['bpe']['example_pieces']['travelling'])
    else:
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
