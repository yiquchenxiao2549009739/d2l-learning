"""7.7 DenseNet。默认完整教材配置，固定第10轮测试。见 --help。"""
import argparse
from collections import OrderedDict
from contextlib import contextmanager, nullcontext, redirect_stdout
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RES = runpy.run_path(str(ROOT / 'lessons/ch07/06_resnet.py'))
COMMON = RES['COMMON']
write_json, write_csv, plot_setup = (RES[key] for key in ('write_json', 'write_csv', 'plot_setup'))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

BASE = dict(stem=64, growth=32, blocks=[4, 4, 4, 4], compression=.5, pool='avg',
            bottleneck=False, recompute=False, size=96, lr=.1, batch_size=256)
PAPER = {121: [6, 12, 24, 16], 169: [6, 12, 32, 32],
         201: [6, 12, 48, 32], 264: [6, 12, 64, 48]}


@contextmanager
def temporary_bn_buffers(module):
    """重算期间更新独立副本。退出后恢复原buffer对象及已提交的一次更新。"""
    slots = [(m, key, value) for m in module.modules() if isinstance(m, nn.BatchNorm2d)
             for key, value in m.named_buffers(recurse=False)]
    try:
        for layer, name, buffer in slots:
            setattr(layer, name, buffer.detach().clone())
        yield
    finally:
        for layer, name, buffer in slots:
            setattr(layer, name, buffer)


class DenseBlock(nn.Module):
    def __init__(self, layers, channels, growth, bottleneck=False, recompute=False, bias=True, device='cpu'):
        super().__init__()
        self.recompute = recompute
        self.units = nn.ModuleList()
        for i in range(layers):
            width = channels + i * growth
            ops = [nn.BatchNorm2d(width, device=device), nn.ReLU()]
            if bottleneck:
                ops.extend([nn.Conv2d(width, 4 * growth, 1, bias=bias, device=device),
                            nn.BatchNorm2d(4 * growth, device=device), nn.ReLU()])
                width = 4 * growth
            ops.append(nn.Conv2d(width, growth, 3, padding=1, bias=bias, device=device))
            self.units.append(nn.Sequential(*ops))
        self.out_channels = channels + layers * growth

    def forward(self, X):
        if self.recompute and self.training and torch.is_grad_enabled():
            features = [X]
            for unit in self.units:
                # 捕获当前unit，避免反传时闭包读到最后一层。
                def append_feature(*parts, unit=unit):
                    return unit(torch.cat(parts, dim=1)) if len(parts) > 1 else unit(parts[0])
                def contexts(unit=unit):
                    return nullcontext(), temporary_bn_buffers(unit)
                features.append(checkpoint(append_feature, *features, use_reentrant=False, context_fn=contexts))
            return torch.cat(features, dim=1)
        for unit in self.units:
            Y = unit(X)
            X = torch.cat((X, Y), dim=1)
        return X


def transition(ci, co, pool='avg', bias=True, device='cpu'):
    pooling = nn.AvgPool2d(2, 2) if pool == 'avg' else nn.MaxPool2d(2, 2)
    return nn.Sequential(nn.BatchNorm2d(ci, device=device), nn.ReLU(),
                         nn.Conv2d(ci, co, 1, bias=bias, device=device), pooling)


def build_net(config=None, device='cpu'):
    cfg = dict(BASE, **(config or {}))
    channels = cfg['stem']
    # 教材默认Conv含bias，论文深度变体采用常见的无卷积bias口径。
    bias = not cfg['bottleneck']
    layers = [('stem', nn.Sequential(nn.Conv2d(1, channels, 7, 2, 3, bias=bias, device=device),
                                      nn.BatchNorm2d(channels, device=device), nn.ReLU(), nn.MaxPool2d(3, 2, 1)))]
    for index, count in enumerate(cfg['blocks'], 1):
        block = DenseBlock(count, channels, cfg['growth'], cfg['bottleneck'], cfg['recompute'], bias, device)
        layers.append((f'dense{index}', block))
        channels = block.out_channels
        if index < len(cfg['blocks']):
            reduced = max(1, int(channels * cfg['compression']))
            layers.append((f'transition{index}', transition(channels, reduced, cfg['pool'], bias, device)))
            channels = reduced
    layers.extend([('bn', nn.BatchNorm2d(channels, device=device)), ('relu', nn.ReLU()),
                   ('gap', nn.AdaptiveAvgPool2d(1)), ('flatten', nn.Flatten()),
                   ('classifier', nn.Linear(channels, 10, device=device))])
    net = nn.Sequential(OrderedDict(layers))
    if device != 'meta':
        with torch.random.fork_rng(devices=[]):
            for name, layer in net.named_modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    seed = 7700 + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
                    torch.manual_seed(seed)
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    return net


def inspect_net(cfg):
    net = build_net(cfg, device='meta').eval()
    outer, weighted = [], []
    def hook(name):
        def record(layer, inp, out):
            fan_in = layer.weight[0].numel()
            weighted.append(dict(name=name, input=list(inp[0].shape), output=list(out.shape),
                                 weight_multiplications=out.numel() * fan_in,
                                 parameters=sum(p.numel() for p in layer.parameters())))
        return record
    for name, layer in net.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.register_forward_hook(hook(name))
    X = torch.zeros(1, 1, cfg['size'], cfg['size'], device='meta')
    with torch.no_grad():
        for name, layer in net.named_children():
            X = layer(X)
            outer.append(dict(name=name, shape=list(X.shape)))
    return dict(config=cfg, outer=outer, weighted_layers=weighted, parameters=sum(p.numel() for p in net.parameters()),
                weight_multiplications=sum(r['weight_multiplications'] for r in weighted),
                conv_count=len(weighted)-1, dense_units=sum(cfg['blocks']))


def inspect_all(record):
    official = {str(n): inspect_net(dict(BASE, blocks=blocks, size=224, bottleneck=True))
                for n, blocks in PAPER.items()}
    result = dict(lesson96=inspect_net(BASE), lesson224=inspect_net(dict(BASE, size=224)), paper=official,
                  note='论文表1块数/BC结构，灰度输入与十类头适配；乘法只数Conv/Linear权重')
    if record:
        write_json('7_7_architecture.json', result)
        write_csv('7_7_layers.csv', result['lesson96']['weighted_layers'])
    return result


def fit(cfg, splits, epochs, device, name, record=False, use_test=False):
    net = build_net(cfg).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg['lr'])
    loader = COMMON['make_loader'](splits['train'], shuffle=True, batch_size=cfg['batch_size'])
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    result = dict(config=cfg, runtime=COMMON['runtime_metadata'](device), epochs=epochs,
                  model_seed_base=7700, loader_seed=3603, initial_state_sha256=RES['state_hash'](net),
                  parameters=sum(p.numel() for p in net.parameters()), train_samples=len(splits['train']),
                  validation_samples=len(splits['validation']), history=[], test_evaluated=use_test)
    started = time.perf_counter()
    for epoch in range(epochs+1):
        row = dict(epoch=epoch)
        if epoch:
            row.update(RES['train_epoch'](net, loader, optimizer, cfg, device))
        val = RES['evaluate'](net, splits['validation'], cfg, device)
        row.update({f'validation_{k}': v for k, v in val.items()})
        if epoch == epochs:
            row.update({f'train_eval_{k}': v for k, v in RES['evaluate'](net, splits['train'], cfg, device).items()})
        result['history'].append(row)
        online_ce = f"{row['online_train_ce']:.6f}" if epoch else '-'
        print(f"{name} epoch {epoch:2d} | online CE {online_ce}"
              f" | val CE {val['ce']:.6f} | val {val['accuracy']:.2%}", flush=True)
    if use_test:
        result['test'] = RES['evaluate'](net, splits['test'], cfg, device)
    result.update(updates=sum(r.get('updates', 0) for r in result['history']),
                  last_batch_size=result['history'][-1]['last_batch_size'],
                  singleton_tail_merged=len(splits['train']) % cfg['batch_size'] == 1,
                  final_state_sha256=RES['state_hash'](net),
                  bn_counters=sorted({int(m.num_batches_tracked) for m in net.modules() if isinstance(m, nn.BatchNorm2d)}),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
    if record:
        folder = ROOT / 'artifacts/ch07'
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(config=cfg, epoch=epochs, model={k: v.detach().cpu() for k,v in net.state_dict().items()},
                        optimizer=optimizer.state_dict()), folder / (name + '.pt'))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    result['seconds_including_evaluation_and_save'] = time.perf_counter() - started
    return result


def experiments(device, record):
    splits, hashes = COMMON['load_data'](include_test=False)
    splits = {key: COMMON['balanced_subset'](splits[key], size, seed)
              for key, size, seed in [('train', 2000, 7701), ('validation', 1000, 7702)]}
    base = dict(BASE, stem=32, growth=16, batch_size=128)
    configs = dict(A=base, B=dict(base, pool='max'), C=dict(base, growth=8), D=dict(base, growth=24),
                   E=dict(base, compression=.75), F=dict(base, lr=.05))
    results = {name: fit(cfg, splits, 20, device, name) for name, cfg in configs.items()}
    assert results['A']['initial_state_sha256'] == results['B']['initial_state_sha256'] == results['F']['initial_state_sha256']
    chosen = min(results, key=lambda name: results[name]['history'][-1]['validation_ce'])
    result = dict(results=results, split_hashes=hashes, subset_seeds=[7701,7702],
                  selected_by_final_validation_ce=chosen, test_evaluated=False)
    if record:
        write_json('7_7_experiments.json', result)
        write_csv('7_7_experiments_epochs.csv', [dict(candidate=name, **row) for name, run in results.items() for row in run['history']])
    return result


def benchmark(device, record):
    if device.type != 'cuda':
        raise ValueError('GPU短测需要CUDA')
    rows = []
    for size, model, recompute in [(96,'dense',False),(224,'dense',False),(224,'dense',True),(224,'resnet18',False)]:
        for mode in ('train', 'eval'):
            if recompute and mode == 'eval':
                continue
            torch.cuda.empty_cache()
            cfg = dict(BASE, size=size, recompute=recompute)
            net = (build_net(cfg) if model == 'dense' else RES['build_net']()).to(device)
            net.train(mode == 'train')
            X = torch.rand(8, 1, size, size, device=device, generator=torch.Generator(device=device).manual_seed(7709))
            y = torch.arange(8, device=device)
            opt = torch.optim.SGD(net.parameters(), lr=.01)
            def step():
                if mode == 'train':
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(net(X), y).backward()
                    opt.step()
                else:
                    with torch.inference_mode():
                        net(X)
            for _ in range(2):
                step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                step()
            end.record()
            torch.cuda.synchronize()
            rows.append(dict(model=model, size=size, recompute=recompute, mode=mode, ms=start.elapsed_time(end)/5,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
            del net, opt, X, y
    result = dict(results=rows, batch_size=8, warmup=2, repeats=5, seed=7709, runtime=COMMON['runtime_metadata'](device),
                  protocol='合成CUDA驻留输入，训练含CE/反传/SGD，eval无梯度，不含读取/Resize')
    if record:
        write_json('7_7_benchmark.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    for command in ('inspect','experiments','benchmark'):
        modes.add_argument('--'+command, action='store_true')
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--train-size', type=int)
    p.add_argument('--validation-size', type=int)
    p.add_argument('--guided', action='store_true')
    p.add_argument('--record', action='store_true')
    args = p.parse_args()
    custom = (args.epochs != 10 or args.batch_size != 256
              or args.train_size is not None or args.validation_size is not None)
    if args.epochs < 1 or args.batch_size < 2 or (args.train_size is not None and not 2 <= args.train_size <= 54000) or (args.validation_size is not None and not 1 <= args.validation_size <= 6000):
        p.error('轮数至少1，批量至少2，子集大小须在原划分范围内')
    if custom and (args.inspect or args.experiments or args.benchmark):
        p.error('固定实验模式不能与自定义训练参数合用')
    device = COMMON['configure'](args.device)
    if args.benchmark and device.type != 'cuda':
        p.error('--benchmark 需要CUDA')
    with nullcontext() if args.guided else redirect_stdout(sys.stderr):
        if args.inspect:
            result = inspect_all(args.record)
        elif args.experiments:
            result = experiments(device, args.record)
        elif args.benchmark:
            result = benchmark(device, args.record)
        else:
            splits, hashes = COMMON['load_data'](include_test=not custom)
            for key, size, seed in [('train',args.train_size,7701),('validation',args.validation_size,7702)]:
                if size is not None:
                    splits[key] = COMMON['balanced_subset'](splits[key], size, seed)
            name = '7_7_custom' if custom else '7_7'
            result = fit(dict(BASE, batch_size=args.batch_size), splits, args.epochs, device, name, args.record, not custom)
            result.update(custom=custom, split_seed=3601, split_hashes=hashes)
            if args.record:
                write_json(name+'_metrics.json', result)
                write_csv(name+'_epochs.csv', result['history'])
    if not args.guided:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.inspect:
        print('教材96配置参数', result['lesson96']['parameters'], '卷积', result['lesson96']['conv_count'])
    elif args.experiments:
        print('最低末轮验证CE候选', result['selected_by_final_validation_ce'])
    elif args.benchmark:
        for row in result['results']:
            print(row['model'],row['size'],row['mode'],'重算',row['recompute'],f"{row['ms']:.3f} ms")
    else:
        print('更新',result['updates'],'尾批',result['last_batch_size'],'合并',result['singleton_tail_merged'],'测试',result.get('test'))


if __name__ == '__main__':
    main()
