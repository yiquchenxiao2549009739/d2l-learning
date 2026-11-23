"""7.2：小通道完整几何的独立NumPy核对、文档跟做、内存练习参考与检查点。

NumPy卷积/汇聚/CE引擎复用7.1的独立实现，测试新的五块/八卷积/三FC路径。
不重跑正式大训练，不修改练习源文件，不宣称独立检验了完整亿级参数模型。
"""
import contextlib
import io
import json
import re
import runpy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
H = runpy.run_path(str(ROOT / "lessons/ch07/02_vgg.py"))
ENGINE = runpy.run_path(str(ROOT / "tools/verify_alexnet.py"))["numpy_pass"]


def numerical():
    H["COMMON"]["configure"]("cpu")
    torch.manual_seed(7207)
    blocks, ci = [], 1
    for n, co in zip([1, 1, 2, 2, 2], [2, 3, 4, 4, 4]):
        blocks.append(H["vgg_block"](n, ci, co))
        ci = co
    net = nn.Sequential(*blocks, nn.Flatten(), nn.Linear(196, 8), nn.ReLU(),
                        nn.Dropout(0.), nn.Linear(8, 6), nn.ReLU(), nn.Dropout(0.), nn.Linear(6, 10)).double().eval()
    for j, layer in enumerate(m for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))):
        torch.default_generator.manual_seed(7200 + j)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    flat = nn.Sequential(*(m for m in net.modules() if not list(m.children())))
    p = {k: v.detach().numpy().copy() for k, v in flat.state_dict().items()}
    gen = torch.Generator().manual_seed(7208)
    X = torch.rand(3, 1, 224, 224, generator=gen, dtype=torch.float64).requires_grad_()
    labels = torch.tensor([2, 4, 9])
    logits = net(X)
    torch.testing.assert_close(logits, flat(X))
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    ce_np, logits_np, gradients, gx = ENGINE(flat, p, X.detach().numpy(), labels.numpy())
    errors = dict(logits=float(np.max(np.abs(logits_np - logits.detach().numpy()))), ce=abs(ce_np - loss.item()),
                  input_gradient=float(np.max(np.abs(gx - X.grad.numpy()))),
                  parameter_gradient=max(float(np.max(np.abs(gradients[k] - v.grad.numpy()))) for k, v in flat.named_parameters()))
    finite = []
    for key in p:
        coord = np.unravel_index(np.argmax(np.abs(gradients[key])), p[key].shape)
        old, eps = p[key][coord], 1e-7
        p[key][coord] = old + eps
        high = ENGINE(flat, p, X.detach().numpy(), labels.numpy(), False)[0]
        p[key][coord] = old - eps
        low = ENGINE(flat, p, X.detach().numpy(), labels.numpy(), False)[0]
        p[key][coord] = old
        finite.append(abs((high - low) / (2 * eps) - gradients[key][coord]))
    errors["central_difference_22_coordinates"] = max(finite)
    optimizer = torch.optim.SGD(net.parameters(), lr=.05)
    for ids in ([0, 1], [2]):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(net(X.detach()[ids]), labels[ids]).backward()
        optimizer.step()
        grads = ENGINE(flat, p, X.detach().numpy()[ids], labels.numpy()[ids])[2]
        for key in p:
            p[key] -= .05 * grads[key]
    errors["sgd_two_updates_tail1"] = max(float(np.max(np.abs(p[k] - v.detach().numpy()))) for k, v in flat.named_parameters())
    assert max(errors.values()) < 1e-7, errors
    structures = H["structures"]()
    for name, expected in (("full11", 11), ("full16", 16), ("full19", 19)):
        assert len(structures[name]["weighted_layers"]) == expected
    assert len(structures["full11"]["outer"]) == 13
    return dict(parameters=sum(v.numel() for v in net.parameters()), input_shape=list(X.shape),
                conv_count=8, linear_count=3, errors=errors,
                finite_difference_step=1e-7,
                numpy_engine="reuse independent NumPy convolution/pooling/CE from verify_alexnet.py")


def practice():
    text = (ROOT / "docs/ch07/02_vgg.md").read_text(encoding="utf8")
    blocks = []
    for page in text.split("<!-- pagebreak -->"):
        if re.search(r"## (13|14|15|16|17) ", page):
            blocks.extend(re.findall(r"```python\n(.*?)\n```", page.split("<!-- side -->")[0], re.S))
    path = ROOT / ".cache/7_2_follow.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf8")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        runpy.run_path(str(path), run_name="__main__")
    exercise = ROOT / "exercises/ch07/02_vgg.py"
    original = exercise.read_bytes()
    source = original.decode("utf8")
    assert len(re.findall(r"# TODO [1-8]：", source)) == 8
    replacements = {"conv = None": "conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)",
                    "next_in = None": "next_in = out_channels", "pool = None": "pool = nn.MaxPool2d(2, 2)",
                    "block = None": "block = nn.Sequential(*layers)", "in_features = None": "in_features = 288",
                    "loss = None": "loss = F.cross_entropy(logits, y)", "pass  # TODO 7": "loss.backward()  # TODO 7",
                    "pass  # TODO 8": "optimizer.step()  # TODO 8"}
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    reference = io.StringIO()
    with contextlib.redirect_stdout(reference):
        exec(compile(source, str(exercise), "exec"), {"__name__": "__main__", "__file__": str(exercise)})
    assert exercise.read_bytes() == original
    page21 = next(p for p in text.split("<!-- pagebreak -->") if "## 21 " in p)
    variant = re.findall(r"```python\n(.*?)\n```", page21, re.S)[0]
    variant_out = io.StringIO()
    namespace = dict(helper=H, torch=torch)
    with contextlib.redirect_stdout(variant_out):
        exec(variant, namespace)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in namespace["model"].parameters())
    return dict(follow_blocks=len(blocks), follow_stdout=stdout.getvalue(), exercise_reference_stdout=reference.getvalue(),
                exercise_source_unchanged=True, variant_stdout=variant_out.getvalue())


def checkpoints():
    device = H["COMMON"]["configure"]("auto")
    splits, _ = H["COMMON"]["load_data"](include_test=False)
    X = torch.stack([splits["validation"][i][0] for i in range(2)])
    results = {}
    for prefix in ("7_2", "7_2_96"):
        path = ROOT / "artifacts/ch07" / (prefix + ".pt")
        if not path.exists():
            results[prefix] = dict(skipped="尚无本地检查点")
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        net = H["build_net"](state["config"]).eval()
        net.load_state_dict(state["model"])
        report_path = ROOT / "reports/ch07" / (prefix + "_metrics.json")
        report = json.loads(report_path.read_text(encoding="utf8"))
        assert H["parameter_hash"](net) == report["final_parameter_sha256"]
        images = H["prepare"](X, state["config"], torch.device("cpu"))
        with torch.no_grad():
            cpu_logits = net(images)
            net = net.to(device)
            device_logits = net(images.to(device)).cpu()
        difference = (cpu_logits - device_logits).abs().max().item()
        assert difference < 1e-4
        assert torch.equal(cpu_logits.argmax(1), device_logits.argmax(1))
        results[prefix] = dict(max_logit_difference=difference, device=str(device), parameter_hash_matches=True,
                               validation_positions=[0, 1], predictions=cpu_logits.argmax(1).tolist())
        del net, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def main():
    result = dict(numerical=numerical(), practice=practice(), checkpoint_recovery=checkpoints())
    H["write_json"]("7_2_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
