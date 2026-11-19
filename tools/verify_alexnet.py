"""7.1 独立NumPy小通道AlexNet核对、实际文档跟做及内存练习参考。

不重跑正式大训练，不填写练习源文件。数值模型保留完整224空间几何、五卷积和三FC，
仅缩通道/隐藏宽度；Dropout关闭时独立检查前向、全部梯度与尾批SGD。
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
H = runpy.run_path(str(ROOT / "lessons/ch07/01_alexnet.py"))


def numpy_pass(net, parameters, X, labels, backward=True):
    a, cache = X, []
    for index, layer in enumerate(net):
        key = str(index)
        if isinstance(layer, nn.Conv2d):
            kh, kw = layer.kernel_size
            sh, sw = layer.stride
            ph, pw = layer.padding
            padded = np.pad(a, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
            patches = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
            cache.append(("conv", key, a.shape, (ph, pw, sh, sw), patches))
            a = np.einsum("nchwkl,ockl->nohw", patches, parameters[key + ".weight"]) + parameters[key + ".bias"][None, :, None, None]
        elif isinstance(layer, nn.ReLU):
            cache.append(("relu", a > 0))
            a = np.maximum(a, 0)
        elif isinstance(layer, nn.MaxPool2d):
            k, s = layer.kernel_size, layer.stride
            patches = np.lib.stride_tricks.sliding_window_view(a, (k, k), axis=(2, 3))[:, :, ::s, ::s]
            shape = patches.shape[:4]
            flat = patches.reshape(*shape, k * k)
            ids = flat.argmax(-1)
            cache.append(("pool", a.shape, k, s, ids))
            a = flat.max(-1)
        elif isinstance(layer, nn.Flatten):
            cache.append(("flat", a.shape))
            a = a.reshape(len(a), -1)
        elif isinstance(layer, nn.Linear):
            cache.append(("linear", key, a))
            a = a @ parameters[key + ".weight"].T + parameters[key + ".bias"]
        elif not isinstance(layer, nn.Dropout):
            raise TypeError(type(layer))
    logits = a.copy()
    shifted = a - a.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    ce = -logp[np.arange(len(labels)), labels].mean()
    if not backward:
        return ce, logits
    grad = np.exp(logp)
    grad[np.arange(len(labels)), labels] -= 1
    grad /= len(labels)
    gradients = {}
    for item in reversed(cache):
        kind = item[0]
        if kind == "linear":
            _, key, before = item
            gradients[key + ".weight"] = grad.T @ before
            gradients[key + ".bias"] = grad.sum(0)
            grad = grad @ parameters[key + ".weight"]
        elif kind == "flat":
            grad = grad.reshape(item[1])
        elif kind == "relu":
            grad = grad * item[1]
        elif kind == "pool":
            _, shape, k, s, ids = item
            before_grad = np.zeros(shape)
            oh, ow = grad.shape[2:]
            for i in range(k):
                for j in range(k):
                    before_grad[:, :, i:i + oh * s:s, j:j + ow * s:s] += grad * (ids == i * k + j)
            grad = before_grad
        else:
            _, key, shape, (ph, pw, sh, sw), patches = item
            weight = parameters[key + ".weight"]
            gradients[key + ".weight"] = np.einsum("nohw,nchwkl->ockl", grad, patches)
            gradients[key + ".bias"] = grad.sum(axis=(0, 2, 3))
            padded_grad = np.zeros((shape[0], shape[1], shape[2] + 2 * ph, shape[3] + 2 * pw))
            oh, ow = grad.shape[2:]
            for i in range(weight.shape[2]):
                for j in range(weight.shape[3]):
                    padded_grad[:, :, i:i + oh * sh:sh, j:j + ow * sw:sw] += np.einsum("nohw,oc->nchw", grad, weight[:, :, i, j])
            grad = padded_grad[:, :, ph:ph + shape[2], pw:pw + shape[3]]
    return ce, logits, gradients, grad


def numerical():
    H["COMMON"]["configure"]("cpu")
    cfg = dict(H["BASE"], channels=[2, 3, 4, 4, 3], hidden=[8, 6], dropout=0.)
    net = H["build_net"](cfg).double().eval()
    p = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
    generator = torch.Generator().manual_seed(7107)
    X = torch.rand(3, 1, 224, 224, generator=generator, dtype=torch.float64).requires_grad_()
    labels = torch.tensor([1, 4, 7])
    logits = net(X)
    ce = F.cross_entropy(logits, labels)
    ce.backward()
    loss_np, logits_np, gradients, gx = numpy_pass(net, p, X.detach().numpy(), labels.numpy())
    errors = dict(logits=float(np.max(np.abs(logits_np - logits.detach().numpy()))), ce=abs(loss_np - ce.item()),
                  input_gradient=float(np.max(np.abs(gx - X.grad.numpy()))),
                  parameter_gradient=max(float(np.max(np.abs(gradients[k] - v.grad.numpy()))) for k, v in net.named_parameters()))
    finite = []
    for key in p:
        coord = np.unravel_index(np.argmax(np.abs(gradients[key])), p[key].shape)
        old, eps = p[key][coord], 1e-6
        p[key][coord] = old + eps
        high = numpy_pass(net, p, X.detach().numpy(), labels.numpy(), False)[0]
        p[key][coord] = old - eps
        low = numpy_pass(net, p, X.detach().numpy(), labels.numpy(), False)[0]
        p[key][coord] = old
        finite.append(abs((high - low) / (2 * eps) - gradients[key][coord]))
    errors["central_difference_16_coordinates"] = max(finite)
    optimizer = torch.optim.SGD(net.parameters(), lr=.01)
    for ids in ([0, 1], [2]):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(net(X.detach()[ids]), labels[ids]).backward()
        optimizer.step()
        grads = numpy_pass(net, p, X.detach().numpy()[ids], labels.numpy()[ids])[2]
        for key in p:
            p[key] -= .01 * grads[key]
    errors["sgd_two_updates_tail1"] = max(float(np.max(np.abs(p[k] - v.detach().numpy()))) for k, v in net.named_parameters())
    assert max(errors.values()) < 1e-7, errors
    # 张量Resize路径与torchvision相同输入类型/参数的实现对照。
    from torchvision.transforms.functional import resize
    source = torch.rand(2, 1, 28, 28, generator=generator)
    actual = H["prepare"](source, H["BASE"], torch.device("cpu"), dict(mean=0., std=1.))
    errors["resize_vs_torchvision_tensor"] = float((actual - resize(source, [224, 224], antialias=True)).abs().max())
    assert errors["resize_vs_torchvision_tensor"] == 0
    probe = nn.Dropout(.5)
    ones = torch.ones(1000)
    with torch.no_grad():
        a, b = probe(ones), probe(ones)
    assert not torch.equal(a, b)
    probe.eval()
    assert torch.equal(probe(ones), ones)
    return dict(config=cfg, parameters=sum(v.numel() for v in net.parameters()), errors=errors,
                dropout_train_random_even_in_no_grad=True, dropout_eval_identity=True)


def practice():
    text = (ROOT / "docs/ch07/01_alexnet.md").read_text(encoding="utf8")
    blocks = []
    for page in text.split("<!-- pagebreak -->"):
        if re.search(r"## (14|15|16|17) ", page):
            main = page.split("<!-- side -->")[0]
            blocks.extend(re.findall(r"```python\n(.*?)\n```", main, re.S))
    path = ROOT / ".cache/7_1_follow.py"
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf8")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        runpy.run_path(str(path), run_name="__main__")
    exercise = ROOT / "exercises/ch07/01_alexnet.py"
    original = exercise.read_bytes()
    source = original.decode("utf8")
    assert len(re.findall(r"# TODO [1-8]：", source)) == 8
    substitutions = {"conv1 = None": "conv1 = nn.Conv2d(1, 16, 3, padding=1)",
                     "conv2 = None": "conv2 = nn.Conv2d(16, 32, 3, padding=1)",
                     "conv3 = None": "conv3 = nn.Conv2d(32, 64, 3, padding=1)",
                     "flatten = None": "flatten = nn.Flatten()", "loss = None": "loss = F.cross_entropy(logits, y)",
                     "pass  # TODO 6": "loss.backward()  # TODO 6", "pass  # TODO 7": "optimizer.step()  # TODO 7",
                     "prediction = None  # TODO 8": "prediction = logits.argmax(1)  # TODO 8"}
    for old, new in substitutions.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    reference = io.StringIO()
    with contextlib.redirect_stdout(reference):
        exec(compile(source, str(exercise), "exec"), {"__name__": "__main__", "__file__": str(exercise)})
    assert exercise.read_bytes() == original
    page19 = next(page for page in text.split("<!-- pagebreak -->") if "## 19 " in page)
    block = re.findall(r"```python\n(.*?)\n```", page19, re.S)[0]
    common = H["COMMON"]
    splits, _ = common["load_data"](include_test=False)
    namespace = dict(helper=H, device=torch.device("cpu"),
                     train=common["balanced_subset"](splits["train"], 37, 7101),
                     validation=common["balanced_subset"](splits["validation"], 20, 7102))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(block, namespace)
    return dict(follow_blocks=len(blocks), follow_stdout=stdout.getvalue(), exercise_reference_stdout=reference.getvalue(),
                exercise_source_unchanged=True, variant_snippet_updates=namespace["result"]["updates"])


def checkpoint_checks():
    device = H["COMMON"]["configure"]("auto")
    splits, _ = H["COMMON"]["load_data"](include_test=False)
    X = torch.stack([splits["validation"][i][0] for i in range(2)])
    checks = {}
    for prefix in ("7_1", "7_1_native"):
        path = ROOT / "artifacts/ch07" / (prefix + "_epoch15.pt")
        if not path.exists():
            checks[prefix] = dict(skipped="本地尚无第15轮检查点")
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        net = H["build_net"](state["config"]).eval()
        net.load_state_dict(state["model"])
        report = json.loads((ROOT / "reports/ch07" / (prefix + "_metrics.json")).read_text(encoding="utf8"))
        assert H["parameter_hash"](net) == report["final_parameter_sha256"]
        inputs = H["prepare"](X, state["config"], torch.device("cpu"), dict(mean=0., std=1.))
        with torch.no_grad():
            cpu_logits = net(inputs)
            net = net.to(device)
            device_logits = net(inputs.to(device)).cpu()
        difference = (cpu_logits - device_logits).abs().max().item()
        assert difference < 1e-4
        assert torch.equal(cpu_logits.argmax(1), device_logits.argmax(1))
        checks[prefix] = dict(epoch=state["epoch"], validation_positions=[0, 1],
                             max_logit_difference=difference, cpu_predictions=cpu_logits.argmax(1).tolist(),
                             device=str(device), parameter_hash_matches=True)
        del net, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return checks


def main():
    result = dict(numerical=numerical(), practice=practice(), checkpoint_recovery=checkpoint_checks())
    H["write_json"]("7_1_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
