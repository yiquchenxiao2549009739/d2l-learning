"""6.6 的独立小型数值核对，不重跑正式训练，不填写练习源文件。

使用已有 openmmlab。NumPy 独立实现完整基线前向和反传，检查所有参数梯度、
含尾批的更新及有限差分；顺序执行文档跟做和仅在内存填充的练习参考。
"""
import contextlib
import io
import json
import re
import runpy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
helper = runpy.run_path(str(ROOT / "lessons/ch06/06_lenet.py"))


def numpy_pass(parameters, X, labels, backward=True):
    cache = []
    a = X
    for kind, key, padding in [("conv", "0", 2), ("sig", "", 0), ("avg", "", 0),
                               ("conv", "3", 0), ("sig", "", 0), ("avg", "", 0),
                               ("flat", "", 0), ("linear", "7", 0), ("sig", "", 0),
                               ("linear", "9", 0), ("sig", "", 0), ("linear", "11", 0)]:
        if kind == "conv":
            w = parameters[key + ".weight"]
            padded = np.pad(a, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
            windows = np.lib.stride_tricks.sliding_window_view(padded, w.shape[2:], axis=(2, 3))
            cache.append((kind, key, a.shape, padding, windows))
            a = np.einsum("nchwkl,ockl->nohw", windows, w) + parameters[key + ".bias"][None, :, None, None]
        elif kind == "sig":
            a = 1. / (1. + np.exp(-a))
            cache.append((kind, a))
        elif kind == "avg":
            cache.append((kind, a.shape))
            n, c, h, w = a.shape
            a = a.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))
        elif kind == "flat":
            cache.append((kind, a.shape))
            a = a.reshape(len(a), -1)
        else:
            cache.append((kind, key, a))
            a = a @ parameters[key + ".weight"].T + parameters[key + ".bias"]
    logits = a.copy()
    shifted = a - a.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    ce = -log_prob[np.arange(len(labels)), labels].mean()
    if not backward:
        return ce, logits
    grad = np.exp(log_prob)
    grad[np.arange(len(labels)), labels] -= 1
    grad /= len(labels)
    gradients = {}
    for item in reversed(cache):
        kind = item[0]
        if kind == "linear":
            _, key, before = item
            gradients[key + ".weight"] = grad.T @ before
            gradients[key + ".bias"] = grad.sum(axis=0)
            grad = grad @ parameters[key + ".weight"]
        elif kind == "flat":
            grad = grad.reshape(item[1])
        elif kind == "avg":
            grad = np.repeat(np.repeat(grad, 2, axis=2), 2, axis=3) / 4
        elif kind == "sig":
            grad = grad * item[1] * (1 - item[1])
        else:
            _, key, shape, padding, windows = item
            weight = parameters[key + ".weight"]
            gradients[key + ".weight"] = np.einsum("nohw,nchwkl->ockl", grad, windows)
            gradients[key + ".bias"] = grad.sum(axis=(0, 2, 3))
            padded_grad = np.zeros((shape[0], shape[1], shape[2] + 2 * padding, shape[3] + 2 * padding))
            for i in range(weight.shape[2]):
                for j in range(weight.shape[3]):
                    padded_grad[:, :, i:i + grad.shape[2], j:j + grad.shape[3]] += np.einsum(
                        "nohw,oc->nchw", grad, weight[:, :, i, j])
            grad = padded_grad[:, :, padding:padding + shape[2], padding:padding + shape[3]]
    return ce, logits, gradients, grad


def numerical_checks():
    helper["configure"]("cpu")
    net = helper["build_net"]().double()
    parameters = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
    generator = torch.Generator().manual_seed(6607)
    X = torch.rand(3, 1, 28, 28, generator=generator, dtype=torch.float64).requires_grad_()
    labels = torch.tensor([2, 4, 9])
    logits = net(X)
    ce = F.cross_entropy(logits, labels)
    ce.backward()
    loss_np, logits_np, grads, input_grad = numpy_pass(parameters, X.detach().numpy(), labels.numpy())
    errors = dict(logits=float(np.max(np.abs(logits_np - logits.detach().numpy()))),
                  ce=abs(loss_np - ce.item()), input_gradient=float(np.max(np.abs(input_grad - X.grad.numpy()))),
                  parameter_gradient=max(float(np.max(np.abs(grads[k] - v.grad.numpy())))
                                         for k, v in net.named_parameters()))
    eps, differences = 1e-5, []
    for name, coordinate in [("0.weight", (0, 0, 2, 2)), ("0.bias", (2,)),
                             ("3.weight", (3, 2, 1, 4)), ("3.bias", (4,)),
                             ("7.weight", (3, 9)), ("7.bias", (7,)),
                             ("9.weight", (8, 3)), ("9.bias", (2,)),
                             ("11.weight", (2, 8)), ("11.bias", (9,))]:
        old = parameters[name][coordinate]
        parameters[name][coordinate] = old + eps
        high = numpy_pass(parameters, X.detach().numpy(), labels.numpy(), False)[0]
        parameters[name][coordinate] = old - eps
        low = numpy_pass(parameters, X.detach().numpy(), labels.numpy(), False)[0]
        parameters[name][coordinate] = old
        differences.append(abs((high - low) / (2 * eps) - grads[name][coordinate]))
    errors["central_difference_10_coordinates"] = max(differences)
    optimizer = torch.optim.SGD(net.parameters(), lr=.9)
    for ids in ([0, 1], [2]):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(net(X.detach()[ids]), labels[ids]).backward()
        optimizer.step()
        gradients = numpy_pass(parameters, X.detach().numpy()[ids], labels.numpy()[ids])[2]
        for key in parameters:
            parameters[key] -= .9 * gradients[key]
    errors["two_updates_with_tail1"] = max(float(np.max(np.abs(parameters[k] - v.detach().numpy())))
                                           for k, v in net.named_parameters())
    assert max(errors.values()) < 1e-8, errors
    return errors


def document_and_exercise_checks():
    source = (ROOT / "docs/ch06/06_lenet.md").read_text(encoding="utf8")
    parts = source.split("<!-- pagebreak -->")
    blocks = []
    for page in parts[12:16]:
        blocks.extend(re.findall(r"```python\n(.*?)\n```", page, re.S))
    target = ROOT / ".cache/6_6_follow.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n\n".join(blocks) + "\n", encoding="utf8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(target), run_name="__main__")
    exercise = ROOT / "exercises/ch06/06_lenet.py"
    original = exercise.read_bytes()
    text = original.decode("utf8")
    assert len(re.findall(r"# TODO [1-8]：", text)) == 8
    replacements = {
        "conv1 = None": "conv1 = nn.Conv2d(1, 6, 5, padding=2)",
        "conv2 = None": "conv2 = nn.Conv2d(6, 16, 5)",
        "flatten = None": "flatten = nn.Flatten()",
        "optimizer = None": "optimizer = torch.optim.SGD(net.parameters(), lr=.1)",
        "loss = None": "loss = F.cross_entropy(logits, y)",
        "pass  # TODO 6": "loss.backward()  # TODO 6",
        "pass  # TODO 7": "optimizer.step()  # TODO 7",
        "prediction = None  # TODO 8": "prediction = logits.argmax(1)  # TODO 8"}
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    reference_out = io.StringIO()
    with contextlib.redirect_stdout(reference_out):
        exec(compile(text, str(exercise), "exec"), {"__name__": "__main__", "__file__": str(exercise)})
    assert exercise.read_bytes() == original
    # 第19页扩展片段按说明单独准备数据后执行。
    variant = re.findall(r"```python\n(.*?)\n```", parts[18], re.S)[0]
    namespace = {"helper": helper, "device": torch.device("cpu")}
    splits, _ = helper["load_data"](include_test=False)
    namespace["train"] = helper["balanced_subset"](splits["train"], 37, 6601)
    namespace["validation"] = helper["balanced_subset"](splits["validation"], 20, 6602)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(variant, "manual-page19", "exec"), namespace)
    return dict(follow_blocks=len(blocks), follow_stdout=out.getvalue(),
                exercise_reference_stdout=reference_out.getvalue(), exercise_source_unchanged=True,
                variant_snippet_updates=namespace["result"]["updates"])


def main():
    errors = numerical_checks()
    docs = document_and_exercise_checks()
    architecture = helper["inspect_net"]()
    assert architecture["parameters"] == 61706
    assert architecture["weight_multiplications_per_image"] == 416520
    counts = {key: helper["inspect_net"](cfg)["parameters"] for key, cfg in helper["candidates"]().items()}
    result = dict(float64_numpy=errors, architecture_parameters=architecture["parameters"],
                  candidate_parameters=counts, documentation=docs)
    (ROOT / "reports/ch06/6_6_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
