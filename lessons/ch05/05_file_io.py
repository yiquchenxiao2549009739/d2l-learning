"""5.5 读写文件。保存本地张量、模型状态和可恢复的训练检查点。"""
import copy
import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

ARTIFACTS = ROOT / "artifacts/ch05/5_5"
REPORTS = ROOT / "reports/ch05"
DTYPE = torch.float64
ARCHITECTURE = {
    "kind": "two_hidden_mlp",
    "in_features": 2,
    "hidden_features": [8, 8],
    "out_features": 1,
}
SAVE_STEP = 60
FINAL_STEP = 120


class Regressor(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        first, second = hidden_features
        self.features = nn.Sequential(
            nn.Linear(in_features, first), nn.ReLU(),
            nn.Linear(first, second), nn.ReLU(),
        )
        self.head = nn.Linear(second, out_features)

    def forward(self, X):
        return self.head(self.features(X))


def build_model(config):
    if config["kind"] != "two_hidden_mlp":
        raise ValueError(f"未知的模型结构 {config['kind']!r}")
    return Regressor(
        config["in_features"], config["hidden_features"], config["out_features"],
    ).double()


def save_local(value, path):
    # 先写完整的临时文件，再用同目录替换更新正式文件。
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_local(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_demo():
    x = torch.arange(4)
    y = torch.zeros(4, dtype=DTYPE)
    values = {"tensor.pt": x, "list.pt": [x, y], "dictionary.pt": {"x": x, "y": y}}
    for name, value in values.items():
        save_local(value, ARTIFACTS / name)
    single = load_local(ARTIFACTS / "tensor.pt")
    sequence = load_local(ARTIFACTS / "list.pt")
    mapping = load_local(ARTIFACTS / "dictionary.pt")
    return {
        "tensor_equal": torch.equal(single, x),
        "tensor_dtype": str(single.dtype),
        "list_equal": torch.equal(sequence[0], x) and torch.equal(sequence[1], y),
        "dictionary_equal": torch.equal(mapping["x"], x) and torch.equal(mapping["y"], y),
        "dictionary_keys": list(mapping),
        "floating_tensor_dtype": str(mapping["y"].dtype),
    }


def make_data():
    rng = torch.Generator().manual_seed(5510)
    X = 2 * torch.rand(96, 2, generator=rng, dtype=DTYPE) - 1
    a, b = X[:, :1], X[:, 1:]
    y = .6 * a - .8 * b + .3 * a * b
    return X[:64], y[:64], X[64:], y[64:]


def train_step(model, optimizer, X, y):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(X) - y).square().mean()
    loss.backward()
    optimizer.step()


def mse(model, X, y):
    with torch.no_grad():
        return (model(X) - y).square().mean().item()


def compare_parameters(first, second):
    return max((a - b).abs().max().item()
               for a, b in zip(first.parameters(), second.parameters()))


def training_and_restore():
    X, y, VX, vy = make_data()
    model = build_model(ARCHITECTURE)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9)
    history = [{"step": 0, "continuous_mse": mse(model, X, y), "resumed_mse": None}]
    for step in range(1, FINAL_STEP + 1):
        train_step(model, optimizer, X, y)
        history.append({"step": step, "continuous_mse": mse(model, X, y), "resumed_mse": None})
        if step == SAVE_STEP:
            save_local({
                "format_version": 1,
                "architecture": ARCHITECTURE,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
            }, ARTIFACTS / "checkpoint_step60.pt")

    save_local(model.state_dict(), ARTIFACTS / "weights.pt")
    weights = load_local(ARTIFACTS / "weights.pt")
    restored = build_model(ARCHITECTURE)
    restored.load_state_dict(weights)
    mode_after_loading = restored.training
    model.eval()
    restored.eval()
    with torch.no_grad():
        output = model(VX)
        restored_output = restored(VX)

    checkpoint = load_local(ARTIFACTS / "checkpoint_step60.pt")
    if checkpoint["format_version"] != 1:
        raise ValueError("检查点格式版本不兼容。")
    resumed = build_model(checkpoint["architecture"])
    resumed.load_state_dict(checkpoint["model_state"])
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=.05, momentum=.9)
    resumed_optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state"]))
    resumed.train()
    history[checkpoint["step"]]["resumed_mse"] = mse(resumed, X, y)
    for step in range(checkpoint["step"] + 1, FINAL_STEP + 1):
        train_step(resumed, resumed_optimizer, X, y)
        history[step]["resumed_mse"] = mse(resumed, X, y)

    # 同一个中间起点，比较有无历史动量的第一步。
    cold = build_model(checkpoint["architecture"])
    cold.load_state_dict(checkpoint["model_state"])
    cold_optimizer = torch.optim.SGD(cold.parameters(), lr=.05, momentum=.9)
    warm = build_model(checkpoint["architecture"])
    warm.load_state_dict(checkpoint["model_state"])
    warm_optimizer = torch.optim.SGD(warm.parameters(), lr=.05, momentum=.9)
    warm_optimizer.load_state_dict(checkpoint["optimizer_state"])
    train_step(cold, cold_optimizer, X, y)
    train_step(warm, warm_optimizer, X, y)

    wider_config = dict(ARCHITECTURE, out_features=2)
    wider = build_model(wider_config)
    original_head = wider.head.weight.detach().clone()
    wider.features.load_state_dict(restored.features.state_dict())
    with torch.no_grad():
        feature_error = (wider.features(VX) - restored.features(VX)).abs().max().item()
        new_shape = list(wider(VX).shape)

    # 对错误加载使用一次性模型，便于观察形状冲突。
    incompatible = build_model(wider_config)
    try:
        incompatible.load_state_dict(weights, strict=False)
    except RuntimeError as error:
        size_mismatch_detected = "size mismatch" in str(error)
    else:
        size_mismatch_detected = False

    momentum_error = max(
        (optimizer.state[a]["momentum_buffer"] -
         resumed_optimizer.state[b]["momentum_buffer"]).abs().max().item()
        for a, b in zip(model.parameters(), resumed.parameters())
    )
    resumed.eval()
    with torch.no_grad():
        resumed_output_error = (output - resumed(VX)).abs().max().item()

    results = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture": ARCHITECTURE,
        "weight_keys": list(weights),
        "initial_train_mse": history[0]["continuous_mse"],
        "checkpoint_train_mse": history[SAVE_STEP]["continuous_mse"],
        "final_train_mse": history[-1]["continuous_mse"],
        "final_validation_mse": (output - vy).square().mean().item(),
        "weights_restore_parameter_error": compare_parameters(model, restored),
        "weights_restore_prediction_error": (output - restored_output).abs().max().item(),
        "restored_training_mode_before_eval": mode_after_loading,
        "checkpoint_step": checkpoint["step"],
        "resume_loss_error": max(abs(row["continuous_mse"] - row["resumed_mse"])
                                 for row in history[SAVE_STEP:]),
        "resume_parameter_error": compare_parameters(model, resumed),
        "resume_momentum_error": momentum_error,
        "resume_prediction_error": resumed_output_error,
        "cold_optimizer_one_step_parameter_difference": compare_parameters(cold, warm),
        "warm_first_step_loss_error": abs(mse(warm, X, y) - history[SAVE_STEP + 1]["continuous_mse"]),
        "transfer_feature_error": feature_error,
        "transfer_new_head_weight_preserved": torch.equal(original_head, wider.head.weight),
        "transfer_output_shape": new_shape,
        "strict_false_detected_size_mismatch": size_mismatch_detected,
    }
    return results, history


def draw_training(history, path):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    resumed = history[SAVE_STEP:]
    ax.semilogy([r["step"] for r in history], [r["continuous_mse"] for r in history],
                color="#147e83", linewidth=2, label="连续训练")
    ax.semilogy([r["step"] for r in resumed], [r["resumed_mse"] for r in resumed],
                "--", color="#b76b1b", linewidth=1.6, label="从检查点恢复")
    ax.axvline(SAVE_STEP, color="#64748b", linestyle=":", label="第60步保存")
    ax.set(xlabel="SGD更新次数", ylabel="训练MSE（对数坐标）", title="恢复后与连续训练的损失重合")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(5501)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    figure_path = REPORTS / "figures/5_5_resume.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = tensor_demo()
    model, history = training_and_restore()
    metrics = {
        "section": "5.5", "date": "2026-11-03", "torch": torch.__version__,
        "device": "cpu", "dtype": str(DTYPE), "tensors": tensors, "model": model,
        "local_files": {
            name: (ARTIFACTS / name).stat().st_size
            for name in ("tensor.pt", "list.pt", "dictionary.pt", "weights.pt", "checkpoint_step60.pt")
        },
    }
    with (REPORTS / "5_5_epochs.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (REPORTS / "5_5_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    draw_training(history, figure_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
