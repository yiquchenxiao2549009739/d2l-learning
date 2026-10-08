"""Small deterministic linear-regression training to verify the installation.

The user will implement and understand the training loop when reaching chapter 3.
"""

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run with --device cpu or fix the GPU setup.")

    torch.set_num_threads(1)
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    x_cpu = torch.randn(256, 2, generator=generator)
    true_weight_cpu = torch.tensor([[2.0], [-3.4]])
    true_bias = 4.2
    noise = 0.01 * torch.randn(256, 1, generator=generator)
    y_cpu = x_cpu @ true_weight_cpu + true_bias + noise
    x, y = x_cpu.to(device), y_cpu.to(device)
    model = torch.nn.Linear(2, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_function = torch.nn.MSELoss()
    initial_loss = loss_function(model(x), y).item()
    history = []
    for epoch in range(1, 301):
        prediction = model(x)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "mse_before_step": loss.item()})

    with torch.no_grad():
        final_loss = loss_function(model(x), y).item()
        learned_weight = model.weight.detach().cpu().reshape(-1)
        learned_bias = model.bias.detach().cpu().item()
    max_weight_error = (learned_weight - true_weight_cpu.reshape(-1)).abs().max().item()
    passed = final_loss < 0.001 and max_weight_error < 0.1 and abs(learned_bias - true_bias) < 0.1
    metrics = {
        "purpose": "environment validation, not learner completion",
        "seed": 42,
        "device": device,
        "torch": torch.__version__,
        "samples": 256,
        "epochs": 300,
        "learning_rate": 0.05,
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "true_weight": [2.0, -3.4],
        "learned_weight": learned_weight.tolist(),
        "true_bias": true_bias,
        "learned_bias": learned_bias,
        "passed": passed,
    }
    output_dir = ROOT / "artifacts" / "smoke" / device
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "mse_before_step"])
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy([row["epoch"] for row in history], [row["mse_before_step"] for row in history])
    ax.set(xlabel="Epoch", ylabel="Mean squared error", title=f"Linear regression validation ({device})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=150)
    plt.close(fig)
    torch.save(
        {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
         "input_features": 2, "output_features": 1},
        output_dir / "model.pt",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("Training validation failed: parameters or loss did not converge.")
    print(f"Training validation passed. Results: {output_dir}")


if __name__ == "__main__":
    main()
