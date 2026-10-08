"""Check the local environment; this is tooling, not a completed lesson."""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    info = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    devices = ["cpu"]
    if info["cuda_available"]:
        devices.append("cuda")
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
    for device in devices:
        x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        y = (x * x).sum()
        y.backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())
        product = torch.ones((2, 3), device=device) @ torch.ones((3, 4), device=device)
        torch.testing.assert_close(product, torch.full((2, 4), 3.0, device=device))
        if device == "cuda":
            torch.cuda.synchronize()
        info[f"{device}_tensor_and_gradient_check"] = "passed"
    output = ROOT / "artifacts" / "environment.local.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("Environment check passed.")


if __name__ == "__main__":
    main()
