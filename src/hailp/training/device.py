from __future__ import annotations

import multiprocessing

import torch


def _try_directml() -> tuple[torch.device, str] | None:
    """Attempt to create a DirectML device and run a tiny validation."""
    try:
        import torch_directml  # type: ignore
    except Exception:
        return None

    try:
        dml = torch_directml.device()
        # Quick validation — does it actually execute?
        x = torch.ones(2, 2).to(dml)
        y = torch.ones(2, 2).to(dml)
        z = x + y
        v = z[0][0].item()
        if abs(v - 2.0) > 1e-4:
            raise RuntimeError("DirectML math check failed")
        return dml, "DirectML"
    except Exception:
        # If DirectML exists but doesn't work, fall back to CPU.
        return None


def _select_device() -> tuple[torch.device, str]:
    """Return (device, device_name) for CUDA / DirectML / CPU."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        dev = torch.device("cuda")
        return dev, f"CUDA ({props.name})"

    directml = _try_directml()
    if directml is not None:
        dev, _name = directml
        # torch-directml may not expose a friendly GPU name consistently,
        # so we keep the device name generic but explicit.
        return dev, "DirectML"

    dev = torch.device("cpu")
    return dev, f"CPU ({multiprocessing.cpu_count()} cores)"


DEVICE, DEVICE_NAME = _select_device()

# AMP is CUDA-only in this codebase (autocast/scaler come from torch.cuda.amp).
USE_AMP: bool = DEVICE.type == "cuda"

# This repo doesn't use torch.compile today, but keep this flag for parity.
USE_COMPILE: bool = False

print(f"H(AI)LP running on: {DEVICE_NAME}")

