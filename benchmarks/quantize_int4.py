"""Offline INT4-ish quantization for H(AI)LP checkpoints.

This script is *storage-focused*: it converts FP32 weights from the
H(AI)LP checkpoint into a compact 4‑bit representation and writes them
to disk so you can point at a **real ~10 MB file**, even before wiring
H(AI)LP into llama.cpp.

It does NOT implement a custom INT4 runtime in PyTorch; llama.cpp (via
GGUF) will eventually provide the production inference engine.

Run:

    # From repo root, after training at least once:
    uv run python benchmarks/quantize_int4.py \\
        --ckpt checkpoints/hailp/best.pt \\
        --out hailp_int4_storage.pt

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Mapping

import numpy as np
import torch


def pack_int4_tight(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a float tensor into 4‑bit values with tight nibble packing.

    Uses per‑tensor affine quantisation:

        q = round(x / scale + zp)  in [0, 15]

    Returns
    -------
    packed : uint8 tensor
        1D vector where each byte stores two 4‑bit values.
    scale : float32 tensor scalar
        Quantisation scale.
    zp : float32 tensor scalar
        Zero‑point (stored as float for simplicity).
    """
    x = tensor.detach().float()
    if x.numel() == 0:
        packed = x.new_empty(0, dtype=torch.uint8)
        scale = x.new_tensor(1.0, dtype=torch.float32)
        zp = x.new_tensor(0.0, dtype=torch.float32)
        return packed, scale, zp

    qmin, qmax = 0.0, 15.0
    x_min = x.min()
    x_max = x.max()
    if x_min == x_max:
        scale = x.new_tensor(1.0, dtype=torch.float32)
        zp = x.new_tensor(0.0, dtype=torch.float32)
        q = torch.zeros_like(x, dtype=torch.int16)
    else:
        scale = (x_max - x_min) / (qmax - qmin)
        # Avoid division by zero
        scale = torch.clamp(scale, min=torch.finfo(torch.float32).eps)
        zp = torch.clamp((qmin - x_min / scale).round(), qmin, qmax)
        q = torch.clamp((x / scale + zp).round(), qmin, qmax).to(torch.int16)

    flat = q.view(-1)
    # Pad to even length so we can pack pairs.
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, flat.new_full((1,), int(zp.item()), dtype=torch.int16)], dim=0)

    low = (flat[0::2] & 0x0F).to(torch.uint8)
    high = ((flat[1::2] & 0x0F) << 4).to(torch.uint8)
    packed = (low | high).contiguous()
    return packed.cpu(), scale.cpu().to(torch.float32), zp.cpu().to(torch.float32)


def unpack_int4_tight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    shape: torch.Size | Tuple[int, ...],
) -> torch.Tensor:
    """Reconstruct an approximate float tensor from packed INT4."""
    if packed.numel() == 0:
        return torch.zeros(shape, dtype=torch.float32)

    bytes_ = packed.view(-1).to(torch.uint8)
    low = (bytes_ & 0x0F).to(torch.int16)
    high = ((bytes_ >> 4) & 0x0F).to(torch.int16)

    flat = torch.empty(bytes_.numel() * 2, dtype=torch.int16)
    flat[0::2] = low
    flat[1::2] = high

    numel = 1
    for d in shape:
        numel *= int(d)
    flat = flat[:numel].to(torch.float32)

    scale_f = scale.to(torch.float32)
    zp_f = zp.to(torch.float32)
    x_hat = (flat - zp_f) * scale_f
    return x_hat.view(*shape)


def _quantize_tensor_int4(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Quantise a tensor using pack_int4_tight; return storage dict."""
    packed, scale, zp = pack_int4_tight(x)
    return {
        "q": packed.cpu(),
        "scale": scale.cpu(),
        "zp": zp.cpu(),
        "shape": torch.tensor(list(x.shape), dtype=torch.int32),
    }


def should_quantize_int4(name: str, tensor: torch.Tensor) -> bool:
    """Decide whether a tensor should be INT4‑quantised.

    Heuristics:
    - Never quantise 1‑D tensors (biases, norms).
    - Never quantise anything with fewer than 32×32 elements.
    - Prefer quantising large 2‑D weight matrices (e.g. linear/attention).
    """
    if tensor.ndim <= 1:
        return False
    if tensor.numel() <= 32 * 32:
        return False
    lower = name.lower()
    if "bias" in lower or "norm" in lower:
        return False
    return True


def pack_model(
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Split a model state_dict into INT4‑quantised and kept‑FP16 buckets.

    Returns
    -------
    quantised :
        Mapping from param name -> quantised storage dict (q, scale, zp, shape).
    kept_fp16 :
        Mapping from param name -> original tensor for parameters that stayed
        in higher precision (biases, norms, small tensors).
    """
    quantised: Dict[str, Dict[str, torch.Tensor]] = {}
    kept_fp16: Dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            kept_fp16[name] = tensor
            continue
        if should_quantize_int4(name, tensor):
            quantised[name] = _quantize_tensor_int4(tensor)
        else:
            kept_fp16[name] = tensor

    return quantised, kept_fp16


def quantize_checkpoint_int4(
    ckpt_path: Path,
    out_path: Path,
    *,
    min_param_elems: int = 16,
) -> None:
    """Quantise FP32 model weights from a HAILP checkpoint to 4‑bit storage."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state: Dict[str, torch.Tensor] = ckpt.get("model", {})

    # Filter tiny tensors up front.
    filtered_state: Dict[str, torch.Tensor] = {}
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.numel() < min_param_elems:
            continue
        filtered_state[name] = tensor

    quantised, kept_fp16 = pack_model(filtered_state)

    payload = {
        "meta": {
            "source_checkpoint": str(ckpt_path),
            "format": "hailp-int4-storage-v1",
        },
        "params_int4": quantised,
        "params_fp16": kept_fp16,
    }
    torch.save(payload, out_path)

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"Wrote INT4 storage file to: {out_path}  (~{size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# NPZ container: save/load helpers for HAILP
# ---------------------------------------------------------------------------


def _encode_name(name: str) -> str:
    """Encode parameter name for use as a safe NPZ key."""
    return name.replace(".", "__DOT__")


def _decode_name(key: str) -> str:
    """Decode NPZ key back into parameter name."""
    return key.replace("__DOT__", ".")


def save_hailp_int4(model: torch.nn.Module, prefix: str | Path) -> Path:
    """Save a HAILP model to a compact INT4+FP16 NPZ container.

    Parameters
    ----------
    model :
        HAILPModel instance with a `state_dict` attribute.
    prefix :
        Path prefix; `.npz` will be appended if not present.
    """
    path = Path(prefix)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    state = model.state_dict()
    quantised, kept_fp16 = pack_model(state)

    arrays: Dict[str, np.ndarray] = {}

    # INT4 tensors
    for name, qdict in quantised.items():
        key = _encode_name(name)
        arrays[f"int4_q/{key}"] = qdict["q"].cpu().numpy()
        arrays[f"int4_scale/{key}"] = qdict["scale"].cpu().numpy()
        arrays[f"int4_zp/{key}"] = qdict["zp"].cpu().numpy()
        arrays[f"int4_shape/{key}"] = qdict["shape"].cpu().numpy()

    # Kept higher‑precision tensors (store as float16 to save space)
    for name, tensor in kept_fp16.items():
        key = _encode_name(name)
        arrays[f"fp16/{key}"] = tensor.detach().cpu().to(torch.float16).numpy()

    # Minimal metadata
    arrays["meta/version"] = np.array(["hailp-int4-npz-v1"], dtype=object)

    np.savez_compressed(path, **arrays)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"[hailp-int4] wrote {path} (~{size_mb:.1f} MB)")
    return path


def load_hailp_int4(prefix: str | Path, model: torch.nn.Module) -> None:
    """Load INT4+FP16 NPZ container back into a HAILP model."""
    path = Path(prefix)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    with np.load(path, allow_pickle=False) as data:
        state: Dict[str, torch.Tensor] = {}

        # First, restore INT4 tensors
        for key in data.files:
            if not key.startswith("int4_q/"):
                continue
            encoded = key[len("int4_q/") :]
            name = _decode_name(encoded)
            q = torch.from_numpy(data[key]).to(torch.uint8)
            scale = torch.from_numpy(data[f"int4_scale/{encoded}"]).view(())
            zp = torch.from_numpy(data[f"int4_zp/{encoded}"]).view(())
            shape_arr = data[f"int4_shape/{encoded}"]
            shape = tuple(int(x) for x in shape_arr.tolist())
            tensor = unpack_int4_tight(q, scale, zp, shape)
            state[name] = tensor

        # Then, restore FP16 tensors
        for key in data.files:
            if not key.startswith("fp16/"):
                continue
            encoded = key[len("fp16/") :]
            name = _decode_name(encoded)
            arr = data[key]
            tensor = torch.from_numpy(arr).to(torch.float32)
            state[name] = tensor

    model.load_state_dict(state, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantise HAILP checkpoint weights to a compact INT4 storage file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/hailp/best.pt"),
        help="Path to HAILP checkpoint (expects 'model' state_dict).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("hailp_int4_storage.pt"),
        help="Output path for the quantised INT4 storage file.",
    )
    args = parser.parse_args()

    quantize_checkpoint_int4(args.ckpt, args.out)


if __name__ == "__main__":
    main()

