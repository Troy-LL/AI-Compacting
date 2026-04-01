from __future__ import annotations

import torch


def main() -> int:
    try:
        import torch_directml  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"torch-directml not available: {e}")
        return 1

    dml = torch_directml.device()
    print(f"DirectML device: {dml}")

    with torch.no_grad():
        x = torch.ones(3, 3).to(dml)
        y = torch.ones(3, 3).to(dml)
        z = x + y
        v = z[0][0].item()
        print(f"Tensor add result: {v}  (expected 2.0)")
        if abs(v - 2.0) > 1e-5:
            return 2

        a = torch.randn(512, 512).to(dml)
        b = torch.randn(512, 512).to(dml)
        c = a @ b
        # Force completion; reading .item() syncs.
        s = c[0][0].item()
        print(f"Matrix multiply: OK  (sample={s:.6f})")

    print("DirectML working correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

