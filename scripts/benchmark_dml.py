from __future__ import annotations

import argparse
import time

import torch


def benchmark(device: torch.device, device_name: str, size: int, warmup: int, iterations: int) -> None:
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    with torch.no_grad():
        # Warmup — helps stabilize first-run overhead.
        for _ in range(warmup):
            c = a @ b
            _ = c[0][0].item()

        start = time.perf_counter()
        for _ in range(iterations):
            c = a @ b
            # Force completion / synchronization.
            _ = c[0][0].item()
        elapsed = time.perf_counter() - start

    ms_per_op = (elapsed / max(iterations, 1)) * 1000.0
    ops_per_sec = iterations / max(elapsed, 1e-9)
    print(f"{device_name}: size={size} iters={iterations}  {ops_per_sec:.1f} ops/sec  ({ms_per_op:.2f} ms/op)")


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU vs DirectML matmul benchmark")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    cpu = torch.device("cpu")
    benchmark(cpu, "CPU", args.size, args.warmup, args.iterations)

    try:
        import torch_directml  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"torch-directml not available: {e}")
        return 1

    dml = torch_directml.device()
    benchmark(dml, "DirectML", args.size, args.warmup, args.iterations)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

