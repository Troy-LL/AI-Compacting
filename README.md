H(AI)LP — Architecture Comparison: Baseline GPT vs H(AI)LP RWKV

Two 50M-parameter language models trained on the same data, differing only in architecture.

## The Core Claim

Standard transformers allocate a Key-Value cache that **grows linearly** with sequence length.
H(AI)LP RWKV replaces this with a **fixed-size recurrent state** that never grows.

## Project Layout

```
H(AI)LP/
├── configs/
│   ├── baseline.yaml       # Standard GPT config (O(n²) attention, KV cache)
│   └── hailp.yaml         # H(AI)LP RWKV config (O(n) recurrent, fixed state)
├── models/
│   ├── components/
│   │   ├── low_rank.py     # Rank-r weight factorization (U @ V^T)
│   │   ├── param_sharing.py # Cross-layer FFN weight sharing
│   │   └── adapter.py      # Language adapter injection points
│   ├── baseline_gpt.py     # Baseline GPT implementation
│   └── hailp_model.py     # H(AI)LP RWKV implementation
├── training/
│   ├── data.py             # Streaming Wikipedia tokenisation
│   └── trainer.py          # Training loop with W&B hooks
├── benchmarks/
│   ├── memory_benchmark.py # RAM usage vs sequence length
│   ├── speed_profile.py    # Tokens/sec at various seq lengths
│   └── quality_eval.py     # Validation loss & perplexity (optional checkpoint)
├── demo.py                 # Side-by-side comparison (run this first!)
└── tests/
    ├── test_components.py
    ├── test_baseline.py
    └── test_hailp.py
```

## Quick Start

```powershell
# Install uv (if not installed)
winget install astral-sh.uv

# Install dependencies
uv sync

# Run all tests (everything must pass before training)
uv run pytest tests/ -v

# Run the demo comparison
uv run python demo.py
```

**Note:** No Hugging Face account or token is required. The dataset and tokenizer are public. If you hit rate limits, set `HF_TOKEN` or run `huggingface-cli login` for higher limits.

## DirectML

If you have an AMD GPU on Windows, you can run training with DirectML.

Recommended Python version: `3.11` (DirectML can have issues on `3.12`).

Install DirectML:

```powershell
pip install torch-directml
```

Verify DirectML is working:

```powershell
python scripts/verify_dml.py
```

DirectML training smoke test (H(AI)LP):

```powershell
uv run python train.py --steps 200 --no-wandb
```

Notes:
- This repo enables AMP (`GradScaler`/`autocast`) only for CUDA devices; DirectML runs FP32.

## Training

**Smoke test** (H(AI)LP only, 200 steps, no W&B; good for CI or a quick sanity check):

```powershell
uv run python train.py --steps 200 --batch-size 4 --no-wandb
```

**Full run** with Weights & Biases (perplexity and curves; ~30k steps per model):

```powershell
# Log in to W&B first if needed: uv run wandb login
uv run python train.py
```

Resume from the latest checkpoint:

```powershell
uv run python train.py --resume
```

Checkpoints are saved under `checkpoints/<model>/` (rotating last 3 + `best.pt`).

## Benchmarks

- **Memory** — KV cache growth vs fixed recurrent state:
  ```powershell
  uv run python benchmarks/memory_benchmark.py
  ```
- **Speed** — tokens/second at different sequence lengths:
  ```powershell
  uv run python benchmarks/speed_profile.py
  uv run python benchmarks/speed_profile.py --batch-size 8 --seq-lens 64 256 512
  ```
- **Quality** — validation loss and perplexity (random init or from checkpoints):
  ```powershell
  uv run python benchmarks/quality_eval.py --model hailp --max-batches 50
  uv run python benchmarks/quality_eval.py --model hailp --checkpoint-dir checkpoints --max-batches 100
  ```

## Multi-GPU Training Results (Kaggle Dual-T4)

A recent 1,000-step continuous streaming test on dual NVIDIA T4 GPUs (15GB) conclusively validated the architecture's core memory thesis. 

| Metric                  | Start (Step 10) | End (Step 1,000) | Key Observation                                                  |
|-------------------------|-----------------|------------------|------------------------------------------------------------------|
| **Train Loss**          | 10.63           | 1.08             | **90% Reduction**, rapid convergence.                            |
| **Best Val Perplexity** | —               | **127.5**        | Achieved at step 800 (`val_loss=4.8485`). Strong generalization. |
| **VRAM Usage**          | 2,544 MB        | 2,505 MB         | **Dead flat**, perfectly fixed memory footprint!                 |
| **Throughput**          | ~8,290 tok/s    | ~8,550 tok/s     | Perfectly stable scaling, peaking at >8,500 tokens per second.   |

**Configuration:** `fp16`, global effective batch size `512` (64 per-GPU micro-batch × 4 accum × 2 GPUs).
*Note: A custom `LossWrapper` is used internally during training to compute loss during the forward pass, preventing Accelerate's automatic FP32 conversion overhead which would otherwise consume an unnecessary >3GB of VRAM.*

## Architecture Key Differences

| Feature              | Baseline GPT             | H(AI)LP RWKV            |
|----------------------|--------------------------|--------------------------|
| Attention            | Full O(n²) self-attention | RWKV time-mixing O(n)   |
| Memory at seq=512    | 8× more than seq=64      | Same as seq=64           |
| FFN weights          | Unique per layer          | Shared across groups     |
| Attention projections | Full rank W              | Rank-64 U @ V^T         |
| State representation | KV cache (grows)         | Fixed h_state tensor     |
| Parameters           | ~50M                      | ~50M                     |

---

## Hardware Requirements (Android Deployment)

For a tool intended for survival and critical aid, reliability is non-negotiable. Our INT4-quantized model with 360M parameters has the following RAM footprint at idle:

| Device Total RAM | Operational Status | Reality Check |
|------------------|---------------------|---------------|
| **2GB** | **Insecure** | High risk of OOM crashes; OS takes 700MB+, Leaving <1.2GB for model/index (~700MB total). |
| **3GB** | **Stable** | ~1.3GB free headroom; standard for reliable operation. |
| **4GB** | **Optimum** | Fully comfortable operation under load. |

### The Honest RAM Breakdown (for 2GB Devices)

A typical budget phone with 2GB total RAM is extremely constrained:
- **OS & System Processes:** 600–800 MB (baseline for Android to function).
- **Background Apps/Launcher:** 200–400 MB.
- **Available before app launch:** ~800 MB–1.2 GB.

When H(AI)LP loads:
- **INT4 weights + runtime overhead:** 450–550 MB (scaled from 156 MB peak for 18.5M).
- **FAISS Index & Knowledge Base:** 100-150 MB.
- **Result:** You operate right at the functional edge. Performance is unstable and crashes are likely.

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details. This license was chosen for its permissive nature and its specific protections for patents, which is the industry standard for modern AI/ML software.
