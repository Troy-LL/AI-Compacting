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

## Training

**Smoke test** (both models, 200 steps, no W&B; good for CI or a quick sanity check):

```powershell
uv run python train.py --model both --steps 200 --batch-size 4 --no-wandb
```

**Full run** with Weights & Biases (perplexity and curves; ~30k steps per model):

```powershell
# Log in to W&B first if needed: uv run wandb login
uv run python train.py --model both
# Or one model: uv run python train.py --model baseline
uv run python train.py --model hailp
```

Resume from the latest checkpoint:

```powershell
uv run python train.py --model both --resume
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
  uv run python benchmarks/quality_eval.py --model both --max-batches 50
  uv run python benchmarks/quality_eval.py --model both --checkpoint-dir checkpoints --max-batches 100
  ```

## Architecture Key Differences

| Feature              | Baseline GPT             | H(AI)LP RWKV            |
|----------------------|--------------------------|--------------------------|
| Attention            | Full O(n²) self-attention | RWKV time-mixing O(n)   |
| Memory at seq=512    | 8× more than seq=64      | Same as seq=64           |
| FFN weights          | Unique per layer          | Shared across groups     |
| Attention projections | Full rank W              | Rank-64 U @ V^T         |
| State representation | KV cache (grows)         | Fixed h_state tensor     |
| Parameters           | ~50M                      | ~50M                     |
