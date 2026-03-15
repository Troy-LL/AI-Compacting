H(AI)LP — Architecture Comparison: Baseline GPT vs LifeLink RWKV

Two 50M-parameter language models trained on the same data, differing only in architecture.

## The Core Claim

Standard transformers allocate a Key-Value cache that **grows linearly** with sequence length.
LifeLink RWKV replaces this with a **fixed-size recurrent state** that never grows.

## Project Layout

```
H(AI)LP/
├── configs/
│   ├── baseline.yaml       # Standard GPT config (O(n²) attention, KV cache)
│   └── lifelink.yaml       # LifeLink RWKV config (O(n) recurrent, fixed state)
├── models/
│   ├── components/
│   │   ├── low_rank.py     # Rank-r weight factorization (U @ V^T)
│   │   ├── param_sharing.py # Cross-layer FFN weight sharing
│   │   └── adapter.py      # Language adapter injection points
│   ├── baseline_gpt.py     # Baseline GPT implementation
│   └── lifelink_rwkv.py    # LifeLink RWKV implementation
├── training/
│   ├── tokenizer.py        # Simple tokenizer (vocab=8000)
│   ├── data.py             # TextDataset for training
│   └── trainer.py          # Training loop with W&B hooks
├── benchmarks/
│   ├── memory_profile.py   # RAM usage vs sequence length
│   ├── speed_profile.py    # Tokens/second benchmark
│   └── quality_eval.py     # Perplexity / accuracy eval
├── demo/
│   └── compare.py          # Side-by-side compare (run this first!)
└── tests/
    ├── test_components.py
    ├── test_baseline.py
    └── test_lifelink.py
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
uv run python demo/compare.py
```

## Architecture Key Differences

| Feature              | Baseline GPT             | LifeLink RWKV            |
|----------------------|--------------------------|--------------------------|
| Attention            | Full O(n²) self-attention | RWKV time-mixing O(n)   |
| Memory at seq=512    | 8× more than seq=64      | Same as seq=64           |
| FFN weights          | Unique per layer          | Shared across groups     |
| Attention projections | Full rank W              | Rank-64 U @ V^T         |
| State representation | KV cache (grows)         | Fixed h_state tensor     |
| Parameters           | ~50M                      | ~50M                     |

