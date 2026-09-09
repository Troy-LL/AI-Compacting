# AI-Compacting

**This season the repo is a theoretical paper.** No new training runs. No invented numbers.

Thesis owner: [`docs/PAPER.md`](docs/PAPER.md)  
Must-cites: [`docs/LITERATURE.md`](docs/LITERATURE.md)  
Eval freeze: [`docs/eval-protocol.md`](docs/eval-protocol.md)

Author: Troy.

---

## Two claims, kept separate

**Claim P (practical).** A phone-resident chatty generalist is the dense **1–3B INT4** class (~**0.5–1.5 GB weights**). **\(\le\)7B INT4** is a published COTS ceiling, not the target. Cites: Phi-3, MobileLLM, BitNet (compression bound only), MELT and related phone studies. Phone-fit means **process RSS** plus **sustained thermal/energy under a named \(T\)**, not file size. NPU is unnamed until measured.

**Claim R (radical).** Under a **fixed process-RSS budget**, how few parameters still buy master-of-none multi-skill competence? A same-data **~50M GPT** is the control. H(AI)LP’s fixed recurrent state is a **candidate** architecture for the RAM-vs-sequence axis, not the thesis.

Do not stack P and R as one win. Thesis **A (data)** is primary; thesis **B (architecture)** is related.

## Own innovation

An operational **compact curriculum recipe** (concepts + broad analogies + dense textbooks) with explicit in/out rules. It is a **hypothesis and a protocol**. We do not claim it wins without a train. Analogical reasoning is not treated as solved. MMLU is not a primary under 300M. MoE and sub-INT4 are theater until measured.

Standing rule, everywhere: **we do not claim X without a run.**

---

## Later / optional code

The tree still has a ~50M GPT vs H(AI)LP scaffold, tests, and trainers. They are instruments for a later run, not this season’s output. Older engineering notes (including Kaggle smoke numbers and Android footprint projections) live in [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) and [`docs/Kaggle.md`](docs/Kaggle.md). **They are not paper results.**

```text
docs/PAPER.md              thesis
docs/LITERATURE.md         must-cites
docs/eval-protocol.md      freeze + first-run table (empty)
src/hailp/                 optional ~50M GPT / H(AI)LP scaffold
scripts/train.py           later
scripts/train_multi.py     later
scripts/demo.py            later
tests/                     scaffold tests (optional to run)
```

If you still want the scaffold:

```bash
uv sync
uv run pytest tests/ -v
uv run python scripts/demo.py
```

Training, DirectML, and benchmark commands are unchanged in the scripts; do not run them as paper work this season. Checkpoints and W&B are optional and off-protocol until `eval.jsonl` is frozen (see the eval stub).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
