# AI-Compacting

**Compact competence under a phone budget**

Theoretical research on (1) literature bounds for phone-resident chatty LLMs and (2) fewest parameters for master-of-none multi-skill competence under a fixed process-RSS budget. The original hypothesis is an operational compact curriculum (CCR).

Author: Troy.

## What this is / what this is not

This is a paper-first repo: two claims kept separate, a literature envelope, and a protocol. It is not a product, not a training run, and not a claimed result. Numbers that are not in [`docs/PAPER.md`](docs/PAPER.md) are not claims. Standing rule: **no claim without a run.**

## Two claims, kept separate

**Claim P (practical).** Literature bounds for a phone-resident chatty generalist: the dense **1–3B INT4** class (~**0.5–1.5 GB weights**). **\(\le\)7B INT4** is a published COTS ceiling, not the target. Phone-fit is process RSS plus sustained thermal/energy under a named \(T\), not file size. Sources and caveats live in [`docs/PAPER.md`](docs/PAPER.md) §2 and [`docs/LITERATURE.md`](docs/LITERATURE.md).

**Claim R (radical).** Under a **fixed process-RSS budget**, how few parameters still buy master-of-none multi-skill competence? A same-data **~50M GPT** is the control. H(AI)LP’s fixed recurrent state is a **candidate** architecture for the RAM-vs-sequence axis, not the thesis.

Do not stack P and R as one result. Thesis **A (data)** is primary; thesis **B (architecture)** is related.

## Own innovation — Compact Curriculum Recipe (CCR)

CCR is a **hypothesis and a protocol**: concepts, then broad analogies, then dense textbooks, with explicit in/out rules. It is not a result.

- Protocol: [`docs/ccr-spec.md`](docs/ccr-spec.md)
- Example shapes (not a corpus): [`docs/ccr-samples.md`](docs/ccr-samples.md)
- Eval freeze: [`docs/eval-protocol.md`](docs/eval-protocol.md)

Analogical reasoning is not treated as solved. MMLU is not a primary under 300M. MoE and sub-INT4 are theater until measured.

## Doc map

| File | Job |
|------|-----|
| [`docs/PAPER.md`](docs/PAPER.md) | Thesis owner |
| [`docs/LITERATURE.md`](docs/LITERATURE.md) | Must-cites |
| [`docs/eval-protocol.md`](docs/eval-protocol.md) | Eval freeze + empty first-run table |
| [`docs/ccr-spec.md`](docs/ccr-spec.md) | CCR operational protocol |
| [`docs/ccr-samples.md`](docs/ccr-samples.md) | CCR example shapes |
| [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) | Older scaffold notes (not paper results) |
| [`docs/Kaggle.md`](docs/Kaggle.md) | Older Kaggle notes (not paper results) |
| [`docs/GITHUB-ABOUT.txt`](docs/GITHUB-ABOUT.txt) | GitHub About text to paste |

## Later / optional code

The tree still has a ~50M GPT vs H(AI)LP scaffold, tests, and trainers. They are instruments for a later run, not this season’s output.

```text
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

Training, DirectML, and benchmark commands remain in the scripts; do not run them as paper work this season.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
