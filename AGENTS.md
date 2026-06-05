# AGENTS.md

Guidance for AI agents working in this repository.

## Cursor Cloud specific instructions

### Project type

H(AI)LP is a **Python ML research/CLI project** (not a web app). There are no HTTP dev servers or Docker services. All workflows run as one-shot CLI commands via `uv run`.

### Package manager

This repo uses **[uv](https://docs.astral.sh/uv/)**. Ensure `uv` is on `PATH` (`~/.local/bin` after the standard install). The VM update script runs `uv sync` from the repo root.

### Common commands

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Tests (62 tests) | `PYTHONPATH=.:src uv run pytest tests/ -v` |
| Architecture demo | `uv run python scripts/demo.py` |
| Lint | `uv run ruff check src tests scripts` |
| Type check | `uv run mypy src/hailp` |
| Training smoke test | `uv run python scripts/train.py --steps 200 --batch-size 4 --no-wandb` |

### pytest PYTHONPATH gotcha

`pyproject.toml` sets `pythonpath = ["src"]`, but `tests/inference/test_pipeline.py` imports sibling test modules as `from tests.inference...`. Until that is fixed upstream, **prefix tests with** `PYTHONPATH=.:src` so collection succeeds.

### Lint / type-check notes

- `ruff check` reports many pre-existing style issues (line length, unused imports, etc.) across `scripts/` and `src/`.
- `mypy` runs in strict mode and reports pre-existing type errors in benchmarks and model code.

### External services (optional)

- **Hugging Face Hub** — required only for training (`scripts/train.py`) and quality benchmarks; not needed for tests or `scripts/demo.py`.
- **Weights & Biases** — optional for training; pass `--no-wandb` to skip.
- **CUDA GPU** — optional; tests and demo run on CPU only.

### Source layout

Application code lives under `src/hailp/` (models, training, inference, benchmarks). Entry-point scripts are in `scripts/`. YAML configs are in `src/hailp/configs/`.
