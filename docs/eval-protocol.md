# Eval protocol (pre-registered stub)

Owner: [`PAPER.md`](PAPER.md) §6. This file is the freeze. Do not grow the item cap or change the primary metric after a run starts.

**Status:** theoretical season — `eval.jsonl` is **not** written yet. Writing it is part of run authorization, not of this stub.

---

## Freeze

- **File:** `eval.jsonl` (repo root or `evals/eval.jsonl` — pick one when the file is created; hash it).
- **Cap:** \(\le\) **30** items. Not 31. Not a living leaderboard.
- **When:** before the first reported train. Hash of `eval.jsonl` + this file goes in the run note.
- **Contamination:** no item, paraphrase, or near-duplicate in the train stream.

## Strata (multi-strata, CCR-aligned)

Target mix (counts sum to \(\le\) 30):

| Stratum | Role | Target count |
|---|---|---|
| C | Concept: necessary/sufficient + contrasts + one failure mode | 8–10 |
| A | Broad analogy: named \(R\), worked map, near-miss | 6–8 |
| T | Textbook procedure: problem \(\rightarrow\) method \(\rightarrow\) check | 6–8 |
| Chat | Short instruction / chatty use of a locked concept | 4–6 |

No MMLU. No “analogical reasoning is solved” items that only reward memorized SAT pairs. Analogy items must include a near-miss distractor or an explicit wrong \(R\).

Each line, minimum schema:

```json
{
  "id": "C-03",
  "stratum": "C",
  "skill": "short-name",
  "input": "...",
  "target": "...",
  "success_rule": "exact|regex|rubric-1-0",
  "concept_ids": ["..."]
}
```

`success_rule` is decided at freeze. Rubrics are binary. No post-hoc softening.

## Metrics

| Axis | Metric | Role | Against |
|---|---|---|---|
| Quality | `task_success` = items correct / items | **Primary** | same-data **~50M GPT** |
| Quality | held-out PPL on a named stream \(\neq\) eval items | Secondary | same control |
| Memory | process RSS vs sequence length | **Second axis** | same-param GPT (KV) vs H(AI)LP (fixed state) |

Claim P (1–3B phone band) is **not** scored on this file.

## First reported run

The first completed train that produces a full eval pass is **the** reported run.

Disclose, or do not do:

- restarts after peeking at `task_success`
- dropping items
- changing `success_rule`
- unregistered mix or LR changes

Secondary runs are labeled `secondary` and do not replace the first row.

## Empty first-run table

| Run | Date | Eval hash | Tokens | Mix | Model | task_success | PPL | RSS@seq (named lens) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| first | — | — | — | — | 50M GPT (CCR) | — | — | — | no run this season |
| first | — | — | — | — | 50M H(AI)LP (CCR) | — | — | — | candidate, not the thesis |
| first | — | — | — | — | 50M GPT (Wiki/web control) | — | — | — | control |
| first | — | — | — | — | 50M H(AI)LP (Wiki/web) | — | — | — | control |

Kill criteria: [`PAPER.md`](PAPER.md) §7.2.
