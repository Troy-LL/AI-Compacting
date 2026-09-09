# Compact Curriculum Recipe — operational spec

**Author:** Troy  
**Season:** theoretical paper — no new training runs  
**Owner of the thesis:** [`PAPER.md`](PAPER.md) §4. This file is the **operational protocol** for writing, checking, sequencing, and disposing CCR cards. Example shapes (not a corpus): [`ccr-samples.md`](ccr-samples.md). Must-cites: [`LITERATURE.md`](LITERATURE.md). Eval freeze: [`eval-protocol.md`](eval-protocol.md).

**Standing rule.** CCR is a **hypothesis plus a protocol**. We do not claim it beats Wikipedia, FineWeb, FineWeb-Edu, TinyStories, or Phi’s pipeline without a run. This file does not contain a result.

**What an agent with only this file must do:** accept, rewrite, park, or kill a card or a mix plan against the schema, the checklist, and the catalog. If a step is not in this file, it is not CCR.

---

## F. Linkage (read first)

| File | Job |
|---|---|
| [`PAPER.md`](PAPER.md) | Thesis. Two claims, kept separate. CCR hypothesis, strata in/out, mixture, hygiene, kill criteria. |
| [`LITERATURE.md`](LITERATURE.md) | Must-cites. Chinchilla token-budget row is **TODO-cite**. FineWeb-Edu is a control candidate, not CCR. Lee et al. (2022) is a related masking prior, not this recipe. |
| [`eval-protocol.md`](eval-protocol.md) | Frozen ≤30-item Claim **R** eval. Primary `task_success`. No MMLU. Claim **P** is not scored here. |
| [`ccr-samples.md`](ccr-samples.md) | Example shapes. Not `eval.jsonl`. Not a training corpus. |

CCR does not convert Claim R’s ~50M instrument into Claim P’s 1–3B phone band. H(AI)LP is a candidate architecture, not the curriculum.

---

## A. Card schema

Every training unit is **one JSON object** with exactly one primary `stratum` (`C` | `A` | `T`). YAML is an authoring form; interchange and checks are JSON. Unknown extra keys are allowed only for the extensions named in this file (`lang`, `disposition`, `reject_code`, `reject_reason`). Invented result fields (`mmlu`, `task_success`, `ppl`, `phone_rss`) are **illegal** on a card.

`text` is the training surface for the first-run hypothesis. Structured fields are mandatory metadata. `text` must **realize** the lock (it may not contradict it, and it may not omit a required lock element). Do not claim a serialization wins.

### A.1 Common (all strata)

| Field | Type | Required | Constraint |
|---|---|---|---|
| `id` | string | yes | Unique. Pattern `^(C\|A\|T)-[a-z0-9]+(?:-[a-z0-9]+)*-[0-9]{3}$`. Rejected attempts use the same pattern inside a `rejected/` record; they are not train ids. |
| `stratum` | `"C"` \| `"A"` \| `"T"` | yes | Exactly one. A unit may *point at* another stratum; it may not mash up to evade in/out. |
| `license` | string | yes | Non-empty. Prefer SPDX (`CC0-1.0`, `CC-BY-4.0`, `Apache-2.0`, `PD`). `unknown` is a hygiene fail. |
| `source_id` | string | yes | Non-empty locator: URI, `{corpus}:{path}`, or `synthetic:{doc}#{id}`. |
| `concept_ids` | string[] | yes | Min 1. Every entry is a C `id` (this card, if stratum C). |
| `version` | integer | yes | ≥ 1. Bump on lock change; do not silently fork. |
| `text` | string | yes | Non-empty. Realizes the lock. No eval surface form and no near-duplicate of `eval.jsonl`. |

Optional (all strata):

| Field | Type | Constraint |
|---|---|---|
| `tags` | string[] | Short slugs. Not a dump of the lock. |
| `difficulty` | `1` \| `2` \| `3` | Authoring hint only: 1 = one lock, 2 = lock plus a numeric/checkable step, 3 = composition of two locks. **Not a model score.** |
| `stage` | `1` \| `2` \| `3` | Hint for when the card is *intended* to enter the mix. Sequencing rules still govern. |
| `lang` | string | BCP-47. Default `en`. See §E multilingual. |

### A.2 Stratum C — required

A concept unit teaches **what a thing is**. One atomic named concept per card.

| Field | Type | Constraint |
|---|---|---|
| `concept` | string | The name. Must match the lock, not a slogan. |
| `conditions` | string | Necessary and/or sufficient conditions in prose. Empty, circular, or “X is important because…” fails. |
| `contrasts` | string[] | **≥ 2** near-miss neighbors. Each contrast names a different object and *why it is not this concept*. Synonym padding does not count. |
| `failure_mode` | string[] | **≥ 1** named common confusion. Pedagogical, not a protocol code. |

`concept_ids` for a C card includes its own `id` (and only that, unless the card is a documented alias merge — then park the duplicate instead).

### A.3 Stratum A — required

An analogy unit teaches a **named transferable relation**, not a pretty metaphor.

| Field | Type | Constraint |
|---|---|---|
| `relation_R` | string | One sentence naming \(R\). Must be checkable. Not a simile. |
| `source_domain` | string | Named domain. |
| `target_domain` | string | Named domain. **Must not be identical** to `source_domain` (identity is not analogy — §E). |
| `worked_mapping` | string | One explicit mapping under \(R\). |
| `near_miss` | string | Shares surface features with the pair but **not** \(R\). Empty / missing fails. |
| `transfers` | string | What carries. ≥ 1 clause. |
| `does_not_transfer` | string | What does not. ≥ 1 clause. Silence here fails. |

`concept_ids` lists the C ids being related. If a domain concept is not yet a C card, **promote to C first** (§E). Analogy is a training objective. Do not imply analogical reasoning is solved ([`LITERATURE.md`](LITERATURE.md): Lewis and Mitchell 2024; Mitchell and Lewis 2024).

### A.4 Stratum T — required

A textbook unit teaches a **worked procedure** that reuses locked concepts.

| Field | Type | Constraint |
|---|---|---|
| `uses` | string[] | **≥ 1** C `id`s. Every entry ∈ `concept_ids`. Every id must refer to an **accepted** C card. |
| `problem` | string | Stated task. |
| `method` | string | Named procedure. |
| `solution` | string | Worked result. |
| `check` | string | A check the learner or a grader can apply. Empty fails. |
| `exercise` | string | At least one exercise **not** solvable by copying the worked example verbatim (different numbers, different identifiers, or a near-miss setup). |

Code or math without exposition is out. `uses` may not introduce a new concept in prose that is not in `uses[]` (§E).

### A.5 JSON Schema (mechanical)

This schema catches **mechanical** fails. Semantic in/out still need the checklist.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://troy-ll.github.io/AI-Compacting/ccr-card.schema.json",
  "title": "CCR card",
  "type": "object",
  "additionalProperties": false,
  "required": ["id", "stratum", "license", "source_id", "concept_ids", "version", "text"],
  "properties": {
    "id": { "type": "string", "pattern": "^(C|A|T)-[a-z0-9]+(?:-[a-z0-9]+)*-[0-9]{3}$" },
    "stratum": { "enum": ["C", "A", "T"] },
    "license": { "type": "string", "minLength": 1, "not": { "const": "unknown" } },
    "source_id": { "type": "string", "minLength": 1 },
    "concept_ids": {
      "type": "array",
      "minItems": 1,
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true
    },
    "version": { "type": "integer", "minimum": 1 },
    "text": { "type": "string", "minLength": 1 },
    "tags": { "type": "array", "items": { "type": "string" } },
    "difficulty": { "enum": [1, 2, 3] },
    "stage": { "enum": [1, 2, 3] },
    "lang": { "type": "string", "minLength": 2 },
    "concept": { "type": "string", "minLength": 1 },
    "conditions": { "type": "string", "minLength": 1 },
    "contrasts": {
      "type": "array",
      "minItems": 2,
      "items": { "type": "string", "minLength": 1 }
    },
    "failure_mode": {
      "type": "array",
      "minItems": 1,
      "items": { "type": "string", "minLength": 1 }
    },
    "relation_R": { "type": "string", "minLength": 1 },
    "source_domain": { "type": "string", "minLength": 1 },
    "target_domain": { "type": "string", "minLength": 1 },
    "worked_mapping": { "type": "string", "minLength": 1 },
    "near_miss": { "type": "string", "minLength": 1 },
    "transfers": { "type": "string", "minLength": 1 },
    "does_not_transfer": { "type": "string", "minLength": 1 },
    "uses": {
      "type": "array",
      "minItems": 1,
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": true
    },
    "problem": { "type": "string", "minLength": 1 },
    "method": { "type": "string", "minLength": 1 },
    "solution": { "type": "string", "minLength": 1 },
    "check": { "type": "string", "minLength": 1 },
    "exercise": { "type": "string", "minLength": 1 }
  },
  "allOf": [
    {
      "if": { "properties": { "stratum": { "const": "C" } }, "required": ["stratum"] },
      "then": { "required": ["concept", "conditions", "contrasts", "failure_mode"] }
    },
    {
      "if": { "properties": { "stratum": { "const": "A" } }, "required": ["stratum"] },
      "then": { "required": ["relation_R", "source_domain", "target_domain", "worked_mapping", "near_miss", "transfers", "does_not_transfer"] }
    },
    {
      "if": { "properties": { "stratum": { "const": "A" } }, "required": ["stratum"] },
      "then": {
        "not": {
          "properties": {
            "source_domain": { "type": "string" },
            "target_domain": { "type": "string" }
          },
          "$comment": "identity domains: enforce in checklist V-A4 (casefold+trim); schema cannot compare two fields portably"
        }
      }
    },
    {
      "if": { "properties": { "stratum": { "const": "T" } }, "required": ["stratum"] },
      "then": { "required": ["uses", "problem", "method", "solution", "check", "exercise"] }
    }
  ]
}
```

Rejected records are a wrapper, **not** train cards:

```json
{
  "disposition": "rejected",
  "reject_code": "FM-C02",
  "reject_reason": "one sentence",
  "card": { }
}
```

Park path: `rejected/{id}.json` (or a JSONL line with the wrapper). Reason is mandatory. Cards without a reason are not parked; they are unfinished.

---

## B. In/out rules and validation checklist

Restated from [`PAPER.md`](PAPER.md) §4. If this table and the paper disagree on a *claim*, the paper wins. If they disagree on a *checkable field*, this file wins.

### B.1 Stratum C

**In:** one atomic named concept; necessary/sufficient conditions in prose; ≥2 explicit contrasts; ≥1 named failure mode; short definitional core.

**Out:** trivia, celebrity, news-of-the-week, brand lists; slogans or “X is important because…” with no conditions; multi-concept tours that never lock a core; unsourced numerical trivia offered as knowledge.

### B.2 Stratum A

**In:** \(R\) named in one sentence; source and target domains specified and distinct; one worked mapping; one near-miss that shares surface features but not \(R\); what transfers and what does not.

**Out:** metaphor-as-decoration; unexplained similes; literary flourish with no checkable \(R\); pair-memorization with no named relation and no near-miss; any implication that analogical reasoning is a solved LLM capability.

### B.3 Stratum T

**In:** cites Stratum C concept ids; problem → method → solution → check; ≥1 exercise not solved by copying the worked example; a check a grader can apply.

**Out:** unfiltered crawl, forum dumps, SEO blogs; duplicated boilerplate and license walls; code or math with no exposition; textbook-shaped prose with no checkable procedure; synthetic text whose only virtue is that an LLM wrote it.

### B.4 Validation checklist

Run **in order**. First FAIL stops the card (still record every fail you saw). Mechanical vs semantic is marked. `PASS` only if every applicable row is PASS.

**Disposition:** FAIL + `reject` → park under `rejected/` with `reject_code`. FAIL + `rewrite` → one rewrite attempt, then park if still FAIL. FAIL + `log` / `kill` are curriculum- or eval-level (catalog §C); they do not “fix” a card by editing the eval.

#### Mechanical (schema / agent)

| ID | Check | Applies | Fail → |
|---|---|---|---|
| V-M1 | JSON parses. Schema validates. `id` / `stratum` / `license` / `source_id` / `concept_ids` / `version` / `text` present and typed. | all | reject |
| V-M2 | `stratum` is exactly one of C, A, T. No second primary stratum field. | all | reject |
| V-M3 | C: `concept`, `conditions` non-empty; `contrasts.length ≥ 2`; `failure_mode.length ≥ 1`. | C | reject |
| V-M4 | A: `relation_R`, `source_domain`, `target_domain`, `worked_mapping`, `near_miss`, `transfers`, `does_not_transfer` all non-empty. | A | reject |
| V-M5 | T: `uses.length ≥ 1`; `problem`, `method`, `solution`, `check`, `exercise` non-empty. Every `uses[]` ∈ `concept_ids[]`. | T | reject |
| V-M6 | `license` is not `unknown` and not empty. `source_id` is a locator, not a slogan. | all | reject |
| V-M7 | `version ≥ 1` integer. Duplicate `id` at a different lock without a bump: fail. | all | reject |
| V-M8 | No illegal result fields on the card (`mmlu`, `task_success`, `ppl`, `phone_rss`, product slogans as metrics). | all | reject |

#### Semantic (human or agent-with-judgment)

| ID | Check | Applies | Fail → |
|---|---|---|---|
| V-C1 | Exactly one atomic concept. The name in `concept` is the lock. | C | rewrite |
| V-C2 | `conditions` are necessary and/or sufficient, not a slogan, not circular, not “X is important because…”. | C | rewrite |
| V-C3 | Each contrast is a *different object* plus a reason it fails the conditions. Two phrasings of the same neighbor do not count as two. | C | rewrite |
| V-C4 | `failure_mode` is a real confusion, not “students might be confused.” | C | rewrite |
| V-C5 | Not trivia / celebrity / news / brand list / unsourced numerical trivia-as-knowledge. | C | reject |
| V-A1 | `relation_R` is a named transferable relation, not a metaphor or unexplained simile. | A | rewrite |
| V-A2 | Worked mapping actually uses \(R\). | A | rewrite |
| V-A3 | Near-miss shares surface features and **fails** \(R\). Missing near-miss is not optional. | A | reject |
| V-A4 | `source_domain` ≠ `target_domain` after casefold and trim. Identity → not an A card. | A | reject |
| V-A5 | `does_not_transfer` is specific. “Nothing is different” fails. | A | rewrite |
| V-A6 | Text does not claim analogical reasoning is solved in LLMs or in this recipe. | A | rewrite |
| V-T1 | Every concept the procedure *needs* is in `uses[]`, and those C cards exist and are accepted. | T | rewrite |
| V-T2 | `check` is applicable by a learner or grader (numeric identity, boolean test, executable assertion, or binary rubric). “Looks right” fails. | T | reject |
| V-T3 | `exercise` cannot be solved by copying `solution` verbatim. | T | reject |
| V-T4 | Exposition exists. Bare code/math dump fails. | T | rewrite |
| V-T5 | T does not introduce a new lockable concept. If it does, promote to C first. | T | rewrite |
| V-H1 | `text` realizes every required lock field and does not contradict them. | all | rewrite |
| V-H2 | Synthetic C/A/T: in-rules hold. A prompt template is not sufficient. LLM-shaped fluff fails. | all | reject |
| V-H3 | License and source are honest. License-wall paste, scraped ToS-blocked text, or missing attribution fails. | all | reject |
| V-H4 | No eval leak: card is not `eval.jsonl` item, paraphrase, or near-duplicate (once that file exists). Until freeze, treat the *future* eval as off-limits for cute “preview” items. | all | reject |
| V-H5 | No HAILP-manifesto / 360M-Android-product language. No stacking Claim P into the card. | all | reject |
| V-H6 | Chatty instruction tone does not create a fourth training stratum. Chat probes live in the eval slice, not as un-checked T. | all | rewrite |

**PASS** = all applicable rows PASS. **FAIL** = any row FAIL. There is no “mostly CCR.”

---

## C. Failure-mode catalog

Every mode has a **required mitigation**. Do not invent a softer one after seeing data.

Mitigation vocabulary:

| Verb | Means |
|---|---|
| **reject** | Park under `rejected/` with this code. Do not train. |
| **rewrite** | One attempt to satisfy the checklist. Second fail → reject. |
| **log** | Write the fact in the run note (mix, tokens, hash). Silence is itself a fail. |
| **kill** | The *claim or run status* is dead under [`PAPER.md`](PAPER.md) §7.2. Do not quietly edit the eval or the kill table. |

### C.1 Card-level

#### FM-C01 Missing conditions

**Shows up as:** C card with a name, vibes, and no necessary/sufficient prose; `conditions` empty, circular, or tautological (“gradient descent is when you do gradient descent”).

**Mitigation:** **rewrite**. If the author cannot state conditions, **reject**.

#### FM-C02 Slogan / no-lock

**Shows up as:** “X is important because…”, “always use Y”, motivational filler, product slogans. Contrasts missing or decorative.

**Mitigation:** **reject**. A slogan is not a C card in waiting; it is a different object.

#### FM-C03 Multi-concept mush

**Shows up as:** one unit tours several cores and never locks one; `concept` is a conjunction (“overfitting and underfitting and regularization”); C `concept_ids` lists many ids that are not aliases.

**Mitigation:** **rewrite** into separate C cards (and T/A that *cite* them). If the tour is the point, **reject**.

#### FM-C04 Trivia

**Shows up as:** celebrity, news-of-the-week, brand lists, unsourced numerical trivia offered as knowledge; conditions replaced by a factoid.

**Mitigation:** **reject**.

#### FM-C05 Metaphor without \(R\)

**Shows up as:** A card that is a simile or decoration (“networks are like brains,” “attention is like a spotlight”) with no checkable `relation_R`.

**Mitigation:** **reject**, or **rewrite** only if a real \(R\) can be named, domains distinguished, mapping and near-miss supplied. Pretty prose alone is not a rewrite.

#### FM-C06 Pair-memorization

**Shows up as:** SAT-style “A is to B as C is to D” with no named relation and no near-miss; reward for recalling the stock pair.

**Mitigation:** **reject**. Do not launder into eval A items either ([`eval-protocol.md`](eval-protocol.md)).

#### FM-C07 Near-miss missing

**Shows up as:** A card with \(R\) and a mapping but no foil; or a foil that still satisfies \(R\); or “near-miss: none.”

**Mitigation:** **reject** (schema + V-A3). Analogies without a foil train surface match.

#### FM-C08 T without check

**Shows up as:** worked prose, no `check`; or check = “the answer seems reasonable.”

**Mitigation:** **reject**.

#### FM-C09 T without citing C

**Shows up as:** empty `uses[]`; `uses` pointing at A/T ids; `uses` pointing at C ids that do not exist or are parked; procedure depends on an unnamed concept.

**Mitigation:** **rewrite** (promote missing concepts to C, then cite). If the T is really a blog dump, **reject**.

#### FM-C10 Exercise copyable

**Shows up as:** exercise is the worked example with a synonym swapped, or the same numbers; solution pastes into the exercise unchanged.

**Mitigation:** **reject**.

#### FM-C11 Synthetic fluff

**Shows up as:** LLM-shaped educational tone, no lock; template filled with adjectives; “textbook-quality” as a vibe rather than in-rules.

**Mitigation:** **reject**. Synthetic C/A/T are allowed **only** if in-rules hold (§E).

#### FM-C12 License / hygiene fail

**Shows up as:** `license=unknown`; license-wall textbook paste; missing `source_id`; boilerplate ToS; claiming CC0 on copied All Rights Reserved text.

**Mitigation:** **reject**. Do not train. Do not “fix” by lying about the license.

#### FM-C13 Eval leak / near-dup of `eval.jsonl`

**Shows up as:** train card shares surface form, paraphrase, or structure+numbers with a frozen eval item; “practice” cards written by peeking at the eval; samples copy-pasted into `eval.jsonl` later without a new freeze.

**Mitigation:** **reject** the train card. **kill** the run if leakage is discovered after train start. Never rewrite a leak into a “sufficiently different” train item after seeing the eval. Freeze first ([`eval-protocol.md`](eval-protocol.md)).

### C.2 Curriculum-level

#### FM-K01 A-before-C sequencing

**Shows up as:** Stage 1 A-only or A-heavy before concepts are locked; analogies as decoration; mix plan that starts at Stage 3.

**Mitigation:** **reject** the mix plan. **rewrite** the schedule to §D. Do not start A-only.

#### FM-K02 Stratum mashup that evades rules

**Shows up as:** one blob tagged C that is actually a T without a check; “C+A+T” as a single primary; chatty filler labeled T; a metaphor labeled C.

**Mitigation:** **reject** the unit. Split and re-check each piece. Mashup is not a fourth stratum.

#### FM-K03 Mixture drift unlogged

**Shows up as:** tokens-per-stratum not logged; silent walk away from the pre-registered mix; “we eyeballed it.”

**Mitigation:** **log** is mandatory on every run. If this is billed as the **first reported run** and the mix was not pre-registered, **kill** that status (it is secondary, or it does not exist).

#### FM-K04 Under-train vs Chinchilla

**Shows up as:** a short smoke (Kaggle 1k steps, laptop demo) used to kill or crown CCR; token budget far below a compute-optimal reading for ~50M, then “CCR failed.”

**Mitigation:** **log** the token budget and the **TODO-cite** Hoffmann et al. (2022) table row used for ~50M ([`LITERATURE.md`](LITERATURE.md)). Do **not** kill the CCR *hypothesis* for under-training. **kill** the comparison if the budget was not pre-registered. Do not invent the multiplier.

#### FM-K05 CCR vs web control confounding (different token budgets)

**Shows up as:** CCR run saw more tokens, a different tokenizer, a different param count, or a different eval than the Wikipedia/web control; then “CCR won.”

**Mitigation:** **kill** the comparison. Same tokenizer, same ~50M class, **same pre-registered token budget**, same frozen eval. A mismatched control is not a control.

#### FM-K06 HAILP manifesto language in CCR docs

**Shows up as:** CCR described as a field appliance, a 360M Android product, a FAISS appliance, or an old product slogan as if that were this protocol.

**Mitigation:** **rewrite** the doc sentence. **reject** any card that uses that language (V-H5). Engineering notes stay in [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) / [`Kaggle.md`](Kaggle.md) and are not paper results.

#### FM-K07 Stacking Claim P into Claim R via CCR

**Shows up as:** one table mixing Phi-3 MMLU / 1–3B INT4 phone band with a 50M CCR `task_success`; “CCR therefore gives a phone chatty generalist”; scoring Claim P on the 30-item file.

**Mitigation:** **kill** the stacked claim. CCR is a Claim **R** protocol. Claim P stays a literature envelope ([`PAPER.md`](PAPER.md) §2, §6).

### C.3 Eval-level

#### FM-E01 MMLU-under-300M

**Shows up as:** MMLU (or a 57-task 4-way cousin) as primary at ~50M or under 300M; “we’re near chance, therefore CCR failed / succeeded.”

**Mitigation:** **reject** MMLU as a primary. Do not run it as a headline at ~50M ([`PAPER.md`](PAPER.md) §3.2; Hendrycks et al., 2021).

#### FM-E02 Unfrozen eval

**Shows up as:** train starts before `eval.jsonl` exists and is hashed with this protocol; living leaderboard; item cap 31+.

**Mitigation:** **kill** the run as a reported run. Freeze ≤30 items first ([`eval-protocol.md`](eval-protocol.md)).

#### FM-E03 Rewriting kill after seeing data

**Shows up as:** ties resolved post-hoc; margin edited; items dropped; `success_rule` softened; kill table quietly rewritten because the first run hurt.

**Mitigation:** **kill** is a scientific outcome ([`PAPER.md`](PAPER.md) §7.2). Restore the frozen kill. Disclose or do not do. Secondary runs are labeled `secondary`.

#### FM-E04 PPL-only scoring

**Shows up as:** reporting held-out PPL as the CCR result; skipping `task_success`; “language modeling got better, so the recipe works.”

**Mitigation:** **reject** the report as a CCR result. Primary is `task_success` vs the same-data ~50M GPT control. PPL is secondary and has its own kill margin (default +15% vs control).

#### FM-E05 Scoring Claim P on the 30-item Claim R eval

**Shows up as:** 1–7B phone-band models scored on `eval.jsonl` as if that were Claim P; or 50M `task_success` advertised as phone-fit.

**Mitigation:** **reject** that score. Claim P is not scored on this file. A later phone probe uses a named COTS device and [`PAPER.md`](PAPER.md) §5.

### C.4 Catalog index

| Code | Name | Level | Mitigation |
|---|---|---|---|
| FM-C01 | Missing conditions | card | rewrite / reject |
| FM-C02 | Slogan / no-lock | card | reject |
| FM-C03 | Multi-concept mush | card | rewrite / reject |
| FM-C04 | Trivia | card | reject |
| FM-C05 | Metaphor without \(R\) | card | reject / rewrite if \(R\) is real |
| FM-C06 | Pair-memorization | card | reject |
| FM-C07 | Near-miss missing | card | reject |
| FM-C08 | T without check | card | reject |
| FM-C09 | T without citing C | card | rewrite / reject |
| FM-C10 | Exercise copyable | card | reject |
| FM-C11 | Synthetic fluff | card | reject |
| FM-C12 | License / hygiene fail | card | reject |
| FM-C13 | Eval leak / near-dup | card / run | reject / kill run |
| FM-K01 | A-before-C sequencing | curriculum | reject mix; rewrite schedule |
| FM-K02 | Stratum mashup | curriculum | reject unit |
| FM-K03 | Mixture drift unlogged | curriculum | log; kill first-run status |
| FM-K04 | Under-train vs Chinchilla | curriculum | log TODO-cite; do not kill hypothesis for under-train |
| FM-K05 | Token-budget confounding | curriculum | kill comparison |
| FM-K06 | HAILP manifesto language | curriculum | rewrite doc; reject card |
| FM-K07 | Stacking Claim P into R | curriculum | kill stacked claim |
| FM-E01 | MMLU-under-300M | eval | reject as primary |
| FM-E02 | Unfrozen eval | eval | kill reported-run status |
| FM-E03 | Rewriting kill after data | eval | restore freeze; disclose |
| FM-E04 | PPL-only scoring | eval | reject report as CCR result |
| FM-E05 | Claim P on Claim R eval | eval | reject that score |

---

## D. Sequencing and mixture

From [`PAPER.md`](PAPER.md) §4.2. Required.

### D.1 Stages

1. **Stage 1 — C-heavy:** lock concepts.
2. **Stage 2 — C+T:** procedures that use locked concepts.
3. **Stage 3 — C+A+T:** transfer via named relations.

**Do not start with A-only.** Analogies without locked concepts are decoration (FM-K01).

A card’s optional `stage` hint may not override this. An A card marked `stage: 1` still waits until the C ids it cites are in the Stage 1 mix.

### D.2 Default hypothesis mixes

Tokens-per-stratum are **logged**. Default *hypothesis* mix for a first run (**not** a claimed optimum):

| Stage | C | T | A |
|---|---|---|---|
| 1 | ~50% | ~40% | ~10% |
| 2 | ~25% | ~65% | ~10% |
| 3 | ~20% | ~50% | ~30% |

A sweep may change these. The first reported run uses the **pre-registered** mix. Drift without a run-note is FM-K03.

Stage 1’s 10% A is **not** A-only and is not Stage 3. Those A tokens still require cited C cards already in the stream (or the A card is held). Prefer Stage 1 A volume at the low end if C coverage is thin.

### D.3 What may enter the stream

| Unit | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Accepted C | yes | yes | yes |
| Accepted T whose `uses[]` are already locked | sparse | yes | yes |
| Accepted A whose `concept_ids` are already locked | sparse, not A-only | sparse | yes |
| Parked / rejected | never | never | never |
| Eval items / near-dups | never | never | never |

### D.4 Hygiene (train)

- Freeze `eval.jsonl` before any train; hash eval + [`eval-protocol.md`](eval-protocol.md) + this spec.
- Prefer public-domain or clearly licensed sources for T.
- Record `license`, `source_id`, `stratum`, `concept_ids` on every unit.
- Same tokenizer, same ~50M class, same token budget for CCR vs Wikipedia/web control (FM-K05).
- INT4 is a storage / later-phone step, not a training dtype unless separately pre-registered ([`PAPER.md`](PAPER.md) §7.1).

---

## E. Situations and edge cases

If a situation is not listed, the default is: **apply the checklist**; do not invent a waiver.

### E.1 Multilingual

- `concept_ids` are language-agnostic. The lock is the conditions, not the English string.
- Set `lang` (BCP-47). Default `en`.
- A translation is a new `version` or a new `id` suffix, not a silent overwrite. Conditions must still lock; do not ship a slogan in another language and call it C.
- Contrasts and near-misses must work in that language (a false-friend is a valid contrast; an untranslated English foil in a `lang: fr` card is a hygiene fail).
- Mix logging is by stratum, not by language, unless a multilingual run is pre-registered as such.
- Eval leak checks are cross-lingual: a translation of an eval item is a near-dup (FM-C13).

### E.2 Code and math

- Allowed in T (and in C conditions when the concept *is* a formal object).
- **Out** if there is no exposition ([`PAPER.md`](PAPER.md) §4.1 T).
- `check` must be executable, numeric, or proof-checkable. Comments in code are not a check.
- Identifiers are not the lock: a function named `gradient_descent` with no conditions is FM-C02.
- Copyable exercises include “change the variable name.” Change the instance (numbers, predicates, shapes).
- Do not paste license-walled library docs (FM-C12). Prefer a tiny self-contained snippet.

### E.3 Ambiguous concept boundaries

- Same conditions, two names → **one** `id`; put the other name in `tags` or `text`. Do not mint a second C.
- One name, two lockable cores → **split** into two C cards. Mush is FM-C03.
- If you cannot state conditions that separate A from B, you do not yet have two concepts; park until you can, or write the one concept you can lock.
- Do not “save tokens” by merging unrelated cores.

### E.4 Analogy that is actually identity

- After casefold+trim, `source_domain` == `target_domain` → **not an A card** (V-A4). Restate as C, or as a T worked example in that domain.
- Near-identity (CPU cache vs GPU cache) is allowed only if \(R\) is non-trivial **and** the domains are distinguished **and** the near-miss is not “the other vendor’s cache.” If \(R\) is “both are caches,” **reject** (FM-C05 / FM-C06).
- A mapping that is a definition (`source` *is* `target`) is C, not A.

### E.5 T that introduces new concepts

- **Promote to C first.** Accept the C card. Then T may cite it in `uses[]`.
- A new lockable term in `text` / `method` that is not in `uses[]` and is not ordinary language → FAIL V-T5. Ordinary language (“number,” “step”) is not a concept card.
- Do not hide a new concept inside an analogy used as exposition inside T. That is mashup (FM-K02): extract C, then A or T.

### E.6 Missing public T sources

- Prefer public-domain or clearly licensed textbooks / notes.
- If none exist for a needed procedure: write a **synthetic T** that still satisfies T in-rules; `source_id` = `synthetic:...`; `license` honest (in-repo default: `Apache-2.0` to match this repo, or `CC0-1.0` if you are the sole author and choose that).
- FineWeb-Edu is a **control candidate**, not a T source that makes crawl into CCR ([`LITERATURE.md`](LITERATURE.md)).
- Do not scrape license-walled books and call it hygiene (FM-C12).
- Missing source is not permission to skip `check` or `uses[]`.

### E.7 Synthetic C/A

- **Allowed only if in-rules hold.** A prompt template, a style guide, or “make it educational” is not sufficient (FM-C11).
- Fill every required field. Run this checklist. `source_id` must say `synthetic:`.
- Synthetic is not a waiver for eval leak, trivia, slogans, or missing near-miss.
- We do not have Phi’s private pipeline. Do not pretend we do ([`PAPER.md`](PAPER.md) §4.2).

### E.8 When to park under `rejected/`

Park **immediately** (no rewrite loop) when the fail is: FM-C02, FM-C04, FM-C06, FM-C07, FM-C08, FM-C10, FM-C11, FM-C12, FM-C13, FM-K02, V-H5.

Park **after one rewrite** when the fail is rewrite-class (FM-C01, FM-C03, FM-C05-if-\(R\)-is-real, FM-C09, V-T1, V-T5, V-H1).

Every parked file includes `reject_code`, `reject_reason`, and the attempted `card`. No reason → not parked, not accepted.

Do **not** park:

- a card you have not checked;
- an eval idea (eval ideas go to the freeze process, not `rejected/`);
- a mix-plan fail (that is a run-note / kill, not a card).

### E.9 Other defaults

| Situation | Handling |
|---|---|
| Empty `text` with perfect fields | FAIL V-M1 / V-H1. The stream needs `text`. |
| Perfect `text` with empty fields | FAIL mechanical. Do not reverse-engineer fields after the fact to launder a blog post. |
| Chat / instruction unit | Eval slice only ([`eval-protocol.md`](eval-protocol.md)). Not a fourth train stratum. |
| Duplicate lock, new prose | `version` bump on the same `id`, or a new id if it is a *different* lock. Near-dup of eval still FM-C13. |
| H(AI)LP / KV vs state as content | Legitimate **Claim R** C/T material. Still no manifesto language, no Claim P stack (FM-K06, FM-K07). |
| Wanting MMLU “just as a side table” | FM-E01. Side tables become headlines. Don’t. |

---

## What this file is not

- Not a claimed win.
- Not `eval.jsonl`.
- Not a token budget (Hoffmann row: **TODO-cite**).
- Not TinyStories, not FineWeb-Edu, not Lee et al.’s masking schedule, not Phi’s private stack.
- Not permission to train this season.

When a run happens, it hashes this file with the eval freeze. It does not rewrite this catalog to absorb Claim P into Claim R.
