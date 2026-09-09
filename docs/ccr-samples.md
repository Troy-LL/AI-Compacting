# CCR sample pack — example shapes, not a corpus

**Author:** Troy  
**Season:** theoretical paper — no new training runs

These cards show the schema and the checklist in [`ccr-spec.md`](ccr-spec.md). They are **example shapes**. They are not a training corpus. They are not `eval.jsonl`. They do not spend a token budget. **No result is claimed.**

Do not copy these items into a later `eval.jsonl` and also into a train stream. If a freeze uses a close cousin, the train side is FM-C13.

Thesis owner: [`PAPER.md`](PAPER.md) §4. Eval freeze: [`eval-protocol.md`](eval-protocol.md).

---

## Accepted C

### C-gradient-descent-001

One atomic update rule. Conditions, two contrasts, one named confusion.

```json
{
  "id": "C-gradient-descent-001",
  "stratum": "C",
  "license": "Apache-2.0",
  "source_id": "synthetic:docs/ccr-samples.md#C-gradient-descent-001",
  "concept_ids": ["C-gradient-descent-001"],
  "version": 1,
  "tags": ["optimization", "ml"],
  "difficulty": 2,
  "stage": 1,
  "lang": "en",
  "concept": "gradient descent",
  "conditions": "A parameter-update rule is gradient descent when (1) a scalar loss L is differentiable, or a named surrogate is, with respect to parameters θ, (2) each step computes or estimates ∇_θ L, and (3) θ is replaced by θ minus a positive step-size times that gradient (or a minibatch estimate of it). Full-batch evaluation of L is not required; stochastic gradient descent is included when the gradient is a minibatch estimate.",
  "contrasts": [
    "Newton or quasi-Newton steps: they use second-order information (a Hessian or an approximation), not the gradient alone.",
    "Zeroth-order or evolutionary search: they update from function values or random perturbations and never form ∇_θ L."
  ],
  "failure_mode": [
    "Calling any iterative improvement gradient descent, including grid search or genetic algorithms that never compute a gradient."
  ],
  "text": "Concept: gradient descent. A parameter-update rule is gradient descent when a scalar loss L is differentiable (or a named surrogate is) in the parameters θ, each step computes or estimates ∇_θ L, and θ is replaced by θ minus a positive step-size times that gradient (or a minibatch estimate). Full-batch evaluation of L is not required. Contrast Newton / quasi-Newton: those steps use a Hessian or an approximation, not the gradient alone. Contrast zeroth-order / evolutionary search: those updates use function values or random perturbations and never form ∇_θ L. Common confusion: treating any iterative improvement as gradient descent, including grid search or genetic algorithms that never compute a gradient."
}
```

Checklist: mechanical V-M1–V-M3, V-M6–V-M8 PASS. Semantic V-C1–V-C5, V-H1–V-H6 PASS. Synthetic, in-rules hold (E.7).

### C-overfitting-001

Held-out comparison is in the conditions. Underfit and shift are contrasts, not synonyms.

```json
{
  "id": "C-overfitting-001",
  "stratum": "C",
  "license": "Apache-2.0",
  "source_id": "synthetic:docs/ccr-samples.md#C-overfitting-001",
  "concept_ids": ["C-overfitting-001"],
  "version": 1,
  "tags": ["generalization", "ml"],
  "difficulty": 1,
  "stage": 1,
  "lang": "en",
  "concept": "overfitting",
  "conditions": "A fit overfits when, on the intended task family, performance on the training split improves (or is already strong) while performance on a held-out split from the same family worsens or stalls relative to an earlier or simpler fit. The held-out split must be the same task family; a comparison is required. High training performance alone is not overfitting.",
  "contrasts": [
    "Underfitting: training and held-out performance are both poor; the fit has not captured the training split either.",
    "Distribution shift: held-out data are a different task family or domain, so a train/held-out gap is not evidence of overfitting to the original family."
  ],
  "failure_mode": [
    "Calling any high training accuracy overfitting without a same-family held-out comparison."
  ],
  "text": "Concept: overfitting. A fit overfits when, on the intended task family, performance on the training split improves (or is already strong) while performance on a held-out split from the same family worsens or stalls relative to an earlier or simpler fit. A comparison is required. High training performance alone is not overfitting. Contrast underfitting: train and held-out are both poor. Contrast distribution shift: the held-out data are a different task family, so a gap is not overfitting to the original family. Common confusion: labeling any high training accuracy as overfitting without a same-family held-out comparison."
}
```

### C-kv-cache-vs-fixed-state-001

Claim **R** material: process memory vs sequence length. Not a phone product. Not a product slogan. Not Claim P.

```json
{
  "id": "C-kv-cache-vs-fixed-state-001",
  "stratum": "C",
  "license": "Apache-2.0",
  "source_id": "synthetic:docs/ccr-samples.md#C-kv-cache-vs-fixed-state-001",
  "concept_ids": ["C-kv-cache-vs-fixed-state-001"],
  "version": 1,
  "tags": ["inference", "memory", "claim-r"],
  "difficulty": 2,
  "stage": 1,
  "lang": "en",
  "concept": "KV cache versus fixed recurrent state",
  "conditions": "At decode, a transformer KV cache stores keys and values for past tokens; the stored cache size grows with sequence length (and with batch, layers, and heads). A fixed recurrent state is a bounded tensor updated in place; its stored size does not grow with sequence length. Weight file size is not this distinction. Process RSS includes weights plus this state (or cache) plus runtime; the concept names the sequence-dependent term, not a device measurement.",
  "contrasts": [
    "Recompute-all attention: keys and values are rebuilt each step and need not persist as a growing cache; compute grows, but the stored KV may not.",
    "Sliding-window or local attention: a cache may still be stored, but it is capped by the window, not by the full sequence length T."
  ],
  "failure_mode": [
    "Equating weight-file size with process RSS, or saying a fixed state means the model has no memory of the past (it has a bounded state, not an empty one)."
  ],
  "text": "Concept: KV cache versus fixed recurrent state. At decode, a transformer KV cache stores keys and values for past tokens; stored cache size grows with sequence length (and batch, layers, heads). A fixed recurrent state is a bounded tensor updated in place; its stored size does not grow with sequence length. Weight file size is not this distinction. Process RSS includes weights plus this state or cache plus runtime; the concept names the sequence-dependent term, not a named-device measurement. Contrast recompute-all attention: K and V are rebuilt each step and need not persist as a growing cache. Contrast sliding-window / local attention: a cache may still exist but is capped by the window, not by full T. Common confusion: treating file size as process RSS, or treating fixed state as “no memory of the past.” This card does not claim a phone chatty generalist and does not report a measured T*."
}
```

---

## Accepted A

### A-cache-working-memory-001

Named \(R\), two domains, worked mapping, near-miss, transfer and non-transfer.

```json
{
  "id": "A-cache-working-memory-001",
  "stratum": "A",
  "license": "Apache-2.0",
  "source_id": "synthetic:docs/ccr-samples.md#A-cache-working-memory-001",
  "concept_ids": ["C-kv-cache-vs-fixed-state-001"],
  "version": 1,
  "tags": ["memory", "analogy"],
  "difficulty": 2,
  "stage": 3,
  "lang": "en",
  "relation_R": "A small working store used for items currently being processed is capacity-limited; new items displace old ones, while a separate long-term store is not under the same per-step cap.",
  "source_domain": "machine working store (CPU cache, or a transformer KV cache as the decode-time working store)",
  "target_domain": "human working memory versus long-term memory",
  "worked_mapping": "Evicting a cache line to admit a new block maps to dropping a digit from a phone number you are rehearsing when a new number is spoken.",
  "near_miss": "Comparing the working store to a library or disk (large long-term memory). Surface word “memory” matches; the per-step capacity cap and displacement rule do not.",
  "transfers": "Bounded capacity, recency, and eviction or overwrite under load.",
  "does_not_transfer": "Cache associativity and line size; biological consolidation; rehearsal as literal DRAM refresh; any claim that analogical reasoning is solved in language models.",
  "text": "Relation R: a small working store used for items currently being processed is capacity-limited; new items displace old ones, while a separate long-term store is not under the same per-step cap. Source domain: machine working store (CPU cache, or a transformer KV cache as the decode-time working store). Target domain: human working memory versus long-term memory. Worked mapping: evicting a cache line to admit a new block maps to dropping a digit from a phone number you are rehearsing when a new number is spoken. Near-miss: calling a library or disk the working store because both are “memory” — the surface word matches; the per-step cap and displacement rule do not. Transfers: bounded capacity, recency, eviction or overwrite under load. Does not transfer: associativity and line size; biological consolidation; rehearsal as DRAM refresh. This unit trains a named relation. It does not claim analogical reasoning is solved. Cited concept: C-kv-cache-vs-fixed-state-001."
}
```

Stage hint is 3. Do not start a run A-only (FM-K01). The cited C card must already be locked.

### A-overfit-answer-key-001

Deliberate near-miss: high train *and* high held-out is not overfitting.

```json
{
  "id": "A-overfit-answer-key-001",
  "stratum": "A",
  "license": "Apache-2.0",
  "source_id": "synthetic:docs/ccr-samples.md#A-overfit-answer-key-001",
  "concept_ids": ["C-overfitting-001"],
  "version": 1,
  "tags": ["generalization", "analogy"],
  "difficulty": 2,
  "stage": 3,
  "lang": "en",
  "relation_R": "Success on the items used to fit is not, by itself, evidence of success on unseen items from the same task family.",
  "source_domain": "a student who memorizes a practice-exam answer key",
  "target_domain": "a model that fits training-split labels",
  "worked_mapping": "Reciting the practice-exam answers maps to driving training loss down on the training split; a new exam from the same syllabus maps to a same-family held-out split.",
  "near_miss": "A student who understands the method and also gets the practice items right (strong train and strong held-out). Surface: both “did well on practice.” Missing R: there is no train/held-out dissociation, so this is not overfitting.",
  "transfers": "You need a same-family held-out check; more drilling on the seen items can look like progress while the unseen items get worse.",
  "does_not_transfer": "Moral categories such as cheating; human intention; a model “wanting” to memorize; treating a different subject’s exam as the held-out split (that is shift, not this R).",
  "text": "Relation R: success on the items used to fit is not, by itself, evidence of success on unseen items from the same task family. Source: a student who memorizes a practice-exam answer key. Target: a model that fits training-split labels. Worked mapping: reciting the practice-exam answers maps to driving training loss down on the training split; a new exam from the same syllabus maps to a same-family held-out split. Near-miss: a student who understands the method and also scores well on practice (strong train and strong held-out). The surface “did well on practice” matches; there is no train/held-out dissociation, so R does not hold and this is not overfitting. Transfers: a same-family held-out check is required; more drilling on seen items can look like progress while unseen items worsen. Does not transfer: cheating as a moral category; human intention; a model wanting to memorize; using a different subject’s exam as held-out (shift, not this R). Cited concept: C-overfitting-001. Analogy is a training objective, not a solved capability."
}
```

---

## Accepted T

### T-gd-one-step-001

One C cite. Problem → method → solution → check. Exercise uses a different step-size; copying `θ=2` fails the check.

```json
{
  "id": "T-gd-one-step-001",
  "stratum": "T",
  "license": "Apache-2.0",
  "source_id": "synthetic:docs/ccr-samples.md#T-gd-one-step-001",
  "concept_ids": ["C-gradient-descent-001"],
  "version": 1,
  "tags": ["optimization"],
  "difficulty": 2,
  "stage": 2,
  "lang": "en",
  "uses": ["C-gradient-descent-001"],
  "problem": "Let L(θ) = (θ − 3)², θ₀ = 1, step-size η = 0.25. Take one gradient-descent step. Report θ₁ and whether L decreased.",
  "method": "Differentiate: ∇L(θ) = 2(θ − 3). Update: θ ← θ − η ∇L(θ). Compare L(θ₀) and L(θ₁).",
  "solution": "∇L(1) = 2(1 − 3) = −4. θ₁ = 1 − 0.25 · (−4) = 2. L(1) = 4. L(2) = 1.",
  "check": "θ₁ equals 2 and L(θ₁) < L(θ₀). A grader rejects any update that does not subtract η times the gradient.",
  "exercise": "Same L and θ₀ = 1, but η = 2. Compute θ₁ and L(θ₁). State whether L decreased. (The numbers are not 2 and 1.)",
  "text": "Uses C-gradient-descent-001. Problem: L(θ) = (θ − 3)², θ₀ = 1, η = 0.25. Take one gradient-descent step; report θ₁ and whether L decreased. Method: ∇L(θ) = 2(θ − 3); θ ← θ − η ∇L(θ); compare L at θ₀ and θ₁. Solution: ∇L(1) = −4; θ₁ = 1 − 0.25 · (−4) = 2; L(1) = 4; L(2) = 1. Check: θ₁ is 2 and L fell. Exercise (not a copy): same L, θ₀ = 1, η = 2; compute θ₁ and L(θ₁) and say whether L decreased. Those values are not 2 and 1; η = 2 overshoots (θ₁ = 9, L = 36)."
}
```

The parenthetical on the exercise in `text` is exposition for this *sample*. A train serialization may omit the numeric spoiler and keep the check on the learner side. Either way the exercise instance is not the worked numbers.

### T-early-stop-001

Two C cites: gradient descent and overfitting.

```json
{
  "id": "T-early-stop-001",
  "stratum": "T",
  "license": "Apache-2.0",
  "source_id": "synthetic:docs/ccr-samples.md#T-early-stop-001",
  "concept_ids": ["C-gradient-descent-001", "C-overfitting-001"],
  "version": 1,
  "tags": ["generalization", "optimization"],
  "difficulty": 3,
  "stage": 2,
  "lang": "en",
  "uses": ["C-gradient-descent-001", "C-overfitting-001"],
  "problem": "A model is updated with gradient descent. After each epoch the losses are: epoch 1 train 2.0 held-out 2.1; epoch 2 train 1.4 held-out 1.6; epoch 3 train 1.0 held-out 1.3; epoch 4 train 0.6 held-out 1.5; epoch 5 train 0.3 held-out 1.9. Choose the last epoch that is not overfitting under C-overfitting-001, and say why epoch 5 is not “better because train is lower.”",
  "method": "After each gradient-descent epoch, compare train and same-family held-out. Overfitting begins when train improves (or stays strong) and held-out worsens or stalls relative to an earlier fit. Stop at the last epoch before that dissociation.",
  "solution": "Epoch 3 is the last epoch before held-out rises (1.3 → 1.5 → 1.9) while train keeps falling. Stop at epoch 3. Epoch 5 has the lowest train loss and is the most overfit on this table.",
  "check": "Chosen epoch is 3. A grader fails any answer that picks epoch 5 because train is lowest, or that calls epoch 1 overfit (held-out has not yet worsened relative to a better earlier fit).",
  "exercise": "New table: epoch 1 train 1.8 held-out 1.9; epoch 2 train 1.2 held-out 1.4; epoch 3 train 0.9 held-out 1.4; epoch 4 train 0.5 held-out 1.8. Pick the stop epoch. Conditions count a held-out stall while train improves as overfitting, so do not copy “3” from the worked example.",
  "text": "Uses C-gradient-descent-001 and C-overfitting-001. Problem: gradient-descent epochs with train/held-out pairs (1: 2.0/2.1, 2: 1.4/1.6, 3: 1.0/1.3, 4: 0.6/1.5, 5: 0.3/1.9). Choose the last epoch that is not overfitting, and say why epoch 5 is not better because train is lower. Method: after each epoch, apply the overfitting conditions to the same-family held-out curve; stop at the last epoch before train keeps improving while held-out worsens or stalls. Solution: stop at epoch 3; epoch 5 is the lowest train and the worst held-out. Check: the chosen epoch is 3; picking 5 for lowest train fails. Exercise: a new table (1: 1.8/1.9, 2: 1.2/1.4, 3: 0.9/1.4, 4: 0.5/1.8). Held-out stalls at 1.4 in epoch 3 while train improves, so the stop is epoch 2; copying “3” fails the check."
}
```

---

## Rejected examples

These are **annotated fails**. They are not train units. They show the checklist saying no.

### rejected/C-gd-slogan-001 — slogan C (FM-C02)

No conditions. No contrasts. No failure mode. Motivational prose.

```json
{
  "disposition": "rejected",
  "reject_code": "FM-C02",
  "reject_reason": "Slogan / no-lock: “gradient descent is important” with no necessary or sufficient conditions, no contrasts, no failure mode. Not a C card. Rewrite is the wrong verb; the object is a slogan.",
  "card": {
    "id": "C-gd-slogan-001",
    "stratum": "C",
    "license": "Apache-2.0",
    "source_id": "synthetic:docs/ccr-samples.md#rejected-C-gd-slogan-001",
    "concept_ids": ["C-gd-slogan-001"],
    "version": 1,
    "concept": "gradient descent",
    "conditions": "",
    "contrasts": [],
    "failure_mode": [],
    "text": "Gradient descent is important because it is how neural networks learn. Always use it. Small models especially need this powerful idea."
  }
}
```

| Checklist | Result |
|---|---|
| V-M3 | FAIL — `conditions` empty; `contrasts` < 2; `failure_mode` < 1 |
| V-C2 | FAIL — slogan, no lock |
| V-C5 | PASS (not celebrity trivia) |
| V-H1 | FAIL — `text` does not realize a lock |
| Disposition | **reject** immediately (FM-C02). Park. Do not train. |

### rejected/A-network-brain-001 — metaphor A (FM-C05)

Simile, no checkable \(R\), no near-miss, domains collapse into decoration.

```json
{
  "disposition": "rejected",
  "reject_code": "FM-C05",
  "reject_reason": "Metaphor without R: “a network is like a brain” is decoration. No checkable relation, no worked mapping, no near-miss. Pair-ish surface match (FM-C06) and missing near-miss (FM-C07) also apply. Filling empty fields with more poetry is not a rewrite.",
  "card": {
    "id": "A-network-brain-001",
    "stratum": "A",
    "license": "Apache-2.0",
    "source_id": "synthetic:docs/ccr-samples.md#rejected-A-network-brain-001",
    "concept_ids": ["C-gradient-descent-001"],
    "version": 1,
    "relation_R": "",
    "source_domain": "brain",
    "target_domain": "neural network",
    "worked_mapping": "",
    "near_miss": "",
    "transfers": "",
    "does_not_transfer": "",
    "text": "A neural network is like a brain: layers are like neurons firing together in a beautiful dance of intelligence."
  }
}
```

| Checklist | Result |
|---|---|
| V-M4 | FAIL — `relation_R`, `worked_mapping`, `near_miss`, `transfers`, `does_not_transfer` empty |
| V-A1 | FAIL — simile, not a named relation |
| V-A3 | FAIL — near-miss missing (FM-C07) |
| V-A6 | FAIL — “intelligence” flourish; reads as capability theater |
| V-H2 | FAIL — synthetic fluff (FM-C11) if shipped as “textbook-quality” |
| Disposition | **reject** immediately (FM-C05, FM-C07). A real \(R\) would be a *different* card, not a patch on this one. |

---

## What this pack does not do

- It does not authorize a train.
- It does not freeze `eval.jsonl`.
- It does not set a Chinchilla token budget (**TODO-cite** the Hoffmann et al., 2022 row if a run is ever authorized).
- It does not score Claim P.
- It does not claim analogical reasoning is solved.
- It does not put H(AI)LP manifesto language on a card.

Further cards, if any, get the same checklist. Volume is not rigor.
