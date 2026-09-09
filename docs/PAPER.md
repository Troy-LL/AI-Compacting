# Compact Competence under a Phone Budget

**Author:** Troy  
**Season:** theoretical paper — no new training runs  
**Repo role:** this file is the thesis owner. Literature notes live in [`LITERATURE.md`](LITERATURE.md). The eval freeze lives in [`eval-protocol.md`](eval-protocol.md). CCR operations live in [`ccr-spec.md`](ccr-spec.md). Training scripts in this repo are later / optional.

**Standing rule.** We do not claim a result without a run. This season states bounds, a gap, a protocol, and kill criteria. Numbers below are from published work or from arithmetic identities (for example, INT4 weight bytes \(\approx 0.5 \times N\)). Anything not pinned to a source is marked **TODO-cite**.

---

## Abstract

Two claims, kept separate.

**Claim P (practical).** A phone-resident *chatty generalist* sits in the dense 1–3B INT4 band: about 0.5–1.5 GB of *weights*. A 7B INT4 model is a published ceiling on current COTS phones, not a target. The supporting literature is Phi-3 on-device, MobileLLM’s on-device design envelope, BitNet’s compression bound, and COTS mobile measurement (MELT and related phone studies).

**Claim R (radical).** Under a *fixed process-RSS budget*, the interesting quantity is the fewest parameters that still yield master-of-none multi-skill competence. A same-data ~50M GPT is the control. H(AI)LP’s fixed recurrent state is a *candidate* architecture for the RSS-vs-sequence axis, not the thesis.

The original contribution this season is not another architecture sketch and not a literature survey. It is an **operational compact curriculum recipe**: concepts, then broad analogies, then dense textbooks, with explicit in/out rules. We state it as a hypothesis plus a protocol. We do not claim it wins without a train.

---

## 1. Two claims, not one win

Do not stack Claim P and Claim R in a single table or a single “we win” sentence. They use different model classes, different success metrics, and different literature.

| | Claim P | Claim R |
|---|---|---|
| Question | What already-published dense models are phone-plausible as a chatty generalist? | Under a fixed process-RSS budget, how few parameters still buy multi-skill competence? |
| Model class | 1–7B dense (INT4 weights) | ~50M class, same data |
| Primary evidence this season | Literature bounds | Gap + protocol (no run) |
| Architecture | One published dense class per row | H(AI)LP fixed-state vs GPT is a *candidate*, not the claim |
| Forbidden merge | Do not put Phi-3 MMLU next to a 50M perplexity and call it one result | Do not put a survival slogan in a parameter table |

Thesis **A (data)** is primary. Thesis **B (architecture)** is related: if the curriculum is held fixed, does a constant-state model buy a longer usable \(T\) inside the same RSS cap? That question is not answered this season.

---

## 2. Claim P — practical phone band

### 2.1 What “phone-fit” means

Phone-fit is **not** “the GGUF file is smaller than device RAM.”

Following the COTS measurement line (Laskaridis et al., 2024; Çöplü et al., 2023) and the on-device design envelope (Liu et al., 2024), a model is phone-fit for a named device class only if both hold:

1. **Process RSS** — weights + KV or recurrent state + runtime + the OS/app share stay under a named budget for the named context length.
2. **Sustained thermal and energy** under a named duration \(T\) of chatty generation (not a one-shot tokens/s peak).

NPU honesty: a speedup is an NPU result only if that backend was the measured path. MELT treats NPU acceleration as a future bet, not a current default (Laskaridis et al., 2024). llama.cpp on Android is often CPU-bound in the same study. We do not write “NPU” next to a CPU or GPU-Metal number.

One model class per claim. Claim P tables contain only the 1–7B dense INT4 class (and explicitly marked BitNet rows if used as a *compression bound*, not as a measured phone stack).

### 2.2 Weight arithmetic, then literature

INT4 stores about 0.5 bytes per parameter. That identity — not a measurement — gives:

- 1B dense INT4 \(\approx\) 0.5 GB weights
- 3B dense INT4 \(\approx\) 1.5 GB weights
- 7B dense INT4 \(\approx\) 3.5 GB weights

Process RSS is larger (KV cache or recurrent state, activations, runtime). Claim P’s “0.5–1.5 GB” is a *weight* band for 1–3B INT4, not a process-RSS measurement.

### 2.3 Published bounds

**Existence of a phone-local chatty 4B-class model.** Phi-3-mini is 3.8B parameters, trained on 3.3T tokens. Abdin et al. (2024) report that 4-bit quantization occupies \(\approx\) 1.8 GB, and that the quantized model ran fully offline on an iPhone 14 (A16 Bionic) at more than 12 tokens/s. This is the strongest public existence proof that a *chatty generalist* can live on a modern phone. It sits just above the 1.5 GB / 3B arithmetic mark; we cite it as the published 4B-class point, not as our measurement.

**On-device design envelope, including a stricter mid-tier reading.** MobileLLM (Liu et al., 2024, ICML) targets on-device use at sub-billion scale. The ICML / arXiv:2402.14905 tables report **125M and 350M** (plus layer-share / +LS variants). Larger Hugging Face MobileLLM variants (600M / 1B / 1.5B) are separate releases — **TODO-cite if used**; do not list them as ICML result rows. The paper’s DRAM figure is 6–12 GB on then-current phones, and it argues a foreground app should not take more than about 10% of DRAM because the OS and other apps share it. That 10% rule is *their* deployment heuristic, not a universal OS fact. It is why they treat LLaMA-2 7B at 8-bit as prohibitively expensive in main memory and motivate the reported 125M / 350M family. They also give an energy order-of-magnitude of 0.1 J/token per billion parameters (citing earlier hardware energy models). We treat that joule figure as a literature estimate, not a phone-lab number.

Tension, not contradiction: MobileLLM’s 10% DRAM heuristic pushes toward \(\le\)1B on mid-tier RAM. Phi-3’s iPhone 14 run shows a 3.8B 4-bit model can execute on a high-end 6 GB class device. Claim P’s 1–3B INT4 band is the *envelope between those readings*, not a single device’s headroom.

**Compression bound, not the phone stack.** BitNet b1.58 (Ma et al., 2024) trains ternary weights. At 3B, they report matching a same-size, same-token FP16 LLaMA-style baseline in perplexity, with measured GPU memory 2.22 GB vs 7.89 GB and a 7 nm *arithmetic* energy model of \(71.4\times\) vs FP16 matmul (Horowitz-style coefficients). Those GPU and 7 nm figures are not COTS-phone RSS or battery.[^bitnet-2b4t] Sub-INT4 and MoE-on-phone stay **theater until measured** on the named device and backend.

**COTS execution, thermal, energy.** MELT (Laskaridis et al., 2024) is the systematic COTS study: TinyLlama 1.1B through Llama-2 7B/13B and Gemma 2B/7B, on mid/high Android and iOS, with Monsoon energy traces. Findings we use as bounds:

- Quantization (especially 4-bit) is what makes execution viable; 3-bit is not automatically better (dequantize + matmul cost).
- Inference is largely memory-bound.
- Continuous chat is thermally and energetically unstable: throughput drops, battery discharge is first-class, QoE (load stalls, OOM, reboots) is part of the metric.
- High-end phones with more than 6 GB RAM can run a chat LLM at a *cost*; they do not make 7B free.
- NPU is named as the likely future accelerator, not as a measured default in their farm.

Çöplü et al. (2023) run a quantized 7B (orca_mini_v3, 3-bit GGUF, 2.95 GiB) on iPhones with \(\ge\) 6 GiB RAM and show that sustained performance is gated by iOS thermal states. This is why 7B INT4 is a **ceiling**, not the chatty-generalist target.

### 2.4 Claim P, stated so it can fail

On current COTS phones, a *chatty generalist* that is actually used as a chat model is the dense **1–3B INT4** class (~0.5–1.5 GB weights), with Phi-3-mini 4-bit (~1.8 GB, 3.8B) as the published high-end existence point. **\(\le\)7B INT4** is the published ceiling: it runs on \(\ge\)6 GB high-end devices in the COTS studies, under thermal and energy tax, and is not the default target.

We do not claim that a model from this repo occupies that band. We do not claim a 50M model is a chatty generalist.

---

## 3. Claim R — fewest params under a fixed RSS budget

### 3.1 The question

Fix a process-RSS cap and a duration \(T\). What is the smallest parameter count that still shows *master-of-none multi-skill competence* — several distinct skills above a frozen task bar, none of them expert?

This is not “smallest model that babels English.” TinyStories (Eldan and Li, 2023) already shows that models below 10M can produce fluent, grammatical stories *inside a deliberately tiny synthetic distribution*. That is a domain-restricted existence proof, not multi-skill competence.

### 3.2 What the literature does and does not give

- **Data dominates at small scale when the architecture is ordinary.** Phi-1 / “Textbooks Are All You Need” (Gunasekar et al., 2023) and Phi-3 (Abdin et al., 2024) locate the win in filtered plus synthetic textbook-quality data, not in a new layer type. That is why thesis A is primary.
- **Architecture is not irrelevant under a storage cap.** MobileLLM finds that for sub-billion models, depth, embedding sharing, and grouped-query attention move zero-shot scores at fixed size (Liu et al., 2024; ICML tables **125M / 350M**, +LS). That supports treating architecture as *related*, not as the headline.
- **Sub-billion data prior.** SmolLM (Allal et al., 2024) is the published curated-corpus prior at **135M / 360M** (under 1B; plus a 1.7B sibling). Sit it next to MobileLLM: Liu et al. for the on-device *architecture* envelope; Allal et al. for *data* at sub-billion. Neither is a 50M multi-skill result; neither is a phone process-RSS measurement.
- **Fixed-state inference is a real RSS lever.** RWKV (Peng et al., 2023) gives constant computational and memory complexity at inference, with models scaled to 14B and reported as on par with similar-size Transformers. Linear-time / constant-state families (RWKV; also Mamba, Gu and Dao, 2023) motivate a KV-vs-state axis. They do not, by themselves, produce a 50M multi-skill generalist.
- **MMLU is the wrong primary at this scale.** Hendrycks et al. (2021) introduced MMLU as a 57-task 4-way exam; random is 25%. Their own GPT-3 size sweep shows models up to 13B near chance in the few-shot setting they report, with only the 175B model clearly off the floor. We therefore **reject MMLU as a primary metric under 300M**, and we do not use it at ~50M at all.

### 3.3 H(AI)LP is a candidate, not the thesis

This repo’s H(AI)LP model is an RWKV-style time-mix with a fixed \(h\)-state, FFN sharing, and low-rank projections, aimed at ~50M, with a same-data GPT control. That pair is the **intended instrument** for Claim R’s second axis (RAM vs sequence) and for a same-data quality comparison.

It is not:

- a survival radio
- a 360M Android product
- a substitute for Claim P’s 1–3B band
- a result

No HAILP “survival” language belongs in a parameter table. No Kaggle smoke-train loss belongs in a competence table. Those are engineering notes, not paper claims.

### 3.4 Crossover \(T^*\)

Let \(T^*\) be the sequence (or conversation) length at which the **process RSS** of a KV-cache transformer exceeds that of a same-param fixed-state model, on a named device, under a named dtype.

Literature gives the *shape*: KV grows with length (Pope et al., 2023; standard Transformer inference); RWKV state does not (Peng et al., 2023). We do not report a measured \(T^*\) this season. A later run that claims a \(T^*\) must name the device, the dtype, the batch, and whether RSS is process RSS or a CUDA allocator peek.

---

## 4. Own innovation — Compact Curriculum Recipe (CCR)

This is the original object. It is a **protocol**, not a win.

**Operational home:** [`ccr-spec.md`](ccr-spec.md) (card schema, validation checklist, failure-mode catalog, sequencing, edge cases). Example shapes, not a corpus: [`ccr-samples.md`](ccr-samples.md). This section states the hypothesis and the in/out rules; it does not duplicate the catalog.

The Phi line showed that *textbook-quality* data can move small models (Gunasekar et al., 2023; Abdin et al., 2024). FineWeb-Edu (Penedo et al., 2024) showed that educational filtering of web text is a scalable cousin of that idea. Neither paper publishes an operational three-stratum recipe with in/out rules aimed at *master-of-none multi-skill* under a phone RSS cap. That recipe is ours to state and, later, to test.

Lee et al. (2022) is the closest published *concept-curriculum* efficiency prior (concept-based curriculum masking; comparable BERT/GLUE at ~½ the MLM training cost). We cite it as **related** for CCR positioning. It is not a solved-analogy result and it is not a substitute for CCR: their object is an MLM masking schedule, not a three-stratum data recipe under a phone RSS cap.

**Hypothesis (not a result).** Under a frozen \(\le\)30-item multi-strata eval and a pre-registered token budget, a ~50M model trained on CCR will beat a same-param, same-token Wikipedia/web control on **task_success**, and will not lose to that control on held-out perplexity by more than a pre-registered margin. We do not claim this without a run.

### 4.1 Three strata

Every training unit is tagged with exactly one primary stratum. A unit may *point at* another stratum (a textbook exercise cites a concept id); it may not be a mashup that evades the in/out rules.

#### Stratum C — Concepts

A concept unit teaches *what a thing is*.

**In**

- One atomic concept per unit, named.
- Necessary and/or sufficient conditions stated in prose.
- At least two explicit contrasts (near-miss neighbors).
- One named failure mode or common confusion.
- Short. Definitional core, not an encyclopedia article.

**Out**

- Trivia, celebrity, news-of-the-week, brand lists.
- Slogans or “X is important because…” with no conditions.
- Multi-concept tours that never lock a core.
- Unsourced numerical trivia offered as knowledge.

#### Stratum A — Broad analogies

An analogy unit teaches a *named transferable relation*, not a pretty metaphor.

**In**

- Relation \(R\) named in one sentence.
- Source domain and target domain both specified.
- One worked mapping under \(R\).
- One near-miss that shares surface features but not \(R\).
- One sentence on what transfers and what does not.

**Out**

- Metaphor-as-decoration; unexplained similes.
- Literary flourish with no checkable \(R\).
- Pair-memorization (“doctor is to hospital as teacher is to school”) with no named relation and no near-miss.
- Any implication that analogical reasoning is a solved capability of LLMs. We treat analogy as a *training objective*. Published stress tests show brittleness on counterfactual variants (Lewis and Mitchell, 2024; Mitchell and Lewis, 2024). Webb, Holyoak, and Lu (2023) claimed emergent analogy in GPT-3; that claim is contested. CCR does not take a side by assertion. It refuses to treat the matter as closed.

#### Stratum T — Dense textbooks

A textbook unit teaches a *worked procedure* that reuses locked concepts.

**In**

- Chapter-scale exposition that cites Stratum C concept ids.
- Worked procedure: problem \(\rightarrow\) method \(\rightarrow\) solution \(\rightarrow\) check.
- At least one exercise that cannot be solved by copying the worked example verbatim.
- A check the learner (or a grader) can apply.

**Out**

- Unfiltered Common Crawl, forum dumps, SEO blogs.
- Duplicated boilerplate and license walls of text.
- Code or math with no exposition.
- Textbook-shaped prose that never states a checkable procedure.
- Synthetic text whose only virtue is that an LLM wrote it.

### 4.2 Sequencing, mixture, hygiene

**Sequencing (required).**

1. Stage 1 — C-heavy: lock concepts.
2. Stage 2 — C+T: procedures that use locked concepts.
3. Stage 3 — C+A+T: transfer via named relations.

Do not start with A-only. Analogies without locked concepts are decoration.

**Mixture.** Tokens-per-stratum are logged. Default *hypothesis* mix for a first run (not a claimed optimum): Stage 1 \(\approx\) 50/40/10 C/T/A; Stage 2 \(\approx\) 25/65/10; Stage 3 \(\approx\) 20/50/30. A sweep may change these. The first reported run uses the pre-registered mix.

**Hygiene.**

- Freeze `eval.jsonl` before any train (see [`eval-protocol.md`](eval-protocol.md)).
- No eval surface form, and no near-duplicate, in the train stream.
- Prefer public-domain or clearly licensed sources for T; synthetic C/A must follow the in-rules, not just a prompt template.
- Record license, source id, stratum, and concept ids on every unit.

**What CCR is not.**

- Not “we filtered the web and called it textbooks.”
- Not TinyStories (single restricted domain).
- Not Phi’s private synthetic pipeline (we do not have it).
- Not Lee et al.’s concept-based curriculum *masking* (related efficiency prior; different object).
- Not a claim that quality data removes the need for scale in Claim P.

---

## 5. Phone constraints (operational)

These constraints apply when Claim P is discussed and when a later on-device probe is designed. They do not convert Claim R’s 50M candidate into a phone product.

1. **Name the device class** (for example: “high-end \(\ge\)6 GB, 2022+” vs “mid-tier 4–6 GB”). MELT’s farm is the template (Laskaridis et al., 2024).
2. **Report process RSS**, not file size. File size is a lower bound.
3. **Name \(T\)** and report tokens/s *and* joules or battery \(\Delta\) *and* thermal state at the end of \(T\). A peak tokens/s with no \(T\) is not phone-fit.
4. **Compute \(T^*\)** only as KV RSS vs fixed-state RSS on the same device, same param class, same dtype. One architecture per curve.
5. **NPU line stays empty** until that backend is measured.
6. **Quantization that is real:** INT4 post-training (GPTQ, Frantar et al., 2022; AWQ, Lin et al., 2024; GGUF k-quants as used by MELT) is in-scope. MoE routing and sub-INT4 kernels are out of scope until measured on the named phone stack.

---

## 6. Evaluation protocol (pre-registered)

Full stub: [`eval-protocol.md`](eval-protocol.md). Constraints, locked:

- Freeze \(\le\) 30 items, multi-strata, in `eval.jsonl`, **before** the first reported run.
- Strata match CCR: C, A, T, plus a small chat/instruction slice. No MMLU.
- **Primary metric:** `task_success` vs a same-data ~50M GPT control (Claim R class).
- **Secondary:** held-out perplexity on a named stream that is not the eval items.
- **Second axis:** RAM vs sequence length (process RSS), for the \(T^*\) shape.
- **First reported run counts.** Restarts, aborts, and hyperparameter fishing are disclosed or they do not exist.

Claim P is *not* scored on this 30-item file. Claim P is a literature envelope. A later phone probe, if any, uses a named COTS device and the phone-fit definition in §5, on a Claim P model class.

---

## 7. Train plan and kill criteria

**This season: no train.** The plan exists so a later run has something to falsify.

### 7.1 When a run is authorized

1. Freeze `eval.jsonl` and this protocol (hash both).
2. Build two ~50M models: GPT control, H(AI)LP candidate. Same tokenizer, same token budget, same CCR mix (and a Wikipedia/web control mix).
3. INT4 is a *storage and later-phone* step, not a training dtype unless separately pre-registered.
4. Report the first completed run against the frozen eval. Secondary runs are labeled as such.

### 7.2 Kill criteria (falsifiers)

| Hypothesis | Kill if |
|---|---|
| CCR beats web/Wiki at ~50M | After the pre-registered token budget, CCR `task_success` is not higher than the same-param Wikipedia/web control by the pre-registered margin (default: **strictly more items correct**, no ties resolved after seeing data). |
| CCR does not wreck language modeling | Held-out PPL on the named stream is worse than the control by more than the pre-registered relative margin (default: **+15% PPL**). |
| Architecture is the path (thesis B) | H(AI)LP loses on `task_success` **and** does not show a lower process RSS at the pre-registered sequence lengths vs the GPT control. Then B stays related-and-failed, and A may still stand. |
| Fixed-state wins \(T^*\) | No crossover, or crossover only at lengths the eval never uses, on the named device/dtype. |
| 7B is a usable target (Claim P) | Literature-consistent COTS conditions show 7B INT4 cannot hold the named \(T\) without OOM or thermal collapse on the named high-end class. Then 7B remains a ceiling, which is already the claim; a “7B is the product” reading is killed. |
| Sub-INT4 / MoE helps on phone | Any such claim without a named-device measurement is out of court (already theater). |

A kill is a scientific outcome. It is not a reason to quietly edit the eval.

---

## 8. What we do not claim

We do not claim, without a run:

- that CCR beats Wikipedia, FineWeb, FineWeb-Edu, TinyStories, or Phi’s pipeline
- that a ~50M model is a phone chatty generalist
- that H(AI)LP is more competent than GPT at the same data
- that H(AI)LP is a survival product, a 360M Android build, or a FAISS-augmented appliance
- that analogical reasoning is solved, in us or in the literature
- that MMLU under 300M (or at 50M) is a meaningful primary score
- that MoE or sub-INT4 is a phone win
- that NPU inference is faster here
- that Kaggle or laptop smoke-train curves are competence evidence
- that any number in [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) or the root README’s old hardware tables is a paper result
- that Claim P and Claim R have been jointly demonstrated

We **do** claim, as literature:

- the Phi-3-mini 4-bit on-device existence point (Abdin et al., 2024)
- the MobileLLM on-device size and DRAM-heuristic envelope (Liu et al., 2024), with ICML tables at **125M / 350M**
- SmolLM’s **135M / 360M** curated-corpus prior at sub-billion (Allal et al., 2024)
- Lee et al. (2022) as a *related* concept-curriculum efficiency prior — not as CCR, and not as analogy-solved
- the BitNet 3B compression/quality bound on *GPU / energy models*, not phones (Ma et al., 2024)
- the MELT / iPhone-7B COTS thermal-energy picture (Laskaridis et al., 2024; Çöplü et al., 2023)
- that INT4 post-training quantization is a real, widely used method (Frantar et al., 2022; Lin et al., 2024)
- that RWKV-style fixed state is a published alternative to a growing KV cache (Peng et al., 2023)

---

## 9. Season output

This repo, this season, is a paper repo. The code is a later instrument: a ~50M GPT vs H(AI)LP scaffold and optional trainers. The paper owns both claims, separately. The innovation is CCR as hypothesis and protocol.

When a run happens, it edits [`eval-protocol.md`](eval-protocol.md) only to fill the first-run table, and it does not rewrite this abstract to absorb Claim P into Claim R.

[^bitnet-2b4t]: Wang et al. (2025), *BitNet b1.58 2B4T* (arXiv:2504.12285), report **~0.4 GB** non-embedding memory as a tighter *compression* bound next to Ma et al. (2024). Still not COTS-phone process RSS.
