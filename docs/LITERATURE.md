# Must-cites (condensed)

Companion to [`PAPER.md`](PAPER.md). Every figure here is from a paper, a standard arithmetic identity, or marked **TODO-cite**. Do not copy repo smoke-train or projected Android tables into this file.

**How to read “Botiful / Bolbol.”** Those reviews locked: thesis A = data; quant + data recipe are real; MoE / sub-INT4 are theater until measured; reject analogical-as-solved and MMLU-under-300M. The numbers below are from the papers, not from those reviews.

---

## Claim P — phone band

| Source | What we take | Figure | Status |
|---|---|---|---|
| Abdin et al., 2024. *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone.* arXiv:2404.14219 | Phone-local chatty 3.8B | Phi-3-mini **3.8B**, **3.3T** tokens; 4-bit occupies **\(\approx\) 1.8 GB**; iPhone 14 A16, offline, **>12 tok/s**. Also reports MMLU 69% / MT-bench 8.38 — **do not reuse as our score**. Phi-3-small **7B** and medium **14B** exist; they are scale-ups, not the phone existence proof. | cited |
| Liu et al., 2024. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* ICML / arXiv:2402.14905 | On-device size envelope | ICML/arXiv tables: **125M / 350M** (and layer-share / +LS). Do **not** list **600M / 1B / 1.5B** as ICML result rows — larger Hugging Face MobileLLM variants are separate / **TODO-cite if used**. DRAM on cited phones **6–12 GB**. Authors’ heuristic: app **\(\lesssim\) 10% of DRAM**. LLaMA-2 **7B 8-bit** called prohibitively expensive in main memory. Energy estimate **0.1 J/token per billion params** (they cite hardware energy models). 125M/350M trained on **1T** tokens for the reported tables. Deep-and-thin + emb-share + GQA + optional layer-share. | cited (125M/350M); larger HF variants TODO-cite if used |
| Allal et al., 2024. *SmolLM — blazingly fast and remarkably powerful.* Hugging Face blog | Claim R / sub-B data prior | Curated SmolLM-Corpus. **135M / 360M** under 1B (each **600B** tokens) plus a 1.7B sibling. Next to MobileLLM: Liu et al. for the on-device *architecture* envelope; Allal et al. for *data* at sub-billion. Not a phone-RSS measurement; not a 50M result. | cited |
| Ma et al., 2024. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764 | Compression *bound* | BitNet b1.58 **3B** matches same-size same-token FP16 LLaMA-style **PPL** (Wiki/C4 setup in paper). GPU memory **2.22 GB vs 7.89 GB** (FasterTransformer, 2-bit kernel). **71.4\(\times\)** arithmetic energy on a **7 nm** model (Horowitz-style), not a phone battery. | cited; **not** a COTS-phone stack |
| Wang et al., 2025. *BitNet b1.58 2B4T Technical Report.* arXiv:2504.12285 | Tighter compression *bound* (optional) | Native 1.58-bit **2B** on **4T** tokens. Paper table: **~0.4 GB** non-embedding memory. Sit next to Ma et al. (2024). Still **not** COTS-phone process RSS. | cited as compression bound only |
| Laskaridis et al., 2024. *MELTing point: Mobile Evaluation of Language Transformers.* arXiv:2403.12844 | COTS study | Zoo includes TinyLlama **1.1B**, Zephyr **3B**, Gemma **2B/7B**, Mistral **7B**, Llama-2 **7B/13B**. Mid/high Android + iOS (S23 8 GB, Pixel 6a 8 GB, iPhone 14 Pro 6 GB, iPhone SE 4 GB). Monsoon energy. **4-bit makes execution viable**; 3-bit not automatically better. Memory-bound decode. Continuous chat **thermally/energetically unstable**. **>6 GB** high-end can run chat at QoE/battery cost. **NPU = future bet**, not their measured default. llama.cpp Android often **CPU**. | cited |
| Çöplü, Loedi, Bendiken, Makohin, Bouw, Cobb, 2023. *A Performance Evaluation of a Quantized Large Language Model on Various Smartphones.* arXiv:2312.12472 | 7B on iPhones | orca_mini_v3 **7B**, GGUF **3-bit K-S**, **2.95 GiB**, iPhones with **\(\ge\) 6 GiB** RAM. Sustained speed gated by iOS thermal states; 90 s inter-prompt delay stayed “fair,” 5 s delay did not. | cited |
| Frantar et al., 2022. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323 | INT4 is real | One-shot weight quantization to 3–4 bit for GPT-scale models; the method MELT lists as a supported path. | cited |
| Lin et al., 2024. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* MLSys / arXiv:2306.00978 | INT4 is real | 4-bit on-device compression/acceleration; paired with GPTQ as the “quant is real” line. | cited |
| Weight arithmetic | INT4 \(\approx\) 0.5 byte/param | **1B \(\approx\) 0.5 GB**, **3B \(\approx\) 1.5 GB**, **7B \(\approx\) 3.5 GB** *weights only*. Not process RSS. | identity, not a lab number |
| Llama 3.2 1B/3B on-device (Meta, 2024) | Extra 1–3B dense points | Meta’s product blog/model card positions 1B/3B for on-device. Exact RSS / tok/s / energy on a named phone: **TODO-cite** a paper or a measured card, do not invent. | TODO-cite |
| Gemma 2 2B (Google, 2024) | Extra 2B dense point | Gemma 2B appears in MELT’s zoo as Gemma-2b-it. Gemma 2 report numbers for phone RSS: **TODO-cite** if used beyond MELT. | partial (MELT lists 2B) |

**Claim P synthesis (literature only).** Chatty generalist \(\approx\) **1–3B dense INT4** (~**0.5–1.5 GB weights**), with Phi-3-mini 4-bit (**\(\approx\) 1.8 GB**, 3.8B) as the published high-end existence point. **\(\le\) 7B INT4** is the COTS ceiling (MELT + iPhone 7B study), not the target. Process RSS + sustained \(T\) are the real phone-fit tests; file size is not.

---

## Claim R — small models, data, state

| Source | What we take | Figure | Status |
|---|---|---|---|
| Gunasekar et al., 2023. *Textbooks Are All You Need.* arXiv:2306.11644 | Data recipe lineage | Phi-1 **1.3B**; filtered web code **6B** tokens + synthetic textbooks **<1B** + CodeExercises **180M**. HumanEval pass@1 **50.6%**, MBPP **55.5%**. Phi-1-small **350M**, HumanEval **45%**. Innovation located in data, not architecture. | cited |
| Eldan and Li, 2023. *TinyStories.* arXiv:2305.07759 | Small \(\neq\) multi-skill | Models **<10M** (or 1-block) can write fluent stories in a **restricted** synthetic child-vocabulary domain. Not a generalist; not a phone claim. | cited |
| Peng et al., 2023. *RWKV: Reinventing RNNs for the Transformer Era.* Findings of EMNLP. arXiv:2305.13048 | Fixed-state candidate | Linear-time / constant-memory inference; models to **14B**; reported on par with similar-size Transformers. Motivates H(AI)LP as a *candidate*, not a result. | cited |
| Gu and Dao, 2023. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752 | Related state-space line | Linear-time selective SSM; another fixed-state family. Do not collapse Mamba into H(AI)LP. | cited |
| Hoffmann et al., 2022. *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556 | Tokens/params | Compute-optimal training uses far more tokens per parameter than the Kaplan et al. (2020) reading many small projects still inherit. A 50M run that under-trains is not a fair kill of CCR. Exact Chinchilla multiplier applied to 50M: **TODO-cite** the table row you use; do not invent a token budget. | cited (law); token budget TODO-cite |
| Penedo et al., 2024. *The FineWeb Datasets* (incl. FineWeb-Edu). arXiv:2406.17557 | Educational web filter | Public cousin of “textbook quality” via educational-value filtering. CCR is stricter and operational; FineWeb-Edu is a control candidate, not CCR. | cited |
| Lee et al., 2022. *Efficient Pre-training of Masked Language Model via Concept-based Curriculum Masking.* EMNLP / arXiv:2212.07617 | Related CCR positioning | Closest published *concept-curriculum* efficiency prior: comparable BERT/GLUE at **~½** the MLM training cost. **Related**, not a solved-analogy claim, and **not a substitute for CCR** (MLM masking schedule vs our three-stratum data recipe). | cited as related |
| Pope et al., 2023. *Efficiently Scaling Transformer Inference.* MLSys / arXiv:2211.05102 | KV grows | KV cache is a first-class inference cost that scales with batch and sequence. Shape source for \(T^*\), not a phone RSS number. | cited |
| This repo’s ~50M GPT vs H(AI)LP | Instrument | Config targets **~50M**. Not a published result. Do not cite PROJECT_SUMMARY tables as literature. | instrument, not a cite |

---

## Rejected as primary / closed

| Source | Why it is here | Rule |
|---|---|---|
| Hendrycks et al., 2021. *Measuring Massive Multitask Language Understanding.* ICLR. arXiv:2009.03300 | MMLU | 57 tasks, 4-way, random **25%**. Their GPT-3 sweep: models up to **13B** near chance few-shot; **175B** at **43.9%**. **Reject MMLU as a primary under 300M**; do not run it at ~50M as a headline. |
| Webb, Holyoak, Lu, 2023. *Emergent analogical reasoning in large language models.* *Nature Human Behaviour.* | Analogy-as-solved claim | They report GPT-3 matching or beating humans on several analogy batteries and call the ability emergent. |
| Lewis and Mitchell, 2024. *Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in Large Language Models.* arXiv:2402.08955; and Mitchell and Lewis, 2024, *Evaluating the Robustness of Analogical Reasoning in Large Language Models.* arXiv:2411.14215 | Analogy is not closed | Humans stay high on counterfactual variants; GPT models drop. **Reject analogical-as-solved.** CCR trains named relations; it does not assert human-like analogy. |
| BitNet / MoE-on-phone blogs | Theater | Memory and 7 nm arithmetic energy are not process RSS on a named phone. **MoE and sub-INT4 stay theater until measured** (Bolbol lock). |

---

## Quantization and runtime (real vs theater)

**Real (methods exist and are used on device in MELT / llama.cpp / Phi-3):**

- INT4 / 4-bit weight quantization: GPTQ, AWQ, GGUF k-quants.
- llama.cpp / MLC-LLM as COTS runtimes (Laskaridis et al., 2024).
- Phi-3-mini 4-bit on iPhone 14 (Abdin et al., 2024).

**Theater until measured on the named phone + backend:**

- BitNet b1.58 as a *phone* stack (GPU table and 7 nm model only).
- MoE “7B quality at 2B active” on a mid-tier phone.
- Sub-INT4 (2-bit / 1.58-bit) tokens/s, RSS, and thermal under a named \(T\).
- NPU tok/s copied from a GPU or CPU run.

---

## Do not promote from this repo

The following appear in older README / `PROJECT_SUMMARY.md` / Kaggle notes. They are **not** literature and **not** paper results:

- 1,000-step Kaggle dual-T4 train loss / val PPL.
- Projected 360M INT4 Android footprints and “survival tool” RAM tables.
- Scaled-from-18.5M INT4 size projections.
- Demo CPU tokens/s as architecture proof.

If a later run produces a number, it enters [`eval-protocol.md`](eval-protocol.md) with a date, a hash, and “first reported run” or “secondary.”
