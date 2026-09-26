# The HOPE Marathon — A Documented Small-Scale Training Experiment

**Date:** 2026-09-25 18:10 → 2026-09-26 09:44 EEST (~15.5 h wall clock, unattended)
**Hardware:** one Apple-silicon Mac (MPS, 36 GB RAM), fp32
**Result:** a fully documented learning curve from random init (loss 10.8) to a fluent,
genre-accurate 86M-parameter language model with measurable factual emergence —
and an honest negative result on factual recall at this data budget.

> **TL;DR:** After making training **5.2× faster** (exact chunk-parallel delta-rule
> scan), we ran a 48k-step Wikipedia foundation (~786M tokens) + 6k-step Q&A
> fine-tune, probing the model hourly with a 20-question factual cloze suite.
> Perplexity: **371 → 65**. Factual top-5: **0% → 20%**. MRR: **0.008 → 0.165**.
> The model learned English, Wikipedia's genre, and a handful of facts at rank #1 —
> but fell short of our pre-registered "wow" bar (top-5 ≥ 60%, ppl ≤ 30), which
> scaling laws suggest would need 10–50× more tokens. Every decision, trigger,
> and measurement is recorded in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); the raw
> per-probe data is in [`probe_stats.jsonl`](probe_stats.jsonl).

---

## 1. Making training 5× faster (the prerequisite)

Before this experiment, the per-token Python loop in `SelfModifyingLayer` consumed
**85% of training compute** (552 ms fwd+bwd per microbatch vs 99 ms for everything
else). The recurrence

```
M_t = α_t·M_{t-1} + β_t·k_tᵀ(v_t − k_t·M_{t-1}),   o_t = q_t·M_{t-1}
```

is the **gated delta rule** (as in DeltaNet / Gated DeltaNet) and admits an *exact*
chunk-parallel form (`_chunked_delta_scan` in `train_hope.py`): within a chunk, the
per-token write vectors solve a unit lower-triangular system built from pairwise
decay ratios `exp(G_{t-1} − G_s) ≤ 1` — no division by cumulative products, so no
underflow blowups. Sequential cost drops from T steps to T/64 batched-matmul steps.

| Measurement (B=4, T=512, D=512, MPS, fp32) | Before | After |
|---|---|---|
| Fast-memory layer fwd+bwd | 513 ms | **23 ms (22×)** |
| Full model microbatch | 575 ms | **119 ms (4.8×)** |
| Training throughput | ~3.2k tok/s | **~16.4k tok/s (5.2×)** |
| Validation (every 60 s) | minutes (re-streamed 100k samples) | **2.5 s** (50-batch startup cache) |

Mathematical equivalence is not taken on faith: `test_chunked.py` verifies
chunked-vs-loop in float64 across 420 combinations of length, chunk size, mask and
extreme gates (worst output diff **1.9e-13**), plus gradient equivalence. The
token-by-token loop is retained bit-identically for single-token generation.
Also in the speed pass: `GPT2TokenizerFast` (~3× tokenize), a background prefetch
thread, fused `_foreach_` tier-gradient updates, deferred MPS syncs.
Commits: `812db78`, `69db99a`, `a6d5662`, `1fe2a33`.

## 2. Experiment design

- **Budget was adaptive by pre-registered rule.** The run started at the repo
  default (12k foundation + 3k fine-tune). Decision rule D3 (registered *before*
  seeing 12k results): extend if val was still improving, factual top-5 < 30%, or
  wow criteria missed on descending curves. The maintainer then chose the **48k
  marathon** option explicitly (D4).
- **Extension mechanics (D4):** resume from the 12k checkpoint with scheduler
  `base_lrs` rewritten 2e-4 → 1e-4 so the cosine re-warm against the 48k horizon
  peaks at ~8.8e-5 instead of ~1.8e-4 (no destabilization observed — monitored
  steps 12k–13k). Data: stream advanced to *fresh* articles (skip 65k, coverage to
  ~213k articles ≈ single pass); val split moved 100k → 600k to stay clear of the
  expanded coverage (old val series bridged and documented; old best snapshotted).
- **Measurement:** 20-min vitals checks; hourly probes (`probe_model.py`) loading a
  *copy* of the latest checkpoint on CPU — zero interference with training (measured
  once: unniced probes cost ~15 min of training per hour; fixed with `nice -n 15`).
- **Probe suite:** 20 factual cloze probes (rank + probability of the correct first
  token), 5 Q&A generations, 3 encyclopedic generations, all seeded.

## 3. The learning curve

### Perplexity & factual knowledge (full table in `probe_stats.jsonl`)

| Step | Val ppl (wiki) | Factual top-5 | MRR | Milestone |
|---|---|---|---|---|
| 2,000 | 371 | 0% | 0.008 | uniform-ish guessing → token frequencies |
| 5,000 | 163 | 5% | 0.021 | **first top-5 fact**: "tallest mountain in the *world*" (#5) |
| 8,000 | 113 | 5% | 0.062 | wiki scaffolding in generations (References, External links) |
| 12,000 | 111 | 5% | 0.071 | baseline end (annealed); "Sun" #9 |
| 18,000 | 87 | 15% | 0.109 | "Earth orbits the *Sun*" enters top-5 (#3) |
| 28,000 | 74 | 15% | 0.113 | mid-cosine rank oscillation damps out |
| 48,000 | **65** | 15% | 0.114 | foundation final (plateau at LR floor) |
| +6,000 QA | 33 (QA domain) | **20%** | **0.165** | **3 facts at rank #1**: world (p=52%), *energy* for photosynthesis (p=69%), *O* for H₂O (p=24%) |

### Emergence of a single fact ("The Earth orbits around the ___" → Sun)

rank #163 (5k) → #46 (8k) → #9 (11.7k) → #24 (14.8k) → #15 (21k) → #3–4 (25k) → **#2 (p=10.8%) (48k+)**

### Findings worth recording

1. **Syntax before semantics, reliably.** Grammar, register and genre were
   fluent by ~5k steps while every factual probe still failed — the classic
   small-LM ordering, captured per-hour.
2. **Rank oscillation is real and damps with LR.** Probes 3k steps apart disagreed
   wildly on mid-rank items (east #9→#40) at LR 6–7e-5; MRR became monotone only
   below ~2e-5. Single-probe snapshots are noisy; judge trends over ≥5k steps.
3. **Fine-tuning metrics can diverge.** In the QA overfit regime, QA val *rose*
   (3.63→4.17) while factual cloze *improved* (MRR 0.096→0.177) — the science-
   flavoured QA set reinforced fact circuits while memorizing its own answers.
   "Best checkpoint" is metric-dependent; we report both.
4. **Late annealing rescued the fine-tune.** Instead of early-stopping at the
   overfit onset (~3k), we let the cosine finish: QA val recovered to a **new best
   3.51** at step ~5.3k. The pre-registered early-stop guard would have left value
   on the table.
5. **Domain shift is visible but bounded**: wiki ppl rose 65 → 72 after 6k QA
   steps — the tiered CMS schedule (slow layers update every 16 steps on averaged
   gradients) is the structural mitigation; quantifying it remains open
   (AGENTS.md improvement item #10).

## 4. The pre-registered "wow" verdict — 1/4, a negative result

| # | Criterion (fixed before results) | Final measurement | Verdict |
|---|---|---|---|
| 1 | Phase 2 complete, Q&A format learned | 100% of probes emit Answer/Reasoning structure | **PASS** |
| 2 | Factual cloze top-5 ≥ 60% | 20% (4/20), MRR 0.165 | **FAIL** |
| 3 | Q&A samples actually answer their questions | format yes, grounding no ("capital of France" → "It's a social media when you're older") | **FAIL** |
| 4 | Val perplexity ≤ 30 | 65.2 wiki / 33.3 QA | **FAIL** |

**Interpretation.** ~786M tokens through an 86M-parameter model buys fluency,
genre and a handful of hardened facts — consistent with scaling laws; criteria 2
and 4 would plausibly require 10–50× more tokens (weeks on this hardware). We
publish this as a *negative result on factual recall at consumer budget* together
with the positive results (5× speed methodology, full emergence documentation).
The model does what the data budget says it should: it sounds like Wikipedia and
knows almost nothing.

Sample (final fine-tuned model, seeded):
```
Question: Why is the sky blue?
Answer:  Sunlight gets scattered by air molecules... When it enters Earth's
         atmosphere, it turns into light and light...   [partially correct mechanism,
                                                         weakly question-conditioned]
Question: What is the capital of France?
Answer:  It's a social media when you're older and can learn about what they can do.
```

### Addendum (2026-09-26, post-verdict): greedy decoding reveals a correct answer

The verdict table above was measured with the probe harness's *sampled* decoding
(temp 0.7, top-k 40). During the inference-optimization pass we re-ran the Q&A
probes **greedily** (argmax) on `hope_final_best.pth`:

```
Question: Why is the sky blue?
Answer:  Sunlight gets scattered by air molecules, making blue light more visible.
Reasoning: Blue light has shorter wavelengths that coll[ide with gas molecules…]
```

— a fully correct Rayleigh-scattering answer with correct reasoning, and the
project's canonical demo question. Two honest caveats, both measured:

1. **It is memorization of the fine-tuning set, not generalization.** Other
   questions ("What happens when water freezes?", "What do plants need to grow?")
   still produce dataset-flavored non-answers under greedy decoding. The sky-blue
   Q&A is (near-)verbatim in the micro dataset.
2. **It is prompt-format fragile.** Removing/adding a single trailing space after
   "Answer:" changes the first-token distribution (top-1 probability is only ~9%
   in the flat regime), and sampled decoding at temp 0.7 usually misses the
   correct continuation entirely.

Criterion 3 therefore stays **FAIL** ("reliably answer"), but the accurate
characterization is: *the model can retrieve a correct, well-reasoned answer for
at least one canonical question under greedy decoding; retrieval is format-
sensitive and does not generalize across questions.* Practical tip: use low
temperature (≤0.3) or greedy with this checkpoint.

### Addendum: inference speedups (measured, 86M model on Apple MPS)

| Path | Before | After | Gain |
|---|---|---|---|
| Prefill T=512 | 12.8 ms | 5.1 ms | **2.5×** (`last_only`: CMS+head on final position — valid because CMS blocks are position-wise) |
| Prefill T=12 | 4.0 ms | 2.4 ms | 1.6× |
| Generation | 387 tok/s | ~500 tok/s | **1.3×** (bf16 weights + `torch.compile`d T=1 steps) |

Quality gates passed before shipping defaults: bf16-vs-fp32 **100% greedy token
agreement** on 5 prompts and **identical** factual-probe metrics (top-5 20%, MRR
0.165). The delta-rule scan and memory state stay fp32 under bf16 weights — MPS
`solve_triangular` is fp32-only (bf16 trips an uncatchable Metal assert), and
fp32 state accumulates more accurately. Sampling upgraded to top-k + nucleus +
repetition penalty (fixes the repetition degeneration seen in §3 finding 5's era:
*"the character of the character…"*).

## 5. Artifacts & reproduction

**Model checkpoints** (GitHub Release `v0.1-marathon`, ~986 MB each):
- `hope_final_best.pth` — Q&A fine-tuned, best QA val 3.51 @ step 5,273 (**use this for chat**)
- `hope_foundation_best.pth` — 48k-step foundation, best wiki val 4.178 (new split)

```bash
gh release download v0.1-marathon --repo obekt/HOPE-nested-learning --pattern "*.pth"
python3 chat.py   # loads CONFIG['save_path']; point it at the downloaded file
```

**Code & data (this repo):**
- `probe_model.py` — the probe harness (`--report` prints the accumulated table)
- `probe_stats.jsonl` — all 34 probe records (ranks, probabilities, samples)
- `EXPERIMENT_LOG.md` — the decision log (D1–D5) with every trigger and number
- `extend_training.py` / `extend_phase2.py` — the extension drivers (resume + re-seed mechanics)

**Reproduce the run:** `python3 run_pipeline.py` (12k+3k default, ~4.5 h at
16.4k tok/s), then optionally `python3 extend_training.py` + `python3
extend_phase2.py` for the 48k+6k marathon (~12 h). Tests: `python3 test_chunked.py
&& python3 test_nested.py && python3 test_fixes.py`.
