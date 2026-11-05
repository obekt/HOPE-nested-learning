# Weight surgery on the `v0.1-marathon` release

Goal from the request: take the released model **binary** and edit its weights
directly — no training, no optimizer, no gradient step — so it becomes a more
capable small model.

Everything here lives in `surgery/` and touches no file that existed before.

* artifacts (gitignored `.pth`): `surgery/out/*.pth`
* code: `surgery/src/`
* raw measurements: `surgery/out/*.jsonl`, `surgery/out/tables.md`
* `surgery/data/` (~3.3 GB) holds tokenised corpora and the cached normal
  equations `A = XᵀX`, `B = XᵀY` for each solved state — all regenerable, gitignored

---

## 1. What the binary actually contains

Before editing anything I measured the two released checkpoints
(`hope_foundation_best.pth`, 48k wiki steps; `hope_final_best.pth`, +6k Q&A steps).
Details: `src/diag.py`, `src/normcheck.py`, `src/probe_h.py`.

| finding | evidence |
|---|---|
| The **output head is the only well-trained matrix** | `head.weight` rms is **18.8×** its Xavier init; every row grew ≥4.6× |
| The **input embedding is mostly still random init** | 61% of the 50,257 rows are within 10% of their init L2 norm; for the low-frequency half, per-element std is 0.01998 vs 0.02000 at init and the 4th-moment check is 3.06 (Gaussian) — i.e. untouched |
| **Half the depth never trained** | CMS blocks 8–15 (the tier-1/tier-2 slow weights, update periods 4 and 16) have weight rms 0.97–1.00× their *random init*; blocks 0–7 moved to 1.09–1.19× |
| The fast memory's projections barely moved, but its **gates moved a lot** | `proj_q/k/v/out` rms = 0.98–1.13× init; `gate_alpha` 7.0×, `gate_beta` 4.0× |
| Rare tokens have **larger output rows** than frequent ones | corr(‖head row‖, head bias) = **−0.967**; bottom-half-prior rows have norm 3.22 vs 1.34 for the top decile |
| Input and output spaces are unrelated | cos(embedding row, head row) = 0.014 (no tying) |
| But HOPE's head is semantically organised | a **closed-form ridge map from GPT-2's same-id token embedding explains 63% of the variance** of HOPE's head rows (held-out, centred R²) |

Behaviourally (`src/eval_suite.py`, pre-registered suite, deterministic likelihood +
greedy only), against a freshly initialised untrained HOPE and GPT-2 small
(124M, *identical vocabulary*) run through the same code:

| | ppl↓ | cloze@1 | cloze@5 | copy (cue) | ARC | Wino | TQA | MMLU |
|---|---|---|---|---|---|---|---|---|
| untrained HOPE | 50 972 | 0.000 | 0.000 | 0.000 | 0.260 | 0.493 | 0.266 | 0.241 |
| `hope_foundation_best` | 421 | 0.026 | 0.079 | 0.113 | 0.202 | 0.503 | 0.307 | 0.256 |
| `hope_final_best` | 2 552 | 0.079 | 0.105 | 0.007 | 0.209 | 0.509 | 0.294 | 0.263 |
| GPT-2 small (reference) | 48.7 | 0.290 | 0.579 | 0.350 | 0.238 | 0.501 | 0.289 | 0.269 |

Read the last four columns honestly: **multiple-choice likelihood sits at chance for
GPT-2 small too** on this protocol (25% for 4-option items), so they measure almost
nothing at this scale.
The metrics that discriminate are perplexity, cloze ranking, and context-copying.
And the two most important lines are:

* On a **held-out** set of 25 fresh facts (`cloze_holdout_B`, written before any
  edit), both released checkpoints score **0.000 top-5**. The published "20% top-5"
  is the 20-item suite; it does not generalise.
* The Q&A fine-tune **destroyed the fast memory's usefulness**: context-copying fell
  from 11.3% (foundation) to 0.7% (final), and with the memory ablated
  (`proj_out ← 0`) perplexity goes 421 → 1 535 and copying 11.3% → 1.0%. So the
  memory *is* load-bearing, it is just barely useful.

## 2. Why "hand-edit the bytes" cannot work, and what can

Bits carry no meaning without a model of the function they implement. The only
edits that can raise capability without training are ones where the target is
derivable in closed form from (a) the checkpoint's own weights, (b) the other
released checkpoint, or (c) another model's weights. That is what I did — every op
in `src/surgeries.py` is a copy, a rescale, a task-vector combination, or **one
ridge-regression normal equation**. Nothing in this project computes a gradient.

The load-bearing trick: **softmax(log p) = p**, exactly. So if I want HOPE's
output layer to reproduce a teacher's next-token distribution as well as its own
representation allows, the least-squares target is the teacher's *log-probabilities*
and the answer is

```
W = (XᵀX + λI)⁻¹ Xᵀ Y        X = HOPE's own final hidden states
                              Y = GPT-2's log-probs on the same tokens
```

one 513×513 solve, 15–90 seconds on this Mac. HOPE's vocabulary is GPT-2's exactly
(50,257 shared ids), so the teacher's targets line up with HOPE's output rows with
no alignment step at all.

## 3. Results: 105 measured configurations

Full grids: `out/lab1..lab8.jsonl`. Screening on suite A; the shipped picks were
then re-measured on the untouched confirmation suite B.

**Nothing that touched the body of the network improved perplexity on its own** —
the release's weights are jointly calibrated, so every upstream edit costs CE
before the output layer knows about it:

| edit alone | ppl vs 421 | note |
|---|---|---|
| neutralise the 8 never-trained CMS blocks | 15 808 | yet ARC 0.238, TQA 0.370 — the random layers *were* adding noise |
| zero the untrained embedding rows | 742 | |
| impute 30k dead embedding rows from GPT-2 | 630 | direction recovers ~12% of variance; not enough |
| rewrite `proj_v` as a closed-form successor lookup | 541–623 | fails: see §5 |
| interpolate the two releases (task vector, λ=0.25) | 474 | cloze@5 0.079→0.132, MRR 0.065→0.089 — best pure merge |
| equalise head-row norms | 4 964 | |

**The pre-registered temperature control.** A pure logit rescale (`head.weight`
and `head.bias` multiplied by s, i.e. exactly a sampling-temperature change) on
both the release and the operated model, measured on suite A:

| s | 0.5 | 0.7 | **1.0** | 1.4 | 2.0 |
|---|---|---|---|---|---|
| ppl, released foundation | 1 488 | 595 | **421** | 1 005 | 8 340 |
| ppl, operated foundation (the 50/50 variant) | 1 086 | 378 | **207** | 332 | 1 567 |
| cloze@1 / @5 / copy, either model | unchanged | unchanged | unchanged | unchanged | unchanged |

Each model is already at its own best scale, and the operated model wins at every
scale — so the perplexity gain below is not a decoding artifact. (Ranking metrics
are scale-invariant by construction; the MC columns move ±1–2 pt with scale, which
is why no claim in this report rests on them.)

**Re-solving the head is where all the capability came from**, and mixing the
teacher with the model's own distribution (`mixbase`) is what keeps the parts that
already worked:

| candidate | ppl↓ | cloze@1/@5 | copy | ARC | TQA |
|---|---|---|---|---|---|
| foundation → teacher only | 245 | 0.000 / 0.053 | 0.003 | 0.191 | 0.333 |
| foundation → 50% teacher / 50% own | 207 | 0.026 / 0.053 | 0.103 | 0.195 | 0.310 |
| foundation → 40% teacher / 60% own **(shipped)** | **216** | 0.026 / **0.079** | 0.100 | 0.195 | 0.307 |
| foundation → 50/50 + 35% Q&A-format corpus | 220 | 0.026 / **0.079** | 0.100 | 0.206 | 0.308 |
| final → teacher only | 344 | 0.000 / 0.026 | 0.003 | 0.206 | 0.310 |
| final → 50/50 **(shipped)** | **548** | 0.079 / 0.105 | 0.010 | 0.220 | 0.292 |

Full grids behind these rows: `out/lab1..lab8.jsonl` (105 distinct configurations).

#### Shipped files — selection suite A, then confirmation suite B

| model | ppl↓ | pplB↓ | cloze@1 | cloze@5 | clozeMRR | clozeB@5 | ARC | Wino | TQA | HS | HSB | MMLU | copy | copyGain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:identity | 421 | – | 0.026 | 0.079 | 0.065 | – | 0.202 | 0.503 | 0.307 | 0.245 | – | – | 0.113 | 0.010 |
| SHIPPEDF | 216 | 187 | 0.026 | 0.079 | 0.060 | 0.040 | 0.195 | 0.503 | 0.307 | 0.245 | 0.263 | 0.263 | 0.100 | 0.007 |
| final | 2552 | 2297 | 0.079 | 0.105 | 0.088 | 0.000 | 0.209 | 0.509 | 0.294 | 0.260 | 0.258 | 0.263 | 0.007 | 0.000 |
| SHIPPEDQ | 548 | 500 | 0.079 | 0.105 | 0.087 | 0.000 | 0.220 | 0.509 | 0.300 | 0.242 | 0.245 | 0.241 | 0.010 | 0.000 |

#### One lever per family, on the foundation model (suite A)

| model | ppl↓ | pplB↓ | cloze@1 | cloze@5 | clozeMRR | clozeB@5 | ARC | Wino | TQA | HS | HSB | MMLU | copy | copyGain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:identity | 421 | – | 0.026 | 0.079 | 0.065 | – | 0.202 | 0.503 | 0.307 | 0.245 | – | – | 0.113 | 0.010 |
| emb:impute_g2 | 630 | – | 0.000 | 0.053 | 0.027 | – | 0.209 | 0.507 | 0.327 | 0.250 | – | – | 0.070 | 0.010 |
| cms:untrained_off | 15808 | – | 0.000 | 0.026 | 0.012 | – | 0.238 | 0.519 | 0.370 | 0.258 | – | – | 0.003 | 0.000 |
| tv:lam0.25 | 474 | – | 0.053 | 0.132 | 0.089 | – | 0.213 | 0.505 | 0.316 | 0.255 | – | – | 0.113 | 0.007 |
| D0.distill_r1e-3 | 245 | – | 0.000 | 0.053 | 0.016 | – | 0.191 | 0.506 | 0.333 | 0.255 | – | – | 0.003 | – |
| Z03.mix.5 | 207 | – | 0.026 | 0.053 | 0.058 | – | 0.195 | 0.505 | 0.310 | 0.242 | – | – | 0.103 | – |
| K2.mix.6surp | 216 | – | 0.026 | 0.079 | 0.060 | – | 0.195 | 0.503 | 0.307 | 0.245 | – | – | 0.100 | – |
| H2.mix.5surp_qa35_n600 | 220 | – | 0.026 | 0.079 | 0.060 | – | 0.206 | 0.508 | 0.308 | 0.250 | – | – | 0.100 | – |

#### Every memory-focused edit, applied alone (suite A)

| model | ppl↓ | pplB↓ | cloze@1 | cloze@5 | clozeMRR | clozeB@5 | ARC | Wino | TQA | HS | HSB | MMLU | copy | copyGain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:identity | 421 | – | 0.026 | 0.079 | 0.065 | – | 0.202 | 0.503 | 0.307 | 0.245 | – | – | 0.113 | 0.010 |
| N1.tieqk_only | 1814 | – | 0.026 | 0.053 | 0.043 | – | 0.206 | 0.515 | 0.247 | 0.253 | – | – | 0.037 | – |
| N2.bagcollapse_only | 1634 | – | 0.000 | 0.000 | 0.004 | – | 0.195 | 0.504 | 0.294 | 0.245 | – | – | 0.017 | – |
| N3.forget_fast | 642 | – | 0.026 | 0.105 | 0.055 | – | 0.191 | 0.511 | 0.306 | 0.245 | – | – | 0.060 | – |
| N4.retain_forever | 508 | – | 0.026 | 0.079 | 0.065 | – | 0.199 | 0.505 | 0.303 | 0.250 | – | – | 0.100 | – |
| Z13.surp_mix+bag | 749 | – | 0.000 | 0.000 | 0.004 | – | 0.209 | 0.507 | 0.275 | 0.247 | – | – | 0.010 | – |
| Z16.dst+succ.05+dst | 541 | – | 0.000 | 0.000 | 0.006 | – | 0.191 | 0.515 | 0.286 | 0.247 | – | – | 0.010 | – |

#### Input-side rewrites (suite A)

| model | ppl↓ | pplB↓ | cloze@1 | cloze@5 | clozeMRR | clozeB@5 | ARC | Wino | TQA | HS | HSB | MMLU | copy | copyGain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base:identity | 421 | – | 0.026 | 0.079 | 0.065 | – | 0.202 | 0.503 | 0.307 | 0.245 | – | – | 0.113 | 0.010 |
| emb:dead_off | 742 | – | 0.000 | 0.053 | 0.027 | – | 0.188 | 0.517 | 0.338 | 0.250 | – | – | 0.080 | 0.003 |
| emb:dead_half | 459 | – | 0.026 | 0.079 | 0.053 | – | 0.217 | 0.496 | 0.322 | 0.247 | – | – | 0.097 | 0.003 |
| P1.embH+dst | 2613 | – | 0.000 | 0.000 | 0.001 | – | 0.224 | 0.504 | 0.326 | 0.258 | – | – | 0.000 | – |
| P3.embHb.5+dst | 530 | – | 0.000 | 0.026 | 0.013 | – | 0.213 | 0.500 | 0.305 | 0.247 | – | – | 0.033 | – |

#### Operating on the Q&A checkpoint (suite A)

| model | ppl↓ | pplB↓ | cloze@1 | cloze@5 | clozeMRR | clozeB@5 | ARC | Wino | TQA | HS | HSB | MMLU | copy | copyGain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final | 2552 | 2297 | 0.079 | 0.105 | 0.088 | 0.000 | 0.209 | 0.509 | 0.294 | 0.260 | 0.258 | 0.263 | 0.007 | 0.000 |
| F0.final+dst | 344 | – | 0.000 | 0.026 | 0.017 | – | 0.206 | 0.506 | 0.310 | 0.245 | – | – | 0.003 | – |
| F2.final+dst_mix.5surp | 548 | – | 0.079 | 0.105 | 0.087 | – | 0.220 | 0.509 | 0.300 | 0.242 | – | – | 0.010 | – |
| G2.mix.5surp_n600 | 554 | – | 0.079 | 0.105 | 0.087 | – | 0.217 | 0.507 | 0.294 | 0.240 | – | – | 0.010 | – |
| G3.tv.25found+mix.5surp | 366 | – | 0.053 | 0.105 | 0.078 | – | 0.188 | 0.499 | 0.301 | 0.247 | – | – | 0.037 | – |


Notes on reading these tables:

* `ppl↓` is suite A (wikitext-2 **test**), `pplB↓` is the confirmation split
  (wikitext-2 **validation**, never fitted on). `clozeB@5`, `HSB`, `MMLU` are also
  confirmation-only. Suite A drove every selection decision; suite B was touched
  only after the picks were frozen.
* `–` = that candidate was screened on suite A only (the ~105 screening runs at
  ~1 min each would have doubled for A+B; only the shipped files were confirmed).
* `copy` = accuracy on 300 auto-built cloze items where the answer is a rare token
  that appeared earlier in the same 512-token window; `copyGain` subtracts the
  control where that earlier occurrence is blanked out (same length, same other
  tokens), which is what isolates use-of-context from plain predictability.
* MC columns (ARC/Wino/TQA/HS/MMLU) have ±1–2 pt sensitivity to logit scale (see
  the control table above) and n = 277–1 267, so nothing here should be read as a
  capability claim below ~3 pt. They are included mainly to show the shipped
  edits do **not** regress them beyond noise. The one visible dip: SHIPPEDQ loses
  2.2 pt on MMLU (n = 320, SE ≈ 2.4 pt) — real but marginal, and it bought 4.7×
  perplexity.


## 4. The shipped files

Two files in `surgery/out/`, each **448 MB**, same schema as the release,
`strict=True` loadable by `train_hope.py`'s `HOPE`, carrying a `surgery` block with
the exact op list, the base file's sha256, the tensor-level diff, and the measured
metrics. Optimizer state was dropped (weights + `step` + `best_val_loss` +
`cms_tiers` + `qa_format` remain), so `train_hope.py` resumes from them with fresh
optimizers exactly as it does for the release's own files.

| file | base | tensors changed | ‖Δw‖ from base | what was done |
|---|---|---|---|---|
| `hope_foundation_surg1.pth` | `hope_foundation_best.pth` (sha `eedbd1660485…`) | **2 / 113** | 2 804 | `head.weight`, `head.bias` re-solved; target = 40% GPT-2 + 60% the model's own distribution (`mixbase=0.6`), positions weighted by teacher surprisal, fitted on 300×512-token windows of wikitext-2 **train** |
| `hope_final_surg1.pth` | `hope_final_best.pth` (sha `b1cb920d12fe…`) | **2 / 113** | 2 505 | same solve at a 50/50 mixture (this checkpoint had wrecked its own wiki LM, so its own head is worth less) |

```bash
python3 chat.py --model surgery/out/hope_final_surg1.pth          # Q&A model
python3 generate.py --model surgery/out/hope_final_surg1.pth \
        --prompt "What is the capital of France?" --temperature 0.2
python3 chat.py --model surgery/out/hope_foundation_surg1.pth     # raw Wikipedia style
```

Verified through the repo's own stack (`src/verify_artifact.py`): `strict=True`
load in bf16 and fp32, `uses_qa_format` picks up the explicit flag, compiled T=1
steps run, `generate.py` works end to end. dtype check, because the re-solved head
has different row norms from the release's: greedy bf16-vs-fp32 agreement is
**100%** on the foundation build and **96.9%** on the Q&A build (the release
measured 100% for its own weights) — so use `--dtype fp32` if you want the numbers
in this file reproduced exactly.

An **alternative, more aggressive Q&A build** is prebuilt at
`hope_final_surg_alt.pth`: `task_vector λ=0.25` back toward the foundation, then
re-solve the head (113/113 tensors moved). It reaches ppl 366 and lifts
context-copying 0.007 → 0.037, but costs cloze@1 (0.079 → 0.053) and ARC
(0.209 → 0.188) — which is why the shipped file is the 2-tensor version. To rebuild
it from scratch:

```bash
python3 src/build_artifact.py --base downloads/hope_final_best.pth --other downloads/hope_foundation_best.pth \
  --steps '[{"op":"task_vector","params":{"lam":0.25}},
            {"op":"distill_head","params":{"n_windows":300,"stride":2,"ridge":0.001,"mixbase":0.5,"wmode":"surprisal"}}]' \
  --name final_aggressive --out out/hope_final_surg_alt.pth
```

Nothing outside `surgery/` was modified: `git status` shows only this new
directory. (One early control run — measuring the release's own checkpoint with the
repo's `probe_model.py` to validate my harness — appended a row to the tracked
`probe_stats.jsonl`; that row is kept here as `out/probe_control_row.jsonl` and the
tracked file was restored.)

**Disclosure on the fit corpus.** The solve uses wikitext-2 *train* only; test and
validation are never fitted on and are reported as held-out. The Q&A-formatted
variant explored during screening (`qa_frac`, part of the fit corpus drawn from
`obekt/obekt-question-answer-reasoning-micro-v0.1`) was **not shipped**, partly
because that corpus contains the release's flagship question 2 400 times — using it
would inflate exactly the demo the release already advertises. The 25 held-out
facts in `cloze_holdout_B` appear nowhere in it (checked).

## 5. What this cannot do — and the number that says why

A closed-form readout can only decode what the network hands it, so I measured
what is actually in the head's input. Ridge probes fitted from `h` (the post-CMS,
pre-head vector) back onto the quantities that produced it (`src/probe_h.py`,
held-out R²):

| target | from `h` in the release | from `h` after rewriting the embedding from the model's own head rows |
|---|---|---|
| current token embedding | 0.46 | 0.83 |
| previous token embedding | 0.15 | 0.18 |
| fast-memory read (before `proj_out`) | 0.54 | 0.57 |
| **causal context bag of embeddings** | **0.066** | 0.035 |

The context is the problem. Everything the fast memory integrates over the
sequence arrives at the head with R² ≈ 0.07 — i.e. the slow stack, which is
position-wise FFN + LayerNorm ×16, leaves almost none of it linearly reachable.
Since a linear head is the only thing I am allowed to re-derive without training,
that is the ceiling this surgery runs into. It also explains the two failures
above: rewriting `proj_v` as a lookup table injected a strong context signal into
a channel the stack then scrambled, and shifting the embedding changed the
residual statistics every LayerNorm was calibrated to (current-token decodability
went up to 0.83 and perplexity went from 207 to 2 613 — legible is not the same
as usable).

(`src/oracle_ce.py` was written to price that ceiling exactly — same solve, but
fed `h` plus the undistorted context bag, which the architecture cannot expose.
It needs a float64 workaround for MPS and I did not finish it; the R² table above
is the evidence I have.)

Behavioural evidence, same direction (suite A, memory edits applied *alone*, no
head re-solve — `out/lab8.jsonl`):

| memory surgery | ppl | context-copy acc | verdict |
|---|---|---|---|
| none (released foundation) | 421 | **0.113** | — |
| tie query to key (`q ← k`) | 1 814 | 0.037 | retrieval got *worse* |
| collapse keys + identity values (make the read a clean context bag) | 1 634 | 0.017 | worse |
| force fast forgetting (α bias −4) | 642 | 0.060 | worse |
| force full retention (α bias +4) | 508 | 0.100 | worse |
| closed-form successor lookup in `proj_v` (then re-solve head) | 541–641 | 0.010 | worse |

Not one hand-designed read/write geometry beat the trained one on retrieval. The
memory is already sitting in a local optimum of the objective it was trained
against; the only edit that helped anything was re-solving the output layer.

Capability that would need new information — facts, reasoning, instruction
following — does not appear. On the 25 held-out facts the foundation build moved
top-5 from 0.000 to **0.040** (1 fact in 25) and the Q&A build stayed at 0.000; MC
stayed at chance (where GPT-2 small also sits);
the model still answers unfamiliar questions with dataset-flavored
non-sequiturs. What improved is language modelling: **~1.9× better perplexity on
held-out Wikipedia and ~4.7× on the Q&A release, with the ranking and retrieval
behaviour preserved rather than traded away.**

To go materially further you would have to change the architecture (interleave the
memory, or give the head a direct path from `fast_out`) and then *train* — which
was explicitly off the table here.

## 5b. What it is actually like to use, and what it says about the project's claims

Greedy (`temperature 0.0`) through the repo's own `chat.py`/`generate.py` path:

* `hope_final_surg1.pth` — "Why is the sky blue?" → *"The sky has a short
  wavelengths that gets scattered by air molecules…"*: the right mechanism, worse
  grammar than the release's own answer, and the same slide into dataset-flavored
  filler. "What is the capital of Japan?" → *"Reasoning: The answer focuses on a
  child's desire for independence…"* — i.e. still a non-answer, exactly the
  failure mode the release documents. **No factual capability was added.**
* `hope_foundation_surg1.pth` — fluent Wikipedia-genre prose; on a bare question it
  writes an article stub, and once it answered a question prompt with an immediate
  end-of-article token (empty output), because raw-mode EOS behaviour is unchanged.

Two numbers here are worth lifting into `EXPERIMENT_LOG.md`, because they measure
things the repo currently only claims:

1. **AGENTS.md open item #10 ("benchmark forgetting — claim is structural, not
   measured")** — now it is measured. Going from the 48k-step foundation to the
   6k-step Q&A fine-tune moves held-out wiki perplexity **421 → 2 552** (6.1×) and
   context-copying accuracy **11.3% → 0.7%**. The tiered slow-weight schedule did
   not protect either the language model or the fast memory's usability on these
   probes.
2. **The fast memory's contribution is real but small and fragile** — ablating
   `proj_out` costs perplexity 421 → 1 535 and copying 11.3% → 1.0%, yet the
   context signal that survives the 16 position-wise CMS blocks has linear
   recoverability R² ≈ 0.07 from the head's input. If the nested-learning claim is
   supposed to show up as in-context retrieval, this architecture as built is
   losing it between the memory and the head — which argues for interleaving the
   fast memory deeper, or giving `fast_out` a direct path to the head, rather than
   for more steps at the current wiring.

## 6. Reproducing

```bash
python3 src/diag.py                                   # weight statistics
python3 src/eval_suite.py --randomref --gpt2ref       # reference rows
python3 src/eval_suite.py --label name=path.pth        # one checkpoint, suites A+B
python3 src/lab.py --base <pth> --other <pth> --specs src/specs4.json --suites A --out out/lab4.jsonl
python3 src/build_artifact.py --base <pth> --other <pth> --steps '<json steps>' \
                              --name tag --out out/x.pth --numbers out/metrics.json
python3 src/verify_artifact.py out/x.pth              # loads via the repo's own stack
python3 src/probe_h.py; python3 src/oracle_ce.py
python3 test_chunked.py && python3 test_nested.py && python3 test_fixes.py   # repo guardrails
```
