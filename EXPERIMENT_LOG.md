# HOPE Training Experiment — Decision & Measurement Log

Living log for the 2026-09-25 training run. Kept locally until the publish
decision (see D3); becomes part of the repo if the experiment is published.

## Mandate

- User goal: faster training + inference; principles verified sound first. (Done — see commits `812db78..83f1e7c`: exact chunked delta-rule scan ~5x total step speedup, validation-cache fix, fast tokenizer, prefetch, fused tier updates.)
- User instruction (verbatim): *"do not stick to 12 000 steps pre-training and 3000 post-training. if you feel results are not good enough, feel free to increase training steps, just do not forget to document decisions and measurement"*
- Publishing constraint (user): document experiment on GitHub + upload model **only if it is a "wow"**.

## WOW criteria (defined before results, to avoid moving goalposts)

1. Phase 2 complete — Q&A format learned.
2. Factual cloze top-5 hit rate >= 60% (20-probe suite in `probe_model.py`).
3. Q&A samples actually answer their questions (qualitative, quoted in report).
4. Val perplexity <= 30 (val loss <= ~3.4).

## Measurement protocol

- Training vitals: every 20 min from `foundation.log` / `finetune.log` (step, loss, val, tok/s) — cron `27bae27c`.
- Learning probes: hourly via `probe_model.py` (CPU copy of latest checkpoint, zero training interference; copy+retry avoids save races). Machine-readable accumulation in `probe_stats.jsonl`; cron `fd5e9b6d` reports trend.
- Probe suite: 20 factual cloze probes (rank + prob of correct first token), 5 Q&A generations (seeded, temp 0.7, top-k 40), 3 encyclopedic generations.

## Decisions

### D1 — 2026-09-25 ~11:20 — Start pipeline at default budget (12k + 3k)
Rationale: fresh optimized code path; wanted live speed/loss data before committing GPU-hours. Started `run_pipeline.py` (Phase 1 Wikipedia 12k steps -> `hope_foundation.pth`; Phase 2 Q&A 3k steps -> `hope_final.pth`).

### D2 — 2026-09-25 ~12:10 — Baseline probe at step 2,757: factual 0% top-5, ppl 268
Interpretation: normal "syntax before facts" stage (~45M tokens seen). No action; continue.

### D3 — 2026-09-25 ~19:10 — Budget policy after user mandate
Protocol adopted (pre-registered decision rule):
- Let the 12k + 3k pipeline complete (~4 h) to get a full Phase 1+2 datapoint — cheaper and more informative than mid-flight changes (the running process reads `max_steps` from memory; it cannot be retargeted without a restart).
- **Extension trigger**: at Phase 1 end, if val loss is still improving by > 0.1 per 1,000 steps OR factual top-5 < 30% OR (after Phase 2) any WOW criterion is missed while curves are still descending, extend:
  - Phase 1 extension: resume `hope_foundation.pth` with `max_steps=20000..24000` (LR cosine recomputes on resume — a mild re-warm, standard for continued training; scheduler lambda is rebuilt from the new CONFIG, optimizer/tier state resumes exactly).
  - Phase 2 redo: re-seed `hope_final.pth` from the NEW `hope_foundation_best.pth` (step=0, fresh optimizer — same logic as `run_pipeline.py` lines 64-87), fine-tune with `max_steps=6000..8000`. Watch for overfitting: the Q&A micro-set is small and `isolate_samples=True` cycles it — if Phase 2 val loss rises while train loss falls, stop at the best checkpoint (already saved automatically).
- Every extension run gets a D-entry here with: trigger evidence (numbers), new budget, expected outcome, and post-run measurement.

### D4 — 2026-09-25 ~21:15 — USER DECISION: 48k marathon (supersedes D3 auto-rule)
Asked user to choose 20k moderate / 48k marathon / keep-12k; user chose **48k marathon**.
Evidence at decision: val 4.73@9.4k plateauing under cosine LR decay (not data exhaustion — only ~197M of the ~768M-token article pool consumed at 12k); factual top-5 5%, MRR 0.062 doubling per ~3k steps; Chinchilla-optimal for 86M params ≈ 1.7B tokens ≈ 100k steps.

**Plan & mechanics (all scripted, chained, unattended-safe):**
1. Let Phase 1 finish its 12k cosine-annealed baseline (~21:50). Watcher kills `run_pipeline.py` (PIDs 3790/3794) at the "PHASE 1 COMPLETE" marker — before Phase 2 wastes ~50 min on a foundation that is about to be extended. Partial `hope_final.pth` from the killed seeding is deleted (re-seeded properly later).
2. `extend_training.py`: snapshot `hope_foundation_best.pth` → `hope_foundation_best_step12k.pth`; patch resume checkpoint (`best_val_loss=inf`, scheduler `base_lrs` 2e-4→**1e-4**); resume 12k→**48k** steps (~10.4 h), log `foundation_ext.log`.
   - **LR re-warm**: on resume the cosine recomputes against the 48k horizon (88% of peak at step 12k). Base lowered to 1e-4 → jump capped at ~8.8e-5, decaying to 1e-6 by 48k. **Monitor first ~500 steps for a loss spike** (grad clip 1.0 + warm AdamW state should absorb it; intervene if val > 5.8 sustained past step 13k).
   - **Data protocol change**: `max_samples` 500k→1.5M windows (~213k articles ≈ single pass over the 48k budget of ~786M tokens); `train_skip_samples=65000` (extension starts near the old coverage boundary ~70k articles → articles 1–70k seen ~2×, 70k–213k seen 1×; original 12k run stayed inside articles 1–70k).
   - **Val protocol change**: `val_skip_samples` 100k→**600k** articles — with coverage extended to ~213k articles, the old val region (100k–100.2k) would leak into training. New val series NOT comparable to old; bridge = first extension validation of the (near-)12k model, logged. Old best snapshotted (above) and `best_val_loss` reset to inf so `_best` tracking is per-protocol.
3. `extend_phase2.py` (auto-chained after extension): re-seed `hope_final.pth` from the NEW foundation best (step=0, fresh optimizer — run_pipeline logic), fine-tune Q&A **6000 steps** (~1.8 h), lr 5e-5, `finetune_ext.log`. Phase-2 val metric may be degenerate on the micro dataset (near-empty held-out slice) → authoritative Phase-2 evaluation is the probe suite.
4. Whole chain runs under `caffeinate -i` (no idle sleep), checkpoints every 1000 steps (max crash loss ~17 min), 20-min vitals + hourly probe crons keep reporting; watcher aborts the chain and reports if any stage exits non-zero.

**Timeline**: extension done ~08:30 Sep 26; Phase 2 done ~10:30 Sep 26; WOW verdict + publish question to user right after (repo docs + 986MB model to GitHub Releases — only on explicit yes).

**Code change (committed)**: `train_hope.py` gains `train_skip_samples` (default 0) and `val_skip_samples` (default 100000) CONFIG knobs — defaults reproduce previous behavior exactly; all 64 tests green.

### D5 — 2026-09-26 09:45 — FINAL VERDICT: WOW bar NOT met (1/4 criteria); publish decision deferred to user
Marathon complete: ALL_DONE 09:44 (48k foundation + 6k Phase 2, ~14.2 h wall-clock end-to-end, zero crashes/interventions). Final artifacts: `hope_foundation.pth` (48k), `hope_foundation_best.pth` (val 4.178 @ ~41k, new-split), `hope_foundation_best_step12k.pth` (snapshot, old-split 4.707), `hope_final.pth` (6k QA), `hope_final_best.pth` (QA val 3.506 @ 5,273).

**Pre-registered criteria vs final measurements (probes @6,000 / @5,273):**
| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | Phase 2 complete, Q&A format learned | Format fully learned by ~1k steps; 100% of probes emit Answer/Reasoning structure | **PASS** |
| 2 | Factual cloze top-5 >= 60% | **20%** (4/20: world #1 p=52%, energy #1 p=69%, H2O #1 p=24%, Sun #2); MRR 0.162-0.165 | **FAIL** |
| 3 | Q&A samples actually answer their questions | Format yes, grounding no — "capital of France" -> "It's a social media when you're older"; near-correct Rayleigh-scattering prose exists in-distribution but is not reliably question-conditioned | **FAIL** |
| 4 | Val perplexity <= 30 | Wikipedia: **65.2** (val 4.187); QA-domain: 33.3 (val 3.51) | **FAIL** |

**Interpretation (honest):** 786M tokens through an 86M-param model buys fluent English + wiki genre mastery + a handful of hardened facts — consistent with scaling laws; criterion 2/4 would plausibly need 10-50x more tokens (weeks on this hardware), so "train longer until wow" is not recommended. The genuinely valuable results are (a) the 5x speedup methodology (already public in git), (b) the fully documented learning curve: ppl 371→65, MRR 0.008→0.177, fact-emergence ranks (Sun #163→#2 over 43k steps), overfit-divergence finding (QA val up while factual probes improved), late-anneal val recovery 4.17→3.51 after the overfit minimum.

**Per user constraint ("only if it is a wow"): default = do NOT publish experiment docs/model.** Options presented to user: (a) keep everything local (default), (b) publish anyway as an honest negative-result/learning-curve report (EXPERIMENT.md + log + stats + probe harness; model optional via GitHub Releases — gh auth verified as obekt), (c) publish docs only, no model.
**DECISION (user, 2026-09-26 ~09:50): publish docs + model** — EXPERIMENT.md writeup + this log + probe_stats.jsonl + probe harness + extension drivers committed; `hope_final_best.pth` and `hope_foundation_best.pth` uploaded to GitHub Release `v0.1-marathon`.

### D6 — 2026-09-26 ~10:40 — Inference optimization pass (post-marathon, user-requested)
Scope: bf16 weights, last_only prefill, compiled T=1 steps, nucleus+repetition-penalty sampler; shared helpers in train_hope.py, all four inference scripts rewired.

**Decisions & measurements:**
- bf16 default for inference, **scan + memory state always fp32**. Gate: 100% greedy token agreement vs fp32 (5 prompts × 48 tok) and identical probe metrics (top-5 20%, MRR 0.165). Speed: 387→463 tok/s (1.18×; generation is kernel-launch-bound, not purely bandwidth-bound).
- `last_only=True` prefill (CMS blocks are position-wise): T=512 prefill 12.8→5.1 ms (2.49×), T=12 4.0→2.4 ms (1.64×). Equivalence proven in test_inference.py (final-position logits atol 1e-4, incl. carried state).
- `torch.compile(dynamic=False)` on T=1 steps only: +1.15× (427→489 tok/s). Prefill MUST stay eager — inductor cannot lower linalg_solve_triangular on MPS. Combined generation ≈ 500 tok/s (1.29× vs fp32-eager baseline).
- Sampler: temperature + top-k 40 + top-p 0.9 + repetition penalty 1.15 + MPS multinomial clamp. Fixes observed repetition degeneration.

**Bugs found by the new tests/validation (all fixed):**
1. Padding mask created `.float()` promoted the fp32 memory state against bf16 params → scan now casts everything to fp32 explicitly.
2. `_tri_solve_ok(device, x.dtype)` probed with bf16 → Metal assert ABORTS the process (uncatchable). Probe pinned to fp32 (scan is fp32 by construction).
3. `sample_next_token` aliased caller logits when already fp32 (`.float()` no-copy) — repetition penalty mutated the caller's tensor across calls. Fixed with explicit clone; regression test added.

**Greedy-decoding finding (published as EXPERIMENT.md addendum):** `hope_final_best` answers "Why is the sky blue?" CORRECTLY under greedy (Rayleigh scattering + reasoning), with/without trailing space. Prompt-format fragile (top-1 p≈9% in the flat regime); does not generalize to other questions (memorized dataset item). Criterion 3 stays FAIL; characterization refined to "retrieves one canonical answer greedily; format-sensitive; no cross-question generalization". Practical: use temp ≤0.3 with this checkpoint.

### Runbook for extension (exact mechanics)
```bash
# 1) Phase 1 extension (auto-resumes from step in hope_foundation.pth)
python3 - <<'EOF'
import train_hope as th
th.CONFIG.update({"max_steps": 20000, "save_path": "hope_foundation.pth",
                  "log_file": "foundation_ext.log", "checkpoint_every": 1000})
th.train()   # dataset/optimizer/tier settings unchanged from Phase 1 defaults
EOF
# 2) Re-seed + re-run Phase 2 (replicates run_pipeline.py seeding, then trains)
#    -> write extend_phase2.py when needed: copy run_pipeline.py lines 49-90,
#       set max_steps 6000-8000, log_file finetune_ext.log
```

## Measurements

### Val-loss trajectory (Phase 1, from foundation.log; every ~60 s validation)
| step | val loss | note |
|---|---|---|
| 201 | 9.734 | first validation |
| 1,346 | 6.385 | warmup ending |
| 2,277 | 5.774 | |
| 2,757 | 5.592 | probe baseline: factual 0% top-5, MRR 0.014 |
| 2,934 | 5.535 | slope ~ -0.04/100 steps (steep, healthy) |
| 3,644 | 5.361 | |
| 4,791 | 5.158 | |
| 5,000 | 5.093 | probe #2: first factual hit (Everest→world, rank 5) |
| 7,084 | 4.775 | flattening (~-0.02/1k steps) |
| 8,249 | 4.759 | |
| 8,487 | 4.754 | probe #3: Everest→world now rank **1**; MRR doubled |
| 10,739 | 4.714 | |
| 11,779 | 4.707 | |
| 12,000 | 4.707 | **Phase-1 baseline FINAL** (ppl 110.7, LR annealed to 2e-6). Watcher killed pipeline at marker 21:31; snapshot `hope_foundation_best_step12k.pth` saved; resume checkpoint patched (best_val_loss=inf, base_lrs→1e-4); extension started 21:31 (PID under caffeinate). Transient ~50% rate dips observed 21:46-22:15 window were probe/diagnostic contention + stream jitter; recovered to 54-60 steps/min. |

### Val trajectory (new split, skip=600k) — extension run
| step | val loss | note |
|---|---|---|
| ~12,600 | 4.6221 | **bridge**: ~12k model on NEW split (old-split final was 4.7065; splits differ, both ~ppl 101-111 difficulty band) |
| 12,976 | 4.6231 | LR re-warm window 12,000-13,000 PASSED: no spike (train 4.5-5.1 band, val flat), LR started 8.81e-5 exactly as predicted, 67 steps/min |
| 14,082 | 4.5864 | |
| 14,769 | 4.5608 | new best; ppl < 100 for the first time (95.7) |
| 16,363 | 4.5178 | |
| 17,495 | 4.4820 | |
| 18,211 | 4.4679 | ppl 86.6; -0.154 since bridge (4.622) in ~5.6k steps |
| 19,795 | 4.4631 | |
| 20,944 | 4.4327 | |
| 21,852 | 4.4055 | ppl 81.9; -0.217 since bridge in ~9.2k steps |
| 23,238 | 4.3813 | |
| 24,386 | 4.3612 | |
| 25,375 | 4.3560 | ppl 77.8; -0.266 since bridge in ~12.8k steps |
| 25,533 | 4.3552 | 10th consecutive val improvement. **Measurement: probe runs stall training** — 01:17-01:27 probe window cost ~15 min of steps (rate 24,386→25,533 = ~32/min avg vs 57/min clean). Mitigation applied: probe cron now runs under `nice -n 15`. |
| 26,682 | 4.3275 | |
| 27,827 | 4.3061 | |
| 28,117 | 4.2972 | ppl 73.5; niced probe 02:17-02:24 — training rate during probe window looked nominal (nice mitigation appears effective, confirmed next check) |
| 28,706 | 4.3019 | train loss hit 3.94 — train/val gap 0.36 (watch item: was 0.20 at 15k; val still improving every check, not yet overfitting) |
| 30,114 | 4.2758 | broke the brief 4.30 plateau |
| 31,256 | 4.2689 | |
| 32,139 | 4.2638 | ppl 71.1; -0.358 since bridge in ~19.5k steps |
| 33,544 | 4.2558 | |
| 34,684 | 4.2508 | train≈val (gap closed, no overfitting) |
| 35,211 | 4.2456 | ppl 69.8 — under 70 |
| 36,971 | 4.2412 | broke the 4.25 band |
| 38,115 | 4.2378 | |
| 38,991 | 4.2307 | ppl ~68.6; -0.39 since bridge in ~26.4k steps |
| 40,414 | 4.2208 | |
| 41,565 | 4.1827 | annealing acceleration begins (-0.038/1.1k steps) |
| 42,440 | 4.1864 | best-ckpt val 4.177 (ppl 65.2); -0.445 since bridge |
| 43,868 | 4.1873 | |
| 45,015 | 4.1889 | val plateau at LR floor (best 4.177 @ ~41k stands) |
| 45,896 | 4.1871 | plateau confirmed; extension ends 48k with val ~4.19 |
| 48,000 | 4.1868 | **EXTENSION COMPLETE 08:01** (final LR 1e-6). Foundation summary: 12k baseline val 4.707 (old split) → 48k val 4.187 / best 4.178 (new split), ppl 111→65, ~786M tokens total. Best ckpt @ ~40,942 seeds Phase 2. |

### Phase 2 (Q&A fine-tune, 6k steps, from foundation best @40,942)
| step | train loss | val loss (QA slice, skip=2000) | note |
|---|---|---|---|
| 472 | 5.011 | 4.655 | started ~08:02; loss up from 4.3 = expected domain shift to Q&A format; val non-degenerate (QA dataset >2000 samples ✓); 16.5k tok/s |
| 1,349 | 4.981 | 4.249 | QA val dropping fast (4.66→4.25); format fully learned by ~1k steps |
| 1,621 | 3.653 | 4.388 | |
| 2,784 | 2.536 | **3.723** | val minimum region |
| 3,960 | 2.960 | 4.139 | **OVERFIT ONSET** (D4 guard condition): val rose 3.72→4.14 while train sits ~2.5-3.0 — QA micro-set memorization. `hope_final_best.pth` (best-val ≈3.72 @ ~2.8k) is preserved automatically and is the deliverable; letting the run finish to 6k (costless, LR annealing to 5e-7 may recover some val), but final verdict/probes must use final_best, not final. |

### Probe history (from probe_stats.jsonl)
| time | checkpoint | step | val ppl | factual top-5 | MRR | highlight |
|---|---|---|---|---|---|---|
| 18:50 | foundation_best | 2,757 | 268.3 | 0% | 0.014 | "sun rises in the" -> north/south (slot learned, fact not); "1945" at rank #57 |
| 18:50 | foundation | 2,000 | 370.8 | 0% | 0.008 | (older checkpoint probed for the record) |
| 19:29 | foundation_best | 4,971 | 162.9 | 5% | 0.030 | first top-5 hit: "tallest mountain in the **world**" (rank 5); "east" #12, H2"O" #13 |
| 19:29 | foundation | 5,000 | 162.9 | 5% | 0.021 | generations gaining narrative structure (dates, bios) but facts still scrambled |
| 20:29 | foundation_best | 6,766 | 113.2 | 5% | 0.059 | Everest→**world** promoted to rank **1**; "Sun" #146→#46 trajectory |
| 20:29 | foundation | 8,000 | 113.2 | 5% | 0.062 | MRR doubled vs 5k; generations now emit full wiki scaffolding ("See also / References / External links", birth-death category stubs) |
| 21:26 | foundation_best | 11,694 | 110.7 | 5% | 0.069 | "Earth orbits around the **Sun**" #163→#46→#9; H2"O" #6; Einstein→"physicist" enters top-100 (#76-100) |
| 21:26 | foundation | 11,000 | 111.0 | 5% | 0.071 | same trend; Phase-1 baseline probes (last before D4 val-protocol switch) |
| 22:26 | foundation_best | 14,769 | 95.7 | **10%** | 0.068 | 2nd hit: H2"**O**" #5 (p=3.1%); Everest→world now p=16.4% top-1; val ppl <100 first time |
| 22:26 | foundation | 15,000 | 95.7 | **10%** | 0.071 | broad rank climbs: Sun #15, Celsius #18, energy #30, physicist #48; "written by"→['John','David','Michael'] (author-name prior, not Shakespeare #1499); "Neil ___"→'Young' (Neil Young beats Armstrong #511 🎸) |
| 23:26 | foundation_best | 18,211 | 86.6 | **15%** | 0.102 | 3rd hit: "Earth orbits around the **Sun**" #3; H2O→#2 (p=8.8%); Everest p=18.9% |
| 23:26 | foundation | 18,000 | 87.1 | **15%** | 0.109 | MRR +50% in 3k steps; east #9, energy #9, Celsius #11 knocking on top-5; Einstein bio now has "Career"/"Medal of Honor" narrative arc (facts still invented) |
| 00:26 | foundation_best | 21,852 | 81.9 | 15% | 0.088 | "physicist" #48→#30; Celsius #8; **observation: individual ranks oscillate between nearby checkpoints (east #9→#40, Celsius #11→#72 on the 21k file) — LR still 6-7e-5; judge trends over 5k+ step windows, not single probes** |
| 00:26 | foundation | 21,000 | 83.8 | 15% | 0.085 | Everest→world #1 stable (p=0.26); Einstein bio gained "professor at the University of New York… 1882… University of Chicago" arc |
| 01:26 | foundation_best | 25,375 | 77.8 | 15% | 0.096 | ppl <80; MRR recovering from oscillation (0.085→0.096); "1945" #54→#40; H2O #2 (p=8.8%) |
| 01:26 | foundation | 25,000 | 77.9 | 15% | 0.090 | energy #12, Celsius #28, physicist #41 — steady teens/20s/40s bands; generations shifting to geography register ("village of Osterre, which lies to the west of the village") |
| 02:24 | foundation_best | 28,117 | 73.5 | 15% | 0.097 | Celsius #28→#14, east #49→#23, physicist #31; Everest p=26% |
| 02:24 | foundation | 28,000 | 73.7 | 15% | 0.103 | MRR trend 0.087@21k→0.093@25k→0.100@28k (oscillation damping as LR decays); Einstein bio: "married to the German-born American author… former professional baseball pitcher" (fluent, scrambled) |
| 03:24 | foundation_best | 32,112 | 71.1 | 15% | 0.100 | Everest p=**38%**; four probes clustered at #10-18 (physicist, east, Celsius, energy) — top-5 break-ins expected soon; niced probe: no training-rate impact ✓ |
| 03:24 | foundation | 32,000 | 71.2 | 15% | 0.099 | same cluster #10-17; "1945" oscillates (#59-89); generations vary by checkpoint (bio boilerplate "Albert Einstein was born in 1820. References") |
| 04:24 | foundation_best | 35,211 | 69.8 | 15% | **0.113** | ppl <70; Sun→#2 (p=7%); "east" #7 (closest ever); MRR trend 0.087→0.093→0.100→0.100→**0.112** monotone over 14k steps |
| 04:24 | foundation | 35,000 | 69.9 | 15% | 0.111 | "1945" #55-89→#44-46 (dates finally moving); energy #9, Celsius #10, physicist #15-16 — the #7-16 cluster should convert to top-5 hits as LR anneals to 1e-6 by 48k |
| 05:24 | foundation_best | 38,672 | 68.6 | 15% | **0.121** | MRR record #3 straight; Sun #2 (p=10%); **Celsius #7, energy #10** — conversions imminent as LR <1e-5; one generation shows repetition degeneration ("the character of the character…") — known small-LM failure at temp 0.7 |
| 05:24 | foundation | 38,000 | 69.1 | 15% | 0.117 | Einstein bio → fluent obituary register ("she worked as an editor of the magazine… died in the late 1960s"); east #13, physicist #15 |
| 06:24 | foundation_best | 40,942 | **65.2** | 15% | 0.113 | ppl −3.4 in 2.3k steps (annealing); Sun #2 p=10.8%; "1945" #46→#38; MRR consolidating ~0.11-0.12 (0.121 @38.7k was a fluctuation high) |
| 06:24 | foundation | 42,000 | 65.2 | 15% | 0.114 | Celsius stuck #6 three probes running — top-5 conversions deferred to final LR anneal (1e-6 by 48k); French Revolution gen gained naval-history register ("replacement for the French fleet… British and French Navy's crew") |
| 07:24 | foundation | 45,000 | 65.2 | 15% | 0.115 | **plateau confirmed**: ppl/MRR/ranks frozen vs 42k (Celsius #6 4th probe, energy #10, east #11, physicist #16-20, 1945 #39). Final 3k steps at LR≤1.5e-6 unlikely to move metrics. Projected final: top-5 15%, MRR ~0.115, ppl ~65 → WOW criteria 2 (≥60%) and 4 (ppl≤30) will FAIL; criteria 1/3 pending Phase 2 |
| 08:24 | foundation | **48,000** | 65.2 | 15% | 0.114 | **FOUNDATION FINAL**: projection confirmed exactly (top-5 15%, MRR 0.114, ppl 65.2); Celsius #6, energy #10, east #13 at the wire |
| 08:24 | final_best | 1,147 | 66.6 | **20%** | **0.152** | Phase 2 @ ~1k steps: **4th hit — photosynthesis→"energy" #1 (p=58%)**; Everest p=70%; QA format fully learned; QA content = micro-dataset themes (sunlight/raindrops/"your brain is still growing"), question-relevance still poor |
| 08:24 | final | 1,000 | 72.2 | **20%** | 0.145 | wiki-ppl rose 65→72 = expected fine-tuning domain shift; QA val improving fast (4.66→4.25 by step 1,349) |
| 09:24 | final_best | 2,664 | 37.7 (QA) | 15% | 0.096 | best-QA-val ckpt; factual probes slightly DOWN vs 4.5k (see finding below) |
| 09:24 | final | 4,500 | 37.7 (QA) | **20%** | **0.177** | **MRR record**: H2O→**#1** (p=24%), energy→**#1** (p=69%), Everest #1 (p=52%), Sun #2. **Finding: QA-val and factual-probe metrics DIVERGE in the overfit regime** — QA val rose (3.63→4.17) while factual cloze kept improving (science-flavored QA set reinforces fact circuits). Deliverable choice (final_best vs final) is metric-dependent → present both at verdict. Encyclopedic prompts now convert to QA format ("…begins with a strong thinking team? Answer: Math helps us…") = full format transfer |

### Speed (verified)
- ~16.4k tok/s steady (~1.0 s/step, 16,384 tok/step) — 5.2x pre-optimization baseline (~3.2k tok/s).
- Validation: ~2.5 s per run from 50-batch startup cache (was: minutes, re-streaming 100k samples).
