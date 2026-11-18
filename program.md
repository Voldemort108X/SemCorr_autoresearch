# program.md — autonomous decoder research for Déjà Vu (5 days, 2026-11-18 → 2026-11-23)

You are an autonomous research agent. The author is away and will NOT answer questions. This file is your
contract. The author edits it; you do not (if you think it is wrong, write that in `journal/ideas.md`).
Modelled on karpathy/autoresearch: one frozen evaluator, one editable file, one metric, keep/discard via git,
never stop.

---------------------------------------------------------------------------------------------------------------
## 0. Hard rules (violating any of these is worse than finding nothing)

1. **Write only inside `/workspace/SemCorr_autoresearch/`.** `/workspace/SemCorr_proj/` (Datasets, Code, Results,
   cvpr2027_*) and `/workspace/SemCorrPaper/` are READ-ONLY: read data, read code, copy code into `vendor/`.
   Never run a script that writes into them (the SemCorr2D scripts write to `../../../Results/...` relative to cwd;
   never run them from their own folder). Never run `b2_backup_all.sh`. Never touch their git repos.
   Do not import Python from there in a way that writes `__pycache__` (keep `PYTHONDONTWRITEBYTECODE=1`; prefer
   copies in `vendor/`).
2. **RAM:** the machine is shared (251 GB total, ~70 GB used by others). Keep YOUR total RSS < 60 GB. Use disk
   caches + `np.load(mmap_mode='r')` / per-category loading, not whole-split tensors in RAM. Check `free -g` before
   any job that may be large.
3. **Disk:** keep this folder < 300 GB (`du -sh cache runs` daily). The network volume is slow for many tiny files:
   cache in a few large files per category.
4. **GPU:** the single RTX 4090 (24 GB) is yours. One job at a time unless both fit.
5. **Git:** this folder is its own local repo (no remote). Commit/reset freely here. Never `git push`.
6. **No new packages** unless an idea is impossible without one (then log why in the journal). Available:
   torch 1.13.1+cu117, numpy, scipy, sklearn, cv2, kornia 0.6, POT (`ot`), faiss, networkx, pandas, matplotlib.
7. **Honesty:** never report a number you did not produce from a log in `runs/`. Record what data each decision was
   selected on (see §3). Negative results are results.

---------------------------------------------------------------------------------------------------------------
## 1. Context: what Déjà Vu is and where the gap is

Paper: `/workspace/SemCorrPaper/cvpr2027/sections/{1_intro,3_methods,5_results}.tex` (read them once).
Decoder D(S, ℓ_i) → m̂_i on the cosine similarity matrix S = F_s F_tᵀ (60×60 grid, L = 3600, masked to dilated
object masks):
- **Nominate:** softmax(S[ℓ_i,:]/τ), τ = 0.1, smallest K with cumulative prob ≥ p = 0.9, clamped to [1, K_max = 30]
  (in practice the cap binds almost always).
- **Re-rank:** R = S[ℓ_i, m_k] − λ·d_cyc/w, with d_cyc = ‖ρ(argmax_ℓ S[ℓ, m_k]) − ρ(ℓ_i)‖₂ in patches and
  w = 60. λ = 0.5 (selected on SPair val).
- **Unchanged, as in the base method:** adaptive flip (pick the flipped source if its feature distance is lower),
  window soft-argmax refinement (window 5) around the chosen patch, masks with dilation 3.

Code (read-only):
- Decoder: `/workspace/SemCorr_proj/Code/SemCorrVLM_cvpr26/semcorr_vfm/SemCorr2D/utils/utils_correspondence_topp_cycle.py`
  → `calculate_keypoint_transformation_topp_cycle`.
- Fast multi-head evaluator (the best template): `.../SemCorr2D/pck_eval_cvpr27_review.py` + `.../SemCorr2D/utils/utils_review_heads.py`
  (`_PairState`, `window_soft_argmax`, `select`, `fast_get_distance`). It imports `get_patch_descriptors` from
  `pck_train_patch_ours_neurips_rebuttal.py`. Run settings per base method: `/workspace/SemCorr_proj/Results/cvpr2027_review/run_task.sh`.
- Features: `/workspace/SemCorr_proj/Datasets/SPair-71k/features/<cat>/<img>_{sd,dino}[_flip].pt` + `_mask[_flip].png`
  (TLR = SD+DINO, `--DUMMY_NET`), `features_cleandift/` (CleanDIFT), DIY-SC = `features/` + AggregationNetwork
  `.../baselines_neurips/DIY-SC/ckpts/0280_spair/best.pth` (config `eval_spair_supervised_rebuttal.yaml`).
  All 100 images/category have features → trn (53,340 pairs), val (5,384), test (12,234) are all usable.
- Existing per-keypoint error dumps (heads incl. `top1`, `lam0.5`, `ours`, dual-softmax, CSLS, NCNet, k-reciprocal):
  `/workspace/SemCorr_proj/Results/cvpr2027_review/{tlr,cleandift,diysc}/{val,test}_<cat>.npz`. Use these for the
  sanity gate in §2.

**The gap (full SPair test, per keypoint, full pipeline):** TLR top-1 69.8 / Déjà Vu λ 0.5 71.2 @0.10
(54.1 → 55.0 @0.05, 11.0 → 11.2 @0.01). DIY-SC 74.4 → 75.0. Top-k oracle (patch level, paper Tab. 1): top-5
80.7, top-10 87.3, top-30 94.9. We recover ~1.4 of ~25 points. Known failure modes (paper + supp): 64.5 % of
failures land on ANOTHER annotated keypoint, mostly the **mirror-symmetric** part (left/right leg, wheel, wing);
cycle distance is a **local** signal; mirror parts can both be cycle-consistent. The paper's limitation section says
closing the gap "likely calls for global reasoning". Already tried and NOT better than Déjà Vu (don't re-run as
ideas, use as baselines): exact MNN, MNN with tolerance, binary penalty, dual-softmax, CSLS, NCNet-style filtering,
k-reciprocal, reverse-rank, Sinkhorn / partial OT re-ranking, fixed top-k pools.

**Research question:** which decoder over the same features (ideally only S) closes the most of the gap between
top-1 and the top-k oracle, without tuning on the reporting split? Training-free is preferred; learned is allowed
(track T3) but must be reported separately.

---------------------------------------------------------------------------------------------------------------
## 2. Day-0 setup (do once; record progress in `journal/STATE.md`)

1. `git init`, commit the skeleton, `git checkout -b autoresearch/nov18`.
2. `vendor/`: copy (don't import) the minimal code to load images, keypoints, masks and features, build the TLR /
   CleanDIFT / DIY-SC descriptors, adaptive-flip decision, soft-argmax, and PCK. Note the source path at the top
   of each copied file.
3. `prepare.py` (frozen after setup, never edited during the loop):
   - `python prepare.py --method {tlr,cleandift,diysc} --split {dev,val,test,trn_train} [--cats ...]` builds the
     cache in `cache/<method>/<split>/<cat>.*`: per image fp16 descriptors (orig + flipped source), patch masks,
     keypoints, bbox size, and the **adaptive-flip arm per pair** (it does not depend on the decoder, so it is
     decided once, exactly as the original code does).
   - Splits: `dev` = 300 pairs/category sampled from SPair **trn** with seed 0 (≈5,400 pairs; the selection split);
     `val` = official val; `test` = official test; `trn_train` = the remaining trn pairs (only for T3).
   - `evaluate(decode_fn, method, split, cats=None)` → per-keypoint errors / bbox → PCK@0.10/0.05/0.01 (per keypoint
     AND per image), plus the reference heads `top1`, `dv` (Déjà Vu λ 0.5), and `oracle@k` (k = 5, 10, 30: best
     candidate among the top-k by S, with the same refinement) on the same pairs. Saves `runs/<id>/err.npz`,
     `metrics.json`; prints `^dev:`-style grep-able summary lines.
   - Paired bootstrap over pairs (1,000 resamples) for Δ vs `dv` → the noise floor (report its std).
4. **Sanity gate (must pass before the loop):** on official **val**, the cached pipeline's `top1` and `dv` per-keypoint
   PCK must equal the `top1` and `lam0.5` heads of `/workspace/SemCorr_proj/Results/cvpr2027_review/tlr/val_*.npz`
   to ±0.05 (ideally exact per keypoint). If not, find the difference (flip arm, mask, window, β) before going on.
   Do the same for `cleandift` and `diysc` (their heads were run with λ 3.0 / 1.0 in that folder; compare `top1`
   and the matching `lam*` head).
5. Measure: seconds per `evaluate` on dev for `top1` and `dv`, cache size, RSS. Target ≤ 5 min per experiment
   (use GPU batching; precompute per-pair S on the fly from the cached descriptors).
6. Write the first `results.tsv` rows (status `ref`): top1, dv, oracle@5/10/30 on dev, for TLR.
7. Commit. Update `README.md` (how to run, layout, current best) and `journal/STATE.md`.

---------------------------------------------------------------------------------------------------------------
## 3. Metric, splits and the keep rule

- **The editable file is `decode.py` only** (plus new files under `decode_lib/` if it grows; same rules).
  Interface: `decode(pair) -> pred_xy [N,2]` (pixel coordinates in the 840 px annotation frame) for the source
  keypoints of one pair (or a batched variant), where `pair` gives the GPU tensors: source descriptors (the arm
  chosen by the adaptive flip), the other source arm, target descriptors, both patch masks, source keypoint patch
  indices, and image sizes. It may compute S, call the vendored soft-argmax, use any of these, and keep state
  (for T3: load a model trained by `decode.py --train`, budget ≤ 10 min).
  It must NOT read ground-truth target keypoints, bbox, or labels of the evaluated split.
- **Selection metric (`dev`, TLR):** `dev = mean over t ∈ {0.10, 0.05, 0.01} of (PCK_t(decoder) − PCK_t(top1))`,
  per keypoint. This is the paper's λ-selection criterion. Columns `d10/d05/d01` hold the three deltas.
- **Keep** iff `dev > best_dev + max(0.05, 1 bootstrap std)` AND no threshold below `top1`. Else **discard**
  (`git reset --hard HEAD~1`, keep the row). Crash → fix trivial bugs and re-run; else log `crash` and move on.
  Simpler code with equal dev → keep (simplicity criterion, as in autoresearch).
- **Official `val`** (TLR) is run for every new best and in the daily check. It is reported but never used for
  keep/discard. If the val gain is < half the dev gain, note "possible overfit to dev" in the journal.
- **`test`:** the author allows test evaluation at any time, but selection still uses dev only. Every test number
  in a report must say "selected on dev (SPair trn subset)".
- **Transfer** (every ~6 h, and for every candidate you consider a finalist): CleanDIFT and DIY-SC on `val`
  (DIY-SC was trained on SPair trn, so never use DIY-SC on dev). A decoder that helps TLR but hurts both others is
  not plug-and-play; note it.
- **Generalization** (finalists, day 4–5): SPair-U, AP-10K cross-family, PF-Pascal (data in
  `/workspace/SemCorr_proj/Datasets/{SPair-U,ap-10k,PF-dataset-PASCAL}`; settings as in the CLAUDE.md /
  run scripts of SemCorr_proj; λ and every other setting fixed from dev).

---------------------------------------------------------------------------------------------------------------
## 4. The loop (autoresearch style)

Each iteration:
1. Read `journal/STATE.md`, the tail of `results.tsv`, `journal/ideas.md`. Pick the next idea (respect track budgets, §5).
2. Edit `decode.py`. Write a one-line hypothesis in the journal **before** running.
3. `git commit -am "<seq>: <short description>"`.
4. `bash scripts/run.sh <seq> dev > runs/<seq>_<commit>/run.log 2>&1` (10-min timeout: kill → discard). Do not print
   the log into your context; `grep "^dev:\|^pck\|^sec:" run.log`, `tail -n 40` only on crash.
5. Append the row to `results.tsv` (do NOT commit results.tsv… exception: commit it once a day with the report so the
   history survives; it is small).
6. Keep or discard per §3. On keep: run val, update `README.md` "current best".
7. Every ~10 experiments: a short **analysis** step instead of an idea: where do the remaining errors of the current
   best sit (mirror swaps vs. other keypoint vs. background vs. near-miss; per category; vs. pool size; vs. oracle@k)?
   Save to `journal/analysis_<seq>.md`. Let the analysis drive the next ideas.

**Session protocol.** You run under `/loop` (dynamic). Work in long iterations (1–3 h of experiments), then update
`journal/STATE.md` (current best, what is running, next 3 ideas), commit, and schedule the next wake-up
(ScheduleWakeup, 60–120 s, same prompt). **NEVER STOP and never ask the author anything**: if blocked, log the
blocker and switch to another track. If out of ideas: re-read the paper's limitations, the analysis files, combine
near-misses, try something radical. If the context is summarized, `journal/STATE.md` is your memory.

---------------------------------------------------------------------------------------------------------------
## 5. Tracks and idea backlog (seed; extend in `journal/ideas.md`)

Budget guide (of ~4 days of loop time): T2 ≥ 40 %, T1 ≤ 20 %, T4 ≤ 15 %, T3 ≤ 25 %. Tag each row with its track.
Do not spend more than ~10 runs on pure hyperparameter tweaks of one idea before an analysis step.

**T1: training-free, per keypoint (Déjà Vu family).**
- Soft cycle: the expected reverse location under softmax(S[:, m]/τ_r) instead of argmax; the reverse entropy as a cue.
- Reverse top-k: the distance to the query from the best of the k reverse matches; reverse rank of ℓ_i in column m.
- Multi-hop cycles S→T→S→T; cycle through the neighbourhood of ℓ_i (a 3×3 source patch window votes).
- **Mirror test:** use the other source arm (the flipped image) and the SPair flip permutation: does candidate m map
  back closer to the query's mirror counterpart than to the query? Penalize those (mirror swaps are the main failure).
- Score calibration: z-scored similarities per row/column; pool-adaptive λ.

**T2: training-free, global per pair (main bet; the paper's stated future work).**
- Dense anchors: mutual-NN pairs over the object masks (and/or high-confidence Déjà Vu matches) → fit a pair-level
  transform (similarity / affine / TPS / piecewise-affine, RANSAC or IRLS, with a flip-aware variant) → score
  candidates by residual to the transform: R = S − λ·d_cyc/w − μ·resid/w.
- Anchor voting: each candidate's displacement relative to K nearby anchors (local affine consistency, as in
  GMS / locally-affine matching).
- Joint decoding of the N query keypoints of a pair: MRF / graph matching over their candidate pools with pairwise
  terms preserving relative geometry (distances, orientation / handedness, ordering), solved by ICM, TRW / loopy
  BP, or exact enumeration on small pools; a one-to-one constraint among the query keypoints.
- Dense flow regularization: decode ALL masked source patches (not just keypoints), smooth the field (bilateral /
  median in source space), read off the keypoints.
- Global orientation: estimate whether the pair is mirrored (left↔right) from anchors and break mirror ties with it.

**T3: learned light re-ranker (allowed; report separately as "trained on SPair trn").**
- Listwise model (MLP / small transformer over the K candidates) on cues from T1/T2 (S, rank, d_cyc, soft-cycle,
  anchor residual, mirror score, pool entropy…), trained on `trn_train` pairs, ≤ 10 min. Category-agnostic inputs
  only (no category id) so it can transfer; must be checked on SPair-U (unseen keypoints) and AP-10K.
- Learned pairwise potentials for the T2 MRF.

**T4: candidate pool and refinement.**
- Separate similarities per feature source (SD s3/s4/s5, DINO) as extra cues or an ensemble; a flip-ensembled S.
- Larger / smaller K_max with the better re-ranker; recall of the pool vs. accuracy of the choice.
- Refinement: soft-argmax window/β around the chosen candidate; sub-patch refinement that uses the cycle.

---------------------------------------------------------------------------------------------------------------
## 6. Reporting

- `journal/DAY<n>.md`: one entry per experiment (hypothesis → result → interpretation), terse.
- **Daily summary** at ~04:00 pod time (2026-11-19 … 2026-11-22): `reports/daily_<date>.md` with the best dev/val so
  far vs. top1 / dv / oracle, the top-5 ideas tried with Δ, what failed and why, the plan for the next day. Send it to
  the author with SendUserFile (or PushNotification with a 3-line summary if that is unavailable). Commit
  `results.tsv` + the report.
- **Final (start 2026-11-22 ~12:00, finish by 2026-11-23 04:00):** stop new ideas; take the 2–3 best decoders (at least
  one training-free); run test + transfer (CleanDIFT, DIY-SC) + generalization (§3). Write `reports/FINAL_REPORT.md`:
  the method(s) in paper-ready prose (equations, hyperparameters, selection protocol), tables vs. top1 / Déjà Vu /
  oracle at all three thresholds, per-category, the failure analysis, runtime, negative results, and a recommendation
  (what is worth adding to the paper or a follow-up). Send it to the author. Then keep exploring until stopped.
