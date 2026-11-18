# DAY 1 — 2026-11-18

## Day-0 setup
- vendor/: core.py (desc assembly, masks, kps, fast_get_distance, PairState, select/graded, window soft-argmax),
  geoware.py, projection_network.py + resnet.py (DIY-SC). Sources noted per file.
- Cache: raw SD/DINO blocks in fp16 per category ([100 img, 2 arms]) + dilated patch masks; descriptors are re-assembled
  and normalised in fp32 on the GPU per pair. Flip arm decided once from the fp32 originals (exact replica).
- Sanity on val/aeroplane: top1 and lam0.5 identical to the review dump at every keypoint (max |Δerr| 1.2e-4, 0 PCK flips).
- Note: dev bbox = target bbox (the original loader appends src+trg thresholds for 'trn'; fixed in the vendored copy).
- Note: the decoder gets `arm_of_kpt` (frozen adaptive flip; it uses mutual visibility, as the original code does) but not
  the evaluated-keypoint mask; joint decoders use the source-visible keypoints of each arm as the query set.

## Loop
- Full-dev failure analysis of dv (cache/tlr/dev/pools.npz): wrong@.10 = 30.4 %; 57.8 % of those have a correct
  candidate in the pool. Wrong predictions land on: mirror kp 21.6 %, other kp 41.9 %, near-miss 18.4 %, far 18.1 %.
- Offline screen (dev, single cues added to dv): consensus-RANSAC residual (cap 5) +1.14, MNN-affine residual (cap 5)
  +1.04, soft-cycle +0.98, flip-ensemble similarity +1.17, ctx2 +0.95, keypoint MRF/ICM +1.15; MLS/handedness ≈ dv.
  Pointwise GBM over all cues (2-fold on dev) only +1.02 → the cues carry little extra information.
- seq 1 hypothesis [T2]: adding μ·min(resid to the pair-level consensus similarity transform, 5)/w (μ 0.5) to Déjà Vu
  fixes part of the mirror / other-kp swaps → dev ≈ +1.1 (offline +1.136).
- seq 2 hypothesis [T1/T4]: flip-ensemble — score the candidate also against the same semantic keypoint in the other source arm (mirrored counterpart) + its cycle there (WFL 0.5, LFL 0.25) → dev ≈ +1.37 (offline).
- seq 3 hypothesis [T4]: refine with the flip-ensembled similarity map (own + 0.5·other-arm map of the same semantic kp) → better localisation at all thresholds (sweep: dev +2.04).
- seq 4 hypothesis [T4]: flip-ensembled forward row (own + 0.75·other-arm row)/1.75 for nomination and the similarity term (replaces WFL) → better pool + choice (sweep +2.19).
- seq 3 → kept (dev +2.036, val +1.971). seq 4 → kept (dev +2.191, val +2.123; vs dv +1.41 dev / +1.23 val).
  Transfer (val, settings from TLR dev): CleanDIFT +1.963 (dv +0.700), DIY-SC +2.805 (dv +0.240).
- Sweeps (not committed): SRC_AVG (query 4-neighbour smoothing) 0.25 → +1.79, 0.5 → +1.29 (hurts @.01): negative.
  Greedy one-to-one constraint among query keypoints (offline, on seq2-like unary): ≤ +0.05: negative.
  Refinement β 0.02 (own map) raises @.10 but hurts @.05/.01; own-map window 3 +1.917 vs window 5 +1.371 (seq2
  settings), window 7 worse (+0.885): the base window 5 is too wide for this metric.
- Refinement sweeps on seq4 (dev): window 5/4/3/2 → +2.19/+2.50/+2.73/+2.81 (β .01); W3 β.015 +3.00, W3 β.02 +3.05,
  W3 β.03 +2.80, W2 β.02 +3.22, W2 β.03 +3.25, W2 β.05 +3.00, W1 β.02 +3.05, W4 β.02 +2.50. Analysis: journal/analysis_4.md.
- seq 5 hypothesis [T4]: refinement window 2 / β 0.03 (on the ensembled map) → dev ≈ +3.25. Baselines with the same
  refinement to be measured (fairness).
- seq 5 → kept: dev +3.251, val +3.036. Fair baselines (same W2/β.03 refinement, own map): top1 dev +0.410 / val +0.250,
  dv dev +1.767 / val +1.610 → decoder-only gain over dv ≈ +1.48 dev / +1.43 val.
- sweep5 on seq5 (dev): K 10 +3.16, K 20 +3.23, REF_ENS 1.0 +3.23, λ 1.0 +3.30, μ 1.0 +3.03 → all within noise; keep seq5.
- T3 setup: decode_lib/feats.py (seq-5 candidates + 19 category-agnostic features; reproduces seq5 exactly),
  scripts/dump_t3.py (tables with per-candidate refined GT error), decode_lib/t3.py (DeepSets listwise MLP,
  expected-metric-gain loss). trn_train cache built (48k pairs); dumping 600 pairs/category for training.
- seq 6 hypothesis [T3]: DeepSets listwise re-ranker (19 cue features, h64, 60 ep, trained on 600 trn_train pairs/cat, 13 s) on the seq-5 pool → dev ≈ +4.4 (offline table: +1.19 over seq5).
- seq 7 hypothesis [T1, training-free]: reverse-rank penalty 0.02·log1p(rank of the query in the candidate's own-arm column) on seq5 — T3 permutation importance ranked revrank/softcyc highest; offline on dev table +0.33 over seq5.
- seq 8 hypothesis [T3]: re-ranker with 12 extra features (competition from other query kps, mirror-partner competition, 3x3 local map shape), trained on ALL trn_train pairs (334k kpts), h128, 60 ep (79 s) → offline +1.50 over seq5 → dev ≈ +4.75.
- Training-free linear-weight coordinate ascent on the dev t3_x table (λ, LFL, μ, cap, RR, softcyc, comp): best +0.17 over seq7 (offline; likely partly dev-overfit) → the linear family is saturated; not run.
- Sweeps on seq7 (dev, not kept): CONS_ITER 1 +3.638 (+0.05); mirror-pair orientation swap SWAP_T 0/.5/.8 →
  +3.636/+3.650/+3.640 (+0.05–0.07, noise); peak (NMS) pools K_PEAK 10 / 20 / 10+top10 / 10 r2 → +2.75/+2.68/+3.43/+2.42;
  block reweighting SD_W .7/1.3 → +3.23/+2.97; refinement quad → +3.05 (@.01 ↑, @.10 ↓), mix → +3.59 (= seq7).
  T3 3-seed ensemble +0.06 over a single model (offline). Analysis: journal/analysis_8.md (mirror errors untouched).
- TEST (selected on dev): seq7 +3.457 (73.57/58.51/13.21), seq8 +4.791 (74.97/60.42/13.90); dv +0.823; dvW2 +1.750.
- Generalisation: SPair-U (no perm): dv +0.51, dvW2 +0.50, seq7 +0.99, seq8 +1.45. PF-Pascal (.10/.05/.15):
  dv 89.44/75.93/93.70, seq7 90.31/77.38/94.12, seq8 91.59/78.33/95.44 (top1 86.50/73.57/91.30).
  Sanity SPair-U top1/dv vs review dumps within ±0.01.
- Sweeps on seq7 (not kept): MU_ADAPT (μ scaled by consensus inlier ratio) .5/μ1 +3.50, .8/μ1 +3.53, .5 +3.58;
  FL_ALIGN 1 (snap the other-arm location to the most query-similar neighbour) +2.95 → negative.
- RR_FL (reverse rank in the other arm) .01/.02 → +3.66/+3.66 (+0.07, noise; not kept).
- T3-joint (decode_lib/t3j.py: DeepSets candidates + 2-layer Transformer across the pair's keypoints), x features,
  20 ep (542 s): +1.508 over seq5 on dev table vs +1.497 for the per-keypoint model → joint attention adds nothing with
  these features.
- AP-10K cross-family cache built (1,454 images, 6,300 pairs, adaptive flip with AP10K_FLIP); evaluating.
- PROF (relational profile correlation) .02/.05/.1 → +3.53/+3.48/+3.40: negative. AP-10K CF (test): dv +0.56, dvW2 +1.30, seq7 +2.50, seq8 +3.17.
- seq 9 hypothesis [T3]: x2 features (+ 5x5 ring of the refinement map, own-map 3x3, query self-similarity to its mirror partner / far source patches, object-relative positions of query and candidate, 2-hop cycle, target mask fraction; 64 feats) → offline +2.07 over seq5 → dev ≈ +5.3.
- DENSE (T2 dense-flow prior: local median displacement of cycle-consistent dense Déjà Vu matches) μ.5 R6 +3.60, μ.5 R10 +3.52, μ1 R4 +3.49 → no gain. seq 9 → kept (T3): dev +5.309, val +4.972, TEST +5.230 (75.50/61.05/14.05).
- !! T3 caveat found: SPair trn/val/test use disjoint images, but dev ⊂ trn shares ALL its images with trn_train
  (T3 training data). Bigger T3 models looked much better on dev (h256/100ep +2.82, h512/150ep +3.25 over seq5 vs
  h128/60ep +2.07) but on an image-disjoint protocol (decode_lib/t3split.py: train on trn_train pairs inside image
  half A, select on dev pairs inside half B) they are equal: h128 +1.72, h512 +1.68, h256 +1.64. → memorisation.
  Keep h128/60 (seq 9). From now on T3 choices use the A/B protocol; T3 dev numbers are optimistic, val/test are
  image-disjoint and trustworthy (seq9 val +4.972 / test +5.230 vs seq7 +3.312 / +3.457).
- seq9 transfer (val): CleanDIFT +4.882 (dv +0.700), DIY-SC +3.657 (dv +0.240).
- seq10 (analysis) T3 x2 without position features: A/B dev +1.38 vs +1.72 with; full model: val +4.820, SPair-U +1.122, Pascal +3.963, AP-10K +3.250 (seq9: +4.972/+0.982/+3.963/+3.456; seq8: +4.557/+1.447/+4.667/+3.169). Position priors are not what hurts SPair-U/Pascal; x→x2 features trade SPair/AP-10K gains for SPair-U/Pascal.
- Arm analysis (dev, seq7 scoring): all-kps arm0 +2.32, all-kps arm1 (queries rebuilt from source info) +2.04, adaptive +3.58; per-pair arm ORACLE +5.06, per-kpt oracle +6.85. Evidence-based arm choice (mean score / sim / MNN / d_cyc / consensus support, ± base feature distance): ≤ +3.68 — no better than the base decision. But using the base pair decision for ALL keypoints (arm-1 queries rebuilt from the source partner) gives +3.715 (offline).
- seq 11 hypothesis [T4]: ARM_PAIR — decode every keypoint in the pair's arm (rebuilt semantic queries) instead of the per-keypoint vis_flip rule (which also depends on target visibility) → +0.13.
- seq 12 [T3] seq9 model + ARM_PAIR: dev +5.400 (+0.091 < boot std 0.134) → DISCARD by the keep rule (val +5.078 vs +4.972, consistent but small). T3 best stays seq 9 (per-kpt arm rule); TF best seq 11 (ARM_PAIR).
- seq11 finalist (training-free): TEST +3.556 (73.72/58.67/13.19); CleanDIFT val +3.270; DIY-SC val +2.945; SPair-U +0.989; Pascal +3.480; AP-10K CF +2.676.
- seq13 (analysis) T3 x2 trained with extra perm-dropped pairs: dev table +1.98 over seq5 (seq9 +2.07); val +4.932, SPair-U +1.221, Pascal +3.797, AP-10K +3.576 (seq9: +4.972/+0.982/+3.963/+3.456) → within noise except SPair-U ↑.
- ZROW (pool-adaptive penalty scaling by top-K sim spread) .5/1 → +3.671/+3.666 vs seq11 +3.698: negative.
- sweeps on seq11: REF_CYC .5 +3.823, REF_SIG 1.5/W4 +3.827, REF_SIG 1/W3 +3.642, REF_SIG 2/W5 +3.580, both combined +3.816 / +3.730 / +3.823 → combinations don't add.
- seq 14 hypothesis [T4/T1]: refine on the Déjà Vu score map (sim − 0.5·d_cyc/w of every target patch in the window) → +0.12.
- seq 14 → kept: dev +3.823, val +3.583. T3 seq9 + T3_REF_CYC .5 (sweep dev) +5.282 vs +5.309 → no gain (labels were made with the plain refinement).
- REF_CONS (consensus-distance in the refinement map) .5/1 → +3.769/+3.478: negative. HAND2 (mirror-pair handedness direction from 4 source views: raw + mirrored per side) .05/.2/.5 → +3.851/+3.897/+3.801 (+0.07 max, noise); error types: mirror 6.75 → 6.59 (.2) → 6.47 (1.0) but other-kp 11.44 → 12.08: weak signal, not kept.
- seq14 finalist: TEST +3.711 (73.81/58.90/13.34), CleanDIFT val +3.463, DIY-SC val +2.917, SPair-U +1.237, Pascal +3.590, AP-10K +2.709.
- T3 with labels/features under the REF_CYC refinement (A/B protocol): B-set mean PCK 49.64 vs 49.83 for the plain-refinement model → no gain; T3 stays seq9.
- Runtime (test, ms/pair, decode only, RTX 4090 shared): top1/dv 2.8, seq7 7.0, seq11 10.9, seq14 14.8, seq8 T3 9.7, seq9 T3 17.8.
- T3 two-arm pool (60 candidates: 30 per source arm, arm features; decode_lib/twoarm.py, scripts/dump_t3_2arm.py), A/B protocol: B-set PCK 74.92/60.80/13.41 (mean 49.71) vs one-arm x2 75.46/60.98/13.05 (49.83) → no gain; the arm-oracle headroom is not learnable from these cues either.
- Leave-one-out ablation of seq14 (dev +3.823): −flip-ensembled row (NOM_ENS 0) +3.392 (−0.43); −other-arm cycle
  +3.738 (−0.09); −consensus +3.538 (−0.29); −reverse rank +3.505 (−0.32); −ensembled refinement map +3.513 (−0.31);
  −cycle-penalised refinement +3.698 (−0.13); −ARM_PAIR +3.675 (−0.15); base refinement (W5 β.01) +2.949 (−0.87);
  −Déjà Vu cycle (λ 0) +3.310 (−0.51).
- No flip components at all (NOM_ENS 0, LFL 0, REF_ENS 0): dev +2.813 (−1.01 vs seq14).
- REF_UP (bilinear up-sampled refinement map, soft-argmax on finer grid) u2 +3.239, u4 +3.386, u4 β.02 +3.653, u4 β.01 +3.739, u4 β.005 W1 +3.443 vs seq14 +3.823 → negative.
- T3 with a mixed pool (top-20 patches + 10 NMS peaks): pool oracle @.10 97.5 vs 94.6 (but @.01 31.4 vs 35.2); A/B B-set mean 49.57 vs 49.83 → negative.
- ENS_MIN (soft-AND disagreement penalty between the two query views) .1/.25/.5 → +3.832/+3.737/+3.309: negative.
- NEW: per-pair test-time channel re-weighting (decode_lib/tta.py; label-free): w ∈ R^D (softplus, init 1) maximising
  the dual-softmax mutual-match confidence (τ .05) of the masked source(arm 0)/target descriptors, Adam lr .05; applied to
  both source arms and the target, then the seq-14 decoder. Subset dev (aeroplane, cat, chair, horse; seq14 = +3.386):
  20 steps reg 1 +3.529, reg .1 +3.956, reg .01 +4.245, reg 0 +4.377; 50 steps reg .1 +3.074; reg 0: 10 steps +4.095,
  30 steps +4.039.
- seq 15 hypothesis [T1, training-free]: seq14 + TTA (20 steps, reg 0) → dev ≈ +4.3–4.8.
- seq 15 → kept: dev +4.714 (+0.89), val +4.657 (+1.07 vs seq14). Decode 58 ms/pair.
- TTA variants (4-cat subset; default 20 steps/τ .05/lr .05/arm0/dsmax = +4.377): pair's arm +4.369; both arms in the loss +4.129; τ .02 +3.844; τ .1 +4.200; lr .1 10 steps +4.219; entropy objective +3.938; soft-cycle objective +4.279 → default kept.
- TTA low-rank adapter (rank 8/32) collapses (dev −35.8/−28.9: degenerate mutual-confidence solution); 40 steps lr .025 +4.257 → diagonal 20 steps kept.
- Launched T3 tables on TTA-adapted descriptors (T3_TTA=20) for dev + trn_train (overnight).
- seq15 finalist so far: TEST +4.378 (74.16/59.74/14.15); CleanDIFT val +4.593 (seq14 +3.463); DIY-SC val +2.355 (seq14 +2.917 → TTA HURTS the supervised DIY-SC features by −0.56; still +2.1 over dv).
- seq15 finalist: SPair-U +1.789 (seq14 +1.237), Pascal +3.977 (+3.590), AP-10K +3.472 (+2.709).
- TTA ablation (dev): dv + TTA +1.593 (dv +0.777); dv W2β.03 + TTA +2.696 (dvW2 +1.767) → TTA adds ≈ +0.8–0.9 on its own, roughly additively.
- seq 16 hypothesis [T3]: T3 (x2 feats) trained on TTA-adapted descriptors (tables T3_TTA=20), decoded after TTA; A/B: B-set mean 50.29 vs 49.83 (non-TTA T3).
- TTA weight sharing (4-cat subset, per-channel = +4.377): 64 groups +3.093, 512 groups +3.431, separate src/trg weights +4.155 → per-channel shared weights kept.
- seq16 finalist (T3+TTA): TEST +5.988 (76.06/62.00/14.83); CleanDIFT val +5.857; DIY-SC val +2.815; SPair-U +2.029; Pascal +3.894; AP-10K +4.196.
- TTA_HAND (TTA term making a patch and its horizontal mirror dissimilar (+) / similar (−)) +.5 −0.957, +2 −1.415, −.5 +3.281 (subset; default +4.377) → negative: the handedness-sensitive channels are not semantic.
- Leave-one-out ablation of seq15 (dev +4.714): −all flip-ensemble +3.463 (−1.25); base refinement W5 β.01 +3.596
  (−1.12); −Déjà Vu cycle +4.042 (−0.67); −consensus +4.281 (−0.43); −reverse rank +4.491 (−0.22); −ARM_PAIR +4.520
  (−0.19); −cycle-penalised refinement (REF_CYC 0) +4.712 (0.00) → with TTA, REF_CYC no longer helps.
- seq 17 hypothesis (simplification): seq15 without REF_CYC — equal dev, simpler.
- seq 17 → kept by the simplicity rule: dev +4.712 (= seq15), val +4.563 (seq15 +4.657, −0.09 within noise).
- Error types (dev, % of kpts): seq17 TF+TTA wrong 27.08 | mirror 7.05 other 11.17 near 4.71 far 4.14; seq16 T3+TTA 25.16 | 6.50 / 10.85 / 4.22 / 3.58 → TTA does not touch mirror errors either.
- TTA_QW (extra query-row confidence term) 1/3 → +2.977/+2.838 (subset; base +4.366): strongly negative.
- TTA prior (θ averaged over 720 unlabeled trn_train pairs, runs/tta_prior.npy): fixed global weights, no per-pair steps +3.848 (subset; no TTA ≈ +3.39, per-pair TTA +4.366) → about half of the TTA gain is a global channel re-weighting; prior as init for per-pair TTA +4.294 (no gain).
- PARTNER_EST (estimate the mirror partner from features, excluding the self-mirror within r; forced on dev to simulate perm-free data) r3 +1.385, r6 +1.445 vs no flip-ensemble +3.277, true perm +4.366 (subset) → estimated partners are wrong and harmful; the flip-ensemble needs the annotated permutation.
- TTA_SEM (source-side semantic contrast in the TTA: query vs its flipped semantic view (+) vs self-mirror / own-arm partner (−)) .1/.5/2 → +2.703/+2.462/+2.440 (subset; base +4.366): strongly negative (few views → over-specialised weights).
