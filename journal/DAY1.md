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
