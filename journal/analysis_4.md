# Analysis after seq 4 (2026-11-18, dev TLR)

Data: cache/tlr/dev/pools.npz (Déjà Vu pool, cues, per-candidate GT error), runs/4_61c2572/err.npz.

## Where dv fails (per kpt @.10; wrong = 30.4 %)
- 57.8 % of wrong keypoints have a correct candidate in the top-p pool; 42.2 % are pool misses.
- Wrong predictions land on: the mirror counterpart 21.6 %, another annotated kp 41.9 %, near-miss (0.1–0.2) 18.4 %,
  far 18.1 %. Pool-oracle @.10 = 87.2 vs dv 69.6.
- Mirror-error share per category is highest for chair (16 %), person (13 %), horse (10 %), sheep / cat (9 %).

## What the cues can and cannot do
- Pair-level geometry (consensus-RANSAC similarity over keypoint pools, MNN affine, MLS, keypoint ICM/MRF, one-to-one)
  saturates at ≈ +0.35–0.4 over dv. GBM over all cues (2-fold on dev) +1.0 only: the per-candidate cues carry little.
- Mirror pairs: the 2-D x-order of a mirror pair (after the adaptive flip) agrees between source and target only for
  rigid / frontal categories (bottle, bus, train, tvmonitor, pottedplant ≥ 0.96); for side-view animals / aeroplane /
  bicycle / motorbike / person it is ≈ chance (0.44–0.60). The left/right confusion there is near-side vs far-side (depth),
  which no 2-D transform resolves → stop pushing 2-D handedness / ordering constraints.
- What worked: MORE EVIDENCE about the query, not better re-ranking logic. The flip-ensemble (similarity of the
  candidate to the mirrored counterpart part in the flipped source, which looks like the query part) adds +0.6 in the
  choice and +0.65 in the refinement map. → Pursue more "views" / better similarity maps (T4) and refinement.

## seq 4 vs dv per category (Δdev vs top1): gains in 17/18 categories; tvmonitor +6.4 (dv +1.8), pottedplant +3.7,
  bottle +3.3, cat +2.9; bus −0.05 (dv +0.33). Fixed/broken vs dv @.10: 1682 / 953.

## Refinement (window soft-argmax) is a large lever
- With the ensembled map: window 5 → +2.19, 4 → +2.50, 3 → +2.73, 2 → +2.81; β 0.015–0.02 better than 0.01.
  The base pipelines' window 5 / β 0.01 is far from optimal for this metric (mean of @.10/.05/.01).
- Fairness: report top1 and dv with the SAME tuned refinement as extra baselines.

## Next
1. Commit the tuned refinement (seq 5); measure top1/dv with the same refinement.
2. T3: listwise re-ranker on trn_train cue tables (cache building).
3. More query evidence: cycle evidence in both arms jointly, ensembled map for the cycle; sub-patch refinement.
