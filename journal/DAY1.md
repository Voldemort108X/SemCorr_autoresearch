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
