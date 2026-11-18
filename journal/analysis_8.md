# Analysis after seq 8 (2026-11-18 ~12:00 UTC, dev TLR)

## Error types (% of ALL evaluated keypoints, wrong @.10) — scripts/error_types.py
| decoder | wrong | mirror kp | other kp | near-miss | far |
|---|---|---|---|---|---|
| dv (base refinement) | 30.42 | 6.56 | 12.75 | 5.60 | 5.51 |
| seq 7 training-free | 27.88 | 6.77 | 11.64 | 4.97 | 4.51 |
| seq 8 T3 | 26.63 | 6.71 | 11.15 | 4.75 | 4.03 |
→ every gain so far comes from other-kp / near / far errors. Mirror swaps (≈ 6.7 % of keypoints, 25 % of errors) are
untouched by ALL decoders, including the learned one.

## Why mirror errors resist
- For dv mirror errors the correct location is in the top-30 pool in only 18.5 % (pool = one blob on the wrong part).
- When it is in the pool, every appearance cue prefers the wrong one: cycle distance (correct has larger d_cyc in 71 %),
  reverse rank (68 %), soft cycle (77 %), flip-ensemble sim (correct higher in only 29 %), handedness (32 %), context.
  The mirror part of the target IS cycle-consistent with the query (appearance is symmetric).
- Diversifying the pool (NMS peaks, K_PEAK 10/20) without a discriminative cue: dev +2.75 / +2.68 vs +3.58 → worse.
- Geometry: with the ORACLE similarity transform (GT-fitted), the offset vector of a mirror pair points the same way in
  source and target in 89 % overall (rigid/frontal ≈ 1.0, cat .87, cow .92, aeroplane .87, chair .79, dog .79,
  sheep .77, person .72) but ≈ chance for horse .56, bird .54, bicycle .57 (side views: near/far side).
  The 2-D x-order alone is ≈ chance for animals (see analysis_4).
- Test now: mirror-pair orientation swap (post-hoc, RANSAC similarity on the decoder's own predictions).

## Next
- If the swap test fails: mirror errors need 3-D / pose reasoning beyond 2-D geometry; leave it as the documented
  limitation and spend effort on generalisation checks + T3 robustness.
