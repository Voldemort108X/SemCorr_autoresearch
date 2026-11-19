# Idea backlog (agent-maintained; seed list is in program.md §5)
Format: `- [track] idea — status — note`. Details/numbers in journal/DAY1.md.

## Kept (current pipeline)
- [T2] consensus RANSAC residual over keypoint pools — seq 1 kept (+0.36)
- [T1/T4] flip ensemble: mirrored-partner view in row (seq 4), score (seq 2), refinement map (seq 3) — kept (+1.0 total)
- [T4] refinement window 2 / β .03 — seq 5 kept (+1.06; also lifts dv → separate baseline)
- [T1] reverse-rank penalty — seq 7 kept (+0.33)
- [T4] pair-level arm for all keypoints (ARM_PAIR) — seq 11 kept (+0.11)
- [T1] per-pair test-time channel re-weighting (TTA, label-free) — seq 15/17 kept (+0.89)
- [T3] DeepSets listwise re-ranker (x/x2 features), on TTA features — seq 6/8/9/16 kept (val +6.09)

## Tried, not kept (all within noise or negative)
- [T2] MNN-affine, MLS, dense-flow prior, ICM/MRF, one-to-one, iterated consensus, adaptive μ, consensus in refinement
- [T2] mirror handling: orientation swap, de-collision, handedness direction (HAND2), TTA handedness/semantic terms,
  estimated partners without perm, NMS-peak pools
- [T1] soft cycle (linear), hand cue, context agreement, relational profile, competition, query smoothing (all/row),
  ENS_MIN, SD/DINO block weights, FL_ALIGN, RR_FL, ZROW, REF_CYC (no gain once TTA is on)
- [T4] K, τ, quadratic / up-sampled / Gaussian-window refinement, arm re-selection by confidence (seq 18)
- [TTA] grouped / separate / low-rank weights, other objectives, more/fewer steps, ensembles, mixing, no mask,
  block-norm, query weighting, global prior init, self-training with pseudo-labels
- [T3] joint attention across keypoints, two-arm pool, peak pool, perm-drop, bigger models (memorisation), dual
  features (seq 19), rc labels, CE loss

## Open (low expected value)
- [T3] method-specific re-rankers (CleanDIFT / DIY-SC trn) — reporting would no longer be plug-and-play
- [T1] occlusion / depth cues for mirror (near vs far side) — no concrete signal found
