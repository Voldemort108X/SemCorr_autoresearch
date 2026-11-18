# Idea backlog (agent-maintained; seed list is in program.md §5)

Format: `- [track] idea — status (untried / running / seq N: Δdev, kept|discarded) — note`

## T2 (global, per pair) — main bet
- [T2] global affine from MNN anchors (RANSAC) → residual penalty μ·resid/w added to Déjà Vu score — untried
- [T2] MLS (moving least squares) local affine from MNN anchors around each query → residual penalty — untried
- [T2] anchors = MNN ∩ cycle-consistent Déjà Vu matches of all masked source patches; IRLS — untried
- [T2] joint decoding of the pair's keypoints: pairwise relative-geometry MRF over pools (ICM / exact on small pools) — untried
- [T2] dense flow: decode all masked source patches, median-smooth displacement field, read keypoints — untried
- [T2] global orientation (handedness) from anchors: sign of affine det / mirror-score, break mirror ties — untried

## T1 (per keypoint)
- [T1] soft cycle (expected reverse location under softmax column) — untried
- [T1] mirror test with the other source arm + perm — untried
- [T1] 3×3 source neighbourhood votes for cycle — untried

## T3 (learned)
- [T3] listwise MLP over pool cues (S, rank, d_cyc, anchor residual, mirror score) trained on trn_train — untried

## T4 (pool/refine)
- [T4] K_max / τ sweep with the better re-ranker — untried
