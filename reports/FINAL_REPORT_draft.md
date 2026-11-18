# FINAL REPORT (draft, in progress) — decoders beyond Déjà Vu

Status: DRAFT written on day 1; numbers are filled from `results.tsv` / `runs/*/metrics.json` and will be re-checked
before the final version (2026-11-23).

## 1. Method (training-free decoder, "DV++", current best = seq 14)

Notation as in the paper: S = F_s F_tᵀ on the 60×60 grid (masked, dilated object masks), query patch ℓ_i, grid width
w = 60. Besides the source arm used by the base method's adaptive pose step (original or horizontally flipped source,
chosen once per pair from the feature distance, unchanged), we use the *other* arm, which is already computed by the base
pipeline, and the dataset's keypoint mirror permutation π (also already used by the adaptive pose step).

**Second view of the query (flip ensemble).** In the horizontally flipped source the mirrored image of the counterpart
part π(i) looks like part i. Let ℓ'_i be the flipped-arm patch of π(i) (column 59 − x of its original patch) when π(i) is
annotated in the source. The forward row of query i becomes
  s_i(m) = (S[ℓ_i, m] + α S'[ℓ'_i, m]) / (1 + α),   α = 0.75,
(own row only if π(i) is not annotated). Candidate nomination (top-p, τ 0.1, p 0.9, K_max 30) uses s_i.

**Score.** For candidates m in the pool:
  R_i(m) = s_i(m) − λ d_cyc(i, m)/w − λ' d'_cyc(i, m)/w − μ min(r_i(m), c)/w − ρ log(1 + rr_i(m)),
with d_cyc the Déjà Vu cycle distance (own arm, λ = 0.5); d'_cyc the cycle distance in the other arm (back-NN of m in
the flipped source vs ℓ'_i, λ' = 0.25); r_i(m) the residual of m to a pair-level similarity transform T fitted by RANSAC
over all query keypoints' candidate pools (2-point hypotheses from each keypoint's top-5 by the Déjà Vu score, support =
rank-discounted inliers within 6 patches, affine refit on inliers; μ = 0.5, c = 5 patches); rr_i(m) the reverse rank of
the query among all masked source patches in column m (ρ = 0.02).

**Arm.** All keypoints of a pair are decoded in the arm chosen by the base adaptive-pose step (the flipped-arm query of
keypoint i is rebuilt from the source as ℓ'_i above), instead of the base per-keypoint rule that also depends on target
visibility.

**Refinement.** Window soft-argmax around the chosen patch on the map
  M_i = (S[ℓ_i, ·] + 0.5 S'[ℓ'_i, ·]) / 1.5 − 0.5 d_cyc(i, ·)/w,
window 2 (5×5 patches), β = 0.03 (base: window 5, β 0.01, own map).

All hyper-parameters were selected on dev (5,400 pairs drawn from SPair-71k trn, seed 0) with the paper's criterion
(mean ΔPCK over @0.10/0.05/0.01 vs top-1); val/test/transfer/generalisation numbers are reports only.

## 2. Learned re-ranker (track T3, reported separately; trained on SPair-71k trn pairs only)
DeepSets listwise MLP over the 30 candidates of the DV++ pool (64 category-agnostic features: the score terms above,
soft cycle, reverse rank, consensus residual, local 3×3 / 5×5 shape of the refinement and own maps, competition from
other query keypoints, query self-similarity, object-relative positions, 2-hop cycle), trained with the expected metric
gain Σ_k softmax(s)_k · mean_t 1[err_k ≤ t] on 334k trn keypoints (60 epochs, 46 s on a 4090). Model capacity was chosen
with an image-disjoint protocol (dev shares images with SPair trn; val/test do not).

## 3. Results (to be completed)
See `results.tsv`; tables generated with `scripts/tables.py`.
