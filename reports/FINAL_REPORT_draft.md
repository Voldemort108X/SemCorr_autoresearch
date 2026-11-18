# FINAL REPORT (draft, in progress) — decoders beyond Déjà Vu

Status: DRAFT written on day 1; numbers are filled from `results.tsv` / `runs/*/metrics.json` and will be re-checked
before the final version (2026-11-23).

## 1. Method (training-free decoder, "DV++", current best = seq 15 = seq 14 + test-time adaptation)

**Per-pair test-time channel re-weighting (TTA, label-free).** Before decoding, learn a diagonal weight w ∈ R^D_{>0}
(w = softplus(θ + 0.5413), θ = 0 → w = 1) shared by all source/target descriptors of the pair, maximising the dual-
softmax mutual-match confidence of the masked source (original arm) and target descriptors:
  max_θ  mean_i max_j P_ij + mean_j max_i P_ij,  P = softmax_row(S_w/τ) ⊙ softmax_col(S_w/τ),
  S_w = norm(F_s ⊙ w) norm(F_t ⊙ w)ᵀ, τ = 0.05, 20 Adam steps (lr 0.05), no regulariser (early stopping by the step
count). All descriptors (both source arms, target) are re-weighted and re-normalised, then decoded as below.
Ablation (dev): +0.82 on Déjà Vu alone, +0.93 on Déjà Vu with our refinement, +0.89 on the full decoder (seq 14 → 15);
grouping channels (64/512 groups), separate source/target weights, a low-rank adapter (collapses), other objectives
(entropy, soft cycle) or more steps are worse. Cost ≈ 45 ms/pair on a 4090.


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

### 3.1 SPair-71k (TLR features), per-keypoint PCK@0.10/0.05/0.01, selected on dev (SPair trn subset)
| decoder | track | dev Δ | val Δ | test PCK | test Δ |
|---|---|---|---|---|---|
| top-1 (base refinement W5 β.01) | – | 0 | 0 | 69.78/54.09/11.04 | 0 |
| Déjà Vu λ0.5 (paper) | – | +0.78 | +0.89 | 71.22/54.98/11.19 | +0.82 |
| top-1, refinement W2 β.03 | – | +0.41 | +0.25 | 69.32/54.48/12.25 | +0.38 |
| Déjà Vu, refinement W2 β.03 | – | +1.77 | +1.61 | 71.28/56.26/12.62 | +1.75 |
| DV++ w/o TTA (seq 14) | training-free | +3.82 | +3.58 | 73.81/58.90/13.34 | +3.71 |
| **DV++ (seq 17)** | training-free | **+4.71** | **+4.56** | **74.08/59.64/14.15** | **+4.32** |
| DV++ + REF_CYC (seq 15) | training-free | +4.71 | +4.66 | 74.16/59.74/14.15 | +4.38 |
| T3 re-ranker on DV++ pool, no TTA (seq 9) | learned | (+5.31) | +4.97 | 75.50/61.05/14.05 | +5.23 |
| **T3 + TTA (seq 16)** | learned | (+6.27) | **+6.09** | **76.06/62.00/14.83** | **+5.99** |
| oracle top-5 / top-30 (base refinement) | – | | | 77.11/59.68/12.23 / 87.46/70.27/15.86 | |
Dev numbers of learned decoders in parentheses: dev shares images with their training pairs.

### 3.2 Transfer (val; all settings fixed from TLR dev) and generalisation (test)
| | CleanDIFT val | DIY-SC val | SPair-U | PF-Pascal (.10/.05/.15) | AP-10K CF |
|---|---|---|---|---|---|
| Déjà Vu | +0.70 | +0.24 | +0.51 | 89.44/75.93/93.70 (+2.57) | +0.56 |
| DV++ w/o TTA (seq 14) | +3.46 | +2.92 | +1.24 | 90.31/77.55/94.28 (+3.59) | +2.71 |
| DV++ (seq 17) | +4.52 | +2.47 | +1.67 | (+3.83) | +3.56 |
| T3 (seq 9) | +4.88 | +3.66 | +0.98 | 90.85/77.42/94.99 (+3.96) | +3.46 |
| T3 + TTA (seq 16) | +5.86 | +2.82 | +2.03 | 90.64/77.38/95.03 (+3.89) | +4.20 |
Δ = mean ΔPCK vs top-1 over the three thresholds (PF-Pascal: .10/.05/.15 w.r.t. 840 px).
Observation: TTA helps every foundation-feature setting but hurts the supervised DIY-SC features (−0.5 vs seq 14).
