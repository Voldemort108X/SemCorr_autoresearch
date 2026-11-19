# Decoders beyond Déjà Vu: final report (autonomous run 2026-11-18 → 2026-11-23)

Version 1, written 2026-11-19 (day 2) after both tracks saturated. Every number below is read from
`runs/<id>/metrics.json` (row ids in `results.tsv`).

**Protocol.**
- **Selection.** Every design decision and hyper-parameter was selected on **dev** = 5,400 pairs sampled from SPair-71k
  **trn** (300 per category, seed 0). The metric is the paper's λ-selection criterion: the mean over
  PCK@{0.10, 0.05, 0.01} (per keypoint) of the gain over top-1.
- **Reporting only.** Val, test, transfer and generalisation numbers were never used for a decision. Every test number
  is "selected on dev (SPair trn subset)".
- **Learned decoder (T3).** It is trained on SPair-71k trn pairs only and is reported separately. Its dev numbers are
  optimistic, because dev shares images with its training pairs; val and test do not.

## 1. Summary

| SPair-71k test, PCK@.10/.05/.01 (per kpt) | track | test | Δ vs top-1 | ms/pair |
|---|---|---|---|---|
| top-1 (window 5, β .01) | – | 69.78 / 54.09 / 11.04 | 0 | – |
| Déjà Vu λ 0.5 (paper) | training-free | 71.22 / 54.98 / 11.19 | +0.82 | ~3 |
| Déjà Vu + our refinement (window 2, β .03) | training-free | 71.28 / 56.26 / 12.62 | +1.75 | ~3 |
| **DV++ (this work)** | **training-free** | **74.08 / 59.64 / 14.15** | **+4.32** | ~60 |
| DV++-fast (fixed channel weights, no per-pair TTA) | unsupervised prior | 73.94 / 58.90 / 13.51 | +3.81 | ~8 |
| **DV++ + learned re-ranker (T3)** | **learned on SPair trn** | **76.06 / 62.00 / 14.83** | **+5.99** | ~74 |
| oracle: best of top-5 / top-30 by S (base refinement) | – | 77.11 / 59.68 / 12.23 · 87.46 / 70.27 / 15.86 | | |

**Gaps closed.**
- DV++ recovers +4.3 of the top-1 → top-5-oracle gap at @0.10 (69.8 → 74.1 of 77.1). At @0.05 and @0.01 it exceeds the
  top-5 oracle, because of better sub-patch refinement and better descriptors after test-time adaptation.
- The T3 re-ranker reaches 76.1 @0.10, i.e. 1.0 below the top-5 oracle.

**Significance.** Paired bootstrap over test pairs (1,000 resamples):

| comparison | Δ | 95 % CI |
|---|---|---|
| DV++ − Déjà Vu with the same refinement | +2.57 | [2.40, 2.74] |
| T3 − DV++ | +1.67 | [1.53, 1.80] |

**Generalisation.** DV++ (settings fixed from dev) helps everywhere. Δ vs top-1 (CleanDIFT and DIY-SC on test):

| | DV++ | Déjà Vu λ .5 |
|---|---|---|
| CleanDIFT | +4.29 | +0.73 |
| DIY-SC | +2.74 | +0.29 |
| AP-10K cross-family | +3.56 | +0.56 |
| PF-Pascal | +3.83 | +2.57 |
| SPair-U | +1.67 | +0.51 |

## 2. Method: DV++ (training-free)

Notation as in the paper: S = F_s F_tᵀ on the 60×60 grid, restricted to the dilated object masks; query patch ℓ_i;
grid width w = 60.

DV++ uses only quantities the base pipeline already computes:
- both source arms (the original and the horizontally flipped source, already extracted for the adaptive-pose step);
- the adaptive-pose decision;
- the dataset's keypoint mirror permutation π (already used by that step).

**(a) Per-pair test-time channel re-weighting (TTA, label-free).**
- Learn w = softplus(θ + 0.5413) ∈ R^D_{>0} with θ initialised to 0, so w starts at 1.
- Objective: maximise the dual-softmax mutual-match confidence of the masked source (original arm) and target
  descriptors,

  J(θ) = mean_i max_j P_ij + mean_j max_i P_ij,
  P = softmax_row(S_w/τ) ⊙ softmax_col(S_w/τ),
  S_w = norm(F_s ⊙ w) · norm(F_t ⊙ w)ᵀ,  τ = 0.05.

- Optimisation: Adam, lr 0.05, 20 steps (the step count is the only regulariser).
- Then re-weight and re-normalise every descriptor of the pair (both source arms and the target) and decode as below.

**(b) Second view of every query (flip ensemble).**
- In the flipped source, the mirror image of the counterpart part π(i) looks like part i.
- Let ℓ'_i be the flipped-arm patch of π(i) (column 59 − x).
- When π(i) is annotated in the source, use the forward row

  s_i(m) = (S[ℓ_i, m] + 0.75 S'[ℓ'_i, m]) / 1.75.

- Nomination is unchanged from Déjà Vu, applied to s_i: top-p with τ = 0.1, p = 0.9, K_max = 30.

**(c) Score of candidate m.**

  R_i(m) = s_i(m) − λ d_cyc(i,m)/w − λ' d'_cyc(i,m)/w − μ min(r_i(m), c)/w − ρ log(1 + rr_i(m))

| term | meaning | value |
|---|---|---|
| d_cyc | Déjà Vu cycle distance | λ = 0.5 |
| d'_cyc | the same cycle measured in the flipped arm against ℓ'_i | λ' = 0.25 |
| r_i | residual to a pair-level similarity transform (see below) | μ = 0.5, c = 5 patches |
| rr_i | reverse rank of the query in column m of S | ρ = 0.02 |

The similarity transform behind r_i:
- It is fitted by RANSAC over all query keypoints' candidate pools.
- Hypotheses are built from 2 points, each drawn from a keypoint's top-5 candidates by the Déjà Vu score.
- Support is the rank-discounted inlier count within 6 patches; the inliers are then refit with an affine model.

**(d) Arm.** Every keypoint of a pair is decoded in the arm chosen by the base adaptive-pose step; arm-1 queries are
rebuilt from the source as ℓ'_i. This replaces the base per-keypoint rule, which also depends on target visibility.

**(e) Refinement.**
- Window soft-argmax around the chosen patch: window 2 (5×5 patches), β = 0.03, on the map
  (S[ℓ_i,·] + 0.5 S'[ℓ'_i,·]) / 1.5.
- The base pipeline uses window 5, β 0.01 and the own map only.

**Robustness.** Without π (SPair-U unseen keypoints, PF-Pascal), (b) and (d) fall back to the own arm. Everything else is
unchanged.

### Ablation (dev, leave one out; full DV++ ≈ +4.71)

| removed | dev | Δ |
|---|---|---|
| all flip-ensemble parts | +3.46 | −1.25 |
| refinement → base (window 5, β .01) | +3.60 | −1.12 |
| TTA | +3.82 | −0.89 |
| Déjà Vu cycle (λ = 0) | +4.04 | −0.67 |
| consensus residual | +4.28 | −0.43 |
| reverse rank | +4.49 | −0.22 |
| pair-level arm | +4.52 | −0.19 |

Reading the TTA ablation in more detail:
- **Selection and refinement.** TTA helps both: adapting for selection only gives +4.28, for refinement only +4.26.
- **On Déjà Vu alone.** TTA adds +0.82, or +0.93 with our refinement.
- **Global vs per-pair.** About half of the TTA gain is a global channel re-weighting. Weights averaged over 720
  unlabeled trn pairs give "DV++-fast": dev +3.98. They down-weight SD s4 / s5 (mean w 0.78 / 0.62) and up-weight DINO
  (1.10).
- **Lite variant.** TTA + flip ensemble + Déjà Vu cycle + refinement only: dev +3.83.

## 3. Learned re-ranker (track T3; trained on SPair-71k trn only)

**Model.**
- A DeepSets listwise MLP (hidden 128) over the 30 candidates of the DV++ pool, computed on TTA-adapted descriptors.
- Training objective: the expected metric gain Σ_k softmax(s)_k · mean_t 1[err_k ≤ t].
- Data: 334k keypoints from trn_train pairs (dev pairs excluded); 60 epochs, 45 s on a 4090.

**64 category-agnostic features.**
- The DV++ score terms.
- Soft cycle, 2-hop cycle and reverse rank.
- Local 3×3 / 5×5 shape of the refinement map and the own map.
- Competition: similarity of the candidate to the other query keypoints and to the mirror partner.
- Query self-similarity: to its mirror partner and to far-away source patches.
- Object-relative positions of the query and the candidate (in their mask boxes).

**Model selection.** Capacity was chosen with an image-disjoint protocol (train on one half of the trn images, select
on dev pairs inside the other half). Without it, bigger models looked +1.2 better on dev purely through memorising the
shared images.

**Stability.** A second seed gives val +5.96 vs +6.09.

## 4. Results

### 4.1 SPair-71k (TLR = SD + DINO features)

| decoder | dev Δ | val Δ | test PCK | test per-image PCK | test Δ |
|---|---|---|---|---|---|
| top-1 | 0 | 0 | 69.78/54.09/11.04 | 65.50/49.70/9.86 | 0 |
| Déjà Vu | +0.78 | +0.89 | 71.22/54.98/11.19 | 66.46/50.25/9.95 | +0.82 |
| Déjà Vu + refinement W2 β.03 | +1.77 | +1.61 | 71.28/56.26/12.62 | 66.44/51.42/11.12 | +1.75 |
| DV++ without TTA | +3.82 | +3.58 | 73.81/58.90/13.34 | 68.75/53.76/11.68 | +3.71 |
| DV++-fast | +3.98 | +3.69 | 73.94/58.90/13.51 | 69.02/53.96/11.86 | +3.81 |
| **DV++** | **+4.71** | **+4.56** | **74.08/59.64/14.15** | **69.17/54.64/12.58** | **+4.32** |
| T3 without TTA | (+5.31) | +4.97 | 75.50/61.05/14.05 | 70.63/56.07/12.46 | +5.23 |
| **T3 + TTA** | (+6.27) | **+6.09** | **76.06/62.00/14.83** | **70.95/56.81/13.20** | **+5.99** |

The per-category test table is in `reports/tables_test.md`:
- DV++ improves over Déjà Vu at @0.10 in 17 of 18 categories; bird is unchanged (±0.0).
- Large gains: tvmonitor +9.7, pottedplant +7.4, bottle +4.7, chair +4.1, sheep +3.5.
- Smallest gains: aeroplane +0.7, bus +0.4, bicycle +0.9, train +1.0.

### 4.2 Transfer to other features (settings fixed from TLR dev)

Δ vs top-1 of the same features; Déjà Vu at λ 0.5.

| | CleanDIFT val | CleanDIFT test | DIY-SC val | DIY-SC test |
|---|---|---|---|---|
| top-1 test PCK | | 70.29/54.52/11.12 | | 74.44/58.65/12.10 |
| Déjà Vu | +0.70¹ | +0.73 | +0.24¹ | +0.29 |
| DV++ without TTA | +3.46 | – | +2.92 | – |
| **DV++** | **+4.52** | **+4.29** (74.48/60.02/14.31) | +2.47 | **+2.74** (76.33/62.29/14.78) |
| T3 + TTA | +5.86 | +5.96 (76.47/62.38/14.97) | +2.82 | +3.08 (76.61/62.92/14.91) |

¹ Val values use the base methods' own λ (3.0 / 1.0).

TTA helps the zero-shot features. It *hurts* the supervised DIY-SC features by ≈ 0.5 (val) compared with DV++ without
TTA; DIY-SC was never used for selection, so this is reported as a limitation.

### 4.3 Generalisation (test; settings fixed from SPair dev)

| | SPair-U (unseen kps) | AP-10K cross-family | PF-Pascal (.10/.05/.15 @ 840 px) |
|---|---|---|---|
| top-1 | 56.22/35.48/4.49 | 53.30/36.26/7.10 | 86.50/73.57/91.30 |
| Déjà Vu | +0.51 | +0.56 | 89.44/75.93/93.70 (+2.57) |
| Déjà Vu + W2 refinement | +0.50 | +1.30 | +2.69 |
| **DV++** | **+1.67** (58.12/37.66/5.42) | **+3.56** (56.89/40.50/9.95) | **90.72/77.55/94.57 (+3.83)** |
| T3 + TTA | +2.03 | +4.20 (57.64/41.44/10.16) | 90.64/77.38/95.03 (+3.89) |

- SPair-U and PF-Pascal have no keypoint mirror permutation, so the flip ensemble is off there.
- Estimating mirror partners from the features instead was tested and is harmful (dev +1.4 vs +4.4).

Other features × new data (settings still from TLR dev; Déjà Vu at λ .5):

| | CleanDIFT SPair-U | CleanDIFT AP-10K CF | DIY-SC SPair-U |
|---|---|---|---|
| Déjà Vu | +0.39 | +0.46 | +0.35 |
| DV++ | +1.67 | +3.73 (57.23/40.77/10.35) | **−0.31** |
| DV++ without TTA | +0.98 | – | +0.78 |

- TTA is what hurts the supervised DIY-SC descriptors (val −0.45, SPair-U −1.09 vs DV++ without TTA); it helps every
  zero-shot feature.
- Recommendation (post-hoc; DIY-SC was never used for any decision): use TTA only for zero-shot foundation features.

## 5. Failure analysis
- **Remaining errors (dev, DV++, wrong @0.10 = 27.1 % of keypoints).** Breakdown, as % of all keypoints:

  | error type | share |
  |---|---|
  | lands on the mirror counterpart | 7.0 % |
  | lands on another annotated keypoint | 11.2 % |
  | near-miss (0.1–0.2) | 4.7 % |
  | far | 4.1 % |

- **Mirror swaps** are untouched by every decoder, including T3 and TTA (Déjà Vu 6.6 %).
  - For Déjà Vu's mirror errors the correct part is in the top-30 pool only 18 % of the time.
  - When it is in the pool, every appearance cue (cycle, reverse rank, flip view, context) prefers the wrong side.
  - The 2-D offset of a mirror pair agrees between source and target (after the adaptive flip, with a GT-fitted
    similarity) for rigid / frontal categories, but is ≈ chance for side-view animals, bicycles and birds. The real
    difference there is near side vs far side.
- **Mirror pairs both visible (dev, n = 8,975).** Both correct 63 % (Déjà Vu 57 %), swapped 4 %, *collided* (both
  predictions on one spot) 13 % (Déjà Vu 9 %). Gains are larger on lateral keypoints (+4.8 @0.10 over top-1) than on
  central ones (+2.7).

## 6. Negative results (dev; details in journal/DAY1.md)
Within noise or harmful:
- **Pair-level 2-D geometry:** MNN affine, MLS, dense-flow prior, keypoint ICM/MRF, one-to-one assignment, iterated
  consensus, adaptive μ, consensus in the refinement map.
- **Mirror handling:** mirror-pair orientation swap and de-collision, a handedness direction from 4 source views, TTA
  handedness / semantic-contrast terms, estimated partners, NMS-peak pools.
- **Cues:** soft cycle (linear), context agreement, relational profile, competition, query smoothing, SD/DINO block
  weights, view-disagreement penalty.
- **Pool and refinement:** K, τ, quadratic / up-sampled / Gaussian-window refinement, arm re-selection by post-TTA
  confidence.
- **TTA variants:** channel groups, separate source/target weights, low-rank adapters (they collapse), entropy or
  cycle objectives, ensembles, mixing with the original similarity, no mask, per-block renormalisation, query
  weighting, self-training with pseudo-labels, sub-sampled objective.
- **T3 variants:** joint attention across keypoints, two-arm pool, peak pool, perm-drop augmentation, dual features,
  bigger models (memorisation), CE loss.

## 7. Recommendation
1. **For the paper / a follow-up: DV++.**
   - Its three main training-free ingredients each add ≈ +1 point over Déjà Vu, for +2.6 over Déjà Vu at equal
     refinement (test, CI [2.40, 2.74]):
     - the flip-ensemble query, which reuses the second source arm the adaptive-pose step already computes;
     - label-free per-pair TTA of channel weights;
     - a tuned sub-patch refinement.
   - Consensus RANSAC and reverse rank are small but consistent extras.
   - All settings transfer unchanged to CleanDIFT, AP-10K and PF-Pascal.
   - Report the refinement change separately: it also lifts Déjà Vu itself (+0.9 test).
2. **The T3 re-ranker (+1.7 over DV++ on test) is cheap to train (< 1 min) and transfers to CleanDIFT and AP-10K.** It is
   a supervised component; present it as an optional learned head with the image-disjoint selection protocol.
3. **The open problem is the left/right (near/far side) ambiguity** (~25 % of the remaining errors). 2-D geometry and
   appearance cues cannot resolve it on these features. It likely needs 3-D-aware features or a learned pose prior.
4. **Caveats.**
   - TTA hurts supervised DIY-SC features slightly.
   - The flip ensemble needs a keypoint mirror permutation; without it the gains drop to +1–2.
   - DV++ costs ~60 ms/pair (TTA is latency-bound: 20 small optimisation steps). DV++-fast keeps most of the gain at
     ~8 ms/pair.

## 8. Reproduction
- Cache: `python prepare.py --method tlr --split dev,val,test` (plus `cleandift` / `diysc`).
  Generalisation sets: `python prepare_gen.py {spairu,pascal,ap10kcf} tlr`.
- DV++: `OVR="T3_MODEL=" bash scripts/run.sh <id> {dev,val,test} [tlr|cleandift|diysc]` (commit bd80053, seq 17).
  Generalisation: `OVR="T3_MODEL=" python prepare_gen.py eval {pascal,ap10kcf} <id> tlr`.
- T3 + TTA (seq 16, commit 0003221, default `decode.py`):
  1. Build tables: `T3_EXTRA=2 T3_TTA=20 python scripts/dump_t3.py trn_train 0`.
  2. Train: `python -c "from decode_lib import t3; t3.train('cache/tlr/trn_train/t3_x2_tta20.npz',
     'runs/t3_models/x2tta_full_60_128.pt', 60, None, h=128)"`.
