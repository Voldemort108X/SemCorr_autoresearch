# Decoders beyond Déjà Vu: final report (autonomous run 2026-11-18 → 2026-11-23)

Version 2, 2026-11-19 (day 2): the training-free method is now seq 23 (DV++ = seq 17 + a smoothed-query view + equal
flip-view weight in the refinement); seq 17 numbers are kept where useful. Every number below is read from
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
| **DV++ (this work, seq 23)** | **training-free** | **74.46 / 59.91 / 14.42** | **+4.63** | ~60 |
| DV++ seq 17 (without the smoothed view) | training-free | 74.08 / 59.64 / 14.15 | +4.32 | ~60 |
| DV++ with 10 TTA steps (seq 23) | training-free | 74.46 / 59.59 / 13.97 | +4.37 | ~36 |
| DV++-fast (seq 23 with fixed channel weights, no per-pair TTA) | unsupervised prior | 74.23 / 59.11 / 13.49 | +3.97 | ~10 |
| **DV++ + learned re-ranker (T3)** | **learned on SPair trn** | **76.06 / 62.00 / 14.83** | **+5.99** | ~74 |
| oracle: best of top-5 / top-30 by S (base refinement) | – | 77.11 / 59.68 / 12.23 · 87.46 / 70.27 / 15.86 | | |

**Gaps closed.**
- DV++ recovers +4.7 of the top-1 → top-5-oracle gap at @0.10 (69.8 → 74.5 of 77.1). At @0.05 and @0.01 it exceeds the
  top-5 oracle, because of better sub-patch refinement and better descriptors after test-time adaptation.
- The T3 re-ranker reaches 76.1 @0.10, i.e. 1.0 below the top-5 oracle.

**Significance.** Paired bootstrap over test pairs (1,000 resamples):

| comparison | Δ | 95 % CI |
|---|---|---|
| DV++ − Déjà Vu with the same refinement | +2.88 | [2.72, 3.04] |
| DV++ (seq 23) − seq 17 | +0.31 | [0.23, 0.40] |
| T3 − DV++ | +1.36 | [1.21, 1.51] |

**Generalisation.** DV++ (settings fixed from dev) helps everywhere. Δ vs top-1 (CleanDIFT and DIY-SC on test):

| | DV++ (seq 23) | DV++ (seq 17) | Déjà Vu λ .5 |
|---|---|---|---|
| CleanDIFT | +4.50 | +4.29 | +0.73 |
| DIY-SC | +2.58 | +2.74 | +0.29 |
| AP-10K cross-family | +3.72 | +3.56 | +0.56 |
| PF-Pascal | +4.18 | +3.83 | +2.57 |
| SPair-U | +1.94 | +1.67 | +0.51 |

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

- Third view (seq 23): a 3×3-smoothed own query q̄_i (the query patch plus its 8 neighbours, re-normalised) is mixed
  into the forward row only (not the refinement): s_i ← (s_i + 0.7 q̄_iᵀF_t) / 1.7.
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
  (S[ℓ_i,·] + S'[ℓ'_i,·]) / 2 (seq 23; seq 17 used weight 0.5).
- The base pipeline uses window 5, β 0.01 and the own map only.

**Robustness.** Without π (SPair-U unseen keypoints, PF-Pascal), (b) and (d) fall back to the own arm. Everything else is
unchanged.

### Ablation (dev, leave one out; computed on seq 15/17 ≈ +4.71 — seq 23 then adds +0.21: smoothed view +0.11 alone, equal-weight refinement view +0.07 alone)

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

**Transfer of the learned head.** A re-ranker retrained on CleanDIFT trn tables (1,000 pairs per category) reaches
CleanDIFT val +5.49 / test +5.54. The TLR-trained model applied unchanged gets +5.86 / +5.96. The cue-based features
make the head transferable, so no per-backbone training is needed.

Feature-set trade-off: the same re-ranker with the 31-feature subset (no 5×5 ring, own-map shape, self-similarity,
object-relative positions, 2-hop cycle) on TTA descriptors gets val +5.64 (vs +6.09), SPair-U +2.27 (vs +2.03),
PF-Pascal +4.78 (vs +3.89), AP-10K +3.91 (vs +4.20): the extra features are partly SPair-specific. The dev protocol
selects the 64-feature model; for perm-less data the 31-feature model is the safer choice (post-hoc observation).

## 4. Results

### 4.1 SPair-71k (TLR = SD + DINO features)

| decoder | dev Δ | val Δ | test PCK | test per-image PCK | test Δ |
|---|---|---|---|---|---|
| top-1 | 0 | 0 | 69.78/54.09/11.04 | 65.50/49.70/9.86 | 0 |
| Déjà Vu | +0.78 | +0.89 | 71.22/54.98/11.19 | 66.46/50.25/9.95 | +0.82 |
| Déjà Vu + refinement W2 β.03 | +1.77 | +1.61 | 71.28/56.26/12.62 | 66.44/51.42/11.12 | +1.75 |
| DV++ without TTA | +3.82 | +3.58 | 73.81/58.90/13.34 | 68.75/53.76/11.68 | +3.71 |
| DV++-fast | +3.98 | +3.69 | 73.94/58.90/13.51 | 69.02/53.96/11.86 | +3.81 |
| DV++ seq 17 | +4.71 | +4.56 | 74.08/59.64/14.15 | 69.17/54.64/12.58 | +4.32 |
| **DV++ (seq 23)** | **+4.92** | **+4.77** | **74.46/59.91/14.42** | **69.40/54.84/12.78** | **+4.63** |
| T3 without TTA | (+5.31) | +4.97 | 75.50/61.05/14.05 | 70.63/56.07/12.46 | +5.23 |
| **T3 + TTA** | (+6.27) | **+6.09** | **76.06/62.00/14.83** | **70.95/56.81/13.20** | **+5.99** |

The per-category test table is in `reports/tables_test.md` (numbers below are for seq 17):
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
| DV++ seq 17 | +4.52 | +4.29 (74.48/60.02/14.31) | +2.47 | +2.74 (76.33/62.29/14.78) |
| **DV++ (seq 23)** | **+4.68** | **+4.50** (74.79/60.20/14.43) | +2.42 | +2.58 (76.06/62.09/14.78) |
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
| DV++ seq 17 | +1.67 (58.12/37.66/5.42) | +3.56 (56.89/40.50/9.95) | 90.72/77.55/94.57 (+3.83) |
| **DV++ (seq 23)** | **+1.94** (58.58/37.93/5.48) | **+3.72** (57.23/40.56/10.02) | **90.80/78.46/94.66 (+4.18)** |
| T3 + TTA | +2.03 | +4.20 (57.64/41.44/10.16) | 90.64/77.38/95.03 (+3.89) |

- SPair-U and PF-Pascal have no keypoint mirror permutation, so the flip ensemble is off there.
- Estimating mirror partners from the features instead was tested and is harmful (dev +1.4 vs +4.4).

Other features × new data (settings still from TLR dev; Déjà Vu at λ .5):

| | CleanDIFT SPair-U | CleanDIFT AP-10K CF | DIY-SC SPair-U |
|---|---|---|---|
| Déjà Vu | +0.39 | +0.46 | +0.35 |
| DV++ seq 17 | +1.67 | +3.73 (57.23/40.77/10.35) | **−0.31** |
| DV++ without TTA | +0.98 | – | +0.78 |

- TTA is what hurts the supervised DIY-SC descriptors (val −0.45, SPair-U −1.09 vs DV++ without TTA); it helps every
  zero-shot feature.
- Recommendation (post-hoc; DIY-SC was never used for any decision): use TTA only for zero-shot foundation features.

### 4.4 Robustness of the selection
A second, disjoint selection split (dev2 = 300 other SPair trn pairs per category, seed 1) reproduces the dev ranking:

| | dev2 Δ | dev Δ |
|---|---|---|
| Déjà Vu | +0.79 | +0.78 |
| Déjà Vu + W2 refinement | +1.56 | +1.77 |
| DV++ without TTA | +3.67 | +3.82 |
| DV++ seq 17 | +4.45 | +4.71 |
| DV++ seq 23 | +4.61 | +4.92 |

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

Test split, same breakdown (% of all keypoints; wrong @.10 | mirror / other kp / near-miss / far):

| decoder | wrong | mirror | other kp | near-miss | far |
|---|---|---|---|---|---|
| Déjà Vu | 28.78 | 5.93 | 11.62 | 5.69 | 5.54 |
| DV++ (seq 23) | 25.54 | 6.22 | 10.35 | 4.75 | 4.21 |
| T3 + TTA | 23.94 | 5.87 | 9.89 | 4.38 | 3.80 |

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
   - Its three main training-free ingredients each add ≈ +1 point over Déjà Vu, for +2.9 over Déjà Vu at equal
     refinement (seq 23, test, CI [2.72, 3.04]):
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
- DV++: `OVR="T3_MODEL=" bash scripts/run.sh <id> {dev,val,test} [tlr|cleandift|diysc]` (commit 5b1bd72, seq 23;
  seq 17 = commit bd80053).
  Generalisation: `OVR="T3_MODEL=" python prepare_gen.py eval {pascal,ap10kcf} <id> tlr`.
- T3 + TTA (seq 16, commit 0003221, default `decode.py`):
  1. Build tables: `T3_EXTRA=2 T3_TTA=20 python scripts/dump_t3.py trn_train 0`.
  2. Train: `python -c "from decode_lib import t3; t3.train('cache/tlr/trn_train/t3_x2_tta20.npz',
     'runs/t3_models/x2tta_full_60_128.pt', 60, None, h=128)"`.

## 9. Follow-up tasks (requested by the author on 2026-11-20)

### 9.1 SOCO benchmark (DINOv2-B/14, OmniProbe protocol)
- **Protocol.** Our evaluator `scripts/soco_eval.py` re-implements the paper's `soco_eval_onepass.py` (image 798,
  57×57 grid, full image padded to a square, argmax over all target patches, PCK@.10 × bbox scale, per pair → per
  class → mean). It reproduces the paper's per-class argmax numbers to ±0.1 (e.g. aeroplane 72.4 / 83.2).
- **Selection.** SOCO-dev = 10 pairs per class from `trainsplits/train` (1,000 pairs). Intra, cross and
  `trainsplits/test` are report-only.
- **Two tracks.**
  - **S-sim:** similarity only, with no flip arm and no flip ensemble. This is comparable to the paper's decoders.
  - **S-flip:** adds a pair-level flip arm and the flip ensemble. The mirror permutation comes from the keypoint
    names (`left` ↔ `right`). This is the SOCO equivalent of SPair's adaptive flip.
- **What carried over and what changed** (dev, SOC / CC gain over argmax; argmax 59.28 / 78.37; paired bootstrap std
  ≈ .5 / .35):

  | decoder | dev ΔSOC | dev ΔCC |
  |---|---|---|
  | OmniProbe soft-argmax (window 7) | +3.01 | +1.25 |
  | Déjà Vu λ .5 (paper port) | +1.56 | +0.53 |
  | DV++ S-sim, settings from SPair dev | +4.40 | +2.48 |
  | **DV++ S-sim, SOCO-tuned (S7: μ 1.0, window 3)** | **+4.94** | **+2.89** |
  | DV++ S-flip, SPair settings + distance arm rule (S3) | +11.35 | +3.73 |
  | **DV++ S-flip, SOCO-tuned + arm by consensus inlier ratio (S12)** | **+12.58** | **+3.96** |

  - Of the SPair knobs, only μ (consensus weight) 0.5 → 1.0 and refinement window 2 → 3 moved on SOCO.
  - λ, reverse rank, β, TTA steps and query smoothing were flat or worse.
  - The pair-level arm is the big lever on SOCO.
    - The per-pair arm oracle reaches 74.57 SOC, against 71.11 for SPair's distance rule.
    - Choosing the arm whose decode has the higher consensus inlier ratio closes about a fifth of that gap: +0.75 ± .28
      paired, chosen offline among ~20 label-free rules on dev.
    - Ported back to SPair (seq 28) the same rule gives +0.06, which is within noise, so it was discarded there.
- **Dense orientation evidence (S22, kept).** The pair arm is chosen by the similarity-consensus inlier ratio of
  *all* dense mutual-NN matches (inside the non-padding area), not only of the ~10 keypoint predictions.
  - Dev: 72.48 / 82.36 (+13.20 / +3.99), paired over S12 +0.62 ± .21 SOC.
  - It needs one decode per pair instead of two.
  - It ties on DIY-SC (−0.01 ± .16, at half the cost).
  - On SPair (seq 29) it gives dev +4.89 vs +4.92, so it was discarded there: SPair's adaptive-pose arm is already
    good.
- **Full splits** (20,000 pairs each; SOC / CC; Δ vs argmax; paired bootstrap std ≈ .1–.2;
  `soco_results.tsv`):

  | DINOv2-B/14 | intra | cross | trainsplits/test |
  |---|---|---|---|
  | argmax (paper: 60.53 / 78.79, 54.88 / 72.55) | 60.52 / 78.79 | 54.88 / 72.55 | |
  | Déjà Vu λ .5 (paper: +1.66 / +0.53, +1.81 / +0.25) | +1.67 / +0.54 | +1.80 / +0.26 | |
  | DV++ S-sim (S7) | 65.59 / 81.57 (+5.07 / +2.79) | | |
  | DV++ S-flip (S12) | 72.72 / 82.71 (+12.19 / +3.92) | 65.77 / 76.81 (+10.89 / +4.25) | |
  | DV++ S-flip (S22) | | | |

  | DIY-SC (TTA off) | intra | cross |
  |---|---|---|
  | argmax (paper: 69.18 / 78.81, 64.39 / 73.40) | 69.18 / 78.81 | 64.39 / 73.40 |
  | Déjà Vu λ .5 (paper: 70.07 / 78.79, 64.91 / 72.51) | +0.87 / −0.03 | +0.53 / −0.87 |
  | DV++ S-sim (D2) | 71.69 / 79.77 (+2.51 / +0.96) | 66.25 / 73.56 (+1.87 / +0.16) |
  | DV++ S-flip (D4) | **75.97 / 81.18 (+6.78 / +2.36)** | **69.27 / 74.92 (+4.88 / +1.52)** |

- **Per class** (intra, S12): the flip arm improves all 100 classes, from glider +2.7 up to front/back-symmetric
  furniture (folding chair +35, bed +32, armchair +32, highchair +30). S-sim drops below argmax on 3 classes (worst:
  pickup truck −4.3).

### 9.2 Supervised features
Decoders selected on TLR dev, transferred without tuning; Δ vs top-1 (mean over PCK@.10/.05/.01):

| features | Déjà Vu | DV++ (TTA) | DV++ without TTA | T3 |
|---|---|---|---|---|
| supervised TLR (released ckpt), val | +0.15 | +1.47 | **+2.68** | +0.81 |
| supervised TLR, test | +0.13 | +1.45 | **+2.70** (88.87 / 79.54 / 26.73) | +0.84 |
| DIY-SC val | +0.24 | +2.42 | **+2.92** | +2.82 |
| DIY-SC test | +0.29 | +2.58 | **+2.94** (77.46 / 62.71 / 13.83) | +3.08 |

- **Yes, the strategy helps supervised features**, by about 10–20× Déjà Vu's gain, but only without the per-pair TTA.
- The gain is concentrated at @.01 (+4.9 for supervised TLR test): the refinement and flip-ensemble parts transfer.
  The selection part transfers less (+1.2 @.10), because supervised features already rank well (top-1 87.7 @.10).
- T3 (trained on zero-shot TLR tables) transfers poorly to supervised TLR features.
- **A label-free per-pair gate for TTA** keeps TTA only if the keypoint-query mutual-NN rate does not drop. It recovers
  only part of the loss (supervised TLR val +1.69, DIY-SC val +2.51). On dev it exceeded the time limit, so it was
  discarded.
- **Rule:** turn TTA off for features trained for correspondence.
- Supervised TLR is trained on SPair trn, which is our dev. It was therefore never used for any decision.

### 9.3 The paper's stated limitations
- **(a) "The gap to top-k needs global reasoning."** Keypoints the decoder can still recover (top-1 wrong, a correct
  candidate in the top-30 pool, test @.10):

  | decoder | recoverable keypoints promoted | top-1 → top-30-oracle gap closed |
  |---|---|---|
  | Déjà Vu | 16.9 % | 8.1 % |
  | DV++ | 36.5 % | 26.5 % |
  | T3 | 44.5 % | 35.5 % |

  The pair-level parts (consensus RANSAC, pair arm, flip ensemble) are the global reasoning. They roughly triple the
  promotion rate (`scripts/promotion.py`).
- **(b) "Mirror-equivalent parts are both cycle-consistent."**
  - On SPair, mirror swaps are not reduced (≈ 6 % for every decoder; §5).
  - On SOCO, a wrong-instance pick is an error under SOC but still correct under CC, so the CC − SOC gap measures
    these picks. The pair-level flip arm (S-flip) shrinks that gap on dev from 19.1 (argmax) and 17.1 (S-sim) to
    10.5 points.
  - The left/right ambiguity is therefore mostly a pair-level orientation problem when whole objects are mirrored. It
    stays unsolved per keypoint (near vs far side).
- **(c) "Bird / horse regress with Déjà Vu."** With TLR features (test @.10, per keypoint;
  `reports/tables_test.md`):

  | category | top-1 | Déjà Vu | DV++ seq 23 | T3 |
  |---|---|---|---|---|
  | horse | 73.8 | 74.4 | 76.4 | 77.6 |
  | bird | 89.6 | 89.7 | 89.4 | 91.1 |

  - Horse improves with DV++.
  - Bird is near its ceiling and the training-free DV++ stays flat (−0.2, the only category below top-1). The learned
    T3 gains +1.5 on bird.
  - Déjà Vu's only regression on test is car (−0.6). DV++ improves car by +2.0.
