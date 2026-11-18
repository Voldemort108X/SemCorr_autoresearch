# Analysis after seq 13 (2026-11-18 ~14:30 UTC)

## Finalists so far (selected on dev; Δ = mean over @.10/.05/.01 of ΔPCK vs top1 with the base refinement)
| | dev | val | TEST | CleanDIFT val | DIY-SC val | SPair-U test | PF-Pascal (.10/.05/.15) | AP-10K CF test |
|---|---|---|---|---|---|---|---|---|
| dv (paper) | +0.78 | +0.89 | +0.82 | +0.70 | +0.24 | +0.51 | +2.57 | +0.56 |
| dv + W2 β.03 refinement | +1.77 | +1.61 | +1.75 | – | – | +0.50 | +2.69 | +1.30 |
| seq 7 (TF) | +3.58 | +3.31 | +3.46 | +3.16 | +2.95 | +0.99 | +3.48 | +2.50 |
| **seq 11 (TF best)** | **+3.70** | **+3.39** | **+3.56** | **+3.27** | **+2.95** | **+0.99** | **+3.48** | **+2.68** |
| seq 8 (T3, x feats) | +4.75* | +4.56 | +4.79 | +4.37 | +3.14 | +1.45 | +4.67 | +3.17 |
| **seq 9 (T3 best, x2 feats)** | **+5.31*** | **+4.97** | **+5.23** | **+4.88** | **+3.66** | **+0.98** | **+3.96** | **+3.46** |
| seq 13 (T3 x2 + perm-drop) | – | +4.93 | – | – | – | +1.22 | +3.80 | +3.58 |
(*) T3 dev numbers are optimistic (dev shares images with trn_train); val/test/generalisation are image-disjoint.

## Observations
- Gains transfer to other feature extractors (CleanDIFT, DIY-SC) and to new images / species (AP-10K) and to PF-Pascal.
- On SPair-U (unseen keypoints, no mirror permutation, no adaptive flip) the gain is ~+1 (vs +3.5 on SPair): the
  flip-ensemble — the biggest single component — needs the keypoint mirror permutation. Remaining components
  (consensus, reverse rank, refinement) give ≈ +0.5 over dv there.
- T3 x2 vs x: better on SPair/AP-10K, worse on SPair-U/Pascal (Pascal: 300 pairs, boot std ≈ 0.35).
- Training-free family saturated: ≥ 15 further cue/rule variants within ±0.1 (journal/DAY1.md).
- The flip (pose) decision: per-pair arm oracle +5.06 vs +3.70 but no decoder evidence predicts the better arm
  (≤ +0.03) → mostly best-of-two noise.

## Next
- Day 2: (a) write the method + table drafts for the final report; (b) look for another *evidence source* comparable
  to the flip-ensemble (the only big training-free lever found); (c) per-category / per-keypoint-type breakdown.
