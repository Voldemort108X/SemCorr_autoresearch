# STATE (resume point; update at the end of every iteration)

- Phase: LOOP (Day 1). Setup done 2026-11-18 ~05:20 UTC; sanity gates passed for tlr / cleandift / diysc (val).
- Caches: tlr (dev, val, test, trn_train building), cleandift (val, dev), diysc (val). ~100 GB.
- Selection = dev (TLR, 5,400 trn pairs). Noise: bootstrap std of Δ vs dv ≈ 0.05–0.12.
- Reference dev: top1 0, dv +0.777; same-refinement (W2 β.03) baselines: top1 +0.410, dv +1.767 (val +0.250 / +1.610).
- CURRENT BEST: seq 5 (commit 0e76374) dev +3.251, val +3.036
  = flip-ensembled forward row (NOM_ENS .75) + Déjà Vu cycle (λ .5) + other-arm cycle (.25) + consensus-RANSAC residual
    (μ .5, cap 5) + refinement window 2 / β .03 on (own + .5·other-arm) map.
  Transfer of seq 4 (val): CleanDIFT +1.963 (dv +0.700), DIY-SC +2.805 (dv +0.240). seq 5 transfer not run yet.
- Running: sweep5 (K, REF_ENS, LAM, MU on seq5) → runs/logs/sweep5.log; prepare trn_train (tlr) → runs/logs/prepare_trn_train.log.
- Next: (1) pick from sweep5; (2) seq5 transfer (cleandift/diysc val); (3) T3 listwise re-ranker on trn_train cue
  tables (dump_pools for trn_train subset); (4) estimated symmetry partner for keypoints w/o annotated counterpart;
  (5) daily report at ~04:00 2026-11-19.
