# STATE (resume point; update at the end of every iteration)

- Phase: LOOP (Day 1). Setup done 2026-11-18 ~05:20 UTC; sanity gates passed for tlr / cleandift / diysc (val).
- Caches: tlr (dev, val, test, trn_train building), cleandift (val, dev), diysc (val). ~100 GB.
- Selection = dev (TLR, 5,400 trn pairs). Noise: bootstrap std of Δ vs dv ≈ 0.05–0.12.
- Reference dev: top1 0, dv +0.777; same-refinement (W2 β.03) baselines: top1 +0.410, dv +1.767 (val +0.250 / +1.610).
- CURRENT BEST training-free: seq 7 (commit f6ff781) dev +3.584, val +3.312; transfer val CleanDIFT +3.158, DIY-SC +2.945
  = flip-ensembled forward row (NOM_ENS .75) + Déjà Vu cycle (λ .5) + other-arm cycle (.25) + consensus-RANSAC residual
    (μ .5, cap 5) + reverse-rank penalty (.02 log1p) + refinement window 2 / β .03 on (own + .5·other-arm) map.
    (= decode.py with T3_MODEL='')
- CURRENT BEST T3 (learned on trn_train): seq 8 (commit ba7b367) dev +4.753, val +4.557 — DeepSets re-ranker on the
  seq-5 pool with 31 features; model runs/t3_models/x_full_60_128.pt (gitignored; rebuild: T3_EXTRA=1 python
  scripts/dump_t3.py trn_train 0 && python -c "from decode_lib import t3; t3.train('cache/tlr/trn_train/t3_x.npz',
  'runs/t3_models/x_full_60_128.pt', 60, 'cache/tlr/dev/t3_x.npz', h=128)").
- Same-refinement baselines (W2 β.03 own map): top1 dev +0.410 / val +0.250; dv dev +1.767 / val +1.610.
- Running: sweep SD_W (block reweighting) on seq7; transfer of seq8 (cleandift/diysc val).
- Next: (1) analysis step (seq ~10): errors of seq7/seq8 by type & category; (2) daily report 2026-11-19 ~04:00 UTC
  (+ git push); (3) test split + SPair-U / AP-10K generalization for finalists (day 4-5); (4) more training-free ideas:
  block reweighting, multi-hop cycle, per-keypoint arm self-selection (protocol deviation: flag it).
