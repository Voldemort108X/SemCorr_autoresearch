# STATE (resume point; update at the end of every iteration)

- Phase: LOOP (Day 1). Setup done 2026-11-18 ~05:20 UTC; sanity gates passed for tlr / cleandift / diysc (val).
- Caches: tlr (dev, val, test, trn_train building), cleandift (val, dev), diysc (val). ~100 GB.
- Selection = dev (TLR, 5,400 trn pairs). Noise: bootstrap std of Δ vs dv ≈ 0.05–0.12.
- Reference dev: top1 0, dv +0.777; same-refinement (W2 β.03) baselines: top1 +0.410, dv +1.767 (val +0.250 / +1.610).
- CURRENT BEST training-free: seq 14 (commit fa6e150) dev +3.823, val +3.583 (decode.py default, T3_MODEL='')
  = seq 11 + refinement on the cycle-penalised score map (REF_CYC .5).
  seq 11 (53446e7): dev +3.698, val +3.390, test +3.556; CleanDIFT +3.270, DIY-SC +2.945, SPair-U +0.989, Pascal +3.480, AP-10K +2.676.
  seq 7 (f6ff781; = seq 11 with ARM_PAIR=False): dev +3.584, val +3.312, test +3.457; transfer CleanDIFT +3.158, DIY-SC +2.945.
- CURRENT BEST T3 (learned on trn_train): seq 9 (commit 3d51e99) dev +5.309 (optimistic: dev shares images with
  trn_train), val +4.972, TEST +5.230 (75.50/61.05/14.05); transfer val CleanDIFT +4.882, DIY-SC +3.657.
  Reproduce with decode.py: OVR="T3_MODEL=runs/t3_models/x2_full_60_128.pt,ARM_PAIR=False".
  DeepSets re-ranker on the seq-5 pool, 64 features (T3_EXTRA=2), h128 60 ep; model runs/t3_models/x2_full_60_128.pt
  (rebuild: T3_EXTRA=2 python scripts/dump_t3.py trn_train 0; python -c "from decode_lib import t3;
  t3.train('cache/tlr/trn_train/t3_x2.npz','runs/t3_models/x2_full_60_128.pt',60,'cache/tlr/dev/t3_x2.npz',h=128)").
  T3 model selection MUST use the image-disjoint A/B protocol (decode_lib/t3split.py; tables t3_x2_A / t3_x2_B).
- Same-refinement baselines (W2 β.03 own map): top1 dev +0.410 / val +0.250 / test +0.379; dv dev +1.767 / val +1.610 / test +1.750.
- TEST (selected on dev): top1 69.78/54.09/11.04, dv 71.22/54.98/11.19, seq7 73.57/58.51/13.21 (+3.457), seq8 74.97/60.42/13.90 (+4.791).
- Generalisation (settings from dev): SPair-U test (no perm → no flip-ensemble): dv +0.51, dvW2 +0.50, seq7 +0.99, seq8 +1.45.
  PF-Pascal (PCK .10/.05/.15 @840px): top1 86.50/73.57/91.30, dv 89.44/75.93/93.70, seq7 90.31/77.38/94.12, seq8 91.59/78.33/95.44.
  AP-10K CF test: top1 53.30/36.26/7.10, dv +0.56, dvW2 +1.30, seq7 +2.50 (55.78/39.05/9.33), seq8 +3.17 (56.59/39.90/9.67).
- Running: ablation_dev3 (no-flip / dvW2-equivalent) → runs/logs/ablation_dev3.log.
- Tracks so far: T1/T4 training-free saturated (~25 variants within noise, see DAY1.md); T2 consensus is the only
  geometry piece that helps; T3 saturated at ≈ +1.7 over seq5 on the image-disjoint A/B protocol (two-arm pool, joint
  attention, bigger models, rc labels, perm-drop: no gain).
- Next: (1) daily report reports/daily_2026-11-19.md at ~04:00 UTC → update with seq11/14/9 numbers + ablation, commit
  results.tsv, git push origin autoresearch/nov18, PushNotification; (2) final report draft reports/FINAL_REPORT_draft.md;
  (3) new evidence sources (only the flip-ensemble was a big training-free lever) — ideas: none strong yet.
