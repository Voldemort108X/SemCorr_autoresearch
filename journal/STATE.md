# STATE (resume point; update at the end of every iteration)

- Phase: LOOP (Day 1). Setup done 2026-11-18 ~05:20 UTC; sanity gates passed for tlr / cleandift / diysc (val).
- Caches: tlr (dev, val, test, trn_train building), cleandift (val, dev), diysc (val). ~100 GB.
- Selection = dev (TLR, 5,400 trn pairs). Noise: bootstrap std of Δ vs dv ≈ 0.05–0.12.
- Reference dev: top1 0, dv +0.777; same-refinement (W2 β.03) baselines: top1 +0.410, dv +1.767 (val +0.250 / +1.610).
- CURRENT BEST training-free: seq 17 (commit bd80053) dev +4.712, val +4.563 = seq 15 without REF_CYC (simplicity rule;
  run with OVR="T3_MODEL="). seq 15 (1b0ebde) finalist: dev +4.714, val +4.657, TEST +4.378, CleanDIFT +4.593,
  DIY-SC +2.355, SPair-U +1.789, Pascal +3.977, AP-10K +3.472.
- CURRENT BEST training-free: seq 15 (commit 1b0ebde) dev +4.714, val +4.657 (decode.py default, T3_MODEL='')
  = seq 14 + per-pair test-time channel re-weighting (decode_lib/tta.py: diagonal softplus weights on the descriptor
  channels, 20 Adam steps (lr .05) maximising the dual-softmax mutual-match confidence of masked src(arm0)/trg; no
  labels). ~58 ms/pair.
  seq 14 (fa6e150): dev +3.823, val +3.583, test +3.711; CleanDIFT +3.463, DIY-SC +2.917, SPair-U +1.237, Pascal +3.590, AP-10K +2.709.
  seq 11 (53446e7): dev +3.698, val +3.390, test +3.556; CleanDIFT +3.270, DIY-SC +2.945, SPair-U +0.989, Pascal +3.480, AP-10K +2.676.
  seq 7 (f6ff781; = seq 11 with ARM_PAIR=False): dev +3.584, val +3.312, test +3.457; transfer CleanDIFT +3.158, DIY-SC +2.945.
- CURRENT BEST T3: seq 16 (commit 0003221) dev +6.273 (optimistic), val +6.090 — x2 DeepSets re-ranker trained on
  TTA-adapted descriptors (T3_EXTRA=2 T3_TTA=20 dump_t3.py), decoded after TTA; model runs/t3_models/x2tta_full_60_128.pt.
  decode.py default (T3_MODEL set; ARM_PAIR only applies to the training-free path). Training-free = OVR "T3_MODEL=".
  Previous T3 seq 9 (3d51e99): val +4.972, test +5.230 (reproduce: OVR "T3_MODEL=runs/t3_models/x2_full_60_128.pt,TTA_STEPS=0").
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
