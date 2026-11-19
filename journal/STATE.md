# STATE (resume point; update at the end of every iteration)

- Phase: LOOP, day 2 (2026-11-19). Setup done; sanity gates passed (tlr/cleandift/diysc val, SPair-U, AP-10K).
- Caches (~170 GB): tlr (dev/val/test/trn_train/spairu_test, gen_pascal, gen_ap10kcf), cleandift (dev/val/test),
  diysc (val/test). T3 tables in cache/tlr/{dev,trn_train}/t3_*.npz.
- Selection = dev (TLR, 5,400 SPair trn pairs). Bootstrap std of Δ vs dv ≈ 0.05–0.14. Run limit 600 s (TTA runs ~410 s).
- BEST training-free: seq 17 (commit bd80053), run with OVR="T3_MODEL=": dev +4.712, val +4.563, TEST +4.318
  (74.08/59.64/14.15); CleanDIFT val/test +4.519/+4.293; DIY-SC +2.470/+2.739; SPair-U +1.674; Pascal +3.825; AP-10K +3.562.
  = TTA (decode_lib/tta.py) + flip ensemble + DV cycle (+other-arm cycle) + consensus RANSAC + reverse rank + ARM_PAIR +
  refinement W2 β.03 on the ensembled map. seq 15 (+REF_CYC .5) is equivalent.
- BEST T3: seq 16 (commit 0003221; decode.py default T3_MODEL runs/t3_models/x2tta_full_60_128.pt): val +6.090,
  TEST +5.988 (76.06/62.00/14.83); CleanDIFT +5.857/+5.963; DIY-SC +2.815/+3.083; SPair-U +2.029; Pascal +3.894; AP-10K +4.196.
  T3 selection must use the image-disjoint A/B protocol (decode_lib/t3split.py).
- Reference: Déjà Vu test +0.823; Déjà Vu + W2 refinement test +1.750; oracle top-5 test 77.11/59.68/12.23.
- Tools: scripts/run.sh (OVR/TAG env), sweep.py ("cats=..." subset), prepare_gen.py eval <ds>, error_types.py,
  tables.py, dump_t3.py (T3_EXTRA/T3_TTA/...), addrow.py, discard.sh (keeps results.tsv).
- Reports: reports/daily_2026-11-19.md (sent), reports/FINAL_REPORT_draft.md, reports/tables_test.md.
- Next: (1) mirror-ambiguity signal (occlusion/depth cues?); (2) DIY-SC-safe TTA (dev-selectable); (3) final report
  writing from 2026-11-22 12:00; daily report 2026-11-20 ~04:00.
