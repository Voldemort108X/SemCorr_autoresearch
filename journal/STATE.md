# STATE (resume point; update at the end of every iteration)

## NEW TASKS from the author (2026-11-20 ~18:30 UTC) — these now drive the loop
1. SOCO benchmark: test DV++ (and parts of it) on SOCO and ITERATE the method on it. Selection on SOCO
   PairAnnotations/trainsplits/train (SOCO-dev); report on intra + cross (the paper's protocol, OmniProbe eval).
   Paper's code: SemCorr_proj/cvpr2027_results/scripts/soco_eval_onepass.py (+ soco_dejavu.py), run in
   SemCorr_proj/Code/external/OmniProbe/.venv; paper numbers in Results/cvpr2027_reruns/soco/*.log (λ0 = argmax).
   Our copy: scripts/soco_eval.py. Mirror permutation from left/right keypoint names.
2. Supervised model: does the strategy help supervised features? DIY-SC on SPair (done: +2.6 test vs dv +0.3; TTA hurts)
   and on SOCO (backbone dinov2_b14_diy_sc); look for other supervised checkpoints; make TTA safe for supervised.
   PROGRESS (2026-11-20 18:15): cache/soco/dinov2_b14 (38 GB, orig+flip, 57x57), evaluator scripts/soco_eval.py validated
   vs paper per-class (±0.1). Runner scripts/soco_run.sh -> soco_results.tsv, runs/soco/. SOCO-dev = trainsplits/train,
   10 pairs/class (1000 pairs, ~4 min). Tracks: S-sim (similarity only: no flip arm, no flip ensemble) and S-flip
   (pair-level flip arm 'dist' + flip ensemble, mirror perm from left/right names). Heads: argmax (exact), soft
   (OmniProbe soft_eval), dv0.5 (paper port). Dev: S2 S-sim refine +4.40 SOC/+2.48 CC (soft +3.01/+1.25, dv +1.56/+0.53);
   S3 S-flip +11.35/+3.73. Padding mask (S4) = noise. queue_soco1 (S5.x sweeps) then queue_sup1 (tlrsup val/test).
   Disk: delete cache/soco/dinov2_b14 before extracting another backbone (300 GB cap).
   Supervised: method 'tlrsup' (scripts/sup_hook.py; released supervised TLR ckpt best_856.PTH; meta symlinked from tlr).
   NOTE dev = SPair trn = the supervised model's training data -> no tuning on it; transfer settings, report val/test.
3. Paper limitations (5_results.tex): (a) gap to top-k needs global reasoning; (b) mirror-equivalent parts both
   cycle-consistent; (c) bird/horse regress with Déjà Vu (ours: bird ±0, horse +1.4 on test).


- Phase: LOOP, day 2 (2026-11-19). Setup done; sanity gates passed (tlr/cleandift/diysc val, SPair-U, AP-10K).
- Caches (~170 GB): tlr (dev/val/test/trn_train/spairu_test, gen_pascal, gen_ap10kcf), cleandift (dev/val/test),
  diysc (val/test). T3 tables in cache/tlr/{dev,trn_train}/t3_*.npz.
- Selection = dev (TLR, 5,400 SPair trn pairs). Bootstrap std of Δ vs dv ≈ 0.05–0.14. Run limit 600 s (TTA runs ~410 s).
- BEST training-free: seq 23 (commit 5b1bd72; OVR="T3_MODEL="): dev +4.918, val +4.770 = seq 17 + ROW_SMOOTH .7
  (3x3-smoothed own query as a third view in the forward row) + REF_ENS 1.0. Finalists: TEST +4.627 (74.46/59.91/14.42), CleanDIFT val/test +4.684/+4.499, DIY-SC +2.416/+2.581,
  SPair-U +1.936, Pascal +4.184, AP-10K +3.722, dev2 +4.614. FINAL_REPORT v2 uses seq 23.
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
- Reports: reports/daily_2026-11-19.md (sent), reports/FINAL_REPORT.md (v1 written 2026-11-19; update if anything
  changes), reports/tables_test.md (per-category, per-image, bootstrap CIs).
- Next: (1) mirror-ambiguity signal (occlusion/depth cues?); (2) DIY-SC-safe TTA (dev-selectable); (3) final report
  writing from 2026-11-22 12:00; daily report 2026-11-20 ~04:00.
