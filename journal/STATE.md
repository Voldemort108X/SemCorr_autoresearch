# STATE (resume point; update at the end of every iteration)

- Phase: Day-0 setup DONE for TLR (2026-11-18 05:20 UTC); CleanDIFT cache (val,dev) building; DIY-SC cache next.
  Sanity gate TLR/val PASSED (top1 / lam0.5 within ±0.005 PCK of the review dumps; 0 flip-arm mismatches).
- Loop: started. Selection = dev (TLR, 5,400 pairs from SPair trn, seed 0+cat index), metric `dev` (prepare.py).
- Reference (dev, per kpt @.10/.05/.01): top1 68.19/52.06/9.66, dv 69.58/52.93/9.74 (dev +0.777),
  or5 75.89, or10 80.15, or30 87.16.
- Current best: dv baseline (dev +0.777).
- Tools: `bash scripts/run.sh <seq> [split] [method]`; `scripts/dump_pools.py` (cue tables of the top-30 pool + per-
  candidate GT error -> cache/<m>/<split>/pools*.npz) + `scripts/offline.py` / `analyze_pools.py` for fast offline
  screening of re-ranking rules (screen on dev only; every claimed number must come from run.sh).
- Running: dump_pools tlr dev.
- Next: (1) failure analysis on full dev pools; (2) T2 cues screen (consensus RANSAC over pools, MLS, context,
  handedness) on full dev; (3) T3 listwise re-ranker on trn_train cue tables if single cues fail.
