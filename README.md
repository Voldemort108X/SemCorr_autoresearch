# SemCorr_autoresearch

Autonomous 5-day exploration (2026-11-18 → 2026-11-23) of decoders that close the gap between top-1 matching and
the top-k oracle for semantic correspondence, building on Déjà Vu (CVPR'27 submission). Instructions for the agent:
`program.md`. Everything produced lives in this folder; `/workspace/SemCorr_proj` and `/workspace/SemCorrPaper` are
read-only inputs.

## Layout
| Path | What |
|---|---|
| `program.md` | rules, metric, loop, idea backlog (author-edited) |
| `prepare.py` | frozen: cache building + `evaluate()` (ground-truth metric) |
| `decode.py` | the only file the loop edits (the decoder under test) |
| `vendor/` | copies of the needed SemCorr2D utilities (source path noted in each file) |
| `scripts/` | `run.sh` (one experiment), val / transfer / final-eval helpers |
| `cache/` | fp16 descriptors, masks, keypoints, flip arm per split and base method (gitignored) |
| `runs/<seq>_<commit>/` | `run.log`, `metrics.json`, per-keypoint errors |
| `results.tsv` | one row per experiment (dev = mean ΔPCK vs. top-1 over @0.10/0.05/0.01) |
| `journal/` | `STATE.md` (resume point), `ideas.md`, `DAY<n>.md`, `analysis_<seq>.md` |
| `reports/` | daily summaries and `FINAL_REPORT.md` |

## How to run
```
python prepare.py --method tlr --split dev,val,test        # cache (once; ~2.5 min/category)
bash scripts/run.sh <seq> [dev|val|test] [tlr|cleandift|diysc]   # evaluates decode.py -> runs/<seq>_<commit>[...]/
python scripts/sanity.py tlr val                           # sanity gate vs. the review dumps
python scripts/sweep.py dev "NAME=val,..." ...             # decode.py with overridden constants (no files written)
python scripts/dump_pools.py tlr dev                       # per-candidate cue tables for offline screening
```

## Current best
- Training-free DV++, seq 23 (commit 5b1bd72; `OVR="T3_MODEL=" bash scripts/run.sh ...`): dev +4.918 vs top1 (dv +0.777);
  val +4.770; TEST 74.46/59.91/14.42 (+4.627; top1 69.78/54.09/11.04, dv 71.22/54.98/11.19).
- Learned (T3, trained on SPair trn_train), seq 16 (decode.py default): val +6.090, TEST 76.06/62.00/14.83 (+5.988).
- Final report: reports/FINAL_REPORT.md. Tables: results.tsv, reports/tables_test.md. Resume point: journal/STATE.md.
