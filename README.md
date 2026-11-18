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
- Training-free, seq 14 (commit fa6e150): dev +3.823 vs top1 (dv +0.777); val +3.583 (dv +0.891).
  Val per-kpt PCK@.10/.05/.01 72.30/57.14/12.10 (top1 68.33/52.39/10.08, dv 69.84/53.38/10.25).
- Learned re-ranker (T3, trained on SPair trn_train), seq 9 (commit 3d51e99): val +4.972, TEST +5.230
  (75.50/61.05/14.05 vs top1 69.78/54.09/11.04, dv 71.22/54.98/11.19).
- Tables: results.tsv; analyses: journal/analysis_*.md; resume point: journal/STATE.md. Selection on dev only.
