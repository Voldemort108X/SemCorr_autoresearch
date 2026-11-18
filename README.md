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

## Current best
(not started yet)
