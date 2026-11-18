#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400 OVR="T3_MODEL=runs/t3_models/x2nopos_full_60_128.pt" TAG=nopos
bash scripts/run.sh 10 val > runs/logs/nopos_val.log 2>&1
bash scripts/run.sh 10 spairu_test > runs/logs/nopos_spairu.log 2>&1
python prepare_gen.py eval pascal 10 tlr > runs/logs/nopos_pascal.log 2>&1
python prepare_gen.py eval ap10kcf 10 tlr > runs/logs/nopos_ap10k.log 2>&1
