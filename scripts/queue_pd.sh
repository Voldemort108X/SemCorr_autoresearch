#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400 OVR="T3_MODEL=runs/t3_models/x2mixpd_full_60_128.pt,ARM_PAIR=False" TAG=mixpd
bash scripts/run.sh 13 val > runs/logs/pd_val.log 2>&1
bash scripts/run.sh 13 spairu_test > runs/logs/pd_spairu.log 2>&1
python prepare_gen.py eval pascal 13 tlr > runs/logs/pd_pascal.log 2>&1
python prepare_gen.py eval ap10kcf 13 tlr > runs/logs/pd_ap10k.log 2>&1
