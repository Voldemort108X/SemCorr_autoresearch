#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400 OVR="T3_MODEL=" TAG=tf
bash scripts/run.sh 17 test > runs/logs/test17.log 2>&1
bash scripts/run.sh 17 val cleandift > runs/logs/transfer17.log 2>&1
bash scripts/run.sh 17 val diysc >> runs/logs/transfer17.log 2>&1
bash scripts/run.sh 17 spairu_test > runs/logs/spairu17.log 2>&1
python prepare_gen.py eval pascal 17 tlr > runs/logs/gen_pascal_tlr_17.log 2>&1
python prepare_gen.py eval ap10kcf 17 tlr > runs/logs/gen_ap10kcf_tlr_17.log 2>&1
