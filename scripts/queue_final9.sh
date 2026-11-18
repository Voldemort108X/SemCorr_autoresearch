#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400
bash scripts/run.sh 9 test > runs/logs/test9.log 2>&1
bash scripts/run.sh 9 val cleandift > runs/logs/transfer9.log 2>&1
bash scripts/run.sh 9 val diysc >> runs/logs/transfer9.log 2>&1
bash scripts/run.sh 9 spairu_test > runs/logs/spairu9.log 2>&1
python prepare_gen.py eval pascal 9 tlr > runs/logs/gen_pascal_tlr_9.log 2>&1
python prepare_gen.py eval ap10kcf 9 tlr > runs/logs/gen_ap10kcf_tlr_9.log 2>&1
