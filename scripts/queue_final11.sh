#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400
bash scripts/run.sh 11 test > runs/logs/test11.log 2>&1
bash scripts/run.sh 11 val cleandift > runs/logs/transfer11.log 2>&1
bash scripts/run.sh 11 val diysc >> runs/logs/transfer11.log 2>&1
bash scripts/run.sh 11 spairu_test > runs/logs/spairu11.log 2>&1
python prepare_gen.py eval pascal 11 tlr > runs/logs/gen_pascal_tlr_11.log 2>&1
python prepare_gen.py eval ap10kcf 11 tlr > runs/logs/gen_ap10kcf_tlr_11.log 2>&1
