#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400
bash scripts/run.sh 14 test > runs/logs/test14.log 2>&1
bash scripts/run.sh 14 val cleandift > runs/logs/transfer14.log 2>&1
bash scripts/run.sh 14 val diysc >> runs/logs/transfer14.log 2>&1
bash scripts/run.sh 14 spairu_test > runs/logs/spairu14.log 2>&1
python prepare_gen.py eval pascal 14 tlr > runs/logs/gen_pascal_tlr_14.log 2>&1
python prepare_gen.py eval ap10kcf 14 tlr > runs/logs/gen_ap10kcf_tlr_14.log 2>&1
