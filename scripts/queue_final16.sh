#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400
bash scripts/run.sh 16 test > runs/logs/test16.log 2>&1
bash scripts/run.sh 16 val cleandift > runs/logs/transfer16.log 2>&1
bash scripts/run.sh 16 val diysc >> runs/logs/transfer16.log 2>&1
bash scripts/run.sh 16 spairu_test > runs/logs/spairu16.log 2>&1
python prepare_gen.py eval pascal 16 tlr > runs/logs/gen_pascal_tlr_16.log 2>&1
python prepare_gen.py eval ap10kcf 16 tlr > runs/logs/gen_ap10kcf_tlr_16.log 2>&1
