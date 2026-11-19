#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400 OVR="T3_MODEL=" TAG=tf
bash scripts/run.sh 23 test > runs/logs/test23.log 2>&1
bash scripts/run.sh 23 val cleandift > runs/logs/transfer23.log 2>&1
bash scripts/run.sh 23 val diysc >> runs/logs/transfer23.log 2>&1
bash scripts/run.sh 23 spairu_test > runs/logs/spairu23.log 2>&1
python prepare_gen.py eval pascal 23 tlr > runs/logs/gen_pascal_tlr_23.log 2>&1
python prepare_gen.py eval ap10kcf 23 tlr > runs/logs/gen_ap10kcf_tlr_23.log 2>&1
bash scripts/run.sh 23 test cleandift > runs/logs/tt23c.log 2>&1
bash scripts/run.sh 23 test diysc > runs/logs/tt23d.log 2>&1
bash scripts/run.sh 23 dev2 > runs/logs/dev2_23.log 2>&1
