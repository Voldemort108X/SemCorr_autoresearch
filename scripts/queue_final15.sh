#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400
bash scripts/run.sh 15 test > runs/logs/test15.log 2>&1
bash scripts/run.sh 15 val cleandift > runs/logs/transfer15.log 2>&1
bash scripts/run.sh 15 val diysc >> runs/logs/transfer15.log 2>&1
bash scripts/run.sh 15 spairu_test > runs/logs/spairu15.log 2>&1
python prepare_gen.py eval pascal 15 tlr > runs/logs/gen_pascal_tlr_15.log 2>&1
python prepare_gen.py eval ap10kcf 15 tlr > runs/logs/gen_ap10kcf_tlr_15.log 2>&1
OVR="T3_MODEL=,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,REF_CYC=0.0,ARM_PAIR=False,WIN=5,BETA=0.01" TAG=dvTTA bash scripts/run.sh ref dev > runs/logs/dvTTA_dev.log 2>&1
OVR="T3_MODEL=,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,REF_CYC=0.0,ARM_PAIR=False" TAG=dvW2TTA bash scripts/run.sh ref dev > runs/logs/dvW2TTA_dev.log 2>&1
