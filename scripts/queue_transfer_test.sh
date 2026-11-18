#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python prepare.py --method cleandift --split test > runs/logs/prepare_cleandift_test.log 2>&1
python prepare.py --method diysc --split test > runs/logs/prepare_diysc_test.log 2>&1
export TIMEOUT=3000
OVR="T3_MODEL=" TAG=tf bash scripts/run.sh 17 test cleandift > runs/logs/tt17c.log 2>&1
OVR="T3_MODEL=" TAG=tf bash scripts/run.sh 17 test diysc > runs/logs/tt17d.log 2>&1
bash scripts/run.sh 16 test cleandift > runs/logs/tt16c.log 2>&1
bash scripts/run.sh 16 test diysc > runs/logs/tt16d.log 2>&1
OVR="T3_MODEL=,TTA_STEPS=0,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,ARM_PAIR=False,WIN=5,BETA=0.01" TAG=dv bash scripts/run.sh ref test cleandift > runs/logs/ttdvc.log 2>&1
OVR="T3_MODEL=,TTA_STEPS=0,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,ARM_PAIR=False,WIN=5,BETA=0.01" TAG=dv bash scripts/run.sh ref test diysc > runs/logs/ttdvd.log 2>&1
