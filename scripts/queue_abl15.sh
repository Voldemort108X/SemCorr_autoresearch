#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python scripts/sweep.py dev "T3_MODEL=,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0" "T3_MODEL=,MU=0.0" "T3_MODEL=,RR=0.0" "T3_MODEL=,WIN=5,BETA=0.01" "T3_MODEL=,LAM=0.0" "T3_MODEL=,ARM_PAIR=False" "T3_MODEL=,REF_CYC=0.0" > runs/logs/ablation15_dev.log 2>&1
