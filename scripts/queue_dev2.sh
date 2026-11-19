#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "^done" runs/logs/make_dev2.log; do sleep 30; done
export TIMEOUT=3000
OVR="T3_MODEL=,TTA_STEPS=0,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,ARM_PAIR=False" TAG=dvW2 bash scripts/run.sh ref dev2 > runs/logs/dev2_dvW2.log 2>&1
OVR="T3_MODEL=,TTA_STEPS=0,REF_CYC=0.5" TAG=seq14 bash scripts/run.sh 14 dev2 > runs/logs/dev2_14.log 2>&1
OVR="T3_MODEL=" TAG=seq17 bash scripts/run.sh 17 dev2 > runs/logs/dev2_17.log 2>&1
TAG=seq16 bash scripts/run.sh 16 dev2 > runs/logs/dev2_16.log 2>&1
