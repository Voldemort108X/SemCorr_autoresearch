#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400
bash scripts/run.sh 8 test > runs/logs/test8.log 2>&1
OVR="T3_MODEL=" TAG=seq7 bash scripts/run.sh 7 test > runs/logs/test7.log 2>&1
OVR="T3_MODEL=,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0" TAG=dvW2 bash scripts/run.sh ref test > runs/logs/test_dvW2.log 2>&1
OVR="T3_MODEL=,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,LAM=0.0" TAG=top1W2 bash scripts/run.sh ref test > runs/logs/test_top1W2.log 2>&1
