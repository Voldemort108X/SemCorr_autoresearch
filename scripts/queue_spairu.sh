#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400
OVR="T3_MODEL=,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,WIN=5,BETA=0.01" TAG=dvbase bash scripts/run.sh ref spairu_test > runs/logs/spairu_dvbase.log 2>&1
OVR="T3_MODEL=" TAG=seq7 bash scripts/run.sh 7 spairu_test > runs/logs/spairu7.log 2>&1
bash scripts/run.sh 8 spairu_test > runs/logs/spairu8.log 2>&1
OVR="T3_MODEL=,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0" TAG=dvW2 bash scripts/run.sh ref spairu_test > runs/logs/spairu_dvW2.log 2>&1
