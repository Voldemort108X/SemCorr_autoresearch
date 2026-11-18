#!/usr/bin/env bash
# usage: scripts/queue_gen.sh <ds> [method]  -> dv(base), seq7, seq8, dvW2 on a generalisation set
cd "$(dirname "$0")/.."
ds=$1; m=${2:-tlr}
OVR="T3_MODEL=,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,WIN=5,BETA=0.01" TAG=dvbase python prepare_gen.py eval $ds ref $m > runs/logs/gen_${ds}_${m}_dvbase.log 2>&1
OVR="T3_MODEL=" TAG=seq7 python prepare_gen.py eval $ds 7 $m > runs/logs/gen_${ds}_${m}_7.log 2>&1
python prepare_gen.py eval $ds 8 $m > runs/logs/gen_${ds}_${m}_8.log 2>&1
OVR="T3_MODEL=,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0" TAG=dvW2 python prepare_gen.py eval $ds ref $m > runs/logs/gen_${ds}_${m}_dvW2.log 2>&1
