#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=3000
for m in cleandift diysc; do
  python prepare_gen.py spairu $m > runs/logs/gen_spairu_$m.log 2>&1
  OVR="T3_MODEL=" TAG=tf bash scripts/run.sh 17 spairu_test $m > runs/logs/spairu17_$m.log 2>&1
  OVR="T3_MODEL=,TTA_STEPS=0,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,ARM_PAIR=False,WIN=5,BETA=0.01" TAG=dv bash scripts/run.sh ref spairu_test $m > runs/logs/spairudv_$m.log 2>&1
done
python prepare_gen.py ap10kcf cleandift > runs/logs/gen_ap10k_cleandift.log 2>&1
OVR="T3_MODEL=" TAG=tf python prepare_gen.py eval ap10kcf 17 cleandift > runs/logs/ap10k17_cleandift.log 2>&1
OVR="T3_MODEL=,TTA_STEPS=0,MU=0.0,NOM_ENS=0.0,LFL=0.0,REF_ENS=0.0,RR=0.0,ARM_PAIR=False,WIN=5,BETA=0.01" TAG=dv python prepare_gen.py eval ap10kcf ref cleandift > runs/logs/ap10kdv_cleandift.log 2>&1
