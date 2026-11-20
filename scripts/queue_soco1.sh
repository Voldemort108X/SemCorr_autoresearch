#!/bin/bash
B="NOM_ENS=0,LFL=0,REF_ENS=0"
i=0
for o in "LAM=0.25" "LAM=1.0" "MU=0" "MU=1.0" "TTA_STEPS=0" "TTA_STEPS=40" "ROW_SMOOTH=0" "ROW_SMOOTH=1.5" "RR=0" "RR=0.05" "WIN=3" "BETA=0.05" "BETA=0.015"; do
  i=$((i+1))
  MAXP=10 SOCO_OUT=refine SOCO_ARM=none SOCO_DV= OVR="$B,$o" TAG="sw_$o" bash scripts/soco_run.sh S5.$i dev dinov2_b14 S-sim sweep "S2 sweep: $o"
done
echo QUEUE_DONE
