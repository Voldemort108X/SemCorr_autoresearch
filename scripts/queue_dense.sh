#!/bin/bash
# SPair port of the SOCO dense arm rule (dev), then SOCO S22 reports + DIY-SC dev
TIMEOUT=900 OVR="T3_MODEL=,ARM_SEL=dense" TAG=armdense bash scripts/run.sh 29 dev tlr
X="MU=1.0,WIN=3"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV=0.5 OVR="$X,TTA_STEPS=0" TAG=diy_dense bash scripts/soco_run.sh D5 dev dinov2_b14_diy_sc S-flip analysis "DIY-SC: S22 config, TTA off (vs D4 = cons rule)"
for sp in intra cross test; do
  TIMEOUT=30000 MAXP=0 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV=0.5 OVR="$X" TAG=S22 bash scripts/soco_run.sh R22 $sp dinov2_b14 S-flip report "S22 config (selected on SOCO dev)"
done
echo QUEUE_DONE
