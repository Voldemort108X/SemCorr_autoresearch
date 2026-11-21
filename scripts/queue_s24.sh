#!/bin/bash
# S24 (S-flip best: dense arm + dense prior .5): DIY-SC dev check, then full splits
X="MU=1.0,WIN=3,DPRIOR=0.5"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV=0.5 OVR="$X,TTA_STEPS=0" TAG=diy_s24 bash scripts/soco_run.sh D6 dev dinov2_b14_diy_sc S-flip analysis "DIY-SC: S24 config, TTA off (vs D4/D5)"
for sp in intra cross test; do
  TIMEOUT=30000 MAXP=0 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV=0.5 OVR="$X" TAG=S24 bash scripts/soco_run.sh R24 $sp dinov2_b14 S-flip report "S24 config (selected on SOCO dev)"
done
echo QUEUE_DONE
