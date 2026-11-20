#!/bin/bash
# S20: S-flip with S19's consensus knobs (dev); DIY-SC full splits (TTA off, selected on SOCO dev D1-D4); R19 full splits
X="MU=1.0,WIN=3"; Y="MU=1.5,WIN=3,CAP=10"
TIMEOUT=900 MAXP=10 SOCO_OUT=refine SOCO_ARM=cons SOCO_DV= OVR="$Y" TAG=flip_cap10mu15 bash scripts/soco_run.sh S20 dev dinov2_b14 S-flip analysis "S12 + CAP 10 + MU 1.5 (S19 knobs)"
for sp in intra cross; do
  TIMEOUT=30000 MAXP=0 SOCO_ARM=cons SOCO_OUT=refine SOCO_DV=0.5 OVR="$X,TTA_STEPS=0" TAG=D4 bash scripts/soco_run.sh RD4 $sp dinov2_b14_diy_sc S-flip report "DIY-SC, D4 config (S12, TTA off; selected on SOCO dev)"
  TIMEOUT=30000 MAXP=0 SOCO_NOFLIP=1 SOCO_ARM=none SOCO_OUT=refine SOCO_DV=0.5 OVR="$X,TTA_STEPS=0" TAG=D2 bash scripts/soco_run.sh RD2 $sp dinov2_b14_diy_sc S-sim report "DIY-SC, D2 config (S7, TTA off; selected on SOCO dev)"
done
for sp in intra cross test; do
  TIMEOUT=30000 MAXP=0 SOCO_NOFLIP=1 SOCO_ARM=none SOCO_OUT=refine SOCO_DV=0.5 OVR="$Y" TAG=S19 bash scripts/soco_run.sh R19 $sp dinov2_b14 S-sim report "S19 config (selected on SOCO dev)"
done
echo QUEUE_DONE
