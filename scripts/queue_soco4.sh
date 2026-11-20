#!/bin/bash
# DIY-SC (supervised) on SOCO-dev: S-sim / S-flip kept configs, with and without TTA
until [ -f cache/soco/dinov2_b14_diy_sc/meta.json ] && ! pgrep -f "soco_extract.py --backbone dinov2_b14_diy_sc" >/dev/null; do sleep 30; done
X="MU=1.0,WIN=3"
TIMEOUT=900 MAXP=10 SOCO_NOFLIP=1 SOCO_OUT=refine SOCO_ARM=none SOCO_DV=0.5 OVR="$X" TAG=diy_sim bash scripts/soco_run.sh D1 dev dinov2_b14_diy_sc S-sim analysis "DIY-SC: S7 config (TTA on)"
TIMEOUT=900 MAXP=10 SOCO_NOFLIP=1 SOCO_OUT=refine SOCO_ARM=none SOCO_DV=0.5 OVR="$X,TTA_STEPS=0" TAG=diy_sim_notta bash scripts/soco_run.sh D2 dev dinov2_b14_diy_sc S-sim analysis "DIY-SC: S7 config, TTA off"
TIMEOUT=900 MAXP=10 SOCO_OUT=refine SOCO_ARM=cons SOCO_DV=0.5 OVR="$X" TAG=diy_flip bash scripts/soco_run.sh D3 dev dinov2_b14_diy_sc S-flip analysis "DIY-SC: S12 config (TTA on)"
TIMEOUT=900 MAXP=10 SOCO_OUT=refine SOCO_ARM=cons SOCO_DV=0.5 OVR="$X,TTA_STEPS=0" TAG=diy_flip_notta bash scripts/soco_run.sh D4 dev dinov2_b14_diy_sc S-flip analysis "DIY-SC: S12 config, TTA off"
echo QUEUE_DONE
