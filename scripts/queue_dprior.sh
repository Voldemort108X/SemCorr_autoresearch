#!/bin/bash
# dense-MNN similarity prior on candidates (SOCO dev): S-sim from S7 and S-flip from S22
X="MU=1.0,WIN=3"
for w in 0.5 1.0; do
  TIMEOUT=900 MAXP=10 SOCO_NOFLIP=1 SOCO_OUT=refine SOCO_ARM=none SOCO_DV= OVR="$X,DPRIOR=$w" TAG=sim_dprior$w bash scripts/soco_run.sh S23_$w dev dinov2_b14 S-sim analysis "S7 + DPRIOR $w (dense-MNN similarity prior)"
done
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,DPRIOR=0.5" TAG=flip_dprior0.5 bash scripts/soco_run.sh S24 dev dinov2_b14 S-flip analysis "S22 + DPRIOR 0.5"
echo QUEUE_DONE
