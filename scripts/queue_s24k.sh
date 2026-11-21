#!/bin/bash
# S24 knob re-check now that the arm and prior are dense (dev)
X="WIN=3,DPRIOR=0.5"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,MU=1.5" TAG=flip_s24_mu15 bash scripts/soco_run.sh S30 dev dinov2_b14 S-flip analysis "S24 + MU 1.5"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,MU=0.5" TAG=flip_s24_mu05 bash scripts/soco_run.sh S31 dev dinov2_b14 S-flip analysis "S24 + MU 0.5 (prior partly replaces keypoint consensus)"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,MU=1.0,CAP=10" TAG=flip_s24_cap10 bash scripts/soco_run.sh S32 dev dinov2_b14 S-flip analysis "S24 + CAP 10"
echo QUEUE_DONE
