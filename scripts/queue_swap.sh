#!/bin/bash
# S24 + per-keypoint mirror-pair handling (existing SPair knobs, never tried on SOCO) + dense-orientation swap test
X="MU=1.0,WIN=3,DPRIOR=0.5"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,SWAP_T=0.5" TAG=flip_s24_swap bash scripts/soco_run.sh S33 dev dinov2_b14 S-flip analysis "S24 + mirror-pair swap test (similarity of predictions, cos < -.5)"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,SWAP_T=0.5,DSWAP=True" TAG=flip_s24_dswap bash scripts/soco_run.sh S34 dev dinov2_b14 S-flip analysis "S24 + mirror-pair swap test with dense-MNN orientation (cos < -.5)"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,DECOL_R=2" TAG=flip_s24_decol bash scripts/soco_run.sh S35 dev dinov2_b14 S-flip analysis "S24 + mirror-pair de-collision (r 2)"
echo QUEUE_DONE
