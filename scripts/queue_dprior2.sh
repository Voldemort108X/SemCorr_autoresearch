#!/bin/bash
X="MU=1.0,WIN=3"; Y="MU=1.5,WIN=3,CAP=10"
TIMEOUT=900 MAXP=10 SOCO_NOFLIP=1 SOCO_OUT=refine SOCO_ARM=none SOCO_DV= OVR="$Y,DPRIOR=1.0" TAG=sim19_dprior1 bash scripts/soco_run.sh S25 dev dinov2_b14 S-sim analysis "S19 + DPRIOR 1.0"
TIMEOUT=900 MAXP=10 SOCO_NOFLIP=1 SOCO_OUT=refine SOCO_ARM=none SOCO_DV= OVR="$X,DPRIOR=2.0" TAG=sim_dprior2 bash scripts/soco_run.sh S26 dev dinov2_b14 S-sim analysis "S7 + DPRIOR 2.0"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,DPRIOR=1.0" TAG=flip_dprior1 bash scripts/soco_run.sh S27 dev dinov2_b14 S-flip analysis "S22 + DPRIOR 1.0"
echo QUEUE_DONE
