#!/bin/bash
# S24 + inlier threshold of the dense-MNN similarity used by the prior (decode DENSE_THR; arm rule keeps thr 3)
X="MU=1.0,WIN=3,DPRIOR=0.5"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,DENSE_THR=2" TAG=flip_dthr2 bash scripts/soco_run.sh S28 dev dinov2_b14 S-flip analysis "S24 + prior inlier thr 2"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="$X,DENSE_THR=5" TAG=flip_dthr5 bash scripts/soco_run.sh S29 dev dinov2_b14 S-flip analysis "S24 + prior inlier thr 5"
echo QUEUE_DONE
