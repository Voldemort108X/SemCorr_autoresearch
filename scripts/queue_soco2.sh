#!/bin/bash
X="MU=1.0,WIN=3"
TIMEOUT=900 MAXP=10 SOCO_OUT=refine SOCO_ARM=dist SOCO_DV= OVR="$X" TAG=flip_mu1w3 bash scripts/soco_run.sh S8 dev dinov2_b14 S-flip analysis "S3 + MU 1.0 + WIN 3"
TIMEOUT=900 MAXP=10 SOCO_OUT=refine SOCO_ARM=none SOCO_DV= OVR="$X" TAG=flip_arm0 bash scripts/soco_run.sh S9 dev dinov2_b14 S-flip analysis "S8 with arm 0 forced (flip ensemble only)"
TIMEOUT=900 MAXP=10 SOCO_OUT=refine SOCO_ARM=one SOCO_DV= OVR="$X" TAG=flip_arm1 bash scripts/soco_run.sh S10 dev dinov2_b14 S-flip analysis "S8 with arm 1 forced (analysis: per-pair arm oracle with S9)"
echo QUEUE_DONE
