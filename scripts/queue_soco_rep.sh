#!/bin/bash
# SOCO reporting runs (DINOv2-B/14), full splits, kept configs: S12 (S-flip) and S7 (S-sim)
F='SOCO_ARM=cons'; X="MU=1.0,WIN=3"
for sp in intra cross; do
  TIMEOUT=30000 MAXP=0 SOCO_ARM=cons SOCO_OUT=refine SOCO_DV=0.5 OVR="$X" TAG=S12 bash scripts/soco_run.sh R12 $sp dinov2_b14 S-flip report "S12 config (selected on SOCO dev = trainsplits/train)"
  TIMEOUT=30000 MAXP=0 SOCO_NOFLIP=1 SOCO_ARM=none SOCO_OUT=refine SOCO_DV=0.5 OVR="$X" TAG=S7 bash scripts/soco_run.sh R7 $sp dinov2_b14 S-sim report "S7 config (selected on SOCO dev = trainsplits/train)"
done
for sp in test; do
  TIMEOUT=30000 MAXP=0 SOCO_ARM=cons SOCO_OUT=refine SOCO_DV=0.5 OVR="$X" TAG=S12 bash scripts/soco_run.sh R12 $sp dinov2_b14 S-flip report "S12 config, image-disjoint test split"
  TIMEOUT=30000 MAXP=0 SOCO_NOFLIP=1 SOCO_ARM=none SOCO_OUT=refine SOCO_DV=0.5 OVR="$X" TAG=S7 bash scripts/soco_run.sh R7 $sp dinov2_b14 S-sim report "S7 config, image-disjoint test split"
done
echo QUEUE_DONE
