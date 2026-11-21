#!/bin/bash
# Timing on an otherwise idle GPU (dev, 1000 pairs): S7 (S-sim), S12 (2 decodes/pair), S24 (1 decode + dense prior)
TIMEOUT=900 MAXP=10 SOCO_NOFLIP=1 SOCO_OUT=refine SOCO_ARM=none SOCO_DV= OVR="MU=1.0,WIN=3" TAG=time_S7 bash scripts/soco_run.sh T7 dev dinov2_b14 S-sim analysis "timing rerun of S7 on idle GPU"
TIMEOUT=900 MAXP=10 SOCO_ARM=cons SOCO_OUT=refine SOCO_DV= OVR="MU=1.0,WIN=3" TAG=time_S12 bash scripts/soco_run.sh T12 dev dinov2_b14 S-flip analysis "timing rerun of S12 on idle GPU"
TIMEOUT=900 MAXP=10 SOCO_ARM=dense SOCO_OUT=refine SOCO_DV= OVR="MU=1.0,WIN=3,DPRIOR=0.5" TAG=time_S24 bash scripts/soco_run.sh T24 dev dinov2_b14 S-flip analysis "timing rerun of S24 on idle GPU"
echo QUEUE_DONE
