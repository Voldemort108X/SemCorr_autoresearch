#!/bin/bash
# SPair port of the SOCO dense similarity prior (S24's DPRIOR .5) on top of seq23 (dev)
TIMEOUT=900 OVR="T3_MODEL=,DPRIOR=0.5" TAG=dprior bash scripts/run.sh 30 dev tlr
echo QUEUE_DONE
