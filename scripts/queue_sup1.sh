#!/bin/bash
# supervised TLR: transfer of TLR-dev-selected decoders (no tuning); val + test
until grep -q QUEUE_DONE runs/logs/queue_soco1.log; do sleep 30; done
for sp in val test; do
  TIMEOUT=1500 OVR="T3_MODEL=" TAG=tf bash scripts/run.sh 23 $sp tlrsup
  TIMEOUT=1500 OVR="T3_MODEL=,TTA_STEPS=0" TAG=notta bash scripts/run.sh 23 $sp tlrsup
  TIMEOUT=1500 TAG=t3 bash scripts/run.sh 16 $sp tlrsup
done
echo QUEUE_DONE
