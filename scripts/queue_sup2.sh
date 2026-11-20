#!/bin/bash
# DIY-SC test without TTA (report only; completes the supervised-features table)
until grep -q QUEUE_DONE runs/logs/queue_soco3.log; do sleep 30; done
TIMEOUT=1500 OVR="T3_MODEL=,TTA_STEPS=0" TAG=notta bash scripts/run.sh 23 test diysc
echo QUEUE_DONE
