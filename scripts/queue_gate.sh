#!/bin/bash
while pgrep -f "soco_eval.py dev dinov2_b14 runs/soco/S11" > /dev/null; do sleep 20; done
OVR="T3_MODEL=,TTA_GATE=True" TAG=gate bash scripts/run.sh 27 dev
OVR="T3_MODEL=,TTA_GATE=True" TAG=gate TIMEOUT=1500 bash scripts/run.sh 27 val tlrsup
OVR="T3_MODEL=,TTA_GATE=True" TAG=gate TIMEOUT=1500 bash scripts/run.sh 27 val diysc
echo QUEUE_DONE
