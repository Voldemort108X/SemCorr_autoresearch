#!/bin/bash
while pgrep -f "soco_eval.py dev dinov2_b14 runs/soco/S12" > /dev/null || pgrep -f "run_exp.py 27 val diysc" > /dev/null; do sleep 20; done
OVR="T3_MODEL=,ARM_SEL=cons" TAG=armcons bash scripts/run.sh 28 dev
echo QUEUE_DONE
