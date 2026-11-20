#!/bin/bash
# S-sim sweeps from S7 (NOFLIP, MU 1.0, WIN 3): consensus geometry knobs (wrong-instance errors are far -> CAP)
B="MU=1.0,WIN=3"
run() { TIMEOUT=900 MAXP=10 SOCO_NOFLIP=1 SOCO_OUT=refine SOCO_ARM=none SOCO_DV= OVR="$B,$2" TAG=sim_$3 bash scripts/soco_run.sh $1 dev dinov2_b14 S-sim analysis "S7 + $2"; }
run S13 CAP=10 cap10
run S14 MODEL=aff aff
run S15 THR=4 thr4
run S16 THR=9 thr9
run S17 MU=1.5 mu15
run S18 K=50 k50
run S19 CAP=10,MU=1.5 cap10mu15
echo QUEUE_DONE
