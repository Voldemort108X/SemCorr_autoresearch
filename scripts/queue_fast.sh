#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export OVR="T3_MODEL=,TTA_STEPS=0,TTA_PRIOR=runs/tta_prior.npy" TAG=fastprior
bash scripts/run.sh 20 dev > runs/logs/fast_dev.log 2>&1
bash scripts/run.sh 20 val > runs/logs/fast_val.log 2>&1
bash scripts/run.sh 20 test > runs/logs/fast_test.log 2>&1
