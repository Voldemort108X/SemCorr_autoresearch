#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export TIMEOUT=2400
OVR="T3_MODEL=,FORCE_ARM=0" TAG=arm0 bash scripts/run.sh an dev > runs/logs/arm0.log 2>&1
OVR="T3_MODEL=,FORCE_ARM=1" TAG=arm1 bash scripts/run.sh an dev > runs/logs/arm1.log 2>&1
