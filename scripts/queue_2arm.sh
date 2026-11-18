#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python scripts/dump_t3_2arm.py dev 0 > runs/logs/dump_2arm_dev.log 2>&1
python scripts/dump_t3_2arm.py trn_train 0 > runs/logs/dump_2arm_trn.log 2>&1
