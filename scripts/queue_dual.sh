#!/usr/bin/env bash
cd "$(dirname "$0")/.."
T3_EXTRA=2 T3_TTA=20 T3_DUAL=1 python scripts/dump_t3.py dev 0 > runs/logs/dump_dual_dev.log 2>&1
T3_EXTRA=2 T3_TTA=20 T3_DUAL=1 python scripts/dump_t3.py trn_train 0 > runs/logs/dump_dual_trn.log 2>&1
