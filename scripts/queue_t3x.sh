#!/usr/bin/env bash
cd "$(dirname "$0")/.."
T3_EXTRA=1 python scripts/dump_t3.py dev 0 > runs/logs/dump_t3x_dev.log 2>&1
T3_EXTRA=1 python scripts/dump_t3.py trn_train 0 > runs/logs/dump_t3x_trn.log 2>&1
METHOD=cleandift python scripts/sweep.py val "T3_MODEL=" > runs/logs/transfer5.log 2>&1
METHOD=diysc python scripts/sweep.py val "T3_MODEL=" >> runs/logs/transfer5.log 2>&1
