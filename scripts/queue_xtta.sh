#!/usr/bin/env bash
cd "$(dirname "$0")/.."
T3_EXTRA=1 T3_TTA=20 python scripts/dump_t3.py trn_train 0 > runs/logs/dump_xtta_trn.log 2>&1
python -c "
from decode_lib import t3
t3.train('cache/tlr/trn_train/t3_x_tta20.npz','runs/t3_models/xtta_full_60_128.pt',60,None,h=128)
" > runs/logs/train_xtta.log 2>&1
export TIMEOUT=3000 OVR="T3_MODEL=runs/t3_models/xtta_full_60_128.pt" TAG=xtta
bash scripts/run.sh 21 val > runs/logs/xtta_val.log 2>&1
python prepare_gen.py eval pascal 21 tlr > runs/logs/xtta_pascal.log 2>&1
bash scripts/run.sh 21 spairu_test > runs/logs/xtta_spairu.log 2>&1
python prepare_gen.py eval ap10kcf 21 tlr > runs/logs/xtta_ap10k.log 2>&1
