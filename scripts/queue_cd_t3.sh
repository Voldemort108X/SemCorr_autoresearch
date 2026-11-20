#!/usr/bin/env bash
# method-specific T3 for CleanDIFT (reported separately; not plug-and-play)
cd "$(dirname "$0")/.."
python prepare.py --method cleandift --split trn_train > runs/logs/prepare_cleandift_trn.log 2>&1
METHOD=cleandift T3_EXTRA=2 T3_TTA=20 python scripts/dump_t3.py trn_train 1000 > runs/logs/dump_cd_trn.log 2>&1
METHOD=cleandift T3_EXTRA=2 T3_TTA=20 python scripts/dump_t3.py dev 0 > runs/logs/dump_cd_dev.log 2>&1
python -c "
from decode_lib import t3
t3.train('cache/cleandift/trn_train/t3_x2_tta20_n1000.npz','runs/t3_models/cd_x2tta_n1000_60_128.pt',60,'cache/cleandift/dev/t3_x2_tta20.npz',h=128)
" > runs/logs/train_cd.log 2>&1
OVR="T3_MODEL=runs/t3_models/cd_x2tta_n1000_60_128.pt" TAG=cdT3 TIMEOUT=3000 bash scripts/run.sh 25 test cleandift > runs/logs/cdT3_test.log 2>&1
OVR="T3_MODEL=runs/t3_models/cd_x2tta_n1000_60_128.pt" TAG=cdT3 TIMEOUT=3000 bash scripts/run.sh 25 val cleandift > runs/logs/cdT3_val.log 2>&1
