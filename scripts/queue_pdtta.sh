#!/usr/bin/env bash
cd "$(dirname "$0")/.."
T3_EXTRA=2 T3_TTA=20 T3_PERMDROP=1 python scripts/dump_t3.py trn_train 1000 > runs/logs/dump_pdtta.log 2>&1
python - <<'PY' > runs/logs/train_pdtta.log 2>&1
import numpy as np
a=np.load('cache/tlr/trn_train/t3_x2_tta20.npz'); b=np.load('cache/tlr/trn_train/t3_x2_pd1_tta20_n1000.npz')
out={k:(np.concatenate([a[k],b[k]]) if a[k].shape[:1]==a['F'].shape[:1] else a[k]) for k in a.files}
np.savez('cache/tlr/trn_train/t3_x2_tta_mixpd.npz', **out)
from decode_lib import t3
t3.train('cache/tlr/trn_train/t3_x2_tta_mixpd.npz','runs/t3_models/x2tta_mixpd_60_128.pt',60,None,h=128)
PY
export TIMEOUT=3000 OVR="T3_MODEL=runs/t3_models/x2tta_mixpd_60_128.pt" TAG=pdtta
bash scripts/run.sh 22 val > runs/logs/pdtta_val.log 2>&1
bash scripts/run.sh 22 spairu_test > runs/logs/pdtta_spairu.log 2>&1
python prepare_gen.py eval pascal 22 tlr > runs/logs/pdtta_pascal.log 2>&1
