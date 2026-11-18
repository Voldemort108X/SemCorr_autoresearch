"""Average TTA channel weights (theta) over unlabeled SPair trn_train pairs -> runs/tta_prior.npy (no labels used)."""
import os, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch
import prepare as P
from decode_lib.tta import adapt_pair
per = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(0)
ths = []
for cat in P.CATS:
    meta = P.load_split('tlr', 'trn_train', cat)
    F_ = P.CatFeats('tlr', cat)
    for j in rng.choice(len(meta['src']), per, replace=False):
        pair = P.make_pair(F_, meta, j, cat)
        with torch.enable_grad():
            th = adapt_pair(pair, steps=20, reg=0.0, return_w=True)
        if torch.is_tensor(th):
            ths.append(th.cpu().numpy())
    del F_
    print(cat, len(ths), flush=True)
T = np.stack(ths)
np.save('runs/tta_prior.npy', T.mean(0))
print('saved', T.shape, 'mean |theta|', np.abs(T.mean(0)).mean(), 'per-pair std', T.std(0).mean())
