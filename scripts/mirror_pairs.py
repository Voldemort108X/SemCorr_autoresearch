"""Mirror-pair outcomes of decode.py on a split: for mirror pairs (i, perm i) both evaluated, classify
(both correct / swap / collision (both on the same target spot) / one wrong / both wrong). Analysis only."""
import os, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch
import prepare as P
import decode as D
split = sys.argv[1]
for kv in filter(None, (sys.argv[2] if len(sys.argv) > 2 else '').split(',')):
    k, v = kv.split('='); setattr(D, k, (v == 'True') if isinstance(getattr(D, k), bool) else type(getattr(D, k))(v))
cnt = dict(n=0, both_ok=0, swap=0, collide=0, one_wrong=0, both_wrong_other=0)
for cat in P.CATS:
    meta = P.load_split('tlr', split, cat); F_ = P.CatFeats('tlr', cat); perm = meta['perm']
    for j in range(len(meta['src'])):
        k1, k2 = meta['k1'][j], meta['k2'][j]
        vis = (k1[:, 2] > 0) & (k2[:, 2] > 0)
        if vis.sum() == 0:
            continue
        with torch.no_grad():
            pred = D.decode(P.make_pair(F_, meta, j, cat)).cpu().numpy()
        bb = float(meta['bbox'][j])
        for i in np.nonzero(vis)[0]:
            q = perm[i]
            if q <= i or not vis[q]:
                continue
            cnt['n'] += 1
            ei = np.linalg.norm(pred[i] - k2[i, :2]) / bb <= .1
            eq = np.linalg.norm(pred[q] - k2[q, :2]) / bb <= .1
            if ei and eq:
                cnt['both_ok'] += 1
            elif np.linalg.norm(pred[i] - k2[q, :2]) / bb <= .1 and np.linalg.norm(pred[q] - k2[i, :2]) / bb <= .1:
                cnt['swap'] += 1
            elif np.linalg.norm(pred[i] - pred[q]) / bb <= .05:
                cnt['collide'] += 1
            elif ei or eq:
                cnt['one_wrong'] += 1
            else:
                cnt['both_wrong_other'] += 1
    del F_
n = cnt['n']
print(split, sys.argv[2:] , {k: (v if k == 'n' else round(100 * v / n, 2)) for k, v in cnt.items()})
