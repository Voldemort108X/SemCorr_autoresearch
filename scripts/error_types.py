"""Error-type breakdown of decode.py (with overrides) on a split: python scripts/error_types.py <split> ['A=b,...'] ...
Types of wrong@.10: mirror kp / other annotated kp / near-miss (.1-.2) / far. Analysis only (uses GT)."""
import os, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch
import prepare as P
import decode as D
split = sys.argv[1]
DEF = {k: v for k, v in vars(D).items() if k.isupper()}
for cfg in sys.argv[2:] or ['']:
    for k, v in DEF.items():
        setattr(D, k, v)
    for kv in filter(None, cfg.split(',')):
        k, v = kv.split('=')
        setattr(D, k, type(getattr(D, k))(v))
    stats = {}
    for cat in P.CATS:
        meta = P.load_split('tlr', split, cat)
        F_ = P.CatFeats('tlr', cat)
        c = stats.setdefault(cat, np.zeros(6))
        for j in range(len(meta['src'])):
            k1, k2 = meta['k1'][j], meta['k2'][j]
            vis = (k1[:, 2] > 0) & (k2[:, 2] > 0)
            if vis.sum() == 0:
                continue
            with torch.no_grad():
                pred = D.decode(P.make_pair(F_, meta, j, cat)).cpu().numpy()
            bb = float(meta['bbox'][j]); perm = meta['perm']
            for i in np.nonzero(vis)[0]:
                e = np.linalg.norm(pred[i] - k2[i, :2]) / bb
                c[0] += 1
                if e <= 0.1:
                    continue
                d = np.linalg.norm(pred[i][None] - k2[:, :2], axis=1) / bb
                d[k2[:, 2] == 0] = 9; d[i] = 9
                jn = int(d.argmin())
                if d[jn] <= 0.1:
                    c[1 if (jn == perm[i] and perm[i] != i) else 2] += 1
                elif e <= 0.2:
                    c[3] += 1
                else:
                    c[4] += 1
        del F_
    tot = sum(stats.values())
    w = tot[1:5].sum()
    print(f'[{cfg}] {split}: n={int(tot[0])} wrong@.10={100*w/tot[0]:.2f}% | mirror {100*tot[1]/tot[0]:.2f} other-kp {100*tot[2]/tot[0]:.2f} '
          f'near {100*tot[3]/tot[0]:.2f} far {100*tot[4]/tot[0]:.2f} (% of all kpts)', flush=True)
    for cat, c in stats.items():
        print(f'   {cat:12s} wrong {100*c[1:5].sum()/c[0]:5.1f} mirror {100*c[1]/c[0]:4.1f} other {100*c[2]/c[0]:4.1f} near {100*c[3]/c[0]:4.1f} far {100*c[4]/c[0]:4.1f}')
