"""Failure breakdown of a re-ranking rule on dumped pools. python scripts/analyze_pools.py <pools.npz> [lam]"""
import sys
import numpy as np
sys.path.insert(0, 'scripts')
from offline import load, pck

d = load(sys.argv[1])
lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
score = d['sim'] - lam * d['d_cyc'] / 60
s = np.where(d['in_pool'], score, -1e9)
b = s.argmax(1)
r = np.arange(len(b))
e = d['err'][r, b]
wrong = e > 0.1
inpool_ok = (np.where(d['in_pool'], d['err'], 9) <= 0.1).any(1)
near = d['near_kp'][r, b]
mir = d['is_mirror'][r, b] > 0
print(f'n={len(e)} pck10={100*(1-wrong.mean()):.2f}  wrong={100*wrong.mean():.2f}%')
print(f'  of wrong: correct cand in pool {100*inpool_ok[wrong].mean():.1f}% | pool miss {100*(~inpool_ok[wrong]).mean():.1f}%')
print(f'  of wrong: lands on mirror kp {100*mir[wrong].mean():.1f}% | other kp {100*((near>=0)&~mir)[wrong].mean():.1f}% '
      f'| near-miss (0.1-0.2) {100*((near<0)&(e<=0.2))[wrong].mean():.1f}% | far {100*((near<0)&(e>0.2))[wrong].mean():.1f}%')
# recoverable among wrong with correct in pool: rank of best
print('per category pck10 dv vs oracle-in-pool:')
for c in np.unique(d['cat']):
    m = d['cat'] == c
    print(f'  {c:12s} n={m.sum():5d} dv {100*(e[m]<=0.1).mean():5.1f} top1 {100*(d["err"][m,0]<=0.1).mean():5.1f} '
          f'pool-oracle {100*inpool_ok[m].mean():5.1f} mirror-err {100*(mir&wrong)[m].mean():4.1f}')
# by number of valid query keypoints
if 'n_valid' in d:
    nv = d['n_valid'][:, 0]
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 99)):
        m = (nv >= lo) & (nv < hi)
        if m.any():
            print(f'  n_valid [{lo},{hi}) n={m.sum()} dv {100*(e[m]<=0.1).mean():.1f} pool-oracle {100*inpool_ok[m].mean():.1f}')
