"""Offline pair-arm rules on a SOCO run decoded in both arms (SOCO_ARM=both [+ SOCO_DENSE=1]).
usage: python scripts/soco_armrules.py <run_dir>   -> SOC of each label-free rule (per pair -> per class -> mean)"""
import json, sys
import numpy as np
d = sys.argv[1]
A = json.load(open(d + '/soco_armstats.json')); P = json.load(open(d + '/soco_perpair.json'))
cls = sorted(A)
rows = []
for ci, c in enumerate(cls):
    for i, a in enumerate(A[c]):
        s0, s1 = a['s0'] or {}, a['s1'] or {}
        dn0, dn1 = a.get('dn0', {}), a.get('dn1', {})
        rows.append(dict(c=ci, y0=P['ours0'][c][i][0], y1=P['ours1'][c][i][0],
                         kc=s1.get('cons', 0) - s0.get('cons', 0), dist=a['d0'] - a['d1'],
                         da=dn1.get('all', 0) - dn0.get('all', 0), dt=dn1.get('top', 0) - dn0.get('top', 0)))
C = np.array([r['c'] for r in rows]); Y0 = np.array([r['y0'] for r in rows]); Y1 = np.array([r['y1'] for r in rows])
f = {k: np.array([r[k] for r in rows]) for k in ('kc', 'dist', 'da', 'dt')}
def soc(pick):
    v = np.where(pick, Y1, Y0); return 100 * np.mean([v[C == k].mean() for k in range(len(cls))])
def tie(primary, fallback):
    return np.where(primary != 0, primary > 0, fallback > 0)
R = {'arm0': np.zeros(len(C), bool), 'arm1': np.ones(len(C), bool), 'oracle': Y1 > Y0,
     'dist': f['dist'] > 0, 'kcons (S12)': tie(f['kc'], f['dist']),
     'dense all': tie(f['da'], f['dist']), 'dense top100': tie(f['dt'], f['dist']),
     'dense all + kcons': tie(f['da'] + f['kc'], f['dist']), 'dense top + kcons': tie(f['dt'] + f['kc'], f['dist']),
     'kcons, tie->dense all': tie(f['kc'], np.where(f['da'] != 0, f['da'], f['dist']))}
for k, v in R.items():
    agree = np.mean(v == (Y1 > Y0)) if k != 'oracle' else 1.0
    print(f'{k:24s} SOC {soc(v):6.2f}')
# paired bootstrap (pairs resampled within class) of each rule vs kcons (S12)
rng = np.random.default_rng(0)
base = np.where(R['kcons (S12)'], Y1, Y0)
for k in ('dense all', 'dense top100', 'dense top + kcons'):
    dv = np.where(R[k], Y1, Y0) - base
    idx = [np.nonzero(C == j)[0] for j in range(len(cls))]
    bs = [100 * np.mean([dv[rng.choice(ix, len(ix))].mean() for ix in idx]) for _ in range(1000)]
    print(f'{k:24s} vs S12: {100*np.mean([dv[ix].mean() for ix in idx]):+.2f} ± {np.std(bs):.2f}')
