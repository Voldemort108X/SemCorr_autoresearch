"""Collect per-(pair, forced arm) evidence on a split: python scripts/arm_stats.py <split> -> runs/an_armstats_<split>.npz"""
import os, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch
import prepare as P
import decode as D
split = sys.argv[1]
D.T3_MODEL = ''
rows = []
for cat in P.CATS:
    meta = P.load_split('tlr', split, cat)
    F_ = P.CatFeats('tlr', cat)
    for j in range(len(meta['src'])):
        vis = (meta['k1'][j][:, 2] > 0) & (meta['k2'][j][:, 2] > 0)
        if vis.sum() == 0:
            continue
        pair = P.make_pair(F_, meta, j, cat)
        rec = {'cat': cat, 'pid': int(meta['pid'][j]), 'used_flip': bool(meta['used_flip'][j]),
               'od': float(meta['od'][j]), 'fd': float(meta['fd'][j])}
        for a in (0, 1):
            D._STATS = []
            with torch.no_grad():
                D.decode(D.force_arm(pair, a) if a == 0 else D.force_arm(pair, 1))
            st = [s for s in D._STATS if s['arm'] == a]
            for k in ('score', 'sim', 'mnn', 'dcyc', 'cons', 'n'):
                rec[f'{k}{a}'] = st[0][k] if st else np.nan
        rows.append(rec)
    del F_
D._STATS = None
keys = rows[0].keys()
np.savez(f'runs/an_armstats_{split}.npz', **{k: np.array([r[k] for r in rows]) for k in keys})
print('saved', len(rows))
