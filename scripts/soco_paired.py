"""Paired bootstrap of SOC/CC between the 'ours' head of two SOCO runs (same split / pairs).
usage: python scripts/soco_paired.py <base_dir> <dir> [<dir> ...]   (resample pairs within class, 1000x)"""
import json, sys
import numpy as np
def load(d):
    return json.load(open(f'{d}/soco_perpair.json'))['ours']
base = load(sys.argv[1])
rng = np.random.default_rng(0)
for d in sys.argv[2:]:
    o = load(d)
    cls = sorted(base)
    diff = [np.array(o[c]) - np.array(base[c]) for c in cls]   # [n,2] per class
    mean = np.mean([x.mean(0) for x in diff], 0) * 100
    bs = []
    for _ in range(1000):
        bs.append(np.mean([x[rng.integers(0, len(x), len(x))].mean(0) for x in diff], 0) * 100)
    sd = np.std(bs, 0)
    print(f'{d.split("/")[-1]:45s} dSOC {mean[0]:+.2f} ± {sd[0]:.2f}   dCC {mean[1]:+.2f} ± {sd[1]:.2f}')
