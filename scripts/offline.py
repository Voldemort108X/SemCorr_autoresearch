"""Offline re-ranking on dumped pools: score = f(cues) -> argmax within pool -> PCK. Analysis helper."""
import numpy as np
THRS = (0.10, 0.05, 0.01)


def load(path):
    d = dict(np.load(path))
    for k in list(d):
        if d[k].dtype == np.float32 and d[k].ndim == 2:
            d[k] = np.nan_to_num(d[k], nan=1e3)
    d['in_pool'] = d['in_pool'] > 0.5
    return d


def pick(d, score, pool=None):
    pool = d['in_pool'] if pool is None else pool
    s = np.where(pool, score, -1e9)
    b = s.argmax(1)
    return d['err'][np.arange(len(b)), b]


def pck(e):
    return np.array([100 * (e <= t).mean() for t in THRS])


def devm(e, e1):
    return float((pck(e) - pck(e1)).mean())


def report(d, name, score, pool=None, base=None):
    e = pick(d, score, pool)
    e1 = d['err'][:, 0]
    p = pck(e)
    print(f'{name:40s} dev {devm(e, e1):+.3f}  pck {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}')
    return e
