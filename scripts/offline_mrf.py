"""Offline joint (per pair+arm) decoding on dumped pools: ICM with relative-geometry pairwise terms."""
import sys
import numpy as np
sys.path.insert(0, 'scripts')
from offline import load, pick, pck, devm


def groups(d):
    key = np.char.add(d['pair'], np.char.add('#', d['arm'].astype(str)))
    order = np.argsort(key, kind='stable')
    ks = key[order]
    cut = np.nonzero(ks[1:] != ks[:-1])[0] + 1
    return np.split(order, cut)


def fit_sim(q, c, w=None):
    """weighted LS similarity c ~ s R q + t (no reflection). returns (M 2x2, t)."""
    if w is None:
        w = np.ones(len(q))
    w = w / max(w.sum(), 1e-9)
    mq, mc = (w[:, None] * q).sum(0), (w[:, None] * c).sum(0)
    Q, Cc = q - mq, c - mc
    a = (w * (Q[:, 0] * Cc[:, 0] + Q[:, 1] * Cc[:, 1])).sum()
    b = (w * (Q[:, 0] * Cc[:, 1] - Q[:, 1] * Cc[:, 0])).sum()
    den = max((w * (Q ** 2).sum(1)).sum(), 1e-9)
    a, b = a / den, b / den
    M = np.array([[a, -b], [b, a]])
    return M, mc - M @ mq


def joint_decode(d, u, mu=0.5, cap=5.0, sigma=None, iters=3, thr=4.0, pool=None, ransac=True):
    pool = d['in_pool'] if pool is None else pool
    u = np.where(pool, u, -1e9)
    choice = u.argmax(1)
    cx = (d['pid'] % 60).astype(np.float64)
    cy = (d['pid'] // 60).astype(np.float64)
    out = choice.copy()
    for g in groups(d):
        n = len(g)
        if n < 3:
            continue
        q = np.stack([d['qx'][g, 0], d['qy'][g, 0]], 1).astype(np.float64)
        C = np.stack([cx[g], cy[g]], -1)                        # [n,K,2]
        ug = u[g]
        ch = choice[g].copy()
        # robust similarity from current choices: RANSAC over 2-point subsets, inliers < thr
        cc = C[np.arange(n), ch]
        best, bi = -1, None
        if ransac:
            for i in range(n):
                for j in range(i + 1, n):
                    M, t = fit_sim(q[[i, j]], cc[[i, j]])
                    s = np.sqrt(abs(np.linalg.det(M)))
                    if not (0.2 < s < 5):
                        continue
                    r = np.linalg.norm(q @ M.T + t - cc, axis=1)
                    inl = r < thr
                    if inl.sum() > best:
                        best, bi = inl.sum(), inl
        if bi is None or bi.sum() < 2:
            bi = np.ones(n, bool)
        M, t = fit_sim(q[bi], cc[bi])
        W = np.ones((n, n)) if sigma is None else np.exp(-((q[:, None] - q[None]) ** 2).sum(-1) / (2 * sigma ** 2))
        np.fill_diagonal(W, 0)
        dq = (q[None, :, :] - q[:, None, :]) @ M.T             # [i,j,2] expected c_j - c_i
        for _ in range(iters):
            changed = False
            for i in range(n):
                cj = C[np.arange(n), ch]                         # [n,2]
                # for each candidate a of i: sum_j W_ij min(|cj - C_ia - dq_ij|, cap)
                diff = cj[None, :, :] - C[i][:, None, :] - dq[i][None]   # [K,n,2]
                psi = np.minimum(np.linalg.norm(diff, axis=-1), cap)     # [K,n]
                pen = (psi * W[i][None]).sum(1) / max(W[i].sum(), 1e-9)
                a = int(np.argmax(ug[i] - mu * pen / 60))
                if a != ch[i]:
                    ch[i] = a
                    changed = True
            if not changed:
                break
        out[g] = ch
    return out


if __name__ == '__main__':
    d = load(sys.argv[1] if len(sys.argv) > 1 else 'cache/tlr/dev/pools.npz')
    S, dc = d['sim'], d['d_cyc'] / 60
    e1 = d['err'][:, 0]
    r = np.arange(len(e1))
    base = S - 0.5 * dc
    print('dv', devm(pick(d, base), e1))
    for mu in (0.25, 0.5, 1.0, 2.0):
        for cap in (3.0, 5.0, 8.0):
            for sigma in (None, 10.0):
                ch = joint_decode(d, base, mu=mu, cap=cap, sigma=sigma)
                e = d['err'][r, ch]
                print(f'mrf mu{mu} cap{cap} sigma{sigma}: dev {devm(e, e1):+.3f} pck {np.round(pck(e), 2)}', flush=True)
