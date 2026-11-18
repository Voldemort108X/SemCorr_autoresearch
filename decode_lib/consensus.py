"""Pair-level consensus over the keypoints' candidate pools (RANSAC on candidate hypotheses).

Each valid query keypoint i has source position q_i and candidates c_ik with unary u_ik. Hypotheses: pick a few
keypoints, one candidate each (sampled from their top-T by unary), fit a transform (similarity or affine), score =
number of keypoints having some candidate within thr of T(q_i) (weighted by unary rank). Returns the best transform
and per-candidate residual |c_ik - T(q_i)| (patch units)."""
import torch


def _fit_sim(X, Y):
    """2-point similarity (no reflection) per hypothesis. X,Y [H,2,2] -> B [H,3,2] (affine form)."""
    dx = X[:, 1] - X[:, 0]
    dy = Y[:, 1] - Y[:, 0]
    den = (dx ** 2).sum(-1).clamp(min=1e-6)
    a = (dx * dy).sum(-1) / den                         # s cos
    b = (dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]) / den   # s sin
    R = torch.stack([torch.stack([a, b], -1), torch.stack([-b, a], -1)], 1)   # row-vector form: y = x R + t
    t = Y[:, 0] - (X[:, 0].unsqueeze(1) @ R)[:, 0]
    return torch.cat([R, t.unsqueeze(1)], 1)


def _fit_aff(X, Y, w=None):
    Xh = torch.cat([X, torch.ones_like(X[..., :1])], -1)
    if w is None:
        w = torch.ones(X.shape[:-1], device=X.device)
    W = w.unsqueeze(-1)
    A = (Xh * W).transpose(-1, -2) @ Xh + torch.eye(3, device=X.device) * 1e-4
    return torch.linalg.solve(A, (Xh * W).transpose(-1, -2) @ Y)


def _apply(B, X):
    return torch.cat([X, torch.ones_like(X[..., :1])], -1) @ B


def consensus(q, c, u, valid, model='sim', H=1024, T=5, thr=4.0, gen=None):
    """q [M,2], c [M,K,2], u [M,K] (higher better; rank order matters), valid [M] bool.
    Returns resid [M,K] (patch units; zeros if < 2-3 valid keypoints), info dict."""
    dev = q.device
    M, K = u.shape
    vidx = torch.nonzero(valid, as_tuple=True)[0]
    n = vidx.numel()
    need = 2 if model == 'sim' else 3
    if n < need:
        return torch.zeros(M, K, device=dev), dict(ok=False, n=n)
    order = u.argsort(1, descending=True)[:, :min(T, K)]                    # top-T by unary
    Tt = order.shape[1]
    # sample keypoints (distinct) and candidate ranks
    r = torch.rand(H, n, device=dev, generator=gen)
    kp = vidx[r.argsort(1)[:, :need]]                                       # [H,need]
    rk = torch.randint(0, Tt, (H, need), device=dev, generator=gen)
    # always include the all-top-1 hypotheses for a few keypoint subsets
    rk[: H // 8] = 0
    cand = order[kp, rk]                                                    # [H,need]
    X = q[kp]                                                               # [H,need,2]
    Y = c[kp, cand]                                                         # [H,need,2]
    B = _fit_sim(X, Y) if model == 'sim' else _fit_aff(X, Y)
    pred = _apply(B, q.unsqueeze(0).expand(H, -1, -1))                       # [H,M,2]
    res = (c.unsqueeze(0) - pred.unsqueeze(2)).norm(dim=-1)                 # [H,M,K]
    # support: per keypoint, best soft inlier score over its top-T candidates
    sub = torch.gather(res, 2, order.unsqueeze(0).expand(H, -1, -1))         # [H,M,T]
    inl = (sub < thr).float() * (1.0 / (1.0 + torch.arange(Tt, device=dev).float()))  # rank-discounted
    sup = (inl.max(2)[0] * valid.float().unsqueeze(0)).sum(1)               # [H]
    # degenerate similarity guard: scale must be sane
    if model == 'sim':
        s = B[:, 0, :].norm(dim=-1)
    else:
        s = torch.det(B[:, :2, :2]).abs().clamp(min=1e-6).sqrt()
    sup = torch.where((s > 0.2) & (s < 5.0), sup, torch.full_like(sup, -1.0))
    best = sup.argmax()
    Bb = B[best]
    # refit on inliers: per keypoint take its best (nearest) top-T candidate if within thr
    rb = sub[best]                                                          # [M,T]
    j = rb.argmin(1)
    ok = (rb.min(1)[0] < thr) & valid
    if int(ok.sum()) >= 3:
        Xi = q[ok]
        Yi = c[ok.nonzero(as_tuple=True)[0], order[ok, j[ok]]]
        if model == 'sim' and int(ok.sum()) >= 2:
            Bb = _fit_aff(Xi, Yi) if int(ok.sum()) >= 4 else Bb
        else:
            Bb = _fit_aff(Xi, Yi)
    pred = _apply(Bb, q)
    resid = (c - pred.unsqueeze(1)).norm(dim=-1)
    return resid, dict(ok=True, n=n, support=float(sup[best]), n_inl=int(ok.sum()), B=Bb)
