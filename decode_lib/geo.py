"""Geometry helpers for global (pair-level) decoders. All coordinates in patch units (x, y)."""
import torch


def pid_xy(pid, nP=60):
    return torch.stack([(pid % nP).float(), (pid // nP).float()], -1)


def mnn_anchors(st):
    """Mutual nearest neighbours over the masked similarity. Returns src_xy [A,2], trg_xy [A,2], sim [A]."""
    S = st.sim_valid
    f = S.argmax(1)                     # [Ps] -> valid trg idx
    b = S.argmax(0)                     # [Pt] -> valid src idx
    s_idx = torch.arange(S.shape[0], device=S.device)
    mutual = b[f] == s_idx
    si, ti = s_idx[mutual], f[mutual]
    return (pid_xy(st.src_valid_idx[si], st.num_patches), pid_xy(st.tgt_valid_idx[ti], st.num_patches),
            S[si, ti])


def fit_affine_wls(X, Y, w):
    """Weighted LS affine Y ~ [X 1] B. X,Y [..., A, 2], w [..., A]. Returns B [..., 3, 2]."""
    Xh = torch.cat([X, torch.ones_like(X[..., :1])], -1)
    W = w.unsqueeze(-1)
    XtX = (Xh * W).transpose(-1, -2) @ Xh
    XtY = (Xh * W).transpose(-1, -2) @ Y
    eye = torch.eye(3, device=X.device) * 1e-3
    return torch.linalg.pinv(XtX + eye) @ XtY


def apply_affine(B, X):
    Xh = torch.cat([X, torch.ones_like(X[..., :1])], -1)
    return Xh @ B


def ransac_affine(X, Y, w=None, iters=512, thr=3.0, gen=None):
    """RANSAC affine on anchors; returns B [3,2], inlier mask [A]. Hypotheses scored by weighted inlier count."""
    A = X.shape[0]
    dev = X.device
    if A < 3:
        B = torch.zeros(3, 2, device=dev)
        B[0, 0] = B[1, 1] = 1
        return B, torch.ones(A, dtype=torch.bool, device=dev)
    if w is None:
        w = torch.ones(A, device=dev)
    idx = torch.randint(0, A, (iters, 3), device=dev, generator=gen)
    Xs, Ys = X[idx], Y[idx]                                   # [I,3,2]
    Bh = fit_affine_wls(Xs, Ys, torch.ones(iters, 3, device=dev))
    res = (apply_affine(Bh, X.unsqueeze(0).expand(iters, -1, -1)) - Y.unsqueeze(0)).norm(dim=-1)  # [I,A]
    score = ((res < thr).float() * w.unsqueeze(0)).sum(1)
    best = score.argmax()
    inl = res[best] < thr
    if inl.sum() >= 3:
        B = fit_affine_wls(X[inl], Y[inl], w[inl])
        inl = (apply_affine(B, X) - Y).norm(dim=-1) < thr
    else:
        B = Bh[best]
    return B, inl


def mls_affine(Xq, X, Y, w, sigma):
    """Moving-least-squares affine prediction of query points Xq [Q,2] from anchors (X->Y) with Gaussian
    spatial weights (sigma, patch units) times anchor weights w [A]. Returns predicted Yq [Q,2]."""
    d2 = ((Xq.unsqueeze(1) - X.unsqueeze(0)) ** 2).sum(-1)    # [Q,A]
    W = torch.exp(-d2 / (2 * sigma ** 2)) * w.unsqueeze(0)
    Q = Xq.shape[0]
    B = fit_affine_wls(X.unsqueeze(0).expand(Q, -1, -1), Y.unsqueeze(0).expand(Q, -1, -1), W)
    return apply_affine(B, Xq.unsqueeze(1))[:, 0]


def mask_scale(mask, nP=60):
    """sqrt(area) of a patch mask (patch units)."""
    return mask.float().sum().clamp(min=1).sqrt()


def context_score(src_desc, trg_desc, trg_mask, qxy, cxy, radii=(3, 6), ndir=8, scale=1.0, pool=1, nP=60):
    """Neighbourhood-context agreement of candidates. For offsets d (radius r, ndir directions) around the query
    q (source, patch units) and the scaled offset s*d around candidate c (target), average over offsets of
    maxpool_{(2pool+1)^2}( S[q+d, :] )[c + s*d]. Source offsets falling outside the grid are skipped; target
    positions outside the mask/grid score -1. Returns dict r -> [M,K]."""
    dev = src_desc.device
    M, K = cxy.shape[:2]
    out = {}
    ang = torch.arange(ndir, device=dev).float() * (2 * torch.pi / ndir)
    dirs = torch.stack([torch.cos(ang), torch.sin(ang)], -1)                    # [ndir,2]
    tm2d = trg_mask.view(1, 1, nP, nP).float()
    for r in radii:
        off = dirs * r                                                           # [ndir,2]
        sp = (qxy.unsqueeze(1) + off.unsqueeze(0)).round().long()                # [M,ndir,2]
        sval = (sp >= 0).all(-1) & (sp < nP).all(-1)
        spc = sp.clamp(0, nP - 1)
        sidx = spc[..., 1] * nP + spc[..., 0]                                    # [M,ndir]
        sim = src_desc[sidx.reshape(-1)] @ trg_desc.t()                          # [M*ndir, P]
        sim = sim.view(-1, 1, nP, nP).masked_fill(tm2d == 0, -1.0)
        if pool > 0:
            sim = torch.nn.functional.max_pool2d(sim, 2 * pool + 1, 1, pool)
        sim = sim.view(M, ndir, nP * nP)
        tp = (cxy.unsqueeze(2) + scale * off.view(1, 1, ndir, 2)).round().long()  # [M,K,ndir,2]
        tval = (tp >= 0).all(-1) & (tp < nP).all(-1)
        tpc = tp.clamp(0, nP - 1)
        tidx = tpc[..., 1] * nP + tpc[..., 0]                                    # [M,K,ndir]
        g = torch.gather(sim.unsqueeze(1).expand(-1, K, -1, -1), 3, tidx.unsqueeze(-1))[..., 0]  # [M,K,ndir]
        g = torch.where(tval, g, torch.full_like(g, -1.0))
        w = sval.unsqueeze(1).float().expand(-1, K, -1)
        out[r] = (g * w).sum(-1) / w.sum(-1).clamp(min=1)
    return out
