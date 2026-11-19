"""Per-pair test-time channel re-weighting (label-free): learn w in R^D (softplus, init 1) maximising the dual-softmax
mutual-match confidence between the masked source and target descriptors; returns re-weighted (renormalised) pair."""
import torch
import torch.nn.functional as F


def adapt_pair(pair, steps=20, lr=0.05, tau=0.05, reg=1.0, a=0, both=False, obj='dsmax', rank=0, groups=0, sep=False, hand=0.0, qw=0.0, init=None, return_w=False, sem=0.0, blocknorm=False, nomask=False, pseudo=None, pw=1.0):
    A = pair['src'][a]
    X = A['desc'] if nomask else A['desc'][A['mask']]
    X2 = pair['src'][1 - a]['desc'][pair['src'][1 - a]['mask']]
    if sem > 0:   # source-side semantic contrast from the annotated keypoints + mirror permutation (no target labels)
        nP = pair['nP']
        k0, v0, perm = pair['src'][0]['kp'], pair['src'][0]['kp_valid'], pair['perm']
        jc = perm.clamp(min=0)
        okp = v0 & (perm >= 0) & (perm != torch.arange(perm.numel(), device=perm.device)) & v0[jc]
        if int(okp.sum()) >= 2:
            li, lj = k0[okp], k0[jc[okp]]
            mi = (li // nP) * nP + (nP - 1 - li % nP)
            mj = (lj // nP) * nP + (nP - 1 - lj % nP)
            D0, D1 = pair['src'][0]['desc'], pair['src'][1]['desc']
            Qi, Vpos, Uneg, Qj = D0[li], D1[mj], D1[mi], D0[lj]
        else:
            sem = 0.0
    if hand != 0.0:
        nP = pair['nP']
        pidx = torch.nonzero(A['mask'], as_tuple=True)[0]
        mir = (pidx // nP) * nP + (nP - 1 - pidx % nP)
        Xo, Xm = A['desc'][pidx], pair['src'][1 - a]['desc'][mir]
    Y = pair['trg']['desc'] if nomask else pair['trg']['desc'][pair['trg']['mask']]
    if X.shape[0] < 4 or Y.shape[0] < 4:
        return pair
    D = X.shape[1]
    G = groups if groups > 0 else D
    gidx = (torch.arange(D, device=X.device) * G // D)
    th = torch.zeros(G, device=X.device) if init is None else init.clone().to(X.device)
    th.requires_grad_(True)
    th_t = torch.zeros(G, device=X.device, requires_grad=True)
    params = [th, th_t] if sep else [th]
    if rank > 0:
        g = torch.Generator(device=X.device); g.manual_seed(0)
        U = (torch.randn(D, rank, device=X.device, generator=g) / D ** 0.5).requires_grad_(True)
        V = torch.zeros(D, rank, device=X.device, requires_grad=True)
        params += [U, V]
    opt = torch.optim.Adam(params, lr=lr)
    def tf(Z, w):
        Z2 = Z * w
        if rank > 0:
            Z2 = Z2 + (Z @ U) @ V.t()
        if blocknorm and Z2.shape[1] == 3968:   # keep TLR's per-block (SD | DINO) normalisation
            Z2 = torch.cat([F.normalize(Z2[:, :3200], dim=1), F.normalize(Z2[:, 3200:], dim=1)], 1)
        return F.normalize(Z2, dim=1)
    for _ in range(steps):
        w = F.softplus(th + 0.5413)[gidx]              # softplus(0.5413) = 1
        wt = F.softplus(th_t + 0.5413)[gidx] if sep else w
        Yw = tf(Y, wt)
        loss = reg * ((w - 1) ** 2).mean()
        if hand != 0.0:   # +: make a patch and its horizontal mirror (other arm) DISsimilar -> handedness channels
            loss = loss + hand * (tf(Xo, w) * tf(Xm, w)).sum(1).mean()
        for Xa in ([X, X2] if both else [X]):
            Xw = tf(Xa, w)
            S = Xw @ Yw.t() / tau
            if obj == 'dsmax':
                P = torch.softmax(S, 1) * torch.softmax(S, 0)
                loss = loss - P.max(1)[0].mean() - P.max(0)[0].mean()
            elif obj == 'ent':      # minimise row/col softmax entropy
                lr_, lc_ = torch.log_softmax(S, 1), torch.log_softmax(S, 0)
                loss = loss + (-(lr_.exp() * lr_).sum(1).mean() - (lc_.exp() * lc_).sum(0).mean()) * 0.1
            elif obj == 'cyc':      # soft cycle: source -> target -> source returns to itself
                P1, P2 = torch.softmax(S, 1), torch.softmax(S.t(), 1)
                loss = loss - torch.diagonal(P1 @ P2).clamp(min=1e-9).log().mean() * 0.1
            elif obj == 'both':
                P = torch.softmax(S, 1) * torch.softmax(S, 0)
                P1, P2 = torch.softmax(S, 1), torch.softmax(S.t(), 1)
                loss = loss - P.max(1)[0].mean() - P.max(0)[0].mean() - torch.diagonal(P1 @ P2).clamp(min=1e-9).log().mean() * 0.1
        if sem > 0:
            a_ = tf(Qi, w)
            sp, sn1, sn2 = (a_ * tf(Vpos, w)).sum(1), (a_ * tf(Uneg, w)).sum(1), (a_ * tf(Qj, w)).sum(1)
            logits = torch.stack([sp, sn1, sn2], 1) / 0.05
            loss = loss + sem * torch.nn.functional.cross_entropy(logits, torch.zeros(len(sp), dtype=torch.long, device=sp.device))
        if pseudo is not None:   # self-training: pseudo-labels (query patch idx, target patch idx) from a first decode
            qi, ti = pseudo
            Yfull = tf(pair['trg']['desc'], wt)
            Qp = tf(A['desc'][qi], w)
            Sp = (Qp @ Yfull.t() / tau).masked_fill(~pair['trg']['mask'].unsqueeze(0), -1e4)
            loss = loss + pw * torch.nn.functional.cross_entropy(Sp, ti)
        if qw > 0:        # extra confidence term on the query keypoint rows (dual-softmax restricted to the queries)
            Q = tf(A['desc'][A['kp'][A['kp_valid']]], w)
            if Q.shape[0] > 0:
                Sq = Q @ Yw.t() / tau
                Pq = torch.softmax(Sq, 1) * torch.softmax(torch.cat([S, Sq], 0), 0)[-Q.shape[0]:]
                loss = loss - qw * Pq.max(1)[0].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w = F.softplus(th + 0.5413)[gidx]
        wt = F.softplus(th_t + 0.5413)[gidx] if sep else w
        if return_w:
            return th.detach()
        src = [dict(S_, desc=tf(S_['desc'], w)) for S_ in pair['src']]
        return dict(pair, src=src, trg=dict(pair['trg'], desc=tf(pair['trg']['desc'], wt)))
