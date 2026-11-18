"""Per-pair test-time channel re-weighting (label-free): learn w in R^D (softplus, init 1) maximising the dual-softmax
mutual-match confidence between the masked source and target descriptors; returns re-weighted (renormalised) pair."""
import torch
import torch.nn.functional as F


def adapt_pair(pair, steps=20, lr=0.05, tau=0.05, reg=1.0, a=0, both=False, obj='dsmax', rank=0):
    A = pair['src'][a]
    X = A['desc'][A['mask']]
    X2 = pair['src'][1 - a]['desc'][pair['src'][1 - a]['mask']]
    Y = pair['trg']['desc'][pair['trg']['mask']]
    if X.shape[0] < 4 or Y.shape[0] < 4:
        return pair
    D = X.shape[1]
    th = torch.zeros(D, device=X.device, requires_grad=True)
    params = [th]
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
        return F.normalize(Z2, dim=1)
    for _ in range(steps):
        w = F.softplus(th + 0.5413)                     # softplus(0.5413) = 1
        Yw = tf(Y, w)
        loss = reg * ((w - 1) ** 2).mean()
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
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w = F.softplus(th + 0.5413)
        f = lambda d: tf(d, w)
        src = [dict(S_, desc=f(S_['desc'])) for S_ in pair['src']]
        return dict(pair, src=src, trg=dict(pair['trg'], desc=f(pair['trg']['desc'])))
