"""Per-pair test-time channel re-weighting (label-free): learn w in R^D (softplus, init 1) maximising the dual-softmax
mutual-match confidence between the masked source and target descriptors; returns re-weighted (renormalised) pair."""
import torch
import torch.nn.functional as F


def adapt_pair(pair, steps=20, lr=0.05, tau=0.05, reg=1.0, a=0):
    A = pair['src'][a]
    X = A['desc'][A['mask']]
    Y = pair['trg']['desc'][pair['trg']['mask']]
    if X.shape[0] < 4 or Y.shape[0] < 4:
        return pair
    D = X.shape[1]
    th = torch.zeros(D, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([th], lr=lr)
    for _ in range(steps):
        w = F.softplus(th + 0.5413)                     # softplus(0.5413) = 1
        Xw, Yw = F.normalize(X * w, dim=1), F.normalize(Y * w, dim=1)
        S = Xw @ Yw.t() / tau
        P = torch.softmax(S, 1) * torch.softmax(S, 0)
        loss = -P.max(1)[0].mean() - P.max(0)[0].mean() + reg * ((w - 1) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w = F.softplus(th + 0.5413)
        f = lambda d: F.normalize(d * w, dim=1)
        src = [dict(S_, desc=f(S_['desc'])) for S_ in pair['src']]
        return dict(pair, src=src, trg=dict(pair['trg'], desc=f(pair['trg']['desc'])))
