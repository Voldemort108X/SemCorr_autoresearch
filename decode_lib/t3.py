"""T3: listwise light re-ranker over the seq-5 candidate pool (trained on SPair trn_train only; category-agnostic).
Train: python -m decode_lib.t3 <out.pt> [epochs] [train_table] [eval_table]
Objective: maximise the expected metric gain sum_k softmax(s)_k * g_k, g_k = mean_t 1[err_k <= t] (t = .10/.05/.01),
over the in-pool candidates (the oracle target of the selection metric)."""
import sys
import time
import numpy as np
import torch
import torch.nn as nn

THRS = (0.10, 0.05, 0.01)


class Reranker(nn.Module):
    def __init__(self, nf, h=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.register_buffer('mu', torch.zeros(nf))
        self.register_buffer('sd', torch.ones(nf))
        self.register_buffer('fmask', torch.ones(nf))      # 0 = feature dropped (ablation)

    def forward(self, F, pool):
        x = (F - self.mu) / self.sd * self.fmask
        h = self.inp(x)                                                    # [B,K,h]
        m = pool.float().unsqueeze(-1)
        ctx = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        s = self.out(torch.cat([h, ctx.expand_as(h)], -1))[..., 0]
        # residual on the training-free score (feature 'base' scaled): start near seq-5
        return s.masked_fill(~pool, -1e4)


def gains(err):
    return np.mean([(err <= t) for t in THRS], 0).astype(np.float32)


def load_table(path):
    d = np.load(path)
    F = torch.from_numpy(d['F'].astype(np.float32))
    pool = torch.from_numpy(d['in_pool'].astype(bool))
    g = torch.from_numpy(gains(d['err']))
    return d, F, pool, g


def evaluate(model, F, pool, err, dev='cuda'):
    with torch.no_grad():
        s = []
        for i in range(0, len(F), 8192):
            s.append(model(F[i:i + 8192].to(dev), pool[i:i + 8192].to(dev)).cpu())
        b = torch.cat(s).argmax(1).numpy()
    e = err[np.arange(len(b)), b]
    return np.array([100 * (e <= t).mean() for t in THRS]), b


def train(train_path, out_path, epochs=30, eval_path=None, seed=0, h=64, lr=2e-3, wd=1e-4, base_w=None, drop=()):
    torch.manual_seed(seed)
    dev = 'cuda'
    d, F, pool, g = load_table(train_path)
    names = list(d['feat_names'])
    model = Reranker(F.shape[-1], h).to(dev)
    Fm = F[pool]
    model.mu.copy_(Fm.mean(0).to(dev))
    model.sd.copy_(Fm.std(0).clamp(min=1e-3).to(dev))
    for n_ in drop:
        model.fmask[names.index(n_)] = 0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(F)
    bs = 1024
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Fd, pd_, gd = F.to(dev), pool.to(dev), g.to(dev)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(n // bs):
            idx = perm[i * bs:(i + 1) * bs]
            s = model(Fd[idx], pd_[idx])
            p = torch.softmax(s, 1)
            loss = -(p * gd[idx]).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if eval_path and (ep % 10 == 9 or ep == epochs - 1):
            de, Fe, pe, _ = load_table(eval_path)
            pk, _ = evaluate(model, Fe, pe, de['err'])
            bb = np.where(de['in_pool'], de['base'], -1e9).argmax(1)
            eb = de['err'][np.arange(len(bb)), bb]
            pb = np.array([100 * (eb <= t).mean() for t in THRS])
            print(f'ep {ep} loss {tot / (n // bs):.4f} eval pck {np.round(pk, 2)} base {np.round(pb, 2)} '
                  f'Δmean {np.mean(pk - pb):+.3f} ({time.time() - t0:.0f}s)', flush=True)
    torch.save(dict(state=model.state_dict(), nf=F.shape[-1], h=h, names=names), out_path)
    return model


def load_model(path, dev='cuda'):
    ck = torch.load(path, map_location=dev)
    m = Reranker(ck['nf'], ck['h']).to(dev)
    st = ck['state']
    if 'fmask' not in st:
        st = dict(st, fmask=torch.ones(ck['nf'], device=dev))
    m.load_state_dict(st)
    m.eval()
    return m


if __name__ == '__main__':
    out = sys.argv[1]
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    tr = sys.argv[3] if len(sys.argv) > 3 else 'cache/tlr/trn_train/t3_n600.npz'
    ev = sys.argv[4] if len(sys.argv) > 4 else 'cache/tlr/dev/t3.npz'
    train(tr, out, ep, ev)
