"""T3-joint: listwise re-ranker with attention ACROSS the query keypoints of a pair (learned pairwise reasoning).
Per keypoint i: candidate embeddings h_ik = MLP(F_ik); keypoint token z_i = soft-attention pool of h_ik (scores from a
first head) ; tokens exchange information with a 2-layer Transformer over the pair's keypoints; final candidate score
= MLP([h_ik, c_i]). Trained on trn_train tables (grouped by pair), same expected-metric-gain loss as t3.py."""
import time
import numpy as np
import torch
import torch.nn as nn

from decode_lib.t3 import gains, THRS


class JointReranker(nn.Module):
    def __init__(self, nf, h=128, layers=2, heads=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nf, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.pre = nn.Linear(h, 1)
        self.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, heads, 2 * h, dropout=0.0, batch_first=True),
                                        layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.register_buffer('mu', torch.zeros(nf))
        self.register_buffer('sd', torch.ones(nf))

    def forward(self, F, pool, kmask):
        """F [B, N, K, nf], pool [B, N, K] bool, kmask [B, N] bool (real keypoints). Returns scores [B, N, K]."""
        h = self.enc((F - self.mu) / self.sd)                              # [B,N,K,h]
        a = self.pre(h)[..., 0].masked_fill(~pool, -1e4)
        z = (torch.softmax(a, -1).unsqueeze(-1) * h).sum(2)                # [B,N,h]
        c = self.tr(z, src_key_padding_mask=~kmask)                          # [B,N,h]
        s = self.out(torch.cat([h, c.unsqueeze(2).expand_as(h)], -1))[..., 0]
        return s.masked_fill(~pool, -1e4)


def group_table(path):
    """-> padded tensors by pair: F [P, Nmax, K, nf], pool, err, kmask, row index map."""
    d = np.load(path)
    pair = d['pair']
    _, start, counts = np.unique(pair, return_index=True, return_counts=True)
    order = np.argsort(start)
    start, counts = start[order], counts[order]
    # rows of one pair are contiguous in the dump
    P, Nm = len(start), int(counts.max())
    K, nf = d['F'].shape[1], d['F'].shape[2]
    F = np.zeros((P, Nm, K, nf), np.float16)
    pool = np.zeros((P, Nm, K), bool)
    err = np.full((P, Nm, K), 9.0, np.float32)
    km = np.zeros((P, Nm), bool)
    idx = np.full((P, Nm), -1, np.int64)
    Fa, pa, ea = d['F'], d['in_pool'], d['err']
    for p, (s, c) in enumerate(zip(start, counts)):
        F[p, :c], pool[p, :c], err[p, :c], km[p, :c] = Fa[s:s + c], pa[s:s + c], ea[s:s + c], True
        idx[p, :c] = np.arange(s, s + c)
    return d, F, pool, err, km, idx


def evaluate(model, F, pool, err, km, dev='cuda', bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(F), bs):
            s = model(torch.from_numpy(F[i:i + bs]).float().to(dev), torch.from_numpy(pool[i:i + bs]).to(dev),
                      torch.from_numpy(km[i:i + bs]).to(dev))
            out.append(s.argmax(-1).cpu().numpy())
    b = np.concatenate(out)
    e = np.take_along_axis(err, b[..., None], -1)[..., 0][km]
    model.train()
    return np.array([100 * (e <= t).mean() for t in THRS])


def train(train_path, out_path, epochs=30, eval_path=None, seed=0, h=128, lr=1e-3, wd=1e-4, bs=64):
    torch.manual_seed(seed)
    dev = 'cuda'
    d, F, pool, err, km, _ = group_table(train_path)
    g = gains(err)
    nf = F.shape[-1]
    model = JointReranker(nf, h).to(dev)
    Fm = d['F'][d['in_pool']].astype(np.float32)
    model.mu.copy_(torch.from_numpy(Fm.mean(0)).to(dev))
    model.sd.copy_(torch.from_numpy(Fm.std(0)).clamp(min=1e-3).to(dev))
    del Fm
    if eval_path:
        de, Fe, pe, ee, ke, _ = group_table(eval_path)
        bb = np.where(de['in_pool'], de['base'], -1e9).argmax(1)
        eb = de['err'][np.arange(len(bb)), bb]
        pb = np.array([100 * (eb <= t).mean() for t in THRS])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(F)
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Ft = torch.from_numpy(F)            # fp16 on CPU; batches moved to GPU
    pt, gt, kt = torch.from_numpy(pool), torch.from_numpy(g), torch.from_numpy(km)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(n // bs):
            idx = perm[i * bs:(i + 1) * bs]
            Fb, pb_, gb, kb = Ft[idx].float().to(dev), pt[idx].to(dev), gt[idx].to(dev), kt[idx].to(dev)
            s = model(Fb, pb_, kb)
            p = torch.softmax(s, -1)
            loss = -((p * gb).sum(-1) * kb).sum() / kb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if eval_path and (ep % 5 == 4 or ep == epochs - 1):
            pk = evaluate(model, Fe, pe, ee, ke)
            print(f'ep {ep} loss {tot / (n // bs):.4f} eval {np.round(pk, 2)} base {np.round(pb, 2)} '
                  f'Δmean {np.mean(pk - pb):+.3f} ({time.time() - t0:.0f}s)', flush=True)
    torch.save(dict(state=model.state_dict(), nf=nf, h=h, kind='joint'), out_path)
    return model


def load_model(path, dev='cuda'):
    ck = torch.load(path, map_location=dev)
    m = JointReranker(ck['nf'], ck['h']).to(dev)
    m.load_state_dict(ck['state'])
    m.eval()
    return m
