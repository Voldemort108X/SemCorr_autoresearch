"""prepare.py -- FROZEN after Day-0 setup (program.md §2.3). Cache building + evaluate().

Cache (per base method m):
  cache/<m>/feat/<cat>.<blk>.npy   fp16 raw blocks [100 img, 2 arm(orig, flip), C, h, w]   (tlr, cleandift)
  cache/<m>/feat/<cat>.desc.npy    fp16 projected+normalized desc [100, 2, 3600, 768]        (diysc)
  cache/<m>/feat/<cat>.mask.npy    bool patch masks (dilation 3) [100, 2, 3600]
  cache/<m>/feat/<cat>.imgs.json   image names (index order)
  cache/<m>/<split>/<cat>.npz      pairs: src/trg image idx, kps (840 frame), bbox, adaptive-flip arm, ...
  cache/<m>/<split>/<cat>.ref.npz  reference heads (top1, dv, oracle@5/10/30) per visible keypoint

Usage:
  python prepare.py --method tlr --split dev,val,test [--cats aeroplane,bird]
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vendor import core as C                                         # noqa: E402
from vendor.geoware import SPAIR_FLIP, renumber_indices, permute_indices, flip_keypoints  # noqa: E402

SPAIR = '/workspace/SemCorr_proj/Datasets/SPair-71k'
DIY_CKPT = '/workspace/SemCorr_proj/Code/SemCorrVLM_cvpr26/semcorr_vfm/baselines_neurips/DIY-SC/ckpts/0280_spair/best.pth'
CATS = sorted(os.listdir(f'{SPAIR}/ImageAnnotation'))
METHODS = {   # feature dir, dummy net (TLR-style normalisation), adaptive flip, Deja Vu lambda of that method
    'tlr': dict(fdir='features', dummy=True, flip=True, lam=0.5),
    'cleandift': dict(fdir='features_cleandift', dummy=True, flip=True, lam=3.0),
    'diysc': dict(fdir='features', dummy=False, flip=False, lam=1.0),
}
BLOCKS = ('s3', 's4', 's5', 'dino')
THRS = (0.10, 0.05, 0.01)
ORACLE_K = (5, 10, 30)
DEV_PER_CAT = 300
WINDOW, BETA, MASK_DIL = 5, 0.01, 3
CACHE = f'{HERE}/cache'


def cdir(method, sub):
    d = f'{CACHE}/{method}/{sub}'
    os.makedirs(d, exist_ok=True)
    return d


# =============================================================================== cache building
def _load_orig(method, cat, names, device='cuda'):
    """fp32 originals for all images x arms -> dict blk -> [100,2,...] on device; pixel masks [100,2,960,960]."""
    fdir = METHODS[method]['fdir']
    raw = {b: [] for b in BLOCKS}
    pmask = []
    for n in names:
        base = f'{SPAIR}/{fdir}/{cat}/{n[:-4]}'
        per = {b: [] for b in BLOCKS}
        pm = []
        for sfx in ('', '_flip'):
            sd = torch.load(f'{base}_sd{sfx}.pt', map_location='cpu')
            for b in ('s3', 's4', 's5'):
                per[b].append(sd[b][0])
            per['dino'].append(torch.load(f'{base}_dino{sfx}.pt', map_location='cpu')[0])
            pm.append(C.convert_to_binary_mask(f'{base}_mask{sfx}.png'))
        for b in BLOCKS:
            raw[b].append(torch.stack(per[b]))
        pmask.append(torch.stack(pm))
    raw = {b: torch.stack(v).to(device) for b, v in raw.items()}
    return raw, torch.stack(pmask)


_DIY_NET = None


def _net(method):
    global _DIY_NET
    if method != 'diysc':
        from vendor.projection_network import DummyAggregationNetwork
        return DummyAggregationNetwork().cuda().eval()
    if _DIY_NET is None:
        from vendor.projection_network import AggregationNetwork
        net = AggregationNetwork(feature_dims=[640, 1280, 1280, 768], projection_dim=768, device='cuda',
                                 feat_map_dropout=0.2)
        net.load_pretrained_weights(torch.load(DIY_CKPT))
        _DIY_NET = net.cuda().eval()
    return _DIY_NET


@torch.no_grad()
def _desc_fp32(raw, i, a, net, dummy):
    g = C.gather_raw(*(raw[b][i, a:a + 1] for b in BLOCKS))
    return C.desc_from_gathered(g, net, dummy)          # [1, P, D]


@torch.no_grad()
def build_category(method, cat, splits, rng_seed=0):
    M = METHODS[method]
    fd = cdir(method, 'feat')
    names = sorted(f for f in os.listdir(f'{SPAIR}/JPEGImages/{cat}') if f.endswith('.jpg'))
    nidx = {n: i for i, n in enumerate(names)}
    need_feat = not os.path.exists(f'{fd}/{cat}.mask.npy')
    todo = [s for s in splits if not os.path.exists(f'{cdir(method, s)}/{cat}.npz')]
    if not need_feat and not todo:
        return
    t0 = time.time()
    raw, pmask = _load_orig(method, cat, names)
    net = _net(method)
    if need_feat:
        pm = torch.stack([torch.stack([C.mask_to_patch_mask(pmask[i, a], C.NP, MASK_DIL) for a in range(2)])
                          for i in range(len(names))]).cpu().numpy()
        if method == 'diysc':
            desc = torch.stack([torch.stack([_desc_fp32(raw, i, a, net, False)[0] for a in range(2)])
                                for i in range(len(names))])
            np.save(f'{fd}/{cat}.desc.npy', desc.half().cpu().numpy())
            del desc
        else:
            for b in BLOCKS:
                np.save(f'{fd}/{cat}.{b}.npy', raw[b].half().cpu().numpy())
        json.dump(names, open(f'{fd}/{cat}.imgs.json', 'w'))
        np.save(f'{fd}/{cat}.mask.npy', pm)          # written last = completion marker
    for split in todo:
        _build_split(method, cat, split, raw, pmask, names, nidx, net, rng_seed)
    print(f'[prepare] {method} {cat} feat={need_feat} splits={todo} {time.time() - t0:.0f}s', flush=True)
    del raw
    torch.cuda.empty_cache()


def _split_pairs(cat, split, rng_seed):
    """-> kps [2P, N, 3], thresholds [P], used_kps, names [(src, trg)], and the pair selection."""
    base = {'dev': 'trn', 'trn_train': 'trn'}.get(split, split)
    kps, thr, used, names = C.load_spair_pairs(SPAIR, cat, base)
    P = len(names)
    sel = np.arange(P)
    if split in ('dev', 'trn_train'):
        rng = np.random.default_rng(rng_seed + CATS.index(cat))
        dev = np.sort(rng.choice(P, size=min(DEV_PER_CAT, P), replace=False))
        sel = dev if split == 'dev' else np.setdiff1d(np.arange(P), dev)
    return kps, np.asarray(thr, dtype=np.float64), used, names, sel


def _permute_list(cat, used_points, nk):
    FLIP_ANNO = SPAIR_FLIP[cat]
    if sum(len(i) if isinstance(i, list) else 1 for i in FLIP_ANNO) == nk:
        return FLIP_ANNO
    filt = []
    for item in FLIP_ANNO:
        item = [item] if isinstance(item, int) else item
        t = [i for i in item if i in used_points]
        if t:
            filt.append(t)
    return renumber_indices(filt, counter=[0])


def _build_split(method, cat, split, raw, pmask, names, nidx, net, rng_seed):
    M = METHODS[method]
    kps, thr, used, pnames, sel = _split_pairs(cat, split, rng_seed)
    nk = kps.shape[1]
    permute_list = _permute_list(cat, used.tolist(), nk)
    perm_full = permute_indices(permute_list, None)
    out = {k: [] for k in ('src', 'trg', 'k1', 'k2', 'k1f', 'bbox', 'used_flip', 'vis_flip', 'od', 'fd', 'pid')}
    for pi in sel:
        k1, k2 = kps[2 * pi], kps[2 * pi + 1]
        vis = k1[:, 2] * k2[:, 2] > 0
        s, t = nidx[pnames[pi][0]], nidx[pnames[pi][1]]
        k1f = flip_keypoints(k1, C.ANNO, permute_indices(permute_list, vis))
        used_flip, vis_flip, od, fd = False, torch.zeros(nk, dtype=torch.bool), np.nan, np.nan
        if M['flip'] and vis.sum() > 0:
            d1 = _desc_fp32(raw, s, 0, net, M['dummy'])
            d1f = _desc_fp32(raw, s, 1, net, M['dummy'])
            d2 = _desc_fp32(raw, t, 0, net, M['dummy'])
            od = float(C.fast_get_distance(d1, d2, pmask[s, 0], pmask[t, 0]))
            fd = float(C.fast_get_distance(d1f, d2, pmask[s, 1], pmask[t, 0]))
            try:
                kfv = flip_keypoints(k1 * vis.unsqueeze(-1).float(), C.ANNO, permute_indices(permute_list, None))
                vis_flip = (kfv[:, 2] * k2[:, 2] * k1[:, 2] > 0)
                used_flip = bool(fd < od)
            except Exception:
                used_flip, vis_flip = False, torch.zeros(nk, dtype=torch.bool)
        assert k1f.shape[0] == nk, (cat, k1f.shape, nk)
        for k, v in (('src', s), ('trg', t), ('k1', k1.numpy()), ('k2', k2.numpy()), ('k1f', k1f.numpy()),
                     ('bbox', thr[pi]), ('used_flip', used_flip), ('vis_flip', vis_flip.numpy()),
                     ('od', od), ('fd', fd), ('pid', pi)):
            out[k].append(v)
    arr = {k: np.asarray(v) for k, v in out.items()}
    pf = np.asarray(perm_full + list(range(len(perm_full), nk)))[:nk] if len(perm_full) < nk else np.asarray(perm_full)[:nk]
    np.savez(f'{cdir(method, split)}/{cat}.npz', **arr, perm=pf, used_kps=used.numpy(),
             names=np.asarray(pnames)[sel])


# =============================================================================== evaluation
class CatFeats:
    """GPU-resident fp16 cache of one category; desc(i, arm) -> fp32 [P, D] normalized descriptors."""

    def __init__(self, method, cat):
        self.method, self.M = method, METHODS[method]
        fd = cdir(method, 'feat')
        if method == 'diysc':
            self.desc16 = torch.from_numpy(np.load(f'{fd}/{cat}.desc.npy')).cuda()
        else:
            self.raw = {b: torch.from_numpy(np.load(f'{fd}/{cat}.{b}.npy')).cuda() for b in BLOCKS}
        self.mask = torch.from_numpy(np.load(f'{fd}/{cat}.mask.npy')).cuda()
        self._lru = {}

    @torch.no_grad()
    def desc(self, i, a):
        key = (int(i), int(a))
        if key in self._lru:
            return self._lru[key]
        if self.method == 'diysc':
            d = self.desc16[i, a].float()
        else:
            g = C.gather_raw(*(self.raw[b][i, a:a + 1].float() for b in BLOCKS))
            d = C.normalize_feats(g.reshape(1, g.shape[1], -1).permute(0, 2, 1), True)[0]
        if len(self._lru) >= 6:
            self._lru.pop(next(iter(self._lru)))
        self._lru[key] = d
        return d


def load_split(method, split, cat):
    return dict(np.load(f'{cdir(method, split)}/{cat}.npz'))


def make_pair(F_, meta, j, cat):
    """The decoder's input for pair j (no target keypoints / bbox / labels)."""
    s, t = int(meta['src'][j]), int(meta['trg'][j])
    k1, k1f = meta['k1'][j], meta['k1f'][j]
    arm = np.zeros(len(k1), dtype=np.int64)
    if bool(meta['used_flip'][j]):
        arm[meta['vis_flip'][j].astype(bool)] = 1
    dev = 'cuda'
    src = []
    for a, kk in ((0, k1), (1, k1f)):
        src.append(dict(desc=F_.desc(s, a), mask=F_.mask[s, a],
                        kp=torch.from_numpy(C.kpts_to_patch_idx(torch.from_numpy(kk))).long().to(dev),
                        kp_valid=torch.from_numpy(kk[:, 2] > 0).to(dev)))
    return dict(src=src, trg=dict(desc=F_.desc(t, 0), mask=F_.mask[t, 0]),
                arm_of_kpt=torch.from_numpy(arm).to(dev),
                perm=torch.from_numpy(meta['perm'].astype(np.int64)).to(dev),
                nP=C.NP, anno=C.ANNO)


@torch.no_grad()
def reference_heads(pair, k2, bbox, lam):
    """top1, dv (graded lambda), oracle@k per keypoint, combined over arms. Returns dict name -> err [N]."""
    N = pair['arm_of_kpt'].numel()
    preds = {n: torch.zeros(N, 2, device='cuda') for n in ('top1', 'dv')}
    orc = {k: torch.full((N,), 1e9, device='cuda') for k in ORACLE_K}
    gt = torch.from_numpy(k2[:, :2]).float().cuda()
    arms = [0, 1] if bool((pair['arm_of_kpt'] == 1).any()) else [0]
    for a in arms:
        A = pair['src'][a]
        st = C.PairState(A['desc'], pair['trg']['desc'], A['kp'], A['mask'], pair['trg']['mask'])
        take = (pair['arm_of_kpt'] == a)
        if st.M == 0:
            for n in preds:
                preds[n][take] = 0
            continue
        vi = st.valid_kpt_indices
        for n, al in (('top1', 0.0), ('dv', lam)):
            by, bx = C.select_graded(st, al)
            nx, ny = C.window_soft_argmax(st, by, bx, WINDOW, BETA)
            o = torch.zeros(N, 2, device='cuda')
            o[vi] = torch.stack([nx, ny], 1)
            preds[n][take] = o[take]
        si, _ = st.sorted_by_temperature(0.1)
        K = min(max(ORACLE_K), si.shape[1])
        pid = st.tgt_valid_idx[si[:, :K]]                       # [M, K]
        s2 = st.full_sim_2d.unsqueeze(1).expand(-1, K, -1, -1).reshape(-1, C.NP, C.NP)
        nx, ny = C.window_soft_argmax_2d(s2, (pid // C.NP).reshape(-1), (pid % C.NP).reshape(-1), WINDOW, BETA)
        d = torch.sqrt((nx.view(-1, K) - gt[vi, 0:1]) ** 2 + (ny.view(-1, K) - gt[vi, 1:2]) ** 2) / bbox
        cm = torch.cummin(d, dim=1)[0]
        for k in ORACLE_K:
            o = torch.full((N,), 1e9, device='cuda')
            o[vi] = cm[:, min(k, K) - 1]
            orc[k][take] = o[take]
    out = {n: ((p - gt).norm(dim=-1) / bbox) for n, p in preds.items()}
    for k in ORACLE_K:
        out[f'or{k}'] = orc[k]
    return out


REF_NAMES = ('top1', 'dv') + tuple(f'or{k}' for k in ORACLE_K)


def evaluate(decode_fn, method='tlr', split='dev', cats=None, out_dir=None, quiet=False, max_pairs=None):
    """Runs decode_fn(pair) -> pred [N, 2] (840 frame) on every pair; returns metrics dict.
    Saves out_dir/err.npz and out_dir/metrics.json when out_dir is given."""
    cats = cats or CATS
    lam = METHODS[method]['lam']
    t0 = time.time()
    rec = dict(err=[], ref=[], cat=[], pair=[], kpt=[])
    sec_decode = 0.0
    for cat in cats:
        meta = load_split(method, split, cat)
        rpath = f'{cdir(method, split)}/{cat}.ref.npz'
        ref = dict(np.load(rpath)) if os.path.exists(rpath) else None
        new_ref = []
        F_ = CatFeats(method, cat)
        n = len(meta['src']) if max_pairs is None else min(max_pairs, len(meta['src']))
        for j in range(n):
            k2 = meta['k2'][j]
            vis = (meta['k1'][j][:, 2] > 0) & (k2[:, 2] > 0)
            if vis.sum() == 0:
                continue
            pair = make_pair(F_, meta, j, cat)
            bbox = float(meta['bbox'][j])
            t1 = time.time()
            with torch.no_grad():
                pred = decode_fn(pair)
            torch.cuda.synchronize()
            sec_decode += time.time() - t1
            gt = torch.from_numpy(k2[:, :2]).float().cuda()
            e = ((pred.float() - gt).norm(dim=-1) / bbox)[torch.from_numpy(vis).cuda()]
            rec['err'].append(e.cpu().numpy())
            if ref is None:
                r = reference_heads(pair, k2, bbox, lam)
                new_ref.append(np.stack([r[h][torch.from_numpy(vis).cuda()].cpu().numpy() for h in REF_NAMES]))
            nv = int(vis.sum())
            rec['cat'] += [cat] * nv
            rec['pair'] += [f'{cat}/{int(meta["pid"][j])}'] * nv
            rec['kpt'].append(np.nonzero(vis)[0])
        if ref is None:
            ref = dict(err=np.concatenate(new_ref, 1).astype(np.float32), heads=np.array(REF_NAMES))
            if max_pairs is None:
                np.savez(rpath, **ref)
        rec['ref'].append(ref['err'])
        del F_
        torch.cuda.empty_cache()
    err = np.concatenate(rec['err']).astype(np.float32)
    ref = np.concatenate(rec['ref'], 1)
    pair = np.array(rec['pair'])
    m = metrics(err, ref, pair, split)
    m.update(sec=time.time() - t0, sec_decode=sec_decode, method=method, split=split, cats=cats)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez(f'{out_dir}/err.npz', err=err, ref=ref, ref_heads=np.array(REF_NAMES), cat=np.array(rec['cat']),
                 pair=pair, kpt=np.concatenate(rec['kpt']))
        json.dump(m, open(f'{out_dir}/metrics.json', 'w'), indent=1)
    if not quiet:
        print_summary(m)
    return m


def _pck(err):
    return [100.0 * float((err <= t).mean()) for t in THRS]


def _pck_img(err, pair):
    _, inv = np.unique(pair, return_inverse=True)
    out = []
    for t in THRS:
        c = np.bincount(inv, weights=(err <= t).astype(np.float64)) / np.bincount(inv)
        out.append(100.0 * float(c.mean()))
    return out


def metrics(err, ref, pair, split, n_boot=1000, seed=0):
    H = dict(zip(REF_NAMES, ref))
    m = dict(n_kpt=int(err.size), n_pair=int(np.unique(pair).size))
    m['pck'] = {'dec': _pck(err), **{h: _pck(v) for h, v in H.items()}}
    m['pck_img'] = {'dec': _pck_img(err, pair), **{h: _pck_img(v, pair) for h, v in H.items()}}
    d = [a - b for a, b in zip(m['pck']['dec'], m['pck']['top1'])]
    m['d10'], m['d05'], m['d01'] = d
    m['dev'] = float(np.mean(d))
    m['dev_dv'] = float(np.mean([a - b for a, b in zip(m['pck']['dv'], m['pck']['top1'])]))
    # paired bootstrap over pairs: delta of the selection metric vs dv
    _, inv = np.unique(pair, return_inverse=True)
    P = inv.max() + 1
    cnt = np.bincount(inv, minlength=P).astype(np.float64)
    corr = lambda e: np.stack([np.bincount(inv, weights=(e <= t).astype(np.float64), minlength=P) for t in THRS])
    cd, cv, c1 = corr(err), corr(H['dv']), corr(H['top1'])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n_boot, P))
    w = np.stack([np.bincount(r, minlength=P) for r in idx]).astype(np.float64)   # [B, P]
    tot = w @ cnt
    dd = 100 * ((w @ (cd - cv).T) / tot[:, None]).mean(1)
    d1 = 100 * ((w @ (cd - c1).T) / tot[:, None]).mean(1)
    m['boot_std_vs_dv'] = float(dd.std())
    m['boot_std_vs_top1'] = float(d1.std())
    m['delta_vs_dv'] = m['dev'] - m['dev_dv']
    m['p_better_than_dv'] = float((dd > 0).mean())
    return m


def print_summary(m):
    sp = m['split']
    f = lambda v: '/'.join(f'{x:.2f}' for x in v)
    print(f"{sp}: {m['dev']:+.3f} d10 {m['d10']:+.2f} d05 {m['d05']:+.2f} d01 {m['d01']:+.2f} | dv {m['dev_dv']:+.3f} "
          f"| vs_dv {m['delta_vs_dv']:+.3f} (boot std {m['boot_std_vs_dv']:.3f}, P>0 {m['p_better_than_dv']:.2f})")
    print('pck_kpt: ' + ' | '.join(f'{h} {f(v)}' for h, v in m['pck'].items()))
    print('pck_img: ' + ' | '.join(f'{h} {f(v)}' for h, v in m['pck_img'].items()))
    print(f"sec: {m['sec']:.0f} (decode {m['sec_decode']:.0f}) n_kpt {m['n_kpt']} n_pair {m['n_pair']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='tlr', choices=list(METHODS))
    ap.add_argument('--split', default='dev')
    ap.add_argument('--cats', default='')
    a = ap.parse_args()
    cats = a.cats.split(',') if a.cats else CATS
    splits = a.split.split(',')
    for c in cats:
        build_category(a.method, c, splits)
    print('[prepare] done', flush=True)
