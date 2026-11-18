"""Generalisation splits (program.md §3), built WITHOUT editing the frozen prepare.py; evaluate with prepare.evaluate.

  python prepare_gen.py spairu <method>   -> cache/<method>/spairu_test/<cat>.npz  (SPair-U unseen keypoints, test pairs)
    SPair images -> reuses cache/<method>/feat. Base protocol of the review runs: no adaptive flip; the unseen keypoints
    have no mirror permutation -> perm = -1 (decoders fall back to the own arm).
  python prepare_gen.py ap10kcf <method>  -> cache/<method>/gen_ap10kcf/  (AP-10K cross-family test, 6,300 pairs,
    adaptive flip with AP10K_FLIP for tlr/cleandift as in run_task_ap.sh)
  python prepare_gen.py pascal <method>   -> cache/<method>/gen_pascal/   (PF-Pascal test, no flip, PCK w.r.t. 840 px;
    reported at .10/.05/.15 as in the original code)
  python prepare_gen.py eval <ds> <seq> [method]  -> runs/<seq>_<commit>_<method>_<ds>/ (decode.py; OVR env as run_exp)
"""
import os
import sys
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prepare as P                                       # noqa: E402
from vendor import core as C                              # noqa: E402
from vendor.geoware import flip_keypoints                 # noqa: E402


def build_spairu(method, cats=None):
    split = 'spairu_test'
    for cat in cats or P.CATS:
        out_path = f'{P.cdir(method, split)}/{cat}.npz'
        if os.path.exists(out_path):
            continue
        names = sorted(f for f in os.listdir(f'{P.SPAIR}/JPEGImages/{cat}') if f.endswith('.jpg'))
        nidx = {n: i for i, n in enumerate(names)}
        kps, thr, used, pnames = C.load_spair_pairs(P.SPAIR, cat, 'test', unseen_kps=True)
        nk = kps.shape[1]
        out = {k: [] for k in ('src', 'trg', 'k1', 'k2', 'k1f', 'bbox', 'used_flip', 'vis_flip', 'od', 'fd', 'pid')}
        for pi in range(len(pnames)):
            k1, k2 = kps[2 * pi], kps[2 * pi + 1]
            k1f = flip_keypoints(k1, C.ANNO, None)
            for k, v in (('src', nidx[pnames[pi][0]]), ('trg', nidx[pnames[pi][1]]), ('k1', k1.numpy()),
                         ('k2', k2.numpy()), ('k1f', k1f.numpy()), ('bbox', thr[pi]), ('used_flip', False),
                         ('vis_flip', np.zeros(nk, bool)), ('od', np.nan), ('fd', np.nan), ('pid', pi)):
                out[k].append(v)
        arr = {k: np.asarray(v) for k, v in out.items()}
        np.savez(out_path, **arr, perm=-np.ones(nk, dtype=np.int64), used_kps=used.numpy(),
                 names=np.asarray(pnames))
        print(f'[gen] {method} {split} {cat}: {len(pnames)} pairs, {nk} kps', flush=True)


# ------------------------------------------------------------------------------------------- AP-10K / PF-Pascal
import json                                               # noqa: E402
import time                                               # noqa: E402
from glob import glob                                     # noqa: E402
from vendor.geoware import AP10K_FLIP, renumber_indices, permute_indices   # noqa: E402

DS = '/workspace/SemCorr_proj/Datasets'
PASCAL_CLS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
              'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def load_ap10k_pairs(split='test_cross_family', size=C.ANNO):
    """= load_ap10k_data(category='all', subsample=0) (copied logic) + per-pair group of image paths."""
    pairs = sorted(glob(f'{DS}/ap-10k/PairAnnotation/{split}/*:all.json'))
    kps, thr, files = [], [], []
    for pair in pairs:
        d = json.load(open(pair))
        S, T = json.load(open(d['src_json_path'])), json.load(open(d['trg_json_path']))
        tb = np.asarray(T['bbox'])
        sk = torch.tensor(S['keypoints']).view(-1, 3).float(); sk[:, -1] /= 2
        tk = torch.tensor(T['keypoints']).view(-1, 3).float(); tk[:, -1] /= 2
        sk, _, _, _ = C.preprocess_kps_pad(sk, S['width'], S['height'], size)
        tk, _, _, ts = C.preprocess_kps_pad(tk, T['width'], T['height'], size)
        thr.append(max(tb[3], tb[2]) * ts)
        kps += [sk, tk]
        files.append((d['src_json_path'].replace('json', 'jpg').replace('ImageAnnotation', 'JPEGImages'),
                      d['trg_json_path'].replace('json', 'jpg').replace('ImageAnnotation', 'JPEGImages')))
    kps = torch.stack(kps)
    used, = torch.where(kps[:, :, 2].any(dim=0))
    return kps[:, used, :], thr, used, files


def load_pascal_pairs(size=C.ANNO):
    import pandas as pd
    path = f'{DS}/PF-dataset-PASCAL'
    td = pd.read_csv(f'{path}/test_pairs_pf_pascal.csv')
    out = {}
    from PIL import Image
    for cat in sorted(os.listdir(f'{path}/Annotations')):
        sub = td.iloc[np.where(td.iloc[:, 2].values.astype(int) - 1 == PASCAL_CLS.index(cat))[0], :]
        kps, files = [], []
        for i in range(len(sub)):
            sf, tf = f'{path}/../{sub.iloc[i, 0]}', f'{path}/../{sub.iloc[i, 1]}'
            pts = []
            for cols in ((3, 4), (5, 6)):
                X = np.fromstring(sub.iloc[i, cols[0]], sep=';'); Y = np.fromstring(sub.iloc[i, cols[1]], sep=';')
                pc = np.zeros((20, 3), np.float32); pc[:, :2] = -1
                pc[:len(X), 0], pc[:len(X), 1], pc[:len(X), 2] = X, Y, 1
                pts.append(torch.from_numpy(pc))
            ssz, tsz = Image.open(sf).size, Image.open(tf).size
            sk, _, _, _ = C.preprocess_kps_pad(pts[0], ssz[0], ssz[1], size)
            tk, _, _, _ = C.preprocess_kps_pad(pts[1], tsz[0], tsz[1], size)
            kps += [sk, tk]
            files.append((sf, tf))
        if not files:
            continue
        kps = torch.stack(kps)
        used, = torch.where(kps[:, :, 2].any(dim=0))
        out[cat] = (kps[:, used, :], [float(size)] * len(files), used, files)
    return out


def _feat_base(img_path, fdir):
    return img_path.replace('JPEGImages', fdir).replace('.jpg', '')


@torch.no_grad()
def build_gen(ds, method):
    """Per-image fp16 raw blocks in ONE memmap per block [Nimg, 2, C, h, w] + masks; pairs meta per 'category'."""
    M = P.METHODS[method]
    gd = P.cdir(method, f'gen_{ds}')
    if ds == 'ap10kcf':
        kps, thr, used, files = load_ap10k_pairs()
        groups = {'all': (kps, thr, used, files)}
        flip_anno, do_flip = AP10K_FLIP, M['flip']
    else:
        groups = load_pascal_pairs()
        flip_anno, do_flip = None, False
    imgs = sorted({f for g in groups.values() for pr in g[3] for f in pr})
    iidx = {f: i for i, f in enumerate(imgs)}
    net = P._net(method)
    feat_done = os.path.exists(f'{gd}/mask.npy')
    if not feat_done:
        n = len(imgs)
        shapes = {'s3': (640, 60, 60), 's4': (1280, 30, 30), 's5': (1280, 15, 15), 'dino': (768, 60, 60)}
        if method == 'diysc':
            arrs = {'desc': np.lib.format.open_memmap(f'{gd}/desc.npy', 'w+', np.float16, (n, 2, 3600, 768))}
        else:
            arrs = {b: np.lib.format.open_memmap(f'{gd}/{b}.npy', 'w+', np.float16, (n, 2) + shapes[b]) for b in P.BLOCKS}
        pm = np.zeros((n, 2, 3600), bool)
        t0 = time.time()
        for i, f in enumerate(imgs):
            base = _feat_base(f, M['fdir'])
            for a, sfx in enumerate(('', '_flip')):
                sd = torch.load(f'{base}_sd{sfx}.pt', map_location='cuda')
                dino = torch.load(f'{base}_dino{sfx}.pt', map_location='cuda')
                raw = dict(s3=sd['s3'], s4=sd['s4'], s5=sd['s5'], dino=dino)
                if method == 'diysc':
                    g = C.gather_raw(raw['s3'], raw['s4'], raw['s5'], raw['dino'])
                    arrs['desc'][i, a] = C.desc_from_gathered(g, net, False)[0].half().cpu().numpy()
                else:
                    for b in P.BLOCKS:
                        arrs[b][i, a] = raw[b][0].half().cpu().numpy()
                mk = f'{base}_mask{sfx}.png'
                pm[i, a] = C.mask_to_patch_mask(C.convert_to_binary_mask(mk).cuda(), C.NP, P.MASK_DIL).cpu().numpy()
            if i % 200 == 0:
                print(f'[gen] {ds} {method} feat {i}/{n} {time.time() - t0:.0f}s', flush=True)
        for v in arrs.values():
            v.flush()
        json.dump(imgs, open(f'{gd}/imgs.json', 'w'))
        np.save(f'{gd}/mask.npy', pm)
    for cat, (kps, thr, used, files) in groups.items():
        if os.path.exists(f'{gd}/{cat}.npz'):
            continue
        nk = kps.shape[1]
        if flip_anno is not None:
            if sum(len(i) if isinstance(i, list) else 1 for i in flip_anno) == nk:
                plist = flip_anno
            else:
                filt = []
                for item in flip_anno:
                    item = [item] if isinstance(item, int) else item
                    t = [i for i in item if i in used.tolist()]
                    if t:
                        filt.append(t)
                plist = renumber_indices(filt, counter=[0])
            perm = np.asarray(permute_indices(plist, None))[:nk]
        else:
            plist, perm = None, -np.ones(nk, np.int64)
        out = {k: [] for k in ('src', 'trg', 'k1', 'k2', 'k1f', 'bbox', 'used_flip', 'vis_flip', 'od', 'fd', 'pid')}
        cache_f32 = {}

        def desc32(f, a):
            key = (f, a)
            if key not in cache_f32:
                if len(cache_f32) > 8:
                    cache_f32.pop(next(iter(cache_f32)))
                base = _feat_base(f, M['fdir'])
                sfx = '_flip' if a else ''
                sd = torch.load(f'{base}_sd{sfx}.pt', map_location='cuda')
                dino = torch.load(f'{base}_dino{sfx}.pt', map_location='cuda')
                g = C.gather_raw(sd['s3'], sd['s4'], sd['s5'], dino)
                cache_f32[key] = (C.desc_from_gathered(g, net, M['dummy']),
                                  C.convert_to_binary_mask(f'{base}_mask{sfx}.png'))
            return cache_f32[key]
        for pi, (sf, tf) in enumerate(files):
            k1, k2 = kps[2 * pi], kps[2 * pi + 1]
            vis = k1[:, 2] * k2[:, 2] > 0
            k1f = flip_keypoints(k1, C.ANNO, permute_indices(plist, vis) if plist is not None else None)
            used_flip, vis_flip, od, fd = False, torch.zeros(nk, dtype=torch.bool), np.nan, np.nan
            if do_flip and vis.sum() > 0:
                (d1, m1), (d1f, m1f), (d2, m2) = desc32(sf, 0), desc32(sf, 1), desc32(tf, 0)
                od = float(C.fast_get_distance(d1, d2, m1, m2))
                fd = float(C.fast_get_distance(d1f, d2, m1f, m2))
                try:
                    kfv = flip_keypoints(k1 * vis.unsqueeze(-1).float(), C.ANNO, permute_indices(plist, None))
                    vis_flip = (kfv[:, 2] * k2[:, 2] * k1[:, 2] > 0)
                    used_flip = bool(fd < od)
                except Exception:
                    used_flip, vis_flip = False, torch.zeros(nk, dtype=torch.bool)
            for k, v in (('src', iidx[sf]), ('trg', iidx[tf]), ('k1', k1.numpy()), ('k2', k2.numpy()),
                         ('k1f', k1f[:nk].numpy()), ('bbox', thr[pi]), ('used_flip', used_flip),
                         ('vis_flip', vis_flip.numpy()), ('od', od), ('fd', fd), ('pid', pi)):
                out[k].append(v)
        np.savez(f'{gd}/{cat}.npz', **{k: np.asarray(v) for k, v in out.items()}, perm=perm, used_kps=used.numpy(),
                 names=np.asarray(files))
        print(f'[gen] {ds} {method} {cat}: {len(files)} pairs', flush=True)


class LazyFeats:
    """prepare.CatFeats-compatible (desc(i, a), mask) over the per-dataset memmaps, GPU LRU of fp16 raw arms."""

    def __init__(self, method, ds, cap=400):
        self.method = method
        gd = P.cdir(method, f'gen_{ds}')
        if method == 'diysc':
            self.mm = {'desc': np.load(f'{gd}/desc.npy', mmap_mode='r')}
        else:
            self.mm = {b: np.load(f'{gd}/{b}.npy', mmap_mode='r') for b in P.BLOCKS}
        self.mask = torch.from_numpy(np.load(f'{gd}/mask.npy')).cuda()
        self.cap, self.raw, self.lru = cap, {}, {}

    def _raw(self, i, a):
        key = (int(i), int(a))
        if key not in self.raw:
            if len(self.raw) >= self.cap:
                self.raw.pop(next(iter(self.raw)))
            self.raw[key] = {b: torch.from_numpy(np.ascontiguousarray(v[i, a])).cuda() for b, v in self.mm.items()}
        return self.raw[key]

    @torch.no_grad()
    def desc(self, i, a):
        r = self._raw(i, a)
        if self.method == 'diysc':
            return r['desc'].float()
        g = C.gather_raw(*(r[b].unsqueeze(0).float() for b in P.BLOCKS))
        return C.normalize_feats(g.reshape(1, g.shape[1], -1).permute(0, 2, 1), True)[0]


def evaluate_gen(decode_fn, method, ds, out_dir=None, thrs=None):
    """prepare.evaluate twin for gen_<ds>: same pair dict, same reference heads; PCK thresholds `thrs`."""
    gd = P.cdir(method, f'gen_{ds}')
    cats = sorted(f[:-4] for f in os.listdir(gd) if f.endswith('.npz') and not f.endswith('.ref.npz'))
    thrs = thrs or (P.THRS if ds != 'pascal' else (0.10, 0.05, 0.15))
    F_ = LazyFeats(method, ds)
    t0, sec_dec = time.time(), 0.0
    rec = dict(err=[], ref=[], pair=[], cat=[])
    for cat in cats:
        meta = dict(np.load(f'{gd}/{cat}.npz'))
        rp = f'{gd}/{cat}.ref.npz'
        ref = dict(np.load(rp)) if os.path.exists(rp) else None
        new = []
        order = np.argsort(meta['src'], kind='stable')                        # locality for the LRU
        errs = {}
        for j in order:
            k2 = meta['k2'][j]
            vis = (meta['k1'][j][:, 2] > 0) & (k2[:, 2] > 0)
            if vis.sum() == 0:
                continue
            pair = P.make_pair(F_, meta, j, cat)
            bbox = float(meta['bbox'][j])
            t1 = time.time()
            with torch.no_grad():
                pred = decode_fn(pair)
            torch.cuda.synchronize()
            sec_dec += time.time() - t1
            gt = torch.from_numpy(k2[:, :2]).float().cuda()
            vt = torch.from_numpy(vis).cuda()
            e = ((pred.float() - gt).norm(dim=-1) / bbox)[vt].cpu().numpy()
            r = None
            if ref is None:
                rr = P.reference_heads(pair, k2, bbox, P.METHODS[method]['lam'])
                r = np.stack([rr[h][vt].cpu().numpy() for h in P.REF_NAMES])
            errs[j] = (e, r)
        keep = sorted(errs)
        for j in keep:
            rec['err'].append(errs[j][0])
            rec['pair'] += [f'{cat}/{j}'] * len(errs[j][0])
            rec['cat'] += [cat] * len(errs[j][0])
        if ref is None:
            ref = dict(err=np.concatenate([errs[j][1] for j in keep], 1).astype(np.float32))
            np.savez(rp, **ref, heads=np.array(P.REF_NAMES))
        rec['ref'].append(ref['err'])
    err = np.concatenate(rec['err'])
    ref = np.concatenate(rec['ref'], 1)
    pair = np.array(rec['pair'])
    saved = P.THRS
    P.THRS = tuple(thrs)
    try:
        m = P.metrics(err, ref, pair, ds)
    finally:
        P.THRS = saved
    m.update(sec=time.time() - t0, sec_decode=sec_dec, method=method, split=ds, thrs=list(thrs), cats=cats)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez(f'{out_dir}/err.npz', err=err, ref=ref, ref_heads=np.array(P.REF_NAMES), pair=pair,
                 cat=np.array(rec['cat']))
        json.dump(m, open(f'{out_dir}/metrics.json', 'w'), indent=1)
    P.print_summary(m)
    return m


if __name__ == '__main__':
    ds = sys.argv[1]
    if ds == 'eval':
        import subprocess
        ds, seq = sys.argv[2], sys.argv[3]
        method = sys.argv[4] if len(sys.argv) > 4 else 'tlr'
        import decode as D
        for kv in filter(None, os.environ.get('OVR', '').split(',')):
            k, v = kv.split('=')
            setattr(D, k, type(getattr(D, k))(v))
            print(f'override {k}={v!r}')
        c = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE).decode().strip()
        tag = os.environ.get('TAG', '')
        out = f'{HERE}/runs/{seq}_{c}_{method}_{ds}' + (f'_{tag}' if tag else '')
        evaluate_gen(D.decode, method, ds, out)
    else:
        method = sys.argv[2]
        {'spairu': build_spairu, 'ap10kcf': lambda m: build_gen('ap10kcf', m),
         'pascal': lambda m: build_gen('pascal', m)}[ds](method)
