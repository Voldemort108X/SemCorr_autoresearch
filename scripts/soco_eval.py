"""SOCO evaluation of our decoders on cached features (scripts/soco_extract.py), faithful to OmniProbe's
correspondence_soco protocol and to the paper's soco_eval_onepass.py:
  * keypoints: int(pos) * S / hw, truncated (SOCODataset stores them in an int tensor), S = effective image size;
  * query feature = bilinear grid_sample (align_corners=True) of the source map at the keypoint;
  * argmax head = argmax over ALL target patches (SOCO has no masks), prediction = patch index / W (patch corner);
  * error = |pred - kp_trg| / S / thresh_scale (thresh_scale = max bbox side / max image side, use_bbox false),
    correct if < 0.10; SOC over semantic_overlap pairs, CC = min over the concept's target keypoints;
  * per pair mean -> per class mean -> mean over classes.
Heads: 'argmax' (exact OmniProbe), 'dv<lam>' (exact paper port: top-p over all patches, tau .1, p .9, K_max 30,
cycle via reverse NN over all source patches, score = sim - lam d / W), 'ours' = decode.decode(pair) with OVR settings
(output mode OUT: 'corner' = chosen patch index like OmniProbe, 'refine' = decode's refined position).

usage: python scripts/soco_eval.py <split> <backbone> <out_dir> [max_pairs_per_class] [classes...]
  split: dev (PairAnnotations/trainsplits/train, selection) | intra | cross | test (trainsplits/test)
  env OVR="NAME=val,..." decode.py overrides; SOCO_OUT=corner|refine; SOCO_ARM=none|dist (pair-level flip arm rule);
      SOCO_QB=1 replace the query patch rows by the bilinear query feature; SOCO_DV="0.5" paper-head lambdas
"""
import glob, json, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vendor import core as C
import decode as D

ROOT = '/workspace/SemCorr_proj/Datasets/SOCOv1'
SPLITS = {'dev': 'trainsplits/train', 'test': 'trainsplits/test', 'intra': 'intra', 'cross': 'cross'}
THRESH = 0.10


def apply_ovr(s):
    for kv in [x for x in s.split(',') if x.strip()]:
        k, v = kv.split('=', 1)
        cur = getattr(D, k)
        if isinstance(cur, bool):
            setattr(D, k, v == 'True')
        elif isinstance(cur, (int, float)) and v != '':
            setattr(D, k, type(cur)(float(v)) if isinstance(cur, int) else float(v))
        else:
            setattr(D, k, v)


def mirror_name(n):
    t = re.split(r'(_)', n)
    m = ['right' if x == 'left' else 'left' if x == 'right' else x for x in t]
    return ''.join(m)


class Feats:
    def __init__(self, bb):
        self.dir = f'cache/soco/{bb}'
        self.c = {}

    def get(self, cls):
        if cls not in self.c:
            if len(self.c) > 6:
                self.c.pop(next(iter(self.c)))
            z = np.load(f'{self.dir}/{cls}.npz')
            names = list(z['names'])
            self.c[cls] = dict(idx={n: i for i, n in enumerate(names)}, hw=z['hw'],
                               feat=torch.from_numpy(z['feat']).cuda(), featf=torch.from_numpy(z['featf']).cuda(),
                               H=int(z['H']), W=int(z['W']))
        return self.c[cls]


def load_kps(kps, hw, S):
    k = torch.tensor([[int(p['pos'][0]), int(p['pos'][1])] for p in kps], dtype=torch.int64)
    return (k.float() * S / hw).to(torch.int64).float()      # int tensor assignment truncates (toward 0)


def score(pred, ks, kt, p, scale, map_s, map_t):
    """pred, kt: [N,2] pixel in the S frame (normalised later). Returns (soc_acc or None, cc_acc or None)."""
    err = lambda i, j: float((pred[i] - kt[j]).norm()) / scale
    sem = [err(map_s[n], map_t[n]) for n in p.get('semantic_overlap', []) if n in map_s and n in map_t]
    cm = {}
    for c in p.get('concept_matches', []):
        tj = [map_t[n] for n in c.get('trg_keypoints', []) if n in map_t]
        if not tj:
            continue
        for n in c.get('src_keypoints', []):
            if n in map_s:
                cm.setdefault(map_s[n], set()).update(tj)
    con = [min(err(i, j) for j in tj) for i, tj in cm.items() if tj]
    f = lambda e: (float(np.mean([x < THRESH for x in e])) if e else None)
    return f(sem), f(con)


def main():
    split, bb, out = sys.argv[1], sys.argv[2], sys.argv[3]
    maxp = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '0' else None
    classes = sys.argv[5:] or None
    D.T3_MODEL = ''   # training-free path (the T3 model assumes the 60x60 SPair grid)
    apply_ovr(os.environ.get('OVR', ''))
    OUT = os.environ.get('SOCO_OUT', 'corner')
    ARM = os.environ.get('SOCO_ARM', 'dist')
    QB = os.environ.get('SOCO_QB', '1') == '1'
    DVL = [float(x) for x in os.environ.get('SOCO_DV', '0.5').split(',') if x]
    OURS = os.environ.get('SOCO_OURS', '1') == '1'
    NOFLIP = os.environ.get('SOCO_NOFLIP', '0') == '1'
    PADM = os.environ.get('SOCO_PADMASK', '0') == '1'
    SOFT = os.environ.get('SOCO_SOFT', '1') == '1'
    meta = json.load(open(f'cache/soco/{bb}/meta.json'))
    S = meta['image_size']
    FS = Feats(bb)
    base = C.PairState

    pdir = f'{ROOT}/PairAnnotations/{SPLITS[split]}'
    classes = classes or sorted(os.listdir(pdir))
    heads = ['argmax'] + (['soft'] if SOFT else []) + [f'dv{l:g}' for l in DVL] + (['ours'] if OURS else [])
    if OURS and ARM == 'both':
        heads += ['ours0', 'ours1']
    armstats = {}
    acc = {h: {} for h in heads}                    # head -> cls -> list of (soc, cc) per pair
    t0 = time.time()
    tdec = 0.0
    npair = 0
    for ci, cls in enumerate(classes):
        files = sorted(glob.glob(f'{pdir}/{cls}/*.json'))
        if maxp:
            rng = np.random.RandomState(ci)
            files = [files[i] for i in sorted(rng.choice(len(files), min(maxp, len(files)), replace=False))]
        for h in heads:
            acc[h][cls] = []
        for f in files:
            p = json.load(open(f))
            A, B = FS.get(p['src_category']), FS.get(p['trg_category'])
            H, W = A['H'], A['W']
            si, ti = A['idx'][p['src_image']], B['idx'][p['trg_image']]
            ks = load_kps(p['src_keypoints'], int(A['hw'][si]), S)
            kt = load_kps(p['trg_keypoints'], int(B['hw'][ti]), S)
            map_s = {k['name']: i for i, k in enumerate(p['src_keypoints'])}
            map_t = {k['name']: i for i, k in enumerate(p['trg_keypoints'])}
            tb = p.get('trg_bndbox')
            scale = (max(tb[2] - tb[0], tb[3] - tb[1]) / max(p['trg_imsize'][:2])) if tb else 1.0
            fs, ft, ff = A['feat'][si].float(), B['feat'][ti].float(), A['featf'][si].float()
            N = ks.shape[0]
            # --- exact OmniProbe / paper heads
            ndc = ((ks / S) * 2 - 1).cuda()
            fmap = fs.t().reshape(-1, H, W)
            q = F.grid_sample(fmap[None], ndc[None, None], mode='bilinear', align_corners=True)[0, :, 0].t()   # [N,C]
            flat = q @ ft.t()
            kx = (ks[:, 0] / S * W).floor().clamp(0, W - 1).cuda()
            ky = (ks[:, 1] / S * H).floor().clamp(0, H - 1).cuda()
            preds = {}
            idx = flat.argmax(-1)
            preds['argmax'] = torch.stack((idx % W, idx // W), -1).float().cpu() * S / W
            if SOFT:   # OmniProbe soft_eval (beta .02, window 7 around the argmax), heatmap index space
                nx, ny = C.window_soft_argmax_2d(flat.view(N, H, W), idx // W, idx % W, 7, 0.02, 1.0)
                preds['soft'] = torch.stack((nx - 0.5, ny - 0.5), -1).cpu() * S / W
            if DVL:
                probs = torch.softmax(flat / 0.1, -1)
                sp, sidx = torch.sort(probs, -1, descending=True)
                cut = ((sp.cumsum(-1) >= 0.9).float().argmax(-1) + 1).clamp(1, 30)
                kk = int(cut.max())
                cand = sidx[:, :kk]
                csim = torch.gather(flat, 1, cand)
                back = (ft[cand.reshape(-1)] @ fs.t()).argmax(-1).reshape(N, kk)
                d = torch.sqrt(((back // W).float() - ky[:, None]) ** 2 + ((back % W).float() - kx[:, None]) ** 2)
                valid = torch.arange(kk, device='cuda')[None] < cut[:, None]
                for lam in DVL:
                    b = (csim - lam * d / W).masked_fill(~valid, -1e9).argmax(-1)
                    ii = cand[torch.arange(N, device='cuda'), b]
                    preds[f'dv{lam:g}'] = torch.stack((ii % W, ii // W), -1).float().cpu() * S / W
            # --- our decoder (decode.py) on a pair dict in the repo's format
            if OURS:
                t1 = time.time()
                names = [k['name'] for k in p['src_keypoints']]
                perm = torch.tensor([map_s.get(mirror_name(n), -1) for n in names], dtype=torch.long)
                if NOFLIP:   # similarity-only track: no mirror information at all (other-arm views fall back to own)
                    perm = torch.full_like(perm, -1)
                kpi = (ky * W + kx).long()
                fsq = fs.clone()
                if QB:   # the query rows carry the bilinear query feature (as OmniProbe)
                    fsq[kpi] = F.normalize(q, dim=-1)
                kf = ((ky * W + (W - 1 - kx))).long()
                pc = perm.cuda()
                ok1 = pc >= 0
                kp1 = torch.where(ok1, kf[pc.clamp(min=0)], torch.zeros_like(kf))
                ffq = ff.clone()
                allm = torch.ones(H * W, dtype=torch.bool, device='cuda')
                srcm, trgm, srcmf = allm, allm, allm
                if PADM:   # exclude the white square-padding (right / bottom) from both images: the valid image area
                    def vm(imsize, hw, flip=False):
                        w_, h_ = imsize[0] * S / hw, imsize[1] * S / hw
                        c = (torch.arange(W, device='cuda').float() + 0.5) * S / W
                        mx, my = c < w_, c < h_
                        if flip:
                            mx = mx.flip(0)
                        return (my.view(-1, 1) & mx.view(1, -1)).reshape(-1)
                    srcm = vm(p['src_imsize'], int(A['hw'][si]))
                    srcmf = vm(p['src_imsize'], int(A['hw'][si]), True)
                    trgm = vm(p['trg_imsize'], int(B['hw'][ti]))
                arm = 0
                if ARM == 'one' and not NOFLIP:
                    arm = 1
                if ARM == 'dist' and not NOFLIP:
                    d0 = (2 - 2 * (fs @ ft.t()).max(1)[0]).clamp(min=0).sqrt().mean()
                    d1 = (2 - 2 * (ff @ ft.t()).max(1)[0]).clamp(min=0).sqrt().mean()
                    arm = int(d1 < d0)
                def run(arm_):
                    pair = dict(src=[dict(desc=fsq, mask=srcm, kp=kpi, kp_valid=torch.ones(N, dtype=torch.bool, device='cuda')),
                                     dict(desc=ffq, mask=srcmf, kp=kp1, kp_valid=ok1)],
                                trg=dict(desc=ft, mask=trgm),
                                arm_of_kpt=torch.full((N,), arm_, dtype=torch.long, device='cuda') * ok1.long(),
                                perm=pc, nP=W, anno=float(S))
                    C.PairState = lambda *a_, **k_: base(*a_, num_patches=W, anno=float(S))
                    D._STATS = [] if ARM == 'both' else None
                    try:
                        pr = D.decode(pair)
                    finally:
                        C.PairState = base
                    stt = [x for x in (D._STATS or []) if x['arm'] == arm_]
                    D._STATS = None
                    st = S / W
                    if OUT == 'corner':   # snap to the chosen patch's corner, OmniProbe's output format
                        pr = torch.floor(pr / st) * st
                    else:                 # refined position in OmniProbe's index-space convention (patch i -> i * S / W)
                        pr = pr - st / 2
                    pr = pr.cpu()
                    dummy = (kpi == 0).cpu()   # keypoints the PairState treats as dummies (patch 0) fall back to argmax
                    pr[dummy] = preds['argmax'][dummy]
                    return pr, (stt[-1] if stt else None)
                if ARM == 'both':   # analysis: decode in both arms; 'ours' = dist rule; per-pair stats saved
                    d0 = float((2 - 2 * (fs @ ft.t()).max(1)[0]).clamp(min=0).sqrt().mean())
                    d1 = float((2 - 2 * (ff @ ft.t()).max(1)[0]).clamp(min=0).sqrt().mean())
                    p0, s0 = run(0)
                    p1, s1 = run(1)
                    preds['ours0'], preds['ours1'] = p0, p1
                    preds['ours'] = p1 if d1 < d0 else p0
                    armstats.setdefault(cls, []).append(dict(d0=d0, d1=d1, s0=s0, s1=s1, nok=int(ok1.sum())))
                else:
                    preds['ours'] = run(arm)[0]
                tdec += time.time() - t1
            for h in heads:
                acc[h][cls].append(score(preds[h], ks, kt, p, scale * S, map_s, map_t))
            npair += 1
        line = '  '.join(f"{h}: SOC {100*np.mean([a for a, _ in acc[h][cls] if a is not None]):5.1f} "
                         f"CC {100*np.mean([b for _, b in acc[h][cls] if b is not None]):5.1f}" for h in heads)
        print(f'[{ci+1}/{len(classes)}] {cls:24s} {line} ({time.time()-t0:.0f}s)', flush=True)
    # aggregate: per class mean over pairs, mean over classes; paired bootstrap over pairs within class (ours vs argmax)
    res = {}
    for h in heads:
        soc = [np.mean([a for a, _ in acc[h][c] if a is not None]) for c in classes if any(a is not None for a, _ in acc[h][c])]
        cc = [np.mean([b for _, b in acc[h][c] if b is not None]) for c in classes if any(b is not None for _, b in acc[h][c])]
        res[h] = {'SOC': 100 * float(np.mean(soc)), 'CC': 100 * float(np.mean(cc))}
    boot = {}
    rng = np.random.RandomState(0)
    for h in heads[1:]:
        ds, dc = [], []
        for _ in range(200):
            s1, c1 = [], []
            for c in classes:
                n = len(acc[h][c])
                if n == 0:
                    continue
                ii = rng.randint(0, n, n)
                a = [(acc[h][c][i][0], acc['argmax'][c][i][0]) for i in ii if acc[h][c][i][0] is not None]
                b = [(acc[h][c][i][1], acc['argmax'][c][i][1]) for i in ii if acc[h][c][i][1] is not None]
                if a:
                    s1.append(np.mean([x - y for x, y in a]))
                if b:
                    c1.append(np.mean([x - y for x, y in b]))
            ds.append(100 * np.mean(s1)); dc.append(100 * np.mean(c1))
        boot[h] = {'dSOC_std': float(np.std(ds)), 'dCC_std': float(np.std(dc))}
    summ = dict(split=split, backbone=bb, image_size=S, n_pairs=npair, n_classes=len(classes), max_pairs=maxp,
                ovr=os.environ.get('OVR', '') + (' PADMASK' if PADM else '') + (' NOFLIP' if NOFLIP else ''), out=OUT, arm=ARM, qb=QB, sec=time.time() - t0, sec_decode=tdec,
                results=res, boot=boot)
    os.makedirs(out, exist_ok=True)
    json.dump(summ, open(f'{out}/soco_metrics.json', 'w'), indent=1)
    json.dump({h: {c: acc[h][c] for c in classes} for h in heads}, open(f'{out}/soco_perpair.json', 'w'))
    if armstats:
        json.dump(armstats, open(f'{out}/soco_armstats.json', 'w'))
    for h in heads:
        r = res[h]
        bs = boot.get(h, {})
        print(f"{h:8s} SOC {r['SOC']:6.2f} ({r['SOC']-res['argmax']['SOC']:+.2f} ±{bs.get('dSOC_std', 0):.2f})  "
              f"CC {r['CC']:6.2f} ({r['CC']-res['argmax']['CC']:+.2f} ±{bs.get('dCC_std', 0):.2f})")
    print(f"pairs {npair} sec {time.time()-t0:.0f} (decode {tdec:.0f})")


if __name__ == '__main__':
    main()
