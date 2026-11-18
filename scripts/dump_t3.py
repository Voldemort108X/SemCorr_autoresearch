"""T3 training/analysis tables: python scripts/dump_t3.py <split> [max_pairs_per_cat] [cats]
-> cache/tlr/<split>/t3[_n<max>].npz : F [n,K,F], in_pool, base, err [n,K] (GT error of each candidate after the
seq-5 refinement), cat, pair, kpt. Uses GT only for 'err' (labels)."""
import os, sys, time
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import prepare as P
from vendor import core as C
from decode_lib import feats as FT

split = sys.argv[1]
maxp = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '0' else None
cats = sys.argv[3].split(',') if len(sys.argv) > 3 else P.CATS
K = FT.CFG['K']
EXTRA = int(os.environ.get('T3_EXTRA', '0'))
rows = dict(F=[], in_pool=[], base=[], err=[], cat=[], pair=[], kpt=[])
t0 = time.time()
rng = np.random.default_rng(1)
with torch.no_grad():
    for cat in cats:
        meta = P.load_split('tlr', split, cat)
        F_ = P.CatFeats('tlr', cat)
        idx = np.arange(len(meta['src']))
        if maxp and len(idx) > maxp:
            idx = np.sort(rng.choice(idx, maxp, replace=False))
        for j in idx:
            k2 = meta['k2'][j]
            vis = (meta['k1'][j][:, 2] > 0) & (k2[:, 2] > 0)
            if vis.sum() == 0:
                continue
            pair = P.make_pair(F_, meta, j, cat)
            bbox = float(meta['bbox'][j])
            gt = torch.from_numpy(k2[:, :2]).float().cuda()
            arm = pair['arm_of_kpt']
            for a in ([0, 1] if bool((arm == 1).any()) else [0]):
                st, o = FT.arm_candidates(pair, a, extra=EXTRA)
                if o is None:
                    continue
                vi = st.valid_kpt_indices
                take = (arm[vi] == a) & torch.from_numpy(vis).cuda()[vi]
                if take.sum() == 0:
                    continue
                pid = o['pid'][take]
                Kc = pid.shape[1]
                mp = o['map'][take].unsqueeze(1).expand(-1, Kc, -1, -1).reshape(-1, C.NP, C.NP)
                nx, ny = C.window_soft_argmax_2d(mp, (pid // C.NP).reshape(-1), (pid % C.NP).reshape(-1),
                                                 FT.CFG['WIN'], FT.CFG['BETA'], st.anno_stride)
                g = gt[vi[take]]
                err = torch.sqrt((nx.view(-1, Kc) - g[:, 0:1]) ** 2 + (ny.view(-1, Kc) - g[:, 1:2]) ** 2) / bbox
                pad = lambda v, fill: torch.cat([v, torch.full((v.shape[0], K - Kc, *v.shape[2:]), fill, device=v.device, dtype=v.dtype)], 1) if Kc < K else v
                rows['F'].append(pad(o['F'][take], 0.0).half().cpu().numpy())
                rows['in_pool'].append(pad(o['in_pool'][take], False).cpu().numpy())
                rows['base'].append(pad(o['base'][take], -1e9).cpu().numpy())
                rows['err'].append(pad(err, 9.0).cpu().numpy())
                n = int(take.sum())
                rows['cat'] += [cat] * n
                rows['pair'] += [f'{cat}/{int(meta["pid"][j])}'] * n
                rows['kpt'].append(vi[take].cpu().numpy())
        print(cat, f'{time.time() - t0:.0f}s', flush=True)
        del F_
out = {k: (np.concatenate(v) if k not in ('cat', 'pair') else np.array(v)) for k, v in rows.items()}
tag = {0: '', 1: '_x', 2: '_x2'}[EXTRA] + (f'_n{maxp}' if maxp else '')
tag += '' if cats == P.CATS else '_' + '-'.join(cats)
np.savez(f'{P.cdir("tlr", split)}/t3{tag}.npz', **out, feat_names=np.array({0: FT.FEAT_NAMES, 1: FT.FEAT_NAMES2, 2: FT.FEAT_NAMES3}[EXTRA]))
print('saved', out['F'].shape, f'{time.time() - t0:.0f}s')
