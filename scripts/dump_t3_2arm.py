"""Two-arm T3 tables: python scripts/dump_t3_2arm.py <split> [max_pairs_per_cat] -> cache/tlr/<split>/t3_2arm[_n].npz"""
import os, sys, time
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import prepare as P
from vendor import core as C
from decode_lib import feats as FT
from decode_lib.twoarm import two_arm_candidates

split = sys.argv[1]
maxp = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '0' else None
rows = dict(F=[], in_pool=[], err=[], cat=[], pair=[], kpt=[])
rng = np.random.default_rng(1)
t0 = time.time()
with torch.no_grad():
    for cat in P.CATS:
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
            tw = two_arm_candidates(pair)
            if tw is None:
                continue
            bbox = float(meta['bbox'][j])
            gt = torch.from_numpy(k2[:, :2]).float().cuda()
            N, K = tw['F'].shape[0], tw['K']
            err = torch.full((N, 2 * K), 9.0, device='cuda')
            for a, (st, o) in enumerate(tw['per']):
                if o is None:
                    continue
                vi = st.valid_kpt_indices
                pid = o['pid']
                Kc = pid.shape[1]
                mp = o['map'].unsqueeze(1).expand(-1, Kc, -1, -1).reshape(-1, C.NP, C.NP)
                nx, ny = C.window_soft_argmax_2d(mp, (pid // C.NP).reshape(-1), (pid % C.NP).reshape(-1),
                                                 FT.CFG['WIN'], FT.CFG['BETA'], st.anno_stride)
                g = gt[vi]
                e = torch.sqrt((nx.view(-1, Kc) - g[:, 0:1]) ** 2 + (ny.view(-1, Kc) - g[:, 1:2]) ** 2) / bbox
                err[vi, a * K:a * K + Kc] = e
            take = torch.from_numpy(vis).cuda() & tw['has']
            if take.sum() == 0:
                continue
            rows['F'].append(tw['F'][take].half().cpu().numpy())
            rows['in_pool'].append(tw['pool'][take].cpu().numpy())
            rows['err'].append(err[take].cpu().numpy())
            n = int(take.sum())
            rows['cat'] += [cat] * n
            rows['pair'] += [f'{cat}/{int(meta["pid"][j])}'] * n
            rows['kpt'].append(torch.nonzero(take, as_tuple=True)[0].cpu().numpy())
        print(cat, f'{time.time() - t0:.0f}s', flush=True)
        del F_
out = {k: (np.concatenate(v) if k not in ('cat', 'pair') else np.array(v)) for k, v in rows.items()}
out['base'] = np.zeros(out['err'].shape, np.float32)
tag = f'_n{maxp}' if maxp else ''
np.savez(f'{P.cdir("tlr", split)}/t3_2arm{tag}.npz', **out, feat_names=np.array(list(FT.FEAT_NAMES3) + ['cand_arm', 'base_arm']))
print('saved', out['F'].shape, f'{time.time() - t0:.0f}s')
