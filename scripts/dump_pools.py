"""Dump per-candidate cue tables (top-30 pool) + per-candidate refined error, for offline re-ranker analysis.
python scripts/dump_pools.py <method> <split> [cats] -> cache/<method>/<split>/pools.npz  (analysis only; uses GT for err)"""
import os, sys, time
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import prepare as P
from vendor import core as C
from decode_lib import cues as Q

K = 30
method, split = sys.argv[1], sys.argv[2]
cats = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] else P.CATS
rows = {}
meta_rows = dict(cat=[], pair=[], kpt=[], arm=[])
t0 = time.time()
with torch.no_grad():
    for cat in cats:
        meta = P.load_split(method, split, cat)
        F_ = P.CatFeats(method, cat)
        for j in range(len(meta['src'])):
            k2 = meta['k2'][j]
            vis = (meta['k1'][j][:, 2] > 0) & (k2[:, 2] > 0)
            if vis.sum() == 0:
                continue
            pair = P.make_pair(F_, meta, j, cat)
            bbox = float(meta['bbox'][j])
            gt = torch.from_numpy(k2[:, :2]).float().cuda()
            arm = pair['arm_of_kpt']
            for a in ([0, 1] if bool((arm == 1).any()) else [0]):
                st = Q.arm_state(pair, a)
                if st.M == 0:
                    continue
                vi = st.valid_kpt_indices
                take = ((arm[vi] == a) & torch.from_numpy(vis).cuda()[vi])
                if take.sum() == 0:
                    continue
                cu = Q.candidate_cues(st, K, other_desc=pair['src'][1 - a]['desc'], trg_desc=pair['trg']['desc'], kp_valid=pair['src'][a]['kp_valid'])
                pid = cu['pid']
                Kc = pid.shape[1]
                s2 = st.full_sim_2d.unsqueeze(1).expand(-1, Kc, -1, -1).reshape(-1, C.NP, C.NP)
                nx, ny = C.window_soft_argmax_2d(s2, (pid // C.NP).reshape(-1), (pid % C.NP).reshape(-1), 5, 0.01)
                g = gt[vi]
                err = torch.sqrt((nx.view(-1, Kc) - g[:, 0:1]) ** 2 + (ny.view(-1, Kc) - g[:, 1:2]) ** 2) / bbox
                cu['err'] = err
                cu['qx'] = st.kpt_x.unsqueeze(1).expand(-1, Kc)
                cu['qy'] = st.kpt_y.unsqueeze(1).expand(-1, Kc)
                cu['bbox'] = torch.full_like(err, bbox)
                cu['px'], cu['py'] = nx.view(-1, Kc), ny.view(-1, Kc)
                # analysis labels: nearest OTHER visible target keypoint (within 0.1 bbox) and mirror flag
                k2v = torch.from_numpy(k2[:, 2] > 0).cuda()
                D = torch.sqrt((cu['px'].unsqueeze(-1) - gt[:, 0]) ** 2 + (cu['py'].unsqueeze(-1) - gt[:, 1]) ** 2) / bbox
                D[..., ~k2v] = 1e3
                D[torch.arange(len(vi)), :, vi] = 1e3
                dm, nk = D.min(-1)
                nk = torch.where(dm <= 0.1, nk, torch.full_like(nk, -1))
                cu['near_kp'] = nk.float()
                pm = pair['perm'][vi] if a == 0 else pair['perm'][vi]
                cu['is_mirror'] = ((nk == pm.unsqueeze(1)) & (pm != vi).unsqueeze(1)).float()
                for k, v in cu.items():
                    v = v[take].float()
                    if v.shape[1] < K:
                        v = torch.cat([v, torch.full((v.shape[0], K - v.shape[1]), float('nan'), device=v.device)], 1)
                    rows.setdefault(k, []).append(v.cpu().numpy().astype(np.float32))
                n = int(take.sum())
                meta_rows['cat'] += [cat] * n
                meta_rows['pair'] += [f'{cat}/{int(meta["pid"][j])}'] * n
                meta_rows['kpt'].append(vi[take].cpu().numpy())
                meta_rows['arm'] += [a] * n
        print(cat, f'{time.time() - t0:.0f}s', flush=True)
        del F_
out = {k: np.concatenate(v) for k, v in rows.items()}
out.update(cat=np.array(meta_rows['cat']), pair=np.array(meta_rows['pair']),
           kpt=np.concatenate(meta_rows['kpt']), arm=np.array(meta_rows['arm']))
tag = '' if cats == P.CATS else '_' + '-'.join(cats)
np.savez(f'{P.cdir(method, split)}/pools{tag}.npz', **out)
print('saved', {k: v.shape for k, v in out.items()}, f'{time.time() - t0:.0f}s')
