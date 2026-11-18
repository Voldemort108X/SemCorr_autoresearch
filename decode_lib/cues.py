"""Per-candidate cues for one arm of a pair (pool = top-K by S, K = 30)."""
import torch
from vendor import core as C
from decode_lib import geo


def arm_state(pair, a):
    A = pair['src'][a]
    return C.PairState(A['desc'], pair['trg']['desc'], A['kp'], A['mask'], pair['trg']['mask'])


def candidate_cues(st, K=30, mls_sigmas=(4.0, 8.0, 16.0), seed=0, other_desc=None, trg_desc=None, kp_valid=None):
    """Returns dict of [M, K] tensors + 'pid' [M, K] (target patch ids) + 'in_pool' (Deja Vu's top-p pool)."""
    nP = st.num_patches
    _, _, _, cutoff_k = C.nominate(st, max_k=K)
    si, _ = st.sorted_by_temperature(0.1)
    topk_idx = si[:, :min(K, si.shape[1])]
    topk_sim = torch.gather(st.sim_kpts, 1, topk_idx)
    in_pool = torch.arange(topk_idx.shape[1], device=st.device).unsqueeze(0) < cutoff_k.unsqueeze(1)
    pid = st.tgt_valid_idx[topk_idx]
    cxy = geo.pid_xy(pid, nP)                                               # [M,K,2]
    qxy = torch.stack([st.kpt_x, st.kpt_y], -1)                             # [M,2]
    d_cyc = torch.sqrt((st.back_nn_y[topk_idx] - st.kpt_y.unsqueeze(1)) ** 2 +
                       (st.back_nn_x[topk_idx] - st.kpt_x.unsqueeze(1)) ** 2)
    out = dict(pid=pid, in_pool=in_pool, sim=topk_sim, d_cyc=d_cyc,
               rank=torch.arange(topk_idx.shape[1], device=st.device).float().expand_as(topk_sim))
    # soft cycle: expected reverse location under softmax over source (column) / tau_r
    S = st.sim_valid
    src_xy = geo.pid_xy(st.src_valid_idx, nP)                               # [Ps,2]
    for tr in (0.02, 0.05):
        Pc = torch.softmax(S[:, topk_idx.reshape(-1)] / tr, dim=0)          # [Ps, M*K]
        exy = (Pc.t() @ src_xy).view(*topk_idx.shape, 2)
        out[f'softcyc{tr}'] = (exy - qxy.unsqueeze(1)).norm(dim=-1)
    # reverse rank of the query among the candidate column
    cols = S[:, topk_idx.reshape(-1)].view(S.shape[0], *topk_idx.shape)     # [Ps,M,K]
    out['revrank'] = (cols > topk_sim.unsqueeze(0)).sum(0).float()
    # handedness / mirror test: similarity to the query vs. to the query's mirror image (other arm, col 59-x)
    if other_desc is not None:
        mir = (st.idx_valid // nP) * nP + (nP - 1 - st.idx_valid % nP)
        qm = other_desc[mir]                                                 # [M,D]
        tc = trg_desc[pid.reshape(-1)].view(*pid.shape, -1)                 # [M,K,D]
        smir = (tc * qm.unsqueeze(1)).sum(-1)
        out['sim_mir'] = smir
        out['hand'] = topk_sim - smir
    # consensus over the keypoints' pools (pair-level transform)
    from decode_lib.consensus import consensus
    valid = kp_valid[st.valid_kpt_indices] if kp_valid is not None else torch.ones(st.M, dtype=torch.bool, device=st.device)
    u = topk_sim - 0.5 * d_cyc / nP
    for model in ('sim', 'aff'):
        for thr in (3.0, 6.0):
            g2 = torch.Generator(device=st.device)
            g2.manual_seed(seed)
            r, info = consensus(qxy, cxy, u, valid, model=model, thr=thr, gen=g2)
            out[f'res_c{model}{int(thr)}'] = r
    out['n_valid'] = torch.full_like(topk_sim, float(valid.sum()))
    # neighbourhood context agreement (scale from mask areas)
    if trg_desc is not None:
        A = st.src_mask
        sc = float(geo.mask_scale(st.tgt_mask) / geo.mask_scale(A))
        sdesc = st._src_desc_full
        for pool in (0, 1):
            cs = geo.context_score(sdesc, trg_desc, st.tgt_mask, qxy, cxy, radii=(2, 4, 8), scale=sc, pool=pool)
            for r, v in cs.items():
                out[f'ctx{r}p{pool}'] = v
            cs = geo.context_score(sdesc, trg_desc, st.tgt_mask, qxy, cxy, radii=(4,), scale=1.0, pool=pool)
            out[f'ctx4p{pool}s1'] = cs[4]
    # anchors
    X, Y, w = geo.mnn_anchors(st)
    g = torch.Generator(device=st.device)
    g.manual_seed(seed)
    B, inl = geo.ransac_affine(X, Y, None, iters=512, thr=3.0, gen=g)
    pred = geo.apply_affine(B, qxy)                                          # [M,2]
    out['res_aff'] = (cxy - pred.unsqueeze(1)).norm(dim=-1)
    out['n_anchor'] = torch.full_like(topk_sim, float(X.shape[0]))
    out['n_inl'] = torch.full_like(topk_sim, float(inl.sum()))
    for sg in mls_sigmas:
        if X.shape[0] >= 3:
            p = geo.mls_affine(qxy, X, Y, torch.ones_like(w), sg)
            out[f'res_mls{int(sg)}'] = (cxy - p.unsqueeze(1)).norm(dim=-1)
            if inl.sum() >= 3:
                p2 = geo.mls_affine(qxy, X[inl], Y[inl], torch.ones_like(w[inl]), sg)
                out[f'res_mlsi{int(sg)}'] = (cxy - p2.unsqueeze(1)).norm(dim=-1)
            else:
                out[f'res_mlsi{int(sg)}'] = out[f'res_mls{int(sg)}']
        else:
            out[f'res_mls{int(sg)}'] = torch.zeros_like(topk_sim)
            out[f'res_mlsi{int(sg)}'] = torch.zeros_like(topk_sim)
    return out
