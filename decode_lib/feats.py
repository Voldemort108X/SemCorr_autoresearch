"""Candidate generation + per-candidate features for the current best decoder (seq 5) and the T3 re-ranker.
arm_candidates(pair, a) -> (st, dict) with [M, K] tensors; 'base' is the seq-5 training-free score, 'map' the
refinement map [M, nP, nP]. Category-agnostic features only."""
import torch
from vendor import core as C
from decode_lib import geo
from decode_lib.consensus import consensus

CFG = dict(LAM=0.5, MU=0.5, CAP=5.0, THR=6.0, LFL=0.25, NOM_ENS=0.75, REF_ENS=0.5, K=30, WIN=2, BETA=0.03)
FEAT_NAMES = ('row', 'row_gap', 'sim_own', 'sim_fl', 'fl_ok', 'rank', 'in_pool', 'd_cyc', 'd_cyc_fl', 'res_cons',
              'softcyc', 'revrank', 'dist_top', 'peak', 'npool', 'base', 'base_gap', 'row_ent', 'n_valid')
FEAT_NAMES2 = FEAT_NAMES + ('comp', 'comp_mir', 'comp_own') + tuple(f'loc{i}' for i in range(9))
RING2 = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if max(abs(dy), abs(dx)) == 2]
FEAT_NAMES3 = FEAT_NAMES2 + tuple(f'ring{i}' for i in range(16)) + tuple(f'own{i}' for i in range(9)) + (
    'self_mir', 'self_far', 'q_relx', 'q_rely', 'c_relx', 'c_rely', 'cyc2', 'tmask_frac')


def other_arm_kp(pair, a, st):
    nP = st.num_patches
    vi = st.valid_kpt_indices
    k0, v0 = pair['src'][0]['kp'], pair['src'][0]['kp_valid']
    if a == 1:
        return k0[vi], v0[vi]
    j = pair['perm'][vi]                      # -1 = mirror partner unknown (datasets without a flip permutation)
    jc = j.clamp(min=0)
    return (k0[jc] // nP) * nP + (nP - 1 - k0[jc] % nP), v0[jc] & (j >= 0)


def mixed_pool(st, row, ktop, kpeak, r=1):
    """top-ktop patches by the row + the kpeak highest 3x3-NMS local maxima not already included (valid-target idx)."""
    nP = st.num_patches
    M = row.shape[0]
    full = torch.full((M, nP * nP), -10.0, device=st.device)
    full[:, st.tgt_valid_idx] = row
    mp = torch.nn.functional.max_pool2d(full.view(M, 1, nP, nP), 2 * r + 1, 1, r).view(M, -1)
    top = full.topk(ktop, dim=1)[1]
    pk = full.masked_fill(~((full >= mp) & (full > -9)), -10.0)
    pk = pk.scatter(1, top, -10.0)                     # exclude those already in the top list
    pv, pp = pk.topk(kpeak, dim=1)
    fill = full.scatter(1, top, -10.0).topk(kpeak, dim=1)[1]
    pp = torch.where(pv > -9, pp, fill)
    sel = torch.cat([top, pp], 1)
    inv = torch.zeros(nP * nP, dtype=torch.long, device=st.device)
    inv[st.tgt_valid_idx] = torch.arange(st.tgt_valid_idx.numel(), device=st.device)
    return inv[sel]


def arm_candidates(pair, a, cfg=CFG, with_feats=True, extra=False):
    A = pair['src'][a]
    st = C.PairState(A['desc'], pair['trg']['desc'], A['kp'], A['mask'], pair['trg']['mask'])
    if st.M == 0:
        return st, None
    nP = st.num_patches
    trg = pair['trg']['desc']
    O = pair['src'][1 - a]
    lo, ok = other_arm_kp(pair, a, st)
    own_rows = st.sim_kpts
    fr = O['desc'][lo] @ st.img2_valid.t()                                      # [M, Pt]
    w = cfg['NOM_ENS']
    row = torch.where(ok.unsqueeze(1), (own_rows + w * fr) / (1 + w), own_rows)
    st.sim_kpts = row
    st._sorted_cache = {}
    _, _, _, cutoff_k = C.nominate(st, max_k=cfg['K'])
    si, _ = st.sorted_by_temperature(0.1)
    topk_idx = si[:, :min(cfg['K'], si.shape[1])]
    if cfg.get('PEAKS', 0) > 0 and topk_idx.shape[1] == cfg['K']:   # replace the tail of the pool by NMS peaks
        topk_idx = mixed_pool(st, row, cfg['K'] - cfg['PEAKS'], cfg['PEAKS'])
        cutoff_k = torch.full_like(cutoff_k, topk_idx.shape[1])
    topk = torch.gather(row, 1, topk_idx)
    Kc = topk_idx.shape[1]
    in_pool = torch.arange(Kc, device=st.device).unsqueeze(0) < cutoff_k.unsqueeze(1)
    pid = st.tgt_valid_idx[topk_idx]
    cxy = geo.pid_xy(pid, nP)
    qxy = torch.stack([st.kpt_x, st.kpt_y], -1)
    d_cyc = torch.sqrt((st.back_nn_y[topk_idx] - st.kpt_y.unsqueeze(1)) ** 2 +
                       (st.back_nn_x[topk_idx] - st.kpt_x.unsqueeze(1)) ** 2)
    # other-arm cycle
    tc = trg[pid.reshape(-1)]
    omask = O['mask'].clone()
    omask[lo[ok]] = True
    ov = torch.nonzero(omask, as_tuple=True)[0]
    bn = ov[(O['desc'][ov] @ tc.t()).argmax(0)].view(*pid.shape)
    dfl = (geo.pid_xy(bn, nP) - geo.pid_xy(lo, nP).unsqueeze(1)).norm(dim=-1)
    okk = ok.unsqueeze(1).expand_as(topk)
    dfl = torch.where(okk, dfl, d_cyc)
    u = topk - cfg['LAM'] * d_cyc / nP
    valid = A['kp_valid'][st.valid_kpt_indices]
    g = torch.Generator(device=st.device)
    g.manual_seed(0)
    res, _ = consensus(qxy, cxy, u, valid, model='sim', thr=cfg['THR'], gen=g)
    base = u - cfg['LFL'] * dfl / nP - cfg['MU'] * res.clamp(max=cfg['CAP']) / nP
    # refinement map
    fm = (O['desc'][lo] @ trg.t()).masked_fill(~pair['trg']['mask'].unsqueeze(0), -float('inf')).view(-1, nP, nP)
    fl_map = torch.where(ok.view(-1, 1, 1), fm, st.full_sim_2d)
    rmap = (st.full_sim_2d + cfg['REF_ENS'] * fl_map) / (1 + cfg['REF_ENS'])
    if cfg.get('REF_CYC', 0) > 0:   # cycle-penalised refinement map (as decode.py REF_CYC)
        bxy = torch.stack([st.back_nn_x, st.back_nn_y], -1)
        pen = torch.zeros(rmap.shape[0], nP * nP, device=st.device)
        pen[:, st.tgt_valid_idx] = (bxy.unsqueeze(0) - qxy.unsqueeze(1)).norm(dim=-1)
        rmap = rmap - cfg['REF_CYC'] * pen.view(-1, nP, nP) / nP
    out = dict(pid=pid, in_pool=in_pool, base=base, map=rmap, valid=valid)
    if not with_feats:
        return st, out
    sim_own = torch.gather(own_rows, 1, topk_idx)
    sim_fl = torch.where(okk, torch.gather(fr, 1, topk_idx), sim_own)
    S = st.sim_valid
    Pc = torch.softmax(S[:, topk_idx.reshape(-1)] / 0.02, dim=0)
    exy = (Pc.t() @ geo.pid_xy(st.src_valid_idx, nP)).view(*topk_idx.shape, 2)
    softcyc = (exy - qxy.unsqueeze(1)).norm(dim=-1)
    cols = S[:, topk_idx.reshape(-1)].view(S.shape[0], *topk_idx.shape)
    revrank = torch.log1p((cols > torch.gather(own_rows, 1, topk_idx).unsqueeze(0)).sum(0).float())
    dist_top = (cxy - cxy[:, :1]).norm(dim=-1)
    # peakiness of the refinement map at the candidate: value minus mean of its 3x3 neighbourhood
    pm = rmap.clamp(min=-1).unsqueeze(1)
    avg = torch.nn.functional.avg_pool2d(pm, 3, 1, 1).view(pm.shape[0], -1)
    peak = torch.gather(pm.view(pm.shape[0], -1), 1, pid) - torch.gather(avg, 1, pid)
    p = torch.softmax(row / 0.1, dim=1)
    ent = -(p * torch.log(p.clamp(min=1e-12))).sum(1, keepdim=True).expand_as(topk)
    bm = base.masked_fill(~in_pool, -1e9).max(1, keepdim=True)[0]
    F = dict(row=topk, row_gap=topk - topk[:, :1], sim_own=sim_own, sim_fl=sim_fl, fl_ok=okk.float(),
             rank=torch.arange(Kc, device=st.device).float().expand_as(topk) / 30, in_pool=in_pool.float(),
             d_cyc=d_cyc / nP, d_cyc_fl=dfl / nP, res_cons=res.clamp(max=20) / nP, softcyc=softcyc.clamp(max=30) / nP,
             revrank=revrank, dist_top=dist_top / nP, peak=peak, npool=cutoff_k.float().unsqueeze(1).expand_as(topk) / 30,
             base=base, base_gap=(base - bm).clamp(min=-1), row_ent=ent / 8,
             n_valid=valid.float().sum().expand_as(topk) / 20)
    names = FEAT_NAMES
    if extra:
        # competition: is the candidate better explained by ANOTHER query keypoint of this pair?
        vi = st.valid_kpt_indices
        M = vi.numel()
        Rj = row[:, topk_idx.reshape(-1)].view(M, M, Kc)                      # [j, i, k] = row_j(cand_ik)
        vmask = valid.view(M, 1, 1).expand(M, M, Kc) & ~torch.eye(M, dtype=torch.bool, device=st.device).unsqueeze(-1)
        comp = (Rj.masked_fill(~vmask, -2).max(0)[0] - topk).clamp(min=-1)
        F['comp'] = comp
        pos = torch.full((pair['perm'].numel(),), -1, dtype=torch.long, device=st.device)
        pos[vi] = torch.arange(M, device=st.device)
        pv = pair['perm'][vi]
        pj = torch.where(pv >= 0, pos[pv.clamp(min=0)], torch.full_like(pv, -1))   # row index of the mirror partner
        okm = (pj >= 0) & (pj != torch.arange(M, device=st.device))
        okm = okm & valid[pj.clamp(min=0)]
        cm = Rj[pj.clamp(min=0), torch.arange(M, device=st.device)] - topk
        F['comp_mir'] = torch.where(okm.unsqueeze(1), cm, torch.zeros_like(cm))
        Ro = own_rows[:, topk_idx.reshape(-1)].view(M, M, Kc)
        F['comp_own'] = (Ro.masked_fill(~vmask, -2).max(0)[0] - sim_own).clamp(min=-1)
        # local 3x3 shape of the refinement map around the candidate (relative to the centre)
        cy, cx = pid // nP, pid % nP
        rm = rmap.clamp(min=-1)
        c0 = rm.view(M, -1).gather(1, pid)
        for t, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]):
            yy, xx = (cy + dy).clamp(0, nP - 1), (cx + dx).clamp(0, nP - 1)
            F[f'loc{t}'] = rm.view(M, -1).gather(1, yy * nP + xx) - c0 if (dy, dx) != (0, 0) else c0
        names = FEAT_NAMES2
    if extra == 2:
        M = st.valid_kpt_indices.numel()
        rmv = rmap.clamp(min=-1).view(M, -1)
        for t, (dy, dx) in enumerate(RING2):
            yy, xx = (cy + dy).clamp(0, nP - 1), (cx + dx).clamp(0, nP - 1)
            F[f'ring{t}'] = rmv.gather(1, yy * nP + xx) - c0
        om = st.full_sim_2d.clamp(min=-1).view(M, -1)
        o0 = om.gather(1, pid)
        for t, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]):
            yy, xx = (cy + dy).clamp(0, nP - 1), (cx + dx).clamp(0, nP - 1)
            F[f'own{t}'] = om.gather(1, yy * nP + xx) - o0 if (dy, dx) != (0, 0) else o0
        # query ambiguity in the source: similarity to its own mirror partner / to far-away source patches
        sd_ = A['desc']
        qd = sd_[st.idx_valid]
        pidx = torch.where(okm, st.idx_valid[pj.clamp(min=0)], st.idx_valid)
        F['self_mir'] = torch.where(okm, (qd * sd_[pidx]).sum(-1), torch.zeros(M, device=st.device)).unsqueeze(1).expand_as(topk)
        Ssrc = qd @ st.img1_valid.t()                                              # [M, Ps]
        sxy = geo.pid_xy(st.src_valid_idx, nP)
        far = (sxy.unsqueeze(0) - qxy.unsqueeze(1)).norm(dim=-1) > 5
        F['self_far'] = Ssrc.masked_fill(~far, -1).max(1)[0].unsqueeze(1).expand_as(topk)
        # positions relative to the object (mask bounding boxes)
        smx = geo.pid_xy(torch.nonzero(A['mask'], as_tuple=True)[0], nP)
        tmx = geo.pid_xy(torch.nonzero(pair['trg']['mask'], as_tuple=True)[0], nP)
        if smx.numel() == 0:
            smx = sxy
        if tmx.numel() == 0:
            tmx = geo.pid_xy(st.tgt_valid_idx, nP)
        s_lo, s_hi = smx.min(0)[0], smx.max(0)[0]
        t_lo, t_hi = tmx.min(0)[0], tmx.max(0)[0]
        qr = (qxy - s_lo) / (s_hi - s_lo).clamp(min=1)
        cr = (cxy - t_lo) / (t_hi - t_lo).clamp(min=1)
        F['q_relx'], F['q_rely'] = qr[:, 0:1].expand_as(topk), qr[:, 1:2].expand_as(topk)
        F['c_relx'], F['c_rely'] = cr[..., 0], cr[..., 1]
        # 2-hop: candidate -> back-NN -> forward NN; distance back to the candidate (patches)
        bn_valid = st.src_orig2valid[st.back_nn_orig[topk_idx]]                      # [M,K] valid-src idx
        fwd = st.sim_valid[bn_valid.reshape(-1)].argmax(1).view(*topk_idx.shape)
        F['cyc2'] = (geo.pid_xy(st.tgt_valid_idx[fwd], nP) - cxy).norm(dim=-1) / nP
        F['tmask_frac'] = torch.full_like(topk, float(pair['trg']['mask'].float().mean()))
        names = FEAT_NAMES3
    out['F'] = torch.stack([F[k] for k in names], -1)
    return st, out


def refine(st, out, best, cfg=CFG):
    nP = st.num_patches
    bp = out['pid'].gather(1, best[:, None])[:, 0]
    return C.window_soft_argmax_2d(out['map'], bp // nP, bp % nP, cfg['WIN'], cfg['BETA'], st.anno_stride)
