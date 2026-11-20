"""decode.py -- THE file the loop edits (program.md §3).

decode(pair) -> pred [N, 2] (x, y in the 840-px annotation frame) for the N source keypoints.
pair (all GPU tensors; no target keypoints / bbox / labels):
  src[a]       a = 0 original source, 1 horizontally flipped source; each:
                 desc [P, D] fp32 l2-normalised, mask [P] bool (dilated patch mask),
                 kp [N] long patch index of every keypoint in this arm (flip arm: mirror-permuted),
                 kp_valid [N] bool (source keypoint annotated)
  trg          desc [P, D], mask [P]
  arm_of_kpt   [N] long: arm whose prediction is scored for each keypoint (frozen adaptive flip)
  perm         [N] long: SPair mirror permutation (keypoint i <-> its left/right counterpart)
  nP, anno     60, 840

Current: Deja Vu unary (top-p pool, S - lam d_cyc / w) + pair-level consensus residual (T2):
  RANSAC over the query keypoints' candidate pools fits a similarity transform T (2-point hypotheses from the
  top-5 candidates by unary, support = rank-discounted inlier count, thr 6 patches, refit affine on inliers);
  R = S - lam d_cyc / w - mu * min(|c - T(q)|, cap) / w.
+ flip-ensemble (other source arm; see flip_ensemble) in the score and in the window soft-argmax refinement map.
"""
import torch
from vendor import core as C
from decode_lib import geo
from decode_lib.consensus import consensus

LAM = 0.5
MU, CAP, THR, MODEL = 0.5, 5.0, 6.0, 'sim'
WFL, LFL = 0.0, 0.25          # flip-ensemble: weight of the other-arm similarity and of its cycle distance
K = 30
WIN, BETA, REF_ENS = 2, 0.03, 1.0
REF_MODE = 'soft'   # 'soft' window soft-argmax | 'quad': local peak (W1) + separable parabola sub-patch offset | 'mix'
TTA_SUB = 0      # sub-sample patches in the TTA objective (0 = all)
SELF_TRAIN = 0.0   # >0: second TTA with pseudo-labels from a first decode (weight of the CE term), then re-decode
ROW_SMOOTH2 = 0.0   # weight of a 5x5-smoothed own-query view in the forward row
ROW_SMOOTH_FL = 0.0   # weight of a smoothed flipped-arm view in the forward row
ROW_SMOOTH = 0.7   # weight of a smoothed-query view in the forward row (selection only)
REF_ORIG, SEL_ORIG = False, False   # TTA ablations: refinement / selection on the non-adapted descriptors
DECOL_R, DECOL_EX, DECOL_MARGIN = 0.0, 3.0, 1e9   # mirror-pair de-collision radius / exclusion / max score drop
SWAP_T = -1.0    # mirror-pair orientation test: swap the two predictions of a mirror pair if cos < -SWAP_T (-1 = off)
K_PEAK, R_NMS, K_TOP = 0, 1, 0   # peak pool: K_PEAK local maxima (NMS radius R_NMS) of the forward row + top K_TOP patches
DENSE_MU, DENSE_R = 0.0, 6.0   # dense-flow residual weight / source radius (patches)
REF_CYC, REF_SIG = 0.0, 0.0   # refinement on sim - REF_CYC*d_cyc/w ; Gaussian window prior sigma (patches)
TTA_STEPS, TTA_REG = 20, 0.0   # per-pair test-time channel re-weighting (label-free)
TTA_LR, TTA_TAU, TTA_ARM, TTA_BOTH, TTA_OBJ = 0.05, 0.05, 0, False, 'dsmax'   # TTA_ARM -1 = pair's arm
PARTNER_EST, PARTNER_FORCE = 0.0, False   # estimate unknown mirror partners (exclusion radius around the self-mirror)
TTA_NOMASK = False   # TTA objective over all patches (not only the dilated masks)
TTA_ENS_OBJ = ''   # second TTA objective to ensemble with ('' = off)
TTA_MIX = 1.0    # <1: mix original and adapted similarities
TTA_BLOCKNORM = False   # TTA: re-normalise SD and DINO blocks separately after weighting
TTA_SEM = 0.0    # TTA: source-side semantic contrast weight (annotated kps + mirror permutation)
TTA_PRIOR = ''   # path of averaged TTA weights (unsupervised, from trn pairs): fixed (TTA_STEPS 0) or init
TTA_QW = 0.0     # TTA: extra weight on the query keypoint rows' mutual confidence
TTA_HAND = 0.0   # TTA: weight of the mirror-dissimilarity (handedness) term
TTA_SEP = False  # separate TTA weights for source and target
TTA_GROUPS = 0   # TTA weights shared within contiguous channel groups (0 = per channel)
TTA_RANK = 0     # extra low-rank linear adapter in the TTA
ENS_MIN = 0.0    # flip-ensemble disagreement penalty
REF_UP = 1       # refinement map up-sampling factor (1 = off)
HAND2 = 0.0      # mirror-pair handedness direction weight
REF_CONS = 0.0   # refinement map also penalised by mu * min(dist to consensus prediction, cap)
ZROW = 0.0       # pool-adaptive penalty scaling exponent (0 = off)
PROF = 0.0       # relational-profile correlation weight
RR_FL = 0.0      # reverse-rank penalty in the other arm
FL_ALIGN = 0     # snap radius (patches) of the other-arm keypoint location (0 = off)
MU_ADAPT = 0.0   # >0: scale mu by min(1, inlier_ratio / MU_ADAPT) of the consensus fit
CONS_ITER = 0    # extra consensus passes using the full score as the RANSAC unary
RR = 0.02        # reverse-rank penalty: RR * log1p(rank of the query in the candidate's column of the own-arm S)
NOM_ENS = 0.75    # weight of the other-arm row in the forward row used for nomination + top-k similarity


def decode_arm(pair, a):
    A = pair['src'][a]
    st = C.PairState(A['desc'], pair['trg']['desc'], A['kp'], A['mask'], pair['trg']['mask'])
    N = A['kp'].numel()
    out = torch.zeros(N, 2, device='cuda')
    if st.M == 0:
        return out
    nP = st.num_patches
    own_rows = st.sim_kpts
    if NOM_ENS > 0:
        lo, ok = other_arm_kp(pair, a, st)
        O = pair['src'][1 - a]
        fr = O['desc'][lo] @ st.img2_valid.t()                              # [M, Pt]
        ens = (st.sim_kpts + NOM_ENS * fr) / (1 + NOM_ENS)
        if ENS_MIN > 0:   # penalise disagreement of the two views (soft AND)
            ens = ens - ENS_MIN * (st.sim_kpts - fr).abs()
        row = torch.where(ok.unsqueeze(1), ens, st.sim_kpts)
        if ROW_SMOOTH > 0:   # third view for the forward row only: 3x3-smoothed own query
            qs = smooth_query(A['desc'], st.idx_valid, nP, w=1.0)
            row = (row + ROW_SMOOTH * (qs @ st.img2_valid.t())) / (1 + ROW_SMOOTH)
            if ROW_SMOOTH2 > 0:   # coarser (5x5) smoothed view
                q2 = smooth_query(A['desc'], st.idx_valid, nP, w=1.0, r=2)
                row = (row + ROW_SMOOTH2 * (q2 @ st.img2_valid.t())) / (1 + ROW_SMOOTH2)
            if ROW_SMOOTH_FL > 0:   # fourth view: smoothed flipped-arm view (where the mirror partner is known)
                qf = smooth_query(O['desc'], lo, nP, w=1.0)
                rf = qf @ st.img2_valid.t()
                row = torch.where(ok.unsqueeze(1), (row + ROW_SMOOTH_FL * rf) / (1 + ROW_SMOOTH_FL), row)
        st.sim_kpts = row
        st._sorted_cache = {}
    if K_PEAK > 0:
        topk_idx = peak_pool(st, K_PEAK, R_NMS, K_TOP)
        topk_sim = torch.gather(st.sim_kpts, 1, topk_idx)
        in_pool = torch.ones_like(topk_idx, dtype=torch.bool)
    else:
        _, _, _, cutoff_k = C.nominate(st, max_k=K)
        si, _ = st.sorted_by_temperature(0.1)
        topk_idx = si[:, :min(K, si.shape[1])]
        topk_sim = torch.gather(st.sim_kpts, 1, topk_idx)
        in_pool = torch.arange(topk_idx.shape[1], device=st.device).unsqueeze(0) < cutoff_k.unsqueeze(1)
    pid = st.tgt_valid_idx[topk_idx]
    cxy = geo.pid_xy(pid, nP)
    qxy = torch.stack([st.kpt_x, st.kpt_y], -1)
    d_cyc = torch.sqrt((st.back_nn_y[topk_idx] - st.kpt_y.unsqueeze(1)) ** 2 +
                       (st.back_nn_x[topk_idx] - st.kpt_x.unsqueeze(1)) ** 2)
    u = topk_sim - LAM * d_cyc / nP
    if ZROW > 0:   # pool-adaptive penalties: scale all penalties by this row's similarity spread (std of the top-K)
        sd = topk_sim.std(1, keepdim=True)
        pen_scale = (sd / sd.median().clamp(min=1e-6)).clamp(0.25, 4.0) ** ZROW
    else:
        pen_scale = 1.0
    u = topk_sim - pen_scale * LAM * d_cyc / nP
    score = u
    if WFL > 0 or LFL > 0 or REF_ENS > 0:
        fe, fl_map = flip_ensemble(pair, a, st, pid, d_cyc, topk_sim)
        score = score + fe * (pen_scale if WFL == 0 else 1.0)
    if DENSE_MU > 0:   # T2 dense flow: local median displacement of cycle-consistent dense Deja Vu matches
        score = score - DENSE_MU * dense_flow_resid(st, qxy, cxy).clamp(max=CAP) / nP
    if HAND2 > 0:   # mirror-pair handedness direction learned from 4 source views (2 per side, raw + mirrored)
        vi = st.valid_kpt_indices
        Mk = vi.numel()
        lo2, ok2 = other_arm_kp(pair, a, st)
        D_a, D_o = A['desc'], pair['src'][1 - a]['desc']
        pos = torch.full((pair['perm'].numel(),), -1, dtype=torch.long, device=st.device)
        pos[vi] = torch.arange(Mk, device=st.device)
        pv = pair['perm'][vi]
        pj = torch.where(pv >= 0, pos[pv.clamp(min=0)], torch.full_like(pv, -1))
        vk = A['kp_valid'][vi]
        use = (pj >= 0) & (pj != torch.arange(Mk, device=st.device)) & vk & vk[pj.clamp(min=0)] & ok2 & ok2[pj.clamp(min=0)]
        if bool(use.any()):
            ci = D_a[st.idx_valid] + D_o[lo2]                                   # class i (raw i + mirrored partner view)
            cj = ci[pj.clamp(min=0)]                                             # class j
            w = ci - cj
            w = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            tc = pair['trg']['desc'][pid.reshape(-1)].view(*pid.shape, -1)
            h = (tc * w.unsqueeze(1)).sum(-1)                                    # [M,K]
            score = score + HAND2 * torch.where(use.unsqueeze(1), h, torch.zeros_like(h))
    if PROF > 0:   # relational profile: candidate's similarities to the OTHER queries vs the query's self-similarities
        vi = st.valid_kpt_indices
        Mk = vi.numel()
        valid_k = A['kp_valid'][vi]
        if int(valid_k.sum()) >= 3:
            Rj = st.sim_kpts[:, topk_idx.reshape(-1)].view(Mk, Mk, -1)          # [j, i, k]
            qd = A['desc'][st.idx_valid]
            Ss = qd @ qd.t()                                                    # [i, j]
            w = (valid_k.view(-1, 1) & valid_k.view(1, -1) & ~torch.eye(Mk, dtype=torch.bool, device=st.device)).float()  # [i, j]
            T = Rj.permute(1, 2, 0)                                             # [i, k, j]
            wn = w.sum(1).clamp(min=1).view(Mk, 1, 1)
            Tm = (T * w.unsqueeze(1)).sum(-1, keepdim=True) / wn
            Sm = (Ss * w).sum(-1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)
            Tc, Sc = (T - Tm) * w.unsqueeze(1), ((Ss - Sm) * w).unsqueeze(1)
            corr = (Tc * Sc).sum(-1) / (Tc.norm(dim=-1) * Sc.norm(dim=-1)).clamp(min=1e-6)
            score = score + PROF * corr
    if RR > 0:
        S = st.sim_valid
        cols = S[:, topk_idx.reshape(-1)].view(S.shape[0], *topk_idx.shape)
        rr = (cols > torch.gather(own_rows, 1, topk_idx).unsqueeze(0)).sum(0).float()
        score = score - pen_scale * RR * torch.log1p(rr)
        if RR_FL > 0:   # reverse rank of the other-arm query among the other arm's (masked) source patches
            lo, ok = other_arm_kp(pair, a, st)
            O = pair['src'][1 - a]
            om = O['mask'].clone()
            om[lo[ok]] = True
            ov = torch.nonzero(om, as_tuple=True)[0]
            tc = pair['trg']['desc'][pid.reshape(-1)]
            colsf = (O['desc'][ov] @ tc.t()).view(ov.numel(), *pid.shape)
            sq = (O['desc'][lo].unsqueeze(1) * tc.view(*pid.shape, -1)).sum(-1)
            rf = (colsf > sq.unsqueeze(0)).sum(0).float()
            score = score - RR_FL * torch.where(ok.unsqueeze(1), torch.log1p(rf), torch.log1p(rr))
    if MU > 0:
        valid = A['kp_valid'][st.valid_kpt_indices]
        g = torch.Generator(device=st.device)
        g.manual_seed(0)
        r, info = consensus(qxy, cxy, u, valid, model=MODEL, thr=THR, gen=g)
        s0 = score
        mu = MU
        if MU_ADAPT > 0 and info.get('ok'):
            mu = MU * min(1.0, (info['n_inl'] / max(info['n'], 1)) / MU_ADAPT)   # trust geometry when the fit is good
        score = s0 - pen_scale * mu * r.clamp(max=CAP) / nP
        for _ in range(CONS_ITER):   # re-estimate the consensus with the full score as unary
            g.manual_seed(0)
            r, _ = consensus(qxy, cxy, score.masked_fill(~in_pool, -1e9), valid, model=MODEL, thr=THR, gen=g)
            score = s0 - MU * r.clamp(max=CAP) / nP
    score = score.masked_fill(~in_pool, -1e9)
    best = score.argmax(1)
    if DECOL_R > 0:
        best = decollide(pair, a, st, score, pid, best)
    bp = pid.gather(1, best[:, None])[:, 0]
    if _STATS is not None:   # analysis hook: pair-level evidence for this arm (source-side info only)
        vk = A['kp_valid'][st.valid_kpt_indices]
        rr0 = torch.arange(best.numel(), device=st.device)
        sc = score[rr0, best][vk]
        dcb = d_cyc[rr0, best][vk]
        _STATS.append(dict(arm=a, n=int(vk.sum()), score=float(sc.mean()) if sc.numel() else 0.0,
                           sim=float(topk_sim[rr0, best][vk].mean()) if sc.numel() else 0.0,
                           mnn=float((dcb <= 1).float().mean()) if sc.numel() else 0.0,
                           dcyc=float(dcb.mean()) if sc.numel() else 0.0,
                           cons=float((info['n_inl'] / max(info['n'], 1)) if (MU > 0 and info.get('ok')) else 0.0)))
    s2d = st.full_sim_2d
    if REF_ENS > 0:
        s2d = (s2d + REF_ENS * fl_map) / (1 + REF_ENS)
    if (REF_ORIG or SEL_ORIG) and '_orig' in pair:   # ablation: refinement maps from the non-adapted descriptors
        po = pair['_orig']
        T0 = po['trg']['desc']
        own = (po['src'][a]['desc'][st.idx_valid] @ T0.t()).masked_fill(~st.tgt_mask.unsqueeze(0), -float('inf'))
        lo_, ok_ = other_arm_kp(pair, a, st)
        flo = (po['src'][1 - a]['desc'][lo_] @ T0.t()).masked_fill(~st.tgt_mask.unsqueeze(0), -float('inf'))
        flo = torch.where(ok_.unsqueeze(1), flo, own)
        s2d = ((own + REF_ENS * flo) / (1 + REF_ENS)).view(-1, nP, nP)
    if REF_CYC > 0:   # refine on a score map: similarity minus the cycle penalty of every target patch
        bxy = torch.stack([st.back_nn_x, st.back_nn_y], -1)                       # [Pt,2]
        dcm = (bxy.unsqueeze(0) - qxy.unsqueeze(1)).norm(dim=-1)                   # [M,Pt]
        pen = torch.zeros(s2d.shape[0], nP * nP, device=st.device)
        pen[:, st.tgt_valid_idx] = dcm
        s2d = s2d - REF_CYC * pen.view(-1, nP, nP) / nP
    if REF_CONS > 0 and MU > 0 and info.get('ok'):   # also penalise distance to the consensus prediction in the map
        from decode_lib.consensus import _apply
        pq = _apply(info['B'], qxy)                                                   # [M,2]
        yy = torch.arange(nP, device=st.device).view(1, nP, 1).float()
        xx = torch.arange(nP, device=st.device).view(1, 1, nP).float()
        rm = ((xx - pq[:, 0].view(-1, 1, 1)) ** 2 + (yy - pq[:, 1].view(-1, 1, 1)) ** 2).sqrt()
        s2d = s2d - REF_CONS * rm.clamp(max=CAP) / nP
    if REF_SIG > 0:   # Gaussian prior around the chosen patch instead of a hard window
        yy = torch.arange(nP, device=st.device).view(1, nP, 1).float()
        xx = torch.arange(nP, device=st.device).view(1, 1, nP).float()
        d2 = (yy - (bp // nP).view(-1, 1, 1).float()) ** 2 + (xx - (bp % nP).view(-1, 1, 1).float()) ** 2
        s2d = s2d - BETA * d2 / (2 * REF_SIG ** 2)
    if REF_UP > 1:   # bilinear up-sampled map, soft-argmax on the finer grid
        u = REF_UP
        m2 = s2d.clamp(min=-2.0).unsqueeze(1)
        mf = torch.nn.functional.interpolate(m2, scale_factor=u, mode='bilinear', align_corners=False)[:, 0]
        tm = st.tgt_mask.view(1, 1, nP, nP).float()
        tmf = torch.nn.functional.interpolate(tm, scale_factor=u, mode='nearest')[0, 0] > 0
        mf = mf.masked_fill(~tmf.unsqueeze(0), -float('inf'))
        nx, ny = C.window_soft_argmax_2d(mf, (bp // nP) * u + u // 2, (bp % nP) * u + u // 2, WIN * u + u // 2, BETA,
                                         st.anno_stride / u)
    else:
        nx, ny = C.window_soft_argmax_2d(s2d, bp // nP, bp % nP, WIN, BETA, st.anno_stride)
    if REF_MODE != 'soft':
        qx, qy = quad_refine(s2d, bp // nP, bp % nP, st.anno_stride)
        if REF_MODE == 'quad':
            nx, ny = qx, qy
        else:
            nx, ny = (nx + qx) / 2, (ny + qy) / 2
    p = torch.stack([nx, ny], 1)
    if SWAP_T > -1.0:
        p = mirror_swap(pair, a, st, qxy, p)
    out[st.valid_kpt_indices] = p
    return out


def dense_flow_resid(st, qxy, cxy, kd=10):
    """Decode every masked source patch with Deja Vu (top-kd, lambda LAM), keep cycle-consistent ones (d_cyc <= 1) as
    anchors, predict each query's target as q + median displacement of anchors within DENSE_R (excluding the query's own
    patch); return |c - p| [M, K] (0 when < 3 anchors)."""
    nP = st.num_patches
    S = st.sim_valid                                                     # [Ps, Pt]
    v, ti = S.topk(min(kd, S.shape[1]), dim=1)
    sxy = geo.pid_xy(st.src_valid_idx, nP)                               # [Ps,2]
    bxy = torch.stack([st.back_nn_x, st.back_nn_y], -1)                 # [Pt,2] (back-NN of every valid target)
    dc = (bxy[ti] - sxy.unsqueeze(1)).norm(dim=-1)                       # [Ps,kd]
    best = (v - LAM * dc / nP).argmax(1)
    t_best = ti.gather(1, best[:, None])[:, 0]
    good = dc.gather(1, best[:, None])[:, 0] <= 1.0
    disp = geo.pid_xy(st.tgt_valid_idx[t_best], nP) - sxy                 # [Ps,2]
    D = (sxy.unsqueeze(0) - qxy.unsqueeze(1)).norm(dim=-1)               # [M,Ps]
    use = (D <= DENSE_R) & (D >= 1.0) & good.unsqueeze(0)
    n = use.sum(1)
    big = torch.full_like(D, float('nan'))
    dx = torch.where(use, disp[:, 0].unsqueeze(0).expand_as(D), big)
    dy = torch.where(use, disp[:, 1].unsqueeze(0).expand_as(D), big)
    med = torch.stack([dx.nanmedian(1)[0], dy.nanmedian(1)[0]], -1)      # [M,2]
    p = qxy + torch.nan_to_num(med)
    r = (cxy - p.unsqueeze(1)).norm(dim=-1)
    return torch.where((n >= 3).unsqueeze(1), r, torch.zeros_like(r))


def quad_refine(s2d, by, bx, stride):
    """Local maximum of the map in the 3x3 around (by, bx), then a separable parabola fit for a sub-patch offset."""
    M, nP = s2d.shape[0], s2d.shape[1]
    m = s2d.clamp(min=-1.0)
    r = torch.arange(M, device=m.device)
    best_v = torch.full((M,), -9.0, device=m.device)
    py, px = by.clone(), bx.clone()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = (by + dy).clamp(0, nP - 1), (bx + dx).clamp(0, nP - 1)
            v = m[r, yy, xx]
            upd = v > best_v
            best_v = torch.where(upd, v, best_v)
            py, px = torch.where(upd, yy, py), torch.where(upd, xx, px)
    def off(a, b, c):
        den = a - 2 * b + c
        o = torch.where(den < -1e-6, 0.5 * (a - c) / den, torch.zeros_like(den))
        return o.clamp(-0.5, 0.5)
    c0 = m[r, py, px]
    ox = off(m[r, py, (px - 1).clamp(0, nP - 1)], c0, m[r, py, (px + 1).clamp(0, nP - 1)])
    oy = off(m[r, (py - 1).clamp(0, nP - 1), px], c0, m[r, (py + 1).clamp(0, nP - 1), px])
    return (px.float() + ox) * stride + stride / 2.0, (py.float() + oy) * stride + stride / 2.0


def mirror_swap(pair, a, st, qxy, p):
    """Fit a similarity q -> p (patch units) by 2-point RANSAC on this arm's predictions; for every mirror pair (i, perm i)
    decoded in this arm, swap their predictions when the target offset points against the transformed source offset."""
    from decode_lib.consensus import _fit_sim, _apply
    vi = st.valid_kpt_indices
    M = vi.numel()
    use = pair['src'][a]['kp_valid'][vi] & (pair['arm_of_kpt'][vi] == a)
    idx = torch.nonzero(use, as_tuple=True)[0]
    if idx.numel() < 3:
        return p
    pp = p / st.anno_stride
    q, t = qxy[idx], pp[idx]
    n = idx.numel()
    ii, jj = torch.triu_indices(n, n, 1, device=p.device)
    B = _fit_sim(torch.stack([q[ii], q[jj]], 1), torch.stack([t[ii], t[jj]], 1))      # [H,3,2]
    res = (_apply(B, q.unsqueeze(0).expand(B.shape[0], -1, -1)) - t.unsqueeze(0)).norm(dim=-1)
    s = B[:, 0, :].norm(dim=-1)
    sup = ((res < 4.0).float().sum(1)).masked_fill((s < 0.2) | (s > 5), -1)
    R = B[sup.argmax(), :2, :]                                                           # row-vector linear part
    pos = torch.full((pair['perm'].numel(),), -1, dtype=torch.long, device=p.device)
    pos[vi] = torch.arange(M, device=p.device)
    out = p.clone()
    for k in idx.tolist():
        i = int(vi[k]); j = int(pair['perm'][i])
        if j <= i:
            continue
        kj = int(pos[j])
        if kj < 0 or not bool(use[kj]):
            continue
        vs = (qxy[kj] - qxy[k]) @ R
        vt = pp[kj] - pp[k]
        if vs.norm() < 1.0 or vt.norm() < 1.0:
            continue
        cos = float((vs * vt).sum() / vs.norm() / vt.norm())
        if cos < -SWAP_T:
            out[k], out[kj] = p[kj], p[k]
    return out


def decollide(pair, a, st, score, pid, best):
    """Mirror-pair de-collision: if a keypoint and its mirror partner (both decoded in this arm) chose candidates within
    DECOL_R patches, the one with the higher score keeps it and the other takes its best candidate farther than
    DECOL_EX patches from the kept location (if any)."""
    nP = st.num_patches
    vi = st.valid_kpt_indices
    M = vi.numel()
    use = pair['src'][a]['kp_valid'][vi] & (pair['arm_of_kpt'][vi] == a)
    pos = torch.full((pair['perm'].numel(),), -1, dtype=torch.long, device=st.device)
    pos[vi] = torch.arange(M, device=st.device)
    cxy = torch.stack([(pid % nP).float(), (pid // nP).float()], -1)            # [M,K,2]
    r = torch.arange(M, device=st.device)
    best = best.clone()
    bs = score[r, best]
    for k in torch.nonzero(use, as_tuple=True)[0].tolist():
        i = int(vi[k]); j = int(pair['perm'][i])
        if j <= i:
            continue
        kj = int(pos[j])
        if kj < 0 or not bool(use[kj]):
            continue
        pk, pj = cxy[k, best[k]], cxy[kj, best[kj]]
        if float((pk - pj).norm()) > DECOL_R:
            continue
        keep, move = (k, kj) if float(bs[k]) >= float(bs[kj]) else (kj, k)
        kl = cxy[keep, best[keep]]
        far = (cxy[move] - kl).norm(dim=-1) > DECOL_EX
        alt = score[move].masked_fill(~far, -1e9)
        if float(alt.max()) > -1e8 and float(score[move, best[move]] - alt.max()) <= DECOL_MARGIN:
            best[move] = alt.argmax()
    return best


def peak_pool(st, kp, r, ktop):
    """Candidate pool = the kp highest local maxima ((2r+1)^2 NMS on the masked 60x60 row map) of the forward row,
    plus the ktop highest patches (deduplicated, padded with the next-best patches). Returns valid-target indices [M, kp+ktop]."""
    nP = st.num_patches
    M = st.sim_kpts.shape[0]
    full = torch.full((M, nP * nP), -10.0, device=st.device)
    full[:, st.tgt_valid_idx] = st.sim_kpts
    mp = torch.nn.functional.max_pool2d(full.view(M, 1, nP, nP), 2 * r + 1, 1, r).view(M, -1)
    is_peak = (full >= mp) & (full > -9)
    pv = full.masked_fill(~is_peak, -10.0)
    n = kp + ktop
    if ktop > 0:
        tv = full.topk(ktop, dim=1)[1]
        pv = pv.scatter(1, tv, 5.0 + full.gather(1, tv))          # force the top patches in (ranked first)
    sel = pv.topk(n, dim=1)[1]                                   # [M, n] full pids
    sel = torch.where(pv.gather(1, sel) > -9, sel, full.topk(n, dim=1)[1])   # pad with best patches if few peaks
    inv = torch.full((nP * nP,), 0, dtype=torch.long, device=st.device)
    inv[st.tgt_valid_idx] = torch.arange(st.tgt_valid_idx.numel(), device=st.device)
    return inv[sel]


def smooth_query(desc, idx, nP, w=1.0, r=1):
    """l2-normalised desc[idx] + w * sum of its in-grid neighbours ((2r+1)^2 box)."""
    y, x = idx // nP, idx % nP
    q = desc[idx].clone()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = (y + dy).clamp(0, nP - 1), (x + dx).clamp(0, nP - 1)
            q = q + w * desc[yy * nP + xx]
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def other_arm_kp(pair, a, st):
    nP = st.num_patches
    vi = st.valid_kpt_indices
    k0, v0 = pair['src'][0]['kp'], pair['src'][0]['kp_valid']
    if a == 1:
        lo, ok = k0[vi], v0[vi]
    else:
        j = pair['perm'][vi]                      # -1 = mirror partner unknown (datasets without a flip permutation)
        jc = j.clamp(min=0)
        lo, ok = (k0[jc] // nP) * nP + (nP - 1 - k0[jc] % nP), v0[jc] & (j >= 0)
    if PARTNER_EST > 0:   # estimate the other-arm view where the mirror partner is unknown (or always, if forced)
        need = ~ok if not PARTNER_FORCE else torch.ones_like(ok)
        if bool(need.any()):
            O = pair['src'][1 - a]
            q = pair['src'][a]['desc'][st.idx_valid]                        # [M,D]
            om = torch.nonzero(O['mask'], as_tuple=True)[0]
            sim = q @ O['desc'][om].t()                                     # [M, Po]
            own = st.idx_valid
            mir_self = (own // nP) * nP + (nP - 1 - own % nP)               # same physical part, mirrored
            oxy = torch.stack([(om % nP).float(), (om // nP).float()], -1)
            mxy = torch.stack([(mir_self % nP).float(), (mir_self // nP).float()], -1)
            near = (oxy.unsqueeze(0) - mxy.unsqueeze(1)).norm(dim=-1) <= PARTNER_EST
            best_far = sim.masked_fill(near, -9).max(1)
            lo_est = om[best_far[1]]
            okk = best_far[0] > -9
            if a == 1:   # in arm 1 the own query is already the mirrored view; analogous search in arm 0
                pass
            lo = torch.where(need & okk, lo_est, lo)
            ok = ok | (need & okk)
    if FL_ALIGN > 0:   # snap the other-arm location to the neighbour most similar to the own query descriptor
        q = pair['src'][a]['desc'][st.idx_valid]
        O = pair['src'][1 - a]['desc']
        y, x = lo // nP, lo % nP
        best, bl = torch.full_like(lo, -9, dtype=q.dtype), lo.clone()
        for dy in range(-FL_ALIGN, FL_ALIGN + 1):
            for dx in range(-FL_ALIGN, FL_ALIGN + 1):
                l2 = (y + dy).clamp(0, nP - 1) * nP + (x + dx).clamp(0, nP - 1)
                v = (O[l2] * q).sum(-1)
                upd = v > best
                best, bl = torch.where(upd, v, best), torch.where(upd, l2, bl)
        lo = bl
    return lo, ok


def flip_ensemble(pair, a, st, pid, d_cyc, topk_sim):
    """The same semantic keypoint seen in the other arm: for arm 1 -> arm-0 kp[i]; for arm 0 -> the mirrored column of
    arm-0 kp[perm[i]] (the counterpart part, mirrored, looks like part i). Adds WFL * S_other(l', m) and
    -LFL * d_cyc_other / w (back-NN of the candidate in the other arm vs l'). Falls back to the own-arm values when the
    counterpart is not annotated."""
    nP = st.num_patches
    lo, ok = other_arm_kp(pair, a, st)
    O = pair['src'][1 - a]
    tc = pair['trg']['desc'][pid.reshape(-1)]
    sfl = (O['desc'][lo].unsqueeze(1) * tc.view(*pid.shape, -1)).sum(-1)
    omask = O['mask'].clone()
    omask[lo[ok]] = True
    ov = torch.nonzero(omask, as_tuple=True)[0]
    bn = ov[(O['desc'][ov] @ tc.t()).argmax(0)].view(*pid.shape)
    dfl = (geo.pid_xy(bn, nP) - geo.pid_xy(lo, nP).unsqueeze(1)).norm(dim=-1)
    okk = ok.unsqueeze(1).expand_as(sfl)
    sfl = torch.where(okk, sfl, topk_sim)
    dfl = torch.where(okk, dfl, d_cyc)
    fl_map = None
    if REF_ENS > 0:   # other-arm similarity map of the semantic keypoint over all target patches (own map where not ok)
        fm = (O['desc'][lo] @ pair['trg']['desc'].t()).masked_fill(~pair['trg']['mask'].unsqueeze(0), -float('inf'))
        fl_map = torch.where(ok.view(-1, 1, 1), fm.view(-1, nP, nP), st.full_sim_2d)
    return WFL * sfl - LFL * dfl / nP, fl_map


T3_MODEL = 'runs/t3_models/x2tta_full_60_128.pt'   # '' = training-free (seq 5); else path of a decode_lib.t3 model (track T3, trained on SPair trn_train)
_T3 = {}
T3_REF_CYC = 0.0  # cycle-penalised refinement map for the T3 path


def decode_arm_t3(pair, a):
    from decode_lib import feats as FT, t3
    if T3_MODEL not in _T3:
        _T3[T3_MODEL] = t3.load_model(T3_MODEL)
    N = pair['src'][a]['kp'].numel()
    out = torch.zeros(N, 2, device='cuda')
    nf = _T3[T3_MODEL].mu.numel()
    st, o = FT.arm_candidates(pair, a, extra={len(FT.FEAT_NAMES): 0, len(FT.FEAT_NAMES2): 1, len(FT.FEAT_NAMES3): 2}[nf])
    if o is None:
        return out
    s = _T3[T3_MODEL](o['F'], o['in_pool'])
    if T3_REF_CYC > 0:
        nP = st.num_patches
        qxy = torch.stack([st.kpt_x, st.kpt_y], -1)
        bxy = torch.stack([st.back_nn_x, st.back_nn_y], -1)
        pen = torch.zeros(qxy.shape[0], nP * nP, device=st.device)
        pen[:, st.tgt_valid_idx] = (bxy.unsqueeze(0) - qxy.unsqueeze(1)).norm(dim=-1)
        o = dict(o, map=o['map'] - T3_REF_CYC * pen.view(-1, nP, nP) / nP)
    nx, ny = FT.refine(st, o, s.argmax(1))
    out[st.valid_kpt_indices] = torch.stack([nx, ny], 1)
    return out


SD_W = 1.0   # TLR-style descriptors only: reweight the SD block vs the DINO block (S = (w S_sd + (2-w) S_dino)/2)


def reweight(pair, w):
    def f(d):
        if d.shape[1] != 3968:
            return d
        return torch.cat([d[:, :3200] * (w ** 0.5), d[:, 3200:] * ((2 - w) ** 0.5)], 1)
    src = [dict(A, desc=f(A['desc'])) for A in pair['src']]
    return dict(pair, src=src, trg=dict(pair['trg'], desc=f(pair['trg']['desc'])))


_STATS = None    # analysis hook (list) — set by scripts/arm_stats.py
ARM_PAIR = True  # seq 11: pair-level arm for every keypoint (instead of the per-keypoint vis_flip rule)
FORCE_ARM = -1   # analysis: 0 / 1 decode every keypoint in that arm (arm-1 keypoints rebuilt from source info only)


def force_arm(pair, a):
    N = pair['arm_of_kpt'].numel()
    if a == 0:
        return dict(pair, arm_of_kpt=torch.zeros(N, dtype=torch.long, device='cuda'))
    nP = pair['nP']
    k0, v0 = pair['src'][0]['kp'], pair['src'][0]['kp_valid']
    j = pair['perm']
    jc = j.clamp(min=0)
    ok = v0[jc] & (j >= 0)
    kp1 = torch.where(ok, (k0[jc] // nP) * nP + (nP - 1 - k0[jc] % nP), torch.zeros_like(k0))
    src1 = dict(pair['src'][1], kp=kp1, kp_valid=ok)
    arm = torch.where(ok, torch.ones(N, dtype=torch.long, device='cuda'), torch.zeros(N, dtype=torch.long, device='cuda'))
    return dict(pair, src=[pair['src'][0], src1], arm_of_kpt=arm)


def decode(pair):
    if SD_W != 1.0:
        pair = reweight(pair, SD_W)
    if TTA_PRIOR and TTA_STEPS == 0:   # fixed channel weights averaged over unlabeled trn pairs
        import numpy as _np
        import torch.nn.functional as _F
        if TTA_PRIOR not in _T3:
            _T3[TTA_PRIOR] = torch.from_numpy(_np.load(TTA_PRIOR)).float().cuda()
        wp = _F.softplus(_T3[TTA_PRIOR] + 0.5413)
        f = lambda d: _F.normalize(d * wp, dim=1)
        pair = dict(pair, src=[dict(S_, desc=f(S_['desc'])) for S_ in pair['src']], trg=dict(pair['trg'], desc=f(pair['trg']['desc'])))
    if TTA_STEPS > 0:
        from decode_lib.tta import adapt_pair
        init = None
        if TTA_PRIOR:
            import numpy as _np
            if TTA_PRIOR not in _T3:
                _T3[TTA_PRIOR] = torch.from_numpy(_np.load(TTA_PRIOR)).float().cuda()
            init = _T3[TTA_PRIOR]
        orig = pair
        if REF_ORIG or SEL_ORIG:
            orig_keep = pair
        with torch.enable_grad():
            ta = (1 if bool((pair['arm_of_kpt'] == 1).any()) else 0) if TTA_ARM < 0 else TTA_ARM
            pair = adapt_pair(pair, steps=TTA_STEPS, reg=TTA_REG, lr=TTA_LR, tau=TTA_TAU, a=ta, both=TTA_BOTH, obj=TTA_OBJ, rank=TTA_RANK, groups=TTA_GROUPS, sep=TTA_SEP, hand=TTA_HAND, qw=TTA_QW, init=init, sem=TTA_SEM, blocknorm=TTA_BLOCKNORM, nomask=TTA_NOMASK, sub=TTA_SUB)
        if TTA_ENS_OBJ:   # ensemble with a second TTA solution (different objective): concatenated half-weighted descs
            with torch.enable_grad():
                p2 = adapt_pair(orig, steps=TTA_STEPS, reg=TTA_REG, lr=TTA_LR, tau=TTA_TAU, a=ta, obj=TTA_ENS_OBJ)
            cat2 = lambda d0, d1: torch.cat([d0 * 0.5 ** 0.5, d1 * 0.5 ** 0.5], 1)
            pair = dict(pair, src=[dict(S1, desc=cat2(S1['desc'], S2['desc'])) for S1, S2 in zip(pair['src'], p2['src'])],
                        trg=dict(pair['trg'], desc=cat2(pair['trg']['desc'], p2['trg']['desc'])))
        if TTA_MIX < 1.0:   # similarity = (1-m) S_orig + m S_tta via concatenated scaled descriptors
            m = TTA_MIX
            cat = lambda d0, d1: torch.cat([d0 * (1 - m) ** 0.5, d1 * m ** 0.5], 1)
            pair = dict(pair, src=[dict(S1, desc=cat(S0['desc'], S1['desc'])) for S0, S1 in zip(orig['src'], pair['src'])],
                        trg=dict(pair['trg'], desc=cat(orig['trg']['desc'], pair['trg']['desc'])))
    if (REF_ORIG or SEL_ORIG) and TTA_STEPS > 0:
        if SEL_ORIG:   # ablation: select on the original descriptors, refine on the adapted ones
            pair = dict(orig_keep, _orig=pair)
        else:
            pair = dict(pair, _orig=orig_keep)
    if FORCE_ARM >= 0:
        pair = force_arm(pair, FORCE_ARM)
    elif ARM_PAIR and not T3_MODEL:   # (training-free path only) decode ALL keypoints in the arm chosen for the pair (arm-1 queries rebuilt from source info)
        pair = force_arm(pair, 1 if bool((pair['arm_of_kpt'] == 1).any()) else 0)
    if SELF_TRAIN > 0 and TTA_STEPS > 0 and not T3_MODEL:
        pred = _decode_after(pair)
        # pseudo-labels: arm-0 query patches of keypoints decoded in arm 0 with a valid source location
        A0 = orig['src'][0]
        use = A0['kp_valid'] & (pair['arm_of_kpt'] == 0)
        if int(use.sum()) >= 2:
            nP = pair['nP']
            st_ = pair['anno'] / nP
            tp = (pred[use, 1] / st_).long().clamp(0, nP - 1) * nP + (pred[use, 0] / st_).long().clamp(0, nP - 1)
            with torch.enable_grad():
                from decode_lib import tta as _tta
                p2 = _tta.adapt_pair(orig, steps=TTA_STEPS, reg=TTA_REG, lr=TTA_LR, tau=TTA_TAU, a=0,
                                     pseudo=(A0['kp'][use], tp), pw=SELF_TRAIN)
            p2 = dict(p2, arm_of_kpt=pair['arm_of_kpt'],
                      src=[dict(S2, kp=S1['kp'], kp_valid=S1['kp_valid']) for S1, S2 in zip(pair['src'], p2['src'])])
            return _decode_after(p2)
        return pred
    return _decode_after(pair)


def _decode_after(pair):
    arm = pair['arm_of_kpt']
    f = decode_arm_t3 if T3_MODEL else decode_arm
    pred = f(pair, 0)
    if bool((arm == 1).any()):
        p1 = f(pair, 1)
        pred[arm == 1] = p1[arm == 1]
    return pred


if __name__ == '__main__':   # python decode.py --train <out.pt> [epochs]: T3 training on trn_train tables (<10 min)
    import sys
    if sys.argv[1] == '--train':
        from decode_lib import t3
        t3.train('cache/tlr/trn_train/t3_n600.npz', sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60,
                 'cache/tlr/dev/t3.npz')
