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
WFL, LFL = 0.5, 0.25          # flip-ensemble: weight of the other-arm similarity and of its cycle distance
K = 30
WIN, BETA, REF_ENS = 5, 0.01, 0.5   # refinement: window, softmax beta, weight of the other-arm map in the refinement


def decode_arm(pair, a):
    A = pair['src'][a]
    st = C.PairState(A['desc'], pair['trg']['desc'], A['kp'], A['mask'], pair['trg']['mask'])
    N = A['kp'].numel()
    out = torch.zeros(N, 2, device='cuda')
    if st.M == 0:
        return out
    nP = st.num_patches
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
    score = u
    if WFL > 0 or LFL > 0:
        fe, fl_map = flip_ensemble(pair, a, st, pid, d_cyc, topk_sim)
        score = score + fe
    if MU > 0:
        valid = A['kp_valid'][st.valid_kpt_indices]
        g = torch.Generator(device=st.device)
        g.manual_seed(0)
        r, _ = consensus(qxy, cxy, u, valid, model=MODEL, thr=THR, gen=g)
        score = score - MU * r.clamp(max=CAP) / nP
    score = score.masked_fill(~in_pool, -1e9)
    best = score.argmax(1)
    bp = pid.gather(1, best[:, None])[:, 0]
    s2d = st.full_sim_2d
    if REF_ENS > 0:
        s2d = (s2d + REF_ENS * fl_map) / (1 + REF_ENS)
    nx, ny = C.window_soft_argmax_2d(s2d, bp // nP, bp % nP, WIN, BETA, st.anno_stride)
    out[st.valid_kpt_indices] = torch.stack([nx, ny], 1)
    return out


def flip_ensemble(pair, a, st, pid, d_cyc, topk_sim):
    """The same semantic keypoint seen in the other arm: for arm 1 -> arm-0 kp[i]; for arm 0 -> the mirrored column of
    arm-0 kp[perm[i]] (the counterpart part, mirrored, looks like part i). Adds WFL * S_other(l', m) and
    -LFL * d_cyc_other / w (back-NN of the candidate in the other arm vs l'). Falls back to the own-arm values when the
    counterpart is not annotated."""
    nP = st.num_patches
    vi = st.valid_kpt_indices
    k0, v0 = pair['src'][0]['kp'], pair['src'][0]['kp_valid']
    if a == 1:
        lo, ok = k0[vi], v0[vi]
    else:
        j = pair['perm'][vi]
        lo = (k0[j] // nP) * nP + (nP - 1 - k0[j] % nP)
        ok = v0[j]
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


def decode(pair):
    arm = pair['arm_of_kpt']
    pred = decode_arm(pair, 0)
    if bool((arm == 1).any()):
        p1 = decode_arm(pair, 1)
        pred[arm == 1] = p1[arm == 1]
    return pred
