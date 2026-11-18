"""T3 over candidates from BOTH source arms: per keypoint, 30 candidates decoded in arm 0 + 30 in arm 1 (arm-1 queries
rebuilt from the source mirror partner, as ARM_PAIR), with 2 extra features (candidate arm, pair-level adaptive-flip
arm). Returns per keypoint index: F [Nv, 60, nf+2], pool, and per-candidate refinement info."""
import torch
from decode_lib import feats as FT


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
    return dict(pair, src=[pair['src'][0], src1], arm_of_kpt=torch.ones(N, dtype=torch.long, device='cuda'))


def two_arm_candidates(pair, extra=2):
    N = pair['arm_of_kpt'].numel()
    base_arm = float(bool((pair['arm_of_kpt'] == 1).any()))
    per = []
    for a in (0, 1):
        pa = force_arm(pair, a)
        st, o = FT.arm_candidates(pa, a, extra=extra)
        per.append((st, o))
    nf = None
    for st, o in per:
        if o is not None:
            nf = o['F'].shape[-1]
            K = o['F'].shape[1]
    if nf is None:
        return None
    dev = 'cuda'
    F = torch.zeros(N, 2 * K, nf + 2, device=dev)
    pool = torch.zeros(N, 2 * K, dtype=torch.bool, device=dev)
    has = torch.zeros(N, dtype=torch.bool, device=dev)
    for a, (st, o) in enumerate(per):
        if o is None:
            continue
        vi = st.valid_kpt_indices
        val = pair['src'][a]['kp_valid'][vi] if a == 0 else force_arm(pair, 1)['src'][1]['kp_valid'][vi]
        sl = slice(a * K, a * K + o['F'].shape[1])
        Fa = torch.cat([o['F'], torch.full_like(o['F'][..., :1], float(a)),
                        torch.full_like(o['F'][..., :1], base_arm)], -1)
        F[vi, sl] = Fa
        pool[vi, sl] = o['in_pool'] & val.unsqueeze(1)
        has[vi] |= val
    return dict(F=F, pool=pool, has=has, per=per, K=K)
