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
Baseline = Deja Vu (lambda 0.5): top-p nomination + graded cycle re-ranking + window soft-argmax.
"""
import torch
from vendor import core as C

LAM = 0.5


def decode_arm(pair, a):
    A = pair['src'][a]
    st = C.PairState(A['desc'], pair['trg']['desc'], A['kp'], A['mask'], pair['trg']['mask'])
    N = A['kp'].numel()
    out = torch.zeros(N, 2, device='cuda')
    if st.M == 0:
        return out
    by, bx = C.select_graded(st, LAM)
    nx, ny = C.window_soft_argmax(st, by, bx, 5, 0.01)
    out[st.valid_kpt_indices] = torch.stack([nx, ny], 1)
    return out


def decode(pair):
    arm = pair['arm_of_kpt']
    pred = decode_arm(pair, 0)
    if bool((arm == 1).any()):
        p1 = decode_arm(pair, 1)
        pred[arm == 1] = p1[arm == 1]
    return pred
