"""Generalisation splits (program.md §3), built WITHOUT editing the frozen prepare.py; evaluate with prepare.evaluate.

  python prepare_gen.py spairu <method>   -> cache/<method>/spairu_test/<cat>.npz  (SPair-U unseen keypoints, test pairs)
    SPair images -> reuses cache/<method>/feat. Base protocol of the review runs: no adaptive flip; the unseen keypoints
    have no mirror permutation -> perm = -1 (decoders fall back to the own arm).
"""
import os
import sys
import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prepare as P                                       # noqa: E402
from vendor import core as C                              # noqa: E402
from vendor.geoware import flip_keypoints                 # noqa: E402


def build_spairu(method, cats=None):
    split = 'spairu_test'
    for cat in cats or P.CATS:
        out_path = f'{P.cdir(method, split)}/{cat}.npz'
        if os.path.exists(out_path):
            continue
        names = sorted(f for f in os.listdir(f'{P.SPAIR}/JPEGImages/{cat}') if f.endswith('.jpg'))
        nidx = {n: i for i, n in enumerate(names)}
        kps, thr, used, pnames = C.load_spair_pairs(P.SPAIR, cat, 'test', unseen_kps=True)
        nk = kps.shape[1]
        out = {k: [] for k in ('src', 'trg', 'k1', 'k2', 'k1f', 'bbox', 'used_flip', 'vis_flip', 'od', 'fd', 'pid')}
        for pi in range(len(pnames)):
            k1, k2 = kps[2 * pi], kps[2 * pi + 1]
            k1f = flip_keypoints(k1, C.ANNO, None)
            for k, v in (('src', nidx[pnames[pi][0]]), ('trg', nidx[pnames[pi][1]]), ('k1', k1.numpy()),
                         ('k2', k2.numpy()), ('k1f', k1f.numpy()), ('bbox', thr[pi]), ('used_flip', False),
                         ('vis_flip', np.zeros(nk, bool)), ('od', np.nan), ('fd', np.nan), ('pid', pi)):
                out[k].append(v)
        arr = {k: np.asarray(v) for k, v in out.items()}
        np.savez(out_path, **arr, perm=-np.ones(nk, dtype=np.int64), used_kps=used.numpy(),
                 names=np.asarray(pnames))
        print(f'[gen] {method} {split} {cat}: {len(pnames)} pairs, {nk} kps', flush=True)


if __name__ == '__main__':
    ds, method = sys.argv[1], sys.argv[2]
    {'spairu': build_spairu}[ds](method)
