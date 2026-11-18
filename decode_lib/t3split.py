"""Image-disjoint T3 selection protocol: per category, the trn images are split in halves A / B (seed 0); T3 is trained
on trn_train pairs with both images in A and selected on dev pairs with both images in B (no image overlap)."""
import numpy as np
import prepare as P


def halves(cat, seed=0):
    m = P.load_split('tlr', 'trn_train', cat)
    imgs = np.unique(np.concatenate([m['src'], m['trg']]))
    rng = np.random.default_rng(seed + P.CATS.index(cat))
    p = rng.permutation(imgs)
    return set(p[: len(p) // 2].tolist()), set(p[len(p) // 2:].tolist())


def row_mask(table_pair, split, side):
    """table_pair: array of 'cat/pid' per row; returns bool mask of rows whose pair lies entirely in half `side`."""
    out = np.zeros(len(table_pair), bool)
    cats = np.array([p.split('/')[0] for p in table_pair])
    pids = np.array([int(p.split('/')[1]) for p in table_pair])
    for cat in np.unique(cats):
        A, B = halves(cat)
        H = A if side == 'A' else B
        m = P.load_split('tlr', split, cat)
        pid2 = {int(p): (int(s), int(t)) for p, s, t in zip(m['pid'], m['src'], m['trg'])}
        idx = np.nonzero(cats == cat)[0]
        out[idx] = [(pid2[q][0] in H and pid2[q][1] in H) for q in pids[idx]]
    return out
