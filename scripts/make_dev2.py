"""Second selection-robustness split 'dev2': 300 pairs/category from SPair trn_train (seed 1), disjoint from dev.
Built with prepare.py's own functions (prepare.py itself is not modified). -> cache/<method>/dev2/<cat>.npz"""
import os, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
import prepare as P

orig_split_pairs = P._split_pairs


def split_pairs_dev2(cat, split, rng_seed):
    if split != 'dev2':
        return orig_split_pairs(cat, split, rng_seed)
    kps, thr, used, names, sel_tt = orig_split_pairs(cat, 'trn_train', rng_seed)
    rng = np.random.default_rng(1000 + P.CATS.index(cat))
    sel = np.sort(rng.choice(sel_tt, size=min(300, len(sel_tt)), replace=False))
    return kps, thr, used, names, sel


P._split_pairs = split_pairs_dev2
method = sys.argv[1] if len(sys.argv) > 1 else 'tlr'
for c in P.CATS:
    P.build_category(method, c, ['dev2'])
print('done')
