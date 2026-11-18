"""Sanity gate (program.md §2.4): cached reference heads vs the review dumps in SemCorr_proj/Results/cvpr2027_review."""
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import prepare as P

method, split = sys.argv[1], sys.argv[2]
cats = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] else P.CATS
lamhead = {'tlr': 'lam0.5', 'cleandift': 'lam3.0', 'diysc': 'lam1.0'}[method]
R = '/workspace/SemCorr_proj/Results/cvpr2027_review/' + method
P.evaluate(lambda pair: __import__('torch').zeros(pair['arm_of_kpt'].numel(), 2, device='cuda'),
           method, split, cats, quiet=True)   # builds ref caches
A, B = {'top1': [], 'dv': []}, {'top1': [], 'dv': []}
for c in cats:
    r = np.load(f'{P.cdir(method, split)}/{c}.ref.npz')
    o = np.load(f'{R}/{split}_{c}.npz')
    h = list(o['heads'])
    for n, on in (('top1', 'top1'), ('dv', lamhead)):
        a = r['err'][list(r['heads']).index(n)]
        b = o['err'][h.index(on)]
        assert a.shape == b.shape, (c, a.shape, b.shape)
        A[n].append(a); B[n].append(b)
        mism = {t: int(((a <= t) != (b <= t)).sum()) for t in P.THRS}
        print(f'{c:12s} {n:5s} maxabs {np.abs(a-b).max():.2e} pck-mismatch kpts {mism}')
for n in A:
    a, b = np.concatenate(A[n]), np.concatenate(B[n])
    print(f'ALL {n}: ours {[round(x,2) for x in P._pck(a)]} review {[round(x,2) for x in P._pck(b)]} '
          f'diff {[round(x-y,3) for x,y in zip(P._pck(a),P._pck(b))]} n={a.size}')
