"""Limitation (a) of the paper: 'of the keypoints where the GT patch is nominated into the pool but is not the top-1
candidate, our criterion promotes X%'. Here measured with the per-keypoint errors in runs/<dir>/err.npz:
 recoverable = top-1 wrong @t and oracle-in-top-30 right @t; promoted = decoder right @t among them;
 broken = top-1 right @t but decoder wrong @t. usage: python scripts/promotion.py <run_dir> [...]"""
import sys, numpy as np
for d in sys.argv[1:]:
    z = np.load(f'{d}/err.npz')
    e, ref, hn = z['err'], z['ref'], list(z['ref_heads'])
    t1, dv, o30, o5 = ref[hn.index('top1')], ref[hn.index('dv')], ref[hn.index('or30')], ref[hn.index('or5')]
    out = []
    for t in (0.10, 0.05, 0.01):
        rec = (t1 > t) & (o30 <= t)
        for name, x in (('dv', dv), ('dec', e)):
            out.append(f"@{t:.2f} {name}: promoted {100*np.mean(x[rec] <= t):5.1f}% of {rec.sum()} recoverable, "
                       f"broken {100*np.mean(x[t1 <= t] > t):4.1f}% of top1-correct, gap closed "
                       f"{100*(np.mean(x <= t)-np.mean(t1 <= t))/(np.mean(o30 <= t)-np.mean(t1 <= t)):5.1f}% (of or30), "
                       f"{100*(np.mean(x <= t)-np.mean(t1 <= t))/(np.mean(o5 <= t)-np.mean(t1 <= t)):5.1f}% (of or5)")
    print(d); print('\n'.join(out))
