"""Markdown tables from run dirs: python scripts/tables.py <name=run_dir> ... (first must be any run of the split; refs
top1/dv/oracles come from its err.npz). Per-category per-keypoint PCK@.10 and overall @.10/.05/.01 (or the run's thrs)."""
import sys, json
import numpy as np
runs = [a.split('=', 1) for a in sys.argv[1:]]
first = np.load(f'{runs[0][1]}/err.npz')
thrs = json.load(open(f'{runs[0][1]}/metrics.json')).get('thrs', [0.10, 0.05, 0.01])
cats = first['cat'] if 'cat' in first.files else None
H = list(first['ref_heads'])
cols = [('top1', first['ref'][H.index('top1')]), ('dv', first['ref'][H.index('dv')])]
cols += [(n, np.load(f'{d}/err.npz')['err']) for n, d in runs]
cols += [('oracle@5', first['ref'][H.index('or5')]), ('oracle@30', first['ref'][H.index('or30')])]
pk = lambda e, t: 100 * (e <= t).mean()
print('| | ' + ' | '.join(n for n, _ in cols) + ' |')
print('|---' * (len(cols) + 1) + '|')
for t in thrs:
    print(f'| PCK@{t:g} | ' + ' | '.join(f'{pk(e, t):.2f}' for _, e in cols) + ' |')
if cats is not None and len(np.unique(cats)) > 1:
    print()
    print(f'Per category, PCK@{thrs[0]:g} (per keypoint):')
    print()
    print('| category | n | ' + ' | '.join(n for n, _ in cols) + ' |')
    print('|---' * (len(cols) + 2) + '|')
    for c in np.unique(cats):
        m = cats == c
        print(f'| {c} | {m.sum()} | ' + ' | '.join(f'{pk(e[m], thrs[0]):.1f}' for _, e in cols) + ' |')
