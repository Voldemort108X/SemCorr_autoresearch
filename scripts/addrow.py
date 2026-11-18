"""python scripts/addrow.py <run_dir> <seq> <track> <status> <description>  -> appends a results.tsv row"""
import json, sys
d, seq, track, status, desc = sys.argv[1:6]
m = json.load(open(f'{d}/metrics.json'))
p = m['pck']['dec']
commit = d.rstrip('/').split('/')[-1].split('_')[1]
row = [seq, commit, f"{m['dev']:.3f}", f"{m['d10']:.2f}", f"{m['d05']:.2f}", f"{m['d01']:.2f}", f'{p[0]:.2f}', f'{p[1]:.2f}',
       f'{p[2]:.2f}', f"{m['sec']:.0f}", track, status, desc + f" [vs_dv {m['delta_vs_dv']:+.3f} std {m['boot_std_vs_dv']:.3f}; split {m['split']}/{m['method']}]"]
open('results.tsv', 'a').write('\t'.join(row) + '\n')
print('\t'.join(row))
