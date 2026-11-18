"""python scripts/sweep.py <split> 'NAME=val,NAME2=val' ['...'] : evaluate decode.py with module attrs overridden (no files)."""
import os, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import prepare as P
import decode as D
split = sys.argv[1]
cats = None
DEFAULTS = {k: v for k, v in vars(D).items() if k.isupper()}
for cfg in sys.argv[2:]:
    for k, v in DEFAULTS.items():
        setattr(D, k, v)
    if cfg.startswith('cats='):
        cats = cfg[5:].split(','); continue
    for kv in filter(None, cfg.split(',')):
        k, v = kv.split('=')
        setattr(D, k, type(getattr(D, k))(v) if not isinstance(getattr(D, k), bool) else v == 'True')
    m = P.evaluate(D.decode, os.environ.get('METHOD', 'tlr'), split, cats=cats, quiet=True)
    pk = '/'.join(f'{x:.2f}' for x in m['pck']['dec'])
    print(f"{cfg:40s} {os.environ.get('METHOD', 'tlr')} pck {pk} {split}: {m['dev']:+.3f} d {m['d10']:+.2f}/{m['d05']:+.2f}/{m['d01']:+.2f} vs_dv {m['delta_vs_dv']:+.3f}", flush=True)
