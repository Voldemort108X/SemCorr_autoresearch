"""One experiment: python scripts/run_exp.py <seq> <split> [method] [out_dir] -> prints ^<split>: / ^pck / ^sec: lines."""
import os, sys
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import prepare as P
if len(sys.argv) > 3 and sys.argv[3] == 'tlrsup':
    import scripts.sup_hook  # noqa: F401  (supervised TLR checkpoint; patches prepare)
import decode as D

seq, split = sys.argv[1], sys.argv[2]
method = sys.argv[3] if len(sys.argv) > 3 else 'tlr'
out = sys.argv[4] if len(sys.argv) > 4 else None
for kv in filter(None, os.environ.get('OVR', '').split(',')):   # e.g. OVR="T3_MODEL=" (logged in run.log)
    k, v = kv.split('=')
    setattr(D, k, (v == 'True') if isinstance(getattr(D, k), bool) else type(getattr(D, k))(v))
    print(f'override {k}={v!r}')
if hasattr(D, 'setup'):
    D.setup(method=method, split=split)
P.evaluate(D.decode, method, split, out_dir=out)
