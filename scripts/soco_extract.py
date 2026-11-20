"""Cache SOCO dense features (original + horizontally flipped image) for one backbone, exactly as OmniProbe's
correspondence_soco task computes them (same cfg / model instantiation / image size / SOCODataset image loading,
use_bbox false -> full image padded to a square at right/bottom, resized to the effective image size).
Run with OmniProbe's venv python; writes only into cache/soco/<backbone>/ of this repo.
  cache/soco/<bb>/<cls>.npz : names [N], hw [N] (padded square side in px), feat [N,HW,F] fp16, featf [N,HW,F] fp16
  (L2-normalised per patch, as in OmniProbe / the paper's soco_eval_onepass.py)."""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as nn_F

ap = argparse.ArgumentParser()
ap.add_argument('--backbone', required=True)
ap.add_argument('--out', default='/workspace/SemCorr_autoresearch/cache/soco')
ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--classes', nargs='*', default=None)
a = ap.parse_args()
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import omniprobe.scripts.eval_correspondence_soco as soco
from omniprobe.datasets.soco import SOCODataset
from omniprobe.tasks import _runner_extra_overrides
from omniprobe.tasks.script_task import build_script_cfg
cfg_dir = os.path.abspath(os.path.join(os.path.dirname(soco.__file__), '..', '..', 'configs'))
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    run_cfg = compose('run', overrides=['task=correspondence_soco', f'backbone={a.backbone}'])
tcfg = build_script_cfg(run_cfg, extra_overrides=_runner_extra_overrides(run_cfg))
soco.set_seed(tcfg.random_seed)
model = instantiate(tcfg.backbone, output='dense', return_multilayer=tcfg.multilayer).cuda().eval()
img_size = int(soco.resolve_correspondence_image_size(tcfg, model)['effective_image_size'])
root = os.environ.get('SOCO_ROOT', '/workspace/SemCorr_proj/Datasets/SOCOv1')
ds = SOCODataset(root=root, image_size=img_size, image_mean=tcfg.image_mean, use_bbox=tcfg.use_bbox,
                 class_name='aeroplane', max_pairs=1, pair_subdir='PairAnnotations/intra')
imgs = {}
for f in glob.glob(f'{root}/PairAnnotations/*/*/*.json') + glob.glob(f'{root}/PairAnnotations/trainsplits/*/*/*.json'):
    p = json.load(open(f))
    imgs.setdefault(p['src_category'], set()).add(p['src_image'])
    imgs.setdefault(p['trg_category'], set()).add(p['trg_image'])
od = os.path.join(a.out, a.backbone); os.makedirs(od, exist_ok=True)
json.dump({'image_size': img_size, 'backbone': a.backbone, 'cfg': {k: str(v) for k, v in dict(tcfg.backbone).items()}},
          open(os.path.join(od, 'meta.json'), 'w'))
t0 = time.time()
for ci, cls in enumerate(sorted(a.classes or imgs)):
    fn = os.path.join(od, f'{cls}.npz')
    if os.path.exists(fn):
        continue
    names = sorted(imgs[cls]); hw = []; F0 = []; F1 = []
    for b in range(0, len(names), a.bs):
        ims = []
        for n in names[b:b + a.bs]:
            im = ds._load_image(cls, n, None)
            hw.append(im.size[0]); ims.append(ds.image_transform(im))
        x = torch.stack(ims).cuda()
        with torch.no_grad():
            for xx, L in ((x, F0), (torch.flip(x, dims=[-1]), F1)):
                f = model(xx)
                if isinstance(f, list):
                    f = torch.cat(f, dim=1)
                f = nn_F.normalize(f.float(), p=2, dim=1)
                B, C, H, W = f.shape
                L.append(f.reshape(B, C, H * W).transpose(1, 2).half().cpu().numpy())
    np.savez(fn, names=np.array(names), hw=np.array(hw), feat=np.concatenate(F0), featf=np.concatenate(F1), H=H, W=W)
    print(f'[{ci+1}] {cls} n={len(names)} grid {H}x{W} C={C} ({time.time()-t0:.0f}s)', flush=True)
print('DONE', img_size)
