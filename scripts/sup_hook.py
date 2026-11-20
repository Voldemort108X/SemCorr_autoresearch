"""Method 'tlrsup': the released SUPERVISED TLR checkpoint (AggregationNetwork trained on SPair trn, best_856.PTH,
the paper's supp. 'Supervised Setting') applied to the cached TLR raw blocks (SD s3/s4/s5 + DINO); split meta (pairs,
keypoints, adaptive-flip arms) shared with 'tlr' via symlinks in cache/tlrsup; reference heads cached separately.
Patches prepare.METHODS / prepare.CatFeats.desc at import (prepare.py itself unchanged)."""
import torch
import prepare as P
from vendor import core as C

SUP_CKPT = '/workspace/SemCorr_proj/Code/SemCorrVLM_cvpr26/semcorr_vfm/baselines_neurips/cleandift/results_spair/best_856.PTH'
P.METHODS['tlrsup'] = dict(fdir='features', dummy=False, flip=True, lam=0.5)
_NET = []
_orig = P.CatFeats.desc


def _net():
    if not _NET:
        from vendor.projection_network import AggregationNetwork
        net = AggregationNetwork(feature_dims=[640, 1280, 1280, 768], projection_dim=768, device='cuda', feat_map_dropout=0.2)
        net.load_pretrained_weights(torch.load(SUP_CKPT))
        _NET.append(net.cuda().eval())
    return _NET[0]


@torch.no_grad()
def desc(self, i, a):
    if self.method != 'tlrsup':
        return _orig(self, i, a)
    key = (int(i), int(a))
    if key in self._lru:
        return self._lru[key]
    g = C.gather_raw(*(self.raw[b][i, a:a + 1].float() for b in P.BLOCKS))
    d = C.desc_from_gathered(g, _net(), False)[0]
    if len(self._lru) >= 6:
        self._lru.pop(next(iter(self._lru)))
    self._lru[key] = d
    return d


P.CatFeats.desc = desc
