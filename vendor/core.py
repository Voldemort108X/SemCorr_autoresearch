"""Vendored SemCorr2D utilities (copies, not imports). Sources (read-only):
  SC = /workspace/SemCorr_proj/Code/SemCorrVLM_cvpr26/semcorr_vfm/SemCorr2D
  - normalize_feats, prepare_feature_paths_and_load (desc assembly): SC/pck_train_patch_ours_neurips_rebuttal.py
  - convert_to_binary_mask, mask_to_patch_mask, kpts_to_patch_idx: SC/utils/utils_correspondence.py
  - preprocess_kps_pad, load_spair_data (pair parsing): SC/utils/utils_dataset.py
  - fast_get_distance, ReviewPairState, window_soft_argmax, select ('top1'/'graded'): SC/utils/utils_review_heads.py
Changes vs. the originals are marked with 'CHANGED'.
"""
import json
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

NP = 60          # patches per side
ANNO = 840       # annotation frame
SD_DIMS = 640 + 1280 + 1280


# ----------------------------------------------------------------------------- features
def normalize_feats(feats, dummy_net, epsilon=1e-10):
    if dummy_net:  # separate norm for the SD and DINO blocks
        feat_sd = feats[..., :SD_DIMS]
        feat_dino = feats[..., SD_DIMS:]
        norms_sd = torch.linalg.norm(feat_sd, dim=-1)[:, :, None]
        norm_feats_sd = feat_sd / (norms_sd + epsilon)
        norms_dino = torch.linalg.norm(feat_dino, dim=-1)[:, :, None]
        norm_feats_dino = feat_dino / (norms_dino + epsilon)
        feats = torch.cat([norm_feats_sd, norm_feats_dino], dim=-1)
    norms = torch.linalg.norm(feats, dim=-1)[:, :, None]
    return feats / (norms + epsilon)


def gather_raw(s3, s4, s5, dino, num_patches=NP):
    """[B,C,h,w] blocks -> [B, 3968, 60, 60] (same concat/interp as prepare_feature_paths_and_load)."""
    return torch.cat([
        s3,
        F.interpolate(s4, size=(num_patches, num_patches), mode='bilinear', align_corners=False),
        F.interpolate(s5, size=(num_patches, num_patches), mode='bilinear', align_corners=False),
        dino,
    ], dim=1)


def desc_from_gathered(gathered, aggre_net, dummy_net, num_patches=NP):
    """gathered [1,C,60,60] -> normalized desc [1, P, D] (= get_patch_descriptors output)."""
    desc = aggre_net(gathered).reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
    return normalize_feats(desc[0], dummy_net)


# ----------------------------------------------------------------------------- masks / kps
def convert_to_binary_mask(img_path, threshold=127):
    img = Image.open(img_path).convert('L')
    img_tensor = torch.from_numpy(np.array(img))
    return (img_tensor > threshold).float()


def mask_to_patch_mask(mask, num_patches=NP, dilation_radius=0):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    device = mask.device
    mask = mask.float()
    mask_patch = F.interpolate(mask, size=(num_patches, num_patches), mode="nearest")
    if dilation_radius > 0:
        k = 2 * dilation_radius + 1
        kernel = torch.ones(1, 1, k, k, device=device)
        dilated = F.conv2d(mask_patch, kernel, padding=dilation_radius)
        mask_patch = (dilated > 0).float()
    mask_patch = mask_patch[0, 0]
    return (mask_patch > 0.5).view(-1)


def kpts_to_patch_idx(img1_kps, num_patches=NP, anno=ANNO):
    img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy()
    img1_y_patch = (num_patches / anno * img1_y).astype(np.int32)
    img1_x_patch = (num_patches / anno * img1_x).astype(np.int32)
    return num_patches * img1_y_patch + img1_x_patch


def preprocess_kps_pad(kps, img_width, img_height, size):
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3].clone()
    return kps, offset_x, offset_y, scale


def load_spair_pairs(path, category, split, size=ANNO, unseen_kps=False):
    """load_spair_data without subsampling. CHANGED: returns ONE threshold per pair for every split
    (target bbox, as the original does for val/test; the original appends src+trg for 'trn'), and the
    src/trg image names."""
    pair_ann_subpath = 'ExtraPairAnnotation' if unseen_kps else 'PairAnnotation'
    im_ann_subpath = 'ExtraImageAnnotation' if unseen_kps else 'ImageAnnotation'
    pairs = sorted(glob(f'{path}/{pair_ann_subpath}/{split}/*{category}.json'))
    thresholds, kps, names = [], [], []
    category_anno = list(glob(f'{path}/{im_ann_subpath}/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    img_cache = {}

    def img_kps(name):
        if name not in img_cache:
            with open(f'{path}/{im_ann_subpath}/{category}/{name.replace("jpg", "json")}') as f:
                img_cache[name] = json.load(f)['kps']
        return img_cache[name]

    for pair in pairs:
        source_kps = torch.zeros(num_kps, 3)
        target_kps = torch.zeros(num_kps, 3)
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        target_bbox = np.asarray(data["trg_bndbox"])
        kpts_src = img_kps(data["src_imname"])
        kpts_trg = img_kps(data["trg_imname"])
        source_size = data["src_imsize"][:2]
        target_size = data["trg_imsize"][:2]
        for i in range(30):
            point = kpts_src[str(i)]
            if point is None:
                source_kps[i, :3] = 0
            else:
                source_kps[i, :2] = torch.Tensor(point).float()
                source_kps[i, 2] = 1
        source_kps, _, _, _ = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        for i in range(30):
            point = kpts_trg[str(i)]
            if point is None:
                target_kps[i, :3] = 0
            else:
                target_kps[i, :2] = torch.Tensor(point).float()
                target_kps[i, 2] = 1
        target_kps, _, _, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]) * trg_scale)
        kps.append(source_kps)
        kps.append(target_kps)
        names.append((data["src_imname"], data["trg_imname"]))
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    return kps, thresholds, used_kps, names


def fast_get_distance(feature1, feature2, mask1, mask2, RESOLUTION=64):
    ps = RESOLUTION
    src = feature1.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
    tgt = feature2.squeeze().permute(1, 0).reshape(1, -1, 60, 60).cuda()
    sm = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(ps, ps), mode='nearest').squeeze().cuda()
    tm = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(ps, ps), mode='nearest').squeeze().cuda()
    su = F.interpolate(src, size=(ps, ps), mode='bilinear').squeeze()
    tu = F.interpolate(tgt, size=(ps, ps), mode='bilinear').squeeze()
    su = su * sm.repeat(su.shape[0], 1, 1)
    tu = tu * tm.repeat(su.shape[0], 1, 1)
    su[su == 0] = -100000
    tu[tu == 0] = -100000
    s2 = su.reshape(su.shape[0], -1).permute(1, 0)
    t2 = tu.reshape(tu.shape[0], -1).permute(1, 0)
    rows = (sm.reshape(-1) == 1)
    s2 = s2[rows].double()
    t2 = t2.double()
    d2 = (s2 * s2).sum(1, keepdim=True) + (t2 * t2).sum(1).unsqueeze(0) - 2 * s2 @ t2.t()
    dmin = d2.clamp(min=0).min(dim=1)[0].sqrt()
    return dmin.float().cpu().mean()


# ----------------------------------------------------------------------------- Deja Vu heads
class PairState:
    """= utils_review_heads.ReviewPairState (args -> explicit anno size). img*_desc: [P, D]."""

    def __init__(self, img1_desc, img2_desc, img1_patch_idx, src_mask, tgt_mask,
                 num_patches=NP, anno=ANNO):
        if img1_desc.dim() == 3:
            img1_desc = img1_desc[0]
        if img2_desc.dim() == 3:
            img2_desc = img2_desc[0]
        self._src_desc_full, self._trg_desc_full = img1_desc, img2_desc   # CHANGED: kept for decoders
        self.device = device = img1_desc.device
        self.num_patches = num_patches
        self.P = P = img1_desc.shape[0]
        self.N = img1_patch_idx.shape[0]
        self.anno_stride = anno / num_patches
        if not torch.is_tensor(img1_patch_idx):
            img1_patch_idx = torch.from_numpy(np.asarray(img1_patch_idx))
        img1_patch_idx = img1_patch_idx.to(device=device, dtype=torch.long)
        src_mask = src_mask.to(device).bool().clone()
        tgt_mask = tgt_mask.to(device).bool()
        kpt_is_dummy = (img1_patch_idx == 0)
        src_mask[img1_patch_idx[~kpt_is_dummy]] = True
        self.src_mask = src_mask
        self.src_valid_idx = torch.nonzero(src_mask, as_tuple=True)[0]
        self.tgt_valid_idx = torch.nonzero(tgt_mask, as_tuple=True)[0]
        self.tgt_mask = tgt_mask
        Ps, Pt = self.src_valid_idx.numel(), self.tgt_valid_idx.numel()
        self.Ps, self.Pt = Ps, Pt
        src_orig2valid = torch.full((P,), -1, dtype=torch.long, device=device)
        src_orig2valid[self.src_valid_idx] = torch.arange(Ps, device=device)
        self.src_orig2valid = src_orig2valid
        self.kpt_is_valid = (~kpt_is_dummy) & src_mask[img1_patch_idx]
        self.valid_kpt_indices = torch.nonzero(self.kpt_is_valid, as_tuple=True)[0]
        self.M = M = self.valid_kpt_indices.numel()
        if M == 0:
            return
        idx_valid = img1_patch_idx[self.valid_kpt_indices]
        self.idx_valid = idx_valid
        img1_valid = img1_desc[self.src_valid_idx]
        img2_valid = img2_desc[self.tgt_valid_idx]
        self.img1_valid, self.img2_valid = img1_valid, img2_valid
        self.sim_valid = img1_valid @ img2_valid.t()                      # [Ps, Pt]
        self.kpt_src_valid_idx = src_orig2valid[idx_valid]                # [M]
        self.sim_kpts = self.sim_valid[self.kpt_src_valid_idx]            # [M, Pt]
        back_nn_orig = self.src_valid_idx[torch.argmax(self.sim_valid.t(), dim=1)]
        self.back_nn_orig = back_nn_orig
        self.back_nn_y = (back_nn_orig // num_patches).float()
        self.back_nn_x = (back_nn_orig % num_patches).float()
        self.kpt_y = (idx_valid // num_patches).float()
        self.kpt_x = (idx_valid % num_patches).float()
        full_sim = img1_valid[self.kpt_src_valid_idx] @ img2_desc.t()     # [M, P]
        full_sim = full_sim.masked_fill(~tgt_mask.unsqueeze(0), -float('inf'))
        self.full_sim_2d = full_sim.view(M, num_patches, num_patches)
        self._sorted_cache = {}

    def sorted_by_temperature(self, temperature):
        key = round(float(temperature), 8)
        if key not in self._sorted_cache:
            probs = F.softmax(self.sim_kpts / temperature, dim=-1)
            sp, si = torch.sort(probs, dim=-1, descending=True)
            self._sorted_cache[key] = (si, torch.cumsum(sp, dim=-1))
        return self._sorted_cache[key]


def window_soft_argmax_2d(sim2d, by, bx, window=5, beta=0.01, stride=ANNO / NP):
    """CHANGED (generalised): window_soft_argmax on an explicit [M, nP, nP] masked similarity."""
    M, nP = sim2d.shape[0], sim2d.shape[1]
    dev = sim2d.device
    y_rng = torch.arange(nP, device=dev).view(1, nP, 1)
    x_rng = torch.arange(nP, device=dev).view(1, 1, nP)
    y_lo = (by - window).clamp(min=0).view(M, 1, 1)
    y_hi = (by + window + 1).clamp(max=nP).view(M, 1, 1)
    x_lo = (bx - window).clamp(min=0).view(M, 1, 1)
    x_hi = (bx + window + 1).clamp(max=nP).view(M, 1, 1)
    in_win = (y_rng >= y_lo) & (y_rng < y_hi) & (x_rng >= x_lo) & (x_rng < x_hi)
    s = sim2d.masked_fill(~in_win, -float('inf'))
    flat = s.reshape(M, -1)
    prob = torch.exp((flat - flat.max(dim=1, keepdim=True)[0]) / beta)
    prob = prob / prob.sum(dim=1, keepdim=True)
    prob2d = prob.view(M, nP, nP)
    coord = torch.arange(nP, device=dev).float()
    nn_x = (prob2d.sum(dim=1) * coord).sum(dim=1) * stride + stride / 2.0
    nn_y = (prob2d.sum(dim=2) * coord).sum(dim=1) * stride + stride / 2.0
    return nn_x, nn_y


def window_soft_argmax(state, by, bx, window=5, beta=0.01):
    return window_soft_argmax_2d(state.full_sim_2d, by, bx, window, beta, state.anno_stride)


def nominate(state, top_p=0.9, temperature=0.1, min_k=1, max_k=30):
    """Top-p pool of select(): returns topk_idx [M,K] (valid-target indices), topk_sim, in_pool, cutoff_k."""
    sorted_idx, cumprobs = state.sorted_by_temperature(temperature)
    cutoff_k = (cumprobs >= top_p).float().argmax(dim=-1) + 1
    cutoff_k = cutoff_k.clamp(min=min_k, max=max_k)
    K = int(cutoff_k.max().item())
    topk_idx = sorted_idx[:, :K]
    topk_sim = torch.gather(state.sim_kpts, 1, topk_idx)
    in_pool = torch.arange(K, device=state.device).unsqueeze(0) < cutoff_k.unsqueeze(1)
    return topk_idx, topk_sim, in_pool, cutoff_k


def select_graded(state, alpha, top_p=0.9, temperature=0.1, min_k=1, max_k=30):
    """select() for rule 'graded' (alpha = lambda; alpha=0 -> 'top1' within the pool = global argmax)."""
    nP = state.num_patches
    topk_idx, topk_sim, in_pool, cutoff_k = nominate(state, top_p, temperature, min_k, max_k)
    d_cyc = torch.sqrt((state.back_nn_y[topk_idx] - state.kpt_y.unsqueeze(1)) ** 2 +
                       (state.back_nn_x[topk_idx] - state.kpt_x.unsqueeze(1)) ** 2)
    score = topk_sim - alpha * (d_cyc / nP) if alpha != 0 else topk_sim
    score = score.masked_fill(~in_pool, -1e9)
    best = torch.argmax(score, dim=1)
    pid = state.tgt_valid_idx[topk_idx.gather(1, best[:, None])[:, 0]]
    return pid // nP, pid % nP
