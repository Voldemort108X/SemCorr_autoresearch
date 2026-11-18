# DAY 1 — 2026-11-18

## Day-0 setup
- vendor/: core.py (desc assembly, masks, kps, fast_get_distance, PairState, select/graded, window soft-argmax),
  geoware.py, projection_network.py + resnet.py (DIY-SC). Sources noted per file.
- Cache: raw SD/DINO blocks in fp16 per category ([100 img, 2 arms]) + dilated patch masks; descriptors are re-assembled
  and normalised in fp32 on the GPU per pair. Flip arm decided once from the fp32 originals (exact replica).
- Sanity on val/aeroplane: top1 and lam0.5 identical to the review dump at every keypoint (max |Δerr| 1.2e-4, 0 PCK flips).
- Note: dev bbox = target bbox (the original loader appends src+trg thresholds for 'trn'; fixed in the vendored copy).
- Note: the decoder gets `arm_of_kpt` (frozen adaptive flip; it uses mutual visibility, as the original code does) but not
  the evaluated-keypoint mask; joint decoders use the source-visible keypoints of each arm as the query set.
