# PENGWIN 2026 Task 1 — V2.6 Updated-Data STU-Net Submission

**Algorithm Description · 2026-07-24**

# 1. Submission summary

This submission performs per-fragment instance segmentation of pelvic and femoral
fractures in CT. It uses a two-stage STU-Net-B cascade:

1. **Stage A — anatomy segmentation:** a whole-volume five-class network predicts
   background, sacrum, left hip, right hip, and femur.
2. **Official case routing:** the pelvic/femur decision functions published by the
   PENGWIN organizers select the target anatomy family.
3. **Stage B — fracture segmentation:** a CT-only network predicts four ABBC channels
   plus nine learned same-instance affinity channels for each target-anatomy ROI.
4. **Instance decoding and assembly:** average-linkage affinity agglomeration produces
   fragment instances, removes components below approximately 1 cm³, and maps the
   instances to the official PENGWIN label ranges.

V2.6 keeps the verified inference code, official case router, network architectures,
decoder, thresholds, and post-processing unchanged. Both deployed checkpoints were
retrained after rebuilding the datasets from the refreshed official PENGWIN release:

- Stage A: `PengwinTrainerSTUNetBaseAnatomyV301`, fold 0
- Stage B: `PengwinTrainerSTUNetBaseAffinityV308`, fold 0

# 2. Data and split

Training uses the official PENGWIN 2026 Task 1 release containing 340 CT cases:
170 pelvic-fracture cases and 170 femoral-fracture cases. A case-grouped five-fold
split with seed 12345 is shared by both stages. The deployed fold-0 models are trained
on 272 cases and evaluated locally on the same held-out 68 cases (34 pelvic and
34 femoral) without patient overlap between training and validation.

Stage A uses a partial-label-aware anatomy target: pelvic cases supervise sacrum and
left/right hips, while femoral cases supervise femur. A marginal Dice+cross-entropy
loss prevents unlabeled anatomies from being treated as true background.

Stage B uses one CT channel only. The Stage-A probability is used to localize the ROI
but is not provided as a Stage-B input, avoiding cascade feature leakage.

# 3. Model architecture

Both stages use STU-Net-B, a 3D residual U-Net with approximately 58 million parameters.
The networks are initialized from a publicly available TotalSegmentator bone pretrain.
Stage A has a five-class softmax head. Stage B emits 13 channels:

- four ABBC logits: background, outer border, inter-fragment boundary, and core;
- nine affinity logits at short, middle, and long offsets along the three spatial axes.

The affinity loss is class-balanced so that rare cross-fragment edges are not
overwhelmed by the more frequent same-fragment voxel pairs.

# 4. Routing and decoding

The container calls the organizers' published `get_image_info()` and
`classify_pelvic_femur()` rules using the specified SimpleITK/NumPy axis mapping.
Pelvic cases process Sacrum, LeftHip, and RightHip. Femur cases process Femur only.
The legacy random-forest router is disabled and no router artifact is required.

For every routed anatomy, Stage A supplies a padded ROI. Stage B predicts the ABBC and
affinity fields in that ROI. The decoder first oversegments the foreground at learned
separation ridges, then merges adjacent supervoxels using average-linkage interface
affinity with `PENGWIN_AGGLO_T=0.45`. Small components are pruned and local instance IDs
are mapped to:

- sacrum: 1–50
- left hip: 51–100
- right hip: 101–150
- femur: 151–200

# 5. Local validation

The end-to-end cascade was rerun on all 68 held-out fold-0 cases with the updated Stage A
and Stage B checkpoints. The project's official-aligned local proxy produced:

| Metric | V2.6 updated-data checkpoints |
|---|---:|
| Fracture IoU | 0.847755 |
| Fracture Dice | 0.887390 |
| Local Dice (20 mm) | 0.887545 |
| HD95 | 3.630712 mm |
| ASSD | 0.930238 mm |
| Instance Recall | 0.949466 |
| Instance Precision | 0.910110 |
| Instance F1 | 0.914869 |
| Topology Consistency | 0.295903 |
| Merge Errors | 10 |
| Split Errors | 500 |

These are local proxy values, not results from the unpublished Grand Challenge test
evaluator. They are reported for reproducibility and model-version verification.

# 6. Deployment contract

The algorithm is packaged as a non-root Docker container based on
`pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime`. Grand Challenge extracts the separate
model archive under `/opt/ml/model`. The container loads Stage A and Stage B serially,
so peak GPU memory is the maximum of the two stages rather than their sum. The runtime
is designed for an NVIDIA T4 16 GiB GPU and the 10-minute per-case limit.

The container logs the network class and a weight checksum at startup. The V2.6
checkpoints produce approximately:

- Stage A `w0sum`: `1.0411e+02`
- Stage B `w0sum`: `1.0841e+02`

# 7. Checkpoint identity

SHA-256:

- Stage A V301:
  `0d52f0fa41a69462d9ff757fb9417e70d1101104a896e5f4ed709b4ea2566509`
- Stage B V308:
  `66c1b47d9df250add49bff9997c373be949653a6f9651a281081be896d53534c`
- Grand Challenge model archive `model_v301_new_v308_new_20260724.tar.gz`:
  `7ba9fa8e6b6ac95bbfcdc006573c67e915d993466019abadb18da0cf36a4d240`

# 8. Limitations

- The official hidden-test evaluator is unavailable locally.
- This is a single-fold, single-seed submission without ensembling.
- The refreshed models improve recall and surface-distance metrics locally but reduce
  precision and instance F1 relative to the previous checkpoints.

# 9. References

- Isensee F. et al. nnU-Net. *Nature Methods* 18, 203–211 (2021).
- Huang Y. et al. STU-Net.
- PENGWIN 2026 Task 1 official baseline.
- PENGWIN 2024 first-place ABBC formulation.
- Bailoni A. et al. GASP average-linkage agglomeration, CVPR 2022.
