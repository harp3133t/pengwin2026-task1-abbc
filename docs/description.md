# PENGWIN Task 1 v3.5 — Always-On Experts

Algorithm Description · 2026-08-05

# 1. Submission status

`v3.5-always-expert-t075` is an upload candidate, not an automatically activated
replacement. The current unified-model package remains available as the
rollback model until this candidate passes the Grand Challenge container test.

Relative to v3.4, this candidate changes only Stage-B checkpoint selection:
each routed anatomy always uses its Sacrum, shared-Hip, or Femur expert. Stage A,
the validated hybrid target-family router, largest-component ROI policy,
agglomeration threshold, and Grand Challenge input/output contract are fixed.

# 2. Method overview

The submission performs per-fragment instance segmentation of pelvic and
femoral fractures using a serial two-stage STU-Net-B cascade:

1. **Stage A — anatomy.** A 5-class whole-volume model identifies sacrum,
   left/right hipbone and femur.
2. **Hybrid family routing.** A random-forest classifier selects pelvic or femur
   when confident. The organizers' official acquisition rule is used only when
   the RF is uncertain.
3. **Stage B — fracture experts.** The routed anatomy selects a Sacrum,
   shared-Left/Right-Hip, or Femur expert. Each emits four ABBC channels and nine
   learned same-instance affinity channels.
4. **Instance decoding.** Average-linkage agglomeration converts the affinity
   field into fragment instances and removes components below about 1 cm3.
5. **Assembly.** Instances are mapped to the official label ranges: sacrum
   1-50, left hip 51-100, right hip 101-150 and femur 151-200.

# 3. Deployed model contract

| Component | Deployed value |
|---|---|
| Stage A dataset | `Dataset539_PelvicFemurAnatomyV3` |
| Stage A trainer | `PengwinTrainerSTUNetBaseAnatomyV301` |
| Stage A fold/checkpoint | fold 0 / `checkpoint_best.pth` |
| Stage A initialization | from scratch on refreshed PENGWIN anatomy data |
| Router | v3.3 hybrid RF, confidence margin 0.15 |
| Stage B dataset | `Dataset538_PelvicFemurBICMFragmentV5` |
| Stage B trainers | `...V308SacrumExpertDeployedVal`, `...V308HipExpertDeployedVal`, `...V308FemurExpertDeployedVal` |
| Stage B fold/checkpoint | fold 0 / `checkpoint_best.pth` |
| Stage B initialization | v3.4 V308 initialized from official TotalSegmentator `base_ep4k` |
| Expert tuning | encoder frozen; decoder and heads tuned for 3 epochs |
| Stage B output | 13 channels: 4 ABBC + 9 affinity |
| Decoder | `decode_affinity_agglo`, `T=0.75` |

Only Stage B is TotalSegmentator-initialized. Stage A is the refreshed-data
scratch model. The three packaged expert artifacts are their deterministic
epoch-3 `checkpoint_best.pth` files.

# 4. Model archive layout

Grand Challenge extracts the uploaded archive directly under `/opt/ml/model`.
The tarball is created with `tar -C model_payload -czf model.tar.gz .`, producing:

```text
/opt/ml/model/
├── nnunet/results/
│   ├── Dataset539_PelvicFemurAnatomyV3/
│   │   └── PengwinTrainerSTUNetBaseAnatomyV301__nnUNetResEncUNetLPlans__3d_fullres/
│   └── Dataset538_PelvicFemurBICMFragmentV5/
│       ├── PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal__.../
│       ├── PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal__.../
│       └── PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal__.../
└── stage1_router/stage1_target_router_fold0.joblib
```

The router artifact is serialized natively with scikit-learn 1.6.1, matching the
container dependency. Its predictions were checked against the original
scikit-learn 1.7.2 serialization on 4,096 deterministic inputs.

# 5. Local evaluation

All rows below use the same 68-patient / 132-anatomy end-to-end
official-aligned proxy-v2 evaluation set.

| Experiment | IoU-F | Dice | HD95 (mm) | ASSD (mm) | Instance F1 | Split |
|---|---:|---:|---:|---:|---:|---:|
| v3.4 unified | 0.811828 | 0.850707 | 6.752 | 1.891 | **0.919150** | **318** |
| Selective expert OOF | 0.814812 | 0.854027 | 6.835 | 1.897 | 0.921675 | 320 |
| v3.5 always expert | **0.816842** | **0.855649** | **6.687** | **1.852** | 0.916377 | 330 |

Relative to unified Stage B, always-on experts improve IoU-F by 0.005013,
Dice by 0.004942, HD95 by 0.065 mm, and ASSD by 0.038 mm. Precision decreases
by 0.010913, instance F1 decreases by 0.002772, and split errors increase by
12. This mixed trade-off is why v3.5 is a separate hidden-score probe rather
than an automatic deployment promotion. These are local proxy results, not
hidden-test leaderboard results.

# 6. Preprocessing and inference

Images are canonicalized to LPS and processed through the nnU-Net plans used in
training. Stage A predicts mutually exclusive anatomy masks. The selected
anatomy ROI is converted to a CT-only bone-window crop for Stage B. Inference
runs Stage A and one expert at a time on one GPU, restores the
prediction to the input geometry and writes one
`/output/images/peripelvic-fracture-ct-segmentation/*.mha` label map.

# 7. Reproducibility and limitations

- Fold 0 is used for both stages; no ensemble is deployed.
- The expert policy and `T=0.75` were evaluated on the reported validation data, so an independent
  Grand Challenge container/hidden-test check is required before activation.
- Always-on experts improve overlap/surface proxies but reduce precision and
  instance F1 and increase fragment splitting.
- Pelvic cases switch between Sacrum and Hip expert checkpoints serially; the
  Grand Challenge runtime test must confirm the 10-minute case limit.
- The official challenge evaluator and hidden test distribution are not local,
  so the reported metrics are comparative proxies.
- Exact artifact hashes and source paths are recorded in the release
  `MODEL_MANIFEST.json`.

# 8. Runtime and licensing

The Docker image targets an NVIDIA T4 16 GiB GPU. Stage A and each Stage B expert
are loaded serially to limit peak memory. STU-Net-B and nnU-Net are used under their
respective open-source licenses; the Stage-B initialization comes from the
official TotalSegmentator `base_ep4k` checkpoint.

# 9. References

- Isensee, F. et al. nnU-Net, *Nature Methods* 18, 203-211 (2021).
- STU-Net / TotalSegmentator bone pretraining.
- PENGWIN 2026 Task 1 baseline:
  `github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline`.
