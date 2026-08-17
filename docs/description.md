# PENGWIN Task 1 v3.6.2 — Guarded-Seed Recovery

Algorithm Description · 2026-08-17

# 1. Submission status

`v3.6.2-guarded-seed` is a separate upload candidate. It retains the v3.5
Stage-A model, hybrid random-forest family router, largest-component ROI policy,
anatomy-specific Stage-B experts, and all neural-network weights. Stage-1 fill,
the v3.6 A1 decoder, and the v3.6.1 split-candidate RF are disabled. Only the
validated deterministic guarded-seed refinement is enabled.

# 2. Method overview

The submission performs per-fragment instance segmentation of pelvic and
femoral fractures with a serial two-stage STU-Net-B cascade:

1. **Stage A — anatomy.** A 5-class whole-volume model identifies sacrum,
   left/right hipbone and femur.
2. **Hybrid family routing.** A random forest selects pelvic or femur when
   confident; the organizers' acquisition rule is the uncertainty tiebreak.
3. **Stage B — fracture experts.** The routed anatomy selects a Sacrum,
   shared-Left/Right-Hip or Femur expert. It emits four ABBC channels and nine
   learned same-instance affinity channels at 1-, 3- and 9-voxel offsets.
4. **Safe base.** The exact v3.5 one-voxel average-linkage affinity decoder at
   `T=0.75` produces the initial instance partition.
5. **Guarded seed recovery.** Within each base instance, ABBC Boundary evidence
   proposes a binary split. It is committed only when each final piece is at
   least 1,000 mm3, each contains at least 50 Core voxels, and all three
   affinity ranges provide separating evidence. At most two refinement passes
   are run. Post-split hard merging is disabled.
6. **Assembly.** Foreground support is unchanged and instances use the official
   ranges: sacrum 1-50, left hip 51-100, right hip 101-150 and femur 151-200.

# 3. Frozen deployment contract

| Component | Deployed value |
|---|---|
| Stage A | `Dataset539_PelvicFemurAnatomyV3`, `PengwinTrainerSTUNetBaseAnatomyV301`, fold 0 |
| Family router | v3.3 hybrid RF, confidence margin 0.15 |
| Stage B dataset | `Dataset538_PelvicFemurBICMFragmentV5` |
| Stage B experts | `...V308SacrumExpertDeployedVal`, `...V308HipExpertDeployedVal`, `...V308FemurExpertDeployedVal` |
| Stage B fold/checkpoint | fold 0 / `checkpoint_best.pth` |
| Stage B output | 13 channels: 4 ABBC + 9 affinity |
| Safe base | v3.5 one-voxel affinity agglomeration, `T=0.75` |
| Split passes | 2 |
| Minimum final piece | 1,000 mm3 |
| Minimum Core support | 50 voxels on each side |
| Affinity agreement | separating evidence from 3/3 ranges: 1, 3 and 9 voxels |
| Post-split hard merge | disabled |
| Stage-1 fill | disabled |

The model archive is byte-identical to v3.5. No new learned parameters or RF
artifact are added for v3.6.2.

# 4. Model archive layout

Grand Challenge extracts the archive directly under `/opt/ml/model`:

```text
/opt/ml/model/
├── nnunet/results/
│   ├── Dataset539_PelvicFemurAnatomyV3/
│   │   └── PengwinTrainerSTUNetBaseAnatomyV301__.../fold_0/
│   └── Dataset538_PelvicFemurBICMFragmentV5/
│       ├── PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal__.../fold_0/
│       ├── PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal__.../fold_0/
│       └── PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal__.../fold_0/
└── stage1_router/stage1_target_router_fold0.joblib
```

# 5. Local evaluation

The frozen evaluation contains 68 patients and 132 anatomy samples and uses the
official-aligned local proxy. It is not a hidden-test leaderboard score.

| Decoder | Dice | HD95 mm | ASSD mm | Recall | Precision | F1 | Merge | Split | Topology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.5 safe base | 0.855651 | 6.687490 | 1.852238 | 0.931412 | **0.932197** | 0.916377 | 21 | **330** | **0.307585** |
| **v3.6.2 guarded seed** | **0.862738** | **6.358605** | **1.765682** | **0.937725** | 0.930934 | **0.920165** | **20** | 337 | 0.301272 |

Guarded seed improves overlap, both surface distances, Recall, Instance F1 and
Merge while increasing Split by seven. Recall for 1-5 cm3 GT fragments remains
32/53; recall for fragments at least 5 cm3 improves from 257/267 to 259/267.

# 6. Reproducibility and limitations

- The guarded-seed parameters above are the exact configuration evaluated on
  the frozen local arrays.
- Runtime checks fail fast if fewer than nine affinity channels are present, if
  any hard merge is accepted, or if foreground support changes.
- The local proxy was used to compare decoder variants, so its results may be
  optimistic. Hidden-test generalization must be established by submission.
- Stage-1 support fill is deliberately disabled because it reduced 1-5 cm3
  fragment recall from 32/53 to 31/53 despite improving geometry metrics.
- The 10-minute platform runtime and non-root execution must be checked before
  activation.

# 7. Runtime and licensing

The image targets an NVIDIA T4 16 GiB GPU. Stage A and each Stage-B expert are
loaded serially; guarded-seed refinement is deterministic CPU post-processing
inside each anatomy crop. STU-Net-B and nnU-Net are used under their respective
open-source licenses; Stage-B initialization derives from the official
TotalSegmentator `base_ep4k` checkpoint.

# 8. References

- Isensee, F. et al. nnU-Net, *Nature Methods* 18, 203-211 (2021).
- STU-Net / TotalSegmentator bone pretraining.
- PENGWIN 2026 Task 1 baseline:
  `github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline`.
