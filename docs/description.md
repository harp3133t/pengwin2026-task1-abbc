# PENGWIN Task 1 v3.6 — A1 Progressive Affinity-ABBC

Algorithm Description · 2026-08-16

# 1. Submission status

`v3.6-a1-progressive-affinity` is a separate upload candidate. It keeps the
v3.5 Stage-A model, hybrid random-forest family router, largest-component ROI
policy, anatomy-specific Stage-B experts and all model weights unchanged. Only
the deterministic instance decoder is replaced by the validation-selected A1
pipeline described below. The previous image/model pair remains available for
rollback until the candidate passes the Grand Challenge container test.

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
4. **A1 decoding.** Short-range affinity creates watershed supervoxels. Robust
   3/9-voxel evidence vetoes weak RAG merges. Full ABBC then splits partitions
   missing a fracture boundary and merges interfaces unsupported by Boundary.
   Finally, candidate fragments from 1 to 5 cm3 require agreement from all
   three affinity ranges; larger candidates require two of three.
5. **Assembly.** Instances use the official ranges: sacrum 1-50, left hip
   51-100, right hip 101-150 and femur 151-200.

# 3. Frozen model contract

| Component | Deployed value |
|---|---|
| Stage A | `Dataset539_PelvicFemurAnatomyV3`, `PengwinTrainerSTUNetBaseAnatomyV301`, fold 0 |
| Router | v3.3 hybrid RF, confidence margin 0.15 |
| Stage B dataset | `Dataset538_PelvicFemurBICMFragmentV5` |
| Stage B experts | `...V308SacrumExpertDeployedVal`, `...V308HipExpertDeployedVal`, `...V308FemurExpertDeployedVal` |
| Stage B fold/checkpoint | fold 0 / `checkpoint_best.pth` |
| Stage B output | 13 channels: 4 ABBC + 9 affinity |
| Initial partition | 1/3/9-voxel RAG-veto, `T=0.75`, minimum 32 range pairs |
| ABBC refinement | 3 split passes, 400-voxel minimum piece, Boundary merge margin `0.05/3` |
| A1 small-candidate rule | predicted smaller side 1,000-5,000 mm3: affinity 3/3; otherwise 2/3 |
| A1 affinity evidence | mean same-instance affinity >=0.55 and at least 8 observations per range |

The v3.6 model archive is byte-identical to v3.5. This isolates the decoder
change and avoids introducing a new training or weight-selection variable.

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

All rows use the same frozen 68-patient / 132-anatomy official-aligned local
proxy. They are not hidden-test leaderboard scores.

| Decoder | Dice | HD95 mm | ASSD mm | Recall | Precision | F1 | Merge | Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.5 short affinity | 0.855649 | 6.686971 | 1.852273 | 0.931412 | **0.932197** | **0.916377** | 21 | **330** |
| Aggressive Boundary-only | 0.882873 | 3.993362 | 0.983234 | **0.948142** | 0.856773 | 0.879598 | **7** | 459 |
| A0, affinity 2/3 | 0.883147 | **3.651583** | **0.901114** | 0.943723 | 0.885290 | 0.896711 | 9 | 409 |
| **v3.6 A1, small 3/3** | **0.885316** | 3.655591 | 0.901434 | 0.946248 | 0.885417 | 0.897919 | 9 | 412 |

Small-fragment (1-5 cm3) recall is 33/53 for A0 and 34/53 for A1; large
fragment recall remains 262/267. A1 improves A0 Dice by 0.002169 and F1 by
0.001209 with three additional split errors. It does not reach the predeclared
35/53 small-fragment target, so v3.6 remains an independent hidden-score probe.

# 6. Reproducibility and limitations

- Stage A, router, ROI policy, Stage-B experts and model archive are frozen.
- The four decoder modules vendored in the container are the same sources used
  for the reported ablation; synthetic regression and deployment-contract tests
  cover the stage order and A1 thresholds.
- A1 was selected on the reported validation set. Hidden-test generalization
  and the 10-minute platform runtime must be checked before activation.
- The stronger Core/Border veto and mutual-best variants were rejected because
  they lowered local instance F1 and increased split errors.

# 7. Runtime and licensing

The image targets an NVIDIA T4 16 GiB GPU. Stage A and each Stage-B expert are
loaded serially. STU-Net-B and nnU-Net are used under their respective
open-source licenses; Stage-B initialization derives from the official
TotalSegmentator `base_ep4k` checkpoint.

# 8. References

- Isensee, F. et al. nnU-Net, *Nature Methods* 18, 203-211 (2021).
- STU-Net / TotalSegmentator bone pretraining.
- PENGWIN 2026 Task 1 baseline:
  `github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline`.
