# PENGWIN Task 1 v3.6.1 — Split-Aware Candidate RF

Algorithm Description · 2026-08-17

# 1. Submission status

`v3.6.1-split-aware-rf` is a separate upload candidate. The existing `v3.7`
release is not modified. v3.6.1 retains the v3.5 Stage-A model, hybrid
random-forest family router, largest-component ROI policy, anatomy-specific
Stage-B experts, and all neural-network weights. It adds a trained
split-candidate RF gate after Stage B. The prior image/model pair remains
available until this candidate passes the Grand Challenge container test.

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
   `T=0.75` produces the default instance partition.
5. **Candidate proposal.** A parallel 1/3/9-voxel RAG-veto decoder is refined
   by full ABBC split/merge logic. It proposes local binary splits inside a
   v3.5 instance but never changes foreground support.
6. **Split-aware RF gate.** Five random-forest regressors estimate candidate
   changes in Merge, Split, Dice, Instance F1, and Precision from 44
   inference-only geometry, CT, affinity, and ABBC context features. The
   learned conservative policy selects at most one candidate per source
   instance. Rejected sources remain exactly v3.5.
7. **Assembly.** Instances use the official ranges: sacrum 1-50, left hip
   51-100, right hip 101-150 and femur 151-200.

# 3. Frozen model and RF contract

| Component | Deployed value |
|---|---|
| Stage A | `Dataset539_PelvicFemurAnatomyV3`, `PengwinTrainerSTUNetBaseAnatomyV301`, fold 0 |
| Family router | v3.3 hybrid RF, confidence margin 0.15 |
| Stage B dataset | `Dataset538_PelvicFemurBICMFragmentV5` |
| Stage B experts | `...V308SacrumExpertDeployedVal`, `...V308HipExpertDeployedVal`, `...V308FemurExpertDeployedVal` |
| Stage B fold/checkpoint | fold 0 / `checkpoint_best.pth` |
| Stage B output | 13 channels: 4 ABBC + 9 affinity |
| Safe base | v3.5 one-voxel affinity agglomeration, `T=0.75` |
| Proposal | 1/3/9 RAG veto, minimum 32 range pairs; 3 ABBC split passes, 400-voxel minimum piece |
| Candidate minimum | both binary pieces at least 1,000 mm3 |
| RF inputs/outputs | 44 inference-only features / five predicted metric deltas |
| RF policy | predicted Merge <= -0.55, Split <= 2.25, Dice >= -0.01, F1 >= -0.05, Precision >= -0.10 |

The Stage-A and Stage-B neural checkpoints and the family router are unchanged
from v3.5. The model archive is new because it additionally contains five
fitted candidate-outcome RF regressors and their fixed policy metadata.

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
├── stage1_router/stage1_target_router_fold0.joblib
└── split_candidate_rf/v35_candidate_split_aware_rf_all_candidates.joblib
```

# 5. Local evaluation

The primary result is nested case-grouped out-of-fold evaluation on the frozen
68-patient / 132-anatomy official-aligned local proxy. It is not a hidden-test
leaderboard score.

| Decoder | Dice | HD95 mm | ASSD mm | Recall | Precision | F1 | Merge | Split | Topology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.5 safe base | 0.855649 | 6.686971 | 1.852273 | 0.931412 | **0.932197** | **0.916377** | 21 | **330** | **0.307585** |
| **v3.6.1 split-aware RF OOF** | **0.856133** | **6.439807** | **1.789200** | 0.931412 | 0.928409 | 0.913852 | **20** | 334 | 0.300009 |

There were 112 candidate actions. The nested OOF policy selected two candidates
in two cases: one reduced a merge error and neither increased merge errors.
The total split increase was four. The deployment regressors are then fit on
all 112 candidates; any full-fit replay on these 68 cases is treated only as a
runtime sanity check, not as unbiased evidence.

# 6. Reproducibility and limitations

- Given the frozen local base/proposal arrays, the submission candidate
  extractor reproduces all 112 training candidate keys and all 44 features
  exactly (maximum absolute error 0.0).
- Candidate fitting and policy selection use case-grouped folds; no case is
  shared between an OOF training and test fold.
- A non-root GPU container smoke test on pelvic case 039 applied one live
  LeftHip candidate. Against a separate live v3.5-safe-base run it changed
  Merge by -1, Split by +2, Dice by +0.0703, HD95 by -2.71 mm and ASSD by
  -0.814 mm. Femur case 268 generated one candidate and conservatively rejected
  it. Offline cached Stage-A crops are not byte-identical to live container
  crops, so these two live checks complement rather than replace nested OOF.
- The sample of candidate actions is small. Hidden-test generalization and the
  10-minute platform runtime must be checked before activation.
- The RF does not create foreground voxels or remove predicted bone. It only
  relabels a selected piece of one base instance with a free anatomy-local ID.

# 7. Runtime and licensing

The image targets an NVIDIA T4 16 GiB GPU. Stage A and each Stage-B expert are
loaded serially; the small CPU RF bundle is loaded only after GPU Stage-B
inference. STU-Net-B and nnU-Net are used under their respective open-source
licenses; Stage-B initialization derives from the official TotalSegmentator
`base_ep4k` checkpoint.

# 8. References

- Isensee, F. et al. nnU-Net, *Nature Methods* 18, 203-211 (2021).
- STU-Net / TotalSegmentator bone pretraining.
- PENGWIN 2026 Task 1 baseline:
  `github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline`.
