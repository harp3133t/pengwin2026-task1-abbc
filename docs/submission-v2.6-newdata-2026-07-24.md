# V2.6 new-data submission manifest

Prepared: 2026-07-24

## Grand Challenge pair

- GitHub tag: `v2.6-newdata`
- Model upload: `model_v301_new_v308_new_20260724.tar.gz`
- Model archive SHA-256:
  `7ba9fa8e6b6ac95bbfcdc006573c67e915d993466019abadb18da0cf36a4d240`

The Docker code uses the organizers' official pelvic/femur case router and loads the
separate model payload from `/opt/ml/model`.

## Active checkpoints

| Stage | Trainer | Fold | SHA-256 |
|---|---|---:|---|
| Anatomy | `PengwinTrainerSTUNetBaseAnatomyV301` | 0 | `0d52f0fa41a69462d9ff757fb9417e70d1101104a896e5f4ed709b4ea2566509` |
| Fracture | `PengwinTrainerSTUNetBaseAffinityV308` | 0 | `66c1b47d9df250add49bff9997c373be949653a6f9651a281081be896d53534c` |

The archive also retains the inactive V302 rollback checkpoint to preserve the
previously validated model-directory contract. Docker selects V308 explicitly.

## Local end-to-end verification

- Evaluation set: refreshed held-out fold 0, 68 cases
- Fracture Dice: `0.887390`
- HD95: `3.630712 mm`
- Instance Recall: `0.949466`
- Instance Precision: `0.910110`
- Instance F1: `0.914869`
- Expected runtime weight checksums:
  - Stage A: `w0sum ≈ 1.0411e+02`
  - Stage B: `w0sum ≈ 1.0841e+02`

These values are from the official-aligned local proxy, not the hidden Grand Challenge
test evaluator.

## Container smoke verification

The exact clean GitHub release tree was rebuilt as `pengwin-v26-newdata:test`
(`sha256:c04f50e52647a2331ba1fba5d8406dca80f3fee26074cc0cc3fc1eecff0dfea4`).
It was run with `--network none`, the new model payload mounted read-only, and the
container's non-root `user:user` runtime account.

| Case | Official route | Output labels | Runtime | Geometry |
|---|---|---|---:|---|
| 001 | pelvic | `0, 1, 51, 101` | 49.3 s | size/spacing/origin/direction PASS |
| 294 | femur | `0, 151` | 35.7 s | size/spacing/origin/direction PASS |

Both outputs were non-empty `uint8` MetaImage volumes. Runtime logs reported the expected
new Stage-A and Stage-B `w0sum` values. The Dockerfile normalizes copied source files with
`chmod -R a+rX`, so a restrictive checkout umask cannot make the non-root entrypoint
unreadable.

## Upload order

1. Push the `v2.6-newdata` Git tag to the GitHub repository.
2. In Grand Challenge, create the container image using **Link to GitHub** and select
   `v2.6-newdata`.
3. Wait until the image build succeeds.
4. Upload `model_v301_new_v308_new_20260724.tar.gz` in the algorithm's **Models** tab.
5. Run a preliminary smoke submission.
6. Confirm the logs contain the official case route and both expected `w0sum` values.
7. Only then select the new image/model pair as the final submission.
