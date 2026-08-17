"""Deployment and behavior checks for the v3.6.2 guarded-seed decoder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from abbc_conservative_refine_decode import (  # noqa: E402
    refine_instances_conservatively,
)


def _run(affinity_ranges: int) -> tuple[np.ndarray, dict]:
    shape = (20, 20, 40)
    labels = np.zeros(shape, dtype=np.int32)
    labels[3:17, 3:17, 3:37] = 1

    abbc = np.full((4, *shape), 0.01, dtype=np.float32)
    abbc[0] = 0.97
    support = labels > 0
    abbc[:, support] = 0.01
    abbc[1, support] = 0.65
    abbc[3, 5:15, 5:15, 5:35] = 0.95
    abbc[2, 3:17, 3:17, 18:22] = 0.99

    affinity = np.ones((9, *shape), dtype=np.float32)
    for range_index in range(affinity_ranges):
        affinity[range_index * 3 : (range_index + 1) * 3, :, :, 18:22] = 0.0

    return refine_instances_conservatively(
        labels,
        abbc,
        affinity,
        preprocessed_spacing_zyx=(1.0, 1.0, 1.0),
        split_passes=2,
        min_split_piece_mm3=1000.0,
        min_core_voxels_per_piece=50,
        split_affinity_required_ranges=3,
        hard_merge_margin=1.0e9,
        return_report=True,
    )


def test_dockerfile_selects_only_guarded_seed():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "PENGWIN_GUARDED_SEED_DECODE=1" in dockerfile
    assert "PENGWIN_A1_PROGRESSIVE_DECODE=0" in dockerfile
    assert "PENGWIN_SPLIT_AWARE_RF_DECODE=0" in dockerfile
    assert "PENGWIN_STAGEA_BONE_RECONCILE=0" in dockerfile
    assert "PENGWIN_AGGLO_T=0.75" in dockerfile


def test_three_of_three_affinity_is_required_and_support_is_preserved():
    rejected, rejected_report = _run(affinity_ranges=2)
    accepted, accepted_report = _run(affinity_ranges=3)

    assert rejected_report["accepted_splits"] == 0
    assert accepted_report["accepted_splits"] == 1
    assert accepted_report["accepted_hard_merges"] == 0
    assert np.array_equal(rejected > 0, accepted > 0)
    assert np.unique(accepted[accepted > 0]).size == 2


def test_runtime_has_guarded_invariants():
    source = (INFERENCE_DIR / "inference.py").read_text()
    assert "split_affinity_required_ranges=3" in source
    assert "enable_small_candidate_branch=False" in source
    assert "hard_merge_margin=1.0e9" in source
    assert "accepted_hard_merges" in source
    assert "np.array_equal(decoded_pp > 0, _v35_pp > 0)" in source
