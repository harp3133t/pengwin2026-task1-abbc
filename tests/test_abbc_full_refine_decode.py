#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


INFERENCE = Path(__file__).resolve().parents[1] / "inference"
if str(INFERENCE) not in sys.path:
    sys.path.insert(0, str(INFERENCE))

from abbc_full_refine_decode import (
    fill_stage1_mask_by_nearest_instance,
    refine_instances_with_full_abbc,
)


def _probs(shape: tuple[int, int, int]) -> np.ndarray:
    probs = np.full((4, *shape), 0.01, dtype=np.float32)
    probs[0] = 0.90
    return probs


def test_missing_boundary_splits_one_fragment() -> None:
    shape = (32, 32, 48)
    labels = np.zeros(shape, dtype=np.int32)
    labels[5:27, 5:27, 4:44] = 1
    probs = _probs(shape)
    support = labels > 0
    probs[:, support] = 0.01
    probs[1, support] = 0.80
    probs[3, 10:22, 10:22, 8:40] = 0.90
    probs[2, 5:27, 5:27, 22:27] = 0.99
    output, report = refine_instances_with_full_abbc(
        labels,
        probs,
        split_passes=1,
        min_split_piece_voxels=400,
        return_report=True,
    )
    assert output.max() == 2
    assert report["accepted_splits"] == 1


def test_unsupported_interface_merges_two_fragments() -> None:
    shape = (32, 32, 48)
    labels = np.zeros(shape, dtype=np.int32)
    labels[5:27, 5:27, 4:24] = 1
    labels[5:27, 5:27, 24:44] = 2
    probs = _probs(shape)
    support = labels > 0
    probs[:, support] = 0.01
    probs[1, support] = 0.80
    probs[3, 9:23, 9:23, 8:40] = 0.90
    output, report = refine_instances_with_full_abbc(
        labels, probs, split_passes=0, return_report=True
    )
    assert output.max() == 1
    assert report["accepted_merges"] == 1


def test_supported_interface_preserves_two_fragments() -> None:
    shape = (32, 32, 48)
    labels = np.zeros(shape, dtype=np.int32)
    labels[5:27, 5:27, 4:24] = 1
    labels[5:27, 5:27, 24:44] = 2
    probs = _probs(shape)
    support = labels > 0
    probs[:, support] = 0.01
    probs[1, support] = 0.80
    probs[3, 9:23, 9:23, 8:40] = 0.90
    probs[2, 5:27, 5:27, 21:27] = 0.99
    output, report = refine_instances_with_full_abbc(
        labels, probs, split_passes=0, return_report=True
    )
    assert output.max() == 2
    assert report["accepted_merges"] == 0


def test_stage1_fill_does_not_clip_existing_instances() -> None:
    shape = (16, 16, 24)
    labels = np.zeros(shape, dtype=np.int32)
    labels[3:13, 3:13, 2:10] = 1
    labels[3:13, 3:13, 14:22] = 2
    stage1 = np.zeros(shape, dtype=bool)
    stage1[4:12, 4:12, 7:17] = True
    before = labels.copy()
    output, report = fill_stage1_mask_by_nearest_instance(labels, stage1)
    assert np.all(output[before > 0] == before[before > 0])
    assert output.max() == 2
    assert report["filled_voxels"] > 0


if __name__ == "__main__":
    test_missing_boundary_splits_one_fragment()
    test_unsupported_interface_merges_two_fragments()
    test_supported_interface_preserves_two_fragments()
    test_stage1_fill_does_not_clip_existing_instances()
    print("ABBC full-refine synthetic tests: PASS")
