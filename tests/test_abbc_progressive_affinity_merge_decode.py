#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


INFERENCE = Path(__file__).resolve().parents[1] / "inference"
if str(INFERENCE) not in sys.path:
    sys.path.insert(0, str(INFERENCE))

from abbc_progressive_affinity_merge_decode import (
    apply_progressive_affinity_merge,
)


def _scene_three() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (20, 20, 48)
    labels = np.zeros(shape, dtype=np.int32)
    labels[4:16, 4:16, 0:16] = 1
    labels[4:16, 4:16, 16:32] = 2
    labels[4:16, 4:16, 32:48] = 3
    probs = np.full((4, *shape), 0.05, dtype=np.float32)
    probs[0] = 0.90
    support = labels > 0
    probs[:, support] = 0.10
    probs[1, support] = 0.30
    probs[3, support] = 0.20
    affinities = np.zeros((9, *shape), dtype=np.float32)
    return labels, probs, affinities


def _run(labels, probs, affinities, **kwargs):
    return apply_progressive_affinity_merge(
        labels,
        probs,
        affinities,
        preprocessed_spacing_zyx=(1.0, 1.0, 1.0),
        return_report=True,
        **kwargs,
    )


def test_a1_small_requires_three_ranges() -> None:
    labels, probs, affinity = _scene_three()
    affinity[:6] = 0.90
    output, report = _run(
        labels, probs, affinity, small_strict=True, max_merges=1
    )
    assert output.max() == 3
    assert report["accepted"] == 0
    affinity[6:] = 0.90
    output, report = _run(
        labels, probs, affinity, small_strict=True, max_merges=1
    )
    assert output.max() == 2
    assert report["accepted"] == 1


def test_a2_core_and_border_vetoes() -> None:
    labels, probs, affinity = _scene_three()
    labels[labels == 3] = 0
    affinity[:] = 0.90
    # Robust hard-Core support in the small fragment blocks the merge.
    probs[3, labels == 1] = 0.95
    output, report = _run(
        labels,
        probs,
        affinity,
        small_strict=True,
        enable_small_veto=True,
        max_merges=1,
    )
    assert output.max() == 2
    assert report["rejected_core_veto"] > 0

    labels, probs, affinity = _scene_three()
    labels[labels == 3] = 0
    affinity[:] = 0.90
    probs[1, labels > 0] = 0.60
    output, report = _run(
        labels,
        probs,
        affinity,
        small_strict=True,
        enable_small_veto=True,
        max_merges=1,
    )
    assert output.max() == 2
    assert report["rejected_border_veto"] > 0


def test_a3_mutual_best_prevents_cascade() -> None:
    labels, probs, affinity = _scene_three()
    affinity[:] = 0.90
    greedy, greedy_report = _run(
        labels,
        probs,
        affinity,
        small_strict=True,
        enable_small_veto=True,
        mutual_best_single_round=False,
    )
    mutual, mutual_report = _run(
        labels,
        probs,
        affinity,
        small_strict=True,
        enable_small_veto=True,
        mutual_best_single_round=True,
    )
    assert greedy.max() == 1
    assert greedy_report["accepted"] == 2
    assert mutual.max() == 2
    assert mutual_report["accepted"] == 1


if __name__ == "__main__":
    test_a1_small_requires_three_ranges()
    test_a2_core_and_border_vetoes()
    test_a3_mutual_best_prevents_cascade()
    print("ABBC progressive-affinity synthetic tests: PASS")
