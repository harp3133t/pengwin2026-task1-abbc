#!/usr/bin/env python
"""Synthetic checks for the experimental multi-scale RAG-veto decoder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parent
INFERENCE = TOOLS.parent / "inference"
sys.path.insert(0, str(INFERENCE))
sys.path.insert(0, str(TOOLS))

from agglo_decode import decode_affinity_agglo  # noqa: E402
from multiscale_affinity_rag_decode import (  # noqa: E402
    decode_affinity_multiscale_rag_veto,
)


def fields(long_cross: bool, mid_cross: bool = False) -> tuple[np.ndarray, np.ndarray]:
    shape = (12, 12, 30)
    abbc = np.zeros((4, *shape), dtype=np.float32)
    abbc[1] = 1.0
    abbc[3, 3:9, 3:9, 3:6] = 1.0
    abbc[1, 3:9, 3:9, 3:6] = 0.0
    abbc[3, 3:9, 3:9, 24:27] = 1.0
    abbc[1, 3:9, 3:9, 24:27] = 0.0
    affinity = np.full((9, *shape), 0.98, dtype=np.float32)
    if mid_cross:
        affinity[5] = 0.0
    if long_cross:
        # Long x-offset pairs crossing the two watershed basins give strong
        # repulsive evidence, without modifying the local ridge.
        affinity[8] = 0.0
    return abbc, affinity


def fragment_count(array: np.ndarray) -> int:
    return len([value for value in np.unique(array) if int(value) > 0])


def main() -> None:
    abbc, affinity = fields(long_cross=True)
    baseline = decode_affinity_agglo(abbc, affinity, T=0.75, min_vox=1)
    candidate, report = decode_affinity_multiscale_rag_veto(
        abbc,
        affinity,
        T=0.75,
        min_vox=1,
        min_range_pairs=1,
        return_report=True,
    )
    assert fragment_count(baseline) == 1, fragment_count(baseline)
    assert fragment_count(candidate) >= 2, fragment_count(candidate)
    assert report["long_veto_edges"] >= 1, report

    mid_only, mid_report = decode_affinity_multiscale_rag_veto(
        abbc,
        affinity,
        T=0.75,
        min_vox=1,
        min_range_pairs=1,
        use_mid=True,
        use_long=False,
        return_report=True,
    )
    assert fragment_count(mid_only) == 1, fragment_count(mid_only)
    assert mid_report["long_veto_edges"] == 0

    abbc, affinity = fields(long_cross=False, mid_cross=True)
    mid_only, mid_report = decode_affinity_multiscale_rag_veto(
        abbc,
        affinity,
        T=0.75,
        min_vox=1,
        min_range_pairs=1,
        use_mid=True,
        use_long=False,
        return_report=True,
    )
    long_only = decode_affinity_multiscale_rag_veto(
        abbc,
        affinity,
        T=0.75,
        min_vox=1,
        min_range_pairs=1,
        use_mid=False,
        use_long=True,
    )
    assert fragment_count(mid_only) >= 2, fragment_count(mid_only)
    assert mid_report["mid_veto_edges"] >= 1, mid_report
    assert fragment_count(long_only) == 1, fragment_count(long_only)

    abbc, affinity = fields(long_cross=False)
    no_veto_baseline = decode_affinity_agglo(
        abbc, affinity, T=0.75, min_vox=1
    )
    no_veto = decode_affinity_multiscale_rag_veto(
        abbc,
        affinity,
        T=0.75,
        min_vox=1,
        min_range_pairs=1,
    )
    assert fragment_count(no_veto) == 1, fragment_count(no_veto)
    assert np.array_equal(no_veto, no_veto_baseline)
    print("multiscale affinity RAG-veto synthetic tests: PASS")


if __name__ == "__main__":
    main()
