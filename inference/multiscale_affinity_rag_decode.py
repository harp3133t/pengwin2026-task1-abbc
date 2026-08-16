#!/usr/bin/env python
"""Experimental multi-scale RAG decoder for v3.5 Task-1 affinities.

The deployed v3.5 short-range decoder remains the reference. This candidate
uses exactly the same short-range watershed and RAG construction, then lets
robust mean separation from the 3- and 9-voxel affinity edges veto an otherwise
weak RAG merge. Long edges are never painted as voxelwise fracture ridges.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

try:
    from skimage.graph import merge_hierarchical, rag_boundary
except ImportError:  # pragma: no cover - compatibility with old scikit-image
    from skimage.future.graph import merge_hierarchical, rag_boundary

from agglo_decode import (
    _drop_small,
    _merge_boundary,
    _relabel,
    _weight_boundary,
)


OFFSETS_ZYX = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (9, 0, 0),
    (0, 9, 0),
    (0, 0, 9),
)


def _offset_slices(
    shape: Sequence[int], offset: Sequence[int]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    source = []
    destination = []
    for size, delta in zip(shape, offset):
        size = int(size)
        delta = int(delta)
        if abs(delta) >= size:
            raise ValueError(f"offset {tuple(offset)} invalid for shape {tuple(shape)}")
        if delta > 0:
            source.append(slice(0, -delta))
            destination.append(slice(delta, None))
        elif delta < 0:
            source.append(slice(-delta, None))
            destination.append(slice(0, delta))
        else:
            source.append(slice(None))
            destination.append(slice(None))
    return tuple(source), tuple(destination)


def _group_pair_evidence(
    supervoxels: np.ndarray,
    affinities: np.ndarray,
    channels: Sequence[int],
) -> dict[int, tuple[float, int]]:
    """Return pair-code -> (mean separation, endpoint-pair count)."""
    labels = np.asarray(supervoxels, dtype=np.int32)
    affinity = np.asarray(affinities, dtype=np.float32)
    base = int(labels.max()) + 1
    sum_by_code: dict[int, float] = {}
    count_by_code: dict[int, int] = {}
    shape = tuple(int(value) for value in labels.shape)

    for channel in channels:
        source, destination = _offset_slices(
            shape, OFFSETS_ZYX[int(channel)]
        )
        source_labels = labels[source]
        destination_labels = labels[destination]
        valid = (
            (source_labels > 0)
            & (destination_labels > 0)
            & (source_labels != destination_labels)
        )
        if not bool(valid.any()):
            continue
        left = source_labels[valid].astype(np.int64, copy=False)
        right = destination_labels[valid].astype(np.int64, copy=False)
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        codes = low * int(base) + high
        separation = 1.0 - affinity[int(channel)][source][valid]
        unique_codes, inverse = np.unique(codes, return_inverse=True)
        sums = np.bincount(
            inverse, weights=separation.astype(np.float64, copy=False)
        )
        counts = np.bincount(inverse)
        for code, value, count in zip(unique_codes, sums, counts):
            key = int(code)
            sum_by_code[key] = sum_by_code.get(key, 0.0) + float(value)
            count_by_code[key] = count_by_code.get(key, 0) + int(count)

    return {
        code: (sum_by_code[code] / count, count)
        for code, count in count_by_code.items()
        if count > 0
    }


def decode_affinity_multiscale_rag_veto(
    abbc_probs: np.ndarray,
    affinities: np.ndarray,
    *,
    T: float = 0.75,
    min_vox: int = 250,
    seed_core: float = 0.5,
    seed_bnd: float = 0.20,
    min_range_pairs: int = 32,
    use_mid: bool = True,
    use_long: bool = True,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Decode with short ridges and mid/long merge-veto evidence.

    For every initial adjacent RAG pair, ``score`` is the maximum of:

    - the deployed short-range interface separation;
    - mean separation over all 3-voxel affinity edges joining the pair;
    - mean separation over all 9-voxel affinity edges joining the pair.

    A range participates only with at least ``min_range_pairs`` observations.
    The existing average-linkage update is unchanged after initialization.
    """
    probs = np.asarray(abbc_probs, dtype=np.float32)
    affinity = np.asarray(affinities, dtype=np.float32)
    if probs.ndim != 4 or int(probs.shape[0]) != 4:
        raise ValueError(f"expected ABBC [4,Z,Y,X], got {probs.shape}")
    if affinity.ndim != 4 or int(affinity.shape[0]) != 9:
        raise ValueError(f"expected affinity [9,Z,Y,X], got {affinity.shape}")
    if tuple(probs.shape[1:]) != tuple(affinity.shape[1:]):
        raise ValueError("ABBC/affinity spatial shape mismatch")

    bg, _border, _boundary, core = probs
    foreground = bg < 0.5
    empty_report = {
        "initial_supervoxels": 0,
        "initial_rag_edges": 0,
        "mid_covered_edges": 0,
        "long_covered_edges": 0,
        "mid_veto_edges": 0,
        "long_veto_edges": 0,
        "any_veto_edges": 0,
        "output_fragments": 0,
    }
    if not bool(foreground.any()):
        output = np.zeros(foreground.shape, dtype=np.int32)
        return (output, empty_report) if return_report else output

    short = affinity[:3]
    separation = 1.0 - short.min(axis=0)
    seeds = (core > float(seed_core)) & (separation < float(seed_bnd)) & foreground
    markers, marker_count = ndi.label(seeds)
    if marker_count <= 1:
        distance = ndi.distance_transform_edt(separation < float(seed_bnd)) * foreground
        from skimage.feature import peak_local_max

        peaks = peak_local_max(distance, min_distance=4, labels=foreground)
        point_markers = np.zeros(foreground.shape, dtype=bool)
        if peaks.size:
            point_markers[tuple(peaks.T)] = True
        markers, marker_count = ndi.label(point_markers)
        if marker_count == 0:
            markers, marker_count = ndi.label(foreground)

    supervoxels = watershed(separation.astype(np.float32), markers, mask=foreground)
    initial_supervoxels = int(supervoxels.max())
    if initial_supervoxels <= 1:
        output = (supervoxels > 0).astype(np.int32)
        output, _ = _relabel(output)
        output = _drop_small(output, int(min_vox))
        report = dict(empty_report)
        report["initial_supervoxels"] = initial_supervoxels
        report["output_fragments"] = int(output.max())
        return (output, report) if return_report else output

    graph = rag_boundary(supervoxels, separation.astype(np.float32))
    if 0 in graph.nodes:
        graph.remove_node(0)
    base = initial_supervoxels + 1
    mid_evidence = (
        _group_pair_evidence(supervoxels, affinity, (3, 4, 5))
        if bool(use_mid)
        else {}
    )
    long_evidence = (
        _group_pair_evidence(supervoxels, affinity, (6, 7, 8))
        if bool(use_long)
        else {}
    )

    report = dict(empty_report)
    report["initial_supervoxels"] = initial_supervoxels
    report["initial_rag_edges"] = int(graph.number_of_edges())
    for left, right, data in graph.edges(data=True):
        low = min(int(left), int(right))
        high = max(int(left), int(right))
        code = low * int(base) + high
        short_score = float(data.get("weight", 0.0))
        score = short_score
        mid = mid_evidence.get(code)
        long = long_evidence.get(code)
        mid_score = None
        long_score = None
        if mid is not None and int(mid[1]) >= int(min_range_pairs):
            mid_score = float(mid[0])
            score = max(score, mid_score)
            report["mid_covered_edges"] += 1
            if short_score < float(T) <= mid_score:
                report["mid_veto_edges"] += 1
        if long is not None and int(long[1]) >= int(min_range_pairs):
            long_score = float(long[0])
            score = max(score, long_score)
            report["long_covered_edges"] += 1
            if short_score < float(T) <= long_score:
                report["long_veto_edges"] += 1
        if short_score < float(T) <= score:
            report["any_veto_edges"] += 1
        data["weight"] = float(score)
        # Preserve the deployed short-interface count so hierarchical updates
        # remain the same count-weighted average-linkage operation.
        data["count"] = int(data.get("count", 1))
        data["short_weight"] = short_score
        data["mid_weight"] = mid_score
        data["long_weight"] = long_score

    merged = merge_hierarchical(
        supervoxels,
        graph,
        thresh=float(T),
        rag_copy=False,
        in_place_merge=True,
        merge_func=_merge_boundary,
        weight_func=_weight_boundary,
    )
    output = merged.astype(np.int32) + 1
    output[~foreground] = 0
    output, _ = _relabel(output)
    output = _drop_small(output, int(min_vox))
    report["output_fragments"] = int(output.max())
    return (output, report) if return_report else output
