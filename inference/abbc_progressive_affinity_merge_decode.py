#!/usr/bin/env python
"""Progressive safeguards for affinity healing after aggressive ABBC.

The variants are cumulative:

* A1: require 3/3 affinity ranges when the smaller fragment is 1--5 cm^3;
* A2: A1 plus small-fragment Core and interface-Border vetoes;
* A3: A2 plus one-round mutual-best matching (no cascading merges).
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from abbc_conservative_refine_decode import (
    RANGE_CHANNELS,
    RANGE_NAMES,
    _pair_affinity_evidence,
)
from abbc_full_refine_decode import (
    _adjacent_label_pairs,
    _all_label_boxes,
    _candidate_interface,
    _dice_distance,
    _remap_adjacency_after_merge,
    _relabel,
)


def _candidate_evidence(
    labels: np.ndarray,
    abbc_probs: np.ndarray,
    affinities: np.ndarray,
    hard_boundary: np.ndarray,
    hard_core: np.ndarray,
    label_boxes: dict[int, tuple[slice, slice, slice]],
    left: int,
    right: int,
    *,
    voxel_volume_mm3: float,
    small_fragment_max_mm3: float,
    min_candidate_fragment_mm3: float,
    min_boundary_improvement: float,
    affinity_connection_threshold: float,
    large_required_ranges: int,
    small_required_ranges: int,
    min_pair_observations: int,
    enable_small_veto: bool,
    small_core_fraction_veto: float,
    small_border_probability_veto: float,
    adjacency_radius: int,
    interface_radius: int,
) -> tuple[dict | None, str]:
    candidate = _candidate_interface(
        labels,
        left,
        right,
        adjacency_radius=int(adjacency_radius),
        interface_radius=int(interface_radius),
        label_boxes=label_boxes,
    )
    if candidate is None:
        return None, "interface"
    crop, interface = candidate
    predicted = hard_boundary[crop]
    split_distance = _dice_distance(interface, predicted)
    merged_distance = _dice_distance(
        np.zeros(interface.shape, dtype=bool), predicted
    )
    improvement = float(split_distance - merged_distance)
    if improvement <= float(min_boundary_improvement):
        return None, "boundary"

    left_voxels = int((labels == int(left)).sum())
    right_voxels = int((labels == int(right)).sum())
    left_volume = float(left_voxels) * float(voxel_volume_mm3)
    right_volume = float(right_voxels) * float(voxel_volume_mm3)
    small_volume = min(left_volume, right_volume)
    large_volume = max(left_volume, right_volume)
    if small_volume < float(min_candidate_fragment_mm3):
        return None, "candidate_size"
    small_label = int(left) if left_voxels <= right_voxels else int(right)
    small_voxels = min(left_voxels, right_voxels)
    local_labels = labels[crop]
    left_core = int(
        np.logical_and(local_labels == int(left), hard_core[crop]).sum()
    )
    right_core = int(
        np.logical_and(local_labels == int(right), hard_core[crop]).sum()
    )
    small_core = left_core if small_label == int(left) else right_core
    small_core_fraction = float(small_core) / float(max(1, small_voxels))
    boundary_mean = float(abbc_probs[2][crop][interface].mean())
    border_mean = float(abbc_probs[1][crop][interface].mean())
    is_small = small_volume < float(small_fragment_max_mm3)
    if bool(enable_small_veto) and is_small:
        if small_core_fraction >= float(small_core_fraction_veto):
            return None, "core_veto"
        if border_mean >= float(small_border_probability_veto):
            return None, "border_veto"

    range_evidence = {}
    supported = 0
    supported_values = []
    local_affinity = affinities[:, crop[0], crop[1], crop[2]]
    for name, channels in zip(RANGE_NAMES, RANGE_CHANNELS):
        mean, count = _pair_affinity_evidence(
            local_labels, local_affinity, left, right, channels
        )
        range_evidence[name] = {
            "mean_affinity": mean,
            "count": int(count),
        }
        if (
            mean is not None
            and count >= int(min_pair_observations)
            and mean >= float(affinity_connection_threshold)
        ):
            supported += 1
            supported_values.append(float(mean))
    required = int(small_required_ranges if is_small else large_required_ranges)
    if supported < required:
        return None, "affinity"
    signal_score = float(np.mean(supported_values))
    score = float(signal_score + improvement)
    return {
        "left": int(left),
        "right": int(right),
        "score": score,
        "signal_score": signal_score,
        "boundary_dice_distance_improvement": improvement,
        "split_distance": float(split_distance),
        "merged_distance": float(merged_distance),
        "boundary_probability": boundary_mean,
        "border_probability": border_mean,
        "left_voxels": left_voxels,
        "right_voxels": right_voxels,
        "small_volume_mm3": small_volume,
        "large_volume_mm3": large_volume,
        "is_small_candidate": bool(is_small),
        "small_label": small_label,
        "left_core_voxels": left_core,
        "right_core_voxels": right_core,
        "small_core_voxels": small_core,
        "small_core_fraction": small_core_fraction,
        "required_affinity_ranges": required,
        "supported_affinity_ranges": int(supported),
        "affinity_evidence": range_evidence,
    }, "accepted"


def apply_progressive_affinity_merge(
    labels: np.ndarray,
    abbc_probs: np.ndarray,
    affinities: np.ndarray,
    *,
    preprocessed_spacing_zyx: Sequence[float],
    small_strict: bool = True,
    enable_small_veto: bool = False,
    mutual_best_single_round: bool = False,
    min_candidate_fragment_mm3: float = 1000.0,
    small_fragment_max_mm3: float = 5000.0,
    min_boundary_improvement: float = -0.25,
    affinity_connection_threshold: float = 0.55,
    large_required_ranges: int = 2,
    small_required_ranges: int = 3,
    min_pair_observations: int = 8,
    small_core_fraction_veto: float = 0.25,
    small_border_probability_veto: float = 0.40,
    adjacency_radius: int = 2,
    interface_radius: int = 3,
    max_merges: int = 100,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    output = np.asarray(labels, dtype=np.int32).copy()
    probs = np.asarray(abbc_probs, dtype=np.float32)
    affinity = np.asarray(affinities, dtype=np.float32)
    spacing = tuple(float(value) for value in preprocessed_spacing_zyx)
    if probs.shape != (4, *output.shape):
        raise ValueError("ABBC/instance spatial shape mismatch")
    if affinity.shape != (9, *output.shape):
        raise ValueError("affinity/instance spatial shape mismatch")
    if len(spacing) != 3:
        raise ValueError(f"expected 3D spacing, got {spacing}")
    voxel_volume_mm3 = float(np.prod(spacing))
    hard = np.argmax(probs, axis=0).astype(np.uint8, copy=False)
    hard_boundary = hard == 2
    hard_core = hard == 3
    initial_fragments = int(np.unique(output[output > 0]).size)
    events: list[dict] = []
    counters = {
        "pairs_examined": 0,
        "rejected_interface": 0,
        "rejected_boundary": 0,
        "rejected_candidate_size": 0,
        "rejected_core_veto": 0,
        "rejected_border_veto": 0,
        "rejected_affinity": 0,
        "eligible_pairs": 0,
        "accepted": 0,
    }

    def examine(
        left: int,
        right: int,
        boxes: dict[int, tuple[slice, slice, slice]],
    ) -> dict | None:
        counters["pairs_examined"] += 1
        row, reason = _candidate_evidence(
            output,
            probs,
            affinity,
            hard_boundary,
            hard_core,
            boxes,
            left,
            right,
            voxel_volume_mm3=voxel_volume_mm3,
            small_fragment_max_mm3=float(small_fragment_max_mm3),
            min_candidate_fragment_mm3=float(min_candidate_fragment_mm3),
            min_boundary_improvement=float(min_boundary_improvement),
            affinity_connection_threshold=float(affinity_connection_threshold),
            large_required_ranges=int(large_required_ranges),
            small_required_ranges=(
                int(small_required_ranges)
                if bool(small_strict)
                else int(large_required_ranges)
            ),
            min_pair_observations=int(min_pair_observations),
            enable_small_veto=bool(enable_small_veto),
            small_core_fraction_veto=float(small_core_fraction_veto),
            small_border_probability_veto=float(
                small_border_probability_veto
            ),
            adjacency_radius=int(adjacency_radius),
            interface_radius=int(interface_radius),
        )
        if row is None:
            counters[f"rejected_{reason}"] += 1
        else:
            counters["eligible_pairs"] += 1
        return row

    adjacent_pairs = _adjacent_label_pairs(output, int(adjacency_radius))
    if bool(mutual_best_single_round):
        boxes = _all_label_boxes(output)
        eligible = []
        for left, right in sorted(adjacent_pairs):
            row = examine(left, right, boxes)
            if row is not None:
                eligible.append(row)
        best_by_label: dict[int, dict] = {}
        for row in eligible:
            pair = (int(row["left"]), int(row["right"]))
            for label in pair:
                current = best_by_label.get(label)
                if current is None or (
                    float(row["score"]), -pair[0], -pair[1]
                ) > (
                    float(current["score"]),
                    -int(current["left"]),
                    -int(current["right"]),
                ):
                    best_by_label[label] = row
        accepted = [
            row
            for row in eligible
            if best_by_label.get(int(row["left"])) is row
            and best_by_label.get(int(row["right"])) is row
        ]
        accepted.sort(
            key=lambda row: (
                -float(row["score"]),
                int(row["left"]),
                int(row["right"]),
            )
        )
        for row in accepted:
            output[output == int(row["right"])] = int(row["left"])
            events.append(row)
        counters["accepted"] = len(accepted)
        output, _ = _relabel(output)
    else:
        for _ in range(int(max_merges)):
            values = [
                int(value) for value in np.unique(output) if int(value) > 0
            ]
            boxes = _all_label_boxes(output)
            best = None
            for left, right in combinations(values, 2):
                if (left, right) not in adjacent_pairs:
                    continue
                row = examine(left, right, boxes)
                if row is None:
                    continue
                if best is None or (
                    float(row["score"]), -int(left), -int(right)
                ) > (
                    float(best["score"]),
                    -int(best["left"]),
                    -int(best["right"]),
                ):
                    best = row
            if best is None:
                break
            left = int(best["left"])
            right = int(best["right"])
            output[output == right] = left
            adjacent_pairs = _remap_adjacency_after_merge(
                adjacent_pairs, values, left, right
            )
            output, _ = _relabel(output)
            events.append(best)
            counters["accepted"] += 1

    output, output_fragments = _relabel(output)
    report = {
        "initial_fragments": initial_fragments,
        "output_fragments": output_fragments,
        "voxel_volume_mm3": voxel_volume_mm3,
        "small_strict": bool(small_strict),
        "enable_small_veto": bool(enable_small_veto),
        "mutual_best_single_round": bool(mutual_best_single_round),
        "thresholds": {
            "min_candidate_fragment_mm3": float(
                min_candidate_fragment_mm3
            ),
            "small_fragment_max_mm3": float(small_fragment_max_mm3),
            "min_boundary_improvement": float(min_boundary_improvement),
            "affinity_connection_threshold": float(
                affinity_connection_threshold
            ),
            "large_required_ranges": int(large_required_ranges),
            "small_required_ranges": int(small_required_ranges),
            "min_pair_observations": int(min_pair_observations),
            "small_core_fraction_veto": float(small_core_fraction_veto),
            "small_border_probability_veto": float(
                small_border_probability_veto
            ),
        },
        **counters,
        "events": events,
    }
    return (output, report) if return_report else output
