#!/usr/bin/env python
"""Conservative ABBC refinement for the frozen v3.5 Task-1 model.

This decoder targets the failure observed in the full-ABBC experiment: ABBC
splitting reduced merge errors but increased the decoded fragment count from
436 to 632.  A split is now accepted only when it creates exactly two viable
pieces, both pieces contain predicted core, and at least two affinity ranges
support the same fracture band.  A second optional stage merges unsupported
interfaces using soft ABBC and pairwise 1/3/9 affinity evidence.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball
from skimage.segmentation import watershed

from abbc_full_refine_decode import (
    CONNECTIVITY_26,
    _bbox,
    _candidate_interface,
    _merge_healing,
    _relabel,
)
RANGE_CHANNELS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
RANGE_NAMES = ("1_voxel", "3_voxel", "9_voxel")
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


def _split_affinity_evidence(
    affinities: np.ndarray,
    evidence_mask: np.ndarray,
    crop: tuple[slice, slice, slice],
) -> dict[str, float | int]:
    local_affinity = np.asarray(affinities[:, crop[0], crop[1], crop[2]])
    result: dict[str, float | int] = {"evidence_voxels": int(evidence_mask.sum())}
    for name, channels in zip(RANGE_NAMES, RANGE_CHANNELS):
        separation = np.max(1.0 - local_affinity[list(channels)], axis=0)
        result[name] = (
            float(separation[evidence_mask].mean())
            if bool(evidence_mask.any())
            else 0.0
        )
    return result


def _conservative_split_pass(
    labels: np.ndarray,
    abbc_probs: np.ndarray,
    affinities: np.ndarray,
    *,
    voxel_volume_mm3: float,
    relative_error_threshold: float,
    absolute_error_threshold: int,
    min_split_piece_mm3: float,
    min_core_voxels_per_piece: int,
    affinity_separation_threshold: float,
    affinity_required_ranges: int,
    enable_small_candidate_branch: bool,
    small_residual_piece_min_mm3: float,
    small_final_piece_min_mm3: float,
    small_final_piece_max_mm3: float,
    small_min_core_voxels_per_piece: int,
    small_min_core_fraction_per_piece: float | None,
    small_affinity_separation_threshold: float,
    small_affinity_required_ranges: int,
    split_dilation_radius: int,
    max_split_dilations: int,
    interface_radius: int,
) -> tuple[np.ndarray, list[dict], dict[str, int]]:
    output = np.asarray(labels, dtype=np.int32).copy()
    probs = np.asarray(abbc_probs, dtype=np.float32)
    affinity = np.asarray(affinities, dtype=np.float32)
    hard = np.argmax(probs, axis=0).astype(np.uint8, copy=False)
    predicted_boundary = hard == 2
    predicted_core = hard == 3
    events: list[dict] = []
    counters = {
        "triggered": 0,
        "rejected_affinity": 0,
        "rejected_multiway": 0,
        "rejected_size": 0,
        "rejected_core": 0,
        "rejected_small_affinity": 0,
        "rejected_small_core": 0,
        "rejected_small_final_size": 0,
        "no_valid_cut": 0,
        "accepted": 0,
        "accepted_large": 0,
        "accepted_small": 0,
    }

    for label in [int(value) for value in np.unique(output) if int(value) > 0]:
        instance = output == label
        crop = _bbox(instance, pad=max(10, int(interface_radius) + 2))
        if crop is None:
            continue
        local_labels = output[crop]
        local_instance = local_labels == label
        local_boundary = predicted_boundary[crop]
        local_other = (local_labels > 0) & (local_labels != label)
        if bool(local_other.any()):
            current_interface = (
                ndi.binary_dilation(
                    local_other, structure=ball(int(interface_radius))
                )
                & ndi.binary_dilation(
                    local_instance, structure=ball(int(interface_radius))
                )
                & local_instance
            )
        else:
            current_interface = np.zeros(local_instance.shape, dtype=bool)

        missing_boundary = local_boundary & local_instance & ~current_interface
        missing_boundary = ndi.binary_erosion(
            missing_boundary, structure=ball(1)
        )
        predicted_count = int((local_boundary & local_instance).sum())
        missing_count = int(missing_boundary.sum())
        relative_error = (
            float(missing_count) / float(predicted_count)
            if predicted_count > 0
            else 0.0
        )
        if not (
            relative_error > float(relative_error_threshold)
            or missing_count > int(absolute_error_threshold)
        ):
            continue
        counters["triggered"] += 1

        affinity_evidence = _split_affinity_evidence(
            affinity, missing_boundary, crop
        )
        supported_ranges = sum(
            float(affinity_evidence[name])
            >= float(affinity_separation_threshold)
            for name in RANGE_NAMES
        )
        small_supported_ranges = sum(
            float(affinity_evidence[name])
            >= float(small_affinity_separation_threshold)
            for name in RANGE_NAMES
        )

        cut_band = local_boundary & local_instance
        accepted = None
        accepted_iteration = None
        accepted_sizes = None
        accepted_final_sizes = None
        accepted_core_counts = None
        accepted_core_requirements = None
        accepted_branch = None
        accepted_grown = None
        rejection = "no_valid_cut"
        for iteration in range(1, int(max_split_dilations) + 1):
            cut_band = ndi.binary_dilation(
                cut_band, structure=ball(int(split_dilation_radius))
            ) & local_instance
            components, n_components = ndi.label(
                local_instance & ~cut_band, structure=CONNECTIVITY_26
            )
            if n_components < 2:
                continue
            # Removing more voxels cannot turn a multiway cut into a binary
            # cut, so reject immediately once more than two pieces appear.
            if n_components > 2:
                rejection = "rejected_multiway"
                break
            sizes = np.bincount(components.ravel())[1:]
            piece_volumes = sizes.astype(np.float64) * float(voxel_volume_mm3)
            residual_minimum = (
                float(small_residual_piece_min_mm3)
                if bool(enable_small_candidate_branch)
                else float(min_split_piece_mm3)
            )
            if float(piece_volumes.min()) < residual_minimum:
                rejection = "rejected_size"
                break
            local_core = predicted_core[crop] & local_instance
            core_counts = [
                int(np.logical_and(components == component, local_core).sum())
                for component in (1, 2)
            ]
            provisional_markers = np.zeros(components.shape, dtype=np.int32)
            provisional_markers[components == 1] = 1
            provisional_markers[components == 2] = 2
            provisional_grown = watershed(
                probs[2][crop].astype(np.float32, copy=False),
                markers=provisional_markers,
                mask=local_instance,
                connectivity=CONNECTIVITY_26,
            )
            final_sizes = np.asarray(
                [
                    int((provisional_grown == component).sum())
                    for component in (1, 2)
                ],
                dtype=np.int64,
            )
            final_volumes = (
                final_sizes.astype(np.float64) * float(voxel_volume_mm3)
            )
            use_small_branch = bool(enable_small_candidate_branch) and (
                float(final_volumes.min())
                <= float(small_final_piece_max_mm3)
            )
            if use_small_branch:
                if float(final_volumes.min()) < float(small_final_piece_min_mm3):
                    rejection = "rejected_small_final_size"
                    break
                if small_min_core_fraction_per_piece is None:
                    core_requirements = [
                        int(small_min_core_voxels_per_piece),
                        int(small_min_core_voxels_per_piece),
                    ]
                else:
                    core_requirements = [
                        max(
                            1,
                            int(
                                np.ceil(
                                    float(small_min_core_fraction_per_piece)
                                    * float(final_size)
                                )
                            ),
                        )
                        for final_size in final_sizes
                    ]
                if any(
                    int(count) < int(required)
                    for count, required in zip(core_counts, core_requirements)
                ):
                    rejection = "rejected_small_core"
                    break
                if small_supported_ranges < int(small_affinity_required_ranges):
                    rejection = "rejected_small_affinity"
                    break
                branch = "small"
            else:
                if float(piece_volumes.min()) < float(min_split_piece_mm3):
                    rejection = "rejected_size"
                    break
                if min(core_counts) < int(min_core_voxels_per_piece):
                    rejection = "rejected_core"
                    break
                if supported_ranges < int(affinity_required_ranges):
                    rejection = "rejected_affinity"
                    break
                branch = "large"
            accepted = components
            accepted_iteration = iteration
            accepted_sizes = sizes
            accepted_final_sizes = final_sizes
            accepted_core_counts = core_counts
            accepted_core_requirements = core_requirements if use_small_branch else [
                int(min_core_voxels_per_piece),
                int(min_core_voxels_per_piece),
            ]
            accepted_branch = branch
            accepted_grown = provisional_grown
            break
        if accepted is None:
            counters[rejection] += 1
            continue

        next_label = int(output.max()) + 1
        grown = np.zeros(accepted_grown.shape, dtype=np.int32)
        grown[accepted_grown == 1] = next_label
        grown[accepted_grown == 2] = next_label + 1
        local_output = output[crop]
        local_output[local_instance] = grown[local_instance]
        output[crop] = local_output
        counters["accepted"] += 1
        counters[f"accepted_{accepted_branch}"] += 1
        events.append(
            {
                "source_label": label,
                "new_components": 2,
                "missing_boundary_voxels": missing_count,
                "relative_error": relative_error,
                "dilation_iterations": int(accepted_iteration),
                "piece_voxels": [int(value) for value in accepted_sizes],
                "piece_volumes_mm3": [
                    float(value) * float(voxel_volume_mm3)
                    for value in accepted_sizes
                ],
                "final_piece_voxels": [
                    int(value) for value in accepted_final_sizes
                ],
                "final_piece_volumes_mm3": [
                    float(value) * float(voxel_volume_mm3)
                    for value in accepted_final_sizes
                ],
                "candidate_branch": accepted_branch,
                "core_voxels": accepted_core_counts,
                "required_core_voxels": accepted_core_requirements,
                "core_fractions": [
                    float(count) / float(max(1, final_size))
                    for count, final_size in zip(
                        accepted_core_counts, accepted_final_sizes
                    )
                ],
                "affinity_evidence": affinity_evidence,
                "supported_affinity_ranges": int(supported_ranges),
                "small_supported_affinity_ranges": int(
                    small_supported_ranges
                ),
            }
        )

    return _relabel(output)[0], events, counters


def _pair_affinity_evidence(
    labels: np.ndarray,
    affinities: np.ndarray,
    left: int,
    right: int,
    channels: Sequence[int],
) -> tuple[float | None, int]:
    values: list[np.ndarray] = []
    shape = tuple(int(value) for value in labels.shape)
    for channel in channels:
        source, destination = _offset_slices(shape, OFFSETS_ZYX[int(channel)])
        source_labels = labels[source]
        destination_labels = labels[destination]
        cross = (
            ((source_labels == int(left)) & (destination_labels == int(right)))
            | ((source_labels == int(right)) & (destination_labels == int(left)))
        )
        if bool(cross.any()):
            values.append(np.asarray(affinities[int(channel)][source][cross]))
    if not values:
        return None, 0
    joined = np.concatenate(values).astype(np.float32, copy=False)
    return float(joined.mean()), int(joined.size)


def apply_soft_abbc_affinity_merge(
    labels: np.ndarray,
    abbc_probs: np.ndarray,
    affinities: np.ndarray,
    *,
    boundary_probability_max: float = 0.30,
    affinity_connection_threshold: float = 0.55,
    affinity_required_ranges: int = 2,
    min_pair_observations: int = 8,
    adjacency_radius: int = 2,
    interface_radius: int = 3,
    max_merges: int = 100,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Merge interfaces unsupported by soft Boundary and multi-range affinity."""
    output = np.asarray(labels, dtype=np.int32).copy()
    probs = np.asarray(abbc_probs, dtype=np.float32)
    affinity = np.asarray(affinities, dtype=np.float32)
    events: list[dict] = []
    counters = {
        "pairs_examined": 0,
        "rejected_boundary": 0,
        "rejected_affinity": 0,
        "accepted": 0,
    }

    for _ in range(int(max_merges)):
        values = [int(value) for value in np.unique(output) if int(value) > 0]
        best = None
        for left, right in combinations(values, 2):
            candidate = _candidate_interface(
                output,
                left,
                right,
                adjacency_radius=int(adjacency_radius),
                interface_radius=int(interface_radius),
            )
            if candidate is None:
                continue
            counters["pairs_examined"] += 1
            crop, interface = candidate
            boundary_mean = float(probs[2][crop][interface].mean())
            border_mean = float(probs[1][crop][interface].mean())
            if not (
                boundary_mean <= float(boundary_probability_max)
                and border_mean > boundary_mean
            ):
                counters["rejected_boundary"] += 1
                continue

            range_evidence = {}
            supported = 0
            supported_values = []
            local_labels = output[crop]
            local_affinity = affinity[:, crop[0], crop[1], crop[2]]
            for name, channels in zip(RANGE_NAMES, RANGE_CHANNELS):
                mean, count = _pair_affinity_evidence(
                    local_labels, local_affinity, left, right, channels
                )
                range_evidence[name] = {"mean_affinity": mean, "count": count}
                if (
                    mean is not None
                    and count >= int(min_pair_observations)
                    and mean >= float(affinity_connection_threshold)
                ):
                    supported += 1
                    supported_values.append(float(mean))
            if supported < int(affinity_required_ranges):
                counters["rejected_affinity"] += 1
                continue
            score = (
                float(border_mean - boundary_mean)
                + float(np.mean(supported_values))
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    int(left),
                    int(right),
                    boundary_mean,
                    border_mean,
                    range_evidence,
                    supported,
                )
        if best is None:
            break
        (
            score,
            left,
            right,
            boundary_mean,
            border_mean,
            range_evidence,
            supported,
        ) = best
        output[output == right] = left
        output, _ = _relabel(output)
        counters["accepted"] += 1
        events.append(
            {
                "left": left,
                "right": right,
                "score": score,
                "boundary_probability": boundary_mean,
                "border_probability": border_mean,
                "supported_affinity_ranges": supported,
                "affinity_evidence": range_evidence,
            }
        )

    output, fragments = _relabel(output)
    report = {
        **counters,
        "output_fragments": fragments,
        "events": events,
    }
    return (output, report) if return_report else output


def refine_instances_conservatively(
    labels: np.ndarray,
    abbc_probs: np.ndarray,
    affinities: np.ndarray,
    *,
    preprocessed_spacing_zyx: Sequence[float],
    split_passes: int = 2,
    relative_error_threshold: float = 0.10,
    absolute_error_threshold: int = 100,
    min_split_piece_mm3: float = 1000.0,
    min_core_voxels_per_piece: int = 50,
    split_affinity_separation_threshold: float = 0.50,
    split_affinity_required_ranges: int = 2,
    enable_small_candidate_branch: bool = False,
    small_residual_piece_min_mm3: float = 250.0,
    small_final_piece_min_mm3: float = 1000.0,
    small_final_piece_max_mm3: float = 5000.0,
    small_min_core_voxels_per_piece: int = 250,
    small_min_core_fraction_per_piece: float | None = None,
    small_affinity_separation_threshold: float = 0.75,
    small_affinity_required_ranges: int = 3,
    split_dilation_radius: int = 3,
    max_split_dilations: int = 10,
    adjacency_radius: int = 2,
    interface_radius: int = 3,
    hard_merge_margin: float = 0.05 / 3.0,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Apply binary/core/affinity-gated splitting then the reference hard merge."""
    output = np.asarray(labels, dtype=np.int32).copy()
    initial_fragments = int(np.unique(output[output > 0]).size)
    spacing = tuple(float(value) for value in preprocessed_spacing_zyx)
    if len(spacing) != 3:
        raise ValueError(f"expected 3D spacing, got {spacing}")
    voxel_volume_mm3 = float(np.prod(spacing))
    pass_reports = []
    split_events = []
    for pass_index in range(int(split_passes)):
        output, events, counters = _conservative_split_pass(
            output,
            abbc_probs,
            affinities,
            voxel_volume_mm3=voxel_volume_mm3,
            relative_error_threshold=float(relative_error_threshold),
            absolute_error_threshold=int(absolute_error_threshold),
            min_split_piece_mm3=float(min_split_piece_mm3),
            min_core_voxels_per_piece=int(min_core_voxels_per_piece),
            affinity_separation_threshold=float(
                split_affinity_separation_threshold
            ),
            affinity_required_ranges=int(split_affinity_required_ranges),
            enable_small_candidate_branch=bool(enable_small_candidate_branch),
            small_residual_piece_min_mm3=float(
                small_residual_piece_min_mm3
            ),
            small_final_piece_min_mm3=float(small_final_piece_min_mm3),
            small_final_piece_max_mm3=float(small_final_piece_max_mm3),
            small_min_core_voxels_per_piece=int(
                small_min_core_voxels_per_piece
            ),
            small_min_core_fraction_per_piece=(
                None
                if small_min_core_fraction_per_piece is None
                else float(small_min_core_fraction_per_piece)
            ),
            small_affinity_separation_threshold=float(
                small_affinity_separation_threshold
            ),
            small_affinity_required_ranges=int(
                small_affinity_required_ranges
            ),
            split_dilation_radius=int(split_dilation_radius),
            max_split_dilations=int(max_split_dilations),
            interface_radius=int(interface_radius),
        )
        for event in events:
            event["pass"] = pass_index + 1
        split_events.extend(events)
        pass_reports.append({"pass": pass_index + 1, **counters})
        if not events:
            # A second identical pass cannot create new candidates when the
            # partition did not change in the first pass.
            break

    fragments_after_split = int(np.unique(output[output > 0]).size)
    hard = np.argmax(np.asarray(abbc_probs), axis=0).astype(np.uint8)
    output, hard_merge_events = _merge_healing(
        output,
        hard,
        adjacency_radius=int(adjacency_radius),
        interface_radius=int(interface_radius),
        merge_margin=float(hard_merge_margin),
        max_merges=100,
    )
    output, final_fragments = _relabel(output)
    report = {
        "initial_fragments": initial_fragments,
        "fragments_after_split": fragments_after_split,
        "output_fragments": final_fragments,
        "voxel_volume_mm3": voxel_volume_mm3,
        "accepted_splits": len(split_events),
        "accepted_hard_merges": len(hard_merge_events),
        "split_pass_reports": pass_reports,
        "split_events": split_events,
        "hard_merge_events": hard_merge_events,
    }
    return (output, report) if return_report else output
