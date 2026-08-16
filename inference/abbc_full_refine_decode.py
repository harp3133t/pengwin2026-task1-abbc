#!/usr/bin/env python
"""Original-style four-class ABBC healing for the v3.5 Task-1 decoder.

The initial partition is deliberately left to the frozen 1/3/9-voxel
multi-scale affinity decoder.  This module then consumes all four predicted
ABBC classes to perform the two operations used by the MIC-DKFZ method:

* split an instance when a predicted fracture boundary is missing from the
  current partition;
* merge adjacent instances when the current interface is unsupported by the
  predicted fracture boundary.

Class mapping in the local model is ``0 background, 1 outer border,
2 fracture boundary, 3 core``.  The public reference implementation uses
``0 background, 1 outer shell, 2 core, 3 fracture band``.  The behavior here
is a clean-room adaptation to the local class mapping and does not change
model weights or training targets.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball
from skimage.segmentation import watershed


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


def _relabel(labels: np.ndarray) -> tuple[np.ndarray, int]:
    output = np.zeros(labels.shape, dtype=np.int32)
    values = [int(value) for value in np.unique(labels) if int(value) > 0]
    for new_value, old_value in enumerate(values, start=1):
        output[labels == old_value] = new_value
    return output, len(values)


def _bbox(mask: np.ndarray, pad: int = 0) -> tuple[slice, slice, slice] | None:
    coordinates = np.where(mask)
    if not coordinates[0].size:
        return None
    slices = []
    for axis, values in enumerate(coordinates):
        start = max(0, int(values.min()) - int(pad))
        stop = min(int(mask.shape[axis]), int(values.max()) + 1 + int(pad))
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def _all_label_boxes(
    labels: np.ndarray,
) -> dict[int, tuple[slice, slice, slice]]:
    """Compute every positive-label bounding box in one volume scan."""
    boxes = ndi.find_objects(np.asarray(labels, dtype=np.int32))
    return {
        index + 1: box
        for index, box in enumerate(boxes)
        if box is not None
    }


def _adjacent_label_pairs(
    labels: np.ndarray,
    radius: int,
) -> set[tuple[int, int]]:
    """Return pairs satisfying the exact spherical-dilation adjacency test."""
    values = np.asarray(labels, dtype=np.int32)
    radius = int(radius)
    footprint = ball(radius)
    center = np.asarray(footprint.shape, dtype=np.int64) // 2
    offsets = np.argwhere(footprint) - center[None]
    pairs: set[tuple[int, int]] = set()
    base = int(values.max()) + 1
    if base <= 1:
        return pairs
    for raw_offset in offsets:
        offset = tuple(int(value) for value in raw_offset)
        if offset <= (0, 0, 0):
            # The footprint is symmetric; inspect one direction per offset.
            continue
        source = []
        destination = []
        for size, delta in zip(values.shape, offset):
            if delta > 0:
                source.append(slice(0, int(size) - delta))
                destination.append(slice(delta, int(size)))
            elif delta < 0:
                source.append(slice(-delta, int(size)))
                destination.append(slice(0, int(size) + delta))
            else:
                source.append(slice(None))
                destination.append(slice(None))
        left_values = values[tuple(source)]
        right_values = values[tuple(destination)]
        valid = (
            (left_values > 0)
            & (right_values > 0)
            & (left_values != right_values)
        )
        if not bool(valid.any()):
            continue
        low = np.minimum(left_values[valid], right_values[valid]).astype(
            np.int64, copy=False
        )
        high = np.maximum(left_values[valid], right_values[valid]).astype(
            np.int64, copy=False
        )
        for code in np.unique(low * base + high):
            pairs.add((int(code // base), int(code % base)))
    return pairs


def _remap_adjacency_after_merge(
    pairs: set[tuple[int, int]],
    values: list[int],
    left: int,
    right: int,
) -> set[tuple[int, int]]:
    """Update an adjacency graph through merge + sorted consecutive relabel."""
    remaining = [int(value) for value in values if int(value) != int(right)]
    mapping = {old: new for new, old in enumerate(remaining, start=1)}
    output: set[tuple[int, int]] = set()
    for first, second in pairs:
        first = int(left) if int(first) == int(right) else int(first)
        second = int(left) if int(second) == int(right) else int(second)
        if first == second:
            continue
        mapped = sorted((mapping[first], mapping[second]))
        output.add((int(mapped[0]), int(mapped[1])))
    return output


def _dice_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    denominator = int(left.sum()) + int(right.sum())
    if denominator == 0:
        return 0.0
    intersection = int(np.logical_and(left, right).sum())
    return 1.0 - (2.0 * float(intersection) / float(denominator))


def _split_healing_pass(
    labels: np.ndarray,
    hard_abbc: np.ndarray,
    boundary_probability: np.ndarray,
    *,
    relative_error_threshold: float,
    absolute_error_threshold: int,
    min_split_piece_voxels: int,
    split_dilation_radius: int,
    max_split_dilations: int,
    interface_radius: int,
) -> tuple[np.ndarray, list[dict]]:
    output = np.asarray(labels, dtype=np.int32).copy()
    predicted_boundary = hard_abbc == 2
    interventions: list[dict] = []

    for label in [int(value) for value in np.unique(output) if int(value) > 0]:
        instance = output == label
        crop = _bbox(instance, pad=max(10, int(interface_radius) + 2))
        if crop is None:
            continue
        local_labels = output[crop]
        local_instance = local_labels == label
        local_predicted = predicted_boundary[crop]
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

        missing_boundary = local_predicted & local_instance & ~current_interface
        missing_boundary = ndi.binary_erosion(
            missing_boundary, structure=ball(1)
        )
        predicted_count = int((local_predicted & local_instance).sum())
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

        cut_band = local_predicted & local_instance
        accepted = None
        accepted_iteration = None
        accepted_sizes = None
        for iteration in range(1, int(max_split_dilations) + 1):
            cut_band = ndi.binary_dilation(
                cut_band, structure=ball(int(split_dilation_radius))
            ) & local_instance
            components, n_components = ndi.label(
                local_instance & ~cut_band, structure=CONNECTIVITY_26
            )
            if n_components < 2:
                continue
            sizes = np.bincount(components.ravel())[1:]
            sizes = np.sort(sizes)[::-1]
            if len(sizes) >= 2 and int(sizes[1]) > int(min_split_piece_voxels):
                accepted = components
                accepted_iteration = iteration
                accepted_sizes = sizes
                break
        if accepted is None:
            continue

        next_label = int(output.max()) + 1
        markers = np.zeros(accepted.shape, dtype=np.int32)
        for component in range(1, int(accepted.max()) + 1):
            markers[accepted == component] = next_label
            next_label += 1
        grown = watershed(
            boundary_probability[crop].astype(np.float32, copy=False),
            markers=markers,
            mask=local_instance,
            connectivity=CONNECTIVITY_26,
        )
        local_output = output[crop]
        local_output[local_instance] = grown[local_instance]
        output[crop] = local_output
        interventions.append(
            {
                "source_label": label,
                "new_components": int(accepted.max()),
                "missing_boundary_voxels": missing_count,
                "relative_error": relative_error,
                "dilation_iterations": int(accepted_iteration),
                "second_piece_voxels": int(accepted_sizes[1]),
            }
        )

    return _relabel(output)[0], interventions


def _candidate_interface(
    labels: np.ndarray,
    left: int,
    right: int,
    *,
    adjacency_radius: int,
    interface_radius: int,
    label_boxes: dict[int, tuple[slice, slice, slice]] | None = None,
) -> tuple[tuple[slice, slice, slice], np.ndarray] | None:
    if label_boxes is None:
        left_box = _bbox(labels == int(left))
        right_box = _bbox(labels == int(right))
    else:
        left_box = label_boxes.get(int(left))
        right_box = label_boxes.get(int(right))
    if left_box is None or right_box is None:
        return None
    # Exact early rejection: if the closest possible voxel coordinates from
    # the two bounding boxes are farther apart than the dilation radius on any
    # axis, the original binary-dilation intersection must be empty.  This
    # avoids dilating a huge crop spanning distant fragments while preserving
    # the original decision for every pair that can possibly be adjacent.
    for left_slice, right_slice in zip(left_box, right_box):
        if int(left_slice.stop) <= int(right_slice.start):
            axis_distance = int(right_slice.start) - int(left_slice.stop) + 1
        elif int(right_slice.stop) <= int(left_slice.start):
            axis_distance = int(left_slice.start) - int(right_slice.stop) + 1
        else:
            axis_distance = 0
        if axis_distance > int(adjacency_radius):
            return None
    pad = max(20, int(interface_radius) + 2)
    crop = tuple(
        slice(
            max(0, min(int(a.start), int(b.start)) - pad),
            min(int(labels.shape[axis]), max(int(a.stop), int(b.stop)) + pad),
        )
        for axis, (a, b) in enumerate(zip(left_box, right_box))
    )
    local_left = labels[crop] == int(left)
    local_right = labels[crop] == int(right)
    if not bool(
        np.logical_and(
            ndi.binary_dilation(
                local_left, structure=ball(int(adjacency_radius))
            ),
            local_right,
        ).any()
    ):
        return None
    interface = ndi.binary_dilation(
        local_left, structure=ball(int(interface_radius))
    ) & ndi.binary_dilation(
        local_right, structure=ball(int(interface_radius))
    )
    if not bool(interface.any()):
        return None
    return crop, interface


def _merge_healing(
    labels: np.ndarray,
    hard_abbc: np.ndarray,
    *,
    adjacency_radius: int,
    interface_radius: int,
    merge_margin: float,
    max_merges: int,
) -> tuple[np.ndarray, list[dict]]:
    output = np.asarray(labels, dtype=np.int32).copy()
    predicted_boundary = hard_abbc == 2
    interventions: list[dict] = []

    adjacent_pairs = _adjacent_label_pairs(output, int(adjacency_radius))
    for _ in range(int(max_merges)):
        values = [int(value) for value in np.unique(output) if int(value) > 0]
        label_boxes = _all_label_boxes(output)
        best = None
        for left, right in combinations(values, 2):
            if (left, right) not in adjacent_pairs:
                continue
            candidate = _candidate_interface(
                output,
                left,
                right,
                adjacency_radius=int(adjacency_radius),
                interface_radius=int(interface_radius),
                label_boxes=label_boxes,
            )
            if candidate is None:
                continue
            crop, split_interface = candidate
            # The public implementation evaluates a padded fracture-interface
            # crop.  Keeping the same local crop allows other predicted fracture
            # bands to penalize an unsafe merge without using anatomy identity.
            predicted = predicted_boundary[crop]
            split_distance = _dice_distance(split_interface, predicted)
            merged_distance = _dice_distance(
                np.zeros(split_interface.shape, dtype=bool), predicted
            )
            improvement = split_distance - merged_distance
            if improvement <= float(merge_margin):
                continue
            if best is None or improvement > best[0]:
                best = (
                    float(improvement),
                    int(left),
                    int(right),
                    float(split_distance),
                    float(merged_distance),
                    int(split_interface.sum()),
                    int(predicted.sum()),
                )
        if best is None:
            break
        (
            improvement,
            left,
            right,
            split_distance,
            merged_distance,
            interface_voxels,
            predicted_voxels,
        ) = best
        output[output == right] = left
        adjacent_pairs = _remap_adjacency_after_merge(
            adjacent_pairs, values, left, right
        )
        output, _ = _relabel(output)
        interventions.append(
            {
                "left": left,
                "right": right,
                "dice_distance_improvement": improvement,
                "split_distance": split_distance,
                "merged_distance": merged_distance,
                "interface_voxels": interface_voxels,
                "predicted_boundary_voxels_in_crop": predicted_voxels,
            }
        )

    return _relabel(output)[0], interventions


def refine_instances_with_full_abbc(
    labels: np.ndarray,
    abbc_probs: np.ndarray,
    *,
    split_passes: int = 3,
    relative_error_threshold: float = 0.10,
    absolute_error_threshold: int = 100,
    min_split_piece_voxels: int = 400,
    split_dilation_radius: int = 3,
    max_split_dilations: int = 10,
    adjacency_radius: int = 2,
    interface_radius: int = 3,
    merge_margin: float = 0.05 / 3.0,
    max_merges: int = 100,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Heal a 1/3/9-affinity partition using the full ABBC prediction."""
    probs = np.asarray(abbc_probs, dtype=np.float32)
    output = np.asarray(labels, dtype=np.int32)
    if probs.ndim != 4 or int(probs.shape[0]) != 4:
        raise ValueError(f"expected ABBC [4,Z,Y,X], got {probs.shape}")
    if tuple(probs.shape[1:]) != tuple(output.shape):
        raise ValueError("ABBC/instance spatial shape mismatch")
    hard = np.argmax(probs, axis=0).astype(np.uint8, copy=False)
    foreground = hard != 0
    initial_fragments = int(np.unique(output[output > 0]).size)
    output = output.copy()
    output[~foreground] = 0
    output, _ = _relabel(output)

    split_events: list[dict] = []
    for pass_index in range(int(split_passes)):
        output, events = _split_healing_pass(
            output,
            hard,
            probs[2],
            relative_error_threshold=float(relative_error_threshold),
            absolute_error_threshold=int(absolute_error_threshold),
            min_split_piece_voxels=int(min_split_piece_voxels),
            split_dilation_radius=int(split_dilation_radius),
            max_split_dilations=int(max_split_dilations),
            interface_radius=int(interface_radius),
        )
        for event in events:
            event["pass"] = pass_index + 1
        split_events.extend(events)

    after_split = int(np.unique(output[output > 0]).size)
    output, merge_events = _merge_healing(
        output,
        hard,
        adjacency_radius=int(adjacency_radius),
        interface_radius=int(interface_radius),
        merge_margin=float(merge_margin),
        max_merges=int(max_merges),
    )
    output[~foreground] = 0
    output, final_fragments = _relabel(output)
    class_counts = {
        name: int((hard == index).sum())
        for index, name in enumerate(
            ("background", "outer_border", "fracture_boundary", "core")
        )
    }
    report = {
        "initial_fragments": initial_fragments,
        "fragments_after_split": after_split,
        "output_fragments": final_fragments,
        "accepted_splits": len(split_events),
        "accepted_merges": len(merge_events),
        "hard_class_voxels": class_counts,
        "split_events": split_events,
        "merge_events": merge_events,
    }
    return (output, report) if return_report else output


def fill_stage1_mask_by_nearest_instance(
    labels: np.ndarray,
    stage1_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Original-style final semantic-mask fill, kept as a separate ablation."""
    output = np.asarray(labels, dtype=np.int32).copy()
    fill_mask = np.asarray(stage1_mask, dtype=bool)
    if tuple(output.shape) != tuple(fill_mask.shape):
        raise ValueError("instance/stage1 mask shape mismatch")
    before = int((output > 0).sum())
    if bool((output > 0).any()):
        # The reference routine adds semantic-mask voxels to existing
        # instances; it does not clip already predicted instance support.
        growth_mask = fill_mask | (output > 0)
        output = watershed(
            np.zeros(output.shape, dtype=np.float32),
            markers=output,
            mask=growth_mask,
            connectivity=CONNECTIVITY_26,
        ).astype(np.int32, copy=False)
    output, fragments = _relabel(output)
    after = int((output > 0).sum())
    return output, {
        "support_voxels_before": before,
        "stage1_mask_voxels": int(fill_mask.sum()),
        "filled_voxels": after - before,
        "output_fragments": fragments,
    }
