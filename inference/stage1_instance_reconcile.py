"""Conservatively reconcile Stage-2 instance support with a Stage-1 mask."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from abbc_full_refine_decode import fill_stage1_mask_by_nearest_instance


def fill_anatomy_support(
    labels: np.ndarray,
    stage1_mask: np.ndarray,
    label_range: tuple[int, int],
    *,
    spacing_zyx_mm: tuple[float, float, float] | None = None,
    max_growth_distance_mm: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Fill unassigned Stage-1 voxels from existing anatomy instances only.

    A disconnected Stage-1 component without a Stage-2 marker stays empty.
    Existing Stage-2 support is never clipped, and labels from other anatomy
    ranges are never overwritten.
    """
    source = np.asarray(labels)
    mask = np.asarray(stage1_mask, dtype=bool)
    if source.ndim != 3 or tuple(source.shape) != tuple(mask.shape):
        raise ValueError("labels and stage1_mask must be shape-matched 3D arrays")
    lo, hi = (int(label_range[0]), int(label_range[1]))
    if lo < 1 or hi < lo:
        raise ValueError(f"invalid anatomy label range: {label_range}")

    anatomy_source = (source >= lo) & (source <= hi)
    source_ids = sorted(int(value) for value in np.unique(source[anatomy_source]))
    missing_before = mask & ~anatomy_source
    if not source_ids:
        return source.astype(np.uint16, copy=True), {
            "source_instances": 0,
            "output_instances": 0,
            "source_support_voxels": 0,
            "stage1_mask_voxels": int(mask.sum()),
            "stage1_missing_before": int(missing_before.sum()),
            "filled_voxels": 0,
            "stage1_missing_after": int(missing_before.sum()),
            "unseeded_stage1_voxels": int(missing_before.sum()),
        }

    allowed = anatomy_source | mask
    coordinates = np.nonzero(allowed)
    box = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
    )
    source_crop = source[box]
    mask_crop = mask[box]
    local = np.zeros(source_crop.shape, dtype=np.int32)
    for local_id, source_id in enumerate(source_ids, start=1):
        local[source_crop == source_id] = local_id
    effective_mask = mask_crop
    if max_growth_distance_mm is not None:
        if spacing_zyx_mm is None or len(spacing_zyx_mm) != 3:
            raise ValueError("spacing_zyx_mm is required for distance-capped fill")
        if float(max_growth_distance_mm) <= 0.0:
            raise ValueError("max_growth_distance_mm must be positive")
        distance = ndi.distance_transform_edt(
            local == 0,
            sampling=tuple(float(value) for value in spacing_zyx_mm),
        )
        effective_mask = mask_crop & (
            distance <= float(max_growth_distance_mm)
        )
    filled, fill_report = fill_stage1_mask_by_nearest_instance(
        local, effective_mask
    )
    output_ids = sorted(int(value) for value in np.unique(filled[filled > 0]))
    if len(output_ids) > hi - lo + 1:
        raise RuntimeError(
            f"filled anatomy has {len(output_ids)} instances but range {lo}..{hi} "
            "has fewer slots"
        )

    global_filled = np.zeros(filled.shape, dtype=np.uint16)
    for index, local_id in enumerate(output_ids):
        global_filled[filled == local_id] = np.uint16(lo + index)

    output = source.astype(np.uint16, copy=True)
    output[anatomy_source] = 0
    output_crop = output[box]
    write_mask = (global_filled > 0) & (output_crop == 0)
    output_crop[write_mask] = global_filled[write_mask]
    output[box] = output_crop
    output_support = (output >= lo) & (output <= hi)
    if not np.all(output_support[anatomy_source]):
        raise RuntimeError("Stage-1 reconciliation removed existing Stage-2 support")
    if np.any(output_support & ~(mask | anatomy_source)):
        raise RuntimeError("Stage-1 reconciliation grew outside allowed support")

    missing_after = mask & ~output_support
    filled_voxels = int(output_support.sum()) - int(anatomy_source.sum())
    return output, {
        "source_instances": len(source_ids),
        "output_instances": len(output_ids),
        "source_support_voxels": int(anatomy_source.sum()),
        "stage1_mask_voxels": int(mask.sum()),
        "stage1_missing_before": int(missing_before.sum()),
        "filled_voxels": filled_voxels,
        "stage1_missing_after": int(missing_after.sum()),
        "unseeded_stage1_voxels": int(missing_after.sum()),
        "max_growth_distance_mm": (
            None
            if max_growth_distance_mm is None
            else float(max_growth_distance_mm)
        ),
        "processing_bbox_zyx": [
            [int(axis.start), int(axis.stop)] for axis in box
        ],
        "processing_voxels": int(np.prod(local.shape)),
        "fill_backend": fill_report,
    }
