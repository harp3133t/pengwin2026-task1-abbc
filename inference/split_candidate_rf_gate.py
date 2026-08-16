#!/usr/bin/env python
"""Inference-only split-candidate Random Forest gate for Task 1 v3.6.1.

The frozen v3.5 partition is the safe base. A 1/3/9-affinity + full-ABBC
partition supplies binary split proposals. Five RF regressors estimate each
proposal's changes in Merge, Split, Dice, Instance F1, and Precision. Only a
proposal passing the packaged split-aware policy changes the instance labels;
foreground support is invariant.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage as ndi


ANATOMY_RANGES = {
    "Sacrum": (1, 50),
    "LeftHip": (51, 100),
    "RightHip": (101, 150),
    "Femur": (151, 200),
}
TARGETS = (
    "delta_merge",
    "delta_split",
    "delta_dice",
    "delta_f1",
    "delta_precision",
)
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def load_split_gate(path: str | Path) -> dict:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"split-candidate RF artifact missing: {artifact}")
    payload = joblib.load(artifact)
    for key in ("models", "feature_names", "policy", "quality_guards"):
        if key not in payload:
            raise KeyError(f"split-candidate RF artifact missing key {key!r}")
    missing_targets = [name for name in TARGETS if name not in payload["models"]]
    if missing_targets:
        raise KeyError(f"split-candidate RF missing targets: {missing_targets}")
    return payload


def _crop(mask: np.ndarray, pad: int = 1) -> tuple[slice, ...]:
    coordinates = np.where(mask)
    if not coordinates[0].size:
        raise ValueError("empty split-candidate mask")
    return tuple(
        slice(
            max(0, int(values.min()) - int(pad)),
            min(mask.shape[axis], int(values.max()) + 1 + int(pad)),
        )
        for axis, values in enumerate(coordinates)
    )


def _component_count(mask: np.ndarray) -> int:
    if not bool(mask.any()):
        return 0
    _, count = ndi.label(mask, structure=CONNECTIVITY_26)
    return int(count)


def _surface_fraction(mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count == 0:
        return 0.0
    eroded = ndi.binary_erosion(mask, structure=CONNECTIVITY_26)
    return float(np.sum(mask & ~eroded)) / float(count)


def _center_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(ndi.center_of_mass(mask), dtype=np.float64) * np.asarray(
        spacing, dtype=np.float64
    )


def _proposal_ids_for_source(
    proposal: np.ndarray,
    source_mask: np.ndarray,
    lo: int,
    hi: int,
) -> list[tuple[int, int]]:
    values, counts = np.unique(proposal[source_mask], return_counts=True)
    pairs = [
        (int(value), int(count))
        for value, count in zip(values, counts)
        if int(lo) <= int(value) <= int(hi)
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs


def _anatomy_context(multiscale: dict, abbc: dict) -> dict[str, float]:
    hard = abbc["hard_class_voxels"]
    hard_total = float(max(1, sum(int(value) for value in hard.values())))
    rag_edges = float(max(1, int(multiscale["initial_rag_edges"])))
    events = list(abbc.get("split_events", []))
    return {
        "context_initial_supervoxels": float(multiscale["initial_supervoxels"]),
        "context_initial_rag_edges": float(multiscale["initial_rag_edges"]),
        "context_mid_coverage": float(multiscale["mid_covered_edges"]) / rag_edges,
        "context_long_coverage": float(multiscale["long_covered_edges"]) / rag_edges,
        "context_veto_fraction": float(multiscale["any_veto_edges"]) / rag_edges,
        "context_abbc_border_fraction": float(hard["outer_border"]) / hard_total,
        "context_abbc_boundary_fraction": float(hard["fracture_boundary"])
        / hard_total,
        "context_abbc_core_fraction": float(hard["core"]) / hard_total,
        "context_accepted_splits": float(abbc["accepted_splits"]),
        "context_max_relative_error": (
            float(max(row["relative_error"] for row in events)) if events else 0.0
        ),
        "context_mean_relative_error": (
            float(np.mean([row["relative_error"] for row in events]))
            if events
            else 0.0
        ),
    }


def _candidate_features(
    *,
    anatomy: str,
    base: np.ndarray,
    proposal: np.ndarray,
    ct: np.ndarray,
    source_id: int,
    proposal_id: int,
    proposal_id_count: int,
    proposal_total: int,
    invert_proposal: bool,
    piece: np.ndarray,
    remainder: np.ndarray,
    spacing: tuple[float, float, float],
    voxel_mm3: float,
    context: dict[str, float],
) -> dict[str, float]:
    source = base == int(source_id)
    crop = _crop(source, pad=1)
    local_source = source[crop]
    local_piece = piece[crop]
    local_remainder = remainder[crop]
    local_ct = np.asarray(ct[crop], dtype=np.float32)
    piece_values = local_ct[local_piece]
    remainder_values = local_ct[local_remainder]
    source_voxels = int(local_source.sum())
    piece_voxels = int(local_piece.sum())
    remainder_voxels = int(local_remainder.sum())
    small_voxels = min(piece_voxels, remainder_voxels)
    large_voxels = max(piece_voxels, remainder_voxels)
    interface = ndi.binary_dilation(
        local_piece, structure=CONNECTIVITY_26
    ) & local_remainder
    bbox_lengths = []
    for axis, values in enumerate(np.where(local_source)):
        length = (int(values.max()) - int(values.min()) + 1) * float(spacing[axis])
        bbox_lengths.append(float(length))
    bbox_lengths.sort()
    center_distance = float(
        np.linalg.norm(
            _center_mm(local_piece, spacing)
            - _center_mm(local_remainder, spacing)
        )
    )
    equivalent_radius = float(
        ((3.0 * source_voxels * voxel_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
    )
    proposal_inside = int((proposal[source] == int(proposal_id)).sum())
    pooled_std = float(
        np.sqrt(0.5 * (float(piece_values.var()) + float(remainder_values.var())))
    )
    lo, hi = ANATOMY_RANGES[anatomy]
    anatomy_ids = [
        int(value) for value in np.unique(base) if lo <= int(value) <= hi
    ]
    return {
        "is_sacrum": float(anatomy == "Sacrum"),
        "is_left_hip": float(anatomy == "LeftHip"),
        "is_right_hip": float(anatomy == "RightHip"),
        "is_femur": float(anatomy == "Femur"),
        "anatomy_v35_fragment_count": float(len(anatomy_ids)),
        "source_log_volume_cm3": float(
            np.log1p(source_voxels * voxel_mm3 / 1000.0)
        ),
        "source_component_count": float(_component_count(local_source)),
        "source_bbox_short_mm": bbox_lengths[0],
        "source_bbox_mid_mm": bbox_lengths[1],
        "source_bbox_long_mm": bbox_lengths[2],
        "source_bbox_elongation": bbox_lengths[2] / max(bbox_lengths[0], 1.0e-6),
        "piece_small_volume_cm3": float(small_voxels * voxel_mm3 / 1000.0),
        "piece_large_volume_cm3": float(large_voxels * voxel_mm3 / 1000.0),
        "piece_small_large_ratio": float(small_voxels / max(large_voxels, 1)),
        "isolated_piece_fraction": float(piece_voxels / max(source_voxels, 1)),
        "isolated_is_complement": float(bool(invert_proposal)),
        "piece_component_count": float(_component_count(local_piece)),
        "remainder_component_count": float(_component_count(local_remainder)),
        "piece_surface_fraction": _surface_fraction(local_piece),
        "remainder_surface_fraction": _surface_fraction(local_remainder),
        "interface_voxels": float(interface.sum()),
        "interface_source_fraction": float(interface.sum() / max(source_voxels, 1)),
        "centroid_distance_mm": center_distance,
        "centroid_distance_radius_ratio": center_distance
        / max(equivalent_radius, 1.0e-6),
        "proposal_ids_in_source": float(proposal_id_count),
        "proposal_foreground_coverage": float(
            np.sum((proposal[source] >= lo) & (proposal[source] <= hi))
            / max(source_voxels, 1)
        ),
        "proposal_label_leakage_fraction": float(
            max(0, int(proposal_total) - proposal_inside) / max(int(proposal_total), 1)
        ),
        "ct_piece_mean": float(piece_values.mean()),
        "ct_remainder_mean": float(remainder_values.mean()),
        "ct_absolute_mean_difference": float(
            abs(float(piece_values.mean()) - float(remainder_values.mean()))
        ),
        "ct_standardized_mean_difference": float(
            abs(float(piece_values.mean()) - float(remainder_values.mean()))
            / max(pooled_std, 1.0e-6)
        ),
        "ct_piece_std": float(piece_values.std()),
        "ct_remainder_std": float(remainder_values.std()),
        **context,
    }


def build_candidate_rows(
    base: np.ndarray,
    proposal: np.ndarray,
    ct: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    decoder_contexts: dict[str, dict],
) -> tuple[list[dict], dict[str, int]]:
    base = np.asarray(base, dtype=np.uint16)
    proposal = np.asarray(proposal, dtype=np.uint16)
    ct = np.asarray(ct, dtype=np.float32)
    if base.shape != proposal.shape or base.shape != ct.shape:
        raise ValueError("base/proposal/CT shape mismatch")
    voxel_mm3 = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    proposal_values, proposal_counts = np.unique(proposal, return_counts=True)
    proposal_totals = {
        int(value): int(count) for value, count in zip(proposal_values, proposal_counts)
    }
    rows = []
    rejected = {
        "fewer_than_two_proposal_ids": 0,
        "piece_below_1cm3": 0,
        "duplicate_piece": 0,
    }
    for anatomy, (lo, hi) in ANATOMY_RANGES.items():
        if anatomy not in decoder_contexts:
            continue
        context = _anatomy_context(
            decoder_contexts[anatomy]["multiscale"],
            decoder_contexts[anatomy]["abbc"],
        )
        used_ids = {
            int(value) for value in np.unique(base) if lo <= int(value) <= hi
        }
        for source_id in sorted(used_ids):
            source = base == int(source_id)
            pairs = _proposal_ids_for_source(proposal, source, lo, hi)
            if len(pairs) < 2:
                rejected["fewer_than_two_proposal_ids"] += 1
                continue
            if len(pairs) == 2:
                candidate_ids = [min(pairs, key=lambda item: (item[1], item[0]))[0]]
            else:
                candidate_ids = [value for value, _ in pairs]
            seen_signatures: set[tuple[int, int, int, int]] = set()
            for proposal_id in candidate_ids:
                raw_piece = source & (proposal == int(proposal_id))
                raw_remainder = source & ~raw_piece
                invert = bool(raw_piece.sum() > raw_remainder.sum())
                piece = raw_remainder if invert else raw_piece
                remainder = source & ~piece
                piece_voxels = int(piece.sum())
                remainder_voxels = int(remainder.sum())
                if (
                    piece_voxels * voxel_mm3 < 1000.0
                    or remainder_voxels * voxel_mm3 < 1000.0
                ):
                    rejected["piece_below_1cm3"] += 1
                    continue
                coordinates = np.where(piece)
                signature = (
                    piece_voxels,
                    int(np.sum(coordinates[0], dtype=np.int64)),
                    int(np.sum(coordinates[1], dtype=np.int64)),
                    int(np.sum(coordinates[2], dtype=np.int64)),
                )
                if signature in seen_signatures:
                    rejected["duplicate_piece"] += 1
                    continue
                seen_signatures.add(signature)
                features = _candidate_features(
                    anatomy=anatomy,
                    base=base,
                    proposal=proposal,
                    ct=ct,
                    source_id=source_id,
                    proposal_id=proposal_id,
                    proposal_id_count=len(pairs),
                    proposal_total=proposal_totals.get(int(proposal_id), 0),
                    invert_proposal=invert,
                    piece=piece,
                    remainder=remainder,
                    spacing=spacing_zyx,
                    voxel_mm3=voxel_mm3,
                    context=context,
                )
                rows.append(
                    {
                        "anatomy": anatomy,
                        "source_id": int(source_id),
                        "proposal_id": int(proposal_id),
                        "invert_proposal": int(invert),
                        **features,
                    }
                )
    return rows, rejected


def predict_and_select(rows: list[dict], payload: dict) -> list[dict]:
    if not rows or payload["policy"].get("action") != "select":
        return []
    feature_names = list(payload["feature_names"])
    missing = [name for name in feature_names if name not in rows[0]]
    if missing:
        raise KeyError(f"split-candidate feature extractor missing: {missing}")
    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    predictions = {
        target: np.asarray(payload["models"][target].predict(x), dtype=np.float64)
        for target in TARGETS
    }
    policy = payload["policy"]
    guards = payload["quality_guards"]
    best: dict[tuple[str, int], tuple[tuple, int]] = {}
    for index, row in enumerate(rows):
        predicted = {target: float(predictions[target][index]) for target in TARGETS}
        row.update({f"predicted_{key}": value for key, value in predicted.items()})
        eligible = bool(
            predicted["delta_merge"] <= float(policy["merge_max"])
            and predicted["delta_split"] <= float(policy["split_max"])
            and predicted["delta_dice"] >= float(guards["delta_dice_min"])
            and predicted["delta_f1"] >= float(guards["delta_f1_min"])
            and predicted["delta_precision"]
            >= float(guards["delta_precision_min"])
        )
        row["eligible"] = int(eligible)
        if not eligible:
            continue
        key = (str(row["anatomy"]), int(row["source_id"]))
        order = (
            predicted["delta_merge"],
            predicted["delta_split"],
            -predicted["delta_f1"],
            -predicted["delta_dice"],
            -predicted["delta_precision"],
            int(row["proposal_id"]),
        )
        if key not in best or order < best[key][0]:
            best[key] = (order, index)
    return [rows[item[1]] for item in best.values()]


def apply_selected(
    base: np.ndarray, proposal: np.ndarray, selected: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    base = np.asarray(base, dtype=np.uint16)
    proposal = np.asarray(proposal, dtype=np.uint16)
    output = base.copy()
    applied = []
    by_anatomy: dict[str, list[dict]] = {}
    for row in selected:
        by_anatomy.setdefault(str(row["anatomy"]), []).append(row)
    for anatomy, anatomy_rows in by_anatomy.items():
        lo, hi = ANATOMY_RANGES[anatomy]
        used = {int(value) for value in np.unique(output) if lo <= int(value) <= hi}
        free = [value for value in range(lo, hi + 1) if value not in used]
        for row in sorted(anatomy_rows, key=lambda item: int(item["source_id"])):
            if not free:
                break
            source_id = int(row["source_id"])
            proposal_id = int(row["proposal_id"])
            raw_piece = (base == source_id) & (proposal == proposal_id)
            piece = (
                (base == source_id) & ~raw_piece
                if bool(int(row["invert_proposal"]))
                else raw_piece
            )
            if not bool(piece.any()):
                raise RuntimeError("empty reconstructed split candidate")
            new_id = int(free.pop(0))
            output[piece] = np.uint16(new_id)
            applied.append(
                {
                    "anatomy": anatomy,
                    "source_id": source_id,
                    "proposal_id": proposal_id,
                    "new_id": new_id,
                    "piece_voxels": int(piece.sum()),
                    **{
                        f"predicted_{target}": float(row[f"predicted_{target}"])
                        for target in TARGETS
                    },
                }
            )
    if not np.array_equal(output > 0, base > 0):
        raise RuntimeError("split-candidate RF changed foreground support")
    return output, applied


def run_split_candidate_rf_gate(
    base: np.ndarray,
    proposal: np.ndarray,
    ct: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    decoder_contexts: dict[str, dict],
    payload: dict,
) -> tuple[np.ndarray, dict]:
    rows, rejected = build_candidate_rows(
        base, proposal, ct, spacing_zyx, decoder_contexts
    )
    selected = predict_and_select(rows, payload)
    output, applied = apply_selected(base, proposal, selected)
    return output, {
        "candidate_count": len(rows),
        "eligible_count": int(sum(int(row.get("eligible", 0)) for row in rows)),
        "selected_count": len(selected),
        "applied_count": len(applied),
        "rejected": rejected,
        "policy": payload["policy"],
        "quality_guards": payload["quality_guards"],
        "applied": applied,
        "foreground_support_identical": True,
    }
