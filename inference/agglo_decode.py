"""Tier-0 PoC: oversegment + average-linkage agglomeration decoder on V302 ABBC probs.
ABBC softmax channels: 0=bg, 1=border(outer surface), 2=boundary(inter-fragment fracture
surface), 3=core(eroded interior). Idea (GASP average-linkage / connectomics consensus):
 - oversegment the fragment foreground at EVERY boundary ridge (watershed on boundary-prob),
 - then conservatively agglomerate adjacent supervoxels whose shared interface boundary-prob
   is WEAK (avg-linkage), keeping only interfaces that look like true fracture surfaces (>= T).
This avoids the watershed MERGE (we split everywhere first) and the mutex OVER-split
(avg-linkage merges weak ridges back). Inference-only on the DEPLOYED V302 probs.
"""
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
try:
    from skimage.graph import rag_boundary, merge_hierarchical
except ImportError:
    from skimage.future.graph import rag_boundary, merge_hierarchical

# canonical skimage average-linkage merge functions for rag_boundary (weighted-mean boundary)
def _weight_boundary(graph, src, dst, n):
    d = {'weight': 0.0, 'count': 0}
    cs = graph[src].get(n, d)['count']; cd = graph[dst].get(n, d)['count']
    ws = graph[src].get(n, d)['weight']; wd = graph[dst].get(n, d)['weight']
    c = cs + cd
    return {'count': c, 'weight': (cs * ws + cd * wd) / c if c else 0.0}

def _merge_boundary(graph, src, dst):
    pass

def decode_agglo(probs, T=0.45, min_vox=250, seed_core=0.5, seed_bnd=0.20, min_frac=None):
    """probs: float array [4,Z,Y,X]. Returns int instance map (0=bg).
    min_frac (env PENGWIN_AGGLO_MINFRAC, default 0): reabsorb any fragment smaller than
    min_frac x the largest fragment -> kills phantom/sliver over-splits (precision guard)."""
    bg, border, boundary, core = probs[0], probs[1], probs[2], probs[3]
    fg = bg < 0.5
    if fg.sum() == 0:
        return np.zeros(fg.shape, np.int32)
    elev = boundary.astype(np.float32)  # high = fracture-surface ridge
    # oversegment: seed at strong-core / low-boundary interiors, watershed the boundary ridges
    seeds = (core > seed_core) & (boundary < seed_bnd) & fg
    markers, nm = ndi.label(seeds)
    if nm <= 1:  # fallback: seed at boundary-distance maxima
        d = ndi.distance_transform_edt(boundary < seed_bnd) * fg
        from skimage.feature import peak_local_max
        pk = peak_local_max(d, min_distance=4, labels=fg)
        m = np.zeros(fg.shape, bool); m[tuple(pk.T)] = True
        markers, nm = ndi.label(m)
        if nm == 0:
            markers, nm = ndi.label(fg)
    sv = watershed(elev, markers, mask=fg)
    if sv.max() <= 1:
        out = (sv > 0).astype(np.int32)
    else:
        rag = rag_boundary(sv, elev)
        if 0 in rag.nodes:          # drop the BACKGROUND node so fg regions can't agglomerate into bg
            rag.remove_node(0)
        merged = merge_hierarchical(sv, rag, thresh=T, rag_copy=False, in_place_merge=True,
                                    merge_func=_merge_boundary, weight_func=_weight_boundary)
        out = merged.astype(np.int32) + 1   # shift: merge_hierarchical is 0-based, so no fg region collides with bg=0
    out[~fg] = 0
    out, _ = _relabel(out)
    out = _drop_small(out, min_vox)
    mf = float(os.environ.get("PENGWIN_AGGLO_MINFRAC", "0.0")) if min_frac is None else float(min_frac)
    if mf > 0.0:
        out = _drop_by_ratio(out, mf)
    return out

def _drop_by_ratio(lab, min_frac):
    """Reabsorb fragments whose voxel count < min_frac x the largest fragment (phantom/sliver guard)."""
    ids, cnts = np.unique(lab[lab > 0], return_counts=True)
    if len(cnts) <= 1:
        return lab
    small = set(ids[cnts < min_frac * int(cnts.max())].tolist())
    if not small:
        return lab
    keep = lab.copy()
    for s in small:
        keep[keep == s] = 0
    if (keep > 0).any():
        idx = ndi.distance_transform_edt(keep == 0, return_distances=False, return_indices=True)
        filled = keep[tuple(idx)]
        m = (lab > 0) & (keep == 0)
        keep[m] = filled[m]
    lab2, _ = _relabel(keep)
    return lab2

def _relabel(lab):
    ids = [i for i in np.unique(lab) if i != 0]
    remap = np.zeros(int(lab.max()) + 1, np.int32)
    for k, i in enumerate(ids, 1):
        remap[i] = k
    return remap[lab], len(ids)

def _drop_small(lab, min_vox):
    ids, cnts = np.unique(lab[lab > 0], return_counts=True)
    small = set(ids[cnts < min_vox].tolist())
    if not small:
        return lab
    keep = lab.copy()
    for s in small:
        keep[keep == s] = 0
    if (keep > 0).any() and (lab > 0).any():
        # reassign small voxels to nearest kept region
        idx = ndi.distance_transform_edt(keep == 0, return_distances=False, return_indices=True)
        filled = keep[tuple(idx)]
        m = (lab > 0) & (keep == 0)
        keep[m] = filled[m]
    lab2, _ = _relabel(keep)
    return lab2

def decode_affinity_agglo(abbc_probs, affinities, T=0.45, min_vox=250, short_idx=(0, 1, 2)):
    """[TIER-1] Decode V307's 13ch output into instances by average-linkage agglomeration on the
    LEARNED affinities (vs the noisy ABBC boundary channel that Tier-0 used).
      abbc_probs : [4,Z,Y,X] softmax (bg,border,boundary,core)
      affinities : [K,Z,Y,X] sigmoid, same-instance prob per loss.AFFINITY_HEAD_OFFSETS
    Separation ridge = 1 - min(short-range affinity): high where any short affinity is LOW = a true
    fracture surface. We splice this affinity-separation into the boundary channel and reuse the
    validated avg-linkage decoder (oversegment at ridges -> conservatively merge weak ridges)."""
    short = np.asarray(affinities)[list(short_idx)]
    sep = 1.0 - short.min(axis=0)                 # [Z,Y,X], high = fracture surface
    probs_aff = np.asarray(abbc_probs, dtype=np.float32).copy()
    probs_aff[2] = sep.astype(np.float32)         # replace ABBC boundary with the affinity separation
    return decode_agglo(probs_aff, T=T, min_vox=min_vox)


def complete_affinity_basin_seeds(fg, core, sep, basin_sep=0.50,
                                  min_basin_vox=250, seed_core=0.50,
                                  seed_bnd=0.20):
    """Add one deterministic marker to each large low-separation basin lacking a core seed.

    This is a diagnostic, GT-free seed-completion rule. Basins use the same 6-neighbour
    connectivity as scipy's default component labelling and the downstream watershed.
    The added marker is the basin voxel farthest from its exterior; ties are resolved by
    C-order ``argmax``. Existing core markers are never changed.
    """
    fg = np.asarray(fg, dtype=bool)
    core = np.asarray(core, dtype=np.float32)
    sep = np.asarray(sep, dtype=np.float32)
    seed_mask = (core > float(seed_core)) & (sep < float(seed_bnd)) & fg
    initial_markers, initial_count = ndi.label(seed_mask)
    basin_mask = fg & (sep < float(basin_sep))
    basins, n_basins = ndi.label(basin_mask)
    counts = np.bincount(basins.ravel(), minlength=n_basins + 1)
    objects = ndi.find_objects(basins)
    added = []
    qualifying = 0
    already_seeded = 0
    for basin_id in range(1, n_basins + 1):
        basin_voxels = int(counts[basin_id])
        if basin_voxels < int(min_basin_vox):
            continue
        qualifying += 1
        box = objects[basin_id - 1]
        if box is None:
            continue
        local_basin = basins[box] == basin_id
        if np.any(seed_mask[box] & local_basin):
            already_seeded += 1
            continue
        padded = np.pad(local_basin, 1, mode="constant", constant_values=False)
        distance = ndi.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
        local_coord = np.unravel_index(int(np.argmax(distance)), distance.shape)
        global_coord = tuple(int(box[axis].start + local_coord[axis]) for axis in range(3))
        bbox_extent = [int(axis_slice.stop - axis_slice.start) for axis_slice in box]
        bbox_voxels = int(np.prod(bbox_extent))
        expanded_box = tuple(
            slice(max(0, axis_slice.start - 1), min(fg.shape[axis], axis_slice.stop + 1))
            for axis, axis_slice in enumerate(box)
        )
        expanded_basin = basins[expanded_box] == basin_id
        expanded_fg = fg[expanded_box]
        expanded_sep = sep[expanded_box]
        interface = (
            ndi.binary_dilation(
                expanded_basin, structure=ndi.generate_binary_structure(3, 1)
            )
            & expanded_fg
            & ~expanded_basin
        )
        interface_values = expanded_sep[interface]
        interface_labels, n_interface_components = ndi.label(interface)
        interface_counts = np.bincount(
            interface_labels.ravel(), minlength=n_interface_components + 1
        )
        largest_interface = (
            int(interface_counts[1:].max()) if n_interface_components > 0 else 0
        )
        coords = np.argwhere(local_basin)
        if len(coords) > 200_000:
            coords = coords[:: int(np.ceil(len(coords) / 200_000))]
        if len(coords) >= 3:
            covariance = np.cov(coords.astype(np.float64), rowvar=False)
            eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-9)
            principal_elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
        else:
            principal_elongation = float("inf")
        seed_mask[global_coord] = True
        added.append(
            {
                "basin_id": int(basin_id),
                "basin_voxels": basin_voxels,
                "seed_zyx": list(global_coord),
                "interior_distance_vox": float(distance[local_coord]),
                "bbox_extent_zyx": bbox_extent,
                "minimum_bbox_extent_vox": int(min(bbox_extent)),
                "bbox_fill_fraction": float(basin_voxels / max(1, bbox_voxels)),
                "principal_elongation": principal_elongation,
                "interface_voxels": int(interface.sum()),
                "largest_interface_component_voxels": largest_interface,
                "interface_sep_mean": float(interface_values.mean())
                if interface_values.size
                else 0.0,
                "interface_sep_q25": float(np.quantile(interface_values, 0.25))
                if interface_values.size
                else 0.0,
                "interface_sep_median": float(np.median(interface_values))
                if interface_values.size
                else 0.0,
                "core_probability": float(core[global_coord]),
                "separation": float(sep[global_coord]),
            }
        )
    markers, final_count = ndi.label(seed_mask)
    diagnostics = {
        "basin_separation_threshold": float(basin_sep),
        "minimum_basin_voxels": int(min_basin_vox),
        "initial_seed_components": int(initial_count),
        "affinity_basins_total": int(n_basins),
        "qualifying_affinity_basins": int(qualifying),
        "qualifying_basins_already_seeded": int(already_seeded),
        "forced_seeds_added": int(len(added)),
        "seed_components_after_completion": int(final_count),
        "added_seed_details": added,
    }
    return markers, diagnostics


def decode_affinity_agglo_seed_completed(
    abbc_probs,
    affinities,
    T=0.45,
    min_vox=250,
    short_idx=(0, 1, 2),
    basin_sep=0.50,
    min_basin_vox=250,
    seed_core=0.50,
    seed_bnd=0.20,
    guard_profile="none",
    guard_min_region_vox=500,
    guard_min_region_fraction=0.0015,
    guard_min_bbox_extent_vox=10,
    guard_min_interior_distance_vox=2.0,
    guard_min_interface_vox=15,
    guard_min_largest_interface_component_vox=5,
    guard_min_interface_sep_median=0.90,
    return_diagnostics=False,
):
    """Affinity agglomeration with one GT-free marker per unseeded affinity basin.

    Only marker construction differs from :func:`decode_affinity_agglo`; foreground,
    watershed energy, average-linkage threshold, and small-component handling are kept.
    ``guard_profile='boundary'`` retains only candidates with a coherent high-separation
    provisional interface (or a fully topology-isolated component). ``'selective_v1'``
    additionally requires a plausible provisional size and three-dimensional thickness.
    """
    abbc = np.asarray(abbc_probs, dtype=np.float32)
    bg, core = abbc[0], abbc[3]
    fg = bg < 0.5
    if not fg.any():
        empty = np.zeros(fg.shape, np.int32)
        diagnostics = {
            "basin_separation_threshold": float(basin_sep),
            "minimum_basin_voxels": int(min_basin_vox),
            "initial_seed_components": 0,
            "affinity_basins_total": 0,
            "qualifying_affinity_basins": 0,
            "qualifying_basins_already_seeded": 0,
            "forced_seeds_added": 0,
            "forced_seed_candidates": 0,
            "forced_seeds_kept": 0,
            "forced_seeds_rejected": 0,
            "guard_profile": str(guard_profile),
            "seed_components_after_completion": 0,
            "fallback_used": False,
            "final_instances": 0,
        }
        return (empty, diagnostics) if return_diagnostics else empty

    short = np.asarray(affinities, dtype=np.float32)[list(short_idx)]
    sep = (1.0 - short.min(axis=0)).astype(np.float32)
    markers, diagnostics = complete_affinity_basin_seeds(
        fg,
        core,
        sep,
        basin_sep=basin_sep,
        min_basin_vox=min_basin_vox,
        seed_core=seed_core,
        seed_bnd=seed_bnd,
    )
    nm = int(markers.max(initial=0))
    fallback_used = False
    if nm <= 1:
        fallback_used = True
        d = ndi.distance_transform_edt(sep < float(seed_bnd)) * fg
        from skimage.feature import peak_local_max
        pk = peak_local_max(d, min_distance=4, labels=fg)
        fallback_mask = np.zeros(fg.shape, bool)
        fallback_mask[tuple(pk.T)] = True
        markers, nm = ndi.label(fallback_mask)
        if nm == 0:
            markers, nm = ndi.label(fg)

    sv = watershed(sep, markers, mask=fg)
    for detail in diagnostics.get("added_seed_details", []):
        seed_coord = tuple(int(value) for value in detail["seed_zyx"])
        marker_id = int(markers[seed_coord])
        region = sv == marker_id
        region_voxels = int(region.sum())
        region_objects = ndi.find_objects(region.astype(np.uint8))
        region_box = region_objects[0] if region_objects else None
        if region_box is None:
            detail.update(
                {
                    "provisional_region_voxels": 0,
                    "provisional_region_fraction_of_foreground": 0.0,
                    "provisional_minimum_bbox_extent_vox": 0,
                    "provisional_bbox_fill_fraction": 0.0,
                    "provisional_interior_distance_vox": 0.0,
                    "provisional_interface_voxels": 0,
                    "provisional_largest_interface_component_voxels": 0,
                    "provisional_interface_sep_median": 0.0,
                }
            )
            continue
        region_extent = [
            int(axis_slice.stop - axis_slice.start) for axis_slice in region_box
        ]
        local_region = region[region_box]
        local_distance = ndi.distance_transform_edt(
            np.pad(local_region, 1, mode="constant", constant_values=False)
        )[1:-1, 1:-1, 1:-1]
        expanded_region_box = tuple(
            slice(
                max(0, axis_slice.start - 1),
                min(fg.shape[axis], axis_slice.stop + 1),
            )
            for axis, axis_slice in enumerate(region_box)
        )
        expanded_region = region[expanded_region_box]
        expanded_fg = fg[expanded_region_box]
        expanded_sv = sv[expanded_region_box]
        expanded_sep = sep[expanded_region_box]
        provisional_interface = (
            ndi.binary_dilation(
                expanded_region, structure=ndi.generate_binary_structure(3, 1)
            )
            & expanded_fg
            & (expanded_sv != marker_id)
        )
        provisional_values = expanded_sep[provisional_interface]
        provisional_interface_labels, n_provisional_interface = ndi.label(
            provisional_interface
        )
        provisional_counts = np.bincount(
            provisional_interface_labels.ravel(),
            minlength=n_provisional_interface + 1,
        )
        detail.update(
            {
                "provisional_marker_id": marker_id,
                "provisional_region_voxels": region_voxels,
                "provisional_region_fraction_of_foreground": float(
                    region_voxels / max(1, int(fg.sum()))
                ),
                "provisional_bbox_extent_zyx": region_extent,
                "provisional_minimum_bbox_extent_vox": int(min(region_extent)),
                "provisional_bbox_fill_fraction": float(
                    region_voxels / max(1, int(np.prod(region_extent)))
                ),
                "provisional_interior_distance_vox": float(local_distance.max()),
                "provisional_interface_voxels": int(provisional_interface.sum()),
                "provisional_largest_interface_component_voxels": int(
                    provisional_counts[1:].max()
                    if n_provisional_interface > 0
                    else 0
                ),
                "provisional_interface_sep_mean": float(provisional_values.mean())
                if provisional_values.size
                else 0.0,
                "provisional_interface_sep_q25": float(
                    np.quantile(provisional_values, 0.25)
                )
                if provisional_values.size
                else 0.0,
                "provisional_interface_sep_median": float(
                    np.median(provisional_values)
                )
                if provisional_values.size
                else 0.0,
            }
        )
    if guard_profile not in {"none", "boundary", "selective_v1"}:
        raise ValueError(f"unsupported seed guard profile: {guard_profile!r}")
    for detail in diagnostics.get("added_seed_details", []):
        interface_voxels = int(detail["provisional_interface_voxels"])
        topology_isolated = interface_voxels == 0
        coherent_boundary = topology_isolated or (
            interface_voxels >= int(guard_min_interface_vox)
            and int(detail["provisional_largest_interface_component_voxels"])
            >= int(guard_min_largest_interface_component_vox)
            and float(detail["provisional_interface_sep_median"])
            >= float(guard_min_interface_sep_median)
        )
        plausible_size_shape = (
            int(detail["provisional_region_voxels"]) >= int(guard_min_region_vox)
            and float(detail["provisional_region_fraction_of_foreground"])
            >= float(guard_min_region_fraction)
            and int(detail["provisional_minimum_bbox_extent_vox"])
            >= int(guard_min_bbox_extent_vox)
            and float(detail["provisional_interior_distance_vox"])
            >= float(guard_min_interior_distance_vox)
        )
        if guard_profile == "none":
            keep = True
        elif guard_profile == "boundary":
            keep = coherent_boundary
        else:
            keep = coherent_boundary and plausible_size_shape
        rejection_reasons = []
        if not coherent_boundary:
            rejection_reasons.append("weak_or_incoherent_boundary")
        if guard_profile == "selective_v1" and not plausible_size_shape:
            rejection_reasons.append("implausible_size_or_shape")
        detail.update(
            {
                "guard_topology_isolated": bool(topology_isolated),
                "guard_boundary_pass": bool(coherent_boundary),
                "guard_size_shape_pass": bool(plausible_size_shape),
                "guard_keep": bool(keep),
                "guard_rejection_reasons": rejection_reasons,
            }
        )

    if guard_profile != "none":
        guarded_seed_mask = (
            (core > float(seed_core)) & (sep < float(seed_bnd)) & fg
        )
        for detail in diagnostics.get("added_seed_details", []):
            if detail["guard_keep"]:
                guarded_seed_mask[tuple(detail["seed_zyx"])] = True
        markers, nm = ndi.label(guarded_seed_mask)
        fallback_used = False
        if nm <= 1:
            fallback_used = True
            d = ndi.distance_transform_edt(sep < float(seed_bnd)) * fg
            from skimage.feature import peak_local_max
            pk = peak_local_max(d, min_distance=4, labels=fg)
            fallback_mask = np.zeros(fg.shape, bool)
            fallback_mask[tuple(pk.T)] = True
            markers, nm = ndi.label(fallback_mask)
            if nm == 0:
                markers, nm = ndi.label(fg)
        sv = watershed(sep, markers, mask=fg)

    kept_count = int(
        sum(bool(detail["guard_keep"]) for detail in diagnostics.get("added_seed_details", []))
    )
    diagnostics.update(
        {
            "guard_profile": str(guard_profile),
            "guard_thresholds": {
                "minimum_region_voxels": int(guard_min_region_vox),
                "minimum_region_fraction_of_foreground": float(
                    guard_min_region_fraction
                ),
                "minimum_bbox_extent_voxels": int(guard_min_bbox_extent_vox),
                "minimum_interior_distance_voxels": float(
                    guard_min_interior_distance_vox
                ),
                "minimum_interface_voxels": int(guard_min_interface_vox),
                "minimum_largest_interface_component_voxels": int(
                    guard_min_largest_interface_component_vox
                ),
                "minimum_interface_separation_median": float(
                    guard_min_interface_sep_median
                ),
            },
            "forced_seed_candidates": int(
                len(diagnostics.get("added_seed_details", []))
            ),
            "forced_seeds_kept": kept_count,
            "forced_seeds_rejected": int(
                len(diagnostics.get("added_seed_details", [])) - kept_count
            ),
        }
    )
    if sv.max() <= 1:
        out = (sv > 0).astype(np.int32)
    else:
        rag = rag_boundary(sv, sep)
        if 0 in rag.nodes:
            rag.remove_node(0)
        merged = merge_hierarchical(
            sv,
            rag,
            thresh=T,
            rag_copy=False,
            in_place_merge=True,
            merge_func=_merge_boundary,
            weight_func=_weight_boundary,
        )
        out = merged.astype(np.int32) + 1
    out[~fg] = 0
    out, _ = _relabel(out)
    out = _drop_small(out, min_vox)
    diagnostics.update(
        {
            "fallback_used": bool(fallback_used),
            "watershed_supervoxels": int(sv.max(initial=0)),
            "final_instances": int(out.max(initial=0)),
            "foreground_voxels": int(fg.sum()),
            "output_foreground_voxels": int((out > 0).sum()),
        }
    )
    return (out, diagnostics) if return_diagnostics else out


def _affinity_subsplit(m, sep, core, T, seed_core, seed_bnd, min_vox):
    """Oversegment ONE base-instance mask `m` at affinity-separation ridges (seeded at high-core,
    low-separation interiors), then avg-linkage agglomerate supervoxels by mean interface separation,
    keeping a split only if its interface sep >= T. Returns an int label map (0 outside m).
    Conservative by construction: if <2 confident seeds are found, returns m as a single piece (no
    split) -> a base instance is broken ONLY when the affinity gives >=2 separated high-confidence
    interiors AND the interface between them survives the T gate."""
    fg = m
    seeds = (core > seed_core) & (sep < seed_bnd) & fg
    markers, nm = ndi.label(seeds)
    if nm <= 1:
        return fg.astype(np.int32)                # not enough evidence -> keep merged (precision-safe)
    sv = watershed(sep.astype(np.float32), markers, mask=fg)
    if sv.max() <= 1:
        return (sv > 0).astype(np.int32)
    rag = rag_boundary(sv, sep.astype(np.float32))
    if 0 in rag.nodes:
        rag.remove_node(0)
    merged = merge_hierarchical(sv, rag, thresh=T, rag_copy=False, in_place_merge=True,
                                merge_func=_merge_boundary, weight_func=_weight_boundary)
    out = merged.astype(np.int32) + 1
    out[~fg] = 0
    return out


def decode_fusion(base, abbc_probs, affinities, T=0.45, min_vox=250, short_idx=(0, 1, 2),
                  seed_core=0.5, seed_bnd=0.20, ridge_sep=0.5, min_ridge_vox=1500):
    """[FUSION] Refine a V302 base partition by sub-splitting each base instance ONLY where the
    LEARNED affinity (V308) confirms a true internal fracture surface. The split is CONFINED to each
    base instance's voxels -- a base boundary is NEVER crossed -> precision floor = V302; recall
    rises only where V302 merged 2 touching fragments that affinity can separate.
      base       : int instance map (0=bg) from the DEPLOYED V302 core-seed watershed decode
      abbc_probs : [4,Z,Y,X] softmax (bg,border,boundary,core) -- core[3] seeds the oversegment
      affinities : [K,Z,Y,X] sigmoid same-instance prob (loss.AFFINITY_HEAD_OFFSETS order)
    T = separation gate (higher = more conservative, closer to V302).

    REAL-FRACTURE GATE (ridge_sep / min_ridge_vox): a base instance is sub-split ONLY if it contains
    a COHERENT high-separation surface -- the LARGEST connected component of {sep >= ridge_sep} inside
    it must exceed min_ridge_vox. Diagnostic (2026-06-25) proved the V308 phantom over-splits are
    single true fragments with sep~0 everywhere (frac>0.5 ~ 0.000, only scattered noise specks) whose
    core channel SPECKLES into many blobs -> the seed-driven watershed shreds them. A true merge has
    ~10k-25k coherent high-sep voxels (294/Femur frac>0.5 = 0.126). This gate keeps the real splits
    and rejects the speckle phantoms (116/RightHip <100 high-sep voxels)."""
    base = np.asarray(base).astype(np.int32)
    abbc = np.asarray(abbc_probs, dtype=np.float32)
    core = abbc[3]
    short = np.asarray(affinities, dtype=np.float32)[list(short_idx)]
    sep = (1.0 - short.min(axis=0)).astype(np.float32)
    hi = sep >= float(ridge_sep)
    out = np.zeros_like(base)
    nxt = 1
    for lab in [int(v) for v in np.unique(base) if v != 0]:
        m = base == lab
        if int(m.sum()) < 2 * int(min_vox):       # too small to be 2 real fragments -> keep as-is
            out[m] = nxt; nxt += 1; continue
        # REAL-FRACTURE GATE: require a coherent high-sep surface (not scattered speckle noise)
        hi_m = hi & m
        if hi_m.sum() < min_ridge_vox:
            out[m] = nxt; nxt += 1; continue       # sep~0 single fragment -> keep V302 (no phantom)
        _hl, _hn = ndi.label(hi_m)
        if _hn == 0 or int(np.bincount(_hl.ravel())[1:].max()) < min_ridge_vox:
            out[m] = nxt; nxt += 1; continue       # only scattered specks, no coherent surface -> keep
        sub = _affinity_subsplit(m, sep, core, T, seed_core, seed_bnd, min_vox)
        sub = _drop_small(np.where(m, sub, 0), min_vox)
        sub_ids = [int(v) for v in np.unique(sub) if v != 0]
        if len(sub_ids) <= 1:                     # affinity found no real internal fracture -> keep V302
            out[m] = nxt; nxt += 1; continue
        for s in sub_ids:                         # adopt the affinity-confirmed split
            out[(sub == s) & m] = nxt; nxt += 1
    out, _ = _relabel(out)
    return out


if __name__ == "__main__":
    import sys, glob, json, SimpleITK as sitk
    RANGES = {'Sacrum': (1, 50), 'LeftHip': (51, 100), 'RightHip': (101, 150), 'Femur': (151, 200)}
    cases = sys.argv[1:] or ["294", "256", "271", "286", "290", "021"]
    T = float(__import__('os').environ.get("AGGLO_T", "0.45"))
    def gtcount(c):
        g = glob.glob(f'{__import__("os").environ.get("PENGWIN_ROOT", ".")}/data/task1_2/extracted/*/{c}/label.mha')
        a = sitk.GetArrayFromImage(sitk.ReadImage(g[0]))
        return {n: len([i for i in np.unique(a) if lo <= i <= hi]) for n, (lo, hi) in RANGES.items()
                if any(lo <= i <= hi for i in np.unique(a))}
    print(f"=== Tier-0 agglo decode (T={T}) vs GT fragment count (per anatomy) ===")
    for c in cases:
        gc = gtcount(c)
        probs_files = sorted(glob.glob(f'/tmp/probs/{c}/probs/probs_*.npy'))
        row = []
        for pf in probs_files:
            anat = pf.split('probs_')[-1].replace('.npy', '')
            p = np.load(pf).astype(np.float32)
            inst = decode_agglo(p, T=T)
            n = len([i for i in np.unique(inst) if i != 0])
            row.append(f"{anat}:agglo={n}/GT={gc.get(anat,'?')}")
        print(f"  case {c}: " + "  ".join(row))
