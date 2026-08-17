"""Deployment and behavior checks for the v3.6.3 Stage-1 instance fill."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from stage1_instance_reconcile import fill_anatomy_support  # noqa: E402


def test_fills_between_instances_without_touching_other_anatomy():
    labels = np.zeros((12, 12, 24), dtype=np.uint16)
    labels[2:10, 2:10, 2:7] = 51
    labels[2:10, 2:10, 17:22] = 52
    labels[0:2, 0:2, 0:2] = 101
    stage1 = np.zeros(labels.shape, dtype=bool)
    stage1[2:10, 2:10, 2:22] = True

    output, report = fill_anatomy_support(labels, stage1, (51, 100))

    assert np.all(output[labels == 101] == 101)
    assert np.all(output[labels == 51] > 0)
    assert np.all(output[labels == 52] > 0)
    assert report["source_instances"] == report["output_instances"] == 2
    assert report["filled_voxels"] > 0
    assert report["stage1_missing_after"] == 0


def test_does_not_invent_instance_for_unseeded_component():
    labels = np.zeros((12, 12, 24), dtype=np.uint16)
    labels[2:6, 2:6, 2:6] = 51
    stage1 = labels == 51
    stage1[8:11, 8:11, 18:21] = True

    output, report = fill_anatomy_support(labels, stage1, (51, 100))

    assert np.all(output[8:11, 8:11, 18:21] == 0)
    assert report["unseeded_stage1_voxels"] == 27


def test_dockerfile_and_runtime_order_match_evaluated_candidate():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "PENGWIN_GUARDED_SEED_DECODE=1" in dockerfile
    assert "PENGWIN_STAGE1_INSTANCE_FILL=1" in dockerfile
    assert "PENGWIN_STAGEA_BONE_RECONCILE=0" in dockerfile
    assert "PENGWIN_SPLIT_AWARE_RF_DECODE=0" in dockerfile

    source = (INFERENCE_DIR / "inference.py").read_text()
    guarded = source.index(
        "decoded_pp, _guarded_report = refine_instances_conservatively("
    )
    fill = source.index("full_label, fill_report = fill_anatomy_support(")
    orientation = source.index("if orientation_code(ref_img) != \"LPS\":")
    assert guarded < fill < orientation
    assert "output_instances" in source
    assert "unseeded_stage1_voxels" in source
