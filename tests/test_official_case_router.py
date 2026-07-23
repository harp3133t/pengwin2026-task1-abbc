"""Regression tests for the PENGWIN 2026 Task 1/2 updated case router."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import SimpleITK as sitk


INFERENCE_PATH = Path(__file__).parents[1] / "inference" / "inference.py"
SPEC = importlib.util.spec_from_file_location("pengwin_inference", INFERENCE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_get_image_info_uses_published_numpy_axis_mapping(tmp_path):
    # NumPy order is (z, y, x), whereas SimpleITK spacing is returned as (x, y, z).
    # The notice explicitly maps sp[0] -> spacing_z and sp[2] -> spacing_x.
    arr = np.zeros((100, 4, 200), dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.60, 0.80, 1.20))
    image_path = tmp_path / "axis_regression.mha"
    sitk.WriteImage(img, str(image_path))

    info = MODULE.get_image_info(image_path)

    assert (info["dim_z"], info["dim_y"], info["dim_x"]) == (100, 4, 200)
    assert info["spacing_z"] == 0.60
    assert info["spacing_y"] == 0.80
    assert info["spacing_x"] == 1.20
    assert info["physical_z_mm"] == 60.0
    assert info["physical_x_mm"] == 240.0

    # Updated mapping: femur. The previous direct SimpleITK (x,y,z) interpretation
    # would enter spacing_x <= 0.71 and incorrectly return pelvic for this example.
    family = MODULE.classify_pelvic_femur(
        info["spacing_x"],
        info["spacing_y"],
        info["spacing_z"],
        info["physical_x_mm"],
        info["physical_z_mm"],
    )
    assert family == "femur"


def test_published_decision_tree_branches_and_inclusive_thresholds():
    classify = MODULE.classify_pelvic_femur

    # physical_x_mm <= 285.35
    assert classify(0.71, 1.00, 1.00, 285.35, 999.0) == "pelvic"
    assert classify(0.72, 1.00, 0.90, 285.35, 999.0) == "femur"
    assert classify(0.72, 0.91, 0.91, 285.35, 999.0) == "pelvic"
    assert classify(0.72, 0.92, 0.91, 285.35, 999.0) == "femur"

    # physical_x_mm > 285.35 and spacing_z <= 0.68
    assert classify(1.00, 1.00, 0.68, 285.36, 193.55) == "pelvic"
    assert classify(1.00, 1.00, 0.68, 285.36, 193.56) == "femur"

    # physical_x_mm > 285.35 and spacing_z > 0.68
    assert classify(1.00, 1.00, 0.69, 285.36, 390.78) == "pelvic"
    assert classify(1.00, 1.00, 0.69, 285.36, 390.79) == "femur"


def test_route_selects_only_the_published_case_family(tmp_path):
    arr = np.zeros((100, 4, 200), dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.60, 0.80, 1.20))
    image_path = tmp_path / "femur.mha"
    sitk.WriteImage(img, str(image_path))

    route, anatomies = MODULE.route_from_official_case_rule(image_path)

    assert route == "official-rule:femur"
    assert anatomies == ("Femur",)
