"""Deployment-contract tests for the v3.6.1 split-aware RF candidate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from split_candidate_rf_gate import apply_selected, predict_and_select  # noqa: E402


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(x), self.value, dtype=np.float64)


def test_dockerfile_selects_v361_split_aware_rf():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "PENGWIN_A1_PROGRESSIVE_DECODE=0" in dockerfile
    assert "PENGWIN_SPLIT_AWARE_RF_DECODE=1" in dockerfile
    assert (
        "PENGWIN_SPLIT_AWARE_RF_PATH=/opt/ml/model/split_candidate_rf/"
        "v35_candidate_split_aware_rf_all_candidates.joblib"
    ) in dockerfile
    assert "PENGWIN_AGGLO_T=0.75" in dockerfile


def test_runtime_builds_safe_base_then_proposal_then_rf_gate():
    source = (INFERENCE_DIR / "inference.py").read_text()
    base = source.index("decoded_pp = decode_affinity_agglo(")
    proposal = source.index("_proposal_pp, _abbc_report = refine_instances_with_full_abbc(")
    gate = source.index("full_label, split_report = run_split_candidate_rf_gate(")
    assert base < proposal < gate
    assert (INFERENCE_DIR / "split_candidate_rf_gate.py").is_file()


def test_gate_selects_one_candidate_per_source_and_preserves_support():
    rows = [
        {
            "anatomy": "Sacrum",
            "source_id": 1,
            "proposal_id": 3,
            "invert_proposal": 0,
            "feature": 0.0,
        },
        {
            "anatomy": "Sacrum",
            "source_id": 1,
            "proposal_id": 4,
            "invert_proposal": 0,
            "feature": 1.0,
        },
    ]
    payload = {
        "models": {
            "delta_merge": ConstantRegressor(-0.8),
            "delta_split": ConstantRegressor(1.5),
            "delta_dice": ConstantRegressor(0.01),
            "delta_f1": ConstantRegressor(0.0),
            "delta_precision": ConstantRegressor(0.0),
        },
        "feature_names": ["feature"],
        "policy": {"action": "select", "merge_max": -0.55, "split_max": 2.25},
        "quality_guards": {
            "delta_dice_min": -0.01,
            "delta_f1_min": -0.05,
            "delta_precision_min": -0.10,
        },
    }
    selected = predict_and_select(rows, payload)
    assert len(selected) == 1
    assert selected[0]["proposal_id"] == 3

    base = np.zeros((4, 4, 4), dtype=np.uint16)
    base[1:3, 1:3, 1:3] = 1
    proposal = np.zeros_like(base)
    proposal[1, 1:3, 1:3] = 3
    proposal[2, 1:3, 1:3] = 4
    output, applied = apply_selected(base, proposal, selected)
    assert len(applied) == 1
    assert np.array_equal(output > 0, base > 0)
    assert len(np.unique(output[output > 0])) == 2
