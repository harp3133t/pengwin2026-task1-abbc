"""Static neural-weight and decoder prerequisites for v3.6.3."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_selects_the_evaluated_candidate():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "chmod -R a+rX /opt/app/inference /opt/app/code_task1" in dockerfile
    assert (
        "PENGWIN_DS538_TRAINER="
        "PengwinTrainerSTUNetBaseAffinityV308DeployedVal"
    ) in dockerfile
    assert (
        "PENGWIN_DS538_TRAINER_SACRUM="
        "PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal"
    ) in dockerfile
    assert (
        "PENGWIN_DS538_TRAINER_HIP="
        "PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal"
    ) in dockerfile
    assert (
        "PENGWIN_DS538_TRAINER_FEMUR="
        "PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal"
    ) in dockerfile
    assert "PENGWIN_DS538_FOLD=0" in dockerfile
    assert "PENGWIN_DS538_OUT_CH=13" in dockerfile
    assert "PENGWIN_AFFINITY_DECODE=1" in dockerfile
    assert "PENGWIN_A1_PROGRESSIVE_DECODE=0" in dockerfile
    assert "PENGWIN_SPLIT_AWARE_RF_DECODE=0" in dockerfile
    assert "PENGWIN_GUARDED_SEED_DECODE=1" in dockerfile
    assert "PENGWIN_STAGE1_INSTANCE_FILL=1" in dockerfile
    assert "PENGWIN_AGGLO_T=0.75" in dockerfile
    assert "PENGWIN_TARGET_ROUTER=1" in dockerfile
    assert "PENGWIN_RF_CONF_MARGIN=0.15" in dockerfile


def test_guarded_seed_decoder_is_vendored_and_wired_after_safe_base():
    inference_dir = REPO_ROOT / "inference"
    for name in (
        "multiscale_affinity_rag_decode.py",
        "abbc_full_refine_decode.py",
        "abbc_conservative_refine_decode.py",
        "abbc_progressive_affinity_merge_decode.py",
    ):
        assert (inference_dir / name).is_file(), name

    source = (inference_dir / "inference.py").read_text()
    base = source.index("_v35_pp = decode_affinity_agglo(")
    guarded = source.index("decoded_pp, _guarded_report = refine_instances_conservatively(")
    assert base < guarded
    assert "split_passes=2" in source
    assert "min_split_piece_mm3=1000.0" in source
    assert "min_core_voxels_per_piece=50" in source
    assert "split_affinity_required_ranges=3" in source
    assert "hard_merge_margin=1.0e9" in source


def test_candidate_trainer_is_exportable_by_the_nnunet_shim():
    core_path = REPO_ROOT / "code_task1" / "core.py"
    spec = importlib.util.spec_from_file_location("pengwin_candidate_core", core_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    candidate = module.PengwinTrainerSTUNetBaseAffinityV308DeployedVal
    base = module.PengwinTrainerSTUNetBaseAffinityV308
    assert issubclass(candidate, base)
