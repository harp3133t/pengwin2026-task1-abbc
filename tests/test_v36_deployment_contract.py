"""Static deployment-contract checks for the v3.6 A1 candidate."""

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
    assert "PENGWIN_A1_PROGRESSIVE_DECODE=1" in dockerfile
    assert "PENGWIN_AGGLO_T=0.75" in dockerfile
    assert "PENGWIN_TARGET_ROUTER=1" in dockerfile
    assert "PENGWIN_RF_CONF_MARGIN=0.15" in dockerfile


def test_a1_decoder_is_vendored_and_wired_in_order():
    inference_dir = REPO_ROOT / "inference"
    for name in (
        "multiscale_affinity_rag_decode.py",
        "abbc_full_refine_decode.py",
        "abbc_conservative_refine_decode.py",
        "abbc_progressive_affinity_merge_decode.py",
    ):
        assert (inference_dir / name).is_file(), name

    source = (inference_dir / "inference.py").read_text()
    initial = source.index("decode_affinity_multiscale_rag_veto(")
    abbc = source.index("refine_instances_with_full_abbc(")
    a1 = source.index("apply_progressive_affinity_merge(")
    assert initial < abbc < a1
    assert "small_fragment_max_mm3=5000.0" in source
    assert "small_required_ranges=3" in source
    assert "enable_small_veto=False" in source
    assert "mutual_best_single_round=False" in source


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
