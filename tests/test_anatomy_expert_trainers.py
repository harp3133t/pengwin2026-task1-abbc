from code_task1.core import (
    PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal,
    PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal,
    PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal,
)


def test_anatomy_expert_key_filters_are_disjoint_and_complete():
    keys = [
        "PENGWIN_001_Sacrum",
        "PENGWIN_001_LeftHip",
        "PENGWIN_001_RightHip",
        "PENGWIN_251_Femur",
    ]
    trainers = (
        PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal,
        PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal,
        PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal,
    )
    selected = {
        trainer.EXPERT_NAME: [
            key for key in keys if trainer._matches_expert(key)
        ]
        for trainer in trainers
    }
    assert selected == {
        "sacrum": ["PENGWIN_001_Sacrum"],
        "hip": ["PENGWIN_001_LeftHip", "PENGWIN_001_RightHip"],
        "femur": ["PENGWIN_251_Femur"],
    }
    assert sorted(sum(selected.values(), [])) == sorted(keys)
