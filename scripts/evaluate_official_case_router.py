#!/usr/bin/env python3
"""Evaluate the published PENGWIN pelvic/femur rule on the 340-case training set."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
INFERENCE_PATH = REPO_ROOT / "inference" / "inference.py"
SPEC = importlib.util.spec_from_file_location("pengwin_inference", INFERENCE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Directory containing PENGWIN26_task1_2_train_part*/<case>/image.mha",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = sorted(args.dataset_root.glob("PENGWIN26_task1_2_train_part*/[0-9]*/image.mha"))
    if len(images) != 340:
        raise RuntimeError(f"expected 340 training images, found {len(images)}")

    confusion: Counter[tuple[str, str]] = Counter()
    failures: list[dict] = []
    for image_path in images:
        case_id = int(image_path.parent.name)
        # The official training release contains 170 pelvic IDs in 001..200
        # (121..150 absent) and 170 femur IDs in 251..420.
        true_family = "pelvic" if case_id <= 200 else "femur"
        info = MODULE.get_image_info(image_path)
        pred_family = MODULE.classify_pelvic_femur(
            info["spacing_x"],
            info["spacing_y"],
            info["spacing_z"],
            info["physical_x_mm"],
            info["physical_z_mm"],
        )
        confusion[(true_family, pred_family)] += 1
        if pred_family != true_family:
            failures.append(
                {
                    "case_id": f"{case_id:03d}",
                    "true": true_family,
                    "pred": pred_family,
                    **info,
                }
            )

    correct = sum(count for (true, pred), count in confusion.items() if true == pred)
    result = {
        "n_cases": len(images),
        "correct": correct,
        "accuracy": correct / len(images),
        "confusion": {
            f"{true}_to_{pred}": confusion[(true, pred)]
            for true in ("pelvic", "femur")
            for pred in ("pelvic", "femur")
        },
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
