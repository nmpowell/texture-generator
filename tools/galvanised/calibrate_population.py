"""Offline exact-area population fitting; never runs during ordinary rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from texture_generators.core.grains import (
    GrainPartition,
    growth_weights,
    initial_nucleus_count,
    place_grains,
)
from texture_generators.core.random_fields import (
    make_rng,
    material_key_from_rng,
    topology_rng,
)


def calibrate(
    *,
    seeds: int = 12,
    candidates: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9, 1.2, 1.285, 1.5),
    size_mm: tuple[float, float] = (100.0, 100.0),
    diameter_mm: float = 8.0,
    targets: tuple[float, ...] = (0.20, 0.28),
) -> dict[str, object]:
    if seeds < 12 or not 1 <= len(candidates) <= 32:
        raise ValueError("offline fitting requires >=12 seeds and <=32 candidates")
    count = initial_nucleus_count(size_mm, diameter_mm)
    outputs: list[dict[str, object]] = []
    for sigma in candidates:
        diameters = []
        cvs = []
        for seed in range(seeds):
            key = material_key_from_rng(make_rng(seed))
            points = place_grains(
                size_mm,
                diameter_mm,
                topology_rng(key, "nucleus_placement"),
                count=count,
                minimum_distance_mm=0.35 * diameter_mm,
            )
            weights = growth_weights(
                count,
                diameter_mm,
                topology_rng(key, "nucleus_growth"),
                growth_sigma=sigma,
            )
            stats = GrainPartition(points, weights, size_mm).area_statistics()
            diameters.append(stats.median_diameter_mm)
            cvs.append(stats.diameter_cv)
        outputs.append(
            {
                "growth_sigma": sigma,
                "median_diameter_mean_mm": float(np.mean(diameters)),
                "diameter_cv_mean": float(np.mean(cvs)),
                "diameter_cv_sd_between_tiles": float(np.std(cvs, ddof=1)),
                "diameter_cv_values": cvs,
            }
        )
        print(f"sigma={sigma:.3f} CV={np.mean(cvs):.4f}", flush=True)
    fits = []
    for target in targets:
        winner = min(
            outputs,
            key=lambda item: abs(cast(float, item["diameter_cv_mean"]) - target),
        )
        fits.append(
            {
                "target_cv": target,
                "growth_sigma": winner["growth_sigma"],
                "mean_cv": winner["diameter_cv_mean"],
                "absolute_error": abs(cast(float, winner["diameter_cv_mean"]) - target),
            }
        )
    return {
        "method": "periodic exact geometric cell areas; unweighted positive-area equivalent diameters",
        "seeds": seeds,
        "candidates": len(candidates),
        "size_mm": size_mm,
        "diameter_mm": diameter_mm,
        "nuclei": count,
        "candidate_results": outputs,
        "fits": fits,
        "status": "single-scale offline authoring fit; not specimen or cross-scale calibration",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument(
        "--out", type=Path, default=Path("data/galvanised/population_calibration.json")
    )
    args = parser.parse_args()
    result = calibrate(seeds=args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
