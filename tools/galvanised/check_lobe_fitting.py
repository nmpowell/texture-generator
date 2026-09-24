"""Reproduce bounded offline lobe-fitting coupons from the source checkout.

Run with the development dependencies installed. The disjoint-angle coupon
fixtures live in tests; this is a scientific diagnostic, not a production fitter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import time
from pathlib import Path

import numpy as np

from texture_generators.core.conductor import resource_hashes
from texture_generators.core.lobe_fitting import (
    FIT_VERSION,
    FitTolerances,
    LobeFitError,
    fit_material_lobes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    helpers = runpy.run_path(str(repository / "tests/test_galvanised_lobe_fitting.py"))
    response = helpers["_physical_responses"]
    strict = FitTolerances(relative_rms=1e-8, dark_absolute_rms=1e-9)
    reports = {}
    arrays: dict[str, np.ndarray] = {}
    axes = np.arange(4) * np.pi / 4
    train = response(0, axes, np.zeros(4))
    valid = response(0, axes, np.zeros(4), held_out=True)
    weights = np.array([0.5, 0, 0.5, 0])
    target = weights @ train
    held = weights @ valid
    start = time.perf_counter()
    for _pixel in range(32 * 32):
        fit = fit_material_lobes(
            train, target, valid, held, material_id=0, coverage=1.0, tolerances=strict
        )
    seconds = time.perf_counter() - start
    reports["orthogonal_32x32"] = {
        "pixels": 1024,
        "candidate_count": 4,
        "training_scalar_samples": train.shape[1],
        "held_out_scalar_samples": valid.shape[1],
        "seconds": seconds,
        "per_pixel_seconds": seconds / 1024,
        "fit": fit.to_mapping(),
    }
    arrays.update(
        orthogonal_train=train,
        orthogonal_valid=valid,
        orthogonal_target=held,
        orthogonal_fitted=np.array(fit.weights) @ valid[list(fit.candidate_indices)],
    )
    for swapped in [False, True]:
        total = np.zeros(valid.shape[1])
        targets = np.zeros(valid.shape[1])
        fits = []
        for material in [0, 1]:
            a = response(material, [0, 0], [-0.15, 0.15])
            b = response(material, [0, 0], [-0.15, 0.15], held_out=True)
            chosen = material if not swapped else 1 - material
            fit = fit_material_lobes(
                a,
                0.5 * a[chosen],
                b,
                0.5 * b[chosen],
                material_id=material,
                coverage=0.5,
                tolerances=strict,
            )
            total += np.array(fit.weights) @ b[list(fit.candidate_indices)]
            targets += 0.5 * b[chosen]
            fits.append(fit.to_mapping())
        key = "slope_swapped" if swapped else "slope_original"
        reports[key] = {"fits": fits}
        arrays[key + "_target"] = targets
        arrays[key + "_fitted"] = total
    axes = np.tile(np.arange(8) * np.pi / 8, 8)
    slopes = np.repeat(np.linspace(-0.18, 0.18, 8), 8)
    a = response(0, axes, slopes)
    b = response(0, axes, slopes, held_out=True)
    w = np.zeros(64)
    w[[0, 17, 42, 63]] = [0.15, 0.25, 0.35, 0.25]
    start = time.perf_counter()
    try:
        fit = fit_material_lobes(a, w @ a, b, w @ b, material_id=0, coverage=1.0)
        status = "accepted"
    except LobeFitError as error:
        fit = error.result
        status = "rejected"
    reports["candidate64"] = {
        "pixels": 1,
        "candidate_count": 64,
        "seconds": time.perf_counter() - start,
        "status": status,
        "source_indices": [0, 17, 42, 63],
        "source_weights": [0.15, 0.25, 0.35, 0.25],
        "fit": fit.to_mapping(),
    }
    arrays.update(
        candidate64_train=a,
        candidate64_valid=b,
        candidate64_source_weights=w,
        candidate64_target=w @ b,
        candidate64_fitted=np.array(fit.weights) @ b[list(fit.candidate_indices)],
    )
    adversarial_weights = helpers["_physical_local_minimum_weights"]()
    start = time.perf_counter()
    try:
        fit = fit_material_lobes(
            a,
            adversarial_weights @ a,
            b,
            adversarial_weights @ b,
            material_id=0,
            coverage=1.0,
        )
        status = "accepted"
    except LobeFitError as error:
        fit = error.result
        status = "rejected"
    reports["candidate64_local_minimum"] = {
        "pixels": 1,
        "candidate_count": 64,
        "seconds": time.perf_counter() - start,
        "status": status,
        "source_indices": np.flatnonzero(adversarial_weights).tolist(),
        "source_weights": adversarial_weights[adversarial_weights > 0].tolist(),
        "fit": fit.to_mapping(),
        "interpretation": "An exact four-source solution exists; rejection exposes the single-swap search limit.",
    }
    arrays.update(
        adversarial64_source_weights=adversarial_weights,
        adversarial64_target=adversarial_weights @ b,
        adversarial64_fitted=np.array(fit.weights) @ b[list(fit.candidate_indices)],
    )
    a, b, target = helpers["_local_minimum_responses"]()
    start = time.perf_counter()
    try:
        fit = fit_material_lobes(a, target, b, target, material_id=0, coverage=1.0)
        status = "accepted"
    except LobeFitError as error:
        fit = error.result
        status = "rejected"
    reports["synthetic_local_minimum"] = {
        "pixels": 1,
        "candidate_count": 6,
        "seconds": time.perf_counter() - start,
        "status": status,
        "source_indices": [4, 5],
        "source_weights": [0.5, 0.5],
        "fit": fit.to_mapping(),
        "interpretation": "Four tetrahedral decoys trap single replacements despite an exact two-source solution.",
    }
    arrays.update(
        adversarial_synthetic_train=a,
        adversarial_synthetic_valid=b,
        adversarial_synthetic_target=target,
        adversarial_synthetic_fitted=np.array(fit.weights)
        @ b[list(fit.candidate_indices)],
    )
    source_paths = [
        "src/texture_generators/core/lobe_fitting.py",
        "src/texture_generators/core/material_render.py",
        "src/texture_generators/core/microfacet.py",
        "src/texture_generators/core/energy_compensation.py",
        "tests/test_galvanised_lobe_fitting.py",
        "tools/galvanised/check_lobe_fitting.py",
    ]
    reports["provenance"] = {
        "fit_version": FIT_VERSION,
        "module_sha256": hashlib.sha256(
            (repository / "src/texture_generators/core/lobe_fitting.py").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "optical_hashes": dict(resource_hashes()),
        "training": "4 polar angles [0.04,0.12,0.25,0.50]rad x16 azimuths, zero phase; linear RGB*positive incident cosine",
        "held_out": "4 disjoint polar angles [0.07,0.18,0.40,0.80]rad x19 azimuths, phase0.37; normal outgoing view",
        "purpose": "controlled coupons only; not sampler integration, not highlight-width or full G7 acceptance",
    }
    np.savez(root / "responses.npz", allow_pickle=False, **arrays)
    (root / "report.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False) + "\n"
    )
    print(root / "report.json")


if __name__ == "__main__":
    main()
