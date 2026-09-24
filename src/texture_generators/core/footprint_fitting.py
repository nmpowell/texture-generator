"""Offline physical-footprint extraction and checked candidate-lobe fitting.

The fixed normal-view angular coupon is deliberately narrower than a complete
material response domain. Nothing in this module enables sampler quality tiers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from texture_generators.core.conductor import resource_hashes
from texture_generators.core.energy_compensation import energy_resource_hashes
from texture_generators.core.lobe_fitting import (
    FIT_VERSION,
    FitTolerances,
    LobeFitError,
    _error,
    fit_material_lobes,
)
from texture_generators.core.material import LOBE_DTYPE, MaterialMaps
from texture_generators.core.material_render import (
    ReferenceDerivativeConvergenceError,
    _bsdf,
    _converged_reference_normals,
    _Optics,
    _parameters,
    _reference_derivative_step_mm,
)
from texture_generators.core.microfacet import openpbr_widths, tangent_frame
from texture_generators.materials.galvanised import (
    GENERATOR_VERSION,
    GalvanisedState,
    optical_parameters,
    sample_points,
    surface_resource_hashes,
)

ADAPTER_VERSION = "physical-footprint-normal-view-1"
ANGLE_VERSION = "normal-view-disjoint-coupon-1"
MAX_FOOTPRINTS = 16
MAX_POINT_BATCH = 64
SPATIAL_TOLERANCES = FitTolerances(0.01, 0.0002, 0.02)
FIT_TOLERANCES = FitTolerances()
COVERAGE_TOLERANCE = 0.001
_COVERAGE_NAMES = ("metallic", "patina_coverage", "white_stain_coverage")
_CANDIDATE_DTYPE = np.dtype(
    [
        ("source_index", "<u2"),
        ("position_mm", "<f8", (2,)),
        ("coverage", "<f8"),
        ("material_id", "u1"),
        ("normal_ts", "<f8", (3,)),
        ("tangent_ts", "<f8", (3,)),
        ("alpha_t", "<f8"),
        ("alpha_b", "<f8"),
        ("optical_parameter_index", "<u4"),
    ]
)


@dataclass(frozen=True)
class PhysicalFootprint:
    center_mm: tuple[float, float]
    size_mm: tuple[float, float] = (0.125, 0.125)

    def __post_init__(self) -> None:
        for name in ("center_mm", "size_mm"):
            value = np.asarray(getattr(self, name))
            if value.shape != (2,) or value.dtype.kind not in "fiu":
                raise ValueError(f"{name} must contain two finite real lengths")
            if not np.all(np.isfinite(value)) or (
                name == "size_mm" and np.any(value <= 0)
            ):
                raise ValueError(f"{name} must contain finite, positive sizes")
            object.__setattr__(self, name, tuple(float(v) for v in value))


@dataclass(frozen=True)
class PhysicalFootprintFit:
    """Accepted records, or a rejected diagnostic with no usable records."""

    report: dict[str, Any]
    arrays: dict[str, np.ndarray]
    records: np.ndarray

    @property
    def accepted(self) -> bool:
        return bool(self.report["accepted"])


class PhysicalFootprintFitError(ValueError):
    def __init__(self, result: PhysicalFootprintFit) -> None:
        self.result = result
        super().__init__(
            f"physical footprint rejected at {result.report['stage']}: "
            f"{result.report['reason']}"
        )


def angular_directions(*, held_out: bool = False) -> np.ndarray:
    """The fixed disjoint coupon grid; outgoing direction is always +Z."""
    count = 19 if held_out else 16
    phase = 0.37 if held_out else 0.0
    polar = [0.07, 0.18, 0.40, 0.80] if held_out else [0.04, 0.12, 0.25, 0.50]
    theta, phi = np.meshgrid(
        polar, 2.0 * np.pi * (np.arange(count) + phase) / count, indexing="ij"
    )
    return np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
    ).reshape(-1, 3)


def _points(footprint: PhysicalFootprint, rate: int) -> np.ndarray:
    offsets = (np.arange(rate, dtype=np.float64) + 0.5) / rate - 0.5
    xx, yy = np.meshgrid(
        footprint.center_mm[0] + offsets * footprint.size_mm[0],
        footprint.center_mm[1] + offsets * footprint.size_mm[1],
    )
    return np.stack((xx, yy), axis=-1).reshape(-1, 2)


def _responses(
    records: np.ndarray, directions: np.ndarray, optical: _Optics
) -> np.ndarray:
    """Unit-coverage BSDF*cos response, including renderer tangent projection."""
    if len(records) > MAX_POINT_BATCH or len(directions) * 3 > 4096:
        raise ValueError("response batch exceeds the fixed offline work bound")
    if not len(records):
        return np.empty((0, len(directions) * 3))
    normal = records["normal_ts"].astype(np.float64)
    tangent = records["tangent_ts"].astype(np.float64)
    tangent -= np.sum(tangent * normal, axis=-1)[:, None] * normal
    tangent /= np.linalg.norm(tangent, axis=-1)[:, None]
    response = _bsdf(
        normal[:, None, :],
        tangent[:, None, :],
        directions[None, :, :],
        np.array([0.0, 0.0, 1.0]),
        records["alpha_t"].astype(np.float64)[:, None],
        records["alpha_b"].astype(np.float64)[:, None],
        optical,
    )
    cosine = np.maximum(np.sum(normal[:, None, :] * directions[None, :, :], -1), 0)
    result = (response * cosine[..., None]).reshape(len(records), -1)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("physical angular response must be finite and nonnegative")
    return result


def _point_records(
    state: GalvanisedState,
    positions: np.ndarray,
    first: int,
    optics: tuple[_Optics, ...],
    step: tuple[float, float],
) -> tuple[list[np.ndarray], dict[str, int | float]]:
    fields = sample_points(state, positions[:, 0], positions[:, 1])
    normal, derivative = _converged_reference_normals(
        state, positions[:, 0], positions[:, 1], step
    )
    result = []
    for index, optical in enumerate(optics):
        coverage = fields[_COVERAGE_NAMES[optical.material_id]]
        active = coverage > 0.0
        records = np.empty(np.count_nonzero(active), dtype=_CANDIDATE_DTYPE)
        records["source_index"] = np.arange(first, first + len(positions))[active]
        records["position_mm"] = positions[active]
        records["coverage"] = coverage[active]
        records["material_id"] = optical.material_id
        records["optical_parameter_index"] = index
        records["normal_ts"] = normal[active]
        if optical.material_id == 0:
            axis = fields["anisotropy_axis"][active]
            roughness = fields["intrinsic_roughness"][active]
            anisotropy = fields["intrinsic_anisotropy"][active]
        else:
            axis = np.broadcast_to([1.0, 0.0], (len(records), 2))
            roughness = np.full(len(records), optical.roughness)
            anisotropy = np.full(len(records), optical.anisotropy)
        records["tangent_ts"], _ = tangent_frame(normal[active], axis)
        records["alpha_t"], records["alpha_b"] = openpbr_widths(roughness, anisotropy)
        result.append(records)
    return result, derivative


def _integrate(
    state: GalvanisedState,
    footprint: PhysicalFootprint,
    rate: int,
    optics: tuple[_Optics, ...],
    step: tuple[float, float],
    batch_size: int,
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    positions = _points(footprint, rate)
    grids = {
        "training": angular_directions(),
        "held_out": angular_directions(held_out=True),
    }
    totals = {name: np.zeros((3, len(grid) * 3)) for name, grid in grids.items()}
    corrections = {name: np.zeros_like(value) for name, value in totals.items()}
    coverage = np.zeros(3)
    coverage_correction = np.zeros(3)
    candidate_parts: list[list[np.ndarray]] = [[], [], []]
    response_parts: dict[str, list[list[np.ndarray]]] = {
        name: [[], [], []] for name in grids
    }
    max_halvings = 0
    max_derivative_error = 0.0
    for first in range(0, len(positions), batch_size):
        records, diagnostic = _point_records(
            state, positions[first : first + batch_size], first, optics, step
        )
        max_halvings = max(max_halvings, int(diagnostic["max_halvings"]))
        max_derivative_error = max(
            max_derivative_error, float(diagnostic["max_accepted_component_difference"])
        )
        for material, candidates in enumerate(records):
            if rate == 8:
                candidate_parts[material].append(candidates)
            # Fixed physical-point accumulation order is independent of batches.
            for value in candidates["coverage"]:
                _accumulate(
                    coverage[material : material + 1],
                    coverage_correction[material : material + 1],
                    value,
                )
            for name, grid in grids.items():
                response = _responses(candidates, grid, optics[material])
                if rate == 8:
                    response_parts[name][material].append(response)
                for value, weight in zip(response, candidates["coverage"], strict=True):
                    _accumulate(
                        totals[name][material],
                        corrections[name][material],
                        weight * value,
                    )
    coverage /= len(positions)
    for name in totals:
        totals[name] /= len(positions)
        arrays[f"target_{rate}_{name}"] = totals[name]
    arrays[f"coverage_{rate}"] = coverage
    if rate == 8:
        for material in range(3):
            arrays[f"candidates_{material}"] = np.concatenate(candidate_parts[material])
            for name in grids:
                arrays[f"candidates_{material}_{name}"] = np.concatenate(
                    response_parts[name][material]
                )
    return (
        coverage,
        totals,
        {
            "rate": rate,
            "points": len(positions),
            "max_derivative_halvings": max_halvings,
            "max_derivative_component_difference": max_derivative_error,
        },
    )


def _accumulate(total: np.ndarray, correction: np.ndarray, value: Any) -> None:
    """Fixed-order compensated sum avoids integration-generated false lobes."""
    increment = value - correction
    updated = total + increment
    correction[...] = (updated - total) - increment
    total[...] = updated


def _compare(
    coverage: np.ndarray,
    targets: dict[str, np.ndarray],
    other_coverage: np.ndarray,
    others: dict[str, np.ndarray],
) -> dict[str, Any]:
    error = float(np.max(np.abs(coverage - other_coverage)))
    checks = {}
    for name, values in targets.items():
        for material in (*range(3), "total"):
            target = values.sum(axis=0) if material == "total" else values[material]
            actual = (
                others[name].sum(axis=0)
                if material == "total"
                else others[name][material]
            )
            checks[f"{name}_{material}"] = _error(
                actual,
                target,
                np.full(len(target), 1.0 / len(target)),
                SPATIAL_TOLERANCES,
            ).to_mapping()
    return {
        "coverage_maximum_change": error,
        "response_errors": checks,
        "passed": error <= COVERAGE_TOLERANCE
        and all(v["passed"] for v in checks.values()),
    }


def _stored_records(
    candidates: np.ndarray,
    indices: tuple[int, ...],
    weights: tuple[float, ...],
    coverage: float,
) -> np.ndarray:
    result = np.zeros(len(indices), dtype=LOBE_DTYPE)
    selected = candidates[list(indices)]
    for name in (
        "normal_ts",
        "tangent_ts",
        "alpha_t",
        "alpha_b",
        "material_id",
        "optical_parameter_index",
    ):
        result[name] = selected[name]
    result["weight"] = weights
    if len(result):
        pivot = int(np.argmax(result["weight"]))
        others = np.delete(result["weight"], pivot).astype(np.float64).sum()
        result["weight"][pivot] = np.float32(coverage - others)
        if (
            np.any(result["weight"] <= 0)
            or abs(float(result["weight"].astype(np.float64).sum()) - coverage) > 1e-7
        ):
            raise ValueError(
                "stored weights cannot close material coverage within 1e-7"
            )
    return result


def _provenance(state: GalvanisedState, footprint: PhysicalFootprint) -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    sources = (
        "core/footprint_fitting.py",
        "core/lobe_fitting.py",
        "core/material_render.py",
        "core/microfacet.py",
        "core/energy_compensation.py",
        "core/conductor.py",
        "core/dendrites.py",
        "core/galvanised_weather.py",
        "core/grains.py",
        "core/_grain_geometry.py",
        "core/random_fields.py",
        "core/physical.py",
        "core/material.py",
        "materials/galvanised.py",
        "materials/galvanised_config.py",
    )
    state_arrays = {
        "micro_spectrum": state.micro_spectrum,
        "defects": state.defects,
        "partition.points_mm": state.partition.points_mm,
        "partition.weights_mm2": state.partition.weights_mm2,
        "partition.nucleus_ids": state.partition.nucleus_ids,
    }
    record_metadata = {
        "population_mapping_version": state.population_mapping_version,
        "partition_size_mm": state.partition.size_mm,
    }
    for name in ("dendrites", "weather"):
        records = getattr(state, name)
        for field in fields(records):
            value = getattr(records, field.name)
            if isinstance(value, np.ndarray):
                state_arrays[f"{name}.{field.name}"] = value
            else:
                record_metadata[f"{name}.{field.name}"] = value
    state_identity = {
        "arrays": {
            name: _array_identity(value) for name, value in state_arrays.items()
        },
        "record_metadata": record_metadata,
        "config": state.config.to_mapping(),
        "material_key": state.material_key,
    }
    return {
        "adapter_version": ADAPTER_VERSION,
        "fitter_version": FIT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "material_key": state.material_key,
        "seed": state.seed,
        "config": state.config.to_mapping(),
        "center_mm": list(footprint.center_mm),
        "footprint_mm": list(footprint.size_mm),
        "angle_grid_version": ANGLE_VERSION,
        "angle_grid": {
            "training_polar_rad": [0.04, 0.12, 0.25, 0.50],
            "training_azimuth_count": 16,
            "training_azimuth_phase": 0.0,
            "held_out_polar_rad": [0.07, 0.18, 0.40, 0.80],
            "held_out_azimuth_count": 19,
            "held_out_azimuth_phase": 0.37,
        },
        "captured_state": state_identity,
        "state_sha256": hashlib.sha256(
            json.dumps(state_identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "view": [0.0, 0.0, 1.0],
        "response_scale": "linear RGB BSDF times positive local incident cosine; no exposure or environment radiance",
        "source_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in sources
        },
        "surface_resource_hashes": surface_resource_hashes(),
        "optical_resource_hashes": dict(resource_hashes()),
        "renderer_resource_hashes": dict(energy_resource_hashes()),
    }


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    identity = {"shape": list(array.shape), "dtype": array.dtype.descr}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    if array.flags.c_contiguous:
        if array.size:
            digest.update(memoryview(array).cast("B"))
    else:
        # Structured/strided captured records may be supplied by diagnostics.
        # Copy only <=64 KiB of complete rows, never the complete state array.
        row_bytes = array.dtype.itemsize * int(np.prod(array.shape[1:]))
        rows = max(1, 65536 // max(1, row_bytes))
        for first in range(0, len(array), rows):
            block = np.ascontiguousarray(array[first : first + rows])
            if block.size:
                digest.update(memoryview(block).cast("B"))
    return {**identity, "sha256": digest.hexdigest()}


def fit_physical_footprint(
    state: GalvanisedState,
    footprint: PhysicalFootprint,
    *,
    point_batch_size: int = MAX_POINT_BATCH,
) -> PhysicalFootprintFit:
    """Fit one explicitly located periodic footprint or raise with diagnostics."""
    if not isinstance(footprint, PhysicalFootprint):
        raise TypeError("footprint must be PhysicalFootprint")
    if (
        isinstance(point_batch_size, bool)
        or not isinstance(point_batch_size, Integral)
        or not 1 <= point_batch_size <= MAX_POINT_BATCH
    ):
        raise ValueError("point_batch_size must be an integer in [1,64]")
    assert state.config.size_mm is not None
    if any(
        length > tile
        for length, tile in zip(footprint.size_mm, state.config.size_mm, strict=True)
    ):
        raise ValueError("offline footprint dimensions must not exceed one state tile")
    footprint = PhysicalFootprint(
        tuple(
            float(v) for v in np.remainder(footprint.center_mm, state.config.size_mm)
        ),
        footprint.size_mm,
    )
    report = _provenance(state, footprint)
    report.update(
        accepted=False,
        stage="extraction",
        spatial_checks=[],
        integration=[],
        material_fits={},
    )
    arrays = {
        "training_directions": angular_directions(),
        "held_out_directions": angular_directions(held_out=True),
    }

    def finish(records: np.ndarray) -> PhysicalFootprintFit:
        for value in (*arrays.values(), records):
            value.setflags(write=False)
        return PhysicalFootprintFit(report, arrays, records)

    def reject(stage: str, reason: str) -> None:
        report.update(stage=stage, reason=reason)
        raise PhysicalFootprintFitError(finish(np.empty(0, dtype=LOBE_DTYPE)))

    optics = _parameters({"optical_parameters": optical_parameters(state)})
    step = _reference_derivative_step_mm(state, None)
    report["derivative_initial_step_mm"] = list(step)
    report["spatial_policy"] = {
        "relative_rms": 0.01,
        "dark_absolute_rms": 0.0002,
        "radiance_floor": 0.02,
        "coverage_component_max": COVERAGE_TOLERANCE,
        "candidate_rate": 8,
        "rates": [8, 16, 17, 32, 33],
        "max_point_batch": MAX_POINT_BATCH,
    }

    def integrate(rate: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        try:
            coverage, targets, detail = _integrate(
                state, footprint, rate, optics, step, point_batch_size, arrays
            )
        except ReferenceDerivativeConvergenceError as error:
            reject("derivative", str(error))
            raise AssertionError("unreachable") from error
        report["integration"].append(detail)
        return coverage, targets

    coarse_c, coarse = integrate(8)
    for rate in (16, 32):
        coverage, targets = integrate(rate)
        independent_c, independent = integrate(rate + 1)
        successive = _compare(coverage, targets, coarse_c, coarse)
        odd = _compare(coverage, targets, independent_c, independent)
        report["spatial_checks"].append(
            {"rate": rate, "successive": successive, "independent": odd}
        )
        if successive["passed"] and odd["passed"]:
            report["accepted_spatial_rate"] = rate
            break
        coarse_c, coarse = coverage, targets
    else:
        reject(
            "spatial",
            "target quadrature did not converge within 32/33 samples per axis",
        )

    report["coverage"] = coverage.tolist()
    selected_parts = []
    stored_predictions = {
        name: np.zeros_like(target) for name, target in targets.items()
    }
    for material in range(3):
        candidates = arrays[f"candidates_{material}"]
        if coverage[material] > 0 and not len(candidates):
            reject(
                "candidates",
                f"material {material} has positive reference coverage but no 8x8 candidate",
            )
        if coverage[material] == 0:
            continue
        try:
            fit = fit_material_lobes(
                arrays[f"candidates_{material}_training"],
                targets["training"][material],
                arrays[f"candidates_{material}_held_out"],
                targets["held_out"][material],
                material_id=material,
                coverage=float(coverage[material]),
                tolerances=FIT_TOLERANCES,
            )
        except LobeFitError as error:
            report["material_fits"][str(material)] = error.result.to_mapping()
            reject("fitting", str(error))
            raise AssertionError("unreachable") from error
        report["material_fits"][str(material)] = fit.to_mapping()
        try:
            stored = _stored_records(
                candidates,
                fit.candidate_indices,
                fit.weights,
                float(coverage[material]),
            )
            MaterialMaps._validate_lobe_fields(
                stored,
                1,
                np.array([optical.material_id for optical in optics], dtype=np.uint8),
            )
        except ValueError as error:
            reject("storage", str(error))
            raise AssertionError("unreachable") from error
        arrays[f"retained_source_indices_{material}"] = candidates["source_index"][
            list(fit.candidate_indices)
        ].copy()
        selected_parts.append(stored)
        for name in stored_predictions:
            response = _responses(
                stored, arrays[f"{name}_directions"], optics[material]
            )
            stored_predictions[name][material] = (
                stored["weight"].astype(np.float64) @ response
            )
    report["storage_errors"] = {}
    for name, predictions in stored_predictions.items():
        arrays[f"stored_{name}"] = predictions
        for material in (*range(3), "total"):
            target = (
                targets[name].sum(axis=0)
                if material == "total"
                else targets[name][material]
            )
            prediction = (
                predictions.sum(axis=0)
                if material == "total"
                else predictions[material]
            )
            error = _error(
                prediction,
                target,
                np.full(len(target), 1.0 / len(target)),
                FIT_TOLERANCES,
            )
            report["storage_errors"][f"{name}_{material}"] = error.to_mapping()
            if not error.passed:
                reject(
                    "storage",
                    f"cast records failed {name} response check for {material}",
                )
    records = (
        np.concatenate(selected_parts)
        if selected_parts
        else np.empty(0, dtype=LOBE_DTYPE)
    )
    if abs(float(records["weight"].astype(np.float64).sum()) - 1.0) > 1e-6:
        reject("storage", "stored record weights do not close total coverage")
    report.update(accepted=True, stage="accepted", record_count=len(records))
    report["claim"] = (
        "spatially checked normal-view coupon only; no grazing-view, highlight-width or production-quality certification"
    )
    return finish(records)


def check_physical_footprints(
    state: GalvanisedState,
    footprints: Sequence[PhysicalFootprint],
    *,
    point_batch_size: int = MAX_POINT_BATCH,
) -> tuple[PhysicalFootprintFit, ...]:
    """Collect at most 16 explicit coupons, retaining rejected diagnostics."""
    if not 1 <= len(footprints) <= MAX_FOOTPRINTS:
        raise ValueError("provide 1..16 explicitly selected footprints")
    results = []
    for footprint in footprints:
        try:
            result = fit_physical_footprint(
                state, footprint, point_batch_size=point_batch_size
            )
        except PhysicalFootprintFitError as error:
            result = error.result
        results.append(result)
    return tuple(results)
