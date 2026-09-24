"""Measure galvanised height LOD and preview approximation error.

Example:
    PYTHONPATH=src python tools/galvanised/check_lod.py \
      --seed 42 --preset regular --size-mm 24 18 \
      --resolutions 32x24 64x48 128x96 --render-size 8x6 \
      --output /tmp/galvanised-lod.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from texture_generators.core.conductor import resource_hashes
from texture_generators.core.energy_compensation import energy_resource_hashes
from texture_generators.core.material import MaterialMaps
from texture_generators.core.material_render import (
    ReferenceDerivativeConvergenceError,
    _reference_derivative_step_mm,
    render_material_array,
    render_reference,
)
from texture_generators.core.physical import height_to_normal_physical
from texture_generators.materials.galvanised import (
    GENERATOR_VERSION,
    build_state,
    sample_points,
    sample_state,
    surface_resource_hashes,
)
from texture_generators.materials.galvanised_config import (
    GalvanisedConfig,
    PreviewConfig,
)

CHECK_VERSION = "3"
HEIGHT_TARGET_UM = 0.1
NORMAL_TARGET_DEG = 0.1
RADIANCE_FLOOR = 0.02
RIGS = ("studio", "oblique", "overcast", "grazing")
_HEIGHT_SAMPLE_LIMIT = 65_536


def _size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from error
    if min(width, height) < 3:
        raise argparse.ArgumentTypeError("both dimensions must be at least 3")
    return width, height


def _check_resolutions(sizes: list[tuple[int, int]]) -> None:
    if len(sizes) < 2:
        raise ValueError("at least two height resolutions are required")
    for low, high in pairwise(sizes):
        if high[0] <= low[0] or high[1] <= low[1]:
            raise ValueError("resolutions must increase on both axes")
        if high[0] % low[0] or high[1] % low[1]:
            raise ValueError(
                "adjacent resolutions must have integer ratios on both axes"
            )
        if high[0] // low[0] != high[1] // low[1]:
            raise ValueError("use the same integer ratio on both physical axes")


def _height_quadrature(state: Any) -> int:
    """Select the production midpoint rate from the immutable resolved config."""
    config = state.config
    if config.quality == "reference":
        raise ValueError(
            "the LOD checker does not support adaptive reference map quality"
        )
    return 2 if config.representation == "rich" or config.quality == "draft" else 4


def _height(
    state: Any, size: tuple[int, int], chunk_rows: int, quadrature: int
) -> np.ndarray:
    """Independently integrate physical point heights on a bounded midpoint grid."""
    width, height = size
    lx, ly = state.config.size_mm
    result = np.empty((height, width), dtype=np.float32)
    if quadrature not in (2, 4):
        raise ValueError("checker height quadrature must be 2 or 4")
    if quadrature**2 > _HEIGHT_SAMPLE_LIMIT:
        raise ValueError("height point limit must fit one complete pixel quadrature")
    max_tile_width = max(1, _HEIGHT_SAMPLE_LIMIT // quadrature**2)
    tile_width = min(width, max_tile_width)
    tile_rows = max(
        1, min(chunk_rows, _HEIGHT_SAMPLE_LIMIT // (tile_width * quadrature**2))
    )
    offsets = (np.arange(quadrature, dtype=np.float64) + 0.5) / quadrature
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        row = (np.arange(y0, y1, dtype=np.float64)[:, None] + offsets).reshape(-1)
        y = ly * row[:, None] / height
        for x0 in range(0, width, tile_width):
            x1 = min(width, x0 + tile_width)
            col = (np.arange(x0, x1, dtype=np.float64)[:, None] + offsets).reshape(-1)
            x = lx * col[None, :] / width
            sampled = np.asarray(
                sample_points(state, x, y)["height_um"], dtype=np.float64
            )
            result[y0:y1, x0:x1] = (
                sampled.reshape(y1 - y0, quadrature, x1 - x0, quadrature)
                .mean(axis=(1, 3))
                .astype(np.float32)
            )
    return result


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _height_pair(
    low: np.ndarray,
    high: np.ndarray,
    size_mm: tuple[float, float],
    relief_budget_um: float,
) -> dict[str, Any]:
    low_h, low_w = low.shape
    high_h, high_w = high.shape
    down = (
        high.astype(np.float64)
        .reshape(low_h, high_h // low_h, low_w, high_w // low_w)
        .mean(axis=(1, 3))
    )
    height_error = _rms(down - low)
    n_high = height_to_normal_physical(down, size_mm).astype(np.float64)
    n_low = height_to_normal_physical(low, size_mm).astype(np.float64)
    dot = np.clip(np.sum(n_high * n_low, axis=-1), -1.0, 1.0)
    angular_error = _rms(np.degrees(np.arccos(dot)))
    return {
        "low_size": [low_w, low_h],
        "high_size": [high_w, high_h],
        "block_factor": high_w // low_w,
        "height_rms_um": height_error,
        "height_rms_fraction_of_declared_relief": height_error / relief_budget_um,
        "normal_rms_deg": angular_error,
        "height_target_um": HEIGHT_TARGET_UM,
        "normal_target_deg": NORMAL_TARGET_DEG,
        "height_failed": not height_error < HEIGHT_TARGET_UM,
        "normal_failed": not angular_error < NORMAL_TARGET_DEG,
        "normal_method": "periodic central differences on high block-mean and direct low height, both at low physical spacing",
    }


def _radiance_error(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError("rendered arrays must share one shape")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    target = reference.astype(np.float64)
    lit = target >= RADIANCE_FLOOR
    dark = ~lit
    return {
        "relative_rms_nondark": _rms(difference[lit] / target[lit])
        if np.any(lit)
        else None,
        "absolute_rms_dark": _rms(difference[dark]) if np.any(dark) else None,
        "absolute_rms_all": _rms(difference),
        "nondark_channel_count": int(np.count_nonzero(lit)),
        "dark_channel_count": int(np.count_nonzero(dark)),
        "nondark_reference_floor": RADIANCE_FLOOR,
    }


def _resource_ids() -> dict[str, Any]:
    energy = dict(energy_resource_hashes())
    return {
        "surface": surface_resource_hashes(),
        "optics": dict(resource_hashes()),
        "renderer": energy,
        "energy_table_sha256": energy["ggx_directional_albedo.npz"],
    }


def measure(
    *,
    seed: int,
    preset: str,
    size_mm: tuple[float, float],
    resolutions: list[tuple[int, int]],
    render_size: tuple[int, int],
    relief_budget_um: float = 5.0,
    chunk_rows: int = 16,
    height_only: bool = False,
    max_height_pixels: int = 1_000_000,
    max_render_pixels: int = 108,
    derivative_step_mm: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe report from one immutable material state."""
    _check_resolutions(resolutions)
    if not np.isfinite(relief_budget_um) or relief_budget_um <= 0:
        raise ValueError("relief_budget_um must be finite and positive")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if max_height_pixels < 1 or max_render_pixels < 1:
        raise ValueError("pixel limits must be positive")
    if any(width * height > max_height_pixels for width, height in resolutions):
        raise ValueError(
            "height resolution exceeds max_height_pixels; raise the explicit limit for a larger run"
        )
    if not height_only and render_size[0] * render_size[1] > max_render_pixels:
        raise ValueError(
            "render size exceeds max_render_pixels; raise the explicit limit for a larger run"
        )
    costs: dict[str, float] = {}
    started = time.perf_counter()
    config = GalvanisedConfig.from_mapping(
        {
            "preset": preset,
            "size_mm": size_mm,
            "representation": "single_lobe" if height_only else "rich",
        }
    )
    state = build_state(config, seed=seed, size=resolutions[-1])
    assert state.config.size_mm is not None
    costs["build_state"] = time.perf_counter() - started
    height_quadrature = _height_quadrature(state)
    resolved_derivative_step = _reference_derivative_step_mm(state, derivative_step_mm)
    heights: list[np.ndarray] = []
    for width, height in resolutions:
        start = time.perf_counter()
        heights.append(_height(state, (width, height), chunk_rows, height_quadrature))
        costs[f"height_{width}x{height}"] = time.perf_counter() - start
    pairs = [
        _height_pair(low, high, size_mm, relief_budget_um)
        for low, high in pairwise(heights)
    ]
    height_failed = any(pair["height_failed"] for pair in pairs)
    normal_failed = any(pair["normal_failed"] for pair in pairs)
    renders: dict[str, Any] = {}
    derivative_reports: dict[str, Any] = {}
    derivative_failed = False
    crosscheck_size = (
        render_size
        if not height_only
        else (min(8, resolutions[0][0]), min(6, resolutions[0][1]))
    )
    if height_only:
        start = time.perf_counter()
        sampled_height = sample_state(state, size=crosscheck_size, maps=["height_um"])
        costs["sample_height_crosscheck_map"] = time.perf_counter() - start
    else:
        sampled_height = None
    if not height_only:
        start = time.perf_counter()
        rich = sample_state(state, size=render_size)
        costs["sample_render_maps"] = time.perf_counter() - start
        sampled_height = rich
        single_metadata = dict(rich.metadata)
        single_metadata["representation"] = "single_lobe"
        single_metadata["approximation"] = {
            "angular_lobes": "map-level single-lobe compatibility approximation",
            "fit_error": None,
        }
        single = MaterialMaps(rich.arrays, single_metadata)
        for rig in RIGS:
            preview = PreviewConfig.from_mapping({"rig": rig})
            start = time.perf_counter()
            rich_render = render_material_array(rich, preview=preview, output="linear")
            costs[f"render_rich_{rig}"] = time.perf_counter() - start
            start = time.perf_counter()
            single_render = render_material_array(
                single, preview=preview, output="linear"
            )
            costs[f"render_single_{rig}"] = time.perf_counter() - start
            references: dict[int, np.ndarray | None] = {}
            derivative_reports[rig] = {}
            for rate in (4, 8):
                diagnostic: dict[str, Any] = {}
                start = time.perf_counter()
                try:
                    references[rate] = render_reference(
                        state,
                        render_size,
                        preview=preview,
                        samples_per_axis=rate,
                        derivative_step_mm=resolved_derivative_step,
                        derivative_diagnostics=diagnostic,
                    )
                except ReferenceDerivativeConvergenceError as error:
                    references[rate] = None
                    derivative_failed = True
                    diagnostic.update(
                        {
                            "converged": False,
                            "unresolved_points_at_failure": error.unresolved_points,
                            "unresolved_pixels_at_failure": error.unresolved_points,
                            "failed_max_component_difference": error.max_component_difference,
                            "error": str(error),
                        }
                    )
                else:
                    diagnostic.update(
                        {
                            "converged": True,
                            "unresolved_points_at_failure": 0,
                            "unresolved_pixels_at_failure": 0,
                        }
                    )
                derivative_reports[rig][f"reference_{rate}"] = diagnostic
                costs[f"reference_{rate}_{rig}"] = time.perf_counter() - start
            reference4, reference8 = references[4], references[8]
            if reference4 is None or reference8 is None:
                renders[rig] = {"unavailable_due_to_derivative_convergence": True}
                continue
            renders[rig] = {
                "reference_4_vs_8": _radiance_error(reference4, reference8),
                "rich_vs_reference_4": _radiance_error(rich_render, reference4),
                "rich_vs_reference_8": _radiance_error(rich_render, reference8),
                "single_lobe_vs_reference_4": _radiance_error(
                    single_render, reference4
                ),
                "single_lobe_vs_reference_8": _radiance_error(
                    single_render, reference8
                ),
            }
    assert sampled_height is not None
    start = time.perf_counter()
    independent_height = _height(state, crosscheck_size, chunk_rows, height_quadrature)
    sampler_rms = _rms(sampled_height["height_um"] - independent_height)
    costs["height_sampler_crosscheck"] = time.perf_counter() - start
    sampler_failed = sampler_rms > 1e-6
    return {
        "check_version": CHECK_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "material_key": state.material_key,
        "config": state.config.to_mapping(),
        "resolutions": [list(value) for value in resolutions],
        "render_size": None if height_only else list(render_size),
        "pixel_limits": {"height": max_height_pixels, "render": max_render_pixels},
        "height_sampling": {
            "quadrature_per_axis": height_quadrature,
            "rule": "resolved immutable config quality and representation",
            "method": "independent phase-aligned physical midpoint grid in bounded 2D tiles",
            "point_limit_per_tile": _HEIGHT_SAMPLE_LIMIT,
            "physical_extent_mm": list(state.config.size_mm),
        },
        "height_pairs": pairs,
        "render_error": renders,
        "reference_derivative": {
            "requested_step_mm": None
            if derivative_step_mm is None
            else list(derivative_step_mm),
            "initial_step_mm": list(resolved_derivative_step),
            "step_source": "immutable physical feature scales"
            if derivative_step_mm is None
            else "caller supplied",
            "spatial_rates_per_axis": [4, 8] if not height_only else [],
            "convergence_by_rig_and_rate": derivative_reports,
            "component_tolerance": 1e-5,
            "maximum_halvings": 4,
        },
        "render_height_sampler_crosscheck_rms_um": sampler_rms,
        "height_sampler_crosscheck_rms_um": sampler_rms,
        "height_sampler_crosscheck_size": list(crosscheck_size),
        "height_sampler_crosscheck_tolerance_um": 1e-6,
        "resource_hashes": _resource_ids(),
        "stage_cost_seconds": costs,
        "total_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "fail_flags": {
            "height_target_failed": height_failed,
            "normal_target_failed": normal_failed,
            "height_sampler_crosscheck_failed": sampler_failed,
            "derivative_convergence_failed": derivative_failed,
            "reference_convergence_target_undefined": True,
            "render_accuracy_target_undefined": True,
            "render_not_run": height_only,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=("regular", "minimised", "weathered", "wet_storage"),
        default="regular",
    )
    parser.add_argument(
        "--size-mm", nargs=2, type=float, metavar=("LX", "LY"), default=[24.0, 18.0]
    )
    parser.add_argument(
        "--resolutions", nargs="+", type=_size, default=[(32, 24), (64, 48), (128, 96)]
    )
    parser.add_argument("--render-size", type=_size, default=(8, 6))
    parser.add_argument("--relief-budget-um", type=float, default=5.0)
    parser.add_argument("--chunk-rows", type=int, default=16)
    parser.add_argument("--max-height-pixels", type=int, default=1_000_000)
    parser.add_argument("--max-render-pixels", type=int, default=108)
    parser.add_argument("--height-only", action="store_true")
    parser.add_argument(
        "--derivative-step-mm", nargs=2, type=float, metavar=("EX", "EY")
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if a measured height or normal target fails",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = measure(
        seed=args.seed,
        preset=args.preset,
        size_mm=tuple(args.size_mm),
        resolutions=args.resolutions,
        render_size=args.render_size,
        relief_budget_um=args.relief_budget_um,
        chunk_rows=args.chunk_rows,
        height_only=args.height_only,
        max_height_pixels=args.max_height_pixels,
        max_render_pixels=args.max_render_pixels,
        derivative_step_mm=None
        if args.derivative_step_mm is None
        else tuple(args.derivative_step_mm),
    )
    path = (
        args.output
        if args.output.suffix.lower() == ".json"
        else args.output / "report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(path)
    if args.strict and (
        report["fail_flags"]["height_target_failed"]
        or report["fail_flags"]["normal_target_failed"]
        or report["fail_flags"]["height_sampler_crosscheck_failed"]
        or report["fail_flags"]["derivative_convergence_failed"]
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
