"""Physically scaled, replayable galvanised surface state and map sampling."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from texture_generators.core.conductor import resource_hashes, zinc_f0
from texture_generators.core.dendrites import (
    DendriteRecords,
    build_dendrites,
    evaluate_dendrite_fields,
)
from texture_generators.core.energy_compensation import energy_resource_hashes
from texture_generators.core.footprint import box_average, midpoint_axis
from texture_generators.core.galvanised_weather import (
    WeatherRecords,
    build_weather,
    weather_at,
)
from texture_generators.core.grains import (
    AreaStatistics,
    GrainPartition,
    growth_weights,
    initial_nucleus_count,
    place_grains,
)
from texture_generators.core.material import (
    CANONICAL_CHANNELS,
    LOBE_DTYPE,
    MaterialMaps,
    validate_sampling_request,
)
from texture_generators.core.microfacet import (
    GGX_WIDTH_FLOOR,
    openpbr_widths,
    tangent_frame,
)
from texture_generators.core.physical import (
    validate_image_size,
    validate_map_size,
)
from texture_generators.core.random_fields import (
    make_rng_with_seed,
    material_key_from_rng,
    topology_rng,
)
from texture_generators.materials.galvanised_config import (
    GalvanisedConfig,
    PreviewConfig,
)

GENERATOR_VERSION = "galvanised-2"
GALVANISED_PRESETS = ("regular", "minimised", "weathered", "wet_storage")

# Exact-area 12-seed fit at 100x100 mm, 8 mm diameter; extrapolation is marked.
_POPULATION_MAPPING_VERSION = "2026-09-23.area12-v1"
_POPULATION_SIGMA = ((0.0, 0.0), (0.20, 0.30), (0.28, 1.285), (0.50, 2.0), (1.0, 2.8))

_LOBE_CACHE_FIELDS = (
    "height_um",
    "metallic",
    "patina_coverage",
    "white_stain_coverage",
    "anisotropy_axis",
    "intrinsic_roughness",
    "intrinsic_anisotropy",
)

_DEFECT_DTYPE = np.dtype(
    [
        ("kind", np.uint8),
        ("position_mm", np.float64, (2,)),
        ("radius_mm", np.float64),
        ("height_um", np.float64),
        ("length_mm", np.float64),
        ("angle_rad", np.float64),
    ]
)


def _readonly(array: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(array)
    return np.frombuffer(packed.tobytes(), dtype=packed.dtype).reshape(packed.shape)


@dataclass(frozen=True)
class GalvanisedState:
    """Immutable physical state; no raster resolution or preview lighting."""

    config: GalvanisedConfig
    material_key: int
    seed: int | None
    partition: GrainPartition
    dendrites: DendriteRecords
    weather: WeatherRecords
    micro_spectrum: np.ndarray
    defects: np.ndarray
    population_stats: AreaStatistics
    population_mapping_version: str


def _config(
    config: GalvanisedConfig | Mapping[str, Any] | None, size: tuple[int, int]
) -> GalvanisedConfig:
    if config is None:
        result = GalvanisedConfig()
    elif isinstance(config, GalvanisedConfig):
        result = config
    elif isinstance(config, Mapping):
        result = GalvanisedConfig.from_mapping(config)
    else:
        raise TypeError("galvanised config must be GalvanisedConfig, mapping, or None")
    return result.resolve(size=size)


def _build_defects(config: GalvanisedConfig, key: int) -> np.ndarray:
    area_cm2 = config.size_mm[0] * config.size_mm[1] / 100.0  # type: ignore[index]
    rng = topology_rng(key, "fabrication_defects")
    count_dross = int(rng.poisson(config.dross_per_cm2 * area_cm2))
    count_runs = int(rng.poisson(config.runs_per_cm2 * area_cm2))
    if count_dross + count_runs > 100_000:
        raise ValueError("defect count exceeds the bounded object budget")
    records = np.empty(count_dross + count_runs, dtype=_DEFECT_DTYPE)
    for index in range(len(records)):
        kind = 0 if index < count_dross else 1
        records[index] = (
            kind,
            (rng.uniform(0.0, config.size_mm[0]), rng.uniform(0.0, config.size_mm[1])),  # type: ignore[index]
            rng.uniform(0.12, 0.5) if kind == 0 else rng.uniform(0.08, 0.22),
            rng.uniform(2.0, 8.0) if kind == 0 else rng.uniform(1.0, 5.0),
            0.0 if kind == 0 else rng.uniform(1.0, 7.0),
            math.radians(config.gravity_angle_deg),
        )
    return _readonly(records)


def _build_micro_spectrum(size_mm: tuple[float, float], key: int) -> np.ndarray:
    """Fixed periodic Fourier packets with a physical 0.25-0.9 mm band.

    Independent phases and directions avoid a sheet-wide checkerboard. The
    integer wave vectors make periodisation exact; rounding is a documented
    finite-tile approximation to the requested physical wavelength band.
    """
    rng = topology_rng(key, "micro_spectrum")
    angle = rng.uniform(0.0, 2.0 * np.pi, 12)
    wavelength = np.exp(rng.uniform(np.log(0.25), np.log(0.9), 12))
    kx = np.rint(size_mm[0] * np.cos(angle) / wavelength)
    ky = np.rint(size_mm[1] * np.sin(angle) / wavelength)
    kx[(kx == 0) & (ky == 0)] = 1.0
    phase = rng.uniform(-np.pi, np.pi, 12)
    return _readonly(np.stack((kx, ky, phase), axis=-1))


def build_state(
    config: GalvanisedConfig | Mapping[str, Any] | None = None,
    *,
    seed: int | None = None,
    material_key: int | None = None,
    size: tuple[int, int] = (512, 512),
) -> GalvanisedState:
    """Construct one replayable state from exactly one key draw or an explicit key."""
    image_size = validate_image_size(size)
    resolved = _config(config, image_size)
    concrete_seed: int | None
    if material_key is None:
        concrete_seed, rng = make_rng_with_seed(seed)
        key = material_key_from_rng(rng)
    else:
        if isinstance(material_key, bool) or not isinstance(
            material_key, (int, np.integer)
        ):
            raise TypeError("material_key must be an unsigned 64-bit integer")
        key = int(material_key)
        if not 0 <= key < 1 << 64:
            raise ValueError("material_key must be an unsigned 64-bit integer")
        concrete_seed = None if seed is None else make_rng_with_seed(seed)[0]
    assert resolved.size_mm is not None
    diameter = resolved.spangle_diameter_mm
    count = initial_nucleus_count(resolved.size_mm, diameter)
    minimum_distance = 0.35 * diameter
    points = place_grains(
        resolved.size_mm,
        diameter,
        topology_rng(key, "nucleus_placement"),
        count=count,
        mode="uniform" if resolved.placement == "uniform" else "poisson_disc",
        minimum_distance_mm=minimum_distance,
    )
    sigma = float(
        np.interp(
            resolved.spangle_cv,
            [pair[0] for pair in _POPULATION_SIGMA],
            [pair[1] for pair in _POPULATION_SIGMA],
        )
    )
    weights = growth_weights(
        count,
        diameter,
        topology_rng(key, "nucleus_growth"),
        growth_sigma=sigma,
    )
    partition = GrainPartition(points, weights, resolved.size_mm)
    stats = partition.area_statistics()
    dendrites = build_dendrites(
        points, diameter, resolved.crystal_tilt_concentration, key
    )
    weather = build_weather(key, resolved.size_mm, weather_seed=resolved.weather_seed)
    return GalvanisedState(
        resolved,
        key,
        concrete_seed,
        partition,
        dendrites,
        weather,
        _build_micro_spectrum(resolved.size_mm, key),
        _build_defects(resolved, key),
        stats,
        _POPULATION_MAPPING_VERSION,
    )


def _defect_height(state: GalvanisedState, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.zeros(x.shape, dtype=np.float64)
    if len(state.defects) == 0:
        return result
    assert state.config.size_mm is not None
    lx, ly = state.config.size_mm
    for record in state.defects:
        px, py = record["position_mm"]
        dx = np.remainder(x - px + lx / 2.0, lx) - lx / 2.0
        dy = np.remainder(y - py + ly / 2.0, ly) - ly / 2.0
        radius = float(record["radius_mm"])
        if record["kind"] == 0:
            distance = np.hypot(dx, dy) / radius
        else:
            angle = float(record["angle_rad"])
            along = dx * np.cos(angle) + dy * np.sin(angle)
            cross = -dx * np.sin(angle) + dy * np.cos(angle)
            length = float(record["length_mm"])
            distance = np.hypot(
                cross / radius,
                np.maximum(np.maximum(-along, along - length), 0.0) / radius,
            )
        compact = np.maximum(1.0 - distance * distance, 0.0) ** 2
        result += float(record["height_um"]) * compact
    return result


def _sample_continuous(
    state: GalvanisedState, x_mm: np.ndarray, y_mm: np.ndarray
) -> dict[str, np.ndarray]:
    x, y = np.broadcast_arrays(
        np.asarray(x_mm, dtype=np.float64), np.asarray(y_mm, dtype=np.float64)
    )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("sample coordinates must be finite")
    config = state.config
    assert config.size_mm is not None
    ownership, distance = state.partition.query_with_boundary(x, y)
    ids = ownership.ids
    branch, branch_direction = evaluate_dendrite_fields(
        state.dendrites, ids, x, y, config.size_mm, config.spangle_diameter_mm
    )
    transition_width = max(0.045 * config.spangle_diameter_mm, 0.02)
    t = np.clip(distance / transition_width, 0.0, 1.0)
    taper = t * t * (3.0 - 2.0 * t)
    morphology_gain = config.texture_strength / 0.45
    trunk = morphology_gain * config.trunk_relief_um * np.tanh(branch)
    dendrite = (
        morphology_gain * config.dendrite_relief_um * (branch / (1.0 + branch) - 0.28)
    )
    lx, ly = config.size_mm
    micro = np.zeros(x.shape, dtype=np.float64)
    u, v = np.remainder(x, lx) / lx, np.remainder(y, ly) / ly
    for kx, ky, phase in state.micro_spectrum:
        micro += np.sin(2.0 * np.pi * (kx * u + ky * v) + phase)
    # Expected RMS is half the control amplitude, matching the former micro
    # component's budget without its two globally coherent crossed waves.
    micro *= config.micro_relief_um / np.sqrt(2.0 * len(state.micro_spectrum))
    groove = -config.boundary_depth_um * np.exp(
        -0.5 * (distance / (transition_width * 0.45)) ** 2
    )
    substrate = (
        taper * (trunk + dendrite + micro) + groove + _defect_height(state, x, y)
    )
    zinc, patina, white, deposit = weather_at(
        state.weather,
        x,
        y,
        exposure=config.exposure,
        wetness=config.wetness,
        confinement=config.confinement,
        salt_exposure=config.salt_exposure,
        white_stain=config.white_stain,
        substrate_height_um=substrate,
    )
    azimuth = state.dendrites.azimuth_rad[ids]
    base_axis = np.stack((np.cos(2.0 * azimuth), np.sin(2.0 * azimuth)), axis=-1)
    # The branch descriptor already carries its support and axis coherence.
    # A weak intrinsic crystal axis remains in empty regions; competing arms
    # may cancel without being normalised back into an arbitrary strong axis.
    directional = 0.15 * base_axis + 0.85 * branch_direction
    coherence = np.linalg.norm(directional, axis=-1)
    axis = np.divide(
        directional,
        coherence[..., None],
        out=np.zeros_like(directional),
        where=coherence[..., None] > 1e-8,
    )
    # Family-locked intrinsic zinc widths: weather only mixes separate deposits.
    intrinsic_roughness = np.clip(
        config.roughness - 0.10 * config.texture_strength * np.tanh(branch), 0.0, 1.0
    )
    intrinsic_anisotropy = np.clip(
        config.anisotropy * np.minimum(coherence, 1.0), 0.0, 1.0
    )
    roughness = zinc * intrinsic_roughness + patina * 0.64 + white * 0.86
    anisotropy = zinc * intrinsic_anisotropy
    return {
        "height_um": substrate + deposit,
        "substrate_height_um": substrate,
        "deposit_height_um": deposit,
        "grain_id": ids,
        "boundary_distance_mm": distance,
        "orientation_id": ids,
        "metallic": zinc,
        "patina_coverage": patina,
        "white_stain_coverage": white,
        "roughness": roughness,
        "anisotropy": anisotropy,
        "anisotropy_axis": axis,
        "intrinsic_roughness": intrinsic_roughness,
        "intrinsic_anisotropy": intrinsic_anisotropy,
    }


def sample_points(
    state: GalvanisedState,
    x_mm: np.ndarray | float,
    y_mm: np.ndarray | float,
    *,
    footprint_mm: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Evaluate shared continuous fields at arbitrary periodic physical points."""
    if len(footprint_mm) != 2 or any(not np.isfinite(v) or v < 0 for v in footprint_mm):
        raise ValueError("footprint_mm must contain two finite non-negative widths")
    if footprint_mm == (0.0, 0.0):
        return _sample_continuous(state, np.asarray(x_mm), np.asarray(y_mm))
    fx, fy = footprint_mm
    samples = [
        _sample_continuous(
            state, np.asarray(x_mm) + sx * fx, np.asarray(y_mm) + sy * fy
        )
        for sy in (-0.25, 0.25)
        for sx in (-0.25, 0.25)
    ]
    result: dict[str, np.ndarray] = {}
    for name in samples[0]:
        if name in ("grain_id", "orientation_id"):
            result[name] = _sample_continuous(
                state, np.asarray(x_mm), np.asarray(y_mm)
            )[name]
        elif name in ("anisotropy_axis", "anisotropy"):
            continue
        else:
            result[name] = np.mean(np.stack([item[name] for item in samples]), axis=0)
    vector = np.mean(
        np.stack(
            [
                item["anisotropy_axis"] * item["anisotropy"][..., None]
                for item in samples
            ]
        ),
        axis=0,
    )
    length = np.linalg.norm(vector, axis=-1)
    result["anisotropy"] = length
    result["anisotropy_axis"] = np.divide(
        vector,
        length[..., None],
        out=np.zeros_like(vector),
        where=length[..., None] > 1e-8,
    )
    return result


_DEPOSIT_OPTICS = (
    {
        "material_id": 0,
        "kind": "conductor",
        "diffuse_color_linear": [0.0, 0.0, 0.0],
        "roughness": None,
        "anisotropy": None,
        "ior": None,
    },
    {
        "material_id": 1,
        "kind": "dielectric",
        "diffuse_color_linear": [0.43, 0.45, 0.37],
        "roughness": 0.64,
        "anisotropy": 0.0,
        "ior": 1.5,
    },
    {
        "material_id": 2,
        "kind": "dielectric",
        "diffuse_color_linear": [0.78, 0.80, 0.74],
        "roughness": 0.86,
        "anisotropy": 0.0,
        "ior": 1.5,
    },
)


def optical_parameters(state: GalvanisedState) -> list[dict[str, Any]]:
    """Return versioned zinc and dielectric authoring parameters for this state."""
    optics = [dict(item) for item in _DEPOSIT_OPTICS]
    optics[0]["f0_linear_srgb"] = zinc_f0().tolist()
    optics[0]["roughness"] = state.config.roughness
    optics[0]["anisotropy"] = state.config.anisotropy
    return optics


def surface_resource_hashes() -> dict[str, str]:
    """Return identities of morphology and exact-area population fit inputs."""
    hashes: dict[str, str] = {}
    for filename in ("morphology.json", "population_calibration.json"):
        resource = resources.files("texture_generators").joinpath(
            "data/galvanised", filename
        )
        hashes[filename] = hashlib.sha256(resource.read_bytes()).hexdigest()
    return hashes


def _metadata(
    state: GalvanisedState, size: tuple[int, int], quadrature: int
) -> dict[str, Any]:
    config = state.config
    stats = state.population_stats
    optics = optical_parameters(state)
    warnings = list(config.warnings)
    calibrated = (
        config.spangle_cv in (0.20, 0.28)
        and config.placement == "poisson"
        and config.spangle_diameter_mm == 8.0
        and config.size_mm == (100.0, 100.0)
    )
    warnings.append(
        "Morphology family directions and populations are authoring templates pending specimen calibration."
    )
    if config.representation == "rich":
        warnings.append(
            "Rich lobe records preserve the provisional 2x2 shared quadrature; error against denser angular reference is unmeasured."
        )
    if not calibrated:
        warnings.append(
            "Population CV mapping is extrapolated beyond the 12-seed, 100x100 mm, 8 mm exact-area fit."
        )
    return {
        "generator_version": GENERATOR_VERSION,
        "schema_version": 1,
        "config": config.to_mapping(),
        "seed": state.seed,
        "material_key": state.material_key,
        "size_mm": list(config.size_mm or ()),
        "size": list(size),
        "representation": config.representation,
        "optical_parameters": optics,
        "brdf_convention": {
            "distribution": "anisotropic GGX",
            "masking": "height-correlated Smith",
            "roughness_mapping": "OpenPBR 1.1: alpha_t=r^2*sqrt(2/(1+(1-a)^2)), alpha_b=(1-a)*alpha_t",
            "directional_width_floor": GGX_WIDTH_FLOOR,
            "anisotropy_axis": "(cos(2*theta),sin(2*theta)); theta clockwise in raster coordinates",
            "optical_parameter_table_version": 1,
        },
        "filtering": {
            "quadrature": f"{quadrature}x{quadrature} shared samples",
            "height_normal": "periodic central difference from exported float32 height",
        },
        "approximation": {
            "base_color": "coverage-weighted compatibility mixture",
            "angular_lobes": (
                "up to four shared-subpixel components per visible material; local slopes from 2x2 finite differences"
                if config.representation == "rich"
                else "single-lobe compatibility mixture; spatial integration does not validate angular fit"
            ),
            "fit_error": None,
        },
        "stats": {
            "nuclei": len(state.partition.points_mm),
            "active_grains": len(stats.active_ids),
            "hidden_grains": stats.hidden_count,
            "median_diameter_mm": stats.median_diameter_mm,
            "diameter_cv": stats.diameter_cv,
            "diameter_cv_numerical_error": stats.diameter_cv_error,
            "median_numerical_error_mm": stats.median_error_mm,
            "statistically_sufficient": stats.statistically_sufficient,
            "small_cell_count": stats.small_cell_count,
            "smallest_area_mm2": stats.smallest_area_mm2,
            "population_mapping_version": state.population_mapping_version,
            "target_diameter_cv": config.spangle_cv,
            "realised_cv_minus_target": stats.diameter_cv - config.spangle_cv,
            "population_fit_error": (
                0.0020374346508843233
                if config.spangle_cv == 0.20
                else 0.002367940167173954
                if config.spangle_cv == 0.28
                else None
            )
            if calibrated
            else None,
            "cross_seed_cv_sd": (
                0.011024148935217476
                if config.spangle_cv == 0.20
                else 0.01621565456902926
                if config.spangle_cv == 0.28
                else None
            )
            if calibrated
            else None,
        },
        "resource_hashes": dict(resource_hashes()),
        "renderer_resource_hashes": dict(energy_resource_hashes()),
        "surface_resource_hashes": surface_resource_hashes(),
        "morphology_table_version": state.dendrites.table_version,
        "no_boundary_semantics": "positive infinity in boundary_distance_mm",
        "warnings": warnings,
    }


def _average(sub: np.ndarray, quadrature: int = 2) -> np.ndarray:
    return box_average(sub, quadrature)


def _subsamples(
    state: GalvanisedState,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    size: tuple[int, int],
    quadrature: int = 2,
) -> dict[str, np.ndarray]:
    assert state.config.size_mm is not None
    width, height = size
    lx, ly = state.config.size_mm
    cols = midpoint_axis(x0, x1, quadrature)
    rows = midpoint_axis(y0, y1, quadrature)
    return _sample_continuous(
        state, lx * cols[None, :] / width, ly * rows[:, None] / height
    )


def _integrate_fields(
    sub: Mapping[str, np.ndarray], quadrature: int
) -> dict[str, np.ndarray]:
    result = {
        name: _average(value, quadrature)
        for name, value in sub.items()
        if name not in ("grain_id", "orientation_id", "anisotropy_axis")
    }
    # Point anisotropy already contains the zinc coverage. Weighting by zinc
    # again would suppress half-covered anisotropy by a spurious factor of two.
    result["anisotropy_vector"] = _average(
        sub["anisotropy_axis"] * sub["anisotropy"][..., None], quadrature
    )
    return result


_SPATIAL_TOLERANCES = {
    "height_um": 0.005,
    "substrate_height_um": 0.005,
    "deposit_height_um": 0.005,
    "metallic": 0.001,
    "patina_coverage": 0.001,
    "white_stain_coverage": 0.001,
    "roughness": 0.002,
    "anisotropy_vector": 0.002,
}
_REFERENCE_MAP_PIXEL_LIMIT = 4096
_SAMPLING_POINT_LIMIT = 65_536


def _reference_fields(
    state: GalvanisedState,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Check each footprint independently with dyadic and non-dyadic grids."""

    def integrate(rate: int) -> dict[str, np.ndarray]:
        return _integrate_fields(_subsamples(state, x0, x1, y0, y1, size, rate), rate)

    coarse = integrate(8)
    result = {name: np.empty_like(value) for name, value in coarse.items()}
    accepted = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    accepted_rate = np.zeros(accepted.shape, dtype=np.uint8)
    for rate in (16, 32):
        fine = integrate(rate)
        independent = integrate(rate + 1)
        passing = np.ones(accepted.shape, dtype=bool)
        for name, tolerance in _SPATIAL_TOLERANCES.items():
            change = np.maximum(
                np.abs(fine[name] - coarse[name]),
                np.abs(fine[name] - independent[name]),
            )
            if change.ndim == 3:
                change = np.max(change, axis=-1)
            passing &= change <= tolerance
        newly_accepted = passing & ~accepted
        for name in result:
            result[name][newly_accepted] = fine[name][newly_accepted]
        accepted_rate[newly_accepted] = rate
        accepted |= passing
        if np.all(accepted):
            return result, accepted_rate
        coarse = fine
    failed_y, failed_x = np.argwhere(~accepted)[0]
    raise ValueError(
        "reference footprint quadrature did not converge at pixel "
        f"({x0 + failed_x}, {y0 + failed_y}) within 32x32 samples; "
        "increase pixel resolution, reduce physical tile extent, or select an "
        "explicitly approximate quality"
    )


def _allocate(
    name: str,
    size: tuple[int, int],
    output_dir: Path | None,
    *,
    temporary: bool,
) -> np.ndarray:
    width, height = size
    descriptor = CANONICAL_CHANNELS[name]
    shape = (
        (height, width)
        if descriptor.components == 1
        else (height, width, descriptor.components)
    )
    if output_dir is None:
        return np.empty(shape, dtype=descriptor.dtype)
    path = output_dir / (f".work_{name}.npy" if temporary else f"{name}.npy")
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=descriptor.dtype, shape=shape
    )


def _close_working_map(array: np.ndarray) -> None:
    if isinstance(array, np.memmap):
        array.flush()
        # NumPy exposes no public close API. Only sampler-owned temporary
        # mappings with no returned views reach this helper.
        array._mmap.close()  # type: ignore[attr-defined]


def _normal_tiles(
    height: np.ndarray,
    normal: np.ndarray,
    size_mm: tuple[float, float],
    chunk_size: int,
) -> None:
    rows, cols = height.shape
    dx, dy = size_mm[0] / cols, size_mm[1] / rows
    for y0 in range(0, rows, chunk_size):
        y1 = min(rows, y0 + chunk_size)
        yi = np.arange(y0 - 1, y1 + 1) % rows
        for x0 in range(0, cols, chunk_size):
            x1 = min(cols, x0 + chunk_size)
            xi = np.arange(x0 - 1, x1 + 1) % cols
            halo = np.asarray(height[np.ix_(yi, xi)], dtype=np.float64)
            p = (halo[1:-1, 2:] - halo[1:-1, :-2]) / (2.0 * dx) * 1e-3
            q = (halo[2:, 1:-1] - halo[:-2, 1:-1]) / (2.0 * dy) * 1e-3
            local = np.stack((-p, q, np.ones_like(p)), axis=-1)
            local /= np.linalg.norm(local, axis=-1, keepdims=True)
            normal[y0:y1, x0:x1] = local


def _append_lobes(
    state: GalvanisedState,
    size: tuple[int, int],
    chunk_size: int,
    working: Mapping[str, np.ndarray],
    output_dir: Path | None,
    sample_cache: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Keep four shared optical samples per visible material and pixel.

    Only actually visible materials occupy records. The maximum buffer is
    mapped to disk when output_dir is supplied, then compacted to the exact
    record count; a dense 12-lobe resident image is never constructed.
    """
    width, height = size
    coverage_names = ("metallic", "patina_coverage", "white_stain_coverage")
    upper = 4 * sum(int(np.count_nonzero(working[name])) for name in coverage_names)
    if width * height >= 1 << 32:
        raise ValueError("rich lobe pixel_index exceeds uint32 range")
    if output_dir is None:
        buffer = np.empty(upper, dtype=LOBE_DTYPE)
        work_path = None
    else:
        lobe_dir = output_dir / "layers" / "lobes"
        lobe_dir.mkdir(parents=True, exist_ok=True)
        work_path = lobe_dir / ".work_records.npy"
        buffer = np.lib.format.open_memmap(
            work_path, mode="w+", dtype=LOBE_DTYPE, shape=(upper,)
        )
    cursor = 0
    assert state.config.size_mm is not None
    dx = state.config.size_mm[0] / width
    dy = state.config.size_mm[1] / height
    for y0 in range(0, height, chunk_size):
        y1 = min(height, y0 + chunk_size)
        for x0 in range(0, width, chunk_size):
            x1 = min(width, x0 + chunk_size)
            if sample_cache is None:
                sub = _subsamples(state, x0, x1, y0, y1, size)
            else:
                sub = {
                    name: values[2 * y0 : 2 * y1, 2 * x0 : 2 * x1]
                    for name, values in sample_cache.items()
                }
            h = sub["height_um"].reshape(y1 - y0, 2, x1 - x0, 2)
            sx = np.empty_like(h)
            sy = np.empty_like(h)
            sx[:, :, :, :] = ((h[:, :, :, 1] - h[:, :, :, 0]) / (0.5 * dx))[
                :, :, :, None
            ]
            sy[:, :, :, :] = ((h[:, 1, :, :] - h[:, 0, :, :]) / (0.5 * dy))[
                :, None, :, :
            ]
            geometric = working["normal_ts"][y0:y1, x0:x1]
            resolved_x = -geometric[..., 0] / geometric[..., 2] / 1e-3
            resolved_y = geometric[..., 1] / geometric[..., 2] / 1e-3
            for material_id, name in enumerate(coverage_names):
                coverage_sub = sub[name].reshape(y1 - y0, 2, x1 - x0, 2)
                for sub_y in range(2):
                    for sub_x in range(2):
                        weight = coverage_sub[:, sub_y, :, sub_x] / 4.0
                        active = weight > 0.0
                        if not np.any(active):
                            continue
                        # The effective lobe carries the subpixel residual to
                        # the exported resolved-height normal, once only.
                        residual_x = sx[:, sub_y, :, sub_x] - resolved_x
                        residual_y = sy[:, sub_y, :, sub_x] - resolved_y
                        local_x = resolved_x + residual_x
                        local_y = resolved_y + residual_y
                        n = np.stack(
                            (-1e-3 * local_x, 1e-3 * local_y, np.ones_like(local_x)),
                            axis=-1,
                        )
                        n /= np.linalg.norm(n, axis=-1, keepdims=True)
                        if material_id == 0:
                            axis = sub["anisotropy_axis"].reshape(
                                y1 - y0, 2, x1 - x0, 2, 2
                            )[:, sub_y, :, sub_x]
                            rough = sub["intrinsic_roughness"].reshape(
                                y1 - y0, 2, x1 - x0, 2
                            )[:, sub_y, :, sub_x]
                            aniso = sub["intrinsic_anisotropy"].reshape(
                                y1 - y0, 2, x1 - x0, 2
                            )[:, sub_y, :, sub_x]
                        else:
                            axis = np.broadcast_to(
                                np.array([1.0, 0.0]), (*n.shape[:-1], 2)
                            )
                            rough = np.full(
                                n.shape[:-1], 0.64 if material_id == 1 else 0.86
                            )
                            aniso = np.zeros(n.shape[:-1])
                        tangent, _ = tangent_frame(n, axis)
                        alpha_t, alpha_b = openpbr_widths(rough, aniso)
                        loc_y, loc_x = np.nonzero(active)
                        amount = len(loc_y)
                        target = buffer[cursor : cursor + amount]
                        target["pixel_index"] = (y0 + loc_y) * width + x0 + loc_x
                        target["material_id"] = material_id
                        target["weight"] = weight[active]
                        target["normal_ts"] = n[active]
                        target["tangent_ts"] = tangent[active]
                        target["alpha_t"] = alpha_t[active]
                        target["alpha_b"] = alpha_b[active]
                        target["optical_parameter_index"] = material_id
                        cursor += amount
    if output_dir is None:
        lobes = buffer[:cursor]
    else:
        assert work_path is not None
        final_path = output_dir / "layers" / "lobes" / "records.npy"
        if cursor == upper:
            assert isinstance(buffer, np.memmap)
            # Windows cannot rename an open mapped file. Reopen the completed
            # file read-only rather than returning a view of working storage.
            _close_working_map(buffer)
            work_path.rename(final_path)
            lobes = np.load(final_path, mmap_mode="r", allow_pickle=False)
        else:
            lobes = np.lib.format.open_memmap(
                final_path, mode="w+", dtype=LOBE_DTYPE, shape=(cursor,)
            )
            for start in range(0, cursor, 262_144):
                stop = min(cursor, start + 262_144)
                lobes[start:stop] = buffer[start:stop]
            lobes.flush()
            _close_working_map(buffer)
            del buffer
            work_path.unlink()
    lobes.setflags(write=False)
    return lobes


def sample_state(
    state: GalvanisedState,
    size: tuple[int, int] = (512, 512),
    *,
    maps: Sequence[str] | None = None,
    chunk_size: int = 128,
    output_dir: str | Path | None = None,
) -> MaterialMaps:
    """Sample selected map channels with bounded tile work and shared quadrature."""
    image_size = validate_map_size(size)
    selected = validate_sampling_request(maps, chunk_size)
    rich = state.config.representation == "rich"
    reference = state.config.quality == "reference"
    if rich and reference:
        raise ValueError(
            "reference rich maps are not yet supported by the angular fitter; "
            "use representation='single_lobe' for checked spatial integration "
            "or render_reference for the joint angular response"
        )
    width, height = image_size
    count = width * height
    if reference and count > _REFERENCE_MAP_PIXEL_LIMIT:
        raise ValueError(
            f"reference map integration is limited to {_REFERENCE_MAP_PIXEL_LIMIT} "
            "pixels per call; use a smaller diagnostic raster"
        )
    quadrature = 2 if rich or state.config.quality == "draft" else 8 if reference else 4
    largest_rate = 33 if reference else quadrature
    chunk_size = min(
        chunk_size, max(1, math.isqrt(_SAMPLING_POINT_LIMIT) // largest_rate)
    )
    needed = set(selected)
    if rich:
        needed.update(
            (
                "height_um",
                "normal_ts",
                "metallic",
                "patina_coverage",
                "white_stain_coverage",
                "roughness",
                "anisotropy",
                "anisotropy_axis",
            )
        )
    if "normal_ts" in selected or "finite_band_slope" in selected:
        needed.update(("height_um", "normal_ts"))
    if "base_color_linear" in selected:
        needed.update(("metallic", "patina_coverage", "white_stain_coverage"))
    selected_bytes = sum(
        count
        * CANONICAL_CHANNELS[name].components
        * CANONICAL_CHANNELS[name].dtype.itemsize
        for name in selected
    )
    internal_bytes = (
        sum(
            count
            * CANONICAL_CHANNELS[name].components
            * CANONICAL_CHANNELS[name].dtype.itemsize
            for name in needed - set(selected)
        )
        if output_dir is None
        else 0
    )
    lobe_output_upper = (
        count * LOBE_DTYPE.itemsize * (4 if state.config.exposure == 0.0 else 12)
        if rich
        else 0
    )
    tile_work = (
        largest_rate**2 * min(chunk_size, width) * min(chunk_size, height) * 8 * 22
    )
    if reference:
        tile_work += min(chunk_size, width) * min(chunk_size, height) * 8 * 80
    budget = state.config.memory_budget_mb * 1024 * 1024
    working_upper = internal_bytes + tile_work
    if working_upper > budget:
        raise MemoryError(
            "estimated temporary arrays and tile work exceed memory_budget_mb; request fewer maps, use output_dir, reduce chunk_size or raise the budget"
        )
    cache_bytes = 4 * count * 8 * 8
    use_cache = (
        rich
        and cache_bytes <= 384 * 1024 * 1024
        and working_upper + cache_bytes <= budget
    )
    if use_cache:
        working_upper += cache_bytes
    destination = None if output_dir is None else Path(output_dir)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    working = {
        name: _allocate(name, image_size, destination, temporary=name not in selected)
        for name in needed
    }
    sample_cache: dict[str, np.ndarray] | None = None
    if use_cache:
        sample_cache = {}
        for name in _LOBE_CACHE_FIELDS:
            shape = (
                (2 * height, 2 * width, 2)
                if name == "anisotropy_axis"
                else (2 * height, 2 * width)
            )
            if destination is None:
                sample_cache[name] = np.empty(shape, dtype=np.float64)
            else:
                sample_cache[name] = np.lib.format.open_memmap(
                    destination / f".work_lobe_cache_{name}.npy",
                    mode="w+",
                    dtype=np.float64,
                    shape=shape,
                )
    assert state.config.size_mm is not None
    reference_rate_counts = {16: 0, 32: 0}
    for y0 in range(0, height, chunk_size):
        y1 = min(height, y0 + chunk_size)
        for x0 in range(0, width, chunk_size):
            x1 = min(width, x0 + chunk_size)
            if reference:
                try:
                    fields, rates = _reference_fields(state, x0, x1, y0, y1, image_size)
                except ValueError:
                    for name, value in working.items():
                        _close_working_map(value)
                        if destination is not None:
                            filename = (
                                f"{name}.npy"
                                if name in selected
                                else f".work_{name}.npy"
                            )
                            (destination / filename).unlink(missing_ok=True)
                    raise
                for rate in reference_rate_counts:
                    reference_rate_counts[rate] += int(np.count_nonzero(rates == rate))
            else:
                sub = _subsamples(state, x0, x1, y0, y1, image_size, quadrature)
                if sample_cache is not None:
                    for name, values in sample_cache.items():
                        values[2 * y0 : 2 * y1, 2 * x0 : 2 * x1] = sub[name]
                fields = _integrate_fields(sub, quadrature)
            region = np.s_[y0:y1, x0:x1]
            for name in (
                "height_um",
                "roughness",
                "metallic",
                "patina_coverage",
                "white_stain_coverage",
            ):
                if name in needed:
                    working[name][region] = fields[name].astype(np.float32)
            # Exact visible-fraction closure after float32 quantisation.
            if "metallic" in needed:
                working["metallic"][region] = np.maximum(
                    1.0
                    - fields["patina_coverage"].astype(np.float32)
                    - fields["white_stain_coverage"].astype(np.float32),
                    0.0,
                )
            if "anisotropy" in needed or "anisotropy_axis" in needed:
                vector = fields["anisotropy_vector"]
                magnitude = np.linalg.norm(vector, axis=-1)
                # Use the contract's coherence threshold for both channels;
                # a vanishing lobe has no meaningful normalised direction.
                coherent = magnitude > 1e-6
                if "anisotropy" in needed:
                    working["anisotropy"][region] = np.where(
                        coherent, np.minimum(magnitude, 1.0), 0.0
                    ).astype(np.float32)
                if "anisotropy_axis" in needed:
                    working["anisotropy_axis"][region] = np.divide(
                        vector,
                        magnitude[..., None],
                        out=np.zeros_like(vector),
                        where=coherent[..., None],
                    ).astype(np.float32)
            if "base_color_linear" in selected:
                fractions = [
                    working[name][region][..., None]
                    for name in ("metallic", "patina_coverage", "white_stain_coverage")
                ]
                colors = [
                    zinc_f0(),
                    np.array(_DEPOSIT_OPTICS[1]["diffuse_color_linear"]),
                    np.array(_DEPOSIT_OPTICS[2]["diffuse_color_linear"]),
                ]
                working["base_color_linear"][region] = sum(
                    f * color for f, color in zip(fractions, colors, strict=True)
                ).astype(np.float32)
            for name in (
                "substrate_height_um",
                "deposit_height_um",
                "boundary_distance_mm",
            ):
                if name in selected:
                    working[name][region] = fields[name].astype(np.float32)
            if "grain_id" in selected or "orientation_id" in selected:
                lx, ly = state.config.size_mm
                xx = lx * (np.arange(x0, x1, dtype=np.float64) + 0.5) / width
                yy = ly * (np.arange(y0, y1, dtype=np.float64) + 0.5) / height
                ids = state.partition.query(xx[None, :], yy[:, None]).ids
                for name in ("grain_id", "orientation_id"):
                    if name in selected:
                        working[name][region] = ids
    if "normal_ts" in needed:
        _normal_tiles(
            working["height_um"], working["normal_ts"], state.config.size_mm, chunk_size
        )
    if "finite_band_slope" in selected:
        n = working["normal_ts"]
        working["finite_band_slope"][..., 0] = -n[..., 0] / n[..., 2]
        working["finite_band_slope"][..., 1] = n[..., 1] / n[..., 2]
    lobes = (
        _append_lobes(state, image_size, chunk_size, working, destination, sample_cache)
        if rich
        else None
    )
    for value in working.values():
        if isinstance(value, np.memmap):
            value.flush()
        value.setflags(write=False)
    arrays = {name: working[name] for name in selected}
    metadata = _metadata(state, image_size, quadrature)
    metadata["filtering"].update(
        {
            "policy_version": "shared-midpoint-2",
            "quality": state.config.quality,
            "spatial_convergence_checked": reference,
            "finite_band_certified": False,
            "maximum_sample_points_per_batch": _SAMPLING_POINT_LIMIT,
        }
    )
    if reference:
        metadata["filtering"].update(
            {
                "quadrature": "8 to 16 to 32 per axis; independent 17/33 checks",
                "accepted_rate_pixel_counts": {
                    str(rate): amount for rate, amount in reference_rate_counts.items()
                },
                "absolute_component_tolerances": dict(_SPATIAL_TOLERANCES),
                "pixel_limit": _REFERENCE_MAP_PIXEL_LIMIT,
            }
        )
    metadata["resource_estimates"] = {
        "resident_selected_output_bytes": selected_bytes
        + (lobe_output_upper if rich and destination is None else 0),
        "temporary_working_upper_bytes": working_upper,
        "cached_subpixel_bytes": cache_bytes if use_cache else 0,
        "mapped_output": destination is not None,
    }
    result = MaterialMaps(arrays, metadata, lobes)
    if sample_cache is not None:
        for value in sample_cache.values():
            _close_working_map(value)
        del sample_cache
    if destination is not None:
        for name in needed - set(selected):
            value = working.pop(name)
            _close_working_map(value)
            (destination / f".work_{name}.npy").unlink(missing_ok=True)
        if use_cache:
            for name in _LOBE_CACHE_FIELDS:
                (destination / f".work_lobe_cache_{name}.npy").unlink(missing_ok=True)
    return result


def generate(
    shape: tuple[int, int] = (512, 512),
    rng: np.random.Generator | None = None,
    *,
    galvanised: GalvanisedConfig | Mapping[str, Any] | None = None,
    galvanised_preset: str | None = None,
    preview: PreviewConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Legacy RGB adapter; ``shape`` is internal ``(height, width)``."""
    if galvanised is not None and galvanised_preset is not None:
        raise ValueError("galvanised and galvanised_preset are mutually exclusive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    height, width = shape
    size = validate_image_size((width, height))
    config = (
        galvanised
        if galvanised_preset is None
        else GalvanisedConfig.from_mapping({"preset": galvanised_preset})
    )
    state = build_state(config, material_key=material_key_from_rng(rng), size=size)
    if preview is None:
        preview_config = PreviewConfig()
    elif isinstance(preview, PreviewConfig):
        preview_config = preview
    elif isinstance(preview, Mapping):
        preview_config = PreviewConfig(**dict(preview))
    else:
        raise TypeError("preview must be PreviewConfig, mapping, or None")
    from texture_generators.core.material_render import render_material_array

    sample_size = (max(3, width), max(3, height))
    sampled = sample_state(state, size=sample_size)
    if sample_size != size:
        # Legacy 1/2-pixel previews use an explicitly coarse representation:
        # the unsupported derivative axis is flat in both the geometric and
        # material-conditioned normals. Map exports still require >=3 axes.
        arrays = dict(sampled.arrays)
        normal = np.array(arrays["normal_ts"], copy=True)
        if width < 3:
            normal[..., 0] = 0.0
        if height < 3:
            normal[..., 1] = 0.0
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        arrays["normal_ts"] = normal.astype(np.float32)
        lobes = None if sampled.lobes is None else np.array(sampled.lobes, copy=True)
        if lobes is not None:
            lobe_normal = lobes["normal_ts"]
            if width < 3:
                lobe_normal[:, 0] = 0.0
            if height < 3:
                lobe_normal[:, 1] = 0.0
            lobe_normal /= np.linalg.norm(lobe_normal, axis=-1, keepdims=True)
            tangent = lobes["tangent_ts"]
            tangent -= (
                np.sum(tangent * lobe_normal, axis=-1, keepdims=True) * lobe_normal
            )
            tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
        metadata = dict(sampled.metadata)
        metadata["filtering"] = {
            **metadata["filtering"],
            "coarse_preview_flat_axes": [
                axis for axis, count in (("x", width), ("y", height)) if count < 3
            ],
        }
        sampled = MaterialMaps(arrays, metadata, lobes)
    rgb = render_material_array(sampled, preview=preview_config)
    if height != sample_size[1]:
        row_positions = np.linspace(0.0, sample_size[1] - 1, height)
        rgb = (
            np.stack(
                [
                    np.interp(
                        row_positions,
                        np.arange(sample_size[1]),
                        rgb[:, column, channel],
                    )
                    for column in range(sample_size[0])
                    for channel in range(3)
                ],
                axis=-1,
            )
            .reshape(height, sample_size[0], 3)
            .astype(np.float32)
        )
    if width != sample_size[0]:
        column_positions = np.linspace(0.0, sample_size[0] - 1, width)
        rgb = (
            np.stack(
                [
                    np.interp(
                        column_positions,
                        np.arange(sample_size[0]),
                        rgb[row, :, channel],
                    )
                    for row in range(height)
                    for channel in range(3)
                ],
                axis=-1,
            )
            .reshape(width, height, 3)
            .transpose(1, 0, 2)
            .astype(np.float32)
        )
    return rgb.astype(np.float32)
