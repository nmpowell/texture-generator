"""Deterministic distant-light rendering of galvanised material samples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from texture_generators.core.colour import linear_to_srgb
from texture_generators.core.conductor import resource_hashes, zinc_fresnel
from texture_generators.core.energy_compensation import (
    directional_albedo,
    energy_resource_hashes,
    evaluate_ggx_compensated,
    schlick_directional_albedo,
    schlick_directional_moment,
    zinc_total_directional_albedo,
    zinc_total_directional_moment,
)
from texture_generators.core.material import MaterialMaps
from texture_generators.core.microfacet import (
    evaluate_ggx,
    openpbr_widths,
    tangent_frame,
)
from texture_generators.core.physical import height_to_normal_physical
from texture_generators.materials.galvanised_config import PreviewConfig

__all__ = ["render_material", "render_material_array", "render_reference"]

_LOBE_CHUNK = 16384
_LIGHT_BATCH = 4
_REFERENCE_POINT_CHUNK = 4096
_DERIVATIVE_TOLERANCE = 1e-5
_MAX_DERIVATIVE_HALVINGS = 8
_LobeChunk = tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


class ReferenceDerivativeConvergenceError(ValueError):
    """A physical point normal did not meet the bounded derivative test."""

    def __init__(self, unresolved_points: int, max_component_difference: float):
        self.unresolved_points = unresolved_points
        self.max_component_difference = max_component_difference
        super().__init__(
            f"physical normal derivative failed to converge for {unresolved_points} "
            f"point(s) after {_MAX_DERIVATIVE_HALVINGS} halvings; maximum component "
            f"difference {max_component_difference:.6g} exceeds {_DERIVATIVE_TOLERANCE}"
        )


@dataclass(frozen=True)
class _Optics:
    material_id: int
    kind: str
    diffuse: NDArray[np.float64]
    roughness: float
    anisotropy: float
    ior: float


_DEFAULT_OPTICS: tuple[dict[str, Any], ...] = (
    {
        "material_id": 0,
        "kind": "conductor",
        "diffuse_color_linear": [0.0, 0.0, 0.0],
        "roughness": 0.30,
        "anisotropy": 0.55,
        "ior": None,
    },
    {
        "material_id": 1,
        "kind": "dielectric",
        "diffuse_color_linear": [0.32, 0.35, 0.32],
        "roughness": 0.70,
        "anisotropy": 0.04,
        "ior": 1.50,
    },
    {
        "material_id": 2,
        "kind": "dielectric",
        "diffuse_color_linear": [0.72, 0.73, 0.69],
        "roughness": 0.86,
        "anisotropy": 0.0,
        "ior": 1.50,
    },
)


def _parameters(metadata: Mapping[str, Any]) -> tuple[_Optics, ...]:
    source = metadata.get("optical_parameters", _DEFAULT_OPTICS)
    if not isinstance(source, (list, tuple)) or not source:
        raise ValueError("optical_parameters must be a nonempty list")
    result = []
    for entry in source:
        if not isinstance(entry, Mapping):
            raise ValueError("optical parameter entries must be mappings")
        kind = str(entry["kind"])
        if kind not in ("conductor", "dielectric"):
            raise ValueError("optical kind must be conductor or dielectric")
        material_id = int(entry["material_id"])
        if material_id not in (0, 1, 2) or (material_id == 0) != (kind == "conductor"):
            raise ValueError("optical material_id/kind mismatch")
        diffuse = np.asarray(
            entry.get("diffuse_color_linear", [0, 0, 0]), dtype=np.float64
        )
        roughness = float(entry.get("roughness", 0.5))
        anisotropy = float(entry.get("anisotropy", 0.0))
        raw_ior = entry.get("ior", 1.5)
        ior = 1.5 if raw_ior is None else float(raw_ior)
        if (
            diffuse.shape != (3,)
            or np.any(~np.isfinite(diffuse))
            or np.any((diffuse < 0) | (diffuse > 1))
        ):
            raise ValueError("diffuse_color_linear must be finite RGB in [0,1]")
        if not (0 <= roughness <= 1 and 0 <= anisotropy <= 1 and ior >= 1):
            raise ValueError("invalid optical roughness, anisotropy or IOR")
        result.append(_Optics(material_id, kind, diffuse, roughness, anisotropy, ior))
    return tuple(result)


def _schlick_dielectric(cosine: NDArray[np.float64], ior: float) -> NDArray[np.float64]:
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    scalar = f0 + (1.0 - f0) * (1.0 - cosine) ** 5
    return np.broadcast_to(scalar[..., None], (*scalar.shape, 3))


def _bsdf(
    normal: NDArray[np.float64],
    tangent: NDArray[np.float64],
    wi: NDArray[np.float64],
    wo: NDArray[np.float64],
    alpha_t: NDArray[np.float64],
    alpha_b: NDArray[np.float64],
    optics: _Optics,
    prepared_view: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    if optics.kind == "conductor":
        return evaluate_ggx(
            normal, tangent, wi, wo, alpha_t, alpha_b, zinc_fresnel
        ) + evaluate_ggx_compensated(
            normal,
            tangent,
            wi,
            wo,
            alpha_t,
            alpha_b,
            outgoing_albedo=None if prepared_view is None else prepared_view[0],
            mean_albedo=None if prepared_view is None else prepared_view[1],
        )

    def fresnel(cosine: NDArray[np.float64]) -> NDArray[np.float64]:
        return _schlick_dielectric(cosine, optics.ior)

    spec = evaluate_ggx(normal, tangent, wi, wo, alpha_t, alpha_b, fresnel)
    # The reciprocal Kulla--Conty residual uses the directional albedo of
    # this same Schlick specular lobe. Its white-diffuse furnace is unity.
    ei, average = schlick_directional_albedo(
        normal, tangent, wi, alpha_t, alpha_b, optics.ior
    )
    if prepared_view is None:
        eo, _ = schlick_directional_albedo(
            normal, tangent, wo, alpha_t, alpha_b, optics.ior
        )
    else:
        eo, average = prepared_view
    diffuse = (1.0 - ei) * (1.0 - eo) / (np.pi * np.maximum(1.0 - average, 1e-12))
    front = (np.sum(normal * wi, axis=-1) > 0.0) & (np.sum(normal * wo, axis=-1) > 0.0)
    return spec + np.where(front[..., None], diffuse[..., None] * optics.diffuse, 0.0)


def _normal_strength(
    normals: NDArray[np.float64], strength: float
) -> NDArray[np.float64]:
    if strength == 1.0:
        return normals
    transformed = np.array(normals, dtype=np.float64, copy=True)
    transformed[..., :2] *= strength
    transformed /= np.linalg.norm(transformed, axis=-1, keepdims=True)
    return transformed


def _constant_environment_response(
    normal: NDArray[np.float64],
    tangent: NDArray[np.float64],
    view: NDArray[np.float64],
    alpha_t: NDArray[np.float64],
    alpha_b: NDArray[np.float64],
    optics: _Optics,
) -> NDArray[np.float64]:
    front = np.sum(normal * view, axis=-1) > 0.0
    if optics.kind == "conductor":
        result = zinc_total_directional_albedo(normal, tangent, view, alpha_t, alpha_b)
    else:
        specular, _ = schlick_directional_albedo(
            normal, tangent, view, alpha_t, alpha_b, optics.ior
        )
        result = specular[..., None] + (1.0 - specular)[..., None] * optics.diffuse
    return np.where(front[..., None], result, 0.0)


def _overcast_directional_response(
    normal: NDArray[np.float64],
    tangent: NDArray[np.float64],
    view: NDArray[np.float64],
    alpha_t: NDArray[np.float64],
    alpha_b: NDArray[np.float64],
    optics: _Optics,
) -> NDArray[np.float64]:
    if optics.kind == "conductor":
        local_moment = zinc_total_directional_moment(
            normal, tangent, view, alpha_t, alpha_b
        )
    else:
        specular_moment, diffuse_z = schlick_directional_moment(
            normal, tangent, view, alpha_t, alpha_b, optics.ior
        )
        specular, _ = schlick_directional_albedo(
            normal, tangent, view, alpha_t, alpha_b, optics.ior
        )
        local_moment = np.broadcast_to(
            specular_moment[..., None, :], (*specular_moment.shape[:-1], 3, 3)
        ).copy()
        local_moment[..., 2] += ((1.0 - specular) * diffuse_z)[
            ..., None
        ] * optics.diffuse
    bitangent = np.cross(normal, tangent)
    world_z_in_frame = np.stack(
        (tangent[..., 2], bitangent[..., 2], normal[..., 2]), axis=-1
    )
    result = np.sum(local_moment * world_z_in_frame[..., None, :], axis=-1)
    return np.where((np.sum(normal * view, axis=-1) > 0.0)[..., None], result, 0.0)


def _light_quadrature(
    preview: PreviewConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fixed Gaussian distant-emitter quadrature.

    Constant and linear-gradient environments are integrated with cached
    directional moments. Returned emitter weights are incident radiance times
    solid-angle weight, independent of surface content and material seed.
    """
    if preview.rig == "overcast":
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)

    elevation, sigma_x, sigma_y, peak = {
        "studio": (np.deg2rad(58), 0.38, 0.10, 3.0),
        "oblique": (np.deg2rad(48), 0.12, 0.12, 9.0),
        "grazing": (np.deg2rad(13), 0.10, 0.055, 15.0),
    }[preview.rig]
    azimuth = np.deg2rad(preview.light_azimuth_deg)
    centre = np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    tx = np.asarray([-np.sin(azimuth), np.cos(azimuth), 0.0])
    ty = np.cross(centre, tx)
    hnodes, hweights = np.polynomial.hermite.hermgauss(
        min(9, max(3, round(np.sqrt(preview.render_samples / 2))))
    )
    x, y = np.meshgrid(hnodes, hnodes, indexing="ij")
    source_dir = (
        centre
        + np.sqrt(2) * sigma_x * x[..., None] * tx
        + np.sqrt(2) * sigma_y * y[..., None] * ty
    )
    radius = np.linalg.norm(source_dir, axis=-1, keepdims=True)
    source_dir /= radius
    # The gnomonic tangent-plane map has dOmega/dxdy=(1+x²+y²)^(-3/2).
    source_weight = (
        peak
        * 2
        * sigma_x
        * sigma_y
        * np.outer(hweights, hweights)
        / radius[..., 0] ** 3
    )
    valid = source_dir[..., 2] > 0
    return source_dir[valid], source_weight[valid]


def _render_lobes(
    indices: NDArray[np.int64],
    weights: NDArray[np.float64],
    _material_ids: NDArray[np.int64],
    parameters: NDArray[np.int64],
    normals: NDArray[np.float64],
    tangents: NDArray[np.float64],
    alpha_t: NDArray[np.float64],
    alpha_b: NDArray[np.float64],
    optics: tuple[_Optics, ...],
    shape: tuple[int, int],
    preview: PreviewConfig,
) -> NDArray[np.float32]:
    def chunks() -> Iterator[_LobeChunk]:
        for start in range(0, len(indices), _LOBE_CHUNK):
            chunk = slice(start, start + _LOBE_CHUNK)
            yield (
                indices[chunk],
                weights[chunk],
                parameters[chunk],
                normals[chunk],
                tangents[chunk],
                alpha_t[chunk],
                alpha_b[chunk],
            )

    return _render_lobe_chunks(chunks(), optics, shape, preview).astype(np.float32)


def _render_rich_lobes(
    records: np.ndarray,
    optics: tuple[_Optics, ...],
    shape: tuple[int, int],
    preview: PreviewConfig,
) -> NDArray[np.float32]:
    def chunks() -> Iterator[_LobeChunk]:
        for start in range(0, len(records), _LOBE_CHUNK):
            block = records[start : start + _LOBE_CHUNK]
            parameters = block["optical_parameter_index"].astype(np.int64)
            if np.any(parameters >= len(optics)):
                raise ValueError("lobe optical_parameter_index out of range")
            for index, optical in enumerate(optics):
                if np.any(
                    block["material_id"][parameters == index] != optical.material_id
                ):
                    raise ValueError("lobe material_id differs from optical parameter")
            yield (
                block["pixel_index"].astype(np.int64),
                block["weight"].astype(np.float64),
                parameters,
                block["normal_ts"].astype(np.float64),
                block["tangent_ts"].astype(np.float64),
                block["alpha_t"].astype(np.float64),
                block["alpha_b"].astype(np.float64),
            )

    return _render_lobe_chunks(chunks(), optics, shape, preview).astype(np.float32)


def _render_lobe_chunks(
    chunks: Iterable[_LobeChunk],
    optics: tuple[_Optics, ...],
    shape: tuple[int, int],
    preview: PreviewConfig,
) -> NDArray[np.float64]:
    directions, radiances = _light_quadrature(preview)
    output = np.zeros((shape[0] * shape[1], 3), dtype=np.float64)
    view = np.asarray(preview.view, dtype=np.float64)
    for pixel, weights, parameters, normals, tangents, alpha_t, alpha_b in chunks:
        normal = _normal_strength(normals, preview.normal_strength)
        tangent = tangents
        # Reproject after artistic normal-strength adjustment.
        tangent = tangent - np.sum(tangent * normal, axis=1)[:, None] * normal
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        local = np.zeros((len(pixel), 3), dtype=np.float64)
        constant_radiance = {
            "studio": 0.24,
            "oblique": 0.14,
            "overcast": 0.60,
            "grazing": 0.10,
        }[preview.rig]
        subsets = []
        for parameter_index, optical in enumerate(optics):
            selected = parameters == parameter_index
            if not np.any(selected):
                continue
            subset = np.flatnonzero(selected)
            subset_normal = normal[subset]
            subset_tangent = tangent[subset]
            subset_at = alpha_t[subset]
            subset_ab = alpha_b[subset]
            if len(directions):
                if optical.kind == "conductor":
                    prepared_view = directional_albedo(
                        subset_normal, subset_tangent, view, subset_at, subset_ab
                    )
                else:
                    prepared_view = schlick_directional_albedo(
                        subset_normal,
                        subset_tangent,
                        view,
                        subset_at,
                        subset_ab,
                        optical.ior,
                    )
                subsets.append(
                    (
                        subset,
                        subset_normal,
                        subset_tangent,
                        subset_at,
                        subset_ab,
                        optical,
                        prepared_view,
                    )
                )
            local[subset] += constant_radiance * _constant_environment_response(
                subset_normal,
                subset_tangent,
                view,
                subset_at,
                subset_ab,
                optical,
            )
            if preview.rig == "overcast":
                local[subset] += 0.40 * _overcast_directional_response(
                    subset_normal,
                    subset_tangent,
                    view,
                    subset_at,
                    subset_ab,
                    optical,
                )
        for light_start in range(0, len(directions), _LIGHT_BATCH):
            light_directions = directions[light_start : light_start + _LIGHT_BATCH]
            light_radiances = radiances[light_start : light_start + _LIGHT_BATCH]
            cosine = np.maximum(
                0.0, np.sum(normal[None] * light_directions[:, None, :], axis=-1)
            )
            if not np.any(cosine > 0):
                continue
            for (
                subset,
                subset_normal,
                subset_tangent,
                subset_at,
                subset_ab,
                optical,
                prepared_view,
            ) in subsets:
                f = _bsdf(
                    subset_normal[None],
                    subset_tangent[None],
                    light_directions[:, None, :],
                    view,
                    subset_at[None],
                    subset_ab[None],
                    optical,
                    prepared_view,
                )
                for light_index, radiance in enumerate(light_radiances):
                    local[subset] += (
                        radiance * cosine[light_index, subset, None] * f[light_index]
                    )
        np.add.at(output, pixel, local * weights[:, None])
    return output.reshape((*shape, 3))


def _single_lobes(
    maps: MaterialMaps, optics: tuple[_Optics, ...]
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    width, height = maps.size
    shape = (height, width)
    if "normal_ts" in maps:
        normal = np.asarray(maps["normal_ts"], dtype=np.float64)
    elif "height_um" in maps:
        size_mm = maps.metadata.get("size_mm")
        if size_mm is None:
            raise ValueError("height-only rendering requires metadata size_mm")
        normal = height_to_normal_physical(maps["height_um"], size_mm).astype(
            np.float64
        )
    else:
        normal = np.broadcast_to([0.0, 0.0, 1.0], (*shape, 3)).astype(np.float64)
    axis = np.asarray(
        maps.arrays.get("anisotropy_axis", np.broadcast_to([1.0, 0.0], (*shape, 2))),
        dtype=np.float64,
    )
    tangent, _ = tangent_frame(normal, axis)
    roughness = np.asarray(
        maps.arrays.get("roughness", np.full(shape, optics[0].roughness)),
        dtype=np.float64,
    )
    anisotropy = np.asarray(
        maps.arrays.get("anisotropy", np.full(shape, optics[0].anisotropy)),
        dtype=np.float64,
    )
    at, ab = openpbr_widths(roughness, anisotropy)
    white = np.asarray(
        maps.arrays.get("white_stain_coverage", np.zeros(shape)), dtype=np.float64
    )
    patina = np.asarray(
        maps.arrays.get("patina_coverage", np.zeros(shape)), dtype=np.float64
    )
    if "patina_coverage" not in maps and "metallic" in maps:
        patina = 1.0 - white - maps["metallic"]
    zinc = np.asarray(
        maps.arrays.get("metallic", 1.0 - patina - white), dtype=np.float64
    )
    coverages = (zinc, patina, white)
    if any(np.any(layer < -1e-6) for layer in coverages) or not np.allclose(
        zinc + patina + white, 1.0, atol=1e-6
    ):
        raise ValueError(
            "visible coverage constants must form disjoint fractions summing to one"
        )
    available = {optical.material_id for optical in optics}
    if any(
        np.any(coverages[identifier] > 0) and identifier not in available
        for identifier in range(3)
    ):
        raise ValueError("optical_parameters omits a visible material")
    pieces = []
    for p, optical in enumerate(optics):
        coverage = coverages[optical.material_id].ravel()
        selected = np.flatnonzero(coverage > 0)
        pieces.append(
            (
                selected.astype(np.int64),
                coverage[selected],
                np.full(len(selected), optical.material_id, dtype=np.int64),
                np.full(len(selected), p, dtype=np.int64),
                normal.reshape(-1, 3)[selected],
                tangent.reshape(-1, 3)[selected],
                at.ravel()[selected],
                ab.ravel()[selected],
            )
        )
    return tuple(np.concatenate([piece[k] for piece in pieces]) for k in range(8))  # type: ignore[return-value]


def render_material_array(
    maps: MaterialMaps,
    *,
    preview: PreviewConfig | None = None,
    output: Literal["display", "linear"] = "display",
) -> NDArray[np.float32]:
    """Render sampled maps to HxWx3 display sRGB or unclipped linear radiance."""
    if output not in ("display", "linear"):
        raise ValueError("output must be 'display' or 'linear'")
    for field, installed in (
        ("resource_hashes", resource_hashes),
        ("renderer_resource_hashes", energy_resource_hashes),
    ):
        recorded = maps.metadata.get(field)
        if recorded is not None and (
            not isinstance(recorded, Mapping) or dict(recorded) != dict(installed())
        ):
            raise ValueError(f"{field} does not match installed rendering resources")
    maps.validate()
    setting = PreviewConfig() if preview is None else preview
    optics = _parameters(maps.metadata)
    width, height = maps.size
    if maps.metadata.get("representation") == "rich":
        if maps.lobes is None:
            raise ValueError("rich representation requires lobe records")
        linear = _render_rich_lobes(maps.lobes, optics, (height, width), setting)
    else:
        items = _single_lobes(maps, optics)
        linear = _render_lobes(*items, optics, (height, width), setting)
    if output == "linear":
        return linear
    exposed = np.maximum(linear.astype(np.float64) * 2.0**setting.exposure_stops, 0.0)
    # Fixed extended-Reinhard display curve, white point 4 linear units.
    mapped = exposed * (1.0 + exposed / 16.0) / (1.0 + exposed)
    return linear_to_srgb(mapped)


def render_material(
    maps: MaterialMaps, *, preview: PreviewConfig | None = None
) -> Image.Image:
    """Render a galvanised material preview as an RGB Pillow image."""
    array = render_material_array(maps, preview=preview, output="display")
    encoded = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(encoded, mode="RGB")


def _reference_derivative_step_mm(
    state: Any,
    requested: tuple[float, float] | None,
) -> tuple[float, float]:
    if requested is not None:
        values = np.asarray(requested)
        if (
            values.shape != (2,)
            or values.dtype.kind == "b"
            or any(isinstance(value, (bool, np.bool_)) for value in requested)
        ):
            raise ValueError(
                "derivative_step_mm must contain two positive finite lengths"
            )
        try:
            requested_step = tuple(float(value) for value in values)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "derivative_step_mm must contain two positive finite lengths"
            ) from error
        if not all(np.isfinite(value) and value > 0 for value in requested_step):
            raise ValueError(
                "derivative_step_mm must contain two positive finite lengths"
            )
        return requested_step[0], requested_step[1]

    lx, ly = state.config.size_mm
    candidates = [0.001]
    segments = state.dendrites.segments
    if len(segments):
        candidates.extend(
            (
                0.45 * float(np.min(segments["width_mm"])) / 32.0,
                float(np.min(segments["period_mm"])) / 64.0,
            )
        )
    spectrum = state.micro_spectrum
    if len(spectrum):
        frequencies = np.hypot(spectrum[:, 0] / lx, spectrum[:, 1] / ly)
        candidates.append(1.0 / float(np.max(frequencies)) / 64.0)
    # The tapered grain-boundary groove and compact defects can impose a
    # shorter physical transition than the branch/micro carriers.
    groove_width = 0.45 * max(0.045 * state.config.spangle_diameter_mm, 0.02)
    candidates.append(groove_width / 32.0)
    if len(state.defects):
        candidates.append(float(np.min(state.defects["radius_mm"])) / 32.0)
    weather_frequency = state.weather.frequency
    if len(weather_frequency):
        weather_cycles = np.hypot(
            weather_frequency[..., 0] / lx, weather_frequency[..., 1] / ly
        )
        candidates.append(1.0 / float(np.max(weather_cycles)) / 64.0)
    default_step = min(candidates)
    if not np.isfinite(default_step) or default_step <= 0:
        raise ValueError(
            "physical surface features do not define a positive derivative step"
        )
    return default_step, default_step


def _physical_point_normal(
    state: Any,
    x_mm: NDArray[np.float64],
    y_mm: NDArray[np.float64],
    step_mm: tuple[float, float],
) -> NDArray[np.float64]:
    from texture_generators.materials.galvanised import sample_points

    sx, sy = step_mm
    if (
        np.any(x_mm + sx == x_mm)
        or np.any(x_mm - sx == x_mm)
        or np.any(y_mm + sy == y_mm)
        or np.any(y_mm - sy == y_mm)
    ):
        raise ValueError("physical derivative step is below coordinate precision")
    footprint = (0.0, 0.0)
    left = sample_points(state, x_mm - sx, y_mm, footprint_mm=footprint)["height_um"]
    right = sample_points(state, x_mm + sx, y_mm, footprint_mm=footprint)["height_um"]
    up = sample_points(state, x_mm, y_mm - sy, footprint_mm=footprint)["height_um"]
    down = sample_points(state, x_mm, y_mm + sy, footprint_mm=footprint)["height_um"]
    px = 1e-3 * (right - left) / (2.0 * sx)
    py = 1e-3 * (down - up) / (2.0 * sy)
    normal = np.stack((-px, py, np.ones_like(px)), axis=-1).astype(np.float64)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal


def _converged_reference_normals(
    state: Any,
    x_mm: NDArray[np.float64],
    y_mm: NDArray[np.float64],
    step_mm: tuple[float, float],
) -> tuple[NDArray[np.float64], dict[str, int | float]]:
    x, y = np.broadcast_arrays(
        np.asarray(x_mm, dtype=np.float64), np.asarray(y_mm, dtype=np.float64)
    )
    flat_x, flat_y = x.ravel(), y.ravel()
    previous = _physical_point_normal(state, flat_x, flat_y, step_mm)
    accepted = np.empty_like(previous)
    active = np.arange(len(flat_x))
    refined_count = 0
    max_accepted_error = 0.0
    max_gradient_error = 0.0
    max_angle_deg = 0.0
    angle_square_sum = 0.0
    max_halvings = 0
    for level in range(1, _MAX_DERIVATIVE_HALVINGS + 1):
        step = (step_mm[0] / 2**level, step_mm[1] / 2**level)
        current = _physical_point_normal(state, flat_x[active], flat_y[active], step)
        error = np.max(np.abs(previous[active] - current), axis=-1)
        passing = error <= _DERIVATIVE_TOLERANCE
        passed = active[passing]
        accepted[passed] = current[passing]
        if len(passed):
            max_halvings = level
            max_accepted_error = max(max_accepted_error, float(np.max(error[passing])))
            old = previous[passed]
            new = current[passing]
            old_gradient = np.stack(
                (-old[:, 0] / old[:, 2], old[:, 1] / old[:, 2]), axis=-1
            )
            new_gradient = np.stack(
                (-new[:, 0] / new[:, 2], new[:, 1] / new[:, 2]), axis=-1
            )
            max_gradient_error = max(
                max_gradient_error, float(np.max(np.abs(old_gradient - new_gradient)))
            )
            angle = np.degrees(
                np.arctan2(
                    np.linalg.norm(np.cross(old, new), axis=-1),
                    np.sum(old * new, axis=-1),
                )
            )
            max_angle_deg = max(max_angle_deg, float(np.max(angle)))
            angle_square_sum += float(np.sum(np.square(angle)))
            if level > 1:
                refined_count += len(passed)
        remaining = active[~passing]
        if not len(remaining):
            return accepted.reshape((*x.shape, 3)), {
                "point_count": len(flat_x),
                "max_halvings": max_halvings,
                "points_refined_beyond_first": refined_count,
                "max_accepted_component_difference": max_accepted_error,
                "max_accepted_gradient_difference": max_gradient_error,
                "max_accepted_angle_deg": max_angle_deg,
                "accepted_angle_square_sum_deg2": angle_square_sum,
            }
        previous[remaining] = current[~passing]
        active = remaining
    raise ReferenceDerivativeConvergenceError(
        len(active), float(np.max(error[~passing]))
    )


def render_reference(
    state: Any,
    size: tuple[int, int],
    *,
    preview: PreviewConfig | None = None,
    samples_per_axis: int = 4,
    output: Literal["linear", "display"] = "linear",
    derivative_step_mm: tuple[float, float] | None = None,
    derivative_diagnostics: dict[str, Any] | None = None,
) -> NDArray[np.float32]:
    """Reference subpixel BRDF integral from the common physical surface state.

    Point positions and light quadrature are deterministic and independent of
    the surface key. Normals are converged physical central differences of
    freshly sampled height, with a state-derived step independent of output
    resolution and subpixel quadrature.
    """
    from texture_generators.materials.galvanised import (
        optical_parameters,
        sample_points,
    )

    if (
        isinstance(samples_per_axis, bool)
        or not isinstance(samples_per_axis, Integral)
        or samples_per_axis < 1
    ):
        raise ValueError("samples_per_axis must be a positive integer")
    if output not in ("linear", "display"):
        raise ValueError("output must be 'linear' or 'display'")
    width, height = size
    if min(width, height) < 1:
        raise ValueError("size must contain positive dimensions")
    lx, ly = state.config.size_mm
    dx, dy = lx / width, ly / height
    derivative_step = _reference_derivative_step_mm(state, derivative_step_mm)
    if derivative_diagnostics is not None:
        derivative_diagnostics.update(
            {
                "initial_step_mm": list(derivative_step),
                "component_tolerance": _DERIVATIVE_TOLERANCE,
                "maximum_halvings": _MAX_DERIVATIVE_HALVINGS,
                "point_count": 0,
                "max_halvings": 0,
                "points_refined_beyond_first": 0,
                "max_accepted_component_difference": 0.0,
                "max_accepted_gradient_difference": 0.0,
                "max_accepted_angle_deg": 0.0,
                "accepted_angle_square_sum_deg2": 0.0,
                "rms_accepted_angle_deg": 0.0,
            }
        )
    setting = PreviewConfig() if preview is None else preview
    optics = _parameters({"optical_parameters": optical_parameters(state)})
    image = np.zeros((height, width, 3), dtype=np.float64)
    tile_width = min(width, _REFERENCE_POINT_CHUNK)
    tile_rows = max(1, _REFERENCE_POINT_CHUNK // tile_width)
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        for x0 in range(0, width, tile_width):
            x1 = min(width, x0 + tile_width)
            rows, cols = y1 - y0, x1 - x0
            row_index, col_index = np.mgrid[y0:y1, x0:x1]
            for sy in range(samples_per_axis):
                for sx in range(samples_per_axis):
                    x = (col_index + (sx + 0.5) / samples_per_axis) * dx
                    y = (row_index + (sy + 0.5) / samples_per_axis) * dy
                    point = sample_points(state, x, y, footprint_mm=(0.0, 0.0))
                    normal, normal_report = _converged_reference_normals(
                        state, x, y, derivative_step
                    )
                    if derivative_diagnostics is not None:
                        derivative_diagnostics["point_count"] += normal_report[
                            "point_count"
                        ]
                        derivative_diagnostics["max_halvings"] = max(
                            derivative_diagnostics["max_halvings"],
                            normal_report["max_halvings"],
                        )
                        derivative_diagnostics["points_refined_beyond_first"] += (
                            normal_report["points_refined_beyond_first"]
                        )
                        derivative_diagnostics["max_accepted_component_difference"] = (
                            max(
                                derivative_diagnostics[
                                    "max_accepted_component_difference"
                                ],
                                normal_report["max_accepted_component_difference"],
                            )
                        )
                        derivative_diagnostics["max_accepted_gradient_difference"] = (
                            max(
                                derivative_diagnostics[
                                    "max_accepted_gradient_difference"
                                ],
                                normal_report["max_accepted_gradient_difference"],
                            )
                        )
                        derivative_diagnostics["max_accepted_angle_deg"] = max(
                            derivative_diagnostics["max_accepted_angle_deg"],
                            normal_report["max_accepted_angle_deg"],
                        )
                        derivative_diagnostics["accepted_angle_square_sum_deg2"] += (
                            normal_report["accepted_angle_square_sum_deg2"]
                        )
                    tangent, _ = tangent_frame(normal, point["anisotropy_axis"])
                    ids = []
                    fractions = []
                    parameter = []
                    selected_normals = []
                    selected_tangents = []
                    selected_at = []
                    selected_ab = []
                    for p, optical in enumerate(optics):
                        name = {
                            0: "metallic",
                            1: "patina_coverage",
                            2: "white_stain_coverage",
                        }[optical.material_id]
                        coverage = np.asarray(point[name], dtype=np.float64).ravel()
                        selected = np.flatnonzero(coverage > 0.0)
                        if optical.material_id == 0:
                            material_at, material_ab = openpbr_widths(
                                point["intrinsic_roughness"],
                                point["intrinsic_anisotropy"],
                            )
                        else:
                            material_at, material_ab = openpbr_widths(
                                np.full((rows, cols), optical.roughness),
                                np.full((rows, cols), optical.anisotropy),
                            )
                        ids.append(selected.astype(np.int64))
                        fractions.append(coverage[selected])
                        parameter.append(np.full(len(selected), p, dtype=np.int64))
                        selected_normals.append(normal.reshape(-1, 3)[selected])
                        selected_tangents.append(tangent.reshape(-1, 3)[selected])
                        selected_at.append(material_at.ravel()[selected])
                        selected_ab.append(material_ab.ravel()[selected])
                    local = _render_lobe_chunks(
                        [
                            (
                                np.concatenate(ids),
                                np.concatenate(fractions),
                                np.concatenate(parameter),
                                np.concatenate(selected_normals),
                                np.concatenate(selected_tangents),
                                np.concatenate(selected_at),
                                np.concatenate(selected_ab),
                            )
                        ],
                        optics,
                        (rows, cols),
                        setting,
                    )
                    image[y0:y1, x0:x1] += local / samples_per_axis**2
    if derivative_diagnostics is not None and derivative_diagnostics["point_count"]:
        derivative_diagnostics["rms_accepted_angle_deg"] = float(
            np.sqrt(
                derivative_diagnostics["accepted_angle_square_sum_deg2"]
                / derivative_diagnostics["point_count"]
            )
        )
    linear = image.astype(np.float32)
    if output == "linear":
        return linear
    exposed = np.maximum(image * 2.0**setting.exposure_stops, 0.0)
    mapped = exposed * (1.0 + exposed / 16.0) / (1.0 + exposed)
    return linear_to_srgb(mapped)
