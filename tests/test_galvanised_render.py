"""Numerical rendering checks on controlled material and angular coupons."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import texture_generators.core.material_render as material_render
import texture_generators.materials.galvanised as galvanised
from texture_generators.core.conductor import resource_hashes
from texture_generators.core.energy_compensation import (
    directional_albedo,
    energy_resource_hashes,
    evaluate_ggx_compensated,
    schlick_directional_albedo,
)
from texture_generators.core.material import LOBE_DTYPE, MaterialMaps
from texture_generators.core.material_render import (
    _bsdf,
    _constant_environment_response,
    _converged_reference_normals,
    _Optics,
    _overcast_directional_response,
    _parameters,
    _physical_point_normal,
    _reference_derivative_step_mm,
    _render_lobes,
    render_material,
    render_material_array,
    render_reference,
)
from texture_generators.core.microfacet import evaluate_ggx
from texture_generators.core.physical import height_to_normal_physical
from texture_generators.materials.galvanised import build_state
from texture_generators.materials.galvanised_config import PreviewConfig


def _coupon(
    *, mixed: bool = False, rich: bool = False, swap: bool = False
) -> MaterialMaps:
    shape = (4, 5)
    arrays = {
        "height_um": np.zeros(shape, np.float32),
        "normal_ts": np.broadcast_to([0.0, 0.0, 1.0], (*shape, 3))
        .astype(np.float32)
        .copy(),
        "roughness": np.full(shape, 0.45, np.float32),
        "anisotropy": np.full(shape, 0.4, np.float32),
        "anisotropy_axis": np.broadcast_to([1.0, 0.0], (*shape, 2))
        .astype(np.float32)
        .copy(),
        "metallic": np.full(shape, 0.5 if mixed else 1.0, np.float32),
        "patina_coverage": np.full(shape, 0.5 if mixed else 0.0, np.float32),
        "white_stain_coverage": np.zeros(shape, np.float32),
    }
    optical = [
        {
            "material_id": 0,
            "kind": "conductor",
            "diffuse_color_linear": [0, 0, 0],
            "roughness": 0.45,
            "anisotropy": 0.4,
            "ior": None,
        },
        {
            "material_id": 1,
            "kind": "dielectric",
            "diffuse_color_linear": [0.28, 0.33, 0.31],
            "roughness": 0.7,
            "anisotropy": 0.0,
            "ior": 1.5,
        },
    ]
    records = None
    if rich:
        records = np.zeros(shape[0] * shape[1] * 2, dtype=LOBE_DTYPE)
        pixels = np.arange(shape[0] * shape[1], dtype=np.uint32)
        records["pixel_index"] = np.repeat(pixels, 2)
        records["material_id"] = np.tile([0, 1], pixels.size)
        records["optical_parameter_index"] = records["material_id"]
        records["weight"] = 0.5
        nz = np.sqrt(1.0 - 0.35**2)
        records["normal_ts"][0::2] = (-0.35 if swap else 0.35, 0.0, nz)
        records["normal_ts"][1::2] = (0.35 if swap else -0.35, 0.0, nz)
        records["tangent_ts"][0::2] = (nz, 0.0, 0.35 if swap else -0.35)
        records["tangent_ts"][1::2] = (nz, 0.0, -0.35 if swap else 0.35)
        records["alpha_t"] = np.tile([0.18, 0.65], pixels.size)
        records["alpha_b"] = np.tile([0.10, 0.65], pixels.size)
    return MaterialMaps(
        arrays,
        {
            "representation": "rich" if rich else "single_lobe",
            "optical_parameters": optical,
        },
        records,
    )


def _white(cosine: np.ndarray) -> np.ndarray:
    return np.ones((*np.shape(cosine), 3))


@pytest.mark.parametrize(
    "alpha_t,alpha_b,mu,azimuth",
    [
        (0.10, 0.10, 1.0, 0.0),
        (0.35, 0.10, 0.3, np.pi / 4),
        (0.50, 0.20, 1.0, 0.0),
        (1.20, 0.80, 0.3, 0.0),
        (1.40, 0.03, 0.3, 0.0),
    ],
)
def test_compensated_white_furnace(
    alpha_t: float, alpha_b: float, mu: float, azimuth: float
) -> None:
    # Outgoing quadrature is independent of the visible-normal table builder.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    m = (nodes + 1.0) / 2.0
    phi = 2 * np.pi * np.arange(192) / 192
    cosine, angle = np.meshgrid(m, phi, indexing="ij")
    wi = np.stack(
        (
            np.sqrt(1 - cosine**2) * np.cos(angle),
            np.sqrt(1 - cosine**2) * np.sin(angle),
            cosine,
        ),
        axis=-1,
    )
    wo = np.asarray(
        [np.sqrt(1 - mu**2) * np.cos(azimuth), np.sqrt(1 - mu**2) * np.sin(azimuth), mu]
    )
    n, t = [0, 0, 1], [1, 0, 0]
    single = evaluate_ggx(n, t, wi, wo, alpha_t, alpha_b, _white)
    multiple = evaluate_ggx_compensated(
        n, t, wi, wo, alpha_t, alpha_b, fresnel_average=np.ones(3)
    )
    energy = np.sum(
        (single[..., 0] + multiple[..., 0]) * cosine * weights[:, None] * np.pi / 192
    )
    assert abs(energy - 1.0) < 0.01
    assert energy <= 1.005


def test_compensation_reciprocal_pi_periodic_and_backface() -> None:
    n = [0, 0, 1]
    t = [1, 0, 0]
    a = np.asarray([0.4, 0.2, np.sqrt(0.8)])
    b = np.asarray([-0.3, 0.5, np.sqrt(0.66)])
    forward = evaluate_ggx_compensated(n, t, a, b, 0.7, 0.2)
    reverse = evaluate_ggx_compensated(n, t, b, a, 0.7, 0.2)
    rotated = evaluate_ggx_compensated(n, [-1, 0, 0], a, b, 0.7, 0.2)
    np.testing.assert_allclose(forward, reverse, atol=1e-12)
    np.testing.assert_allclose(forward, rotated, atol=1e-12)
    np.testing.assert_array_equal(
        evaluate_ggx_compensated(n, t, [0, 0, -1], b, 0.7, 0.2), 0
    )


def test_prepared_outgoing_albedo_preserves_brdf() -> None:
    normal = np.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, np.sqrt(0.96)]])
    tangent = np.asarray([[1.0, 0.0, 0.0], [np.sqrt(0.96), 0.0, -0.2]])
    wi = np.asarray([0.3, 0.2, np.sqrt(0.87)])
    wo = np.asarray([-0.2, 0.3, np.sqrt(0.87)])
    at = np.asarray([0.12, 0.7])
    ab = np.asarray([0.35, 0.2])
    zinc_view = directional_albedo(normal, tangent, wo, at, ab)
    np.testing.assert_array_equal(
        evaluate_ggx_compensated(normal, tangent, wi, wo, at, ab),
        evaluate_ggx_compensated(
            normal,
            tangent,
            wi,
            wo,
            at,
            ab,
            outgoing_albedo=zinc_view[0],
            mean_albedo=zinc_view[1],
        ),
    )
    deposit = _Optics(1, "dielectric", np.asarray([0.3, 0.4, 0.5]), 0.6, 0.1, 1.5)
    dielectric_view = schlick_directional_albedo(
        normal, tangent, wo, at, ab, deposit.ior
    )
    np.testing.assert_array_equal(
        _bsdf(normal, tangent, wi, wo, at, ab, deposit),
        _bsdf(normal, tangent, wi, wo, at, ab, deposit, dielectric_view),
    )


def test_anisotropic_table_retains_bitangent_endpoint() -> None:
    mu = 0.3
    sx = np.sqrt(1 - mu * mu)
    along_t, _ = directional_albedo([0, 0, 1], [1, 0, 0], [sx, 0, mu], 0.9, 0.1)
    along_b, _ = directional_albedo([0, 0, 1], [1, 0, 0], [0, sx, mu], 0.9, 0.1)
    swapped, _ = directional_albedo([0, 0, 1], [1, 0, 0], [sx, 0, mu], 0.1, 0.9)
    assert abs(float(along_t - along_b)) > 0.02
    np.testing.assert_allclose(along_b, swapped, atol=0.003)


def test_display_matches_pil_and_linear_is_unclipped() -> None:
    maps = _coupon()
    snapshot = maps["roughness"].copy()
    display = render_material_array(maps)
    image = render_material(maps)
    expected = np.uint8(np.clip(display * 255 + 0.5, 0, 255))
    np.testing.assert_array_equal(np.asarray(image), expected)
    assert image.mode == "RGB"
    np.testing.assert_array_equal(maps["roughness"], snapshot)
    linear = render_material_array(maps, output="linear")
    assert linear.dtype == np.float32
    assert np.all(np.isfinite(linear))


def test_render_rejects_mismatched_recorded_optical_resources() -> None:
    original = _coupon()
    metadata = dict(original.metadata)
    metadata["resource_hashes"] = dict(resource_hashes())
    metadata["renderer_resource_hashes"] = dict(energy_resource_hashes())
    pinned = MaterialMaps(original.arrays, metadata)
    np.testing.assert_array_equal(
        render_material_array(pinned, output="linear"),
        render_material_array(original, output="linear"),
    )
    for field in ("resource_hashes", "renderer_resource_hashes"):
        changed = dict(metadata)
        changed[field] = {"wrong": "sha256"}
        with pytest.raises(ValueError, match=field):
            render_material_array(
                MaterialMaps(original.arrays, changed), output="linear"
            )


def test_lights_move_and_rigs_are_finite() -> None:
    maps = _coupon(mixed=True)
    views = []
    for rig in ("studio", "oblique", "overcast", "grazing"):
        result = render_material_array(
            maps, preview=PreviewConfig(rig=rig, view=(0.0, 0.0, 1.0)), output="linear"
        )
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        views.append(result)
    assert np.max(np.abs(views[0] - views[1])) > 0.01
    left = render_material_array(
        maps,
        preview=PreviewConfig(rig="oblique", light_azimuth_deg=0, view=(0.0, 0.0, 1.0)),
        output="linear",
    )
    right = render_material_array(
        maps,
        preview=PreviewConfig(
            rig="oblique", light_azimuth_deg=180, view=(0.0, 0.0, 1.0)
        ),
        output="linear",
    )
    # Flat isotropic coupon is azimuth invariant; a rich tilted coupon below
    # verifies light movement. This check catches accidental seed variation.
    np.testing.assert_allclose(left, right, atol=1e-5)


def test_rich_lobes_preserve_material_slope_association() -> None:
    first = _coupon(mixed=True, rich=True)
    swapped = _coupon(mixed=True, rich=True, swap=True)
    # Both coupons share every marginal map and lobe weight. Only the pairing
    # between optical identity and slope sign differs.
    assert all(np.array_equal(first[name], swapped[name]) for name in first)
    a = render_material_array(
        first,
        preview=PreviewConfig(rig="oblique", light_azimuth_deg=0, view=(0.0, 0.0, 1.0)),
        output="linear",
    )
    b = render_material_array(
        swapped,
        preview=PreviewConfig(rig="oblique", light_azimuth_deg=0, view=(0.0, 0.0, 1.0)),
        output="linear",
    )
    assert np.max(np.abs(a - b)) > 0.01
    opposite = render_material_array(
        first,
        preview=PreviewConfig(
            rig="oblique", light_azimuth_deg=180, view=(0.0, 0.0, 1.0)
        ),
        output="linear",
    )
    assert np.max(np.abs(a - opposite)) > 0.01


def test_rich_multichunk_strided_memmap_matches_full_conversion(tmp_path: Path) -> None:
    width, height = 50, 32
    pixels = width * height
    template = _coupon(mixed=True, rich=True)
    assert template.lobes is not None
    arrays = {
        name: np.resize(value, (height, width, *value.shape[2:]))
        for name, value in template.arrays.items()
    }
    records = np.empty(pixels * 3, dtype=LOBE_DTYPE)
    records[0::3] = template.lobes[0]
    records[1::3] = template.lobes[0]
    records[2::3] = template.lobes[1]
    records["pixel_index"] = np.repeat(np.arange(pixels, dtype=np.uint32), 3)
    records["weight"] = np.tile([0.25, 0.25, 0.5], pixels)
    assert (
        len(records) > 4096
        and records[4095]["pixel_index"] == records[4096]["pixel_index"]
    )

    path = tmp_path / "rich-lobes.npy"
    disk = np.lib.format.open_memmap(
        path, mode="w+", dtype=LOBE_DTYPE, shape=(len(records) * 2,)
    )
    disk[::2] = records
    disk.flush()
    del disk
    read_only = np.lib.format.open_memmap(path, mode="r")
    strided = read_only[::2]
    maps = MaterialMaps(
        arrays,
        {
            "representation": "rich",
            "optical_parameters": template.metadata["optical_parameters"],
        },
        strided,
    )
    assert maps.lobes is not None
    assert np.shares_memory(maps.lobes, read_only)
    assert maps.lobes.strides[0] == 2 * LOBE_DTYPE.itemsize

    optics = _parameters(maps.metadata)
    full_conversion = _render_lobes(
        records["pixel_index"].astype(np.int64),
        records["weight"].astype(np.float64),
        records["material_id"].astype(np.int64),
        records["optical_parameter_index"].astype(np.int64),
        records["normal_ts"].astype(np.float64),
        records["tangent_ts"].astype(np.float64),
        records["alpha_t"].astype(np.float64),
        records["alpha_b"].astype(np.float64),
        optics,
        (height, width),
        PreviewConfig(),
    )
    np.testing.assert_array_equal(
        render_material_array(maps, output="linear"), full_conversion
    )


def test_reference_samples_shared_state_deterministically() -> None:
    state = build_state(seed=4, size=(4, 4))
    reports = [{}, {}, {}]
    low = render_reference(
        state, (3, 3), samples_per_axis=2, derivative_diagnostics=reports[0]
    )
    medium = render_reference(
        state, (3, 3), samples_per_axis=4, derivative_diagnostics=reports[1]
    )
    high = render_reference(
        state, (3, 3), samples_per_axis=8, derivative_diagnostics=reports[2]
    )
    assert low.shape == (3, 3, 3)
    assert low.dtype == np.float32
    assert np.all(np.isfinite(high))
    assert np.mean(np.abs(medium - high)) < np.mean(np.abs(low - medium))
    np.testing.assert_array_equal(
        medium, render_reference(state, (3, 3), samples_per_axis=4)
    )
    assert (
        reports[0]["initial_step_mm"]
        == reports[1]["initial_step_mm"]
        == reports[2]["initial_step_mm"]
    )
    assert [report["point_count"] for report in reports] == [36, 144, 576]
    assert all(report["maximum_halvings"] == 8 for report in reports)
    assert all(report["max_halvings"] <= 8 for report in reports)


def test_physical_reference_derivatives_on_rectangular_analytic_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lx, ly = 3.0, 5.0
    state = SimpleNamespace(config=SimpleNamespace(size_mm=(lx, ly)))

    def analytic_points(
        _state: object,
        x: np.ndarray,
        y: np.ndarray,
        *,
        footprint_mm: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        assert footprint_mm == (0.0, 0.0)
        return {
            "height_um": 200.0 * x
            - 350.0 * y
            + 20.0 * np.sin(2 * np.pi * x / lx)
            + 30.0 * np.cos(2 * np.pi * y / ly)
        }

    monkeypatch.setattr(galvanised, "sample_points", analytic_points)
    x = np.asarray([[0.3, 0.8], [1.4, 2.2]])
    y = np.asarray([[0.6, 1.1], [2.0, 3.7]])
    step = (0.002, 0.006)
    normal, report = _converged_reference_normals(state, x, y, step)
    slope_x = 1e-3 * (200.0 + 20.0 * 2 * np.pi / lx * np.cos(2 * np.pi * x / lx))
    slope_y = 1e-3 * (-350.0 - 30.0 * 2 * np.pi / ly * np.sin(2 * np.pi * y / ly))
    expected = np.stack((-slope_x, slope_y, np.ones_like(slope_x)), axis=-1)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(normal, expected, atol=1e-7, rtol=0)
    coarse_error = np.max(
        np.abs(_physical_point_normal(state, x, y, (0.02, 0.06)) - expected)
    )
    fine_error = np.max(
        np.abs(_physical_point_normal(state, x, y, (0.01, 0.03)) - expected)
    )
    assert fine_error < coarse_error / 3.5
    assert report["max_halvings"] >= 1
    np.testing.assert_allclose(
        _physical_point_normal(state, x + lx, y + ly, step),
        _physical_point_normal(state, x, y, step),
        atol=1e-12,
        rtol=0,
    )
    left, _ = _converged_reference_normals(state, x[:, :1], y[:, :1], step)
    right, _ = _converged_reference_normals(state, x[:, 1:], y[:, 1:], step)
    np.testing.assert_array_equal(np.concatenate((left, right), axis=1), normal)


def test_reference_derivative_step_is_state_based_and_validated() -> None:
    state = build_state(seed=4, size=(4, 4))
    first = _reference_derivative_step_mm(state, None)
    assert first[0] == first[1] and 0 < first[0] <= 0.001
    segments = state.dendrites.segments
    spectrum = state.micro_spectrum
    lx, ly = state.config.size_mm
    shortest_micro = 1.0 / np.max(np.hypot(spectrum[:, 0] / lx, spectrum[:, 1] / ly))
    assert first[0] == min(
        0.001,
        0.45 * np.min(segments["width_mm"]) / 32,
        np.min(segments["period_mm"]) / 64,
        shortest_micro / 64,
    )
    assert _reference_derivative_step_mm(state, (0.002, 0.003)) == (0.002, 0.003)
    for invalid in (
        (0.0, 0.1),
        (float("nan"), 0.1),
        (-1.0, 0.1),
        (0.1,),
        (True, False),
    ):
        with pytest.raises(ValueError, match="derivative_step_mm"):
            _reference_derivative_step_mm(state, invalid)  # type: ignore[arg-type]


def test_reference_derivative_fails_when_unresolved_or_below_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cubic_points(
        _state: object,
        x: np.ndarray,
        y: np.ndarray,
        *,
        footprint_mm: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        # At x=0 the central-difference slope error falls by four per halving,
        # but this curvature remains unresolved after the full eight-step cap.
        return {"height_um": 1e7 * x**3 + np.zeros_like(y)}

    monkeypatch.setattr(galvanised, "sample_points", cubic_points)
    with pytest.raises(
        material_render.ReferenceDerivativeConvergenceError, match="after 8 halvings"
    ) as caught:
        _converged_reference_normals(
            object(), np.asarray([0.0]), np.asarray([0.0]), (0.01, 0.01)
        )
    assert caught.value.unresolved_points == 1
    assert caught.value.max_component_difference > 1e-5
    with pytest.raises(ValueError, match="coordinate precision"):
        _converged_reference_normals(
            object(), np.asarray([1.0]), np.asarray([1.0]), (1e-20, 1e-20)
        )


def test_reference_point_batch_changes_no_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = build_state(seed=13, size=(7, 5))
    preview = PreviewConfig(rig="overcast")
    original = render_reference(state, (7, 5), preview=preview, samples_per_axis=2)
    monkeypatch.setattr(material_render, "_REFERENCE_POINT_CHUNK", 6)
    tiled = render_reference(state, (7, 5), preview=preview, samples_per_axis=2)
    np.testing.assert_array_equal(tiled, original)


def test_reference_preserves_zinc_deposit_slope_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optical = _coupon(mixed=True).metadata["optical_parameters"]
    monkeypatch.setattr(galvanised, "optical_parameters", lambda _state: optical)

    def paired_points(
        state: SimpleNamespace,
        x: np.ndarray,
        y: np.ndarray,
        *,
        footprint_mm: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        shape = np.broadcast_shapes(np.shape(x), np.shape(y))
        x = np.broadcast_to(x, shape)
        height = 150.0 * np.sin(2 * np.pi * x)
        zinc = (np.cos(2 * np.pi * x) > 0).astype(np.float64)
        if state.swap:
            zinc = 1.0 - zinc
        return {
            "height_um": height,
            "anisotropy_axis": np.broadcast_to([1.0, 0.0], (*shape, 2)),
            "intrinsic_roughness": np.full(shape, 0.3),
            "intrinsic_anisotropy": np.full(shape, 0.5),
            "metallic": zinc,
            "patina_coverage": 1.0 - zinc,
            "white_stain_coverage": np.zeros(shape),
        }

    monkeypatch.setattr(galvanised, "sample_points", paired_points)
    preview = PreviewConfig(rig="oblique", light_azimuth_deg=0, view=(0.0, 0.0, 1.0))
    first = SimpleNamespace(config=SimpleNamespace(size_mm=(1.0, 1.0)), swap=False)
    swapped = SimpleNamespace(config=SimpleNamespace(size_mm=(1.0, 1.0)), swap=True)
    a = render_reference(
        first,
        (1, 1),
        preview=preview,
        samples_per_axis=4,
        derivative_step_mm=(0.001, 0.001),
    )
    b = render_reference(
        swapped,
        (1, 1),
        preview=preview,
        samples_per_axis=4,
        derivative_step_mm=(0.001, 0.001),
    )
    assert np.max(np.abs(a - b)) > 0.01


def test_cached_environment_matches_independent_tilted_integral() -> None:
    n = np.asarray([0.3, 0.2, 0.9327379])
    n /= np.linalg.norm(n)
    t = np.asarray([1.0, 0.0, 0.0])
    t -= np.dot(t, n) * n
    t /= np.linalg.norm(t)
    b = np.cross(n, t)
    view = np.asarray([0.0, 0.0, 1.0])
    optics = _Optics(0, "conductor", np.zeros(3), 0.45, 0.4, 1.5)
    nodes, weights = np.polynomial.legendre.leggauss(192)
    mu = (nodes + 1.0) / 2.0
    phi = 2 * np.pi * np.arange(384) / 384
    cosine, angle = np.meshgrid(mu, phi, indexing="ij")
    local = np.stack(
        (
            np.sqrt(1 - cosine**2) * np.cos(angle),
            np.sqrt(1 - cosine**2) * np.sin(angle),
            cosine,
        ),
        axis=-1,
    )
    direction = (
        local[..., 0, None] * t + local[..., 1, None] * b + local[..., 2, None] * n
    )
    weight = cosine * weights[:, None] * np.pi / 384
    brdf = _bsdf(n, t, direction, view, np.asarray(0.5), np.asarray(0.2), optics)
    expected = np.sum(
        brdf * weight[..., None] * (0.6 + 0.4 * direction[..., 2, None]), axis=(0, 1)
    )
    cached = 0.6 * _constant_environment_response(
        n, t, view, np.asarray(0.5), np.asarray(0.2), optics
    )
    cached += 0.4 * _overcast_directional_response(
        n, t, view, np.asarray(0.5), np.asarray(0.2), optics
    )
    np.testing.assert_allclose(cached, expected, atol=0.003, rtol=0)


def test_dielectric_white_furnace_at_sharp_grazing() -> None:
    # Higher quadrature is essential here: 96x192 spuriously estimates 1.099.
    nodes, weights = np.polynomial.legendre.leggauss(384)
    mu = (nodes + 1.0) / 2.0
    phi = 2 * np.pi * np.arange(768) / 768
    cosine, angle = np.meshgrid(mu, phi, indexing="ij")
    incoming = np.stack(
        (
            np.sqrt(1 - cosine**2) * np.cos(angle),
            np.sqrt(1 - cosine**2) * np.sin(angle),
            cosine,
        ),
        axis=-1,
    )
    view = np.asarray([np.sqrt(1 - 0.05**2), 0.0, 0.05])
    optics = _Optics(1, "dielectric", np.ones(3), 0.3, 0.0, 1.5)
    brdf = _bsdf(
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 0.0, 0.0]),
        incoming,
        view,
        np.asarray(0.1),
        np.asarray(0.1),
        optics,
    )
    energy = np.sum(brdf[..., 0] * cosine * weights[:, None] * np.pi / 768)
    assert abs(energy - 1.0) < 0.01
    assert energy <= 1.005


def test_height_only_compact_maps_derive_physical_normals() -> None:
    width, height = 7, 5
    x = np.arange(width, dtype=np.float64)[None, :]
    elevation = (
        np.broadcast_to(50 * np.sin(2 * np.pi * x / width), (height, width))
        .astype(np.float32)
        .copy()
    )
    normal = height_to_normal_physical(elevation, (7.0, 5.0))
    metadata = {"representation": "single_lobe", "size_mm": [7.0, 5.0]}
    compact = MaterialMaps({"height_um": elevation}, metadata)
    explicit = MaterialMaps({"height_um": elevation, "normal_ts": normal}, metadata)
    np.testing.assert_allclose(
        render_material_array(compact, output="linear"),
        render_material_array(explicit, output="linear"),
        atol=1e-6,
    )
