from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from texture_generators.core.material import LOBE_DTYPE, MaterialMaps
from texture_generators.core.physical import (
    height_to_normal_physical,
    pixel_centres,
    validate_map_size,
)
from texture_generators.core.random_fields import (
    derive_key,
    make_rng,
    material_key_from_rng,
    periodic_lattice_uniform,
    topology_key,
    weather_key,
)
from texture_generators.materials import galvanised_config as config_module
from texture_generators.materials.galvanised_config import (
    PRESET_REVISION,
    GalvanisedConfig,
    PreviewConfig,
)


def test_config_resolution_order_and_frozen_contract() -> None:
    minimised = GalvanisedConfig(preset="minimised", spangle_cv=0.31)
    assert minimised.spangle_diameter_mm == 1.5
    assert minimised.spangle_cv == 0.31
    assert "realised equivalent-diameter CV" in " ".join(minimised.warnings)
    assert minimised.preset_version == PRESET_REVISION
    with pytest.raises(FrozenInstanceError):
        minimised.roughness = 0.5  # type: ignore[misc]


def test_resolved_mapping_replay_does_not_reapply_changed_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = GalvanisedConfig(preset="minimised")
    stored = original.to_mapping()
    monkeypatch.setitem(
        config_module._PRESET_DEFAULTS["minimised"],
        "spangle_diameter_mm",
        2.75,
    )
    assert GalvanisedConfig(preset="minimised").spangle_diameter_mm == 2.75
    replayed = GalvanisedConfig.from_resolved_mapping(stored)
    assert replayed == original


def test_config_strict_conversion_resolution_and_experimental_labelling() -> None:
    with pytest.raises(ValueError, match="unknown"):
        GalvanisedConfig.from_mapping({"cooling_rate_K_s": 1.0})
    with pytest.raises(TypeError):
        GalvanisedConfig(spangle_diameter_mm=True)
    with pytest.raises(ValueError):
        GalvanisedConfig(roughness=float("nan"))
    resolved = GalvanisedConfig().resolve(size=(800, 400))
    assert resolved.size_mm == (100.0, 50.0)
    experimental = GalvanisedConfig(preset="batch")
    assert experimental.is_experimental
    assert "experimental" in " ".join(experimental.warnings)


@pytest.mark.parametrize("field", ["roughness", "wetness", "spangle_cv"])
def test_config_rejects_invalid_ranges(field: str) -> None:
    with pytest.raises(ValueError):
        GalvanisedConfig.from_mapping({field: 1.01})


def test_preview_config_is_separate_and_strict() -> None:
    preview = PreviewConfig(
        rig="grazing",
        exposure_stops=-1.5,
        normal_strength=1.25,
        light_azimuth_deg=45.0,
        view=(0.0, 0.6, 0.8),
        render_samples=128,
    )
    assert preview.rig == "grazing"
    assert not hasattr(GalvanisedConfig(), "normal_strength")
    with pytest.raises(ValueError, match="unit vector"):
        PreviewConfig(view=(0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="unknown"):
        PreviewConfig.from_mapping({"exposure": 1.0})


def test_pixel_centres_are_broadcastable_and_use_rectangular_spacing() -> None:
    x, y = pixel_centres((8.0, 3.0), size=(4, 3))
    assert x.shape == (1, 4)
    assert y.shape == (3, 1)
    np.testing.assert_allclose(x, [[1.0, 3.0, 5.0, 7.0]])
    np.testing.assert_allclose(y[:, 0], [0.5, 1.5, 2.5])
    with pytest.raises(ValueError, match="at least 3"):
        validate_map_size((2, 20))


def test_derivative_treats_an_axis_with_fewer_than_three_samples_as_flat() -> None:
    x, y = pixel_centres((4.0, 1.0), size=(4, 1))
    normals = height_to_normal_physical(
        100.0 * x + 500.0 * y, (4.0, 1.0), periodic=False
    )
    expected = np.array([-0.1, 0.0, 1.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(normals, np.broadcast_to(expected, normals.shape))


def test_nonperiodic_rectangular_ramps_have_canonical_signs() -> None:
    width, height = 7, 5
    width_mm, height_mm = 14.0, 5.0
    x, y = pixel_centres((width_mm, height_mm), size=(width, height))
    # dh/du=200 um/mm and dh/dv=-100 um/mm.
    height_um = 200.0 * x - 100.0 * y
    normals = height_to_normal_physical(
        height_um, (width_mm, height_mm), periodic=False
    )
    expected = np.array([-0.2, -0.1, 1.0], dtype=np.float64)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(
        normals, np.broadcast_to(expected, normals.shape), atol=1e-6
    )


def test_periodic_sine_normal_matches_independent_analytic_derivative() -> None:
    width, height = 128, 96
    width_mm, height_mm = 20.0, 12.0
    x, y = pixel_centres((width_mm, height_mm), size=(width, height))
    height_um = 4.0 * np.sin(2.0 * np.pi * x / width_mm) + 3.0 * np.sin(
        4.0 * np.pi * y / height_mm
    )
    normals = height_to_normal_physical(height_um, (width_mm, height_mm))
    # Compare with the central-difference analytic multiplier, not continuum
    # derivatives, so this fixture also checks independent X/Y spacings.
    dx = width_mm / width
    dy = height_mm / height
    dhdu = (
        4.0
        * np.cos(2.0 * np.pi * x / width_mm)
        * np.sin(2.0 * np.pi * dx / width_mm)
        / dx
    )
    dhdv = (
        3.0
        * np.cos(4.0 * np.pi * y / height_mm)
        * np.sin(4.0 * np.pi * dy / height_mm)
        / dy
    )
    expected = np.stack(
        np.broadcast_arrays(-1e-3 * dhdu, 1e-3 * dhdv, np.ones((height, width))),
        axis=-1,
    )
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(normals, expected, atol=1e-6)


def _mixed_material_maps() -> MaterialMaps:
    shape = (3, 4)
    coverages = {
        "metallic": np.full(shape, 0.5, dtype=np.float32),
        "patina_coverage": np.full(shape, 0.3, dtype=np.float32),
        "white_stain_coverage": np.full(shape, 0.2, dtype=np.float32),
    }
    records = np.empty(shape[0] * shape[1] * 3, dtype=LOBE_DTYPE)
    row = 0
    tangent = np.array([2**-0.5, -(2**-0.5), 0.0], dtype=np.float32)
    for pixel in range(shape[0] * shape[1]):
        for material_id, weight in enumerate((0.5, 0.3, 0.2)):
            records[row] = (
                pixel,
                material_id,
                weight,
                (0.0, 0.0, 1.0),
                tangent,
                0.1,
                0.2,
                material_id,
            )
            row += 1
    return MaterialMaps(
        coverages,
        {
            "schema_version": 1,
            "representation": "rich",
            "coordinate_frame": "X=u, Y=Ly-v, +Z=out",
            "warnings": [],
        },
        records,
    )


def test_material_maps_validate_mixture_weights_and_angle_frame() -> None:
    maps = _mixed_material_maps()
    assert maps.size == (4, 3)
    assert maps["metallic"].flags.writeable is False
    assert maps.lobes is not None
    np.testing.assert_allclose(
        maps.lobes["tangent_ts"][0],
        [2**-0.5, -(2**-0.5), 0.0],
        atol=1e-6,
    )


def test_material_maps_reject_bad_per_material_weights_and_metadata() -> None:
    valid = _mixed_material_maps()
    assert valid.lobes is not None
    bad = valid.lobes.copy()
    bad[0]["weight"] = np.float32(0.4)
    bad[1]["weight"] = np.float32(0.4)  # pixel total stays one, identity does not.
    with pytest.raises(ValueError, match="do not match metallic"):
        MaterialMaps(dict(valid.arrays), dict(valid.metadata), bad)
    with pytest.raises(ValueError, match="JSON-safe"):
        MaterialMaps(
            {"roughness": np.zeros((3, 3), dtype=np.float32)},
            {"bad": object()},
        )


def test_material_maps_reject_wrong_units_dtype_and_axis() -> None:
    with pytest.raises(TypeError, match="float32"):
        MaterialMaps(
            {"height_um": np.zeros((3, 3), dtype=np.float64)},
            {},
        )
    axis = np.full((3, 3, 2), 0.25, dtype=np.float32)
    with pytest.raises(ValueError, match="unit axes"):
        MaterialMaps({"anisotropy_axis": axis}, {})
    with pytest.raises(ValueError, match="map_units"):
        MaterialMaps(
            {"height_um": np.zeros((3, 3), dtype=np.float32)},
            {"map_units": {"height_um": "mm"}},
        )


def test_semantic_keys_and_periodic_lattice_are_stable_and_separated() -> None:
    assert derive_key(42, "nuclei", 7) == derive_key(42, "nuclei", 7)
    assert topology_key(42, "nuclei") != weather_key(42, "nuclei")
    x = np.arange(-2, 7)
    field = periodic_lattice_uniform(12, "warp", x, 3, period=(4, 5))
    repeated = periodic_lattice_uniform(12, "warp", x + 4, 8, period=(4, 5))
    np.testing.assert_array_equal(field, repeated)
    assert np.all((field >= 0.0) & (field < 1.0))


def test_material_key_consumes_exactly_one_uint64_draw() -> None:
    expected_rng = make_rng(123)
    expected = int(expected_rng.integers(0, 1 << 64, dtype=np.uint64))
    expected_next = int(expected_rng.integers(0, 1 << 64, dtype=np.uint64))
    actual_rng = make_rng(123)
    assert material_key_from_rng(actual_rng) == expected
    assert material_key_from_rng(actual_rng) == expected_next
