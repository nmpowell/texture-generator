"""Independent footprint integrals, alias rejection and quality contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from texture_generators.core.footprint import box_average, midpoint_axis
from texture_generators.core.physical import height_to_normal_physical
from texture_generators.materials import galvanised
from texture_generators.materials.galvanised import build_state, sample_state
from texture_generators.materials.galvanised_config import GalvanisedConfig


@pytest.fixture(scope="module")
def state():
    return build_state(
        GalvanisedConfig(
            size_mm=(8, 6), spangle_diameter_mm=3, representation="single_lobe"
        ),
        seed=42,
    )


def _analytic_fields(state, x, y):
    x, y = np.broadcast_arrays(x, y)
    lx, ly = state.config.size_mm
    signal = 2.0 * np.sin(2 * np.pi * (3 * x / lx + 2 * y / ly) + 0.31)
    scalar = np.ones(x.shape)
    return {
        "height_um": signal,
        "substrate_height_um": signal,
        "deposit_height_um": scalar * 0,
        "boundary_distance_mm": scalar,
        "metallic": scalar * 0.5,
        "patina_coverage": scalar * 0.5,
        "white_stain_coverage": scalar * 0,
        "roughness": scalar * 0.4,
        "intrinsic_roughness": scalar * 0.3,
        "intrinsic_anisotropy": scalar * 0.6,
        "anisotropy": scalar * 0.3,
        "anisotropy_axis": np.stack((scalar, scalar * 0), axis=-1),
        "grain_id": np.zeros(x.shape, dtype=np.uint32),
        "orientation_id": np.zeros(x.shape, dtype=np.uint32),
    }


def test_rectangular_fourier_box_integral_and_global_phase():
    width, height, rate = 7, 5, 64
    x = midpoint_axis(0, width, rate) / width
    y = midpoint_axis(0, height, rate) / height
    phase = 0.43
    values = np.sin(2 * np.pi * (3 * x[None, :] - 2 * y[:, None]) + phase)
    expected = (
        np.sin(
            2
            * np.pi
            * (
                3 * (np.arange(width)[None, :] + 0.5) / width
                - 2 * (np.arange(height)[:, None] + 0.5) / height
            )
            + phase
        )
        * np.sinc(3 / width)
        * np.sinc(2 / height)
    )
    np.testing.assert_allclose(box_average(values, rate), expected, atol=9e-5)
    np.testing.assert_array_equal(
        midpoint_axis(3, 7, rate), midpoint_axis(0, 7, rate)[3 * rate :]
    )


@pytest.mark.parametrize("rate", [0, -1, True, 1.5])
def test_invalid_quadrature(rate):
    with pytest.raises(ValueError, match="positive integer"):
        midpoint_axis(0, 2, rate)
    with pytest.raises(ValueError, match="positive integer"):
        box_average(np.ones((4, 4)), rate)


def test_normal_tiles_retain_exact_global_spacing():
    height = np.zeros((7, 11), dtype=np.float32)
    height[:, 2::3] = 1.0
    size_mm = (0.6140318707407483, 7.0)
    expected = height_to_normal_physical(height, size_mm)
    for chunk in (1, 2, 3, 7, 11):
        normal = np.empty((*height.shape, 3), dtype=np.float32)
        galvanised._normal_tiles(height, normal, size_mm, chunk)
        np.testing.assert_array_equal(normal, expected)


def test_single_lobe_qualities_integrate_one_surface(state, monkeypatch):
    monkeypatch.setattr(galvanised, "_sample_continuous", _analytic_fields)
    images = {}
    for quality, rate in (("draft", 2), ("production", 4), ("reference", 8)):
        setting = replace(
            state,
            config=GalvanisedConfig(
                size_mm=state.config.size_mm,
                representation="single_lobe",
                quality=quality,
            ),
        )
        maps = sample_state(
            setting, (7, 5), maps=("height_um", "normal_ts", "anisotropy")
        )
        images[quality] = maps["height_um"]
        np.testing.assert_array_equal(
            maps["normal_ts"],
            height_to_normal_physical(maps["height_um"], state.config.size_mm),
        )
        np.testing.assert_allclose(maps["anisotropy"], 0.3, atol=1e-7)
        assert maps.metadata["filtering"]["spatial_convergence_checked"] == (rate == 8)
        assert not maps.metadata["filtering"]["finite_band_certified"]
    x = (np.arange(7)[None, :] + 0.5) * 8 / 7
    y = (np.arange(5)[:, None] + 0.5) * 6 / 5
    expected = (
        _analytic_fields(state, x, y)["height_um"] * np.sinc(3 / 7) * np.sinc(2 / 5)
    )
    errors = [
        np.max(np.abs(images[q] - expected))
        for q in ("draft", "production", "reference")
    ]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.002


def test_reference_selection_chunks_and_mapping_agree(state, monkeypatch, tmp_path):
    monkeypatch.setattr(galvanised, "_sample_continuous", _analytic_fields)
    reference = replace(
        state,
        config=GalvanisedConfig(
            size_mm=state.config.size_mm,
            representation="single_lobe",
            quality="reference",
        ),
    )
    first = sample_state(reference, (7, 5), chunk_size=2)
    second = sample_state(
        reference,
        (7, 5),
        maps=("height_um", "normal_ts"),
        chunk_size=7,
        output_dir=tmp_path,
    )
    for name in second:
        np.testing.assert_array_equal(first[name], second[name])
    assert first.metadata["filtering"] == second.metadata["filtering"]
    assert sum(first.metadata["filtering"]["accepted_rate_pixel_counts"].values()) == 35


def test_non_dyadic_check_rejects_false_dyadic_agreement(state, monkeypatch, tmp_path):
    def aliased(state, x, y):
        result = _analytic_fields(state, x, y)
        x, _ = np.broadcast_arrays(x, y)
        # 32 cycles in each output pixel: 8 and 16 midpoint grids both report
        # the same constant although the exact box integral is zero.
        result["height_um"] = np.cos(2 * np.pi * 32 * x / (8 / 3))
        result["substrate_height_um"] = result["height_um"]
        return result

    monkeypatch.setattr(galvanised, "_sample_continuous", aliased)
    reference = replace(
        state,
        config=GalvanisedConfig(
            size_mm=state.config.size_mm,
            representation="single_lobe",
            quality="reference",
        ),
    )
    with pytest.raises(ValueError, match="did not converge"):
        sample_state(
            reference, (3, 3), maps=("height_um", "normal_ts"), output_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_reference_work_and_angular_limits_fail_before_output(state, tmp_path):
    reference = replace(
        state,
        config=GalvanisedConfig(
            size_mm=state.config.size_mm,
            representation="single_lobe",
            quality="reference",
        ),
    )
    with pytest.raises(ValueError, match="limited to 4096"):
        sample_state(reference, (65, 64), output_dir=tmp_path / "too_large")
    rich = replace(
        state,
        config=GalvanisedConfig(size_mm=state.config.size_mm, quality="reference"),
    )
    with pytest.raises(ValueError, match="angular fitter"):
        sample_state(rich, (3, 3), output_dir=tmp_path / "rich")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("quality", ["draft", "production", "reference"])
def test_selected_metallic_keeps_identical_quantised_coverage(
    state, monkeypatch, quality
):
    def mixed(state, x, y):
        result = _analytic_fields(state, x, y)
        for name, value in (
            ("metallic", 0.1),
            ("patina_coverage", 0.2),
            ("white_stain_coverage", 0.7),
        ):
            result[name] = np.full_like(result[name], value)
        return result

    monkeypatch.setattr(galvanised, "_sample_continuous", mixed)
    setting = replace(
        state,
        config=GalvanisedConfig(
            size_mm=state.config.size_mm,
            representation="single_lobe",
            quality=quality,
        ),
    )
    all_maps = sample_state(setting, (7, 5))
    selected = sample_state(setting, (7, 5), maps=("metallic",))
    np.testing.assert_array_equal(selected["metallic"], all_maps["metallic"])
