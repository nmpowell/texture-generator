"""State, weather and shared-sample invariants for galvanised surfaces."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from texture_generators.core.dendrites import build_dendrites, evaluate_dendrites
from texture_generators.core.grains import GrainPartition
from texture_generators.core.material import LOBE_DTYPE
from texture_generators.core.physical import height_to_normal_physical
from texture_generators.core.random_fields import make_rng, material_key_from_rng
from texture_generators.materials.galvanised import (
    build_state,
    generate,
    sample_points,
    sample_state,
    surface_resource_hashes,
)
from texture_generators.materials.galvanised_config import GalvanisedConfig


@pytest.fixture(scope="module")
def regular():
    return build_state(seed=42, size=(32, 24))


def test_seed_and_key_routes_are_equivalent(regular):
    key = material_key_from_rng(make_rng(42))
    replay = build_state(material_key=key, seed=42, size=(32, 24))
    assert regular.material_key == key == replay.material_key
    assert regular.seed == 42
    np.testing.assert_array_equal(
        regular.partition.points_mm, replay.partition.points_mm
    )
    np.testing.assert_array_equal(regular.dendrites.segments, replay.dendrites.segments)


def test_periodic_points_and_one_grain_branch_support(regular):
    lx, ly = regular.config.size_mm
    x = np.array([-8.2, 0.1, 19.5, 151.2])
    y = np.array([4.2, -3.1, 70.3, 23.4])
    a = sample_points(regular, x, y)
    b = sample_points(regular, x + 3 * lx, y - 2 * ly)
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=1e-10, rtol=0.0)
    point = np.array([[0.1, 0.1]])
    records = build_dendrites(point, 1.1, 4.0, 91)
    ids = np.zeros(4, dtype=np.uint32)
    xx = np.array([0.01, 0.99, 1.01, -0.01])
    yy = np.full(4, 0.10)
    field = evaluate_dendrites(records, ids, xx, yy, (1.0, 1.0), 1.1)
    np.testing.assert_allclose(field[:2], field[2:], atol=1e-12, rtol=0.0)


def test_weather_nested_monotonic_and_substrate_fixed(regular):
    zero = sample_points(regular, np.array([2.0, 12.0]), np.array([4.0, 16.0]))
    assert np.all(zero["deposit_height_um"] == 0.0)
    assert np.all(zero["metallic"] == 1.0)
    weathered = replace(
        regular,
        config=GalvanisedConfig(preset="wet_storage", size_mm=regular.config.size_mm),
    )
    half = replace(
        weathered,
        config=GalvanisedConfig(
            preset="wet_storage", size_mm=regular.config.size_mm, exposure=0.5
        ),
    )
    a = sample_points(half, np.array([2.0, 12.0]), np.array([4.0, 16.0]))
    b = sample_points(weathered, np.array([2.0, 12.0]), np.array([4.0, 16.0]))
    np.testing.assert_array_equal(a["substrate_height_um"], b["substrate_height_um"])
    np.testing.assert_array_equal(zero["substrate_height_um"], b["substrate_height_um"])
    assert np.all(b["metallic"] <= a["metallic"])
    assert np.all(b["white_stain_coverage"] <= 1.0 - b["metallic"])
    np.testing.assert_allclose(
        b["metallic"] + b["patina_coverage"] + b["white_stain_coverage"], 1.0
    )
    assert np.any(b["deposit_height_um"] > 0)
    assert np.all(b["deposit_height_um"][b["metallic"] == 1.0] == 0.0)
    assert weathered.config.roughness == regular.config.roughness
    assert weathered.config.anisotropy == regular.config.anisotropy


def test_release_weather_presets_have_distinct_coverage_without_repainting_zinc(
    regular,
):
    lx, ly = regular.config.size_mm
    x, y = np.meshgrid(
        (np.arange(48) + 0.5) * lx / 48,
        (np.arange(36) + 0.5) * ly / 36,
    )
    fresh = sample_points(regular, x, y)
    exposed = sample_points(
        replace(
            regular,
            config=GalvanisedConfig(preset="weathered", size_mm=(lx, ly)),
        ),
        x,
        y,
    )
    stored_state = replace(
        regular,
        config=GalvanisedConfig(preset="wet_storage", size_mm=(lx, ly)),
    )
    stored = sample_points(stored_state, x, y)
    shifted = sample_points(stored_state, x + 2 * lx, y - 3 * ly)
    for name in ("metallic", "patina_coverage", "white_stain_coverage", "height_um"):
        np.testing.assert_allclose(stored[name], shifted[name], atol=1e-10, rtol=0)
    # Ordinary weathering must dull most of the sheet; the former sparse
    # moisture mask left it visually indistinguishable from fresh metal.
    assert 0.65 < exposed["patina_coverage"].mean() < 0.95
    assert not np.any(exposed["white_stain_coverage"])
    assert stored["white_stain_coverage"].mean() > 0.25
    assert np.ptp(stored["white_stain_coverage"]) > 0.4
    for name in ("substrate_height_um", "intrinsic_roughness", "intrinsic_anisotropy"):
        np.testing.assert_array_equal(fresh[name], exposed[name])
        np.testing.assert_array_equal(fresh[name], stored[name])
    # Reflections persist between the fine growth arms, rather than outlining
    # every branch against a uniformly isotropic grain interior.
    assert np.median(fresh["intrinsic_anisotropy"]) > 0.35


@pytest.mark.parametrize("exposure", [5.0, 100.0])
def test_saturated_weather_keeps_nonnegative_fractions_and_defined_axes(exposure):
    config = GalvanisedConfig(
        size_mm=(24, 18),
        exposure=exposure,
        wetness=0.9,
        confinement=0.85,
        salt_exposure=0.2,
        white_stain=0.85,
    )
    maps = sample_state(build_state(config, seed=42), (24, 18))
    for name in ("metallic", "patina_coverage", "white_stain_coverage"):
        assert np.all((maps[name] >= 0.0) & (maps[name] <= 1.0))
    total = maps["metallic"] + maps["patina_coverage"] + maps["white_stain_coverage"]
    np.testing.assert_allclose(total, 1.0, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(
        np.linalg.norm(maps["anisotropy_axis"], axis=-1) > 0.5,
        maps["anisotropy"] > 1e-6,
    )


def test_sampling_chunks_selected_maps_and_final_normals(regular):
    selected = (
        "height_um",
        "normal_ts",
        "metallic",
        "patina_coverage",
        "white_stain_coverage",
        "grain_id",
    )
    first = sample_state(regular, (8, 7), maps=selected, chunk_size=3)
    second = sample_state(regular, (8, 7), maps=selected, chunk_size=8)
    assert tuple(first) == selected
    for name in selected:
        np.testing.assert_array_equal(first[name], second[name])
    expected = height_to_normal_physical(first["height_um"], regular.config.size_mm)
    np.testing.assert_array_equal(first["normal_ts"], expected)
    assert first["grain_id"].dtype == np.uint32
    assert first.lobes.dtype == LOBE_DTYPE
    left = np.argsort(
        np.rec.fromarrays(
            (first.lobes["pixel_index"], first.lobes["material_id"]),
            names="pixel,material",
        ),
        kind="stable",
    )
    right = np.argsort(
        np.rec.fromarrays(
            (second.lobes["pixel_index"], second.lobes["material_id"]),
            names="pixel,material",
        ),
        kind="stable",
    )
    np.testing.assert_array_equal(first.lobes[left], second.lobes[right])
    assert np.all(first.lobes["optical_parameter_index"] == first.lobes["material_id"])
    assert first.metadata["approximation"]["fit_error"] is None


def test_rich_conditional_material_identity_and_memmap(tmp_path, regular):
    state = replace(
        regular,
        config=GalvanisedConfig(preset="wet_storage", size_mm=regular.config.size_mm),
    )
    maps = sample_state(
        state,
        (7, 6),
        maps=("height_um", "metallic", "patina_coverage", "white_stain_coverage"),
        output_dir=tmp_path,
        chunk_size=3,
    )
    assert isinstance(maps["height_um"].base, np.memmap)
    assert isinstance(maps.lobes.base, np.memmap)
    assert (tmp_path / "height_um.npy").exists()
    assert (tmp_path / "layers" / "lobes" / "records.npy").exists()
    assert {0, 1, 2} == set(maps.lobes["material_id"])
    for material_id, name in enumerate(
        ("metallic", "patina_coverage", "white_stain_coverage")
    ):
        weight = np.bincount(
            maps.lobes["pixel_index"][maps.lobes["material_id"] == material_id],
            weights=maps.lobes["weight"][maps.lobes["material_id"] == material_id],
            minlength=42,
        )
        np.testing.assert_allclose(weight.reshape(6, 7), maps[name], atol=1e-6)


def test_working_memmaps_are_closed_before_rename_or_unlink(
    tmp_path, regular, monkeypatch
):
    """Exercise Windows' open-mapping restriction on every platform."""
    opened = {}
    real_open = np.lib.format.open_memmap
    real_rename = Path.rename
    real_unlink = Path.unlink

    def tracked_open(filename, *args, **kwargs):
        result = real_open(filename, *args, **kwargs)
        opened[Path(filename)] = result
        return result

    def assert_closed(path):
        if path.name.startswith(".work_"):
            assert path in opened
            assert opened[path]._mmap.closed

    def checked_rename(path, destination):
        assert_closed(path)
        return real_rename(path, destination)

    def checked_unlink(path, *args, **kwargs):
        assert_closed(path)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", tracked_open)
    monkeypatch.setattr(Path, "rename", checked_rename)
    monkeypatch.setattr(Path, "unlink", checked_unlink)
    result = sample_state(
        regular, (8, 7), maps=("height_um",), output_dir=tmp_path, chunk_size=3
    )
    assert np.isfinite(result["height_um"]).all()
    assert np.isfinite(result.lobes["normal_ts"]).all()
    assert not list(tmp_path.rglob(".work_*"))
    assert not list(tmp_path.glob(".work_*"))


def test_cv_changes_realised_diameter_distribution(regular):
    config = GalvanisedConfig(size_mm=regular.config.size_mm, spangle_cv=0.0)
    low = build_state(config, material_key=regular.material_key)
    config = GalvanisedConfig(size_mm=regular.config.size_mm, spangle_cv=0.5)
    high = build_state(config, material_key=regular.material_key)
    np.testing.assert_array_equal(low.partition.points_mm, high.partition.points_mm)
    assert low.population_stats.diameter_cv < high.population_stats.diameter_cv
    assert regular.population_stats.statistically_sufficient


def test_budget_and_dimensions(regular):
    with pytest.raises(ValueError, match="at least 3"):
        sample_state(regular, (2, 8))
    low_budget = replace(
        regular,
        config=GalvanisedConfig(size_mm=regular.config.size_mm, memory_budget_mb=1),
    )
    with pytest.raises(MemoryError):
        sample_state(low_budget, (512, 512), chunk_size=64)
    with pytest.raises(ValueError, match="distinct"):
        sample_state(regular, (4, 4), maps=("height_um", "height_um"))


def test_single_lobe_selection_and_rich_quality_limits(regular):
    single = replace(
        regular,
        config=GalvanisedConfig(
            size_mm=regular.config.size_mm, representation="single_lobe"
        ),
    )
    result = sample_state(single, (9, 8), maps=("height_um",), chunk_size=4)
    assert tuple(result) == ("height_um",)
    assert result.lobes is None
    assert (
        result.metadata["resource_estimates"]["temporary_working_upper_bytes"] < 50_000
    )
    draft = replace(
        regular,
        config=GalvanisedConfig(
            size_mm=regular.config.size_mm,
            quality="draft",
        ),
    )
    draft_maps = sample_state(draft, (9, 8), maps=("height_um",))
    production_maps = sample_state(regular, (9, 8), maps=("height_um",))
    np.testing.assert_array_equal(draft_maps["height_um"], production_maps["height_um"])
    np.testing.assert_array_equal(draft_maps.lobes, production_maps.lobes)
    unsupported = replace(
        regular,
        config=GalvanisedConfig(
            size_mm=regular.config.size_mm,
            quality="reference",
        ),
    )
    with pytest.raises(ValueError, match="not yet supported"):
        sample_state(unsupported, (9, 8), maps=("height_um",))


def test_bounded_lobe_cache_and_resampling_fallback(regular):
    cached = sample_state(regular, (64, 64), chunk_size=16)
    low_budget = replace(
        regular,
        config=GalvanisedConfig(size_mm=regular.config.size_mm, memory_budget_mb=1),
    )
    fallback = sample_state(low_budget, (64, 64), chunk_size=16)
    assert cached.metadata["resource_estimates"]["cached_subpixel_bytes"] > 0
    assert fallback.metadata["resource_estimates"]["cached_subpixel_bytes"] == 0
    for name in cached:
        np.testing.assert_array_equal(cached[name], fallback[name])
    np.testing.assert_array_equal(cached.lobes, fallback.lobes)


def test_rich_keeps_four_shared_subpixel_axes(regular):
    maps = sample_state(
        regular,
        (9, 8),
        maps=("height_um", "metallic", "patina_coverage", "white_stain_coverage"),
        chunk_size=4,
    )
    assert len(maps.lobes) == 4 * 9 * 8
    counts = np.bincount(maps.lobes["pixel_index"], minlength=9 * 8)
    np.testing.assert_array_equal(counts, 4)
    assert np.ptp(maps.lobes["normal_ts"][:, 0]) > 0


def test_explicit_one_grain_has_no_boundary(regular):
    partition = GrainPartition(
        np.array([[0.3, 0.4]]), np.zeros(1), regular.config.size_mm
    )
    state = replace(
        regular,
        partition=partition,
        dendrites=build_dendrites(partition.points_mm, 8.0, 4.0, 91),
    )
    result = sample_points(state, 0.1, 0.2)
    assert np.isinf(result["boundary_distance_mm"])
    assert np.isfinite(result["height_um"])


def test_two_sample_preview_and_route_validation():
    image = generate((2, 5), make_rng(82))
    assert image.shape == (2, 5, 3)
    assert image.dtype == np.float32
    assert np.all(np.isfinite(image))
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate(
            (4, 4),
            make_rng(82),
            galvanised=GalvanisedConfig(),
            galvanised_preset="regular",
        )


def test_packaged_surface_source_hashes(regular):
    hashes = surface_resource_hashes()
    assert set(hashes) == {"morphology.json", "population_calibration.json"}
    sampled = sample_state(regular, (3, 3), maps=("height_um",))
    assert sampled.metadata["surface_resource_hashes"] == hashes
