"""Public image/map/CLI integration and legacy isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from texture_generators import (
    generate,
    generate_array,
    generate_maps,
    load_material,
    render_material,
    resolve_variant,
    sample_sheet,
    variants,
)
from texture_generators.cli import main
from texture_generators.materials import metal
from texture_generators.materials.galvanised_config import GalvanisedConfig

RECIPE = GalvanisedConfig(size_mm=(12.0, 8.0))


def test_image_and_map_paths_render_identically() -> None:
    maps = generate_maps("metal", (24, 16), 42, "galvanised", galvanised=RECIPE)
    image = generate("metal", (24, 16), 42, "galvanised", galvanised=RECIPE)
    assert image.tobytes() == render_material(maps).tobytes()
    array = generate_array("metal", (24, 16), 42, "galvanised", galvanised=RECIPE)
    assert array.shape == (16, 24, 3)
    assert array.dtype == np.float32
    assert np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1


def test_new_variant_is_explicit_and_defaults_remain_frozen() -> None:
    assert "galvanised" in variants("metal")
    assert "galvanised" not in metal.DEFAULT_VARIANTS
    assert all(resolve_variant("metal", seed) != "galvanised" for seed in range(100))


@pytest.mark.parametrize(
    "material,variant", [("wood", "board"), ("metal", "brushed"), ("metal", None)]
)
@pytest.mark.parametrize("control", ["galvanised", "galvanised_preset"])
def test_recipe_cannot_be_silently_ignored(material, variant, control) -> None:
    value = RECIPE if control == "galvanised" else "regular"
    with pytest.raises(ValueError, match="explicit"):
        generate_array(material, 8, 42, variant, **{control: value})


def test_recipe_cannot_be_broadcast_over_catalogue() -> None:
    with pytest.raises(ValueError, match="explicit"):
        sample_sheet(size=4, seed=1, galvanised=RECIPE)


@pytest.mark.parametrize(
    "params",
    [
        {"film": False},
        {"iridescence": 0.2},
        {"brush_angle": 0.0},
        {"specular": 1.0},
        {"roughness": 0.2},
        {"typo": True},
    ],
)
def test_galvanised_rejects_legacy_and_unknown_controls(params) -> None:
    with pytest.raises(ValueError):
        generate_array("metal", 8, 42, "galvanised", **params)


def test_map_capability_and_routes_are_strict() -> None:
    with pytest.raises(ValueError, match="require"):
        generate_maps("wood", size=8, variant="board")
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_maps(
            "metal", 8, 42, "galvanised", galvanised=RECIPE, galvanised_preset="regular"
        )
    with pytest.raises(TypeError):
        generate_maps("metal", 8, 42, "galvanised", preview={})


@pytest.mark.parametrize(
    "options",
    [
        {"maps": []},
        {"maps": ["height_um", "height_um"]},
        {"maps": ["unknown"]},
        {"maps": [["height_um"]]},
        {"chunk_size": 0},
        {"chunk_size": True},
        {"chunk_size": 1.5},
    ],
)
def test_invalid_sampling_request_precedes_expensive_state_creation(
    monkeypatch, options
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid map request constructed a physical state")

    monkeypatch.setattr(
        "texture_generators.materials.galvanised.build_state", forbidden
    )
    with pytest.raises(ValueError, match=r"maps|chunk_size"):
        generate_maps("metal", 8, 42, "galvanised", **options)


def test_two_pixel_images_and_minimum_map_size() -> None:
    assert generate("metal", (2, 7), 42, "galvanised", galvanised=RECIPE).size == (2, 7)
    with pytest.raises(ValueError, match="at least 3"):
        generate_maps("metal", (2, 7), 42, "galvanised", galvanised=RECIPE)


def test_image_cli_reports_concrete_physical_recipe(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "metal",
            "--variant",
            "galvanised",
            "--size",
            "16x12",
            "--size-mm",
            "12x8",
            "--seed",
            "42",
            "--json",
            "-o",
            str(tmp_path / "image.png"),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)["written"][0]
    assert report["galvanised"]["size_mm"] == [12.0, 8.0]
    assert isinstance(report["galvanised"]["material_key"], int)
    assert report["galvanised"]["preview_rig"] == "studio"
    assert "Warning:" not in result.stdout


def test_map_cli_bundle_report_and_selection(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    result = CliRunner().invoke(
        main,
        [
            "maps",
            "metal",
            "--variant",
            "galvanised",
            "--size",
            "12x8",
            "--size-mm",
            "12x8",
            "--seed",
            "42",
            "--map",
            "height_um",
            "--map",
            "grain_id",
            "--json",
            "--outdir",
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert set(report["maps"]) == {"height_um", "grain_id"}
    assert report["maps"]["height_um"]["units"] == "um"
    assert report["maps"]["grain_id"]["dtype"] in ("uint32", "<u4")
    loaded = load_material(destination)
    assert loaded.size == (12, 8)
    assert loaded["grain_id"].dtype == np.uint32


@pytest.mark.parametrize(
    "arguments",
    [
        ["metal", "--galvanised-preset", "regular"],
        ["metal", "--variant", "brushed", "--size-mm", "100x100"],
        ["metal", "--variant", "galvanised", "--size-mm", "nanx1"],
        [
            "maps",
            "metal",
            "--variant",
            "galvanised",
            "--size",
            "2",
            "--outdir",
            "unused",
        ],
        ["maps", "wood", "--variant", "board", "--outdir", "unused"],
    ],
)
def test_invalid_cli_combinations_are_usage_errors(arguments) -> None:
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output


def test_config_conflicts_and_duplicates_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"roughness":0.2,"roughness":0.3}')
    args = ["metal", "--variant", "galvanised", "--config", str(path)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2 and "duplicate" in result.output
    path.write_text('{"roughness":0.2}')
    result = CliRunner().invoke(main, [*args, "--galvanised-preset", "regular"])
    assert result.exit_code == 2 and "cannot be combined" in result.output
