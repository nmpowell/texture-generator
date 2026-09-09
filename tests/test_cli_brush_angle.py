"""CLI coverage for explicit brushed-metal direction."""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from texture_generators import MATERIALS, generate
from texture_generators.cli import main


def test_brush_angle_render_matches_public_api(tmp_path: Path) -> None:
    """A CLI angle produces the same pixels as the public Python API."""
    output = tmp_path / "brushed.png"

    result = CliRunner().invoke(
        main,
        [
            "metal",
            "--variant",
            "brushed",
            "--brush-angle",
            "90",
            "--size",
            "32",
            "--seed",
            "7",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    with Image.open(output) as actual:
        expected = generate(
            "metal", size=32, seed=7, variant="brushed", brush_angle=90.0
        )
        assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_brush_angle_is_reported_for_every_batch_render(tmp_path: Path) -> None:
    """Batch JSON preserves the user-supplied degrees on every written image."""
    result = CliRunner().invoke(
        main,
        [
            "metal",
            "--variant",
            "brushed",
            "--brush-angle",
            "450.5",
            "--size",
            "32",
            "--seed",
            "11",
            "--count",
            "2",
            "--json",
            "--outdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    first, second = json.loads(result.stdout)["written"]
    assert first["brush_angle"] == 450.5
    assert second["brush_angle"] == 450.5
    assert Path(first["path"]).name == "metal-brushed-11.png"
    assert Path(second["path"]).name == "metal-brushed-12.png"
    with Image.open(first["path"]) as actual_first:
        expected_first = generate(
            "metal", size=32, seed=11, variant="brushed", brush_angle=450.5
        )
        assert np.array_equal(np.asarray(actual_first), np.asarray(expected_first))
    with Image.open(second["path"]) as actual_second:
        expected_second = generate(
            "metal", size=32, seed=12, variant="brushed", brush_angle=450.5
        )
        assert np.array_equal(np.asarray(actual_second), np.asarray(expected_second))


def test_omitted_brush_angle_keeps_existing_json_shape(tmp_path: Path) -> None:
    """Without the option, JSON records and filenames retain their old shape."""
    result = CliRunner().invoke(
        main,
        [
            "metal",
            "--variant",
            "brushed",
            "--size",
            "16",
            "--seed",
            "5",
            "--json",
            "--outdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    entry = json.loads(result.stdout)["written"][0]
    assert set(entry) == {
        "path",
        "material",
        "variant",
        "seed",
        "width",
        "height",
    }
    assert Path(entry["path"]).name == "metal-brushed-5.png"


@pytest.mark.parametrize(
    "variant_args",
    [[], ["--variant", "polished"]],
    ids=["missing", "other-metal-variant"],
)
def test_brush_angle_requires_explicit_brushed_variant(
    tmp_path: Path, variant_args: list[str]
) -> None:
    """An angle without explicit brushed metal is a clean usage error."""
    result = CliRunner().invoke(
        main,
        [
            "metal",
            *variant_args,
            "--brush-angle",
            "90",
            "--size",
            "16",
            "--outdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2, result.output
    assert "requires explicit --variant brushed" in result.output


@pytest.mark.parametrize("angle", ["nan", "inf", "-inf"])
def test_brush_angle_must_be_finite(angle: str) -> None:
    """Non-finite angle spellings are rejected as Click usage errors."""
    result = CliRunner().invoke(
        main,
        ["metal", "--variant", "brushed", "--brush-angle", angle],
    )

    assert result.exit_code == 2, result.output
    assert "finite" in result.output


def test_metal_help_describes_brush_angle_contract() -> None:
    """Metal help documents units, orientation, finiteness, and precondition."""
    result = CliRunner().invoke(main, ["metal", "--help"])
    help_text = " ".join(result.output.split())

    assert result.exit_code == 0, result.output
    assert "--brush-angle FLOAT" in help_text
    assert "finite degrees clockwise" in help_text
    assert "0 horizontal" in help_text
    assert "90 vertical" in help_text
    assert "requires explicit --variant brushed" in help_text


@pytest.mark.parametrize(
    "command",
    [*[name for name in MATERIALS if name != "metal"], "all", "sheet", "samples"],
)
def test_brush_angle_is_unavailable_outside_metal(command: str) -> None:
    """No other material or aggregate command accepts the metal-only flag."""
    result = CliRunner().invoke(main, [command, "--brush-angle", "90"])

    assert result.exit_code == 2, result.output
    assert "No such option" in result.output
    assert "--brush-angle" in result.output
