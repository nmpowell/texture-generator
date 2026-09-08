"""Test suite for the Click CLI: help text, JSON reports, usage errors.

Every render here is tiny (16px, or 24x16 for the non-square cases) with a
fixed seed, so the whole file runs in a couple of seconds. The material and
variant lists come from the public API rather than being repeated, so a new
material or variant is covered without editing this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner, Result

import texture_generators
from texture_generators import (
    MATERIALS,
    VARIANTS_BY_MATERIAL,
    _pick_variant,
    all_pairs,
    generate,
    resolve_variant,
)
from texture_generators.cli import main

SIZE = "16"
WIDE = "24x16"
COMMANDS = [*MATERIALS, "all", "sheet", "samples", "list"]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _report(result: Result) -> dict:
    """Assert a clean exit and return the parsed --json report."""
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_help_lists_every_command(runner: CliRunner) -> None:
    """Every subcommand is registered on the group and listed under Commands."""
    assert set(COMMANDS) <= set(main.commands)
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    _, _, listing = result.output.partition("Commands:")
    assert listing, result.output
    for name in COMMANDS:
        assert name in listing, result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_each_subcommand_has_help(runner: CliRunner, command: str) -> None:
    """Every subcommand documents itself with a copy-pasteable example."""
    result = runner.invoke(main, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert result.output.strip()
    assert "Example:" in result.output, result.output
    examples = [
        line for line in result.output.splitlines() if "texture-gen" in line.strip()
    ]
    assert examples, result.output


@pytest.mark.parametrize("material", list(MATERIALS))
def test_material_help_has_variant_and_example(
    runner: CliRunner, material: str
) -> None:
    """Material help offers --variant and a copy-pasteable example."""
    result = runner.invoke(main, [material, "--help"])
    assert result.exit_code == 0, result.output
    assert "--variant" in result.output
    examples = [
        line
        for line in result.output.splitlines()
        if "texture-gen" in line and material in line
    ]
    assert examples, result.output


def test_list_human_output(runner: CliRunner) -> None:
    """`list` names every material and every one of its variants."""
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0, result.output
    for name, variants in VARIANTS_BY_MATERIAL.items():
        assert name in result.output
        for variant in variants:
            assert variant in result.output


def test_list_json_output(runner: CliRunner) -> None:
    """`list --json` reports the registry, in registry order."""
    result = runner.invoke(main, ["list", "--json"])
    assert _report(result) == {
        "materials": [
            {"name": name, "variants": variants}
            for name, variants in VARIANTS_BY_MATERIAL.items()
        ]
    }


def test_material_json_render(runner: CliRunner, tmp_path: Path) -> None:
    """A material's --json report describes the file it just wrote."""
    result = runner.invoke(
        main,
        [
            "wood",
            "--size",
            SIZE,
            "--seed",
            "3",
            "--variant",
            "board",
            "--json",
            "--outdir",
            str(tmp_path),
        ],
    )
    written = _report(result)["written"]
    assert len(written) == 1
    entry = written[0]
    assert entry["material"] == "wood"
    assert entry["variant"] == "board"
    assert entry["seed"] == 3
    assert entry["width"] == 16
    assert entry["height"] == 16
    assert Path(entry["path"]).is_file()


def test_non_square_size(runner: CliRunner, tmp_path: Path) -> None:
    """`--size WxH` renders W wide by H tall, not a square."""
    result = runner.invoke(
        main,
        [
            "wood",
            "--size",
            WIDE,
            "--seed",
            "1",
            "--json",
            "--outdir",
            str(tmp_path),
        ],
    )
    entry = _report(result)["written"][0]
    assert entry["width"] == 24
    assert entry["height"] == 16


def test_all_json(runner: CliRunner, tmp_path: Path) -> None:
    """`all` writes one PNG per material/variant pair."""
    result = runner.invoke(
        main,
        ["all", "--size", SIZE, "--seed", "0", "--json", "--outdir", str(tmp_path)],
    )
    written = _report(result)["written"]
    assert len(written) == len(all_pairs())
    for entry in written:
        assert entry["material"] in MATERIALS
        assert entry["variant"] in VARIANTS_BY_MATERIAL[entry["material"]]
        assert Path(entry["path"]).is_file()


def test_sheet_json(runner: CliRunner, tmp_path: Path) -> None:
    """`sheet` reports one contact sheet, sized from its tiles."""
    result = runner.invoke(
        main,
        ["sheet", "--size", WIDE, "--seed", "7", "--json", "--outdir", str(tmp_path)],
    )
    written = _report(result)["written"]
    assert len(written) == 1
    entry = written[0]
    assert entry["kind"] == "sheet"
    assert entry["seed"] == 7
    assert "7" in Path(entry["path"]).name
    assert Path(entry["path"]).is_file()
    assert isinstance(entry["width"], int) and entry["width"] > 0
    assert isinstance(entry["height"], int) and entry["height"] > 0


def test_samples_json(runner: CliRunner, tmp_path: Path) -> None:
    """`samples --json` reports a run directory holding PNGs and an index."""
    result = runner.invoke(
        main,
        [
            "samples",
            "--size",
            SIZE,
            "--count",
            "1",
            "--no-sheet",
            "--no-sweeps",
            "--jobs",
            "1",
            "--outdir",
            str(tmp_path),
            "--json",
        ],
    )
    report = _report(result)
    assert set(report) == {"run_dir", "index", "count"}
    assert report["count"] >= 1
    assert Path(report["run_dir"]).is_dir()
    assert Path(report["index"]).exists()


def test_samples_outdir_help_names_default(runner: CliRunner) -> None:
    """`samples --help` documents its own --outdir default, not the group's."""
    result = runner.invoke(main, ["samples", "--help"])
    assert result.exit_code == 0, result.output
    assert "TEXTURE_GEN_SAMPLES_DIR" in result.output, result.output
    assert "/tmp/texture-gen-samples" in result.output, result.output


def test_sheet_json_seed_is_populated(runner: CliRunner, tmp_path: Path) -> None:
    """`sheet` without --seed still reports the seed it drew, so it can be redone."""
    result = runner.invoke(
        main, ["sheet", "--size", SIZE, "--json", "--outdir", str(tmp_path)]
    )
    entry = _report(result)["written"][0]
    assert isinstance(entry["seed"], int)


def test_sheet_json_seed_reproduces(runner: CliRunner, tmp_path: Path) -> None:
    """The seed `sheet` reports really does re-render that same sheet."""
    result = runner.invoke(
        main, ["sheet", "--size", SIZE, "--json", "--outdir", str(tmp_path)]
    )
    entry = _report(result)["written"][0]
    seed = entry["seed"]
    assert isinstance(seed, int)
    first = Path(entry["path"]).read_bytes()

    again = tmp_path / "again.png"
    result = runner.invoke(
        main, ["sheet", "--size", SIZE, "--seed", str(seed), "-o", str(again)]
    )
    assert result.exit_code == 0, result.output
    assert again.read_bytes() == first


def test_variant_reported_when_omitted(runner: CliRunner, tmp_path: Path) -> None:
    """Without --variant, the report and the filename name the variant rendered."""
    result = runner.invoke(
        main,
        ["wood", "--size", SIZE, "--seed", "1", "--json", "--outdir", str(tmp_path)],
    )
    entry = _report(result)["written"][0]
    assert entry["variant"] in MATERIALS["wood"].VARIANTS
    assert entry["variant"] in Path(entry["path"]).name


@pytest.mark.parametrize("seed", [0, 1, 4, 17])
def test_resolve_variant_names_the_variant_rendered(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """What the CLI reports is what the material module was actually asked for.

    The report is built from ``resolve_variant`` rather than from the render, so
    the two must agree — spy on the material module to see which variant the
    seeded draw inside ``generate`` really selected. The spy also records the
    rng's next value, pinning the variant pick as the *only* draw made before
    the render: that is what keeps reporting the variant byte-neutral.
    """
    module = MATERIALS["wood"]
    original = module.generate
    seen: list[str] = []
    next_draw: list[int] = []

    def spy(
        shape: tuple[int, int],
        rng: np.random.Generator,
        variant: str,
        **params: Any,
    ) -> np.ndarray:
        """Record the variant asked for and the rng's next value, then render."""
        seen.append(variant)
        next_draw.append(int(rng.integers(0, 2**31)))
        return original(shape, rng, variant, **params)

    monkeypatch.setattr(module, "generate", spy)
    generate("wood", 16, seed=seed)
    assert seen == [resolve_variant("wood", seed)]

    expected = np.random.default_rng(seed)
    _pick_variant(module, expected)
    assert next_draw == [int(expected.integers(0, 2**31))]


def test_resolve_variant_api() -> None:
    """The public helper the CLI reports from: seeded, stable, and validating."""
    assert resolve_variant("wood", 3) in MATERIALS["wood"].VARIANTS
    assert resolve_variant("wood", 3) == resolve_variant("wood", 3)
    with pytest.raises(ValueError, match="unknown wood variant"):
        resolve_variant("wood", 3, "sparkly")
    assert "resolve_variant" in texture_generators.__all__


@pytest.mark.parametrize("size", ["1", "16x0", "0x0", "-4"])
def test_invalid_size_is_usage_error(
    runner: CliRunner, tmp_path: Path, size: str
) -> None:
    """A size below the 2x2 floor is a usage error, not a leaked ValueError."""
    result = runner.invoke(main, ["wood", "--size", size, "--outdir", str(tmp_path)])
    assert result.exit_code == 2, result.output
    assert not isinstance(result.exception, (ValueError, OSError)), result.exception


@pytest.mark.parametrize("name", ["noext", "x.zzz"], ids=["no-extension", "unknown"])
def test_bad_output_path_is_clean_error(
    runner: CliRunner, tmp_path: Path, name: str
) -> None:
    """A path Pillow cannot infer a format from exits 1 with a wrapped error."""
    result = runner.invoke(main, ["wood", "--size", SIZE, "-o", str(tmp_path / name)])
    assert result.exit_code == 1, result.output
    assert not isinstance(result.exception, (ValueError, OSError)), result.exception
    assert "Error" in result.output
    assert "could not write" in result.output


def test_missing_parent_directory_is_created(runner: CliRunner, tmp_path: Path) -> None:
    """A -o path under a directory that does not exist yet still writes."""
    nested = tmp_path / "newsub" / "x.png"
    result = runner.invoke(main, ["wood", "--size", SIZE, "-o", str(nested)])
    assert result.exit_code == 0, result.output
    assert nested.is_file()


@pytest.mark.parametrize(
    "args",
    [
        ["all", "--size", SIZE, "--seed", "0", "--outdir", "/dev/null/nope"],
        ["sheet", "--size", SIZE, "--seed", "0", "-o", "/dev/null/x.png"],
    ],
    ids=["all", "sheet"],
)
def test_all_and_sheet_wrap_io_errors(runner: CliRunner, args: list[str]) -> None:
    """`all` and `sheet` wrap an unwritable destination too, not just the
    material commands."""
    result = runner.invoke(main, args)
    assert result.exit_code == 1, result.output
    assert not isinstance(result.exception, (ValueError, OSError)), result.exception
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_out_and_outdir_conflict(runner: CliRunner, tmp_path: Path) -> None:
    """-o and --outdir together is a usage error rather than a silent --outdir."""
    result = runner.invoke(
        main,
        [
            "wood",
            "--size",
            SIZE,
            "-o",
            str(tmp_path / "x.png"),
            "--outdir",
            str(tmp_path / "d"),
        ],
    )
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize(
    "args",
    [
        ["wood", "--count", "0"],
        ["wood", "--count", "2", "-o", "x.png"],
        ["all", "-o", "x.png"],
        ["frobnicate"],
        ["metal", "--variant", "sparkly"],
    ],
    ids=[
        "count-zero",
        "out-with-count",
        "all-has-no-out",
        "unknown-command",
        "variant",
    ],
)
def test_usage_errors(runner: CliRunner, args: list[str]) -> None:
    """Bad invocations are usage errors, not tracebacks or silent successes."""
    result = runner.invoke(main, args)
    assert result.exit_code == 2, result.output


def test_unknown_variant_lists_valid_variants(runner: CliRunner) -> None:
    """An unknown --variant tells you which variants exist."""
    result = runner.invoke(main, ["metal", "--variant", "sparkly"])
    assert result.exit_code == 2
    for variant in MATERIALS["metal"].VARIANTS:
        assert variant in result.output


def test_single_file_output(runner: CliRunner, tmp_path: Path) -> None:
    """`-o` writes exactly that file and prints its path."""
    out = tmp_path / "w.png"
    result = runner.invoke(
        main, ["wood", "--size", SIZE, "--seed", "2", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert str(out) in result.output


def test_cli_is_deterministic(runner: CliRunner, tmp_path: Path) -> None:
    """The same seed renders byte-identical PNGs; a different seed does not."""

    def render(name: str, seed: str) -> bytes:
        path = tmp_path / name
        result = runner.invoke(
            main, ["wood", "--size", SIZE, "--seed", seed, "-o", str(path)]
        )
        assert result.exit_code == 0, result.output
        return path.read_bytes()

    assert render("a1.png", "5") == render("a2.png", "5")
    assert render("a3.png", "6") != render("a1.png", "5")


def test_seed_increments_for_count(runner: CliRunner, tmp_path: Path) -> None:
    """`--count N` walks the seed, so a fixed seed gives a reproducible run."""
    result = runner.invoke(
        main,
        [
            "wood",
            "--size",
            SIZE,
            "--seed",
            "10",
            "--count",
            "3",
            "--json",
            "--outdir",
            str(tmp_path),
        ],
    )
    written = _report(result)["written"]
    assert [entry["seed"] for entry in written] == [10, 11, 12]
    paths = {entry["path"] for entry in written}
    assert len(paths) == 3
    assert all(Path(path).is_file() for path in paths)


def test_module_entry_point(tmp_path: Path) -> None:
    """`python -m texture_generators` is the same CLI as the console script."""
    out = tmp_path / "m.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "texture_generators",
            "wood",
            "--size",
            SIZE,
            "--seed",
            "2",
            "-o",
            str(out),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert out.stat().st_size > 0


def test_shell_completion() -> None:
    """Click's completion hook is wired up under the script's own name."""
    result = subprocess.run(
        [sys.executable, "-m", "texture_generators"],
        env={**os.environ, "_TEXTURE_GEN_COMPLETE": "bash_source"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert "complete" in result.stdout or "_texture_gen" in result.stdout
