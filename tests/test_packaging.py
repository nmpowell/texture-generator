"""Verify the installed package and its distribution contract."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import typing
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from build import ProjectBuilder
from PIL import Image

import texture_generators
from texture_generators import (
    MATERIALS,
    Material,
    all_pairs,
    generate,
    generate_array,
    materials,
    to_image,
)
from texture_generators.samples import default_outdir

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def runtime_files() -> frozenset[str]:
    """Name every source module that must remain usable after installation."""
    source = PROJECT_ROOT / "src"
    modules = source.glob("texture_generators/**/*.py")
    return frozenset(path.relative_to(source).as_posix() for path in modules) | {
        "texture_generators/py.typed"
    }


def test_version_matches_distribution() -> None:
    """Check the public version matches installed distribution metadata."""
    version = texture_generators.__version__
    assert isinstance(version, str)
    assert version
    assert re.match(r"^\d+\.\d+\.\d+", version)
    assert version == importlib.metadata.version("texture-generator")


def test_py_typed_marker_ships() -> None:
    """Check the installed package includes its typing marker."""
    marker = importlib.resources.files("texture_generators") / "py.typed"
    assert marker.is_file()


def test_distribution_metadata_describes_public_project() -> None:
    """Expose the public project's licence and support links to installers."""
    metadata = importlib.metadata.metadata("texture-generator")

    assert metadata["License-Expression"] == "Apache-2.0"
    assert metadata.get_all("License-File") == ["LICENSE"]
    assert set(metadata.get_all("Project-URL", [])) == {
        "Homepage, https://github.com/nmpowell/texture-generator",
        "Issues, https://github.com/nmpowell/texture-generator/issues",
        "Changelog, https://github.com/nmpowell/texture-generator/releases",
    }


def test_all_names_resolve() -> None:
    """Check every public export resolves and the core API is exported."""
    for name in texture_generators.__all__:
        assert hasattr(texture_generators, name), name
    assert {
        "generate",
        "generate_array",
        "to_image",
        "sample_sheet",
        "MATERIALS",
        "Material",
        "__version__",
    } <= set(texture_generators.__all__)


def test_generate_array_contract() -> None:
    """Check generated arrays have the documented shape, type and range."""
    array = generate_array("plastic", size=32, seed=3, variant="matte")
    assert array.dtype == np.float32
    assert array.shape == (32, 32, 3)
    assert np.isfinite(array).all()
    assert ((array >= 0.0) & (array <= 1.0)).all()


@pytest.mark.parametrize(
    ("material", "variant"),
    all_pairs() + [(material, None) for material in MATERIALS],
)
def test_generate_array_matches_generate(material: str, variant: str | None) -> None:
    """Check array conversion and direct rendering produce identical pixels."""
    seed = 5 if variant is None else 9
    array = generate_array(material, size=24, seed=seed, variant=variant)
    image = generate(material, size=24, seed=seed, variant=variant)
    assert to_image(array).tobytes() == image.tobytes(), (material, variant)


def test_material_protocol() -> None:
    """Check registered materials and the exported protocol agree."""
    assert MATERIALS
    for module in MATERIALS.values():
        assert isinstance(module.VARIANTS, Sequence)
        assert not isinstance(module.VARIANTS, (str, bytes))
        assert module.VARIANTS
        assert all(isinstance(variant, str) and variant for variant in module.VARIANTS)
        assert callable(module.generate)

    assert Material is materials.Material
    assert all(isinstance(module, Material) for module in MATERIALS.values())
    assert not isinstance(object(), Material)

    variants_getter = Material.VARIANTS.fget
    generate_getter = Material.generate.fget
    assert variants_getter is not None
    assert generate_getter is not None
    assert typing.get_type_hints(variants_getter)["return"] == Sequence[str]
    assert typing.get_type_hints(generate_getter)["return"] == Callable[..., np.ndarray]


def test_console_script_renders(tmp_path: Path) -> None:
    """Check the installed console script renders outside the project."""
    script = Path(sys.executable).parent / (
        "texture-gen.exe" if os.name == "nt" else "texture-gen"
    )
    if not script.is_file():
        pytest.fail(
            f"Missing console script {script}: the package must be installed into "
            "the test interpreter's environment."
        )
    output = tmp_path / "w.png"
    result = subprocess.run(
        [script, "wood", "--size", "24x16", "--seed", "2", "-o", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(output) in result.stdout
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (24, 16)


def test_console_script_and_module_agree(tmp_path: Path) -> None:
    """Check script and module invocations produce identical PNG files."""
    script = Path(sys.executable).parent / (
        "texture-gen.exe" if os.name == "nt" else "texture-gen"
    )
    if not script.is_file():
        pytest.fail(
            f"Missing console script {script}: the package must be installed into "
            "the test interpreter's environment."
        )
    script_output = tmp_path / "script.png"
    module_output = tmp_path / "module.png"
    for command, output in (
        ([script], script_output),
        ([sys.executable, "-m", "texture_generators"], module_output),
    ):
        result = subprocess.run(
            [*command, "wood", "--size", "24x16", "--seed", "2", "-o", str(output)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert script_output.read_bytes() == module_output.read_bytes()


def test_help_names_prog(tmp_path: Path) -> None:
    """Check module help uses the console script's public name."""
    result = subprocess.run(
        [sys.executable, "-m", "texture_generators", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("Usage: texture-gen")


def test_version_flag(tmp_path: Path) -> None:
    """Check ``--version`` reports the package version under the script name."""
    result = subprocess.run(
        [sys.executable, "-m", "texture_generators", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"texture-gen {texture_generators.__version__}"


def test_bad_variant_is_a_usage_error(tmp_path: Path) -> None:
    """Check an unknown variant is reported as a usage error, not a traceback."""
    result = subprocess.run(
        [sys.executable, "-m", "texture_generators", "wood", "--variant", "nope"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "is not one of" in result.stderr
    for variant in MATERIALS["wood"].VARIANTS:
        assert variant in result.stderr, variant
    assert "Traceback" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_wheel_excludes_tests(tmp_path: Path, runtime_files: frozenset[str]) -> None:
    """Check a wheel built by the declared backend ships the package, not the tests."""
    wheel_path = Path(ProjectBuilder(PROJECT_ROOT).build("wheel", str(tmp_path)))
    assert wheel_path.is_file()

    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        assert "texture_generators/py.typed" in names
        assert "texture_generators/__init__.py" in names
        assert runtime_files <= set(names)
        assert not any(name.startswith("tests/") for name in names)
        entry_points = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(entry_points) == 1, entry_points
        entry_point_text = wheel.read(entry_points[0]).decode("utf-8")
        assert "texture-gen = texture_generators.cli:main" in entry_point_text


def test_source_distribution_rebuilds_complete_wheel(
    tmp_path: Path, runtime_files: frozenset[str]
) -> None:
    """Ship the source and development files needed to rebuild the package."""
    source_archive = Path(ProjectBuilder(PROJECT_ROOT).build("sdist", str(tmp_path)))
    test_files = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "tests").glob("*.py")
    }
    required = (
        {f"src/{name}" for name in runtime_files}
        | test_files
        | {
            "pyproject.toml",
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
            "uv.lock",
            ".python-version",
            "docs/reference.md",
            "docs/releasing.md",
        }
    )

    with tarfile.open(source_archive) as archive:
        names = {name.partition("/")[2] for name in archive.getnames()}
        assert required <= names, sorted(required - names)
        archive.extractall(tmp_path / "unpacked", filter="data")
    source_root = tmp_path / "unpacked" / source_archive.name.removesuffix(".tar.gz")
    wheel_path = ProjectBuilder(source_root).build("wheel", str(tmp_path / "rebuilt"))

    with zipfile.ZipFile(wheel_path) as wheel:
        assert runtime_files <= set(wheel.namelist())


def test_samples_default_outdir_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check sample output honours its override and defaults outside the project."""
    override = tmp_path / "x"
    monkeypatch.setenv("TEXTURE_GEN_SAMPLES_DIR", str(override))
    assert default_outdir() == override

    monkeypatch.delenv("TEXTURE_GEN_SAMPLES_DIR", raising=False)
    default = default_outdir()
    assert default.name == "texture-gen-samples"
    expected_parent = (
        Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    )
    assert default.parent == expected_parent
    assert not default.resolve().is_relative_to(PROJECT_ROOT)


def test_samples_cli_honours_env_override(tmp_path: Path) -> None:
    """Check the samples subcommand writes where the environment points it."""
    outdir = tmp_path / "samples-out"
    env = {**os.environ, "TEXTURE_GEN_SAMPLES_DIR": str(outdir)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "texture_generators",
            "samples",
            "--only",
            "plastic/matte",
            "--count",
            "1",
            "--size",
            "16",
            "--no-sweeps",
            "--jobs",
            "1",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    runs = [p for p in outdir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    assert len(runs) == 1, list(outdir.iterdir())
    assert (runs[0] / "plastic-matte-0.png").is_file()
    assert (runs[0] / "index.html").is_file()
    assert str(runs[0]) in result.stdout
    latest = outdir / "latest"
    if os.name == "nt" and not latest.is_symlink():
        # ``samples`` treats the ``latest`` symlink as best effort, and Windows
        # may refuse to create one without elevated privileges.
        pass
    else:
        assert latest.resolve() == runs[0].resolve()
    assert not any(p.is_dir() and p.name.startswith("run-") for p in tmp_path.iterdir())


def test_samples_default_is_evaluated_at_run_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check the CLI reads the environment when it runs, not when it was imported."""
    from texture_generators.cli import main

    outdir = tmp_path / "late-override"
    monkeypatch.setenv("TEXTURE_GEN_SAMPLES_DIR", str(outdir))
    argv = ["samples", "--only", "plastic/matte", "--count", "1", "--size", "16"]
    # Click's group exits the interpreter itself rather than returning a status.
    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--no-sweeps", "--jobs", "1"])
    assert excinfo.value.code == 0
    runs = [p for p in outdir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    assert len(runs) == 1, list(outdir.iterdir())
    assert str(runs[0]) in capsys.readouterr().out
