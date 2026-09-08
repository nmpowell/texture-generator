"""Test suite: rendering, determinism, sizes, error handling, CLI smoke test.

Kept deliberately small (128px and below) so the whole file runs in a couple
of seconds.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from texture_generators import MATERIALS, all_pairs, generate, sample_sheet, samples
from texture_generators.core.noise import gradient_noise
from texture_generators.materials import plastic

SIZE = 128


@pytest.mark.parametrize("material,variant", all_pairs())
def test_every_variant_renders(material: str, variant: str) -> None:
    """Every material x variant renders a non-degenerate RGB image."""
    img = generate(material, size=SIZE, seed=3, variant=variant)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (SIZE, SIZE)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    for channel in range(3):
        assert arr[..., channel].std() > 0.005, f"{material}/{variant} channel flat"


def test_determinism_same_seed() -> None:
    """The same seed produces byte-identical output."""
    for material in MATERIALS:
        first = generate(material, size=64, seed=1234)
        second = generate(material, size=64, seed=1234)
        assert first.tobytes() == second.tobytes(), material


def test_different_seeds_differ() -> None:
    """Different seeds produce different textures."""
    for material in MATERIALS:
        a = generate(material, size=64, seed=1)
        b = generate(material, size=64, seed=2)
        assert a.tobytes() != b.tobytes(), material


def test_variant_none_is_seeded_choice() -> None:
    """variant=None picks a variant from the seeded rng, reproducibly."""
    a = generate("metal", size=64, seed=99)
    b = generate("metal", size=64, seed=99)
    assert a.tobytes() == b.tobytes()


def test_non_square_size_honoured() -> None:
    """A (width, height) tuple is honoured exactly, for every material."""
    for material in MATERIALS:
        img = generate(material, size=(160, 96), seed=7)
        assert img.size == (160, 96), material


def test_unknown_material_raises() -> None:
    """An unknown material raises ValueError listing the valid names."""
    with pytest.raises(ValueError) as excinfo:
        generate("unobtanium", size=32, seed=0)
    message = str(excinfo.value)
    assert "unobtanium" in message
    for name in MATERIALS:
        assert name in message


def test_unknown_variant_raises() -> None:
    """An unknown variant raises ValueError listing the valid variants."""
    with pytest.raises(ValueError) as excinfo:
        generate("wood", size=32, seed=0, variant="chipboard")
    message = str(excinfo.value)
    assert "chipboard" in message
    assert "board" in message


def test_material_module_contract() -> None:
    """Each module returns float32 (H, W, 3) in [0, 1] and lists variants."""
    for name, module in MATERIALS.items():
        assert module.VARIANTS, name
        rng = np.random.default_rng(0)
        out = module.generate((48, 64), rng, module.VARIANTS[0])
        assert out.dtype == np.float32, name
        assert out.shape == (48, 64, 3), name
        assert out.min() >= 0.0 and out.max() <= 1.0, name


def test_paper_is_bright() -> None:
    """White paper keeps a high mean luminance (>= 0.75)."""
    arr = np.asarray(
        generate("paper", size=SIZE, seed=5, variant="white"), dtype=np.float32
    )
    lum = arr @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    assert lum.mean() / 255.0 >= 0.75


def test_sample_sheet() -> None:
    """The contact sheet renders and is larger than a single tile."""
    sheet = sample_sheet(size=48, seed=2)
    assert sheet.mode == "RGB"
    assert sheet.width > 48 and sheet.height > 48


def test_noise_does_not_band_repeat_on_tall_canvas() -> None:
    """Gradient noise must not repeat every width pixels on a tall canvas.

    A unit-periodic lattice would make rows exactly one unit (= width pixels)
    apart identical; the lattice must instead span the full y extent.
    """
    rng = np.random.default_rng(0)
    field = gradient_noise((256, 16), 4.0, rng)  # aspect 16: y spans 16 units
    assert not np.allclose(field[0:16], field[16:32], atol=1e-6)


@pytest.mark.parametrize("material,variant", all_pairs())
@pytest.mark.parametrize("size", [(16, 16), (256, 24), (24, 256)])
def test_extreme_sizes_finite(
    material: str, variant: str, size: tuple[int, int]
) -> None:
    """Tiny and extreme-aspect canvases render finite, correctly sized output."""
    img = generate(material, size=size, seed=11, variant=variant)
    assert img.size == size
    arr = np.asarray(img, dtype=np.float32)
    assert np.isfinite(arr).all(), f"{material}/{variant} at {size}"


def _run_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "texture_generators", *args],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_sheet_honours_wxh(tmp_path: Path) -> None:
    """`sheet --size WxH` uses non-square tiles rather than dropping H."""
    out_a = tmp_path / "sheet-a.png"
    out_b = tmp_path / "sheet-b.png"
    res = _run_cli(
        ["sheet", "--size", "48x32", "--seed", "1", "-o", str(out_a)], tmp_path
    )
    assert res.returncode == 0, res.stderr
    res = _run_cli(
        ["sheet", "--size", "48x48", "--seed", "1", "-o", str(out_b)], tmp_path
    )
    assert res.returncode == 0, res.stderr
    with Image.open(out_a) as a, Image.open(out_b) as b:
        assert a.width == b.width
        assert a.height < b.height


def test_cli_rejects_bad_flag_combinations(tmp_path: Path) -> None:
    """Non-positive --count, and -o with 'all' or --count>1, are errors."""
    res = _run_cli(["wood", "--count", "0"], tmp_path)
    assert res.returncode != 0
    res = _run_cli(["all", "-o", str(tmp_path / "x.png")], tmp_path)
    assert res.returncode != 0
    res = _run_cli(["wood", "--count", "2", "-o", str(tmp_path / "x.png")], tmp_path)
    assert res.returncode != 0
    res = _run_cli(["samples", "-o", str(tmp_path / "x.png")], tmp_path)
    assert res.returncode != 0
    res = _run_cli(["wood", "--only", "wood"], tmp_path)
    assert res.returncode != 0


def test_samples_run_is_deterministic_and_symlinked(tmp_path: Path) -> None:
    """samples.run renders every pair with fixed seeds and repoints latest."""
    kwargs = dict(
        outdir=tmp_path,
        size=32,
        count=1,
        base_seed=0,
        sheet=False,
        workers=1,
        sweeps=False,
    )
    first = samples.run(**kwargs)
    second = samples.run(**kwargs)
    assert first != second
    names = sorted(p.name for p in first.glob("*.png"))
    assert len(names) == len(all_pairs())
    assert (first / "index.html").exists()
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    latest = tmp_path / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == second.resolve()


def test_samples_only_filter_and_seeds(tmp_path: Path) -> None:
    """--only restricts pairs; seeds run base..base+count-1; sheet skipped."""
    run_dir = samples.run(
        outdir=tmp_path,
        size=32,
        count=2,
        base_seed=5,
        only=["wood/board"],
        workers=1,
        sweeps=False,
    )
    assert sorted(p.name for p in run_dir.glob("*.png")) == [
        "wood-board-5.png",
        "wood-board-6.png",
    ]
    assert not (run_dir / "sheet.png").exists()


def test_samples_rejects_unknown_only(tmp_path: Path) -> None:
    """An unknown --only material or variant raises ValueError."""
    with pytest.raises(ValueError):
        samples.run(outdir=tmp_path, size=32, count=1, only=["granite"], workers=1)
    with pytest.raises(ValueError):
        samples.run(
            outdir=tmp_path, size=32, count=1, only=["wood/chipboard"], workers=1
        )


def test_samples_sweeps_render_one_distinct_png_per_value(tmp_path: Path) -> None:
    """Each sweep writes one PNG per value, all different, with a section each."""
    run_dir = samples.run(
        outdir=tmp_path,
        size=48,
        count=1,
        only=["wood/board"],
        sheet=False,
        workers=1,
    )
    index = (run_dir / "index.html").read_text(encoding="utf-8")
    wood_sweeps = [s for s in samples.SWEEPS if s.material == "wood"]
    assert wood_sweeps
    for sweep in wood_sweeps:
        blobs = {}
        for value in sweep.values:
            path = run_dir / sweep.filename(value)
            assert path.exists(), path.name
            blobs[value] = path.read_bytes()
        assert len(set(blobs.values())) == len(sweep.values), sweep.title
        assert f">{sweep.title} —" in index
        for value in sweep.values:
            assert f"<figcaption>{value}</figcaption>" in index
    # Sweep names cannot collide with the pair renders.
    assert (run_dir / "wood-board-0.png").exists()


def test_samples_no_sweeps_omits_them(tmp_path: Path) -> None:
    """sweeps=False (and --no-sweeps) leaves the gallery pair-only."""
    run_dir = samples.run(
        outdir=tmp_path,
        size=32,
        count=1,
        only=["wood/board"],
        sheet=False,
        workers=1,
        sweeps=False,
    )
    assert list(run_dir.glob("sweep-*.png")) == []
    assert "wood cut" not in (run_dir / "index.html").read_text(encoding="utf-8")

    # fmt: off
    res = _run_cli(
        [
            "samples", "--outdir", str(tmp_path), "--size", "32", "--count", "1",
            "--only", "wood/board", "--no-sheet", "--no-sweeps", "--jobs", "1",
        ],
        tmp_path,
    )
    # fmt: on
    assert res.returncode == 0, res.stderr
    assert list(tmp_path.glob("run-*/sweep-*.png")) == []


def test_samples_only_filters_sweeps(tmp_path: Path) -> None:
    """--only keeps the selected pair's sweeps and drops the others'."""
    defs = [
        samples.Sweep("board cut", "wood", "board", "cut", ["flatsawn", "quartersawn"]),
        samples.Sweep(
            "plank cut",
            "wood",
            "planks",
            "cut",
            ["flatsawn"],
            slug="sweep-wood-planks-cut",
        ),
    ]
    kwargs = dict(
        outdir=tmp_path, size=32, count=1, sheet=False, workers=1, sweep_defs=defs
    )
    run_dir = samples.run(only=["wood/board"], **kwargs)
    assert sorted(p.name for p in run_dir.glob("sweep-*.png")) == [
        "sweep-wood-cut-flatsawn.png",
        "sweep-wood-cut-quartersawn.png",
    ]
    other = samples.run(only=["metal/polished"], **kwargs)
    assert list(other.glob("sweep-*.png")) == []


def test_samples_sweep_rejects_unknown_values(tmp_path: Path) -> None:
    """A value (or pin) the material does not accept raises before rendering."""
    bad = samples.Sweep("bad", "wood", "board", "species", ["oak", "unobtainium"])
    with pytest.raises(ValueError, match="unobtainium"):
        bad.validate()
    with pytest.raises(ValueError, match="unobtainium"):
        samples.run(
            outdir=tmp_path, size=32, count=1, workers=1, sweep_defs=[bad], sheet=False
        )
    assert list(tmp_path.glob("run-*/*.png")) == []
    with pytest.raises(ValueError, match="birdseye"):
        samples.Sweep(
            "bad pin",
            "wood",
            "board",
            "cut",
            ["flatsawn"],
            fixed={"figure": "birdseye"},
        ).validate()
    with pytest.raises(ValueError, match="variant"):
        samples.Sweep(
            "bad variant", "wood", "plywood", "species", ["mahogany"]
        ).validate()


def test_cli_samples(tmp_path: Path) -> None:
    """The samples subcommand writes a run dir with PNGs and an index."""
    res = _run_cli(
        [
            "samples",
            "--outdir",
            str(tmp_path),
            "--size",
            "32",
            "--count",
            "1",
            "--only",
            "paper",
            "--no-sheet",
            "--jobs",
            "1",
        ],
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    runs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("run-")]
    assert len(runs) == 1
    assert len(list(runs[0].glob("paper-*.png"))) == len(MATERIALS["paper"].VARIANTS)
    assert (runs[0] / "index.html").exists()
    assert str(runs[0]) in res.stdout


def test_cli_smoke(tmp_path: Path) -> None:
    """The CLI writes the requested PNG and prints its path."""
    out = tmp_path / "cli-wood.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "texture_generators",
            "wood",
            "--size",
            "96x64",
            "--seed",
            "8",
            "-o",
            str(out),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert str(out) in result.stdout
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (96, 64)


def test_plastic_palette_is_100_valid_unique_colours() -> None:
    """The surveyed palette is 100 unique colours as a float32 (100, 3) array."""
    assert len(plastic.PALETTE) == 100
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in plastic.PALETTE)
    # Pin the transcription against accidental edits: the survey's rank-order
    # endpoints (rank 1 and rank 100).
    assert plastic.PALETTE[0] == "#d7d9d6"
    assert plastic.PALETTE[-1] == "#947c66"
    rgb = plastic._PALETTE_RGB
    assert rgb.dtype == np.float32
    assert rgb.shape == (100, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert len({tuple(row) for row in rgb.tolist()}) == 100


def test_plastic_base_colour_prefers_palette() -> None:
    """Most base colours land near a palette entry, not on the HSV tail.

    Jitter (brightness gain + per-channel noise) means exact membership never
    holds, so proximity is measured with an L-inf tolerance; the 85/15 split
    should put well over 60% of draws within it.
    """
    rng = np.random.default_rng(0)
    colours = np.stack([plastic._base_colour(rng) for _ in range(300)])
    assert colours.dtype == np.float32
    dist = np.abs(colours[:, None, :] - plastic._PALETTE_RGB[None, :, :]).max(axis=2)
    near = (dist.min(axis=1) <= 0.08).mean()
    assert near >= 0.6, f"only {near:.0%} of draws near the palette"


def test_plastic_base_colour_deterministic() -> None:
    """Equal-seeded generators give identical base-colour sequences."""
    first = np.random.default_rng(5)
    second = np.random.default_rng(5)
    for _ in range(10):
        assert np.array_equal(plastic._base_colour(first), plastic._base_colour(second))
