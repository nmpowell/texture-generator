"""Command-line interface for the texture generator, built on Click.

``texture-gen`` is a command group: one subcommand per material, plus ``all``,
``sheet``, ``samples`` and ``list``. Examples::

    texture-gen wood --variant board --size 1024x512 --seed 42 -o wood.png
    texture-gen metal --variant brushed --size 512 -n 5 --outdir /tmp/m
    texture-gen all --size 256 --outdir /tmp/textures
    texture-gen sheet --size 256 -o sheet.png
    texture-gen samples --only wood --count 6
    texture-gen list --json

``python -m texture_generators ...`` is equivalent to ``texture-gen ...``;
``texture-gen --version`` prints the package version, and an unknown material,
variant or flag is a usage error (exit status 2).

The materials, their variants and the ``--variant`` choices are all derived
from :data:`texture_generators.MATERIALS` at import time, so a new material or
variant appears in the CLI — and in its help — without editing this module.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

import click
from PIL import Image

from . import (
    MATERIALS,
    __version__,
    all_pairs,
    generate,
    resolve_variant,
    sample_sheet,
    samples,
)


class SizeType(click.ParamType):
    """A pixel size: ``N`` for a square texture, or ``WxH`` (also ``W*H``)."""

    name = "size"

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> int | tuple[int, int]:
        """Parse ``N`` or ``WxH``, rejecting any dimension below the 2x2 floor.

        The generator itself raises on a smaller size, so checking here turns a
        degenerate ``--size`` into a usage error rather than a traceback.
        """
        parsed = self._parse(value, param, ctx)
        dimensions = parsed if isinstance(parsed, tuple) else (parsed,)
        if any(dimension < 2 for dimension in dimensions):
            self.fail(f"size must be at least 2x2, got {value!r}", param, ctx)
        return parsed

    def _parse(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> int | tuple[int, int]:
        """Parse ``N`` or ``WxH`` into an int or a ``(width, height)`` tuple."""
        if isinstance(value, (int, tuple)):  # already-converted default
            return value
        cleaned = str(value).strip().lower().replace("*", "x")
        if "x" in cleaned:
            parts = cleaned.split("x")
            if len(parts) != 2:
                self.fail(f"bad size {value!r}; use N or WxH", param, ctx)
            try:
                return (int(parts[0]), int(parts[1]))
            except ValueError:
                self.fail(f"bad size {value!r}; use N or WxH", param, ctx)
        try:
            return int(cleaned)
        except ValueError:
            self.fail(f"bad size {value!r}; use N or WxH", param, ctx)


SIZE = SizeType()

size_option = click.option(
    "-s",
    "--size",
    type=SIZE,
    default=512,
    show_default=True,
    metavar="SIZE",
    help="pixel size: N for a square, or WxH.",
)
seed_option = click.option(
    "--seed",
    type=int,
    default=None,
    metavar="INTEGER",
    help="integer seed; the same seed always renders the same texture. "
    "Omit for fresh entropy.",
)
outdir_option = click.option(
    "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="PATH",
    help="output directory (default: the current directory).",
)
out_option = click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    metavar="PATH",
    help="output image path; the format follows the file extension (PNG by default).",
)
json_option = click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="emit a JSON report on stdout instead of plain path lines.",
)


def _default_name(material: str, variant: str | None, seed: int | None) -> str:
    """Build ``<material>[-<variant>][-<seed>].png``."""
    parts = [material]
    if variant:
        parts.append(variant)
    if seed is not None:
        parts.append(str(seed))
    return "-".join(parts) + ".png"


def _resolve_path(out: Path | None, outdir: Path | None, name: str) -> Path:
    """Choose the output path: ``out`` verbatim, else ``name`` under ``outdir``."""
    return Path(out) if out else Path(outdir or ".") / name


def _save(img: Image.Image, path: Path) -> None:
    """Write ``img`` to ``path``, creating parents, and wrap any failure.

    Pillow raises ``ValueError`` for a path it cannot infer a format from and
    ``OSError`` for one it cannot write; both are the user's problem, not a bug,
    so they surface as ``Error: …`` and exit 1 rather than as a traceback.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"could not write {path}: {exc}") from exc


def _one_destination(out: Path | None, outdir: Path | None) -> None:
    """Reject ``-o`` together with ``--outdir``: ``--outdir`` would be ignored."""
    if out is not None and outdir is not None:
        raise click.UsageError(
            "use either --out (one file) or --outdir (a directory), not both"
        )


def _record(path: Path, img: Image.Image, **fields: Any) -> dict[str, str | int | None]:
    """One ``written`` entry: the path, the given fields, then the pixel size."""
    width, height = img.size
    return {"path": str(path), **fields, "width": width, "height": height}


def _emit(records: list[dict[str, str | int | None]], as_json: bool) -> None:
    """Report what was written, as JSON or as one path per line."""
    if as_json:
        click.echo(json.dumps({"written": records}, indent=2))
    else:
        for record in records:
            click.echo(record["path"])


def _seed_for(seed: int | None, index: int = 0) -> int:
    """``seed + index`` for a fixed seed, else a fresh random seed."""
    return seed + index if seed is not None else secrets.randbelow(2**31)


MATERIAL_HELP = """Render a {material} texture as a PNG.

Variants: {variants} (default: one picked with the seed).

Writes to --out, or to <material>-<variant>-<seed>.png under --outdir (the
current directory by default), and prints every path written — naming the
variant the seed picked, even when --variant was omitted. With -n/--count the
seeds increment from --seed, so a fixed seed gives a reproducible run.

\b
Example:
  texture-gen {material} --variant {variant} --size 1024x512 --seed 42 -o {material}.png
"""


def _material_command(material: str) -> click.Command:
    """Build the subcommand for one material, from that material's variants."""
    variants = list(MATERIALS[material].VARIANTS)

    @click.command(
        name=material,
        short_help=f"Render a {material} texture.",
        help=MATERIAL_HELP.format(
            material=material,
            variants=", ".join(variants),
            variant=variants[0],
        ),
    )
    @size_option
    @seed_option
    @click.option(
        "--variant",
        type=click.Choice(variants),
        default=None,
        help=f"which {material} variant to render.",
    )
    @click.option(
        "-n",
        "--count",
        type=click.IntRange(min=1),
        default=1,
        show_default=True,
        metavar="COUNT",
        help="render N textures (seeds increment from --seed).",
    )
    @out_option
    @outdir_option
    @json_option
    def command(
        size: int | tuple[int, int],
        seed: int | None,
        variant: str | None,
        count: int,
        out: Path | None,
        outdir: Path | None,
        as_json: bool,
    ) -> None:
        _one_destination(out, outdir)
        if out is not None and count > 1:
            raise click.UsageError(
                "--out names a single file; use --outdir with --count"
            )
        records = []
        for index in range(count):
            seed_i = _seed_for(seed, index)
            # Name the variant before rendering, so an omitted --variant is
            # reported and filed as the variant the seed actually picks.
            resolved = resolve_variant(material, seed_i, variant)
            img = generate(material, size, seed=seed_i, variant=variant)
            path = _resolve_path(out, outdir, _default_name(material, resolved, seed_i))
            _save(img, path)
            records.append(
                _record(path, img, material=material, variant=resolved, seed=seed_i)
            )
        _emit(records, as_json)

    return command


class TextureGenGroup(click.Group):
    """The group, always naming itself ``texture-gen`` in help and errors.

    Click would otherwise take the program name from ``sys.argv[0]``, which
    makes ``python -m texture_generators --help`` advertise a different
    interface from the identical console script.
    """

    def main(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the group, defaulting the program name to the script's."""
        kwargs.setdefault("prog_name", "texture-gen")
        return super().main(*args, **kwargs)


@click.group(
    cls=TextureGenGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(
    __version__,
    "-V",
    "--version",
    prog_name="texture-gen",
    message="%(prog)s %(version)s",
)
def main() -> None:
    """Generate random but realistic procedural material textures (metal,
    plastic, wood, paper) as PNGs.

    Every texture comes from a single seeded generator, so the same seed
    renders the same image; omit --seed for fresh entropy. --size takes N for a
    square or WxH. Run "texture-gen list" to see every material and variant.

    \b
    Example:
      texture-gen wood --variant board --size 1024x512 --seed 42 -o wood.png
      texture-gen metal --size 512 -n 5 --outdir /tmp/metal
      texture-gen all --size 256 --outdir /tmp/textures
      texture-gen samples --only wood --count 6
    """


for _material in MATERIALS:
    main.add_command(_material_command(_material))


@main.command(name="all")
@size_option
@seed_option
@outdir_option
@json_option
def all_command(
    size: int | tuple[int, int],
    seed: int | None,
    outdir: Path | None,
    as_json: bool,
) -> None:
    """Render every material/variant combination, one PNG each.

    Files are named <material>-<variant>-<seed>.png under --outdir (the current
    directory by default). There is no -o/--out: this writes many files.

    \b
    Example:
      texture-gen all --size 256 --seed 42 --outdir /tmp/textures
    """
    records = []
    for material, variant in all_pairs():
        seed_i = _seed_for(seed)
        img = generate(material, size, seed=seed_i, variant=variant)
        path = _resolve_path(
            None, outdir or Path("."), _default_name(material, variant, seed_i)
        )
        _save(img, path)
        records.append(
            _record(path, img, material=material, variant=variant, seed=seed_i)
        )
    _emit(records, as_json)


@main.command(name="sheet")
@size_option
@seed_option
@out_option
@outdir_option
@json_option
def sheet_command(
    size: int | tuple[int, int],
    seed: int | None,
    out: Path | None,
    outdir: Path | None,
    as_json: bool,
) -> None:
    """Render a labelled contact sheet: one tile per material/variant.

    --size is the per-tile size, so WxH gives non-square tiles. Writes to
    --out, or sheet-<seed>.png under --outdir.

    \b
    Example:
      texture-gen sheet --size 256 --seed 42 -o sheet.png
    """
    _one_destination(out, outdir)
    # Resolve the seed up front rather than letting sample_sheet draw one, so
    # the reported seed is the one that rendered this sheet.
    seed_i = _seed_for(seed)
    img = sample_sheet(size=size, seed=seed_i)
    path = _resolve_path(out, outdir, _default_name("sheet", None, seed_i))
    _save(img, path)
    _emit([_record(path, img, kind="sheet", seed=seed_i)], as_json)


@main.command(name="samples")
@size_option
@click.option(
    "--seed",
    type=int,
    default=None,
    metavar="INTEGER",
    help="base seed; each pair renders seeds base..base+count-1 (default 0).",
)
@click.option(
    "-n",
    "--count",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    metavar="COUNT",
    help="seeds to render per material/variant pair.",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="PATH",
    help="run-directory root (default: $TEXTURE_GEN_SAMPLES_DIR, else "
    "/tmp/texture-gen-samples).",
)
@click.option(
    "--only",
    metavar="MAT[/VARIANT]",
    multiple=True,
    help="restrict to a material or material/variant (repeatable).",
)
@click.option("--no-sheet", is_flag=True, help="skip the contact sheet.")
@click.option("--no-sweeps", is_flag=True, help="skip the parameter-sweep rows.")
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=None,
    metavar="INTEGER",
    help="worker processes (default: one per CPU; 1 = serial).",
)
@json_option
def samples_command(
    size: int | tuple[int, int],
    seed: int | None,
    count: int,
    outdir: Path | None,
    only: tuple[str, ...],
    no_sheet: bool,
    no_sweeps: bool,
    jobs: int | None,
    as_json: bool,
) -> None:
    """Render the visual-review set: every pair at fixed seeds, plus a gallery.

    Writes a fresh timestamped run directory (under --outdir, else
    $TEXTURE_GEN_SAMPLES_DIR or /tmp/texture-gen-samples) holding the PNGs, the
    contact sheet, the parameter sweeps and an index.html, and repoints a
    "latest" symlink at it. Seeds are fixed, so the same command before and
    after a change renders the same textures.

    \b
    Example:
      texture-gen samples --only wood/board --count 6 --size 512
    """
    try:
        run_dir = samples.run(
            outdir=Path(outdir) if outdir else samples.default_outdir(),
            size=size,
            count=count,
            base_seed=seed if seed is not None else 0,
            only=list(only) or None,
            sheet=not no_sheet,
            workers=jobs,
            sweeps=not no_sweeps,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    n_images = len(list(run_dir.glob("*.png")))
    index = run_dir / "index.html"
    if as_json:
        click.echo(
            json.dumps(
                {"run_dir": str(run_dir), "index": str(index), "count": n_images},
                indent=2,
            )
        )
    else:
        click.echo(f"{n_images} PNGs -> {run_dir}")
        click.echo(index)


@main.command(name="list")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="emit the registry as JSON instead of one line per material.",
)
def list_command(as_json: bool) -> None:
    """List every material and its variants, in registry order.

    \b
    Example:
      texture-gen list --json
    """
    materials = [
        {"name": name, "variants": list(module.VARIANTS)}
        for name, module in MATERIALS.items()
    ]
    if as_json:
        click.echo(json.dumps({"materials": materials}, indent=2))
    else:
        for entry in materials:
            click.echo(f"{entry['name']}: {', '.join(entry['variants'])}")
