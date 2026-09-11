"""Regenerate a browsable set of sample textures in one command.

The visual-iteration loop for this library is: tweak an algorithm, re-render
lots of outputs, eyeball them. This module makes that one fast command::

    texture-gen samples                  # every pair x 3 seeds
    texture-gen samples --only wood      # just wood while tuning
    texture-gen samples --count 6 --size 384

(``python -m texture_generators samples ...`` is equivalent.)

Each run writes into a fresh timestamped directory under
:func:`default_outdir` — ``$TEXTURE_GEN_SAMPLES_DIR`` if set to a non-empty
value, else ``/tmp/texture-gen-samples/`` where ``/tmp`` exists, else that name
under the system temp dir, so by default never inside the repo — containing the
PNGs, a contact sheet and an ``index.html`` gallery, and repoints a ``latest``
symlink at it (best effort). Seeds are fixed (``base_seed + i``) so successive runs render
the *same* textures — comparing run directories before and after an algorithm
change shows exactly what the change did.

Pair rows can miss parameters that the random draw does not pick, such as
quartersawn oak with its prominent ray fleck. Each :data:`SWEEPS` entry adds
a row that walks one parameter across its
values at a single fixed seed, which is also the only way to *read* a parameter:
the cells differ by that parameter and nothing else.
"""

from __future__ import annotations

import datetime
import html
import os
import re
import tempfile
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import MATERIALS, all_pairs, generate, sample_sheet


def default_outdir() -> Path:
    """Where ``samples`` runs go by default: a temp location, not the repo.

    A non-empty ``$TEXTURE_GEN_SAMPLES_DIR`` wins (it is used as given, so it
    can point anywhere the caller chooses); otherwise ``/tmp`` where it exists
    (the documented ``/tmp/texture-gen-samples/latest`` habit keeps working on
    macOS and Linux); otherwise the platform temp dir.
    """
    override = os.environ.get("TEXTURE_GEN_SAMPLES_DIR")
    if override:
        return Path(override)
    tmp = Path("/tmp")
    base = tmp if tmp.is_dir() else Path(tempfile.gettempdir())
    return base / "texture-gen-samples"


DEFAULT_OUTDIR = default_outdir()

# A job: material, variant, seed, size, path, extra params for the material.
Job = tuple[str, str, int, int | tuple[int, int], Path, Mapping[str, object]]


@dataclass(frozen=True)
class Sweep:
    """One gallery row varying ``param`` across ``values`` at a fixed seed.

    ``fixed`` pins the other parameters that would otherwise be drawn, so the
    cells differ only by ``param``; ``seed`` defaults to the run's base seed.
    ``values`` are checked against the material module's constant for ``param``
    (``species`` -> ``SPECIES``, ``cut`` -> ``CUTS``, ...; override with
    ``valid_attr``) before anything renders, because a cell that is quietly
    skipped is worse than a crash. ``slug`` names the PNGs and defaults to
    ``sweep-<material>-<param>``.
    """

    title: str
    material: str
    variant: str
    param: str
    values: Sequence[str]
    fixed: Mapping[str, object] = field(default_factory=dict)
    seed: int | None = None
    valid_attr: str | None = None
    slug: str | None = None

    @property
    def prefix(self) -> str:
        """Filename stem shared by this sweep's cells."""
        return self.slug or f"sweep-{_slug(self.material)}-{_slug(self.param)}"

    def filename(self, value: str) -> str:
        """PNG name for one cell — distinct from ``material-variant-seed.png``."""
        return f"{self.prefix}-{_slug(value)}.png"

    def params(self, value: str) -> dict[str, object]:
        """Material params for one cell: the fixed pins plus this value."""
        return {**self.fixed, self.param: value}

    def validate(self) -> None:
        """Raise ValueError unless the material accepts every value named."""
        if self.material not in MATERIALS:
            names = ", ".join(sorted(MATERIALS))
            raise ValueError(
                f"sweep {self.title!r}: unknown material {self.material!r}; "
                f"choose from: {names}"
            )
        mod = MATERIALS[self.material]
        if self.variant not in mod.VARIANTS:
            names = ", ".join(mod.VARIANTS)
            raise ValueError(
                f"sweep {self.title!r}: unknown {self.material} variant "
                f"{self.variant!r}; choose from: {names}"
            )
        if not self.values:
            raise ValueError(f"sweep {self.title!r}: no values to sweep")
        accepted = _accepted_values(mod, self.param, self.valid_attr)
        if accepted is None:
            looked = ", ".join(_attr_candidates(self.param, self.valid_attr))
            raise ValueError(
                f"sweep {self.title!r}: {self.material} declares no values for "
                f"{self.param!r} (looked for {looked}); set valid_attr"
            )
        for value, where in [(v, self.param) for v in self.values] + [
            (v, k) for k, v in self.fixed.items()
        ]:
            valid = (
                accepted if where == self.param else _accepted_values(mod, where, None)
            )
            if valid is not None and value not in valid:
                names = ", ".join(sorted(str(v) for v in valid))
                raise ValueError(
                    f"sweep {self.title!r}: unknown {self.material} {where} "
                    f"{value!r}; choose from: {names}"
                )


# Four wood rows, all at the run's base seed (0 by default). The pins are the
# point of each row: ``cut`` sits on oak because ray fleck is oak's feature and
# only quartersawn shows it; ``figure`` sits on maple because that is what curl
# is drawn on; ``species`` sits on one cut so the eight cells differ only by
# species.
SWEEPS: list[Sweep] = [
    Sweep(
        "wood species",
        "wood",
        "board",
        "species",
        ["pine", "maple", "ash", "oak", "cherry", "walnut", "sapele", "mahogany"],
        fixed={"cut": "flatsawn"},
    ),
    Sweep(
        "wood cut",
        "wood",
        "board",
        "cut",
        ["cathedral", "flatsawn", "quartersawn"],
        fixed={"species": "oak"},
    ),
    Sweep(
        "wood figure",
        "wood",
        "board",
        "figure",
        ["plain", "curly", "ribbon"],
        fixed={"species": "maple"},
    ),
    Sweep(
        "wood finish",
        "wood",
        "board",
        "finish",
        ["none", "oil", "polyurethane", "acrylic"],
        fixed={"species": "walnut"},
    ),
]


def _slug(text: str) -> str:
    """Filename-safe form of ``text``."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def _attr_candidates(param: str, valid_attr: str | None) -> list[str]:
    """Module constants that might hold ``param``'s accepted values."""
    if valid_attr:
        return [valid_attr]
    upper = param.upper()
    # ``species`` -> SPECIES, ``cut`` -> CUTS, ``finish`` -> FINISHES.
    return [upper, upper + "S", upper + "ES"]


def _accepted_values(mod, param: str, valid_attr: str | None) -> Collection[str] | None:
    """The material module's declared values for ``param``, or None if it has none."""
    for name in _attr_candidates(param, valid_attr):
        values = getattr(mod, name, None)
        if values is not None:
            return values
    return None


def _render_one(job: Job) -> Path:
    """Render one texture (runs in a worker process)."""
    material, variant, seed, size, path, params = job
    generate(material, size=size, seed=seed, variant=variant, **params).save(path)
    return path


def _parse_only(only: Sequence[str] | None) -> set[tuple[str, str | None]] | None:
    """Resolve ``--only`` specs (``material`` or ``material/variant``)."""
    if not only:
        return None
    wanted: set[tuple[str, str | None]] = set()
    for spec in only:
        material, _, variant_part = spec.partition("/")
        wanted.add((material, variant_part or None))
    valid = set(all_pairs())
    for material, variant in wanted:
        if material not in MATERIALS:
            names = ", ".join(sorted(MATERIALS))
            raise ValueError(
                f"unknown material {material!r} in --only; choose from: {names}"
            )
        if variant is not None and (material, variant) not in valid:
            names = ", ".join(MATERIALS[material].VARIANTS)
            raise ValueError(
                f"unknown variant {variant!r} in --only; {material} variants: {names}"
            )
    return wanted


def _selected(
    wanted: set[tuple[str, str | None]] | None, material: str, variant: str
) -> bool:
    """Whether ``--only`` (already parsed) asks for this material/variant."""
    return wanted is None or (material, None) in wanted or (material, variant) in wanted


def _filter_pairs(wanted: set[tuple[str, str | None]] | None) -> list[tuple[str, str]]:
    """Pairs the run should render."""
    return [(m, v) for m, v in all_pairs() if _selected(wanted, m, v)]


def _filter_sweeps(
    sweeps: Sequence[Sweep], wanted: set[tuple[str, str | None]] | None
) -> list[Sweep]:
    """Validate every sweep, then keep the ones ``--only`` asks for.

    Validation runs over all of them, not just the selection, so a typo in one
    material's sweep still surfaces on a ``--only`` run of another.
    """
    for sweep in sweeps:
        sweep.validate()
    chosen = [s for s in sweeps if _selected(wanted, s.material, s.variant)]
    seen: dict[str, str] = {}
    for sweep in chosen:
        clash = seen.setdefault(sweep.prefix, sweep.title)
        if clash != sweep.title:
            raise ValueError(
                f"sweeps {clash!r} and {sweep.title!r} share the filename prefix "
                f"{sweep.prefix!r}; give one an explicit slug"
            )
    return chosen


def _write_index(
    run_dir: Path,
    rows: list[tuple[str, str, list[tuple[int, str]]]],
    size: int | tuple[int, int],
    base_seed: int,
    sheet_name: str | None,
    sweep_rows: list[tuple[Sweep, int, list[tuple[str, str]]]] | None = None,
) -> None:
    """Write an ``index.html`` gallery for the run."""
    if isinstance(size, (tuple, list)):
        size_text = f"{size[0]}x{size[1]}"
    else:
        size_text = f"{size}x{size}"
    n_seeds = len(rows[0][2]) if rows else 0
    lines = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(run_dir.name)} textures</title>",
        "<style>",
        "body{background:#141416;color:#ddd;font:14px/1.5 system-ui,sans-serif;"
        "margin:24px}",
        "h1{font-size:18px}h2{font-size:15px;margin:26px 0 8px;color:#fff}",
        "p.meta{color:#999;font-size:12px}a{color:#8ab4f8}",
        ".row{display:flex;flex-wrap:wrap;gap:8px}",
        "figure{margin:0}figcaption{font-size:11px;color:#999;text-align:center}",
        ".row img{width:280px;max-width:96vw;height:auto;display:block;"
        "border-radius:4px}",
        "</style></head><body>",
        f"<h1>texture_generators — {html.escape(run_dir.name)}</h1>",
        f'<p class="meta">{size_text} px, {n_seeds} seed(s) per pair from base '
        f"seed {base_seed}. Seeds are fixed, so the same command after an "
        "algorithm change renders the same textures for comparison.</p>",
    ]
    if sweep_rows:
        lines.append(
            f'<p class="meta">{len(sweep_rows)} parameter sweep(s) below the pair '
            "rows: one row per parameter, one cell per value, everything else "
            "held fixed.</p>"
        )
    if sheet_name:
        lines.append(f'<p><a href="{html.escape(sheet_name)}">contact sheet</a></p>')
    for material, variant, entries in rows:
        lines.append(f"<h2>{html.escape(material)}/{html.escape(variant)}</h2>")
        lines.append('<div class="row">')
        for seed, name in entries:
            quoted = html.escape(name)
            lines.append(
                f'<a href="{quoted}"><figure><img src="{quoted}" loading="lazy">'
                f"<figcaption>seed {seed}</figcaption></figure></a>"
            )
        lines.append("</div>")
    for sweep, seed, cells in sweep_rows or []:
        pins = ", ".join(f"{k}={v}" for k, v in sorted(sweep.fixed.items()))
        detail = f"{sweep.material}/{sweep.variant}, seed {seed}"
        lines.append(
            f"<h2>{html.escape(sweep.title)} — "
            f"{html.escape(detail + (', ' + pins if pins else ''))}</h2>"
        )
        lines.append('<div class="row">')
        for value, name in cells:
            quoted = html.escape(name)
            lines.append(
                f'<a href="{quoted}"><figure><img src="{quoted}" loading="lazy">'
                f"<figcaption>{html.escape(value)}</figcaption></figure></a>"
            )
        lines.append("</div>")
    lines.append("</body></html>")
    (run_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def _point_latest(outdir: Path, run_dir: Path) -> None:
    """Repoint ``outdir/latest`` at the newest run (best-effort)."""
    link = outdir / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(run_dir.name, link)
    except OSError:
        pass


def run(
    outdir: Path | str = DEFAULT_OUTDIR,
    size: int | tuple[int, int] = 512,
    count: int = 3,
    base_seed: int = 0,
    only: Sequence[str] | None = None,
    sheet: bool = True,
    workers: int | None = None,
    sweeps: bool = True,
    sweep_defs: Sequence[Sweep] | None = None,
) -> Path:
    """Render ``count`` fixed seeds of each (filtered) pair into a fresh run dir.

    Returns the run directory, which also contains ``index.html`` and (for
    unfiltered runs) ``sheet.png``. ``workers=None`` uses one process per CPU;
    ``workers=1`` renders serially in-process.

    ``sweeps=False`` skips the :data:`SWEEPS` parameter rows; ``sweep_defs``
    replaces them. Sweeps honour ``only`` and share the pairs' process pool.
    """
    wanted = _parse_only(only)
    pairs = _filter_pairs(wanted)
    chosen_sweeps = (
        _filter_sweeps(SWEEPS if sweep_defs is None else sweep_defs, wanted)
        if sweeps
        else []
    )
    outdir = Path(outdir)
    stamp = datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = outdir / stamp
    suffix = 2
    while run_dir.exists():
        run_dir = outdir / f"{stamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    rows: list[tuple[str, str, list[tuple[int, str]]]] = []
    jobs: list[Job] = []
    for material, variant in pairs:
        entries = []
        for i in range(count):
            seed = base_seed + i
            name = f"{material}-{variant}-{seed}.png"
            jobs.append((material, variant, seed, size, run_dir / name, {}))
            entries.append((seed, name))
        rows.append((material, variant, entries))

    sweep_rows: list[tuple[Sweep, int, list[tuple[str, str]]]] = []
    for sweep in chosen_sweeps:
        seed = base_seed if sweep.seed is None else sweep.seed
        cells = []
        for value in sweep.values:
            name = sweep.filename(value)
            jobs.append(
                (
                    sweep.material,
                    sweep.variant,
                    seed,
                    size,
                    run_dir / name,
                    sweep.params(value),
                )
            )
            cells.append((value, name))
        sweep_rows.append((sweep, seed, cells))

    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if workers <= 1:
        for job in jobs:
            _render_one(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_render_one, jobs, chunksize=1))

    sheet_name = None
    if sheet and not only:
        sheet_name = "sheet.png"
        sample_sheet(size=size, seed=base_seed).save(run_dir / sheet_name)

    _write_index(run_dir, rows, size, base_seed, sheet_name, sweep_rows)
    _point_latest(outdir, run_dir)
    return run_dir


if __name__ == "__main__":
    import sys

    from .cli import main

    raise SystemExit(main(["samples", *sys.argv[1:]]))
