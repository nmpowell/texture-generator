# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-09-08

### Added

- Public package documentation, complete material parameter reference, and
  GitHub Actions checks with Trusted Publishing.
- Source archive completeness and rebuild checks, plus clean installation
  checks for both release artifacts.
- `resolve_variant(material, seed=None, variant=None)`: the concrete variant
  `generate()` would render for those arguments. For a concrete seed it is the
  variant `generate(material, seed=seed)` will render, so a caller can name a variant
  before rendering it; with `seed=None` it just draws one, each call independent.

### Changed

- The distribution is now named `texture-generator`; the `texture_generators`
  import and `texture-gen` command remain available.
- Use Apache-2.0 licensing, setuptools and a literal version in `pyproject.toml`.
  The public `__version__` reads installed distribution metadata.
- Use Ruff for formatting and keep development dependencies in `uv.lock`.
- Require Python 3.14 or later.

### Fixed

- A `--size` below the generator's 2x2 minimum (`--size 1`, `16x0`, `-4`) is now
  a usage error (exit 2) instead of a `ValueError` traceback.
- `--json` and the default filename now report the variant actually rendered
  when `--variant` is omitted, rather than `null` — the report was unusable for
  reproducing a render. Reproduce with the reported `seed`, leaving `--variant`
  off: an omitted `--variant` is the seed's first draw, so passing the reported
  variant back explicitly skips that draw and renders a different image. The
  rendered bytes are unchanged by this reporting change.
- `sheet --json` reports the seed it drew instead of `null`, so a sheet rendered
  without `--seed` can be reproduced by re-running with that `--seed`.
- An output path Pillow cannot write (no extension, unwritable directory) is now
  a clean `Error: could not write …` and exit 1, not a traceback. Missing parent
  directories are still created.
- `-o` together with `--outdir` is a usage error; `--outdir` used to be silently
  ignored.
- `samples --help` documents its own `--outdir` default
  (`$TEXTURE_GEN_SAMPLES_DIR`, else `/tmp/texture-gen-samples`); it was
  inheriting the other commands' "the current directory", which it never uses.

## [0.2.0] - 2026-09-07

### Added

- `list` subcommand, printing every material and its variants (`--json` for the
  machine-readable form) — the CLI can now be enumerated without reading the
  source or the README.
- `--json` on every command, replacing the plain path lines with a report:
  `{"written": [{"path", "material", "variant", "seed", "width", "height"}, …]}`
  for the material commands and `all`, `"kind": "sheet"` for `sheet`,
  `{"run_dir", "index", "count"}` for `samples`, and `{"materials": [...]}` for
  `list`. `width`/`height` are the saved image's real pixel size.
- `-h` as a synonym for `--help`, `-V` for `--version`, and short forms
  `-s`/`--size` and `-n`/`--count`.
- Per-command `--help` with the material's variants and a worked example, and
  shell completion for bash, zsh and fish (`_TEXTURE_GEN_COMPLETE`), both free
  from Click.

### Changed

- **The CLI is now a command group built on Click** (a new `click>=8.1`
  dependency) rather than a single argparse parser with a positional `TARGET`.
  `texture-gen wood …`, `all`, `sheet` and `samples` are real subcommands, so
  each carries only the options that apply to it and its own help. Every
  command from 0.1.0 keeps working, but **options must now follow their
  subcommand**: `texture-gen wood --size 16`, not `texture-gen --size 16 wood`,
  which 0.1.0's single flat parser accepted. What else changes is that misuse is
  now caught by the parser instead of by hand-written checks — `all -o x.png`,
  `samples -o x.png` and `wood --only wood` are rejected as unknown options,
  and `--count 0` by an integer range.
- `--variant` is validated per material against that material's own variants
  (a `click.Choice`, so it is also completable). The material commands, their
  variants and the choices are all derived from `MATERIALS` at import time, so
  a new material or variant appears in the CLI and its help automatically.
- Usage-error text now comes from Click: an unknown variant reports
  `Invalid value for '--variant': 'nope' is not one of 'board', 'planks'.`
  rather than `choose from: …`. The exit status is still 2.
- Help output is capitalised `Usage:` (Click's convention) rather than
  argparse's `usage:`, and `python -m texture_generators --help` still names the
  program `texture-gen`.

## [0.1.0] - 2026-09-06

### Added

- Packaged as `texture-generators`: a src layout built with hatchling, so the
  generator can be installed and imported rather than run from a checkout.
- `texture-gen` console script, exposing the same interface as
  `python -m texture_generators`.
- `generate_array()`, returning the rendered texture as a float32 `(H, W, 3)`
  numpy array in `[0, 1]` without the PIL conversion.
- `Material` protocol, documenting the contract every material module in
  `MATERIALS` satisfies.
- `__version__` on the package.
- `samples.default_outdir()`, the rule the CLI uses when `--outdir` is omitted.
- `texture-gen --version`.
- `py.typed` marker, so type checkers use the package's own annotations.

### Changed

- `samples` now chooses its default output directory by rule:
  `$TEXTURE_GEN_SAMPLES_DIR` if set to a non-empty value, otherwise
  `/tmp/texture-gen-samples` where
  `/tmp` exists, otherwise the same directory under the system temp directory.
- Tests moved out of the package into a top-level `tests/` directory.
- Invalid material or variant names on the command line now produce a usage
  error (exit status 2) instead of a traceback.

### Removed

- Running from the checkout without installing: install the package (editable
  for development) instead.
