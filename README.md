# texture-generator

Generate random material textures with Python: metal, plastic, wood and paper,
as Pillow images or NumPy arrays. Choose a size and seed to create a reproducible
texture, then save it as a PNG. Generation runs locally using NumPy and Pillow,
without models, downloaded assets or network access.

## Installation

Requires Python 3.10 or later.

```bash
pip install texture-generator
```

The distribution is named `texture-generator`; the Python import is
`texture_generators`, and the command-line program is `texture-gen`.
Installation includes NumPy, Pillow and Click (used by the CLI).

## Quickstart

```python
from texture_generators import generate

image = generate("wood", size=(640, 480), seed=42, variant="board")
image.save("wood.png")
print(image.mode, image.size)
# RGB (640, 480)
```

This writes `wood.png` in the current directory. Sizes are `(width, height)`;
an integer such as `size=512` creates a square. Both dimensions must be at least
2 pixels. The same arguments reproduce the same image with the same package,
dependency versions and platform. Omit `seed` for a fresh texture.

For raw pixels or a labelled sheet of every variant:

```python
from texture_generators import generate_array, sample_sheet

pixels = generate_array("metal", size=256, seed=7, variant="brushed")
print(pixels.shape, pixels.dtype)
# (256, 256, 3) float32

sheet = sample_sheet(size=128, seed=7)
sheet.save("materials.png")
```

Arrays contain RGB values in `[0, 1]`. The
[API reference](https://github.com/nmpowell/texture-generator/blob/main/docs/reference.md#python-api)
covers all public functions and material parameters.

## Command line

```bash
texture-gen wood --variant board --size 640x480 --seed 42 -o wood.png
texture-gen list
texture-gen sheet --size 128 --seed 7 -o materials.png
texture-gen wood --help
```

`python -m texture_generators` runs the same CLI. It also supports batches,
JSON reports, shell completion and browser galleries for comparing seeds and
parameters. See the
[CLI reference](https://github.com/nmpowell/texture-generator/blob/main/docs/reference.md#cli).

## Materials

| Material | Variants |
| --- | --- |
| Metal | `brushed`, `radial`, `polished`, `heat_tinted`, `oil_film`, `anodised_titanium`, `engine_turned` |
| Plastic | `glossy`, `matte`, `textured` |
| Wood | `board`, `planks` |
| Paper | `white`, `kraft`, `recycled`, `newsprint`, `laid`, `coated` |

The generators combine noise, material anatomy and lighting. Albedo, height and
roughness share underlying fields so visible features also affect shading.
These are procedural approximations with documented calibration assumptions.
Metal, plastic and wood do not tile seamlessly; paper tiles when creases are
disabled with `creases=0.0`.

Read the
[technical reference](https://github.com/nmpowell/texture-generator/blob/main/docs/reference.md)
for the algorithms, wood and paper models, core modules, performance estimates
and limitations. See the
[changelog](https://github.com/nmpowell/texture-generator/blob/main/CHANGELOG.md)
for changes and
[GitHub issues](https://github.com/nmpowell/texture-generator/issues)
to report a bug.

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/nmpowell/texture-generator.git
cd texture-generator
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`uv sync --locked` installs the project editable and uses the committed lockfile.
Use `uv run ruff format .` to apply formatting. The tests exercise rendering,
determinism, material physics, the CLI and built-package contents. Internal mypy
checks have existing findings and are informational; see the
[development notes](https://github.com/nmpowell/texture-generator/blob/main/docs/reference.md#development).

The version lives in `pyproject.toml`; `texture_generators.__version__` reads
the installed package metadata. The
[release guide](https://github.com/nmpowell/texture-generator/blob/main/docs/releasing.md)
covers artifact checks and Trusted Publishing.

## License

[Apache License 2.0](https://github.com/nmpowell/texture-generator/blob/main/LICENSE).
