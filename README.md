# texture-generator

Generate random material textures with Python: metal, plastic, wood and paper,
as Pillow images or NumPy arrays. Choose a size and seed to create a reproducible
texture, then save it as a PNG. Generation runs locally using NumPy and Pillow,
without models, downloaded assets or network access.

> ⚠️ Much of this is AI-generated, and not formally reviewed by hand or eye. It's published chiefly for myself: for my own reference, use, and for experimentation with the whole open-source publishing process. I also *use* this code: I dogfood it. It works, for me. I also write tests, and run them to check that it works, and does what it says.

## Installation

Requires Python 3.12 or later. CI covers Python 3.12, 3.13 and 3.14 with locked
dependencies and the lowest compatible NumPy/Pillow wheels and Click release.

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

## Brushing direction

From version 0.4.0, set `brush_angle` for the `brushed` metal variant. Angles are
degrees clockwise in image coordinates: `0` is horizontal, `90` is vertical,
and `45` runs from top left to bottom right. Finite angles wrap at 360 degrees.

```python
image = generate("metal", size=(640, 480), seed=42, variant="brushed", brush_angle=90)
image.save("vertical-brushing.png")
```

The same keyword works with `generate_array()`, and can be combined with a metal
`film` override. It controls grooves, scratches, directional shading, sheen and
glints, plus groove diffraction on unfilmed metal. The light itself keeps its
direction. Omit `brush_angle` or use `None` to retain the random direction and
existing seeded output. An explicit angle requires `material="metal"` and
`variant="brushed"`.

These examples share seed 42 and a size of 384 × 256 pixels:

| 0° — horizontal | 45° | 90° — vertical | 135° |
| --- | --- | --- | --- |
| <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/metal-brush-angle-0.png" alt="Horizontal brushed metal, 0 degrees" width="160" height="107"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/metal-brush-angle-45.png" alt="Diagonal brushed metal, 45 degrees clockwise" width="160" height="107"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/metal-brush-angle-90.png" alt="Vertical brushed metal, 90 degrees" width="160" height="107"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/metal-brush-angle-135.png" alt="Diagonal brushed metal, 135 degrees clockwise" width="160" height="107"> |

## Command line

```bash
texture-gen wood --variant board --size 640x480 --seed 42 -o wood.png
texture-gen list
texture-gen sheet --size 128 --seed 7 -o materials.png
texture-gen wood --help
texture-gen metal --variant brushed --brush-angle 90 --seed 42 -o vertical-brushing.png
```

`python -m texture_generators` runs the same CLI. It also supports batches,
JSON reports, shell completion and browser galleries for comparing seeds and
parameters. See the
[CLI reference](https://github.com/nmpowell/texture-generator/blob/main/docs/reference.md#cli).

`--brush-angle` is available on `metal` and requires `--variant brushed`.
It also works with `--count`; JSON reports include the angle when it is supplied.

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

## Examples

Every variant below is rendered at **384 × 384 pixels with seed 42** and an
explicit variant name. The
[full results gallery](https://github.com/nmpowell/texture-generator/blob/main/examples/results.md)
includes the notebook's outputs and links to each PNG; the
[recipe manifest](https://github.com/nmpowell/texture-generator/blob/main/examples/manifest.json)
records how they were generated.

### Metal

| Brushed | Radial | Polished | Heat tinted |
| --- | --- | --- | --- |
| <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/metal-brushed.png" alt="Brushed metal texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/metal-radial.png" alt="Radial metal texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/metal-polished.png" alt="Polished metal texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/metal-heat_tinted.png" alt="Heat tinted metal texture" width="160" height="160"> |
| **Oil film** | **Anodised titanium** | **Engine turned** |  |
| <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/metal-oil_film.png" alt="Oil film metal texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/metal-anodised_titanium.png" alt="Anodised titanium metal texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/metal-engine_turned.png" alt="Engine turned metal texture" width="160" height="160"> |  |

### Plastic

| Glossy | Matte | Textured |
| --- | --- | --- |
| <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/plastic-glossy.png" alt="Glossy plastic texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/plastic-matte.png" alt="Matte plastic texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/plastic-textured.png" alt="Textured plastic texture" width="160" height="160"> |

### Wood

| Board | Planks |
| --- | --- |
| <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/wood-board.png" alt="Board wood texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/wood-planks.png" alt="Planks wood texture" width="160" height="160"> |

### Paper

| White | Kraft | Recycled |
| --- | --- | --- |
| <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/paper-white.png" alt="White paper texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/paper-kraft.png" alt="Kraft paper texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/paper-recycled.png" alt="Recycled paper texture" width="160" height="160"> |
| **Newsprint** | **Laid** | **Coated** |
| <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/paper-newsprint.png" alt="Newsprint paper texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/paper-laid.png" alt="Laid paper texture" width="160" height="160"> | <img src="https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/variants/paper-coated.png" alt="Coated paper texture" width="160" height="160"> |

### Run the notebook

Open the
[workflow notebook](https://github.com/nmpowell/texture-generator/blob/main/examples/texture_generator_workflow.ipynb)
to work through the importable Python API, save images and arrays as PNGs, and
build a contact sheet. Preview the standalone
[wood image](https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/wood-board.png),
[metal array saved as an image](https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/metal-array.png)
and [contact sheet](https://raw.githubusercontent.com/nmpowell/texture-generator/main/examples/images/contact-sheet.png).

From the repository root, install the optional notebook tools and launch JupyterLab:

```bash
uv sync --group examples
uv run --group examples jupyter lab examples/texture_generator_workflow.ipynb
```

Choose **Run → Run All Cells**. The notebook writes the separate PNGs to
`examples/images/`, records their recipes in `examples/manifest.json`, and
regenerates `examples/results.md`. Notebook tools live in the optional
`examples` dependency group.

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
The `.python-version` file selects Python 3.14 for development; pass
`--python 3.12` or `--python 3.13` to `uv sync` and `uv run` to use an older
supported interpreter. See the [release guide](https://github.com/nmpowell/texture-generator/blob/main/docs/releasing.md) for
commands to check locked and minimum runtime dependencies on all three versions
in separate environments. Minimum checks preserve the locked development tools
and leave `uv.lock` unchanged.
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
