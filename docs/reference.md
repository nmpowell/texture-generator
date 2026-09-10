# API and technical reference

[Back to the README](../README.md) · [Releasing](releasing.md)

Procedural generator for **random but realistic material textures** — metal,
plastic, wood and paper — as PNGs at any pixel size. Pure `numpy` + `Pillow`,
no models, no assets, no network. Same seed → byte-identical image on a given
platform and numpy build.

## Research grounding

The design follows the classic procedural-synthesis literature (Perlin
gradient noise and fBm, Quilez-style domain warping, Worley/cellular noise,
anisotropic streak noise, height-to-normal shading) rather than a neural
texture synthesiser. That choice buys the three properties this library needs:
**determinism** (a seed reproduces a texture exactly, with no model weights to
pin), **resolution independence** (fields are evaluated on whatever grid you
ask for, not upsampled from a fixed tile), and **self-contained randomised
generation** (every seed draws a fresh palette, scale and feature set from
plausible physical ranges instead of resampling a training set). The realism
comes from one discipline applied everywhere: albedo, height and roughness are
all derived from the *same* underlying fields, so a scratch that catches the
light is the same scratch that tints the surface — the coherence real
photographs have and independently-generated maps do not.

For installation and a first render, see the [quickstart](../README.md#installation).

## Python API

```python
from texture_generators import (
    generate,
    generate_array,
    to_image,
    sample_sheet,
    MATERIALS,
    variants,
    all_pairs,
)

img = generate("wood", size=512, seed=42)  # PIL.Image, RGB
img = generate("metal", size=(640, 480), variant="brushed")  # (width, height)
img = generate("plastic", size=256, seed=7, specular=0.9)  # override params
arr = generate_array("wood", size=512, seed=42)  # float32 (H,W,3)
img = to_image(arr)  # identical to generate(...)
sheet = sample_sheet(size=256, seed=7)  # labelled grid

img.save("wood.png")
```

The full public surface, all importable from `texture_generators`:

- `generate(material, size=512, seed=None, variant=None, **params) -> PIL.Image.Image` —
  RGB image at exactly the requested size.
- `generate_array(material, size=512, seed=None, variant=None, **params) -> np.ndarray` —
  the float32 `(H, W, 3)` array in `[0, 1]` that `generate` converts via
  `to_image`. Uses the same rng sequence, so `to_image(generate_array(...))` is
  identical to `generate(...)`.
- `to_image(rgb) -> PIL.Image.Image` — convert a float32 `(H, W, 3)` array to
  an RGB image.
- `sample_sheet(size=256, seed=None, *, columns=None, **params) -> PIL.Image.Image` —
  labelled contact sheet of every material/variant pair.
- `resolve_variant(material, seed=None, variant=None) -> str` — the concrete
  variant `generate` would render for those arguments. For a concrete `seed` it
  is the variant `generate(material, seed=seed)` will render, so a caller can name a
  variant (in a report or a filename) before rendering it; with `seed=None` it
  just draws one such variant, each call independent.
- `MATERIALS: dict[str, Material]` — maps material name to its module.
- `Material` — a `typing.Protocol` with `VARIANTS: Sequence[str]` and
  `generate(shape, rng, variant, **params) -> np.ndarray`.
- `VARIANTS_BY_MATERIAL`, `variants(material)`, `all_pairs()` — enumeration
  helpers.
- `__version__` — the installed version string.

Contracts: `size` is an int (square) or `(width, height)`. `seed=None` draws
fresh entropy; an integer seed is fully reproducible. `variant=None` picks a
variant with the seeded rng. Unknown material or variant raises `ValueError`
listing the valid names. The package ships `py.typed` (PEP 561), so type
checkers use the public API's annotations; mypy over the package's own
internals is not yet clean (see *Development*).

For brushed metal, `generate()` and `generate_array()` accept `brush_angle` in
degrees clockwise in image coordinates: 0 is horizontal, 90 is vertical, and
45 runs from top left to bottom right. A supplied angle requires the explicit
arguments `material="metal"`, `variant="brushed"`; other combinations raise
`ValueError`. `None` or omission retains the random direction. See
[brushing direction](#brushing-direction) for the full contract.

## CLI

`texture-gen` is a [Click](https://click.palletsprojects.com/) command group:
one subcommand per material, plus `all`, `sheet`, `samples` and `list`. It is
installed alongside the package, and `python -m texture_generators` is
identical — down to the program name in `--help`. `-h` is a synonym for
`--help` and `-V` for `--version`, which prints `texture-gen` and the installed version; an unknown
command, variant or flag, and a size below the 2×2 minimum, are usage errors
(exit status 2). Options follow their subcommand: `texture-gen wood --size 16`,
not `texture-gen --size 16 wood`.

```bash
texture-gen --help                  # the group, with every subcommand
texture-gen wood --help             # one command, with its variants
texture-gen list                    # every material and its variants

texture-gen wood --variant board --size 1024x512 --seed 42 -o wood.png
texture-gen metal --variant brushed --size 512 -n 5 --outdir /tmp/metal
texture-gen all --size 256 --outdir /tmp/textures      # every pair
texture-gen sheet --size 256 -o sheet.png             # contact sheet
texture-gen samples --only wood --count 6             # visual-review set
```

| Command | What it renders |
| --- | --- |
| `metal`, `plastic`, `wood`, `paper` | one texture of that material, or `-n N` of them |
| `all` | every material/variant pair, one PNG each |
| `sheet` | the labelled contact sheet |
| `samples` | a visual-review run directory with an `index.html` gallery |
| `list` | the materials and their variants, as lines or JSON |

The material commands are generated from `MATERIALS` at import time, so a new
material or variant becomes a subcommand — and a validated, tab-completable
`--variant` choice — without touching the CLI.

Shared options: `-s`/`--size` takes `N` or `WxH` (default 512; the *per-tile*
size for `sheet`). `--seed` fixes the seed, so the same command renders the
same image; omit it for fresh entropy. `-n`/`--count N` (≥ 1) renders N
textures with seeds incrementing from `--seed`. `-o`/`--out` names a single
file, whose **format follows the file extension** (PNG by default — a path with
no extension is an error, not a silent PNG); `--outdir` names a directory; and
without either, files land in the current directory as
`<material>-<variant>-<seed>.png`, naming the variant the seed picked even when
`--variant` was omitted. Every path written is printed. Because `-o` is one
file, combining it with `--count > 1` is rejected, as is `-o` together with
`--outdir`, and `all` and `samples` have no `-o` at all — they write many files,
so use `--outdir`.

The `metal` command also accepts `--brush-angle FLOAT`, with explicit
`--variant brushed`. The angle applies to every image in a `--count` batch.
Non-finite values, other variants or an omitted variant are usage errors
(exit status 2). Other commands do not accept this option.

```bash
texture-gen metal --variant brushed --brush-angle 90 --size 640x480 --seed 42 -o vertical.png
texture-gen metal --variant brushed --brush-angle 45 --seed 42 --count 3 --outdir diagonal --json
```

### JSON output

Every command takes `--json`, which replaces the plain path lines with a
machine-readable report on stdout, so a script never has to parse the human
output:

```bash
$ texture-gen wood --variant board --size 64 --seed 42 --json
{
  "written": [
    {
      "path": "wood-board-42.png",
      "material": "wood",
      "variant": "board",
      "seed": 42,
      "width": 64,
      "height": 64
    }
  ]
}
```

`width`/`height` are the saved image's real pixel size, and `variant` and `seed`
are always concrete — omit either on the command line and the report still names
what was rendered: the concrete `seed`, and the variant that seed selected. To
reproduce the image exactly, use the reported `--seed` and preserve whether
the original command supplied `--variant`. If it was omitted, leave it off;
if it was supplied, pass the same variant again. An omitted `--variant`
consumes the seed's first random draw. The JSON report does not record whether
the variant was explicit, so keep the original command alongside the report.

When `--brush-angle` is supplied, each `written` entry also contains a numeric
`brush_angle` in degrees. Pass that value back with `--variant brushed` to
reproduce the direction. The key is absent when the option was omitted.

`all` reports the same shape with one entry per pair; `sheet` reports one entry
with `"kind": "sheet"`; `samples` reports `{"run_dir": …, "index": …, "count":
N}`; and `list` reports `{"materials": [{"name": …, "variants": [...]}, …]}`.

### Shell completion

Click generates completion for the subcommands, the options and the
`--variant` values:

```bash
# bash: add to ~/.bashrc
eval "$(_TEXTURE_GEN_COMPLETE=bash_source texture-gen)"
# zsh: add to ~/.zshrc
eval "$(_TEXTURE_GEN_COMPLETE=zsh_source texture-gen)"
# fish
_TEXTURE_GEN_COMPLETE=fish_source texture-gen | source
```

Bash completion requires Bash 4.4 or later; macOS's bundled Bash 3.2 is too old.
With Click 8.1, generating Bash completion also checks the `bash` executable on
`PATH`. Use a supported Bash installation on `PATH`, or use Zsh completion.
The test suite generates Zsh source through the subprocess completion hook;
that generation does not require a Zsh executable or invoke a local shell.

### Regenerating the visual-review set

The `samples` subcommand is the fast eyeball loop for algorithm work:

```bash
texture-gen samples                       # every pair × 3 seeds
texture-gen samples --only wood --count 6 # tuning one material
texture-gen samples --seed 100            # a different fixed set
```

Each run renders in parallel (one process per CPU) into a fresh timestamped directory under `samples.default_outdir()`:
`$TEXTURE_GEN_SAMPLES_DIR` if set to a non-empty value (used as given), else
`/tmp/texture-gen-samples/` where `/tmp` exists (macOS and Linux), else `texture-gen-samples` under the platform temp directory — so
by default never inside the repo. The run directory holds the PNGs, the
contact sheet and an `index.html` gallery, and the run repoints a `latest`
symlink at it (best effort; skipped where symlinks are unavailable):

Open the printed `index.html` path in a browser. On macOS:

```bash
open /tmp/texture-gen-samples/latest/index.html
```

Seeds are **fixed** (`base_seed + i`, base 0 unless `--seed` is given), so
running the same command before and after a change renders the same textures
— compare run directories to see exactly what the change did.
`--only MATERIAL[/VARIANT]` (repeatable) filters the set, `--jobs N` caps the
worker processes (`1` = serial), `--no-sheet` skips the contact sheet (it is
also skipped whenever `--only` filters).

Below the pair rows the gallery has a **parameter sweep** row per entry in
`samples.SWEEPS`: one cell per value of a single parameter, at one fixed seed,
with everything else pinned, captioned by the value. Pair rows alone can only
show a parameter if the random draw happens to pick it. Small random samples
can miss combinations such as quartersawn oak, whose ray fleck is particularly
prominent. The four wood sweeps (species, cut, figure, finish; 18 renders)
show these parameters deliberately. Sweeps are data, so a material adds one by appending a `Sweep`;
values are checked against the material's own constants and a bad one is an
error, not a missing cell. `--only` filters them by material/variant like the
pairs, and `--no-sweeps` skips them.

## Variants

| Material | Variants | One-line algorithm |
| --- | --- | --- |
| `metal` | `brushed` | severely anisotropic fbm streaks in a rotated frame, break-modulated along the brush; dipole scratch grooves (lit lip + shadowed lip) with log-uniform lengths; pass-band roughness (the coarse overlap bands run dimmer/broader via per-pixel specular); conductor-tinted highlight; sheen band perpendicular to the brushing; sparse skew-bright glints gated by the band |
| | `radial` | the same streak recipe in polar coordinates (θ-periodic, seamless) plus a faint Archimedean tool-feed spiral, a centre-fade that kills the pinwheel singularity, a 65%-chance machining witness mark (tight rings + centre-drill dimple), area-uniform heavy-tailed arc scratches, and a two-spoke "bow-tie" sheen at the light azimuth instead of a straight band |
| | `polished` | near-uniform albedo, undulation + smudges (which also *broaden/dim* the local gloss), 2–10 dipole hairlines, 5–30 sub-pixel pits, conductor-tinted specular, then a reflected-room gradient (cool bright half / warm dark half) under a crisp-edged reflected-light streak |
| | `heat_tinted` | an oxide interference film, with a spatial thickness field that produces temper colours over the reflected light |
| | `oil_film` | a transparent oil film whose thickness and angle control interference colour, including desaturation as thicker fringes average spectrally |
| | `anodised_titanium` | a titania interference film over metal, using the same spectral integration with its own optical system and thickness preset |
| | `engine_turned` | overlapping radial swirl marks in per-cell local frames, composited in machining order so each disc cuts a crescent from the previous one |
| `plastic` | `glossy` | flat albedo (±2% mottling) + broad shallow "orange peel", tight specular; gloss is sold by the *reflection*: a flat-topped additive white streak whose edges wobble with the peel field, over a soft multiplicative fill |
| | `matte` | bead-blast Worley craters (not white noise) + weak wide specular, wide soft sheen band |
| | `textured` | two-scale Worley stipple kept separated, low relief, cavity-shadowed hollows — moulded-equipment finish |
| | *all* | ~1/3 of parts get injection-moulding history: curved flow-line gloss bands radiating from a gate, sometimes one weld line |
| `wood` | `board` | a flatsawn board on a real millimetre scale (`mm_across`, 150–300 mm). Three cuts — cathedral (the pith is an *axis dipping below the face*, so ring traces widen into nested crowns), gentle arcs, and rift-like near-straight — over eight species specified in **CIELAB**, with heartwood/sapwood splits, a finish transform, an occasional knot and slow oxidation drift |
| | `planks` | 3–6 strips across the grain, **each its own board**: own pith, ring spacing, sapwood draw, finish magnitude and colour (a Lab offset of ΔL\* ±4, Δa\* ±1.5, Δb\* ±2.5, since identically coloured boards read as printed). Dark gap lines have bevel highlights; knots are rare on narrow strips. Finish class and ageing are shared across the panel |
| | *all* | colour is **specified in CIELAB per species**, not hand-picked RGB (`SPECIES`) — typical published figures rather than measurements of these grades: eight species from walnut at L\* 40 to maple at L\* 81.5, an occasional pale **sapwood** band on cherry/walnut (L\* ~80 against a dark heart), a cherry ageing draw shared across a panel (boards laid together have aged together), and a per-board **finish** (`FINISHES`) applied in Lab as ΔL\* −2…−10 with ΔC\* +2…+12 — deepening *and* saturating, which is what index-matching away the air/cell-wall veil does and what an RGB multiply cannot. The albedo mean is pinned to the species value and the lighting is mean-normalised, so a rendered board measures the colour it was specified as (within 3/255) |
| `paper` | `white` | uncoated woodfree copier stock: short hardwood-rich furnish at ~9 layers of coverage, heavy filler, calender freckling (high spots iron flat and go glossy), and an optical-brightener draw that runs from creamy natural white (b\* +8) through neutral to premium blue-white (b\* −5) |
| | `kraft` | unbleached sack kraft: long coarse softwood fibre (2.5 mm, 31 µm) with no filler to bury it, so the felt is fully exposed; mild machine-direction alignment (κ ≈ 0.35–0.55, *not* hatching), dark shive bundles softened by the pulp they sit in, and the highest Sq of the three |
| | `recycled` | deinked mixed furnish: high fines, and residual ink as **three** populations — a sub-visible clustered darkening field that supplies the grey cast, hundreds of irregular 0.05–0.5 mm specks placed *on the fines*, and a handful of rare coloured fibre-shaped contraries from printed waste |
| | `newsprint` | groundwood newsprint: short coarse mechanical fibre, only ~5 layers of coverage (45 g/m², so the sheet is thin enough to show through), the strongest machine-direction grain of the set, a 250 µm point-spread from the unfilled sheet, and a lignin-ageing draw that runs from fresh (b\* +9.5) to a browned copy several years old |
| | `laid` | mould-made laid paper: a nearly isotropic hand-formed sheet (κ ≈ 0, no machine direction) with the **wire marks in the height field** — 0.95–1.30 mm laid lines at 6.5–10 µm and 20–30 mm chain lines at 16.5–25 µm, both as narrowed cosine troughs modulated by formation, with a proud lip beside each chain wire. Watermark physics, so 1–3 % in reflection and obvious only when the light rakes |
| | `coated` | matte/silk coated stock: white's fibre network with the coating's three effects applied to it — ~80 % of the fibre height and ~95 % of the per-fibre albedo spread suppressed, sub-50 µm relief filled entirely, a 25 µm point-spread (the pigment scatters at the surface, so it looks crisper than uncoated), 30–50 GU sheen in a much tighter lobe, and a 1–4 mm **coat-weight mottle** that lives in the gloss (1 % of it in the albedo, 55 % in the sheen) |

All six paper variants are built as an actual fibre stack rather than as a tinted ground
with marks on it — see *How paper works* below.

Plastic base colours are drawn 85% from a 100-colour sample palette taken from
photos of real device housings and jittered into a continuum, 15% from a free
HSV sampler that keeps saturated novelty colours possible.

### Material parameters

Pass these keyword arguments to `generate()` or `generate_array()` after the
common `material`, `size`, `seed` and `variant` arguments. A seeded draw is
reproducible; omitting a setting lets the material choose it. The table lists
which materials use each parameter. Unrecognised material keywords are
currently ignored, so check the spelling and material when an override has no
effect.

| Material | Parameter | Type / choices | Default and effect |
| --- | --- | --- | --- |
| Wood, paper | `mm_across` | `float`, physical image width in millimetres | Wood draws 150–300 mm and requires a positive value. Paper draws a stock-specific range: white/coated 12–22, kraft 16–30, recycled 13–24, newsprint 14–26, laid 26–44 mm. Controls the capture scale. |
| Wood, plastic | `specular` | `float` | Surface reflection weight. Wood derives its default from the chosen finish; plastic draws by variant: glossy 0.5–0.8, matte 0.1–0.2, textured 0.18–0.35. Local gloss still modulates the supplied weight. |
| Wood, plastic | `shininess` | `float` | Surface highlight exponent; larger values sharpen it. Wood derives its default from the finish; plastic draws glossy 60–120, matte 6–10, textured 14–30. |
| Wood, plastic, paper | `normal_strength` | `float` | Strength of relief in shading. Wood draws 1.1–2.2; plastic draws glossy 0.5–1.1, matte 1.5–3.0, textured 1.4–2.6. Paper defaults to 1.0 and applies it to the diffuse normal. |
| Wood | `species` | `str`: `pine`, `maple`, `ash`, `oak`, `cherry`, `walnut`, `sapele`, `mahogany`; or `None` | Omitted/`None`: uniform seeded choice. One species for the whole panel. |
| Wood | `finish` | `str`: `none`, `oil`, `polyurethane`, `acrylic`; or `None` | Omitted/`None`: seeded choice with probabilities 0.06, 0.30, 0.40, 0.24 respectively. The finish class is shared across a panel; its colour-change magnitude varies per board. |
| Wood | `colour_variation` | `float` | Defaults to 1.0. Scales the board-to-board colour spread and ageing contribution; 0 starts from the nominal species colour. Finishing and surface features still apply. |
| Wood | `age` | `float` from 0 (fresh) to 1 (aged), or `None` | Omitted/`None`: one seeded draw per panel. Changes the colour of species with an aged-colour definition, currently cherry. |
| Wood | `sapwood` | `float` probability from 0 to 1, or `None` | Omitted/`None`: the species' probability. Zero disables pale sapwood bands; an override applies only to species with a sapwood colour. |
| Wood | `knots` | `float` probability from 0 to 1, or `None` | Omitted/`None`: 0.4 for broader boards, 0.12 for narrow strips. Zero disables knots. |
| Wood | `ring_sigma` | `float`, or `None` | Standard deviation of log ring widths. Omitted/`None`: a seeded draw from the species' porosity-class range, between 0.22 and 0.35. Larger values increase ring-width variation. |
| Wood | `figure` | `str`: `plain`, `curly`, `ribbon`; or `None` | Omitted/`None`: usually plain, with occasional curly maple and curly/ribbon sapele. Shared across a panel; controls the fibre reflection pattern. Explicit choices work for every species. |
| Wood | `cut` | `str`: `cathedral`, `flatsawn`, `quartersawn`; or `None` | Omitted/`None`: each board draws a cut with probabilities 0.35, 0.45, 0.20 respectively. Controls ring geometry; quartersawn exposes ray fleck where the species supports it. |
| Wood | `light_dir` | Three floats `(x, y, z)` | Defaults to `(-0.5, -0.55, 0.78)`, pointing towards the light. Change it to inspect chatoyance under another lighting direction. |
| Paper | `extinction` | `float` | Optical extinction per layer of fibre coverage; larger values make each layer more opaque. Defaults: white/laid/coated 0.55, kraft 0.45, recycled 0.50, newsprint 0.40. |
| Paper | `creases` | `float`, or `None` | Omitted/`None`: a stock-dependent chance of a crease, with a seeded strength of 0.35–1.0 when present. Set 0.0 for a seamless sheet; positive strengths above 0.02 add creases. |
| Paper | `ambient` | `float` | Defaults to 0.62. Adjusts the ambient contribution to lighting; the sheet's lighting is still normalised to its mean. |
| Paper | `out` | `dict`, or `None` | Defaults to `None`. A supplied dictionary receives `(H, W)` arrays named `mass` (fibre coverage) and `formation` (normalised formation field) for measurement. It does not change the returned image or pixels. |
| Metal | `film` | `None`, `True`, a system-name `str`, or a `dict` | `None` uses the selected variant's film preset, if any. `True` selects that preset, falling back to `heat_tinted` for a plain variant. Strings choose `oxide`, `oil` or `titania`; a dictionary overrides `system`, `nm` (a two-float thickness range in nanometres) and/or `field` (`weld`, `spill`, `patch`). |
| Metal | `iridescence` | `float` | Defaults to 0.0 (off). Positive values weight groove diffraction on unfilmed `brushed`, `radial` and `engine_turned` textures. |
| Metal / brushed | `brush_angle` | Finite `float` in degrees, or `None` | Clockwise in image coordinates: 0 horizontal, 90 vertical. Values wrap at 360 degrees. Requires the explicit `brushed` variant, including when combined with `film`. Omitted/`None` preserves the existing random direction and seeded pixels. |
| Metal | `source_angular_radius` | `float`, degrees | Defaults to 0.53. Controls the light-source width for groove diffraction; a wider source suppresses its colour. |
| Metal | `groove_pitch_um` | Two floats `(lo, hi)`, with `0 < lo <= hi` | Defaults to `(1.0, 30.0)` micrometres. Bounds the groove spacing used for diffraction. |

For example, pin wood anatomy or request a flat paper tile:

```python
wood = generate(
    "wood",
    size=512,
    seed=42,
    variant="board",
    species="oak",
    cut="quartersawn",
    finish="oil",
    knots=0.0,
)
paper = generate(
    "paper",
    size=512,
    seed=42,
    variant="laid",
    mm_across=40.0,
    creases=0.0,
)
```

### Brushing direction

```python
image = generate("metal", size=(640, 480), seed=42, variant="brushed", brush_angle=90)
pixels = generate_array(
    "metal", size=256, seed=42, variant="brushed", brush_angle=45, iridescence=0.5
)
filmed = generate(
    "metal", size=256, seed=42, variant="brushed", brush_angle=0, film="oil"
)
```

`brush_angle` uses degrees; the generator converts to radians internally.
Negative and multi-turn values wrap at 360 degrees (for example, -90 and 270
choose the same angle). NaN, infinity and non-numeric values are rejected.
The option is for an explicitly selected brushed-metal texture, not a whole
`sample_sheet()` or another material/variant.

The chosen direction is shared by the groove fields, their breaks and roughness,
scratch strokes, anisotropic shading, optional groove diffraction, the
perpendicular sheen band and glints. With a `film` override, the underlying
brushed surface uses that direction too. Groove diffraction remains an unfilmed
metal effect. The angle does not set the light direction or directly set the
reflected-room geometry and film-thickness patterns.

Omitting the angle (or passing `None`) preserves existing seeded renders.
An explicit angle still consumes the original random-angle draw. Rotating the
noise domain can change its lattice size and subsequent random draws, so changing
the angle can also change fine detail and other seeded properties. The same
angle and arguments reproduce the same pixels in the same environment.

The [workflow notebook](../examples/texture_generator_workflow.ipynb) and
[results gallery](../examples/results.md) show 0°, 45°, 90° and 135° examples.

### Metal film and diffraction parameters

The [metal generator](../src/texture_generators/materials/metal.py) accepts
`film=None` by default: the three film variants enable their presets, while
the plain finishes retain their original random sequence. Set `film=True`
to use a preset, a system name (`"oxide"`, `"oil"`, `"titania"`), or a mapping
overriding `system`, `nm` and `field`.

```python
from texture_generators import generate

generate("metal", variant="heat_tinted", seed=42).save("heat-tinted.png")
generate("metal", variant="engine_turned", seed=42, iridescence=0.8).save(
    "engine-turned.png"
)
```

`iridescence` defaults to `0.0` and adds groove diffraction to `brushed`,
`radial` and `engine_turned`. `source_angular_radius` is in degrees: wider
sources wash out the rainbow. `groove_pitch_um=(lo, hi)` controls the
sub-pixel grating pitch in micrometres. These parameters belong to the Python
API; the metal CLI additionally exposes `--brush-angle` for the brushing direction.

## How wood works

Wood was rebuilt after the verdict that it *"looked like veneer rather than real
wood"* — it read as printed melamine laminate. The diagnosis was an inversion:
the generator had **high-contrast smooth rings and almost no fine anatomy**,
where real timber has **moderate rings and high-contrast axial pore texture**.
Photographs of sixteen real species make the point — on most of them the axial
pore and fibre texture dominates and the ring arcs are secondary.

| Stage | What it does |
| --- | --- |
| Scale | `mm_across` (150–300 mm board width) gives `px_per_mm`; pore size, ring pitch and relief all follow from it |
| Ring series | Ring widths are a **grown series, not a `linspace`**: log-normal widths (mean 2.4–2.7 mm measured, furniture stock is 1.5–3.5 mm) drawn as an **AR(1)** process — dendrochronology models a ring-width series as autocorrelated, and the measured lag-1 is 0.64. Plus an age trend (rings narrow outward). Evenly-ruled spacing is one of the two things that read as printed |
| Ring profile | A **sawtooth**, not a symmetric ramp: a gradual earlywood→latewood progression inside the ring and an **abrupt step** at the boundary. Abruptness is per porosity class — measured transition widths 0.32 mm ring-porous, 1.65 semi, 2.40 diffuse. A symmetric soft ramp is the signature of a colour-ramp lookup, which is the other thing that reads as printed |
| **Pores / vessels** | The single biggest win. On a flatsawn face you do not see pores as dots — the saw cuts each vessel tube lengthwise, so you see **axial streaks**, and that is what a laminate physically cannot have. Streak length is *geometric*, not anatomical: for a vessel of diameter `d` at out-of-plane angle `θ`, `L ≈ d/tan θ`, and since θ varies the lengths come out strongly **right-skewed** (measured median 4.4 mm, p90 14.5, p99 42, skew 4.6). Sizing streaks from vessel-*element* length is a known mistake that makes them 10× too short — a vessel is many elements fused and runs decimetres. Written to albedo **and** height, since an open pore is a trough that catches light |
| Porosity class | Ring-porous oak/ash put 2–4 rows of coarse vessels in a band at each ring start, and **that band is a fixed 0.5–1.5 mm regardless of ring width** — a wide ring means more latewood, not a wider pore band. Semi-ring-porous walnut tapers over 30–60 % of the ring; diffuse-porous cherry/maple/sapele/mahogany are uniform. **`pine` has no vessels at all**, and that absence is what makes a softwood read as a softwood |
| Colour | Eight species in **CIELAB** (rendered means land within 0.13/255 of their nominal Lab), with the lighting normalised to its own mean so the render *is* the specified colour. Earlywood→latewood is a ΔL\* offset; measured 20.5 softwood / 12.8 ring-porous / 6.3 semi / 4.5 diffuse — strong in softwoods and ring-porous hardwoods, weak in diffuse-porous ones, as it should be |
| Finish | Finishing is not a tint: raw wood's air/cell-wall interface (n 1.00 → 1.53) scatters light back as a white veil, and a finish index-matches it away. Applied in CIELAB — oil ΔL\* −8 / ΔC\* +8, polyurethane −6/+9, acrylic −4/+4 |
| **Chatoyance** | Aligned fibres reflect anisotropically, so wood's luster shifts with angle. The fibre-tangent field is **the gradient of the same domain warp that bends the rings** — following Liu et al., *Simulating the Structure and Texture of Solid Wood* (ACM TOG 35(6), 2016), where the fibre directions "follow naturally from the distortions", so no second figure system is needed. Shaded with **two lobes**: a sharp isotropic surface lobe from the finish film over a soft anisotropic fibre lobe coloured `albedo**0.5`. That layering is what gives a polished panel depth |

Species anatomy, colour and BRDF figures are largely **calibration assumptions
from the trade and anatomical literature rather than measurements of these
grades** — the code flags which is which. The `L ≈ d/tan θ` streak relation is
derived geometry, not a published figure.

| Stage | What it does |
| --- | --- |
| **Ray fleck** | Ray tissue is **~17 % of hardwood xylem** (some species over 30 %) and rays run *radially*, so their appearance flips between faces. The cut is now explicit — `cathedral` / `flatsawn` / `quartersawn` — and on a **quartersawn** face the rays are sliced along their length and exposed as broad sheets: **ray fleck**, the figure of "tiger oak". Lens-shaped, tapering at both ends, measured median 9.6 mm long × 0.40 mm wide at aspect 23:1, right-skewed in length (p90 21 mm, max 55). Gated on species *and* cut, with target coverage ranges: oak 0.10–0.25, maple 0.02–0.06, ash 0.004–0.015, cherry/walnut 0.003–0.012, sapele/mahogany 0.004–0.014, **pine 0.000**, and zero on every non-quartersawn cut |
| Why it *flashes* | Rays are radial, so on a quartersawn face the ray tissue's fibre direction is perpendicular to the axial grain. Inside a fleck the tangent field is therefore **rotated 90°** (measured: 80.5° inside vs 0.02° outside, softened by the mask ramp), which puts the fleck crosswise in the anisotropic fibre lobe and makes it light up and go dark as the light moves. Ray fleck is a chatoyance effect, not a pigment one — the albedo contrast is only ΔL\* 3–8 and the rotation does the work |

Not yet modelled: the volumetric 3-D field of Liu et al. that would make
flatsawn, quartersawn and end grain all fall out of one model instead of being
separate cases.

## How paper works

Paper got a physical rewrite, because the noise-and-strokes approach the other
materials use cannot produce it. Three facts drive the design:

1. **Coverage.** Total fibre length per unit area is grammage ÷ coarseness, so
   80 g/m² office paper carries ~440 mm of fibre per mm² — about **13 fibre
   layers**. A few hundred strokes on an empty ground is coverage well under 1:
   isolated marks, which the eye reads as *scratches*. `count_for_coverage`
   turns a target coverage into a fibre count, so a 20 mm crop asks for tens of
   thousands of fibres and gets them (16 k fibres deposit in ~80 ms).
2. **Sub-pixel width.** A 30 µm fibre is 0.15–1.0 px wide at a typical capture
   scale, so it can only ever contribute *partial* coverage. Bilinear mass
   accumulation does that exactly; an opaque 1-px line is several times too
   wide *and* too dark, which is the scratch signature. Above a pixel — which
   happens at 2000 px — the fibre is a real ribbon and is laid down as
   parallel strands across its own normal, because a 3 px fibre drawn as a
   1 px line at triple opacity is just a narrow hard-edged scratch again.
   Samples are also kept within a pixel of each other along the fibre: any
   further apart and the bilinear footprints stop overlapping, so the "fibre"
   becomes a string of beads.
3. **Translucency.** Light enters the sheet, spreads sideways over a
   point-spread radius, and leaves. That radius is **per stock**, not one
   figure: 25 µm for `coated` (the pigment scatters at the surface), 90 µm for
   `white`, 130–150 µm for `recycled`, `laid` and `kraft`, and 250 µm for
   `newsprint`, whose unfilled groundwood sheet spreads light furthest — so
   25–250 µm across the six. Height detail finer than that
   radius produces almost no *diffuse* shading while the surface sheen keeps
   all of it — so the shading pass uses **two normals**, blurred for diffuse
   and sharp for sheen. It is also why formation, obvious when you hold a
   sheet to a window, is nearly invisible flat-lit: reflectance saturates with
   grammage, so a ±8 % mass swing is only ±1–2 % in reflected light.

The pipeline, in order:

| Stage | What it does |
| --- | --- |
| Scale | `px_per_mm = width / sheet_mm`. Every feature below is specified in millimetres, so a 2000 px render of the same sheet shows **more detail**, not bigger features |
| Placement | Matérn cluster (Neyman–Scott) process mixed with uniform, MD-elongated. Flocculation *emerges* from where fibres land, so formation and fibre texture agree by construction instead of being two independent layers |
| Deposition | 3 depth slabs × 2 tone groups of worm-like-chain fibres — heading variance `2·ds/ℓp` per step (tangent correlation `exp(−s/ℓp)`) plus a Poisson kink process. Persistence ≈ 1 fibre length, matching measured curl indices of 0.08–0.25. Projected width varies **along** each fibre (CV ≈ 0.20, correlation length ≈ 350 µm — **both figures are calibration assumptions, not measurements**: fibre analysers report a *population* width distribution, which is a different quantity and cannot be substituted for the variation along one fibre) because collapse is intermittent — an uncollapsed tracheid presents ~20–30 µm, a collapsed one ~25–45 µm, and thick-walled latewood resists what thin-walled earlywood gives way to. Ends come in two populations: a **native** tip tapering over ~300 µm to a blunt 30–60 % of mid-width (not to zero), and a **cut** end from refining that does not taper and instead frays into a brush |
| Fibrillation | Refining peels microfibril bundles off the outer wall, but they are **50 nm – 1 µm** wide — far below one pixel even at 2000 px (10 µm/px). So they are rendered as extra sub-pixel **opacity** in a 0–5 µm fringe hugging each fibre edge, never as drawn geometry; putting a sub-pixel object on the raster as a pixel-scale mark is the same category error that made the first fibres read as scratches. Parameterised as fibre analysers do, by **fibrillation index** (fibril area ÷ parent-fibre area): 0.010 unrefined kraft sack → 0.060 heavily-worked recycled. Those per-stock index values are **plausible instrument outputs assigned as calibration assumptions, not measurements of these grades**. The index is a *total* budget — the cut-end brushes take their share of it and the wall fringe gets the rest — so the extra mass deposited is the index (6.18 % measured at 0.060) and not more; it used to be added on top of the fringe, which made 0.060 deposit 9.49 %. Divergence follows the microfibril angle of the layer peeled: **the layer angles are textbook** (S1 an outer crossed helix at 45–75°, S2 the bulk at 10–25°), but the **two-mode mixture, its 30/70 weights and the 0.55 lay-back factor that lands the apparent 10–40° are inference from those angles, not a measured distribution** |
| Two accumulators | Mass is **additive** (it is extensive); light is **not**, so the optical layer is a depth-ordered Beer–Lambert `over` composite (`α = 1 − e^(−τ·mass)`) in linear light, each slab blurred by its depth. That is what produces occlusion and the felted look; summing would saturate to flat grey |
| Inclusions | Kraft shives; recycled's three ink populations |
| Felt | Below the resolvable-fibre scale — fines, mechanical pulp, the weave seen *through* a coating — there is no point drawing individual fibres, so a **LIC** felt field (`core/lic.py`) supplies that texture and feeds mass, albedo and height at once. Stacked over several *independently oriented* layers: one direction field gives every filament in a neighbourhood the same heading, which is combed hair rather than felt. Cost is `O(H·W·layers·length)`, so `layers × length_px` is budgeted to ≈150 at any resolution. The kernel comes from **two** populations, not from whole-fibre length: a **fines** scale (`fines_mm`, 0.05–0.30 mm; 0.12–0.26 mm as configured) taking two thirds of the budget, and an optional second, longer one at fibre length (`fibre_mm`, 1.7 mm `newsprint` / 1.9 mm `laid`) as a single octave — the fines need many orientations before they stop looking combed, the long grain reads fine at two or three. `coated` sets `fibre_mm` to 0 and carries the fines alone |
| Wire marks | `laid` only: laid and chain wires as height, not albedo. Both are parameterised by *count per tile* rather than by pitch — a periodic feature needs a whole number of periods or the sheet seams — so the realised pitch is whatever the nearest integer count implies (`wire_counts` reports it) |
| Height | Built in **micrometres**, dominated by a 0.2–1.0 mm Matérn "tooth" band, then converted to pixel-equivalent units so `np.gradient` yields true surface slopes. Tuned to paper's measured **RMS slope** — 0.079 white / 0.114 kraft / 0.087 recycled / 0.102 newsprint / 0.081 laid, i.e. 4.5–6.5°, and 0.028 (1.6°) for `coated`, at the reference sampling of 25.6 px/mm. Slope is not scale-free (finer sampling resolves steeper features), so the calibration carries a resolution the way a real roughness figure does. Cockle always; Voronoi-facet creases occasionally |
| Surface statistics | The height field is deliberately **not Gaussian**. At this sampling (39 µm/px) you resolve *fibres*, not pores, so protruding fibres and crossings are the extreme values and the upper tail is heavy: **Ssk > 0** for every uncoated grade. Only once the peaks are pressed flat does the pore population win and Ssk go negative — which is why `coated` is the one variant measuring below zero. The *sign pattern* is an inference from that resolution argument rather than a measurement of these grades; the target magnitudes it is calibrated to are given, and flagged, under the table below. Ssk and Sku are only defined against a filter cut-off, and everything here is quoted at `lc = 1 mm` |
| Calendering | A nip is carried by the tallest asperities first, and paper is highly compressible in z but has no lateral mass transport beyond a fibre width — so **peaks are crushed and valleys survive** (a pore stays a pore). Implemented as a one-sided **softplus clip** on the roughness band, `z' = z − k·w·log1p(exp((z − z₀)/w))`, with per-stock `calender_k`: 0 hand-formed `laid`, 0.05 unbleached `kraft`, 0.35 machine-finished, 0.45 calendered copier, 0.80 gloss-calendered `coated`. A hard `min(z, z₀)` would be wrong — it makes a delta spike at the clip level, where viscoelastic recovery gives a smooth knee. The same map drives the gloss freckling and a slight darkening, because the crushed spots are locally densified: that is **calender blackening**, and deriving all three effects from one map is what makes them cohere |
| Shading | Two normals, Oren–Nayar diffuse, wrap diffuse, a weak broad sheen (uncoated paper is 4–10 gloss units at 75°), and lighting normalised to its own mean so the sheet's average reflectance is exactly the L\*a\*b\* colour it was given |

**Measurements of this generator's own output** — not of real paper, and not
independent evidence for the model. Roughness band only, S-filtered at an ISO
16610-21 cut-off of **`lc = 1 mm`**, 6 seeds at the reference 25.6 px/mm
sampling:

| variant | Ssk | Sku | Sa/Sq | RMS slope | `calender_k` |
| --- | --- | --- | --- | --- | --- |
| `white` | +0.49 | 3.22 | 0.793 | 0.076 | 0.45 |
| `kraft` | +0.57 | 3.52 | 0.787 | 0.110 | 0.05 |
| `recycled` | +0.45 | 3.21 | 0.793 | 0.088 | 0.35 |
| `newsprint` | +0.53 | 3.25 | 0.793 | 0.100 | 0.35 |
| `laid` | +0.71 | 4.12 | 0.767 | 0.074 | 0.00 |
| `coated` | −0.37 | 3.58 | 0.789 | 0.026 | 0.80 |

The cut-off is not a detail: it is half of what the numbers mean. Ssk and Sku are
properties of a *band*, so the same surface reports different values in every
band and a figure quoted without its cut-off says nothing. The ISO 16610-21
Gaussian filter transmits 50 % at `lc` and its weighting function has sd
`0.18739·lc` — so the "1 mm high-pass" this table used to claim, implemented as a
Gaussian of **sigma** 1 mm, was really a 5.34 mm cut-off, five times too wide,
with cockle and millimetre-scale periodic structure counted as roughness. The
old numbers in this table were measured in that wrong band; `laid` in particular
reported Sku 5.46 there and 26.9 once the band was corrected, which is what
exposed the two artefacts described in `_skew_warp` and `_asperities`.

The target these are tuned against — **Ssk within ±(0.9–1.3) and Sku 3–5 at
`lc = 1 mm`, positive for uncoated grades and negative for coated** — is
*unverified domain knowledge used as a calibration assumption*, not a measurement
of these six grades. Its two ends are load-bearing in different ways: `Sku < 3`
is a platykurtic surface, lighter-tailed than a Gaussian, which no real surface
is; `Sku` in the tens is a single-pixel artefact rather than a rougher surface.

`Sa/Sq` is a free cross-check: it equals `sqrt(2/pi) = 0.79788` exactly for a
Gaussian height distribution and departs from it as `|Ssk|` grows (Sampson &
Wang, *J. Mater. Sci.* **54**, 2019). The 0.767–0.793 spread above is that
departure, and it is the evidence the field really is non-Gaussian rather than
merely reported as such. `laid` sits highest in both Ssk and Sku, which is the
one place the ordering is a prediction rather than a fit: it is the only stock
with no nip in its history, so nothing has truncated its fibre crowns.

**There is no published bearing-area (Abbott–Firestone) curve for paper.** The
Sk-family parameters were not fitted to one, and nothing here should be read as
implying such a dataset exists.

Colour is specified in **CIELAB** measured on real stock rather than picked in
RGB, so whiteness is −b\*, ageing is +b\*, kraft sits at L\* 59 a\* 9.5 b\* 24,
newsprint's lignin yellowing is a +b\* draw, coated stock is brightened to
b\* −2, and recycled is a warm low-chroma grey rather than a dead neutral.

## Core modules

| Module | Contents |
| --- | --- |
| `core/noise.py` | `gradient_noise`, `fbm`, `ridged` (+ `*_at` variants for arbitrary coordinate arrays); vectorised Perlin-style lattice noise with quintic fade and anisotropic `(freq_x, freq_y)` support. The gradient lattice is sized to the coordinates actually sampled, so tall canvases and warped/ring coordinates never band-repeat; a per-axis `periodic` flag restores unit wrapping where it is wanted (radial metal's θ axis) |
| `core/worley.py` | `worley` → `(F1, F2)` from a jittered feature grid sized to the sampled domain, 3×3 neighbourhood scan, `stretch` for elongated cells, euclidean/manhattan/chebyshev metrics |
| `core/warp.py` | `warp` (`p' = p + A·W(p)` from two independent fbm fields), `double_warp`, `rotate` |
| `core/fields.py` | `normalize01`, `remap`, `smoothstep`, `height_to_normal`, `blur`, `draw_segments` (PIL-rasterised signed line scatter) |
| `core/spectral.py` | `matern_field`, `band_field`, `shaped_noise`, `freq_grid`, `resize_periodic` — noise built by shaping white noise in Fourier space. fBm is scale-free; the Matérn/von Kármán spectrum `S(k) = (1 + (2πλk)²)^−(ν+1)` has a *characteristic* scale and an exponential autocorrelation at ν = 0.5, which is what measured paper formation has. FFT filtering also makes these fields tile exactly. `resize_periodic` resamples such a field by zero-padding or cropping its spectrum — exact for content the field can hold (up then back down is an identity to float error, Nyquist bins included), and unlike bilinear it keeps the result seamless, which is what lets the expensive LIC felt be computed small and scaled up. The felt *scaling* is still approximate, but because of the smaller canvas' Nyquist limit and a clamped kernel length, not because of the resize |
| `core/fibres.py` | `deposit`, `count_for_coverage`, `cluster_points`, `axial_von_mises` — stochastic fibre networks: worm-like-chain centrelines with Poisson kinks, Matérn-cluster placement, and sub-pixel mass accumulated by bilinear splatting via `np.bincount` (an order of magnitude faster than `np.add.at`), wrapping at the edges |
| `core/lic.py` | `direction_field`, `lic`, `felt`, `slope_blur` — Line Integral Convolution (Cabral & Leedom '93): white noise smeared along the streamlines of a smooth heading field, which correlates pixels *along* the flow and leaves them independent *across* it. That is the signature of a fibrous surface, and the right tool below the resolvable-fibre scale where depositing fibres one by one is both wasteful and wrong. Direction is treated as an **axis**, not an arrow (interpolated in the doubled angle, sign resolved against the walk's own heading) — get that wrong and streamlines fold back, the integral collapses to a local average, and the result is a mushy blotch with no filaments in it. `felt` blends octaves for a mixed furnish and sums `layers` **independently oriented** stacks, which is what separates a felt from combed hair |
| `core/colour.py` | `lab_to_linear_rgb`, `lab_to_srgb`, `srgb_to_linear`, `linear_to_srgb`, and the inverses `linear_rgb_to_lab`/`srgb_to_lab` — CIELAB is how paper *and* wood colour is actually specified, translucent fibres must composite in linear light, and the inverse is how a rendered colour gets checked against the measurement it came from (a ΔL\*, not a few display levels) |
| `core/film.py` | `FilmSystem`, `SYSTEMS`, `film_table`, `film_tint`, `wavelength_rgb` — thin-film interference integrated over 36 wavelengths from 380–730 nm against the CIE 1931 observer, then stored in a thickness/angle lookup table. Spectral integration avoids the aliasing of a three-wavelength RGB approximation; `wavelength_rgb` supplies the groove-diffraction colour |
| `core/shading.py` | `shade(...)` — Lambert diffuse over height-derived normals with a high (slightly sky-directional) ambient floor, Blinn-Phong specular with Ward-style anisotropic lobe; `specular`/`shininess` accept per-pixel `(H, W)` arrays (ring gloss, pass bands, smudges), `spec_tint` tints the highlight towards the albedo for conductors (with Schlick whitening on steep micro-slopes), `specular2`/`shininess2` add a second tight clearcoat lobe, `cavity` darkens local concavities, and `normalise` (**off by default**, on only for wood) divides the lighting by its own mean so a measured albedo survives the shading pass. `shade_translucent(...)` shades a thin scattering sheet with **two** normals — blurred for diffuse (the subsurface point-spread radius), sharp for sheen — with Oren–Nayar roughness, wrap diffuse, and mean-normalised lighting; `gaussian_blur` is the separable wrap-around blur both use |

Randomness flows through a single `np.random.default_rng(seed)`; no module
touches the global `np.random` state.

## Tests

From a checkout, install the locked development environment and run the suite:

```bash
uv sync --locked
uv run pytest
```

The suite covers every material × variant rendering at 128px, determinism,
seed variation, non-square sizes, the module-level contract, `ValueError` on
unknown material/variant, paper luminance, the contact sheet, the `samples`
run directory (fixed-seed determinism, `--only` filtering, the `latest`
symlink), CLI subprocess smoke tests, and packaging (`tests/test_packaging.py`).
`tests/test_cli_click.py` covers the Click CLI itself: that every subcommand is
registered and documents itself with an example, `list`, the `--json` report
schemas, the usage-error exit codes (bad size, `-o` with `--count` or
`--outdir`, unknown variant) and clean errors for an unwritable output path,
render determinism, `python -m texture_generators` parity, and shell-completion
generation.
CI runs the full suite on Python 3.12, 3.13 and 3.14 with both locked and minimum
runtime dependencies. The [release guide](releasing.md#prepare-and-check-the-artifacts)
gives the commands. Parallel full-suite runs require separate
project copies because packaging tests build in the project root.
Paper additionally gets the physical checks — coverage, worm-like-chain curl,
spectral field statistics, LIC orientation and kernel length, per-variant RMS
slope against its measured target, and `laid`'s wire pitch as an FFT line —
which is most of the ~2 minutes the suite takes. Tests run against the
installed package (src layout), so install editable first.

## Development

See the [README development instructions](../README.md#development) for the
locked environment, tests, linting and formatting. Ruff is the formatter.

mypy is configured but is not yet clean over the package internals; its output
is informational, not a passing release gate. Run it in the development
environment:

```bash
uv run mypy
```

The package ships `py.typed`, so downstream type checkers can use its public
annotations. That marker does not imply the internal mypy checks pass.

Project layout:

```text
texture-generator/
├── pyproject.toml
├── uv.lock
├── CHANGELOG.md
├── LICENSE
├── docs/
├── src/
│   └── texture_generators/
└── tests/
```

## Versioning

The package follows semantic versioning. The literal `project.version` in
`pyproject.toml` is the source of truth; the public `__version__` reads the
installed distribution metadata. Changes are recorded in
[CHANGELOG.md](../CHANGELOG.md). See [Releasing](releasing.md) for building,
checking and publishing a version.

## Performance

The original development measurements below are approximate and have not been
re-benchmarked for this packaging change; hardware and dependency versions
affect timings. In those measurements, a 512×512 texture took 40–410 ms; the slowest is `paper`, which deposits tens
of thousands of fibres as continuous sub-pixel ribbons. 1024×1024 paper is
~1.1 s and 2000×2000 paper is ~4.2 s — the splat dominates, and it scales with
total fibre *area*, so a bigger render genuinely draws more fibre.

The three variants that carry a LIC felt cost more, because LIC is a per-pixel
streamline walk on top of the deposition: at 2000×2000, `newsprint` is ~8.5 s,
`laid` ~8.4 s and `coated` ~5.3 s, against ~4.7 s for the three felt-free
variants (0.7–1.5 s at 512×512). LIC costs `O(H·W·layers·length)` and the felt
is specified in millimetres, so its cost would otherwise grow as the *cube* of
the sampling density; `_felt_stack` caps the canvas it computes on at ~1.1 Mpx
and `spectral.resize_periodic` band-limits it back up, which is what keeps the
felt variants inside twice the cost of the others rather than four times.

## Notes and limits

- Metal and plastic size fine features such as brush streaks and stipple
  in **pixels**, so a 1024px render keeps a similar fineness of brushing; large
  features scale with the canvas. Wood uses `mm_across` for its pore anatomy
  while ring geometry also uses normalised coordinates. Paper is parameterised
  in millimetres throughout, so raising its resolution resolves more of the
  same sheet. Pass `mm_across=` to choose the wood or paper capture scale.
- Metal, plastic and wood are **not** seamlessly tileable: their noise fields,
  scratch scatter, plank splits and highlight bands do not all wrap, so edges
  will not match up. **Paper does tile** when creases
  are off (`creases=0.0`) — its spectral fields, fibre splatting, specks and
  shading derivatives all wrap. The crease network does not, so a creased
  sheet will show a seam.
- Seamlessness costs `laid` some pitch accuracy: a periodic feature must fit a
  whole number of periods into the tile, so the realised pitch is the requested
  one rounded to the nearest integer count. Laid lines quantise finely (15–21
  periods on a 20 mm crop, i.e. 0.95–1.33 mm), but the 20–30 mm chain lines do
  not — only one period fits a 20 mm crop, so *every* chain pitch collapses to
  20.0 mm there. Render a wider `mm_across` if the chain spacing matters.
- Paper composites its fibre stack with Beer–Lambert `over` in **linear light**
  and encodes to sRGB once at the end (`core/colour.py`). Plastic, wood and
  the plain metal finishes shade in display values. Filmed metal converts its
  reflected term to linear light for tinting and then back to sRGB, with
  percentile scaling to control clipping; this is not a full linear-light
  rendering pipeline.
- Determinism is byte-exact for identical arguments and package/dependency
  versions on a given platform. Across platforms or numpy builds, exact bytes
  can differ
  (`sin`/`cos` rounding at the float32 boundary).
- Peak memory scales with area: noise evaluation holds several full-frame
  intermediates, roughly a few hundred MB transiently at 2048×2048.
