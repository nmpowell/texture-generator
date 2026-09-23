# Galvanised zinc

`metal/galvanised` is a separate physically scaled surface generator. The four
release presets provide a procedural approximation of zinc sheet, with tested
numerical contracts, optics and file round trips. They are authoring defaults,
not a measured material calibration. The [implementation record](galvanised-implementation.md)
separates release acceptance from the unfinished research programme.

## Generate and relight one surface

```python
from texture_generators import generate_maps, render_material, export_material
from texture_generators.materials.galvanised import GalvanisedConfig, PreviewConfig

recipe = GalvanisedConfig(preset="regular", size_mm=(100, 75))
maps = generate_maps(
    "metal",
    variant="galvanised",
    size=(512, 384),
    seed=42,
    galvanised=recipe,
)
render_material(maps).save("galvanised.png")
render_material(
    maps,
    preview=PreviewConfig(rig="oblique", light_azimuth_deg=120),
).save("galvanised-relit.png")
export_material(maps, "galvanised-42")
```

Map dimensions use `(width, height)`; arrays use `(height, width, components)`.
`size_mm` is the physical width and height of one repeating tile. When omitted,
the width is 100 mm and the physical aspect ratio follows the image. Changing
pixel count at a fixed physical extent samples the same crystallisation state.

The existing `generate` and `generate_array` functions also accept explicit
`variant="galvanised"`, `galvanised=recipe`, and `preview=PreviewConfig(...)`.
Their default preview agrees with rendering the corresponding maps. Existing
metal variants and their seeded default selection remain unchanged.

The default camera has a fixed 24° tilt and 135° azimuth, slightly off the
studio light's reflected direction. This makes directional lobe differences
visible without increasing physical relief or zinc reflectance. Use
`PreviewConfig(view=(0.0, 0.0, 1.0))` for a normal-incidence view. Changing the
light azimuth leaves the camera fixed; notebook recipes record both controls.

The gallery presets are `regular`, `minimised`, `weathered`, and `wet_storage`.
`galvanised_preset="regular"` is a shorthand for a recipe, mutually exclusive
with `galvanised`. `inconspicuous` and `batch` exist as experimental config
presets with separate, incomplete scale and fabrication validation.

The fresh presets use coherent grain reflections with a restrained fine growth
pattern. `regular` has a nominal 8 mm spangle diameter; `minimised` uses 1.5 mm.
`weathered` adds widespread dull grey patina while retaining the original zinc
substrate. `wet_storage` adds uneven pale deposits with finer moisture variation.
Weathering controls are appearance parameters rather than elapsed years.
Directional lights show grain contrast most clearly; diffuse overcast light
intentionally makes clean zinc more uniform. Grain contrast comes from authored
roughness and reflection directions, while the zinc reflectance stays constant.

## Sampling quality

`quality="draft"` uses shared 2 × 2 midpoint samples. With
`representation="single_lobe"`, production uses 4 × 4 samples and reference
checks 8 → 16 → 32 samples per axis. The reference path also checks independent
17/33 grids to catch false agreement between dyadic grids. Each footprint must
pass the recorded absolute field tolerances; unresolved footprints raise an
error. Reference map calls are limited to 4096 pixels for diagnostic use.

The default rich production representation still retains four actual shared
samples per material. Its angular fitter is under development, so rich reference
maps fail explicitly. Higher spatial quality does not remove the single-lobe
representation's angular approximation. Complete spectrum filtering remains
unvalidated; metadata records `finite_band_certified=false`.

The [offline physical-footprint checker](galvanised-footprint-fitting.md)
extracts candidates and dense targets from real surface rectangles and tests
stored fitted records at a fixed normal view. It preserves rejected diagnostics
and provides a development tool for the angular fitter.

The generator revision is `galvanised-3`, with preset revision
`2026-09-23.release-1`. It changes the grain reflection model, relief defaults
and weather distribution. Replay rejects older generator versions instead of
silently regenerating different samples.

## Material data

The default maps include signed `height_um`, signed unit-vector `normal_ts`,
linear `base_color_linear`, `roughness`, `anisotropy`, doubled-angle
`anisotropy_axis`, and three visible fractions: `metallic`, `patina_coverage`,
`white_stain_coverage`. Fractions sum to one. A weathered pixel can contain
different zinc and deposit responses; its compatibility base colour and
roughness alone do not preserve that mixture.

The default `representation="rich"` also carries structured, material-specific
angular lobe records and optical parameters. Export retains those records.
The representation's approximation is recorded in metadata; numerically
lossless storage does not imply an exact integral of the continuous surface.

`maps=("height_um", "normal_ts", "grain_id")` selects output channels.
`grain_id` is an integer centre-sample diagnostic, unsuitable for interpolation.
For repeated sampling, use `build_state(recipe, seed=42)` and
`sample_state(state, size=(512, 384))` from `materials.galvanised`.
Use `output_dir=` to memory-map sampled arrays and `chunk_size=` to limit tile
work. Large rich bundles have substantial storage requirements; consult the
measured resource record before scheduling concurrent jobs. Automatic gallery
execution serialises galvanised jobs; explicit `--jobs` remains an override.

Normals use the right-handed frame `X=u`, `Y=Ly-v`, `+Z` out of the surface,
where raster `u` increases right and `v` increases down. If `p` and `q` are
physical slopes, the normal is the normalised `(-p, +q, 1)`. Height is in
micrometres; millimetre tile dimensions determine derivative spacing. The axis
stores `(cos(2θ), sin(2θ))` for a clockwise raster angle. It must be decoded
and converted to a tangent, not connected directly to a tangent input.

Map channels contain no preview lighting, exposure or display curve. Fresh
zinc colour is uniform, pinned conductor reflectance. Deposit colour, thickness
and exposure progression are authoring models, not a calendar-time corrosion
prediction. `normal_strength` belongs to `PreviewConfig` and is an explicit
artistic preview adjustment.

## CLI and exports

```sh
texture-gen metal --variant galvanised --galvanised-preset regular \
  --size 512x384 --size-mm 100x75 --seed 42 -o galvanised.png

texture-gen maps metal --variant galvanised --size 512x384 \
  --size-mm 100x75 --seed 42 --outdir galvanised-42 --json

texture-gen maps metal --variant galvanised --config galvanised.json \
  --size 512 --seed 42 --profile tiff --outdir galvanised-tiff
```

A config file is a strict JSON object of `GalvanisedConfig` fields. It cannot
be combined with `--galvanised-preset` or `--size-mm`. Unknown and duplicate
keys are errors. Galvanised flags require explicit `--variant galvanised`.
Legacy film and brushing controls do not apply to this material.

The core `lossless` profile stores numerical NPY arrays and a JSON manifest,
including signed floats, integer IDs, layer records, checksums and replay
metadata. The `tiff` profile needs `pip install 'texture-generator[export]'`
and writes uncompressed float32/integer TIFF maps with the same canonical
semantics, plus NPY lobe records. These are data files; RGB previews alone are
not a material bundle. See [export details](galvanised-export.md).

```python
from texture_generators import load_material, replay_material

loaded = load_material("galvanised-42", mmap_mode="r")
replayed = replay_material("galvanised-42")
```

Replay checks the stored generator and resource identity and uses the complete
resolved recipe rather than current preset defaults. Incompatible recipes fail
explicitly. Existing export destinations require an explicit Python
`overwrite=True`; the CLI fails rather than replacing them.

The zinc/CIE numerical resources have their own attribution and licence notices
in the installed package. See the [source register](galvanised-sources.md).
Blender integration is outside this task, following the user's clarification.
