# Galvanised material bundles

`export_material(maps, outdir, profile="lossless")` writes a directory and returns
the path to `material.json`. `load_material(outdir, mmap_mode="r")` verifies and
loads it. `replay_material(outdir, size=(width, height), maps=(...))` reconstructs
the physical state from the stored resolved configuration and concrete material
key. The user seed is provenance; the key is what controls replay. Read-only
memory mapping applies to NPY and to the uncompressed TIFF maps written here.

The **lossless** profile stores each requested sampled channel as NPY without
pickle. Physical height remains signed float32 micrometres. Normals, slopes and
axes remain signed float32 values; grain and orientation IDs remain unsigned
integers. This is exact storage of the sampled representation, including rich
lobe records when the material uses that representation. It cannot recreate
subpixel details that sampling filtered out; replay can resample the stored
state at another resolution. Optional diagnostic IDs live under `diagnostics/`.

The optional **tiff** profile writes float32, uncompressed IEEE TIFF for float
maps and retains NPY for integer diagnostics and structured lobe records. Install
`texture-generator[export]` to enable it. `validate_export_profile("tiff")` checks
the optional dependency before the surface is generated. TIFF samples have no
per-image scaling, clipping, colour transform or tone curve. `base_color_linear`
is linear RGB; all other channels are data. Three-component maps use contiguous
RGB samples and two-component maps use contiguous data samples. A standard
image viewer may display data maps as if they were colour; the manifest gives
their actual semantics. `tools/galvanised/check_export.py` independently reads
the uncompressed float TIFF storage using Python's `struct`, with no tifffile
decoder, and validates checksums and dimensions. It is an interchange check,
not an import adapter for arbitrary TIFF variants.

The version 1 manifest contains the profile, complete JSON-safe provenance,
selected map descriptors (relative path, SHA-256, shape, dtype, units and colour
space), optional `layers/lobes/records.npy` and optional `preview.png`. A preview
is sRGB presentation only. The reader rejects absent files, checksum failures,
unexpected formats, type/shape/unit mismatches, unsupported versions, path
traversal and symlinks that escape the bundle. Replay also requires the installed
optical, morphology/population and renderer energy-table identities to match
the recorded hashes. Rendering a loaded material also verifies any recorded
optical and renderer identities before shading. It uses
`GalvanisedConfig.from_resolved_mapping`, so changes to preset defaults cannot
silently change the stored recipe.

Export first writes to a sibling staging directory and validates the written
data. It writes `material.json` last, then renames the directory into place.
An existing destination raises `FileExistsError` unless `overwrite=True`.
Overwrite moves the old bundle aside and restores it if publishing the new one
fails. A failed write removes staging files and leaves the previous bundle
usable.

These exports describe a physically scaled procedural model with documented
authoring choices. Calibration to measured specimens and a quantified angular
fit remain separate release gates; `lossless` names numeric storage fidelity,
not perfect physical prediction.
