# Galvanised surface state and sampling

Status: implemented with procedural release presets. Coherent grain reflections and restrained branch relief replace the earlier dominant comb motif; normal weathering produces broad grey patina. The morphology remains an authored approximation. No specimen-calibrated appearance claim is made.

## Contract and replay

`materials.galvanised` exposes `GENERATOR_VERSION = "galvanised-3"`, `GALVANISED_PRESETS`, frozen `GalvanisedState`, `build_state`, `sample_points`, `sample_state`, `optical_parameters`, `surface_resource_hashes`, and the legacy float32 RGB `generate` adapter. A seeded state uses PCG64 and exactly one `material_key_from_rng` draw. An explicit key bypasses that draw; an optional seed alongside the key is replay metadata. Topology, weather and fabrication object streams are separately named. Resolution, chunk size, lighting and exposure do not enter topology creation.

`sample_points(state,x_mm,y_mm,footprint_mm=(0,0))` accepts broadcast physical millimetre coordinates over any number of tile periods. It returns continuous final/substrate/deposit height, IDs, real inter-ID boundary distance, visible coverages, compatibility roughness/anisotropy/axis and separate intrinsic zinc roughness/anisotropy. A positive footprint uses shared 2 × 2 quadrature. Zero exposure produces exactly zero deposit height. Where the partition has no inter-ID edge, boundary distance is positive infinity.

`sample_state(state,size=(W,H),maps=...,chunk_size=...,output_dir=...)` requires both axes ≥3. It returns only selected channels, with dependencies allocated temporarily. `output_dir` maps arrays to `.npy`; temporary dependency files are removed after construction. Final float32 height is sampled first and canonical normals are derived from that exact array by independent X/Y physical central differences using one-pixel periodic halos. The preview adapter accepts 1–2 sample axes via an explicitly coarse three-sample representation with a flat derivative on each undersampled axis. Draft uses 2 × 2 shared midpoint samples. Single-lobe production uses 4 × 4. Single-lobe reference checks 8 → 16 → 32 samples per axis with independent 17/33-grid comparisons, freezing the first passing rate for each pixel. Rich reference remains unsupported pending angular fitter integration; rich production retains the provisional four-sample representation.

The rich representation retains up to four actual shared-subpixel lobe records per visible material/pixel, including identity, weight, local effective normal, tangent and material-specific GGX widths. Local slopes are estimated by the four height samples; the exported resolved slope is subtracted before their residual is associated with the material lobe. This is an initial quadrature representation, not a calibrated angular fitter. Record arrays are allocated only for visible materials, and mapped/compacted to exact length when streaming. MaterialMaps checks weight closure and ≤4 records per material. The compatibility base-colour, roughness and metallic maps remain approximations of the separate conductor/dielectric responses.

Metadata contains the full resolved configuration, seed/key, dimensions and physical extent, optical parameters, source hashes, filtering, representation, population statistics, resource estimates, warnings, and explicit unknown angular fit error. `resource_hashes` identifies the optical resources expected by export/replay; `surface_resource_hashes` identifies the package's morphology and population-fit files.

## Morphology and weathering choices

The package's `data/galvanised/morphology.json` is versioned `2026-09-23.authoring-4`. It retains six, eight, X, X+C, two and four-arm directional templates, with family probabilities assigned by tilt quantiles rather than measured transition angles. Thin unequal primary trunks support attached secondary and tertiary branches with compact C1 envelopes. All overlapping translated supports contribute, avoiding discontinuities where the nearest nucleus image changes. The grain-local relief tapers to zero at true inter-ID boundaries before the shared groove is added. The detailed construction, rejected experiment and measured comparison are recorded in [galvanised-morphology.md](galvanised-morphology.md).

Doubled-angle branch directions retain their bounded coherence rather than turning vanishing support into a strong arbitrary axis. The sampler combines this descriptor with the coherent intrinsic crystal axis (weights 0.34 and 0.66). Crystal tilt modulates intrinsic roughness between 0.76 and 1.24 times the requested roughness; fine branches add a restrained local change. These are authored unresolved slope distributions, not measured crystal optics. No colour variation is used to simulate grain reflections. The regular relief defaults are 1.0 µm for dendrites and 2.0 µm for trunks, and default anisotropy is 0.70.

The weak micro-height field consists of twelve fixed-phase periodic Fourier waves with nominal physical wavelengths 0.25–0.9 mm. Integer tile wave vectors make the field periodic; finite-tile rounding approximates the requested physical wavelengths. The scale gives an ensemble RMS of half `micro_relief_um` without raster-dependent normalisation. This replaces the earlier sheet-wide crossed-sine checkerboard. Explicit filtering of the complete procedural spectrum remains unfinished. Higher-rate quadrature and independent grids detect some unresolved footprints, but do not establish a finite-band certificate.

The earlier authoring-4 seed-42, 24 × 18 mm, 192 × 144 diagnostic had height RMS 0.349 µm, a −1.316 to +1.090 µm range and normal XY RMS 0.00230. Those measurements precede the release relief changes and are retained as historical prototype evidence, not a fitted industrial envelope. Fine branch patterns remain visible under revealing lights and close crops. Warp remains disabled.

Environmental coefficients are fixed periodic fields in a weather-only namespace. Broad moisture variations combine with twelve weak Fourier directions at nominal physical wavelengths of 3–8 mm. Integer tile frequencies preserve periodicity. A positive atmospheric rate lets ordinary patina spread over the sheet, while a moisture threshold confines white deposits to wetter regions. `Cp=1−exp(−k·exposure)` is monotone and `Cw` is a wetness/confinement/white-stain-controlled subset. Visible fractions are `(1−Cp, Cp−Cw, Cw)`. Deposits share these masks and contribute no height where coverage is absent. The fresh zinc intrinsic roughness/anisotropy and substrate are unchanged by weather presets; the visible compatibility maps mix separate neutral-grey deposit optics. Dross and drainage are sparse physical object records with Poisson area counts, compact radius/height/path and gravity, experimental outside the initial sheet presets.

## Population fit

`tools/galvanised/calibrate_population.py` ran 12 PCG64 seeds and seven growth-spread candidates using exact periodic cell areas, well inside the ≤32-candidate gate. At 100 × 100 mm and target 8 mm diameter, growth sigma 0.300 gave realised mean diameter CV 0.2020 (between-tile SD 0.0110); sigma 1.285 gave 0.2776 (SD 0.0162). The fitted mapping `2026-09-23.area12-v1` uses those points for target CV 0.20 and 0.28. The full result and individual tile CVs are in `data/galvanised/population_calibration.json`, also packaged for replay. Other target CVs, diameters, placement modes and tile populations are explicitly marked extrapolated; metadata reports each tile's realised CV, numerical uncertainty and realised-minus-target difference. This is an exact-area authoring fit, not a measured industrial population distribution.

## Checks and remaining limits

`tests/test_galvanised_surface.py` covers seed/key equivalence, arbitrary-period point sampling, branch periodisation, nested/monotone weather, saturated exposure, invariant substrate, exact map chunk independence, normals from exported height, selected maps, rich material-weight closure, mapped records, target CV effect, low working budgets, quality limits, coarse RGB previews and source hashes. Independent rectangular Fourier fixtures check integration and phase; a 32-cycle-per-pixel signal checks that reference sampling rejects false agreement between dyadic grids. Rich records have the same per-pixel values across chunks, although their storage order follows tile traversal. The API also rejects invalid map selectors and chunk sizes before state construction. Temporary mappings are closed before rename/unlink, including on the successful compacted-lobe path.

The sampler optionally retains a bounded subpixel cache, capped at 384 MiB and included in the working-memory estimate. If it does not fit, the lobe stage resamples the exact same deterministic fields. The measured 512² case improved from 19.59 to 8.03 seconds with byte-identical maps and records on the morphology revision then in use. This is an isolated optimisation comparison, not current end-to-end throughput. Material validation now uses bounded spatial/record blocks and disk-backed reduction buckets instead of whole-image float64 scratch.

The current four-lobe quadrature is not an offline constrained angular fit. Reference comparisons and their limits are in [galvanised-lod.md](galvanised-lod.md). Distant mip roughness is not calibrated to lost height frequencies. Weather controls are a phenomenological appearance model, not a corrosion clock. Morphology, fine-frequency sampling and full resource acceptance remain open in the [implementation record](galvanised-implementation.md).

## Reference spatial integration

The reference map path is bounded to 4096 pixels per call and 65,536 procedural
points per batch. It retains all field associations during each shared quadrature
evaluation. Per-pixel acceptance checks final/substrate/deposit height to 0.005 µm,
visible fractions to 0.001, and compatibility roughness and doubled-angle
anisotropy vectors to 0.002. Both coarse-versus-fine and fine-versus-independent
grid differences must pass. These are fixed diagnostic tolerances, independent
of chunk boundaries and requested channels. They are not specimen calibration
or a guarantee against every possible alias. The final accepted height is
quantised once, then supplies the exported normal.

Metadata records the number of pixels accepted at each rate, tolerances and the
explicitly uncertified finite-band status. Nonconvergence raises an error and
closes/removes the current call's mapped arrays. Single-lobe output remains an
angular approximation even when its spatial integration passes.

Compatibility anisotropy includes zinc coverage exactly once. The point field
already carries that factor, so footprint averaging no longer multiplies it
by the zinc fraction a second time. The generator-version increment makes this
and the sampling changes explicit to export/replay.
