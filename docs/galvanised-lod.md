# Galvanised level-of-detail measurements

Status: version-3 checker results below include one current production
height-only sheet and one tiny rich-render smoke. The older full measurements
are historical; they predate the current morphology, default camera,
derivative policy and spatial quadrature. No current global angular accuracy
target or width-error bound has been established.

## Reproducible checker

`tools/galvanised/check_lod.py` (report version 3) builds **one immutable material state per run**.
It records the concrete material key, resolved config, generator version,
optical/surface/energy table hashes, Python/NumPy/platform identity and stage
timings in a JSON report. Its default limits are one million height pixels and
108 render pixels; larger runs require explicit `--max-height-pixels` or
`--max-render-pixels`. `--strict` writes the report and exits 1 if a defined
height or normal target, independent sampler crosscheck, or derivative
convergence check fails. Render accuracy and spatial reference
convergence have no acceptance threshold; the JSON says so explicitly.

The independent height checker uses the resolved production quadrature from
the immutable config: 4×4 midpoint points for `single_lobe` height-only runs
and 2×2 for rich render comparisons. The latter retains offsets 0.25 and
0.75. `height_sampling.quadrature_per_axis` records the actual rate. The
checker samples physical points directly in bounded, phase-aligned 2D tiles
and crosschecks a small production height map; it does not use the sampled
map as its own LOD reference. Adjacent dimensions must have a common
integer scale factor. A high-resolution height raster is averaged in exact,
phase-aligned blocks; the low raster is sampled independently from the same
state. There is no image resizing or per-image height normalisation. Both
the block-mean height and direct low height get fresh periodic physical
central-difference normals at the **low** resolution, using independent X/Y
spacings. The report gives RMS height difference in µm, its fraction of the
declared 5 µm default relief budget, and RMS angular difference in degrees.
Provisional strict targets are height RMS <0.1 µm and normal RMS <0.1°.

For rendering, the checker samples one small rich map and constructs a
`single_lobe` view of exactly those ordinary arrays. Each studio, oblique,
overcast and grazing render is compared in unclipped linear radiance with
fresh `render_reference` integrations at 4×4 and 8×8 physical subpixels.
Both rates use a state-derived physical derivative step independent of raster
size or rate. The checker records the initial step, pointwise epsilon-versus-half
convergence statistics and any unresolved point failure separately for each rig
and rate under `reference_derivative`. It accepts an explicit X/Y step with
`--derivative-step-mm EX EY`. A failed derivative test marks that reference
unavailable, sets `derivative_convergence_failed`, and causes `--strict` to
exit 1. Reference 4×4 versus 8×8 is a separate spatial self-convergence
result. Relative RMS
uses only RGB channels whose 8×8 reference radiance is at least 0.02 and
divides each channel error by that reference channel. Dark channels are
reported with absolute RMS; if none exist, the metric is `null` and the count
is zero. This avoids calling a large percentage error on near-black pixels a
surface failure while keeping their absolute error visible.

Example:

```sh
PYTHONPATH=src .venv/bin/python tools/galvanised/check_lod.py \
  --seed 4 --preset regular --size-mm 24 18 \
  --resolutions 32x24 64x48 128x96 --render-size 8x6 \
  --output /tmp/galvanised-lod-regular-seed4.json
```

## Current version-3 examples

The seed-4 regular 24×18 mm production `single_lobe` height-only run used
4×4 quadrature at 32×24, 64×48 and 128×96. Independent direct-height versus
`sample_state` crosscheck RMS was 0.0 µm. Phase-aligned height RMS was
0.05945 µm for 32→64 and 0.03775 µm for 64→128; normal RMS was 0.00663°
and 0.00880°. Both pairs passed the provisional 0.1 µm and 0.1° gates on
this one state. It took 1.08 s on the developer machine. Report:
`/tmp/galvanised-lod-v3-height-regular4.json`.

A tiny 3×3→6×6 rich run on the same preset/seed retained 2×2 height
quadrature and crosschecked at 0.0 µm. Its coarse height gate failed. On a
3×3 render, studio and oblique 4×4→8×8 spatial-reference relative radiance
RMS were 2.11% and 4.60%; rich versus 8×8 was 2.95% and 2.74%. Derivative
convergence passed, but these nonzero spatial differences preclude an angular
accuracy claim. Report: `/tmp/galvanised-lod-v3-rich-regular4.json`.

## Historical height results

The historical first two runs used seed 4 regular and seed 5 wet-storage sheet, both
24×18 mm. Rich map height matched the checker's direct midpoint sampler
exactly at the 8×6 render size (RMS 0.0 µm).

| Preset/seed | Low → high | Height RMS (µm) | Normal RMS (°) | Height / normal target |
| --- | --- | ---: | ---: | --- |
| Regular / 4 | 32×24 → 64×48 | 0.2530 | 0.0201 | fail / pass |
| Regular / 4 | 64×48 → 128×96 | 0.1217 | 0.0175 | fail / pass |
| Wet storage / 5 | 32×24 → 64×48 | 0.2279 | 0.0190 | fail / pass |
| Wet storage / 5 | 64×48 → 128×96 | 0.1243 | 0.0169 | fail / pass |
| Regular / 4 | 512×384 → 1024×768 | 0.00490 | 0.00957 | pass / pass |
| Wet storage / 5 | 512×384 → 1024×768 | 0.00428 | 0.00936 | pass / pass |

The 512/1024 checks used the same point surface and midpoint quadrature with
a `single_lobe` height-only state, so they never allocated rich records or
rendered a full bundle. They took 13.0 and 16.7 seconds respectively; the
1024×768 height stage alone took 10.3 and 13.0 seconds. The small complete
checks took 26.3 seconds (regular) and 40.9 seconds (wet storage), mostly in
the 8×8 reference evaluations. These are stage observations, not a broad
throughput benchmark or a 4K claim.

## Historical render gaps

Historical relative linear RGB RMS versus the former 8×8 reference, as a percentage, on the
8×6 raster:

| Preset/seed | Rig | 4×4 reference vs 8×8 | Rich vs 8×8 | Single lobe vs 8×8 |
| --- | --- | ---: | ---: | ---: |
| Regular / 4 | Studio | 0.79% | 1.50% | 2.58% |
| Regular / 4 | Oblique | 1.57% | 2.44% | 3.54% |
| Regular / 4 | Overcast | 0.02% | 0.05% | 0.05% |
| Regular / 4 | Grazing | 0.22% | 0.33% | 0.51% |
| Wet storage / 5 | Studio | 0.48% | 0.94% | 12.84% |
| Wet storage / 5 | Oblique | 0.56% | 1.38% | 17.36% |
| Wet storage / 5 | Overcast | 0.08% | 0.33% | 2.17% |
| Wet storage / 5 | Grazing | 0.08% | 0.25% | 3.22% |

Every reference channel in these tiny coupons exceeded the 0.02 radiance
floor, so absolute dark RMS is `null` with zero dark channels. The rich records
preserved material/slope associations that the single-lobe adapter lost in
wet storage. The 4×4→8×8 difference is nonzero, especially under the oblique
light, and must remain visible when judging the production error. These
measurements do not prove convergence at 8×8 for all footprints or lights.

A historical version-2 smoke on seed 4 regular, 24×18 mm, 3×3 render size and
3×3→6×6 height sizes selected 0.000337528 mm for both derivative axes. All
four rigs at 4×4 and 8×8 passed the pointwise derivative test in one halving,
with no points needing further refinement. The studio and oblique 4×4→8×8
relative radiance RMS were 1.19% and 1.96% respectively, so this smoke does
not establish spatial convergence. Its coarse height target failed. The JSON
report is `/tmp/galvanised-lod-derivative-small.json` on the developer machine;
it is a bounded diagnostic, not an acceptance claim.
