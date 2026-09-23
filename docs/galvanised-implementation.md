# Galvanised metal implementation record

Status: 0.7.0 procedural release candidate; measured-material calibration and
the original full acceptance plan remain open.

The accepted design is `data/plans/2026-09-22-galvanised-metal/implementation-plan.md`
in the sibling data repository, dated 22 September 2026. The supporting research is
`data/llm-output/2026-09/2026-09-22/084026383-b5634651-269b4f34_output.md`.
The baseline commit is `cf455ee3a4904ac7fbece87948f6a16cb0c8f948` (0.6.1).
This record tracks the full design; a working fresh preview does not complete it.

The 0.7.0 release scope is narrower than that design: four visually reviewed
presets, deterministic public image and map paths, independent relighting,
lossless/TIFF export and replay. It does not claim a measured zinc specimen fit,
complete angular fitting, finite-band certification or full 4K/8K rich-export
support. `inconspicuous` and `batch` remain experimental. The open gates below
continue to track the original research ambition rather than blocking this
explicitly scoped procedural approximation.

On 23 September the user clarified that Blender is not part of this task.
No Blender adapter or Blender scene is being implemented. Numerical rendering,
map/export conventions and independent file decoding remain in scope. Current
workers use local Codex agents only, following the user's explicit instruction;
no further external gateway work is authorised by this record.

## Work ownership

| Work | Owner and route | Files | Acceptance |
| --- | --- | --- | --- |
| Completed foundation strands | Prior Sol/Astra workers | material, physical, random_fields, config, optics, grains and fixtures | 97 targeted foundation/compatibility tests passed |
| Surface | Local Sol high | galvanised, dendrites, weather, morphology/population tables and surface tests | G1/G4/G6/G10/G11 |
| Rendering | Local Sol high | material renderer, energy compensation, table builder and render tests | G7/G8/G9 |
| Export and bounded validation | Local Sol high | export, material validator and associated tests | G12/G13 |
| Morphology review and offline lobe fitting | Local Astra xhigh | dendrites, lobe_fitting, fixtures and review records | Measured prototype improvement; full visual and angular acceptance open |
| Integration, evidence and acceptance | Parent | Other files, source register, legacy baseline, subsequent orchestration | Complete matrix below |

Workers have distinct file ownership. Their reports are inputs to review, not
acceptance evidence by themselves. The legacy generator now has only the new
early dispatch and explicit frozen default choices; its old numerical paths are
protected by the pre-change hashes and RNG-state fixtures.

## Phase record

| Phase | Required deliverable | Current evidence |
| --- | --- | --- |
| P0 | Source/licence manifest, specimen plan, legacy hashes, machine record | Legacy fixtures and optical source/licence manifests present; `galvanised-sources.md` distinguishes inspected figures from missing measurements |
| P1 | Typed config/state/maps, deterministic keys, coordinates, selected outputs, professional I/O choice | Contracts implemented; float32 TIFF selected, minimum writer independently decoded; selection/memory refinements continuing |
| P2 | Periodic placement, exact power oracle, bounded accelerator, real edge distances, geometric statistics | Exact geometry and 100k ownership comparisons pass; 12-seed population fit produced; cross-scale fit remains provisional |
| P3 | Orientation families, attached branches/Gabor, physical relief, chunked footprint sampling | Attached branches, bounded directional coherence and periodic micro spectrum implemented; camera scan still exposes fan-like morphology |
| P4 | Zinc integration, GGX/energy treatment, independent relighting, image adapter | Sourced zinc LUT, GGX, multiple-scattering table and bounded rich rendering implemented; reference physical derivatives now converge independently of spatial quadrature |
| P5 | Weathering, deposit geometry, disjoint layers, material-conditioned lobe fitting, reference rendering | Four presets, shared-sample reference and bounded offline candidate fitter implemented; fitter is not integrated into production sampling |
| P6 | Public API, frozen legacy choices, CLI, lossless/professional export, replay, notebook/gallery/docs | Python/CLI, lossless/TIFF, replay, preset sweeps, guide and ten notebook/gallery assets implemented; installed-artifact smoke tests cover wheel and rebuilt sdist |
| P7 | Calibration dossier, held-out comparison, 4K/8K measurements, review | Source and review records started; full performance/appearance acceptance open; Blender excluded by user |
| P8 | Inconspicuous and batch with independent references and scale/defect validation | Experimental states/defects exist; not accepted or advertised as calibrated |

## Required evidence ledger

Each row remains open until the named evidence covers its whole scope.

| Gate | Scope | Required evidence | Status |
| --- | --- | --- | --- |
| G1 | Immutable topology and features independent of resolution, chunks, traversal, workers, ageing and lights | State/key replay and byte comparisons | Open |
| G2 | Conservative weighted ownership and stable ties | At least 100k oracle comparisons, distant weights, hidden cells, seams | Passed targeted analytic/adversarial checks |
| G3 | Physical edges and all positive-area cells | Analytic two/three-cell fixtures, occluded-bisector and 2e-8 mm² cell, accelerator/oracle agreement | Passed targeted analytic/adversarial checks |
| G4 | All fields periodic, including self-image morphology transitions | Same-coordinate period translations, one/two-grain and spanning-branch fixtures, derivatives | Open |
| G5 | Physical units and exact exported-height normals | Rectangular ramp/sine fixtures and exported derivative agreement <=1e-5 | Passed analytic, sampling and lossless I/O checks |
| G6 | Pixel-centre phase and filtering | 512/1K/2K/4K height RMS <2% relief budget, normal angle <0.1° target | Open |
| G7 | LOD and material/angular correlation | Shared-sample reference vs rich fit under rigs; slope-swapped mixtures; width/radiance errors; separately measured single-lobe error | Open |
| G8 | Zinc Fresnel and colour | Source nodes, grazing, neutral D65 white, angular RGB error <1e-3 | Passed source-node, dense-angle and independent spectral integration checks |
| G9 | GGX and multiple scattering | Reciprocity, hemisphere/rotation/axis, <=1.005 energy; white furnace within 1% | Open |
| G10 | Micron relief budget | Specimen envelope and >=12 held-out seeds, RMS/band slopes/trunk outliers | Open |
| G11 | Weathering geometry and fractions | Exposure zero, monotonicity, nested coverage, confinement, finite limits, same substrate | Open |
| G12 | Serialization/professional I/O/replay | Exact arrays and rich records, tiny-slope round trip with independent reader, interrupted-export atomicity, version/resource errors | Open |
| G13 | Resource limits | Machine/stage/RSS/candidate data at 512,1K,4K,8K and >=80k nuclei; streaming and selected output equality | Open |
| G14 | Compatibility and distribution | Legacy hashes/RNG/default selections; all tests/lint/typing/build/metadata; installed wheel/sdist resources outside source | Passed local 3.12/3.13/3.14 locked/minimum tests and artifact smoke checks; remote platform CI pending |
| Visual | Four initial presets across seeds/lights/views/crops/meshes/mips | >=12 seeds each, four rigs, front/oblique/grazing, >=2 crops, 3x3 tiles, flat/curved external renders; reference and reviewer record | Open |
| Examples | Repeatable documented usage | Notebook recipe, maps/scale/light/resolution/weathering, concrete seeds and hashes in examples manifest, preset sweeps | Implemented; new cells executed and ten asset hashes verified |
| Blender adapter | Original pinned renderer integration | Blender-specific scene, node and displacement validation | Excluded by the user; data-frame and independent float-I/O checks remain in scope |
| P8 | Fine/batch coverage | >=80k nuclei, sparse periodic dross/runs with gravity, separate alloy approximation and specimen validation | Open |

Warp remains disabled until Jacobian direction/covector/covariance transport,
area integration and groove-width tests pass. Transparent passivation, substrate
breakthrough/rust, edge-aware drips, growth simulation and per-grain anisotropic
metrics are outside the committed initial release, as the design specifies.

## Verification commands

Run the repository's complete checks after integration:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build --no-sources
uv run twine check --strict dist/*
```

Keep the Python 3.12/3.13/3.14 locked/minimum CI matrix. A separate professional
TIFF job now covers minimum/locked writer on Linux/macOS/Windows and Python
3.12/3.14; this is configured CI, not a claim that those remote jobs have run.
Slow numerical tests and large benchmarks need separate registered markers and
archived reports. Do not replace failed legacy hashes with new ones.

## Integration evidence, 23 September

The API, nine standard channels, rich records, relighting, CLI, lossless/TIFF
bundles, replay and ten notebook examples are implemented. Preset authoring
parameters remain uncalibrated. Experimental fine/batch configurations are not
accepted P8 coverage.

The unchanged non-flat RGB preview check exposed the original camera's weak
view of the metal. A fixed 24° camera tilt, slightly off the studio emitter's
reflected direction, passes that check without increasing relief or F0.
Normal-incidence rendering remains explicitly available. All four initial
presets were rendered at 128² over 100 × 100 mm for seeds 0–11 using the same
camera. All 48 images are finite and their normalized per-channel standard
deviations exceed 0.005 (minimum 0.0078193). This is a non-flat regression
check, not visual calibration. The summary and source identities are archived
in `data/galvanised/validation/2026-09-23/default-camera-summary.json`.

The preceding checkpoint passed **697 tests in 283.45 seconds**, using Python
3.14.7, NumPy 2.5.2 and Pillow 12.3.0. It includes wheel and rebuilt-sdist
installation checks outside the source tree. Ruff lint and formatting, mypy
(41 source files), the locked dependency check, wheel/sdist builds and strict
Twine metadata checks also pass locally. The configured remote CI matrix has
not run. Earlier integration failures were corrected: the renderer is
importable at the artifact checkpoint, the camera is fixed, and the editable
installation has refreshed distribution licence metadata.

Subsequent targeted evidence includes:

- 49 generator/API/config tests passed after the camera correction; legacy
  seeded choice, RGB, float and RNG-state fixtures remain intact.
- 22 morphology/surface tests passed after vectorised support-box culling.
  Independent old/new evaluations agree byte-for-byte on 30,000 arbitrary
  periodic points across three seeds and on a 512² field.
- 24 replay/export tests passed after verifying morphology, optical sources,
  optical manifest/F0 and renderer energy-table identities. Missing, changed
  or extra recorded resource identities fail explicitly.
- Rich validation uses bounded blocks and spill buckets. Renderer conversion
  now uses 4096-record blocks. Cached and uncached sampled records agree.
  Temporary mapped work files close before rename or removal on Windows.
- Reference normals use a state-derived physical derivative step, independently
  refined at each point. Changing spatial quadrature no longer changes the
  derivative policy. Bounded offline lobe fitting selects actual candidates
  and rejects failed held-out response fits; production integration remains open.
- The default-view notebook cells execute and their ten image hashes match the
  manifest. Recipe metadata records the camera and all resource identities.

The new [performance record](galvanised-performance.md) preserves stage data at
512², 1024² and selected-map 4096². The 4K sample took 461 seconds while other
work was running; its sampling peak was 389 MiB and process peak after export
645 MiB. This is neither a full rich bundle nor a latency pass. The 8K run was
interrupted during sampling to prioritise the measured optimisation gap.

The latest [morphology review](galvanised-morphology.md) rejects the remaining
fan-like construction. The [LOD record](galvanised-lod.md) separates historical
morphology results from current reference measurements. Raw specimen
profilometry/controlled reflection, a validated finite-band production fitter,
the full visual matrix and large rich-output performance remain necessary for
P7. No production, measured-material or complete-goal sign-off is claimed.

## Continuation: sampling, physical footprints and support search

The generator is now `galvanised-3`. Draft supports the existing shared 2 × 2
samples; single-lobe production uses 4 × 4. Single-lobe reference uses per-pixel
8 → 16 → 32 comparisons with independent 17/33 grids and explicit rejection.
Its 4096-pixel diagnostic cap bounds work. It does not certify a finite band or
the single-lobe angular approximation. Rich reference remains unsupported.
Reference spatial integration passed a real 64 × 48, 3 × 2.25 mm coupon; all
3072 pixels accepted 16 × 16. Analytic Fourier and deliberate alias fixtures
check the method separately.

Independent review found and corrected selector-dependent coverage rounding
and chunk-dependent normal rounding. Metallic closure uses the same float32
fractions for every selection, and halo derivatives retain exact global
spacings. Averaged anisotropy also weights zinc coverage once, fixing its
previous double suppression in weathered pixels. Export and replay import one
shared generator version.

The offline fitter now refines greedy support with bounded swaps. It recovers
the known exact four-source, 64-candidate target in two sweeps (730 trials,
0.102 seconds). A second physical target and a synthetic target expose local
minima; both still reject explicitly. This is improved support search, not a
universal four-lobe fit guarantee.

Morphology authoring-4 aligns attached branches with family directions,
corrects the qualitative elongated X/X+C projections, and bounds sector
impingement. Three fixed-seed comparisons show more grain-interior coverage
and less nucleus concentration. Ordered fan/comb sectors remain. Stronger
single-axis reflection was rejected because it invents a dominant axis for
balanced families; explicit higher-order angular mixtures remain a model gap.

Exact geometry optimisations passed 100,000-point baseline equivalence and
reduced a paired cached-boundary median from 0.3183 to 0.2226 seconds. Stage
reports are archived under `data/galvanised/validation/2026-09-23/continuation`.
Large-output acceptance remains open. The dendrite builder now packs bounded
blocks instead of retaining all Python segment objects: isolated 2,000-grain
process peak fell from 204.0 to 94.5 MB, with unchanged bytes across 27 fixtures.
Full-state memory accounting still needs completion.

The [physical-footprint adapter](galvanised-footprint-fitting.md) now extracts
actual candidates and dense response targets from the same continuous state.
Its offline normal-view coupons use disjoint training and held-out angles,
independent spatial grids, material-conditioned coverage constraints and
rerendering of the final float32 records. Every failed derivative, spatial,
fitting or storage check rejects explicitly. Independent review corrected an
oversized-periodic-footprint alias and added fingerprints of the actual
captured state. This provides a route to testing the fitter on real surface
samples; it does not yet certify outgoing-view or highlight-width coverage,
nor enable production rich-reference sampling.

The wet-transition coupon exposed a derivative stencil crossing the weather
valley clamp at substrate height zero. Raising the bounded refinement ceiling
from four to eight, without changing the physical step or `1e-5` component
tolerance, resolves that point on its fifth refinement. The full coupon then
passes at spatial rate 32 with 12 records and 1.291% held-out total relative
RMS. A real-state regression also compares against a still finer derivative.
The 1 mm regular coupon still rejects on spatial convergence. Original failed
reports and the diagnostic trace are retained alongside the current evidence.

The authoring-4 sweep rerendered all four initial presets at 128² over
100 × 100 mm for seeds 0–11. All 48 decoded previews clear the unchanged
per-channel standard-deviation diagnostic, with a minimum of 0.00781852.
The archive explicitly distinguishes decoded 8-bit PNG finiteness from
unclipped linear-response finiteness. Source hashes were stable during the
sweep; resource identities checked afterwards are labelled as post-run
verification. These results are in `continuation/preview-summary.json` and
`continuation/preview-progress.json`, under the validation directory above.
The ten updated notebook/gallery assets also match their manifest hashes.

Earlier foundation verification covered **732 collected tests**: 682 passed in the main
run and 49 in the packaging/adapter run before the derivative-ceiling change;
the subsequent 58-test rendering/optics/adapter rerun passed, including the new
wet-transition regression and the strengthened unresolved-derivative fixture.
The two earlier runs took 207.41 and 14.10 seconds; the targeted rerun took
16.46 seconds. Ruff lint and formatting (102 Python files), and mypy (43 source
files), pass on the final source. Packaging checks include installed wheel and
rebuilt-sdist resources outside the checkout. Remote CI remains unrun.

## 0.7.0 release validation

The final procedural branch was reviewed across seeds 17, 42 and 73, two
physical extents, two light azimuths, two weather states, a 2 × 2 repeat and
384 → 192 pixel downsampling. The repeat has no visible edge seam; the direct
versus downsampled display RGB mean absolute error was 0.00370. The resulting
standard metal tile, four-preset comparison, relighting examples and contact
sheet were inspected. The full notebook executed all 22 code cells, wrote 61
images, and all image hashes match its 0.7.0 manifest.

On macOS ARM64, the complete suite passed on Python 3.12, 3.13 and 3.14 with
both locked and minimum compatible runtime dependencies: **732 passed, one
optional TIFF test skipped** in each environment. The minimum selections were
Click 8.1.0 throughout; NumPy/Pillow were 2.0.0/10.0.0 on 3.12,
2.1.0/10.4.0 on 3.13 and 2.3.2/11.3.0 on 3.14. The optional TIFF export suite
passed all 15 tests on 3.12 and 3.14 with both locked `tifffile` and its
minimum supported 2025.5.10 release. Ruff lint/format, mypy on 3.12, and
dependency checks passed. The release performance record gives the final
512 × 384 RGB and four-preset rich-map measurements and states the limits.

The 0.7.0 wheel and source archive passed strict Twine checks. Both artifacts
were installed outside the checkout and passed API, CLI, map/export/replay,
sample-gallery and dependency smoke checks on all three supported Python
versions. Remote Ubuntu, macOS and Windows CI remains to be run on the pushed
branch; these local macOS checks do not substitute for that platform matrix.

## Next implementation steps

1. Extend the physical-footprint coupons to disjoint outgoing views and
   highlight-width checks, including remaining spatial and support-search
   failures. Integrate only a measured domain with an explicit failure path.
2. Complete rich quality tiers and footprint frequency control. Preserve the
   joint material/slope samples; do not average them into four arbitrary lobes.
3. Profile and accelerate morphology and boundary work before repeating full
   rich 4K/8K workloads. Include dense complete-state construction in the memory
   policy, separately from the already-measured grain partition alone.
4. Fit the remaining morphology and weathering against measured specimens,
   acquire missing physical evidence, and complete the original visual matrix.
   The procedural release does not close those calibration gates.
