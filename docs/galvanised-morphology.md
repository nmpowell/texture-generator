# Galvanised morphology revision

The `2026-09-23.authoring-4` construction reduces crossed branches and isolated leaf outlines by extending finer branches across more of each grain and terminating adjacent growth sectors before they overlap. It remains an authored prototype: ordered fans and comb-like striae are still visible. **The plan's complete visual morphology gate is not passed.**

## Evidence and construction

Kim et al. (2019), [Fig. 9](https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs11661-019-05263-4/MediaObjects/11661_2019_5263_Fig9_HTML.png), depicts fine attached branching, unequal sectors and impingement. [Fig. 15](https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs11661-019-05263-4/MediaObjects/11661_2019_5263_Fig15_HTML.png) and [Fig. 16](https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs11661-019-05263-4/MediaObjects/11661_2019_5263_Fig16_HTML.png) distinguish the projected families. The previous equal-angle X template did not represent the elongated X shown in Fig. 16. The revised X uses two oblique axis pairs, and X+C adds the transverse axis. The chosen 20° projection, family frequencies, branch dimensions and amplitudes are **authoring values**, not digitised measurements or a fit to the figures. Sources, acquired figure hashes and rights are recorded in `docs/galvanised-sources.md` and the external reference manifest.

Each nucleus retains its crystal-orientation stream and per-grain morphology stream. The six, eight, X, X+C, two and four templates remain coherent within a grain. Sidearms follow adjacent projected primary directions with at most 2° jitter. The two-direction family has no lateral primary direction and retains an explicitly authored secondary angle.

The revised growth construction has three geometric constraints:

- A sidearm stops before the angular bisector between its parent and the neighbouring primary arm. This bounds interpenetration of different growth sectors.
- Primary reach is 0.95 nominal grain diameters before the existing competition and length variation. Secondary reach no longer has the same decreasing leaf envelope on every trunk. Grain ownership still clips the visible outer branches.
- Every centreline stays within a 1.15-diameter support circle. Width plus compact lateral support remains below the evaluator's 1.25-diameter periodic-image budget. Tertiary reach is capped at 0.65 secondary spacings to reduce long crossed grids.

Secondary spacing is 0.135 mm, with bounded renewal jitter; primary, secondary and tertiary width caps are 0.045, 0.029 and 0.018 mm before their existing random multipliers. Small spangles additionally limit widths and spacing by their physical diameter. Construction is bounded by 48 secondary attachments per side. These parameters describe an authoring model, not a solidification solver or a universal metallurgical length scale.

The ridge envelope and evaluator are unchanged: smooth root/tip windows, a Gaussian times a compact lateral polynomial, tapered width, and a weak phase-modulated carrier. Their values and first derivatives vanish at compact support. Every overlapping periodic image is summed in stable order; culling removes only exact zero support. Coordinates remain millimetres, independent of lighting, output resolution and chunk size.

The local doubled-angle coherence descriptor remains:

`E = sum(w_i * (cos(2 theta_i), sin(2 theta_i))) / (sum(w_i) + 0.12)`

Here `w_i` is branch amplitude times its compact ridge envelope. Cancellation and vanishing support reduce its magnitude; the descriptor is not normalised back into a strong arbitrary axis. It is an authoring proxy, not a fitted finite-band microfacet covariance. The sampler's treatment of this descriptor is a separate optical-authoring decision.

No F0, physical-height conversion, amplitude control, macroplane or weathering value was increased in this revision. The primary and secondary amplitude coefficients remain 0.26 and 0.34.

## Frozen comparison

`/tmp/galvanised-morphology-next/` contains source snapshots, source hashes, comparison scripts, raw map/branch arrays, measurements and renders. `before-*` uses authoring-3; `after-*` uses the retained authoring-4 implementation. Only the morphology module and resource differ between their frozen package snapshots, so later sampler development does not confound this comparison.

All cases use a 24 × 18 mm tile, nine nuclei, nominal 8 mm spangles, 192 × 144 sampled maps and 512 × 384 branch diagnostics. Three seeds are shown under studio lights at 15° and 105°, grazing light and overcast light, with fixed exposure and physical normal strength.

| Diagnostic | Seed 42 before → after | Seed 91 before → after | Seed 137 before → after |
| --- | ---: | ---: | ---: |
| Branch mean | 0.0500 → 0.0496 | 0.0344 → 0.0388 | 0.0349 → 0.0421 |
| Branch field above 0.03, area | 40.0% → 44.3% | 28.4% → 34.5% | 28.6% → 37.3% |
| Branch mass within 1.6 mm of nuclei | 30.5% → 16.0% | 29.8% → 14.3% | 30.4% → 15.2% |
| Full height standard deviation | 0.466 → 0.349 µm | 0.412 → 0.352 µm | 0.411 → 0.348 µm |
| Full height minimum–maximum | −1.368..2.373 → −1.316..1.090 µm | −1.360..1.747 → −1.343..1.813 µm | −1.449..1.708 → −1.449..1.685 µm |
| Normal XY RMS | 0.00264 → 0.00230 | 0.00232 → 0.00215 | 0.00240 → 0.00217 |
| Segment count | 2,137 → 5,793 | 1,826 → 4,686 | 1,738 → 4,582 |
| State and map sampling time | 0.61 → 0.72 s | 0.51 → 0.67 s | 0.49 → 0.66 s |

These diagnostics show redistribution of relief across grain interiors, with lower concentration at nuclei. They are not specimen-fit scores or performance gates. Timings are individual observations from this machine. The higher object count remains a performance risk for dense populations.

Rejected scratch alternatives are retained with the measurements. Trial 1 changed projected directions and impingement alone but retained large pinwheels. Trial 2 also made most sidearms short; it left sparse fern silhouettes. Trial 4 strengthened the carrier packets and applied them to the direction descriptor; it produced dotted combs and mainly reduced relief. Neither suppression method was accepted as a shape correction. Trial 3 is the retained construction.

## Validation and remaining work

The morphology and surface suites passed together: **24 tests**. Ruff and mypy pass for the morphology module. Tests cover true secondary/tertiary parent attachment, every family, an actually elongated X, coherent projected directions, sector termination, immutability, concentration limits, arbitrary chunk/traversal equality, compact-support C1 limits and cancellation of orthogonal descriptors. The 1.25-diameter support budget is checked at 0.4, 1.5, 8, 16 and 37 mm. Fixed-grid translations had zero measured error; non-grid seam and derivative fixtures retain numerical tolerances.

The revision removes much of the coarse crosshatching and reduces isolated botanical outlines. Fine fans, herringbone sectors and some sparse two-direction grains remain conspicuous in branch diagnostics and studio previews. Three seeds and four lights establish a bounded comparison, not the plan's held-out calibration programme. Fitting projected families, fine-scale topography and grain-wide unresolved reflection still needs independent evidence. No completed P7 or calibrated-appearance claim follows from these changes.

## Historical grain-wide reflection diagnostic

The 0.7.0 procedural release later superseded the rejection below for its
bounded visual scope. The sampler now uses a 0.66 intrinsic crystal-axis and
0.34 local-branch blend, with tilt-dependent roughness and reduced branch
relief. Multi-seed, scale and light previews favoured the resulting spangle
contrast over the previous comb pattern. Balanced-family angular fitting and
specimen-calibrated evidence remain open, so this release choice is an
appearance model rather than a physical validation of a single crystal axis.

A frozen scratch experiment also tested whether the sampler's weak grain-wide response makes individual arms dominate the preview. Geometry, normal, roughness and metallic arrays were asserted bitwise unchanged. All three seeds were rendered under the same four lights. No repository sampler change was made.

Raising the fixed intrinsic crystal-axis contribution from 0.15 to 0.65, while reducing the local branch contribution from 0.85 to 0.35, increases studio-15° between-grain luminance CV from 0.031–0.033 to 0.123–0.148. Ordinary views gain stronger grain patches without height or F0 inflation. This uniformly strong axis is **rejected**, because it imposes a preferred direction on balanced sixfold, eightfold and fourfold families.

A second variant uses the equal-weight doubled-angle mean of each projected family. It retains balanced-family cancellation and improves X/two-rich seeds, but reduces seed-42 between-grain contrast to 0.013: most of that seed's families cancel. A length × width × amplitude weighted mean over each grain's branch records also mostly cancels. It additionally includes support outside visible ownership, so it is only a diagnostic proxy.

These results identify grain-wide unresolved reflection as part of the remaining problem, but do not validate a stronger arbitrary single axis. A balanced family can have higher angular structure even when its doubled-angle mean vanishes. Retaining the family's lobe mixture before rendering/fitting, or adding independently justified grain-wide roughness information, remains a separate modelling task. The source does not provide measured parameters for either. Scratch scripts, metrics and four-column comparison sheets are in `/tmp/galvanised-morphology-next/`.
