# Galvanised rendering progress

Status: numerical renderer implemented and tested on controlled coupons.
Material-conditioned angular fitting and specimen calibration remain open.

The renderer consumes `MaterialMaps` and `PreviewConfig`. The linear output is
unclipped float32 radiance. Display output applies a fixed exposure and tone
curve followed by sRGB encoding. Rich lobe records preserve the association
between material, normal, tangent and optical parameters. The single-lobe
representation is explicitly approximate for unresolved mixtures.

The conductor uses the pinned D65 zinc Fresnel LUT in its single-scattering
term. The additional separable missing-energy term follows the directional
albedo construction in Kulla and Conty, *Revisiting Physically Based Shading at
Imageworks* (SIGGRAPH 2017), section on energy-preserving specular reflection.
The table is generated offline by
`tools/galvanised/build_energy_tables.py` using 8192 deterministic GGX
visible-normal samples per grid direction. That sampler follows Heitz,
*Sampling the GGX Distribution of Visible Normals* (JCGT 7(4), 2018).
Both tangent and bitangent widths, view cosine and azimuth
are tabulated. The source's `Favg*Eavg/(1-Favg*(1-Eavg))` factor, evaluated per
RGB channel with a hemispherical zinc Fresnel average, attenuates the
returning-bounce lobe; the result is reciprocal and is an approximation to
full multiple scattering, not an exact Smith random-walk evaluation.

An independent 96×192 Gauss–Legendre/equispaced hemisphere integration of
five white-conductor coupons measured directional albedo from 0.9978 to
1.0033 across tested anisotropic widths and views. A 50-case random sweep over
widths 0.05–1.4, view cosines 0.1–1 and azimuths 0–90° measured
0.9973–1.0042. The deliberately sampled domain passes the 1% white-furnace
and 1.005 upper-energy targets. These are numerical checks, not a proof over
every possible width/view combination. The independent quadrature does not
resolve arbitrarily sharp mirror lobes below width 0.05; no gate is claimed
for those extremes yet.

The Schlick dielectric table also records its Fresnel fifth-power term. Its
diffuse BRDF uses the reciprocal directional residual, so a white diffuse
deposit has a unit furnace in the measured domain. An independently
integrated sharp grazing coupon at widths 0.1/0.1, view cosine 0.05 and IOR
1.5 measured 1.00054 with 384×768 quadrature. A 96×192 quadrature falsely
reported 1.099 for that coupon because it missed the narrow specular peak.

The dielectric patina and white-stain lobes use Schlick Fresnel at their
declared IOR and diffuse colour scaled by residual specular capacity. Their
parameters are authoring choices, not measured optical constants. Zinc and
deposit lobes are mixed once by absolute visible fractions. Rich records carry
each material's local normal, tangent and widths. `single_lobe` uses map-level
normal/roughness/axis with coverage weights and is a labelled approximation.

Rigs use cached directional albedo for the constant environment and cached
first angular moments for the overcast environment; Gaussian distant emitters
use deterministic Hermite quadrature with the gnomonic solid-angle Jacobian.
`studio` has a broad strip, `oblique` a small source, `overcast` has radiance
`0.6 + 0.4*direction_world_z` over the sphere (a ground-bounce floor of 0.2),
and `grazing` has a low-angle inspection source. Radiance is integrated before
display conversion. The display transform applies `2**exposure_stops`, then the fixed extended
Reinhard curve `x*(1+x/16)/(1+x)`, then sRGB encoding. Linear output is
unclipped float32 HDR. The default camera has a fixed 24° polar tilt at 135°
azimuth, slightly off the studio source's mirror direction; `(0, 0, 1)` remains
available for normal-incidence checks. `normal_strength` is a preview-only
artistic adjustment. When maps record optical or renderer resource hashes,
rendering verifies them against the installed tables before shading.

`render_reference` samples `sample_points` on a deterministic physical
subpixel grid. Each point derives its own normal from periodic physical height
differences before BRDF evaluation; no raster map is upsampled. The default
derivative step is the smaller of 0.001 mm, 0.45 times the narrowest branch
width divided by 32, the shortest branch carrier period divided by 64, and
the shortest realised micro-spectrum wavelength divided by 64; narrower
grain-boundary grooves, compact defects or weather harmonics also constrain
the step. It depends on
the immutable physical state, not output resolution or `samples_per_axis`.
Callers can pass independent finite positive X/Y steps through
`derivative_step_mm`. Every point compares normal components at epsilon and
epsilon/2, freezes its first passing finer estimate at a maximum difference
of 1e-5, and fails explicitly if eight halvings do not establish convergence.
The pointwise rule is independent of neighbouring points and the bounded 2D
work tiles. Optional `derivative_diagnostics` records the step, refinement
counts, accepted gradient/angle differences and RMS angle; normal angles use
`atan2` of cross and dot products. Each subpixel's radiance stays float64
through accumulation, with one final float32 cast. The number of subpixels is
independent of light quadrature and material key. The finite grid is a
numerical reference, so spatial convergence still requires increasing
`samples_per_axis` at the same derivative policy. The current rich
fitter uses four shared subpixel components per visible material. Two small
historical error cases are recorded in [the LOD report](galvanised-lod.md);
they predate this derivative policy and recent morphology changes and do not
establish a current angular error bound or specimen calibration.

The ceiling increased from four to eight after a real wet-storage footprint
revealed one point close to the weather valley clamp at substrate height zero.
Its first difference stencils crossed the change of slope; one additional
halving established convergence without changing the physical step or `1e-5`
tolerance. The regression compares the accepted normal with a still finer
estimate and checks the complete footprint fit. Stronger unresolved curvature
and coordinate-precision fixtures retain explicit failure coverage. See the
[physical-footprint evidence](galvanised-footprint-fitting.md) for the original
rejection, diagnosis and current bounded result.

The current derivative-ceiling validation run passed all 58 tests across
`test_galvanised_render.py`, `test_galvanised_optics.py`,
`test_galvanised_footprint.py` and `test_galvanised_footprint_fitting.py`. Ruff
and mypy passed for the changed runtime modules and checker.

`PYTHONPATH=src .venv/bin/pytest -q tests/test_galvanised_render.py
tests/test_galvanised_optics.py` passed 34 tests; Ruff passed on the renderer,
builder, relight tool and test; mypy passed on the two runtime renderer modules.
The five-rig relight smoke at 8×8 wrote PNGs, linear arrays and a manifest.
A warm synthetic 128² zinc coupon took 0.54 s for one default linear render on
this developer machine. In a 256² mixed, rich synthetic coupon with 262,144
lobes, caching each lobe's outgoing directional albedo across the light loop
reduced a warm linear render from 26.0 to 12.9 s. A 256² regular-sheet rich
sample at the same lobe count fell from 15.8 to 7.5 s. Both outputs matched
their pre-change float32 linear arrays exactly. Evaluating light directions in
four-wide NumPy blocks did not improve these timings (13.0 and 7.5 s), so that
change was discarded. The provisional 512² full-preview ≤2 s budget remains
unmet; these are renderer-only timings on one developer machine.

Rich records are converted to float64 only within each 4096-record render
block. The former full-record expansion allocated 96 bytes per lobe for eight
converted fields, including an unused material-ID copy: 96 MiB for 1,048,576
lobes. Current per-block converted fields use at most 88×4096 bytes (0.34
MiB), independent of image dimensions; the float64 output image and BRDF
temporaries are additional. A strided, read-only memmap with a pixel crossing
the block boundary rendered byte identically to the full-conversion path.
Saved 256² rich synthetic and regular-sheet renders likewise matched their
pre-change linear float32 arrays exactly (maximum absolute difference zero).
Warm renderer-only times for those 262,144-lobe samples were 10.9 and 8.1 s
after chunking, versus 11.3 and 8.0 s immediately before; the small timing
differences are within run-to-run variation.

Sources: [Kulla–Conty SIGGRAPH course slides](https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf),
[Heitz visible-normal sampling paper](https://jcgt.org/published/0007/04/01/paper.pdf).
