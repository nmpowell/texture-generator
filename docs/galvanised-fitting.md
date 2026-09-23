# Offline candidate-lobe fitting

`core/lobe_fitting.py` implements the bounded offline P5 fitting experiment. It
selects at most four actual candidate lobes **per material**, preserves their
frames, widths and optical parameters, and fits only their nonnegative weights.
It is not connected to material sampling or the `draft`, `production` and
`reference` quality selectors. These results do not establish the plan's full
footprint-filtering or visual acceptance gates.

## Input and output contract

`fit_material_lobes(training_responses, training_target,
validation_responses, validation_target, *, material_id, coverage, max_lobes=4,
tolerances=None, training_sample_weights=None, validation_sample_weights=None)`
takes two candidate-response matrices and two target vectors. Each matrix has
shape `(candidate_count, scalar_angular_samples)`; RGB can be flattened into
scalar samples with corresponding weights. Responses must be finite,
nonnegative, linear quantities in a declared, consistent radiometric scale.

Candidate rows represent unit-coverage responses. Targets are the absolute
response contributed by this material over the footprint. Consequently the
returned weights sum to `coverage`, rather than to one, and must not be
multiplied by material coverage again. Fit different materials separately;
retain their association with the sampled slope and local frame when forming
targets. A zero-coverage fit has no selected lobes and must still pass the
zero-response acceptance check.

Both matrices must use the same candidate row identities. The caller must
provide disjoint training and validation angles and generate the response of
each complete candidate, including its material, normal, tangent, intrinsic
widths and optics. The generic array interface cannot verify these physical
conditions or angle disjointness. Sample weights are normalized within each
set; a zero-weight sample does not contribute to acceptance.

The result contains source candidate indices, absolute weights, coverage,
training and held-out errors, candidate/trial counts, tolerances and the fitter
version `candidate-simplex-swap-2`. Integration must copy candidate parameters
by these indices without averaging them. Weights and fitting calculations use
float64. Coverage closure and response acceptance after conversion to the
eventual float32 storage format still need an integration check.

## Selection and acceptance

The training objective is squared error weighted by
`sample_weight / max(target, radiance_floor)**2`. At each of at most four
additions, the search tries every remaining candidate. For each trial support,
it enumerates all nonempty active faces of the coverage simplex (at most 15)
and solves the equality-constrained least-squares problem on that face. Faces
with a nonpositive weight are represented by a smaller face. This solves the
nonnegative weight problem for the proposed support.

The greedy result is refined with at most eight best-improving single-swap
sweeps. Each sweep evaluates replacing each selected candidate with every
unselected candidate. If an active-face solution has fewer than the permitted
lobes, additions are considered too. The best training solution becomes the
next sweep's support. Refinement stops at a numerical-zero training loss, a
single-swap local optimum, or the fixed sweep budget. The result records
`refinement_sweeps`, `max_refinement_sweeps` and `search_termination`; the latter
is `numerical_zero`, `local_optimum`, `sweep_limit`, or `zero_coverage`. These
diagnostics describe the search, independently of response acceptance.

This is a bounded local method, not a global sparse optimum. Numerical zero
uses the same `32 * float64_epsilon**2` loss floor as the tie comparator; it
does not use the held-out response or an early acceptance test.

Candidate iteration order is fixed. Numerically tied fits prefer fewer lobes,
then the lexicographically smaller original candidate indices. The held-out
responses never influence selection. With identical arrays and numerical
libraries the algorithm is repeatable, consumes no random state, and has no
image-wide or batch-dependent decisions. Bitwise equivalence across different
BLAS/LAPACK implementations is not promised.

The default acceptance thresholds are:

| Measurement | Default threshold |
| --- | ---: |
| Weighted relative RMS on targets at least `radiance_floor` | 0.05 |
| Weighted absolute RMS on darker targets | 0.001 |
| `radiance_floor` | 0.02 |

Each group is normalized by its own positive sample-weight total. Empty groups
have zero error. Both training and held-out sets must pass both applicable
thresholds. Overall absolute RMS and maximum lit-sample relative error are
reported as diagnostics, without additional acceptance thresholds. The dark
threshold and floor depend on the declared response scale; they are not
exposure-independent physical constants. Angular response RMS does not itself
establish the plan's highlight-width criterion.

Failure raises `LobeFitError`; its `.result` retains the chosen support and
diagnostics. A caller must handle rejection explicitly. Returning these weights
as an accepted production fit, or silently broadening/averaging the candidates,
would bypass the contract.

## Bounds and evidence

One call permits 1–64 candidates and 1–4096 scalar samples in each angular set,
with 1–4 retained lobes. Two maximum-size float64 response matrices occupy
4 MiB in total; fitting creates additional bounded matrix copies and small
least-squares workspaces. No image-sized cache is allocated by this helper.
For `N` candidates the conservative support-trial bound is
`4*N + 8*4*N`, at most 2304 trials. Each trial has at most 15 faces, whose
equality elimination leaves at most three unknowns. The fitter does not cache
all visited supports or enumerate all four-candidate combinations.
The caller remains responsible for limiting coupon size and total work. The
current execution scope is offline coupons of at most 32×32 footprints, not a
4K/8K throughput claim.

The 20 tests cover exact-candidate reconstruction, roundoff ties, four-weight
coverage closure, deterministic duplicate candidates, held-out rejection,
dark-sample error, sample weighting, zero coverage and input bounds. Actual
renderer responses additionally test orthogonal GGX axes and conductor/
dielectric mixtures with the material-to-slope association swapped. A synthetic
five-peak target verifies explicit failure when four candidates cannot meet
the declared tolerance. New cases cover the repaired 64-candidate fit,
physical and synthetic local minima, held-out independence after swaps, and
reported rejection when a reduced one-sweep test budget is exhausted.

On the development machine, the recorded controlled renderer coupons gave:

| Coupon | Candidates | Outcome | Fit time |
| --- | ---: | --- | ---: |
| Orthogonal zinc axes, 1024 independent repeated fits | 4 per footprint | Retains the two source axes at 0.5 each; held-out relative RMS `1.54e-16` | 0.276 s total |
| Zinc/patina opposite slopes, original and swapped association | 2 per material | Retains the appropriate source slope per material at 0.5 coverage; zero error | Not separately timed |
| Original four-source zinc mixture | 64 | Exact source support after two swaps; held-out relative RMS `2.72e-16` | 0.102 s for one footprint |
| Adversarial four-source zinc mixture | 64 | Rejected at a local optimum: 16.6% training and 22.4% held-out relative RMS | 0.059 s for one footprint |
| Synthetic tetrahedron with exact two-source solution | 6 | Rejected at a local optimum: 7.5% relative RMS | 0.0022 s for one footprint |

The original target uses source indices `[0, 17, 42, 63]` and weights
`[0.15, 0.25, 0.35, 0.25]`. The previous greedy method stopped at
`[9, 42, 48, 63]`, with 22.9% held-out error. The two swaps recover
`[9, 17, 42, 63]` then the exact source support, using 730 total support trials.

The physical adversarial target uses indices `[2, 9, 12, 30]` and weights
`[0.31507729041159366, 0.3215818310722124, 0.15036227833922555,
0.21297860017696849]`. The search is trapped at `[1, 10, 12, 30]` after 490
trials. An exact four-candidate solution exists by construction; rejection
exposes the single-swap search limit. The synthetic fixture similarly requires
a joint replacement to escape four decoys even though two other candidates
give an exact result. Increasing the sweep count cannot escape either local
minimum. These are deliberate failure regressions, not claims of successful
approximation.

Timings exclude generating the responses. The run used macOS 26.6.2 arm64,
Python 3.14.7 and NumPy 2.5.2. The 1024-fit coupon repeats one controlled
footprint without result caching; it is not a spatially varied 32×32 surface.

These matrices use the current renderer's BSDF multiplied by positive incident
cosine, in linear RGB. Training uses polar angles `[0.04, 0.12, 0.25, 0.50]`
radians and 16 azimuths. Held-out data uses disjoint polar angles
`[0.07, 0.18, 0.40, 0.80]` and 19 azimuths with phase 0.37. The outgoing view is
normal. Zinc widths are `(0.16, 0.045)`; patina uses `(0.4096, 0.4096)`. These
controlled response coupons do not sample a complete continuous material
footprint or cover all grazing views.

Local evidence is preserved outside the repository:

- `/tmp/galvanised-fitting-next/report.md`: baseline, search decisions,
  adversarial exploration and integration proposal.
- `/tmp/galvanised-fitting-next/coupons/report.json`: timings, diagnostics,
  angular choices, source hashes, environment and optical resource hashes.
- `/tmp/galvanised-fitting-next/coupons/responses.npz`: actual response matrices,
  targets and fit responses for independent inspection.

The durable runner is `tools/galvanised/check_lobe_fitting.py`. It imports the
controlled fixtures from `tests/test_galvanised_lobe_fitting.py` and needs the
source checkout's development environment:

```sh
.venv/bin/python tools/galvanised/check_lobe_fitting.py \
  --output /tmp/galvanised-fitting-new
```

It requires a new output directory and writes `report.json` and `responses.npz`.
The original small report is archived in `data/galvanised/validation/2026-09-23`.
The latest targeted run passed all 20 tests. Ruff passed for implementation,
tests and checker; mypy passed for implementation and checker. The updated
runner reproduced the repaired original 64-candidate coupon and both remaining
adversarial rejections.

## Next bounded step

The smallest realistic integration boundary is an **offline single-footprint
response adapter**, initially run on at most 16 selected footprints spanning a
grain interior, a grain boundary and deposit transitions. Keep the production
sampler unchanged while establishing this adapter:

1. Build at most 64 actual candidates per material on an 8×8 physical midpoint
   grid. Use the current pointwise-converged normal evaluator and each point's
   intrinsic zinc widths, local tangent and material optics. Remove only
   candidates whose material coverage is exactly zero; do not average frames.
2. Generate the material target and fixed coverage from independent, spatially
   converged quadrature, initially 16/17 per axis with a bounded 32/33 refinement.
   The independent odd grid is an alias check. If material appears in the
   target but no candidate exists, or spatial/derivative convergence fails,
   reject that footprint explicitly.
3. First report the existing normal-view training/held-out response domain.
   This exercises actual surface samples without claiming broader view
   acceptance. Run the generic fitter separately per material and retain
   physical point ordinals alongside the result.
4. Copy retained candidate fields to the intended lobe dtype, close the stored
   material coverage, and rerender both angular sets after conversion. Reject
   if storage changes the accepted response. Preserve rejected dense targets
   as evidence rather than returning them as fitted maps.

This boundary is proposed, not implemented here. The current failure cases
also justify evaluating a fixed-width beam or bounded two-candidate swaps
offline if a wider fitting domain is needed; more single-swap sweeps alone are
insufficient. Exhaustive four-element search over 64 candidates is not the
current production proposal.

Before sampler integration, evidence must additionally cover jointly sampled
real footprints with a converged physical derivative and spatial quadrature,
multiple disjoint outgoing views and grazing angles, highlight profiles,
serialization/rerender error, and a declared finite domain with an explicit
failure policy. Any candidate/response caching must be keyed by immutable
physical state and the declared angular grid; batching must leave candidate
order, responses and pointwise convergence decisions unchanged. The broader
working design is recorded in `/tmp/galvanised-quality-design.md`. No quality
tier is enabled by this offline experiment.
