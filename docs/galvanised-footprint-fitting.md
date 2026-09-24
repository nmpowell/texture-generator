# Offline fitting of real physical footprints

`core/footprint_fitting.py` connects the current continuous galvanised state,
the converged physical normal evaluator, and the offline candidate-lobe fitter.
It extracts explicitly selected rectangular footprints and retains actual
sampled lobe records. It does not change the production sampler or enable a
rich `reference` quality tier.

The supported response domain is the fixed **normal outgoing view** coupon
described below. Passing means the declared spatial checks, angular response
fit, and float32 record checks passed on those sample sets. It does not certify
grazing views, highlight width, a finite frequency tail, or arbitrary lighting.

## Supported work and response scale

- A check accepts 1–16 explicitly selected footprints. Each rectangle defaults
  to 0.125 × 0.125 mm and must be no larger than one state tile on either axis.
  Centres wrap periodically into the tile. A rectangle may cross its edge.
- Each material gets at most 64 actual candidates from the same 8 × 8 midpoint
  locations. A material's candidate is omitted only when its point coverage
  is exactly zero. Candidate ordinals retain the physical point ordinal and
  position, local material coverage, normal, tangent, intrinsic widths and
  optical table index.
- Normals use the renderer's state-dependent physical derivative step and
  pointwise convergence check. The first accepted finer estimate is frozen
  per point at a `1e-5` component tolerance, with at most eight halvings;
  unresolved derivatives reject the footprint.
- The response is linear RGB **BSDF multiplied by positive local incident
  cosine**, with unit incident radiance, normal outgoing view and no exposure
  or environment multiplier. Float thresholds refer to this named scale.
- Training uses polar angles `[0.04, 0.12, 0.25, 0.50]` radians and 16 azimuths
  with zero phase. Held-out angles are `[0.07, 0.18, 0.40, 0.80]` and 19 azimuths
  with phase 0.37. These grids are disjoint and contain 192 and 228 scalar RGB
  samples respectively. Their version is `normal-view-disjoint-coupon-1`.

The one-tile limit excludes a reproduced alias: a rectangle spanning 544 tile
periods put the 8, 16 and 17 grids at the same periodic point and falsely
appeared converged. The bounded domain and independent grid improve the
diagnostic; they do not constitute an all-frequency alias proof.

## Target extraction and acceptance

The 8 × 8 responses supply candidates and the initial coarse reference. The
adapter integrates the same state at 16 and 17 samples per axis. It accepts
the 16-grid target only if it agrees with both the 8-grid and independent
17-grid targets. Otherwise it tries 32, comparing against both 16 and 33.
There is no further refinement or silent fallback.

For each comparison, every material **and the combined response** must pass
both training and held-out angular sets:

| Spatial convergence measurement | Limit |
| --- | ---: |
| Relative RMS where reference response ≥ 0.02 | 0.01 |
| Absolute RMS on darker samples | 0.0002 |
| Maximum material-coverage component change | 0.001 |

Angular samples have equal weight, with each lit/dark group normalized by its
own count. The finer target is the comparison denominator. These are empirical
agreement tests on the specified grids. A maximum of 2722 physical locations
is evaluated per footprint across rates 8, 16, 17, 32 and 33, excluding the
additional height evaluations used by derivative convergence.

Geometry and BSDF work is batched at at most 64 physical points. The largest
current BSDF response batch is 64 × 228 scalar values, and the response helper
also enforces a 4096-scalar-per-point ceiling. It never allocates a tensor of
all reference points by all angular samples. Only candidate response matrices,
aggregate targets and diagnostics are retained. Compensated sums follow fixed
physical-point order, so changing point batch size leaves extraction unchanged.
The live regression compares batches of 7 and 64 exactly.

An accepted reference coverage fixes each material's weight sum in the generic
fitter. Its deterministic training-only search retains at most four actual
candidates per material. Held-out targets do not choose the support. A
positive reference coverage with no 8 × 8 candidate rejects explicitly.
The fitter's separate acceptance limits are 5% relative RMS and 0.001 dark
absolute RMS, at the same 0.02 floor; spatial extraction has the stricter limits
above. See [offline lobe fitting](galvanised-fitting.md) for its known local
search failures.

Selected frames, widths, material IDs and optical indices are copied into
`LOBE_DTYPE`; they are not averaged. Stored float32 weights adjust the largest
weight for coverage closure within `1e-7` absolute per material. The canonical
lobe-field validator checks the stored frames and identities. Both angular
sets are then rerendered from the **stored records**, including the renderer's
tangent projection, and checked per material and in total against the accepted
dense targets. Total stored coverage must close within `1e-6`. Storage does
not bypass the fitting gate.

## API and failure policy

```python
from texture_generators.core.footprint_fitting import (
    PhysicalFootprint,
    PhysicalFootprintFitError,
    fit_physical_footprint,
)

try:
    result = fit_physical_footprint(
        state, PhysicalFootprint(center_mm=(2.0, 2.0), size_mm=(0.125, 0.125))
    )
except PhysicalFootprintFitError as error:
    rejected = error.result
    # rejected.report and rejected.arrays preserve the completed diagnostics.
    # rejected.records is empty; it is not an accepted material approximation.
else:
    records = result.records  # One footprint, pixel_index=0; at most 12 records.
```

`PhysicalFootprintFitError` identifies derivative, spatial, absent-candidate,
fitting or storage rejection. Invalid inputs fail before extraction.
`check_physical_footprints(state, footprints)` collects at most 16 outcomes,
including rejected diagnostics; callers must inspect each `accepted` value.
Arrays and returned records are read-only. The optional `point_batch_size`
argument accepts 1–64 and changes work partitioning only.

## Provenance and runnable evidence

Reports contain the material key, seed, complete resolved configuration,
rectangle, angle policy, derivative step, integration comparisons, candidate
fits, storage errors and adapter/fitter/generator versions. Source hashes
include the response implementation and state-construction dependencies.
Optical, renderer and surface resource hashes are recorded separately.

Source identity and captured-state identity are distinct. An existing immutable
state may have deliberately replaced records. The report therefore fingerprints
the actual partition, dendrite, weather, micro-spectrum and defect arrays,
including dtype and shape, plus scalar record metadata and morphology/population
versions. Contiguous arrays feed SHA256 through a memoryview; strided records
use bounded row blocks. It does not duplicate an entire large state array.
The regression changes an immutable micro-spectrum phase while retaining the
same key/config and confirms that `state_sha256` changes.

Run the checker from a source checkout with development dependencies:

```sh
uv run python tools/galvanised/check_physical_footprints.py \
  --seed 42 --size-mm 8 6 --diameter-mm 3 \
  --center-mm 5.1875 5.4375 --center-mm 2.0625 2.4375 \
  --output physical-regular-new

uv run python tools/galvanised/check_physical_footprints.py \
  --preset wet_storage --center-mm 0.0625 0.8125 \
  --center-mm 6.6875 0.4375 --output physical-wet-new
```

The output directory must be new. Each footprint writes an NPZ archive of its
candidate records/responses, reference targets and final records, plus a JSON
report updated after each coupon. Any rejected coupon makes the command exit
with status 1 while preserving the evidence. For example, the recorded regular
centre `(2,2)` with `--footprint-mm 1 1` rejects spatial convergence.

The full authoring-4 evidence, including NPZ archives, was written to a
scratch directory and is not retained. The compact current reports are archived for
[regular coupons](../data/galvanised/validation/2026-09-23/continuation/physical-footprints-regular.json),
[wet-storage coupons](../data/galvanised/validation/2026-09-23/continuation/physical-footprints-wet-storage.json)
and the [coarse rejection](../data/galvanised/validation/2026-09-23/continuation/physical-footprints-coarse.json).
Centres were explicitly chosen from a bounded 64 × 48 point survey: largest
grain-boundary distance, smallest boundary distance, largest weather gradient,
and maximum white-deposit coverage. They are recorded physical samples, not
specimen calibration.

| Coupon, seed 42, 8 × 6 mm state | Extent | Outcome | Stored held-out total relative RMS | Time |
| --- | --- | --- | ---: | ---: |
| Regular interior `(5.1875,5.4375)` | 0.125 mm | Accepts at 16; 4 zinc records | 1.65% | 0.392 s |
| Regular grain boundary `(2.0625,2.4375)` | 0.125 mm | Accepts at 16; 4 zinc records | 0.254% | 0.251 s |
| Wet deposit boundary `(6.6875,0.4375)` | 0.125 mm | Accepts at 16; 12 records | 0.154% | 0.830 s |
| Wet transition `(0.0625,0.8125)` | 0.125 mm | Accepts at 32; 12 records; maximum 5 derivative halvings | 1.291% | 1.989 s |
| Regular `(2,2)` | 1 mm | Rejects spatial convergence at the work limit | — | 0.773 s |

The accepted wet deposit-boundary coupon has material coverages
`[0.093595, 0.355021, 0.551384]`. The coarse regular coupon's held-out response changes by 2.01% from 16
to 32, exceeding the 1% spatial limit despite its 32/33 comparison being closer.
Timings are observations from concurrent local runs, including response
extraction and fitting but excluding state construction.

A four-halving policy originally rejected the wet transition at normal
component difference `6.52e-5`. Its
[original report](../data/galvanised/validation/2026-09-23/continuation/physical-footprints-wet-storage-four-halvings.json)
is preserved. An explicitly recorded
[scratch eight-halving experiment](../data/galvanised/validation/2026-09-23/continuation/physical-footprints-derivative-refinement-8.json)
showed that one additional refinement was sufficient. The
[point trace](../data/galvanised/validation/2026-09-23/continuation/physical-footprints-derivative-trace.json)
isolates a 33-grid point close to the weather valley clamp at substrate height
zero: its early difference stencils crossed that change of slope. At the fifth
refinement the stencil remained on one smooth branch and the normal change
fell to `1.97e-9`. The current renderer adopts the bounded eight-halving
ceiling with the same `1e-5` tolerance and physical-step policy. Original
[regular](../data/galvanised/validation/2026-09-23/continuation/physical-footprints-regular-four-halvings.json)
and [coarse](../data/galvanised/validation/2026-09-23/continuation/physical-footprints-coarse-four-halvings.json)
reports also remain available.

Eleven adapter tests cover the real candidate/record identity and exact batch
equality, disjoint angles, independent-grid alias rejection, held-out spatial
checks, absent material candidates, derivative failure, stored-record
rerendering, work limits and captured-state provenance. Ruff and mypy pass for
the new implementation and checker. No production quality claim follows from
these bounded coupons.
