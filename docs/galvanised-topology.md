# Galvanised periodic topology

## Status

`core/grains.py` and the private `core/_grain_geometry.py` implement periodic grain placement, weighted ownership, exact cell geometry, boundary distance and area statistics. `tests/test_galvanised_grains.py` covers them in ordinary CI, and `tools/galvanised/benchmark_grains.py` reproduces the performance record below. The module adds no runtime dependencies. Warp is disabled. Preset calibration is outside this module.

## Public API

Import from `texture_generators.core.grains`.

```python
rng = np.random.Generator(np.random.PCG64(seed))
points = place_grains(size_mm, diameter_mm, rng)
weights = growth_weights(len(points), diameter_mm, rng)
partition = GrainPartition(points, weights, size_mm)
result = partition.query(x, y)
```

Use independent semantic RNG streams for placement and growth when integrating with material state. The API requires an explicit `Generator(PCG64(...))`; it never obtains entropy implicitly.

| API | Contract |
| --- | --- |
| `GrainPartition(points_mm, weights_mm2, size_mm)` | Copies and freezes float64 `(N,2)` points, float64 `(N,)` weights, `(Lx,Ly)` lengths and uint32 `nucleus_ids == arange(N)`. Input points are reduced periodically in float64. Original row IDs survive hidden cells. Immutable byte-backed arrays cannot have WRITEABLE re-enabled. |
| `query(x, y, *, chunk_size=2048)` | Broadcast X/Y arrays or scalars; returns `OwnershipResult` of the broadcast shape. |
| `ownership(points, *, chunk_size=2048)` | Same result for arbitrary finite `(...,2)` coordinates, including negative coordinates and multiple periods. |
| `ownership_oracle(points, *, chunk_size=1024)` | Exhaustive all-nucleus reference, in blocks of at most 64 seeds. Isotropic distance permits choosing each seed's exact nearest periodic image before comparing nuclei. Independent tests explicitly enumerate all nine images. |
| `boundary_distance(points, *, chunk_size=2048)` | Float64 distance in mm to actual inter-ID segments, including their periodic translations. Returns infinity when no grain boundary exists. |
| `boundary_distance_oracle(points, *, chunk_size=1024)` | Exhaustive all-cell/all-constraint clipped-segment reference; deliberately slow. |
| `cell_fragments(grain_id)` | Tuple of CCW float64 polygons clipped into the fundamental tile. Every rationally positive fragment is included. |
| `cell_segments(grain_id)` | Immutable `(E,2,2)` float64 segments of the canonical image cell; endpoints can lie outside the tile and require periodic translations. Artificial tile/self-image edges are absent. |
| `cell_neighbours(grain_id)` | Immutable uint32 IDs across each corresponding `cell_segments` edge, preserving both sides' original IDs. |
| `cell_area_exact(grain_id)` | `fractions.Fraction` area in mm², including all periodic fragments of that original nucleus. |
| `areas()` | Immutable float64 `(N,)` areas, hidden IDs retained as zeros. Positive areas that underflow float64 raise explicitly; exact areas remain accessible. |
| `area_statistics(*, small_area_fraction=1e-6)` | `AreaStatistics` described below. Computes every original nucleus geometrically. |
| `adjacency()` | Unique sorted uint32 `(E,2)` inter-ID pairs; sparse storage, no dense adjacency matrix. |
| `precompute_geometry()` | Explicit all-cell cache construction. Normal construction and queries remain lazy. |
| `geometry_stats()` | Cached-cell, candidate-image, visited-node and segment counts. |
| `initial_nucleus_count(size_mm, diameter_mm)` | `max(1, round(4*A/(pi*D**2)))`, with capacity/finite checks. A population estimate, not a median-diameter guarantee. |
| `place_grains(size_mm, diameter_mm, rng, *, count=None, mode="poisson_disc", minimum_distance_mm=None, candidates_per_point=30)` | Immutable float64 positions. Uniform alternative uses `mode="uniform"`. The Poisson-disc separation defaults to `0.35*D`, an authoring choice. |
| `growth_weights(count, diameter_mm, rng, *, growth_sigma=0.35, max_weight_fraction=0.2)` | Immutable bounded zero-mean-gauge weights in mm². Growth spread is not a realised diameter CV. |

`OwnershipResult` contains:

- `ids`: uint32, query shape.
- `displacement_mm`: float64, query shape plus `(2,)`; query minus the winning periodic seed image. `local_mm` is an alias.
- `metric_mm2`: float64, squared displacement minus original weight.
- `image_offsets`: int8, query shape plus `(2,)`, relative to the wrapped query and stored seed.
- `stats`: `QueryStats(points, candidate_evaluations, max_candidates_per_point, node_point_tests, rational_comparisons, peak_candidate_pairs)`, plus a `mean_candidates_per_point` property. Rational counts refer to metric evaluations during ambiguous comparisons.

The state is immutable; its private geometry cache is a performance detail and does not affect results, seed consumption or sampling. Query output arrays are ordinary caller-owned arrays. The implementation has no resolution, chunk-origin, lighting or weather seed. Winning-image local coordinates can jump at a same-ID image cut; morphology must periodise its compact supports rather than assuming those local coordinates are a globally continuous chart.

## Ownership and bounds

The ordering is `(exact power metric, original nucleus ID, image_offset_x, image_offset_y)`. The normal calculation uses float64; ambiguous image/metric comparisons use exact rational arithmetic on the represented inputs. In particular, negative coordinates so close to a seam that `remainder` rounds to the period retain their original coordinate in the rational comparison. Metrics and displacements are returned in float64 and can round even when IDs require an exact distinction.

The hierarchy stores axis-aligned seed bounds and each node's maximum weight. Its lower bound is the minimum toroidal squared distance to that box minus maximum weight. Nodes are excluded only when that lower bound exceeds the current winner's upper threshold. Both are enlarged outwards by a conservative arithmetic error envelope; equality and uncertainty cause expansion. There is no fixed nearest-neighbour count. Per-point depth-first stacks are evaluated together with NumPy, avoiding Python work per pixel.

The guard is `256*eps*(Lx²+Ly²+max(abs(weights)))`, floored at 1,024 float64 subnormal steps. It covers subtraction, periodic translation, squared distances and the polygon-plane float filter within the accepted metric range. Lengths must be at least `sqrt(float64.tiny)`; squared domain norm and weight magnitude must stay below `float64.max/1024`. Invalid and nonfinite inputs fail explicitly. Exact predicates do not permit overflowing public metrics.

The accelerator allocates at most `chunk_size*16` candidate pairs; the oracle uses at most `chunk_size*64`. Traversal stacks use `O(chunk_size*log N)` storage. Wrapping, candidate counters and comparison buffers are chunk-local. Returned arrays necessarily use `O(P)` memory; callers should stream calls if those outputs also need a bound. `query(x,y)` additionally constructs the broadcast coordinate array. No default `P*N` or nine-image population matrix is allocated.

## Geometry and boundary distance

Each canonical nucleus image starts inside its four self-image half-planes, the rectangle `[-Lx/2,Lx/2] x [-Ly/2,Ly/2]` in local coordinates. Rival images contribute ordinary power half-planes. Self-image constraints remain present during all clipping. Intersections, degeneracy signs and polygon areas use rational arithmetic; there is no area epsilon or minimum-area cutoff.

Cell construction has a separate termination proof from point ownership. Over the current polygon, a node's lower metric bound is compared with the owning image's maximum metric at polygon vertices. Convexity makes that maximum a valid upper bound everywhere in the polygon. Excluded nodes cannot cut this polygon or any later subset. A float plane filter skips only strictly inactive constraints outside its error envelope; ambiguous predicates and every actual cut go through the rational path. The exhaustive boundary oracle disables both node exclusion and that plane filter.

After geometry is complete, edge provenance identifies real inter-ID segments. A coincident-plane case uses the strongest outward power derivative, followed by ID, to select the grain on the other side. This removes zero-area intermediate neighbours correctly. Same-ID image edges are discarded only at this stage. Cached geometry keeps exact vertices/area and compact segment/neighbour arrays; redundant rational clipping planes are released.

The canonical image polygon covers one copy of its periodic grain, so its exact area equals the sum of all tile-clipped fragments. Tile clipping happens only when fragments are requested. Distances use the active segments and all nine relevant translations. They are distances to finite segments, never arbitrary infinite bisectors.

A point's closest boundary belongs to its own grain: any path to another grain's boundary crosses the owning grain's boundary first. This permits complete-cell caching without collecting unrelated boundary candidates at every sample. A lower-dimensional tie owner has no positive-area cell and uses the all-cell reference. A single positive-area grain has full torus area and no boundary, even if another ID wins at an isolated exact tie.

Returned fragment/segment coordinates are rounded float64. Extremely small positive geometry can collapse in those display coordinates; the cached rational polygon and `cell_area_exact()` remain authoritative. This is distinct from dropping a cell by area. Warp, world-space warp integration and transported morphology are outside this implementation.

## Statistics and placement

`AreaStatistics` contains full `areas_mm2`/`area_error_mm2` arrays; original `active_ids`; their corresponding `equivalent_diameters_mm`; `median_diameter_mm`, `diameter_cv`, propagated `median_error_mm`/`diameter_cv_error`; `smallest_area_mm2`, `small_cell_count`, `hidden_count`, `statistically_sufficient`, and `total_area_error_mm2`.

Every positive-area cell has equal weight in median and population CV. Areas are constructed exactly, then rounded once, with an outward-ULP error estimate. Diameter and CV errors propagate that rounding and floating arithmetic. Square roots and normalised CV calculations avoid losing positive subnormal areas or overflowing at extreme physical scales. These are numerical uncertainty estimates for a complete finite-tile census, not confidence intervals for a fitted material. The provisional `statistically_sufficient` flag requires at least 12 active cells; it is not a calibration approval. `small_area_fraction` is only a diagnostic threshold relative to the entire tile area and never excludes a cell.

Poisson-disc placement uses bounded uniform sequential inhibition on the torus: every proposal covers the whole domain and is accepted only if it respects periodic minimum separation. At most `candidates_per_point*requested_count` proposals are made in draw order. Sparse bins, sorted neighbour iteration and robust ambiguous separation comparisons provide deterministic, finite behaviour. There is no active-list ordering to vary. Failure raises `PlacementError` with `requested` and `achieved` counts; it does not silently reduce population or separation.

This is a documented variation from the Bridson construction suggested in the design. An initial truncated growing-front implementation left large unsampled regions and was replaced. Research source: Bridson, “Fast Poisson Disk Sampling in Arbitrary Dimensions”, SIGGRAPH 2007 Sketches, DOI 10.1145/1278780.1278807. The runtime algorithm here is uniform sequential inhibition, not a claim to reproduce Bridson's distribution or physical nucleation.

Growth controls are positive lognormal samples `g`; `(g-1)/(g+1)` bounds their influence. Subtracting the sample mean fixes the weight gauge to zero within floating roundoff, and final scaling bounds absolute weights by `max_weight_fraction*D²`. Neither `growth_sigma` nor weight spread is labelled a realised diameter CV. Preset calibration must fit placement and growth against the geometric results.

## Verification record

Ordinary CI tests cover:

- 100,000 oracle/accelerator points over small weighted, clustered and distant-dominant populations, including seams; IDs, image offsets, metrics and displacements agree exactly between the paths.
- An independent rational reference explicitly enumerating all nine periodic images; exact ID and image ties; sub-ULP seam/antipode comparisons using original coordinates.
- Broadcast/scalar/empty queries, arbitrary periodic translations and different chunk sizes.
- Analytic one-, two- and three-cell areas/distances; full-torus single cell; duplicates; zero-area line and isolated-point tie owners; sparse original-ID adjacency.
- Hidden seeds `(0.1,y),(0.7,y)`, weights `0.4,0`: no boundary.
- Seeds `(0,0),(0.5,0.5)`, weights `0.1,-0.1`, query `(0.49,0)`: distance `sqrt(0.2²+0.01²)`.
- Near-hidden weight `0.4999`: positive `2e-8 mm²` cell; one-ULP approach to the hiding threshold; positive subnormal area statistics.
- Metric-runner-up/nearest-boundary disagreement; full-constraint boundary reference; exact area conservation and tile-fragment totals.
- Seeded finite toroidal placement, achieved-count failure, minimum separation, global coverage, bounded growth weights, dtype/immutability and numeric validation.

An additional development probe passed 60 random weighted populations with rational total-area equality, positive-area-only adjacency and accelerated/all-constraint boundary agreement. No slow marker was added; the full grain test file is ordinary CI. An additional 414 accelerated cell areas matched exhaustive all-constraint rational clipping across 12 weighted populations, including thin rectangular tori.

## Performance record

Single-seed development measurements, not a 12-seed median/p95 release measurement. Environment: macOS 26.6.2 arm64, Python 3.14.7, NumPy 2.5.2. One Python process, no child processes or SciPy; NumPy backend thread settings were not pinned. CPU model and RAM were not recorded.

All cases use seed 42, a 100 x 100 mm tile, Poisson-disc placement, default growth settings, chunk size 2,048, 100,000 random ownership points and 10,000 boundary points. All geometry was built before the two boundary timings; those are cached-geometry queries. The first two cases verified 128 sampled points against the oracle; the 80,001 case verified 256.

| Nuclei | Ownership (100k) | All exact geometry/statistics | Cached boundary (10k) | Median diameter | Diameter CV |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 199 | 0.288 s | 0.411 s | 0.053 s | 7.71493 mm | 0.19877 |
| 5,659 | 0.522 s | 13.904 s | 0.079 s | 1.45516 mm | 0.20743 |
| 80,001 (final compact cache) | 0.656 s | 219.337 s | 0.099 s | 0.38665 mm | 0.20853 |

The final 80,001 case averaged 14.975 candidate nuclei per point (maximum 50), cached 480,006 directed boundary segments, used at most 20,480 candidate pairs in one query batch, and reached **335.9 MiB peak process RSS**. Releasing redundant rational planes reduced peak RSS from the initial 651.6 MiB. The benchmark driver prints machine/settings/stage timings, candidate statistics, geometry counts, realised statistics and peak RSS as JSON. The development JSON records were written to a scratch directory and are not retained; this table preserves the durable results.

```sh
uv run python tools/galvanised/benchmark_grains.py \
  --nuclei 80001 --points 100000 --geometry --boundary-points 10000 \
  --oracle-points 256 --placement poisson_disc
```

Two additional synthetic stress runs used 199 nuclei and 100,000 query points. `--scenario dominant` hid 198 nuclei, produced zero boundary segments, and used exactly 13 candidates per point; ownership took 0.646 s and geometry 0.229 s. `--scenario clustered` compressed seeds into 10% of each tile dimension, hid 135 nuclei, and used 15.24 candidates per point on average (maximum 75); ownership took 1.220 s and geometry 2.885 s. Both verified 128 oracle points. These are stress configurations, not manufacturing presets. Their development records were written to a scratch directory and are not retained.

## Completed checks

- `uv run python -m pytest tests/test_galvanised_grains.py -q`: **23 passed in 1.31 s**.
- `uv run ruff check` on `grains.py`, `_grain_geometry.py`, `test_galvanised_grains.py` and `benchmark_grains.py`: passed.
- `uv run ruff format --check` on those four files: passed.
- `uv run python -m mypy` on those four files: passed.
- Analytic/adversarial fixtures, 100k ownership comparison, 60 random weighted geometry probes, 414 exhaustive area comparisons, >80k complete geometry and the two stress benchmarks: passed as described above.
- Architecture, OS, Python and NumPy versions were captured by the benchmark.

## Remaining limits

- Exact all-cell geometry is the dense-state setup bottleneck. Ownership scalability does not establish a fast 80k-cell state build; the measured final setup took 219 seconds. Lazy queries avoid paying for untouched cells.
- The geometry cache is linear in realised vertices/edges and retains exact rational vertices. Full precomputation can use hundreds of MiB. Material integration must account for this state separately from raster outputs and query buffers.
- Pathological overlapping weights or near-degenerate candidates can defeat pruning and trigger many rational comparisons. Correctness takes precedence; no fixed-k shortcut or hidden approximate fallback is used.
- The constructor's stored seed coordinates define its float64 geometry after periodic reduction. Sub-ULP information lost before or during seed conversion cannot be reconstructed. Query comparison retains the original represented coordinates for ambiguous wrapping.
- Requested float64 areas below representable range raise, with exact rational area still available. Tiny fragment display vertices can coalesce while the cell remains geometrically active.
- Presets are uncalibrated. The observed diameter CVs are approximately 0.20–0.21, not the design target of 0.28 for `regular`. Calibration should adjust authored settings using geometric statistics across seeds.
- No rendering, morphology, weathering, warp, 4K/8K export timing or full-feature performance acceptance is claimed here.
