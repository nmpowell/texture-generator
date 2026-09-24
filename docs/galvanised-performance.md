# Galvanised performance measurements

These are development measurements on macOS 26.6.2 ARM64, Python 3.14.7 and
NumPy 2.5.2. They do not meet the design's latency targets. Some runs overlapped
rendering and test work, so wall times may include contention. Peak RSS is
process-wide and includes resident mapped output pages as well as working data.

The reports and launch-source identities are retained in
[`data/galvanised/performance/2026-09-23`](../data/galvanised/performance/2026-09-23).
Every completed case uses regular sheet, seed 42, 100 × 100 mm, 199 nuclei,
128-pixel processing tiles and mapped output. The morphology resource is
`2026-09-23.authoring-3`. The full binary bundles were written to a scratch
directory; they are not retained in the repository or packaged.

| Workload | State | Sampling | Preview | Export | Read/validate | Total | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512², nine maps and rich lobes | 0.75 s | 8.95 s | 28.91 s | 0.69 s | 0.21 s | 39.52 s | 368.34 MiB |
| 1024², nine maps and rich lobes | 0.76 s | 32.77 s | — | 2.22 s | 0.99 s | 36.75 s | 625.58 MiB |
| 4096², height and normal, single-lobe | 0.99 s | 461.37 s | — | 1.15 s | 0.38 s | 463.90 s | 645.00 MiB |

The 512² and 1024² runs store respectively 1,048,576 and 4,194,304 rich records.
The 4K run's sampling peak was 388.94 MiB; export raised the process peak to
645 MiB. This selected-output case does not establish the memory or timing of
a full 4K rich bundle. The 8K selected-output run was stopped during sampling
after the 4K result showed a substantial optimisation gap. Its retained report
is explicitly interrupted, with no completed sampling or total measurement.

The 512 preview was measured with the earlier normal-incidence camera and
before the latest record-conversion memory change. Later selector validation
and temporary-map lifetime changes do not change sampled values. Source
companions identify launch hashes and any observed file changes; these timings
must not be silently attributed to future revisions.

Separate comparisons verified optimisations before integration:

- Combining ownership/boundary queries and retaining a bounded subpixel cache
  reduced a 512² sampling case from 19.59 to 8.03 seconds, with identical map
  and lobe bytes on the then-current morphology revision.
- Reusing outgoing directional albedo across emitter samples roughly halved
  two 256² warm renders, with identical linear float32 output.
- Converting rich records in blocks of 4096 reduced conversion scratch from
  96 MiB for 1,048,576 lobes to 0.34 MiB per block. This excludes the rendered
  image and BRDF work arrays. Strided mapped records and cross-block pixel
  reductions preserve the previous output exactly.

The recorded profile puts most sampling time in compact morphology evaluation,
ownership and physical boundary distance. Larger sample-rate caches are not a
solution: they multiply both memory and field work. Further acceleration must
retain exact support culling, ownership and deterministic field values. The
full 4K/8K rich-map and dense complete-state gates remain open.

The next exact-geometry optimisation reduced cached boundary-query median time
from 0.3183 to 0.2226 seconds (1.43×) for 65,536 points and 199 nuclei, over five
paired runs in one process. It hoists invariant segment geometry, retains squared
minima and reduces node-bound scratch. Comparison with frozen old sources found
bitwise equal ownership fields at 100,000 adversarial points and equal boundary
distances at 10,240 adversarial points. This is a geometry-stage measurement,
not an end-to-end speedup. The reproducible diagnostic is
`tools/galvanised/diagnose_geometry_hotpaths.py`.

The table above predates `galvanised-2`. In that revision, single-lobe production
integration increases from 2 × 2 to 4 × 4 samples; its numerical effort and
latency therefore differ from the historical selected-map workload. Draft
retains 2 × 2 sampling. Rich production currently retains its provisional
2 × 2 sampling.

Run a fresh case with a new output directory:

```sh
uv run python tools/galvanised/benchmark_surface.py \
  --size 512 --preview --out galvanised-512-new
```

Use `--representation single_lobe --map height_um --map normal_ts` to reproduce
the selected-map workload. Report it separately from the rich representation.

The dendrite constructor also packs segment records in bounded blocks instead
of retaining every Python tuple and endpoint array until the final conversion.
At 2,000 nuclei, the isolated process peak fell from 204.0 to 94.5 MB; the
finished array remained 14.95 MB and byte-identical. Construction time rose
from 1.805 to 1.913 seconds. Twenty-seven seed/diameter/aspect fixtures and
the measured 500/2,000-nucleus records matched frozen sources exactly. The
complete state's geometry and other buffers remain outside this measurement.
The reproducible runner is `tools/galvanised/diagnose_dendrite_builder.py`.

## Ordinary RGB generation in the release

The measurements in this section, including its size guidance, are developer
measurements without a retained record in
[`data/galvanised/performance/`](../data/galvanised/performance). The headline
512 × 384 figure was re-measured at about 20.3 s and 236 MiB peak RSS during
release review on 2026-09-24.

An uncontended `generate_array("metal", (512, 384), 42,
"galvanised")` profile on the developer machine took **45.73 s wall / 45.40 s
CPU**, with **220.27 MiB peak RSS**. State construction took 2.31 s, map
sampling 8.21 s and rich relighting 35.13 s under the profiler. These
profiled stages include instrumentation overhead and were measured before the
last appearance adjustments. The bottleneck was 6,912 small BSDF calls across
36 emitter directions and 4,096-record chunks.

The renderer now processes four emitter directions at once and uses
16,384-record chunks. On the *same* 512 × 384 regular-sheet, seed-42 maps from
an intermediate appearance revision, the old 36-direction renderer took **29.87 s**
and the batched renderer took **22.57 s**, measured without the profiler. Their
float32 RGB SHA-256 hashes match exactly. Sampling those maps took **11.28 s**;
the state, sampling and new render therefore took about **35 s** in that case.
Process peak RSS was **229.5 MiB** during the batched comparison. The source
revision matters: do not compare this 11.28 s sampling stage directly against
the earlier 8.21 s profile as an optimisation result.

On the final `galvanised-3` release source, a fresh uncontended
`generate_array("metal", (512, 384), seed=42, variant="galvanised")` call
completed in **20.30 s wall time** at **241.5 MiB peak RSS**. This is the
release's ordinary RGB measurement; the 35 s figure above describes the
intermediate comparison revision, not its final latency.

The identical-map rendering comparison and its PNGs were written to a scratch
directory on the development machine and are not retained.
Using 16 emitter directions reduced the new renderer to 10.47 s, but changed
the default image visibly (display RGB mean 0.587 versus 0.625; RMSE 0.0393).
The release keeps 36 directions. Increasing the light batch above four gave no
consistent further speedup on a 256² case and raised peak RSS sharply.

For interactive previews where the four-lobe angular mixture is unnecessary,
an **explicit** draft single-lobe recipe retains the same surface seed and
lighting controls:

```python
generate_array(
    "metal",
    (512, 384),
    seed=42,
    variant="galvanised",
    galvanised={"representation": "single_lobe", "quality": "draft"},
)
```

That 512 × 384 comparison took 1.43 s to build the state, 7.85 s to sample and
4.66 s to render (**13.94 s total**, 193.0 MiB peak RSS). Its display RGB RMSE
against the rich image was 0.014; fine angular variation is visibly flatter.
Single-lobe production took 26.45 s total and had a similar 0.014 RGB RMSE in
this one intermediate-revision case. On the final release source, an
uncontended draft single-lobe call took **9.95 s** with **201.5 MiB peak RSS**.
The ordinary default and `generate_maps()` remain rich so the same recipe
still produces the same preview through either public route.

### Size and storage guidance

The final `galvanised-3` sampler completed all nine rich channels at 512 × 384,
seed 42 and 100 × 75 mm for every release preset, in separate uncontended
processes:

| Preset | Sample time | Peak RSS | Lobe records |
| --- | ---: | ---: | ---: |
| regular | 6.92 s | 226 MiB | 786,432 |
| minimised | 28.60 s | 259 MiB | 786,432 |
| weathered | 7.24 s | 259 MiB | 1,572,864 |
| wet_storage | 7.43 s | 288 MiB | 2,206,756 |

A separate wet-storage lossless bundle at that size passed load and exact
height/lobe replay: sampling took 7.53 s, the complete sample/export/load/replay
sequence took 16.50 s, the bundle occupied 110.3 MB, and process peak RSS was
689 MiB while original, loaded and replayed records coexisted. Allow at least
1 GiB process headroom for that workflow on this machine; it is guidance, not
a hard memory guarantee.

The tested size envelope is **up to 512 × 384 for ordinary rich RGB,
contact-sheet renders and complete rich map sampling/export** across the four
release presets. The older 1024² rich measurement in the table above predates
both the final generator and its weathered lobe counts, so it does not qualify
1024² as a release-wide supported size. These are measured workload envelopes,
not physical image-size validators or hard memory guarantees.

Larger sizes, including 4096², can still be requested; there is no fixed
maximum image size. Pass `output_dir` for large map exports. A selected
height/normal 4096² export completed in about 464 s at 645 MiB peak RSS, but
on an earlier sampler revision with 2 × 2 single-lobe sampling. Current
production single-lobe sampling is 4 × 4, so that result does not qualify 4K
for this release. Full rich 4K and 8K bundles and RGB previews have **not been
validated or timed** and are outside the tested size envelope.

The code enforces at least 3 pixels on each map axis, a 512 MiB default
*temporary-work* budget, a bounded sample tile and a 32-bit rich-lobe pixel
index. The budget does not include final resident arrays, mapped files,
renderer scratch or operating-system page cache. Keep ordinary RGB/contact-sheet
work within the tested size envelope and run jobs serially on a machine of this
class. For exploratory larger map exports, keep a single job active. No larger
rich size is claimed supported by the current measurements.

A rich lobe record occupies 45 bytes. Four zinc records per pixel alone need
2.81 GiB at 4096² and 11.25 GiB at 8192²; three visible materials can raise
the upper bound to 12 records per pixel, or 8.44 and 33.75 GiB respectively.
Those figures exclude channel arrays and files written during export. The
selected-output 4K and interrupted 8K measurements above cannot establish a
safe full-rich 4K/8K limit. Measure those workloads separately on appropriate
hardware before offering them as a supported release setting.
