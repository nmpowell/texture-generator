# Galvanised performance measurements

These are development measurements on macOS 26.6.2 ARM64, Python 3.14.7 and
NumPy 2.5.2. They do not meet the plan's latency targets. Some runs overlapped
rendering and test work, so wall times may include contention. Peak RSS is
process-wide and includes resident mapped output pages as well as working data.

The reports and launch-source identities are retained in
[`data/galvanised/performance/2026-09-23`](../data/galvanised/performance/2026-09-23).
Every completed case uses regular sheet, seed 42, 100 × 100 mm, 199 nuclei,
128-pixel processing tiles and mapped output. The morphology resource is
`2026-09-23.authoring-3`. Full binary bundles remain under
`/tmp/galvanised-review/current-*` and are not packaged.

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
.venv/bin/python tools/galvanised/benchmark_surface.py \
  --size 512 --preview --out /tmp/galvanised-512-new
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
