#!/usr/bin/env python3
"""Reproducible topology benchmark; JSON output, no optional dependencies.

Example: .venv/bin/python tools/galvanised/benchmark_grains.py --nuclei 80001
Geometry is opt-in because exact rational all-cell construction is expensive.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from dataclasses import asdict

import numpy as np

from texture_generators.core.grains import GrainPartition, growth_weights, place_grains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuclei", type=int, default=199)
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--boundary-points", type=int, default=0)
    parser.add_argument("--oracle-points", type=int, default=128)
    parser.add_argument(
        "--scenario", choices=("regular", "dominant", "clustered"), default="regular"
    )
    parser.add_argument(
        "--placement", choices=("uniform", "poisson_disc"), default="uniform"
    )
    args = parser.parse_args()
    random = np.random.Generator(np.random.PCG64(args.seed))
    size = (100.0, 100.0)
    diameter = np.sqrt(4 * np.prod(size) / (np.pi * args.nuclei))
    timings = {}
    start = time.perf_counter()
    seeds = place_grains(
        size, float(diameter), random, count=args.nuclei, mode=args.placement
    )
    weights = growth_weights(args.nuclei, float(diameter), random)
    if args.scenario == "clustered":
        seeds = seeds * 0.1
    elif args.scenario == "dominant":
        weights = weights.copy()
        weights[-1] = float(np.sum(np.square(size)))
    timings["placement_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    partition = GrainPartition(seeds, weights, size)
    timings["hierarchy_seconds"] = time.perf_counter() - start
    points = random.random((args.points, 2)) * size
    start = time.perf_counter()
    result = partition.ownership(points, chunk_size=args.chunk_size)
    timings["ownership_seconds"] = time.perf_counter() - start
    check = min(args.oracle_points, len(points))
    start = time.perf_counter()
    oracle = partition.ownership_oracle(points[:check])
    np.testing.assert_array_equal(result.ids[:check], oracle.ids)
    np.testing.assert_array_equal(result.image_offsets[:check], oracle.image_offsets)
    np.testing.assert_array_equal(result.metric_mm2[:check], oracle.metric_mm2)
    timings["oracle_seconds"] = time.perf_counter() - start
    area_summary = None
    if args.geometry:
        start = time.perf_counter()
        stats = partition.area_statistics()
        timings["all_geometry_seconds"] = time.perf_counter() - start
        area_summary = {
            key: value
            for key, value in asdict(stats).items()
            if not isinstance(value, np.ndarray)
        }
    if args.boundary_points:
        start = time.perf_counter()
        distances = partition.boundary_distance(
            points[: args.boundary_points], chunk_size=args.chunk_size
        )
        timings["boundary_first_query_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        np.testing.assert_array_equal(
            distances,
            partition.boundary_distance(
                points[: args.boundary_points], chunk_size=args.chunk_size
            ),
        )
        timings["boundary_warm_seconds"] = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "machine": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                },
                "settings": vars(args),
                "size_mm": size,
                "timings": timings,
                "ownership_stats": asdict(result.stats),
                "geometry_stats": partition.geometry_stats(),
                "area_statistics": area_summary,
                "oracle_points_verified": check,
                "peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
                "notes": "Single process; geometry cache is already warm before boundary queries when --geometry is supplied. Dominant/clustered scenarios are synthetic stress cases. Exact geometry is rational; no raster outputs or renderer included.",
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
