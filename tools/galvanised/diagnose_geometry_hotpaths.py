"""Repeatable geometry hot-path timing and output snapshot.

Run with ``PYTHONPATH=src .venv/bin/python tools/galvanised/diagnose_geometry_hotpaths.py
--output /tmp/galvanised-geometry-next/before.npz``. Timing JSON is placed next to it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from texture_generators.core.grains import GrainPartition, growth_weights, place_grains


def run(output: Path) -> None:
    size = (100.0, 100.0)
    placement = np.random.Generator(np.random.PCG64(42))
    growth = np.random.Generator(np.random.PCG64(43))
    points = place_grains(size, 8.0, placement)
    weights = growth_weights(len(points), 8.0, growth)
    partition = GrainPartition(points, weights, size)
    queries = np.random.Generator(np.random.PCG64(44)).uniform(
        -100.0, 200.0, (65536, 2)
    )
    started = time.perf_counter()
    partition.precompute_geometry()
    geometry_seconds = time.perf_counter() - started
    started = time.perf_counter()
    owner = partition.ownership(queries, chunk_size=2048)
    ownership_seconds = time.perf_counter() - started
    started = time.perf_counter()
    distance = partition.boundary_distance(queries, chunk_size=2048)
    boundary_seconds = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        ids=owner.ids,
        offsets=owner.image_offsets,
        metric=owner.metric_mm2,
        displacement=owner.displacement_mm,
        boundary=distance,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "nuclei": len(points),
                "queries": len(queries),
                "geometry_seconds": geometry_seconds,
                "ownership_seconds": ownership_seconds,
                "boundary_seconds": boundary_seconds,
                "candidate_evaluations": owner.stats.candidate_evaluations,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
