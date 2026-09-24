"""Measure isolated dendrite construction timing and peak process memory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np


def builder_from(path: Path | None):
    if path is None:
        from texture_generators.core.dendrites import build_dendrites

        return build_dendrites
    spec = importlib.util.spec_from_file_location("dendrites_frozen", path)
    if spec is None or spec.loader is None:
        raise ValueError("unable to load baseline module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_dendrites


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--nuclei", type=int, required=True)
    parser.add_argument("--diameter", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build = builder_from(args.baseline)
    points = np.random.Generator(np.random.PCG64(1234)).uniform(
        0.0, 100.0, (args.nuclei, 2)
    )
    started = time.perf_counter()
    records = build(points, args.diameter, 4.0, args.seed)
    seconds = time.perf_counter() - started
    arrays = (
        records.azimuth_rad,
        records.tilt_rad,
        records.family_id,
        records.segments,
    )
    print(
        json.dumps(
            {
                "implementation": "baseline" if args.baseline else "current",
                "nuclei": args.nuclei,
                "diameter_mm": args.diameter,
                "seconds": seconds,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "state_array_bytes": sum(array.nbytes for array in arrays),
                "segment_count": len(records.segments),
                "sha256": [
                    hashlib.sha256(array.tobytes()).hexdigest() for array in arrays
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
