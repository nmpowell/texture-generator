"""Extract and fit <=16 explicitly selected real physical footprints offline.

Example: PYTHONPATH=src python tools/galvanised/check_physical_footprints.py
  --center-mm 2 2 --center-mm 4 3 --output /tmp/physical-coupons
Rejected coupons retain reference arrays and diagnostics and cause exit status 1.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from texture_generators.core.footprint_fitting import (
    ADAPTER_VERSION,
    PhysicalFootprint,
    check_physical_footprints,
)
from texture_generators.materials.galvanised import build_state, sample_points
from texture_generators.materials.galvanised_config import GalvanisedConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=("regular", "minimised", "weathered", "wet_storage"),
        default="regular",
    )
    parser.add_argument("--size-mm", type=float, nargs=2, default=(8.0, 6.0))
    parser.add_argument("--diameter-mm", type=float, default=3.0)
    parser.add_argument(
        "--center-mm", type=float, nargs=2, action="append", required=True
    )
    parser.add_argument("--footprint-mm", type=float, nargs=2, default=(0.125, 0.125))
    parser.add_argument("--point-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= len(args.center_mm) <= 16:
        parser.error("provide 1..16 --center-mm locations")
    if not 1 <= args.point_batch_size <= 64:
        parser.error("--point-batch-size must lie in [1,64]")
    if any(
        length > tile
        for length, tile in zip(args.footprint_mm, args.size_mm, strict=True)
    ):
        parser.error("footprint dimensions must not exceed one state tile")
    footprints = [
        PhysicalFootprint(tuple(center), tuple(args.footprint_mm))
        for center in args.center_mm
    ]
    config = GalvanisedConfig(
        preset=args.preset,
        size_mm=tuple(args.size_mm),
        spangle_diameter_mm=args.diameter_mm,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = build_state(config, seed=args.seed)
    build_seconds = time.perf_counter() - started
    report: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "state_build_seconds": build_seconds,
        "point_batch_size": args.point_batch_size,
        "coupons": [],
    }
    accepted = True
    for index, footprint in enumerate(footprints):
        started = time.perf_counter()
        result = check_physical_footprints(
            state, [footprint], point_batch_size=args.point_batch_size
        )[0]
        item = dict(result.report)
        item["seconds"] = time.perf_counter() - started
        item["index"] = index
        center = sample_points(state, *footprint.center_mm)
        item["center_fields"] = {
            name: float(center[name]) if np.isfinite(center[name]) else None
            for name in (
                "grain_id",
                "boundary_distance_mm",
                "metallic",
                "patina_coverage",
                "white_stain_coverage",
            )
        }
        archive = f"footprint-{index:02d}.npz"
        np.savez(
            args.output / archive,
            allow_pickle=False,
            records=result.records,
            **result.arrays,
        )
        item["arrays"] = archive
        report["coupons"].append(item)
        accepted &= result.accepted
        # Publish progress even if a later coupon is interrupted.
        report["all_accepted"] = accepted
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"{index}: {'accepted' if result.accepted else 'rejected'} at {result.report['stage']} ({item['seconds']:.3f}s)",
            flush=True,
        )
    print(args.output / "report.json")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
