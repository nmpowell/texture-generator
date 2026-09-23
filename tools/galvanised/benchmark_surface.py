"""Measure one surface/export workload in a fresh process; retain JSON evidence.

Run each workload separately so peak RSS is attributable to one case::

    python tools/galvanised/benchmark_surface.py --size 512 --preview --out /tmp/g512
    python tools/galvanised/benchmark_surface.py --size 4096 --out /tmp/g4096

Large artifacts stay at the explicitly chosen output directory. The report
records failed stages too; it never converts an unmeasured budget into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from texture_generators import export_material, load_material, render_material
from texture_generators.materials.galvanised import (
    GENERATOR_VERSION,
    GalvanisedConfig,
    build_state,
    sample_state,
)


def _peak_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--preset", default="regular")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-mm", type=float, default=100.0)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--map", dest="maps", action="append")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--representation", choices=("rich", "single_lobe"), default="rich"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = GalvanisedConfig(
        preset=args.preset,
        size_mm=(args.size_mm, args.size_mm),
        representation=args.representation,
    )
    report = {
        "schema_version": 1,
        "status": "running",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "package": importlib.metadata.version("texture-generator"),
        },
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "size": [args.size, args.size],
        "config": config.to_mapping(),
        "chunk_size": args.chunk_size,
        "stage_seconds": {},
        "stage_peak_rss_mib": {},
    }
    stages = report["stage_seconds"]
    peaks = report["stage_peak_rss_mib"]
    started = time.perf_counter()
    stage = "state"

    def record_stage(name: str, began: float) -> None:
        stages[name] = time.perf_counter() - began
        peaks[name] = _peak_mib()
        (args.out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        began = time.perf_counter()
        state = build_state(config, seed=args.seed)
        report["material_key"] = state.material_key
        report["nuclei"] = len(state.partition.points_mm)
        record_stage(stage, began)
        stage = "sampling"
        began = time.perf_counter()
        maps = sample_state(
            state,
            (args.size, args.size),
            maps=args.maps,
            chunk_size=args.chunk_size,
            output_dir=args.out / "sampled",
        )
        report["channels"] = list(maps)
        report["sampled_bytes"] = sum(a.nbytes for a in maps.values())
        report["lobe_count"] = 0 if maps.lobes is None else len(maps.lobes)
        report["lobe_bytes"] = 0 if maps.lobes is None else maps.lobes.nbytes
        report["statistics"] = maps.metadata["stats"]
        record_stage(stage, began)
        if args.preview:
            stage = "preview"
            began = time.perf_counter()
            render_material(maps).save(args.out / "preview.png")
            record_stage(stage, began)
        stage = "export"
        began = time.perf_counter()
        manifest = export_material(maps, args.out / "bundle")
        report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        record_stage(stage, began)
        stage = "load_validation"
        began = time.perf_counter()
        loaded = load_material(manifest, mmap_mode="r")
        if tuple(loaded) != tuple(maps):
            raise RuntimeError("selected channel order changed in serialization")
        record_stage(stage, began)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["failed_stage"] = stage
        report["error"] = f"{type(error).__name__}: {error}"
    report["total_seconds"] = time.perf_counter() - started
    report["peak_rss_mib"] = _peak_mib()
    report["disk_bytes"] = sum(
        p.stat().st_size for p in args.out.rglob("*") if p.is_file()
    )
    report_path = args.out / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
