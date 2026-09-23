"""Profile an unchanged galvanised state/map sample with cProfile."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from pathlib import Path

from texture_generators.materials.galvanised import build_state, sample_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile", type=Path, default=Path("/tmp/galvanised-sampler.prof")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    state = build_state(seed=args.seed, size=(args.size, args.size))
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    maps = sample_state(
        state,
        (args.size, args.size),
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
    )
    profile.disable()
    elapsed = time.perf_counter() - start
    profile.dump_stats(str(args.profile))
    print(
        f"sample_seconds={elapsed:.3f} lobes={len(maps.lobes) if maps.lobes is not None else 0}"
    )
    pstats.Stats(profile).sort_stats("cumtime").print_stats(30)


if __name__ == "__main__":
    main()
