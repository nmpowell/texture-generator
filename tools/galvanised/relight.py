"""Render one sampled galvanised state under several deterministic light rigs.

Example:
    PYTHONPATH=src python tools/galvanised/relight.py /tmp/galvanised-lights --seed 42 --size 256 256
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from texture_generators.core.material_render import (
    render_material,
    render_material_array,
)
from texture_generators.materials.galvanised import build_state, sample_state
from texture_generators.materials.galvanised_config import (
    GalvanisedConfig,
    PreviewConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=[256, 256]
    )
    parser.add_argument(
        "--preset",
        default="regular",
        choices=["regular", "minimised", "weathered", "wet_storage"],
    )
    parser.add_argument("--render-samples", type=int, default=64)
    args = parser.parse_args()

    size = tuple(args.size)
    config = GalvanisedConfig(preset=args.preset)
    state = build_state(config, seed=args.seed, size=size)
    maps = sample_state(state, size=size)
    args.output.mkdir(parents=True, exist_ok=True)
    rigs = (
        PreviewConfig(rig="studio", render_samples=args.render_samples),
        PreviewConfig(rig="oblique", render_samples=args.render_samples),
        PreviewConfig(rig="overcast", render_samples=args.render_samples),
        PreviewConfig(rig="grazing", render_samples=args.render_samples),
        PreviewConfig(
            rig="oblique", light_azimuth_deg=135.0, render_samples=args.render_samples
        ),
    )
    entries = []
    for index, preview in enumerate(rigs):
        stem = f"{index:02d}-{preview.rig}-{int(preview.light_azimuth_deg)}deg"
        radiance = render_material_array(maps, preview=preview, output="linear")
        np.save(args.output / f"{stem}-linear.npy", radiance, allow_pickle=False)
        render_material(maps, preview=preview).save(args.output / f"{stem}.png")
        entries.append({"stem": stem, "preview": asdict(preview)})
    manifest = {
        "material_key": state.material_key,
        "seed": state.seed,
        "size": list(size),
        "config": state.config.to_mapping(),
        "material_metadata": dict(maps.metadata),
        "renders": entries,
        "linear_output": "float32 unclipped radiance, no display transform",
        "display_output": "exposure, fixed extended Reinhard, sRGB, RGB8 PNG",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
