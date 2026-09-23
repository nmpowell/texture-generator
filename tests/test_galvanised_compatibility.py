"""Freeze the pre-galvanised metal output and random stream contracts."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pytest

from texture_generators import generate_array, resolve_variant, to_image
from texture_generators.materials import metal

BASELINE = json.loads(
    (Path(__file__).parent / "data" / "metal_legacy_baseline.json").read_text()
)


def _check_pixels(result: np.ndarray, record: dict) -> None:
    assert (
        hashlib.sha256(to_image(result).tobytes()).hexdigest() == record["rgb8_sha256"]
    )
    # Exact float bytes are a same-runtime contract, unlike the existing
    # quantised-image regression checks. libm/NumPy can differ across platforms.
    if (
        platform.platform() == BASELINE["platform"]
        and np.__version__ == BASELINE["numpy"]
    ):
        assert hashlib.sha256(result.tobytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("record", BASELINE["explicit"])
def test_explicit_legacy_bytes_and_rng(record: dict) -> None:
    width, height = record["size"]
    rng = np.random.default_rng(record["seed"])
    result = metal.generate((height, width), rng, record["variant"])
    _check_pixels(result, record)
    assert rng.bit_generator.state == record["rng_state"]


@pytest.mark.parametrize("record", BASELINE["random_choices"])
def test_legacy_random_choice_and_bytes(record: dict) -> None:
    assert resolve_variant("metal", seed=record["seed"]) == record["variant"]
    result = generate_array("metal", size=(32, 24), seed=record["seed"])
    _check_pixels(result, record)
