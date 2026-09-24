"""Versioned semantic keys and stateless periodic random fields."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Sequence
from numbers import Integral

import numpy as np

__all__ = [
    "KEY_DERIVATION_VERSION",
    "capture_seed",
    "derive_key",
    "make_rng",
    "make_rng_with_seed",
    "material_key_from_rng",
    "periodic_lattice_u64",
    "periodic_lattice_uniform",
    "topology_key",
    "topology_rng",
    "weather_key",
    "weather_rng",
]

KEY_DERIVATION_VERSION = 1
_UINT64_MAX = (1 << 64) - 1


def _uint64_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT64_MAX:
        raise ValueError(f"{name} must be in the unsigned 64-bit range")
    return result


def capture_seed(seed: int | None) -> int:
    """Return a concrete unsigned 64-bit seed, capturing entropy for ``None``."""
    if seed is None:
        return secrets.randbits(64)
    return _uint64_integer(seed, "seed")


def make_rng(seed: int) -> np.random.Generator:
    """Create a generator using the explicitly selected PCG64 algorithm."""
    return np.random.Generator(np.random.PCG64(_uint64_integer(seed, "seed")))


def make_rng_with_seed(seed: int | None) -> tuple[int, np.random.Generator]:
    """Capture a replayable seed and construct its PCG64 generator."""
    concrete = capture_seed(seed)
    return concrete, make_rng(concrete)


def material_key_from_rng(rng: np.random.Generator) -> int:
    """Consume exactly one fixed-width unsigned draw for a material branch."""
    value = rng.integers(0, 1 << 64, dtype=np.uint64)
    return int(value)


def derive_key(material_key: int, stage: str, *identifiers: int) -> int:
    """Derive a stable BLAKE2b key from semantic, versioned components."""
    key = _uint64_integer(material_key, "material_key")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")
    stage_bytes = stage.encode("utf-8", errors="strict")
    if len(stage_bytes) > _UINT64_MAX:
        raise ValueError("stage encoding is too long")

    encoded = bytearray(b"texture-generator\x00key")
    encoded.extend(KEY_DERIVATION_VERSION.to_bytes(2, "little"))
    encoded.extend(key.to_bytes(8, "little"))
    encoded.extend(len(stage_bytes).to_bytes(8, "little"))
    encoded.extend(stage_bytes)
    encoded.extend(len(identifiers).to_bytes(8, "little"))
    for index, identifier in enumerate(identifiers):
        encoded.extend(
            _uint64_integer(identifier, f"identifiers[{index}]").to_bytes(8, "little")
        )
    digest = hashlib.blake2b(encoded, digest_size=8, person=b"tg-mat-key-v1").digest()
    return int.from_bytes(digest, "little")


def topology_key(material_key: int, stage: str, *identifiers: int) -> int:
    """Derive a topology namespace key, isolated from weather evolution."""
    return derive_key(material_key, f"topology/{stage}", *identifiers)


def weather_key(material_key: int, stage: str, *identifiers: int) -> int:
    """Derive a weather namespace key, isolated from substrate topology."""
    return derive_key(material_key, f"weather/{stage}", *identifiers)


def topology_rng(
    material_key: int, stage: str, *identifiers: int
) -> np.random.Generator:
    """Return a PCG64 stream for bounded topology object creation."""
    return make_rng(topology_key(material_key, stage, *identifiers))


def weather_rng(
    material_key: int, stage: str, *identifiers: int
) -> np.random.Generator:
    """Return a PCG64 stream for bounded weather object creation."""
    return make_rng(weather_key(material_key, stage, *identifiers))


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def periodic_lattice_u64(
    material_key: int,
    stage: str,
    x: np.ndarray | Sequence[int] | int,
    y: np.ndarray | Sequence[int] | int,
    *,
    period: tuple[int, int],
    identifiers: tuple[int, ...] = (),
) -> np.ndarray:
    """Return stateless uint64 values on a periodic integer lattice."""
    period_x = _uint64_integer(period[0], "period[0]")
    period_y = _uint64_integer(period[1], "period[1]")
    if period_x == 0 or period_y == 0:
        raise ValueError("lattice periods must be positive")
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if x_values.dtype.kind not in "iu" or y_values.dtype.kind not in "iu":
        raise TypeError("lattice coordinates must be integers")
    x_values, y_values = np.broadcast_arrays(x_values, y_values)
    x_wrapped = np.mod(x_values, period_x).astype(np.uint64)
    y_wrapped = np.mod(y_values, period_y).astype(np.uint64)
    seed = np.uint64(derive_key(material_key, f"lattice/{stage}", *identifiers))
    combined = (
        seed
        ^ _splitmix64(x_wrapped)
        ^ np.bitwise_left_shift(_splitmix64(y_wrapped), np.uint64(1))
    )
    return _splitmix64(combined)


def periodic_lattice_uniform(
    material_key: int,
    stage: str,
    x: np.ndarray | Sequence[int] | int,
    y: np.ndarray | Sequence[int] | int,
    *,
    period: tuple[int, int],
    identifiers: tuple[int, ...] = (),
) -> np.ndarray:
    """Return deterministic periodic values in ``[0, 1)`` without raster fitting."""
    hashed = periodic_lattice_u64(
        material_key, stage, x, y, period=period, identifiers=identifiers
    )
    return ((hashed >> np.uint64(40)).astype(np.float32) * np.float32(2.0**-24)).astype(
        np.float32
    )
