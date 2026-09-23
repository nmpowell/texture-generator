"""Periodic environmental fields and nested opaque deposit coverages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from texture_generators.core.random_fields import weather_rng


def _readonly(array: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(array)
    return np.frombuffer(packed.tobytes(), dtype=packed.dtype).reshape(packed.shape)


@dataclass(frozen=True)
class WeatherRecords:
    """Fixed random environment, independent of exposure and crystallisation."""

    frequency: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray
    size_mm: tuple[float, float]
    weather_key: int


def build_weather(
    material_key: int,
    size_mm: tuple[float, float],
    *,
    weather_seed: int | None = None,
) -> WeatherRecords:
    key = material_key if weather_seed is None else weather_seed
    rng = weather_rng(key, "environment")
    # Integer harmonics make the full field exactly periodic at the tile seam.
    frequency = np.array(
        [
            [[1, 0], [0, 1], [1, 1], [2, -1]],
            [[1, -1], [2, 1], [0, 2], [3, 0]],
            [[2, 0], [0, 1], [1, -2], [2, 2]],
        ],
        dtype=np.int16,
    )
    phase = rng.uniform(-np.pi, np.pi, (3, 4))
    amplitude = rng.uniform(0.12, 0.32, (3, 4))
    return WeatherRecords(
        _readonly(frequency), _readonly(phase), _readonly(amplitude), size_mm, key
    )


def environment_fields(
    records: WeatherRecords, x_mm: np.ndarray, y_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.broadcast_arrays(x_mm, y_mm)
    u = 2.0 * np.pi * np.remainder(x, records.size_mm[0]) / records.size_mm[0]
    v = 2.0 * np.pi * np.remainder(y, records.size_mm[1]) / records.size_mm[1]
    fields = []
    for index in range(3):
        signal = np.zeros(x.shape, dtype=np.float64)
        for harmonic in range(4):
            fx, fy = records.frequency[index, harmonic]
            signal += records.amplitude[index, harmonic] * np.sin(
                fx * u + fy * v + records.phase[index, harmonic]
            )
        fields.append(np.clip(0.5 + 0.5 * signal, 0.0, 1.0))
    return fields[0], fields[1], fields[2]


def weather_at(
    records: WeatherRecords,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    exposure: float,
    wetness: float,
    confinement: float,
    salt_exposure: float,
    white_stain: float,
    substrate_height_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return zinc, visible patina, nested white stain and deposit height.

    Cp=1-exp(-k*exposure), Cw=Cp*propensity, and the visible fractions are
    (1-Cp, Cp-Cw, Cw). Exposure zero has exactly zero height and coverage.
    """
    moisture, shelter, salt = environment_fields(records, x_mm, y_mm)
    valley = np.clip(-substrate_height_um / 8.0, 0.0, 1.0)
    # A fixed smooth environmental threshold leaves exposed zinc gaps.  It is
    # an authoring mask, not a predicted corrosion onset or elapsed-time rate.
    moisture_active = np.clip((moisture - 0.42) / 0.58, 0.0, 1.0) ** 1.3
    salt_active = np.clip((salt - 0.45) / 0.55, 0.0, 1.0)
    rate = moisture_active * (0.35 + 2.5 * wetness) * (0.75 + 0.25 * shelter)
    rate *= 1.0 + 0.20 * valley + 0.50 * salt_exposure * salt_active
    cp = -np.expm1(-exposure * rate)
    propensity = (
        white_stain
        * wetness
        * confinement
        * np.clip((moisture - 0.42) / 0.58, 0.0, 1.0)
        * (0.6 + 0.4 * shelter)
    )
    cw = cp * np.clip(propensity, 0.0, 1.0)
    patina = cp - cw
    zinc = 1.0 - cp
    deposit = 2.0 * patina**1.5 + 7.0 * cw**1.5
    return zinc, patina, cw, deposit
