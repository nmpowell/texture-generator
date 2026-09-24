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
    # Millimetre-scale variations break up wet-storage deposits inside the
    # broad sheltered patches. Physical wavelengths stay fixed as output
    # resolution changes; integer wave vectors retain exact tile periods.
    angle = rng.uniform(-np.pi, np.pi, (3, 12))
    wavelength = rng.uniform(3.0, 8.0, (3, 12))
    detail_frequency = np.stack(
        (
            np.rint(size_mm[0] * np.cos(angle) / wavelength),
            np.rint(size_mm[1] * np.sin(angle) / wavelength),
        ),
        axis=-1,
    )
    frequency = np.concatenate((frequency, detail_frequency), axis=1)
    phase = np.concatenate((phase, rng.uniform(-np.pi, np.pi, (3, 12))), axis=1)
    amplitude = np.concatenate((amplitude, rng.uniform(0.035, 0.07, (3, 12))), axis=1)
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
        for harmonic in range(records.frequency.shape[1]):
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
    if exposure == 0.0:
        shape = np.broadcast_shapes(x_mm.shape, y_mm.shape)
        zero = np.zeros(shape, dtype=np.float64)
        return np.ones(shape, dtype=np.float64), zero, zero.copy(), zero.copy()
    moisture, shelter, salt = environment_fields(records, x_mm, y_mm)
    valley = np.clip(-substrate_height_um / 8.0, 0.0, 1.0)
    # Atmospheric patina develops over the full exposed sheet; local moisture
    # modulates it. Restricting all patina to the wet-storage threshold leaves
    # an ordinary weathered sheet almost fresh. This rate is an authoring
    # control, not a calendar-time corrosion prediction.
    moisture_active = np.clip((moisture - 0.42) / 0.58, 0.0, 1.0) ** 1.3
    salt_active = np.clip((salt - 0.45) / 0.55, 0.0, 1.0)
    rate = (0.40 + moisture_active) * (0.35 + 2.5 * wetness)
    rate *= 0.75 + 0.25 * shelter
    rate *= 1.0 + 0.20 * valley + 0.50 * salt_exposure * salt_active
    cp = -np.expm1(-exposure * rate)
    propensity = (
        white_stain
        * wetness
        * confinement
        * np.clip((moisture - 0.32) / 0.24, 0.0, 1.0)
        * (0.6 + 0.4 * shelter)
    )
    cw = cp * np.clip(propensity, 0.0, 1.0)
    patina = cp - cw
    zinc = 1.0 - cp
    deposit = 2.0 * patina**1.5 + 7.0 * cw**1.5
    return zinc, patina, cw, deposit
