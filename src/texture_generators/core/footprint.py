"""Phase-aligned rectangular midpoint integration for material fields."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def midpoint_axis(start: int, stop: int, rate: int) -> np.ndarray:
    """Global pixel coordinates with ``rate`` positive, equal-weight samples."""
    if isinstance(rate, bool) or not isinstance(rate, Integral) or rate < 1:
        raise ValueError("quadrature rate must be a positive integer")
    return np.repeat(np.arange(start, stop, dtype=np.float64), rate) + np.tile(
        (np.arange(rate, dtype=np.float64) + 0.5) / rate, stop - start
    )


def box_average(values: np.ndarray, rate: int = 2) -> np.ndarray:
    """Integrate an interleaved sample grid, preserving trailing components."""
    if isinstance(rate, bool) or not isinstance(rate, Integral) or rate < 1:
        raise ValueError("quadrature rate must be a positive integer")
    rows, columns = values.shape[:2]
    if rows % rate or columns % rate:
        raise ValueError("sample grid dimensions must be divisible by quadrature rate")
    return values.reshape(
        rows // rate, rate, columns // rate, rate, *values.shape[2:]
    ).mean(axis=(1, 3))
