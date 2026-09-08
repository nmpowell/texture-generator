"""Domain warping: p' = p + A * W(p) with W built from independent fbm fields.

Warping the *input* of a pattern (rather than adding noise to its output) is
what turns a mathematically regular field -- concentric rings, straight
streaks -- into something organic. ``double_warp`` applies the classic
recursive variant (warp the warp) for stronger, more turbulent distortion.
"""

from __future__ import annotations

import numpy as np

from .noise import fbm_at

__all__ = ["double_warp", "rotate", "warp"]


def warp(
    coords: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    amp: float = 0.1,
    freq: float | tuple[float, float] = 3.0,
    octaves: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset ``coords`` by ``amp`` times a two-channel fbm vector field."""
    x, y = coords
    wx = fbm_at(x, y, rng, freq, octaves)
    wy = fbm_at(x, y, rng, freq, octaves)
    return (
        (x + np.float32(amp) * wx).astype(np.float32),
        (y + np.float32(amp) * wy).astype(np.float32),
    )


def double_warp(
    coords: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    amp: float = 0.1,
    freq: float | tuple[float, float] = 3.0,
    octaves: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Recursive warp: warp the coordinates, then warp the result again.

    The second pass runs at higher frequency and lower amplitude, adding
    fine turbulence on top of the large-scale distortion.
    """
    first = warp(coords, rng, amp, freq, octaves)
    return warp(first, rng, amp * 0.45, _scale_freq(freq, 2.3), octaves)


def rotate(
    coords: tuple[np.ndarray, np.ndarray],
    angle: float,
    centre: tuple[float, float] = (0.5, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``coords`` by ``angle`` radians about ``centre``."""
    x, y = coords
    cx, cy = centre
    ca, sa = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    dx = x - np.float32(cx)
    dy = y - np.float32(cy)
    return (
        (cx + ca * dx - sa * dy).astype(np.float32),
        (cy + sa * dx + ca * dy).astype(np.float32),
    )


def _scale_freq(
    freq: float | tuple[float, float], k: float
) -> float | tuple[float, float]:
    """Multiply a scalar or ``(fx, fy)`` frequency by ``k``."""
    if isinstance(freq, (tuple, list, np.ndarray)):
        return (float(freq[0]) * k, float(freq[1]) * k)
    return float(freq) * k
