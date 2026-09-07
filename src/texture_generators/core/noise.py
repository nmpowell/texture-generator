"""Vectorised lattice gradient noise (Perlin-style) and fBm helpers.

All functions are fully vectorised over numpy arrays: no per-pixel Python
loops. Noise is periodic on its integer lattice, so warped or rotated
coordinates that fall outside the unit square still sample cleanly.

Frequencies are expressed in *lattice cells across the unit domain* and may
be anisotropic: pass a scalar for isotropic noise or ``(freq_x, freq_y)``
for stretched features (the basis of brushed metal and wood grain).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fbm",
    "fbm_at",
    "gradient_noise",
    "gradient_noise_at",
    "grid_coords",
    "ridged",
    "ridged_at",
]


def _as_freq_pair(freq: float | tuple[float, float]) -> tuple[float, float]:
    """Return ``(freq_x, freq_y)`` from a scalar or pair."""
    if isinstance(freq, (tuple, list, np.ndarray)):
        fx, fy = float(freq[0]), float(freq[1])
    else:
        fx = fy = float(freq)
    return fx, fy


def _fade(t: np.ndarray) -> np.ndarray:
    """Quintic fade curve 6t^5 - 15t^4 + 10t^3 (C2 continuous)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def grid_coords(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` coordinate arrays for ``shape`` = (H, W).

    ``x`` spans [0, 1) across the width and ``y`` spans [0, aspect) down the
    height, so features keep their aspect ratio on non-square canvases.
    """
    h, w = int(shape[0]), int(shape[1])
    aspect = h / max(w, 1)
    x = np.linspace(0.0, 1.0, w, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, aspect, h, endpoint=False, dtype=np.float32)
    return np.meshgrid(x, y)


def gradient_noise_at(
    x: np.ndarray,
    y: np.ndarray,
    freq: float | tuple[float, float],
    rng: np.random.Generator,
    periodic: tuple[bool, bool] = (False, False),
) -> np.ndarray:
    """Gradient noise sampled at arbitrary coordinates ``x``, ``y``.

    Returns values roughly in [-1, 1] with the same shape as ``x``.

    By default the gradient lattice is sized to the coordinates actually
    sampled, so the noise never repeats however far the domain extends
    (a unit-sized lattice would band-repeat on tall canvases and on
    ring/warp coordinates that span several units). Set ``periodic`` per
    axis to wrap that axis with period 1.0 instead -- polar sampling uses
    this so theta is seamless across the branch cut.
    """
    fx, fy = _as_freq_pair(freq)

    xs = np.asarray(x, dtype=np.float32) * fx
    ys = np.asarray(y, dtype=np.float32) * fy

    xi = np.floor(xs)
    yi = np.floor(ys)
    xf = xs - xi
    yf = ys - yi

    if periodic[0]:
        lx = max(
            1,
            round(fx),
        )
        i0 = xi.astype(np.int64) % lx
        i1 = (i0 + 1) % lx
    else:
        x_lo = float(xi.min())
        lx = int(xi.max() - x_lo) + 2
        i0 = (xi - np.float32(x_lo)).astype(np.int64)
        i1 = i0 + 1
    if periodic[1]:
        ly = max(
            1,
            round(fy),
        )
        j0 = yi.astype(np.int64) % ly
        j1 = (j0 + 1) % ly
    else:
        y_lo = float(yi.min())
        ly = int(yi.max() - y_lo) + 2
        j0 = (yi - np.float32(y_lo)).astype(np.int64)
        j1 = j0 + 1

    ang = rng.uniform(0.0, 2.0 * np.pi, size=(ly, lx)).astype(np.float32)
    gx = np.cos(ang)
    gy = np.sin(ang)

    n00 = gx[j0, i0] * xf + gy[j0, i0] * yf
    n10 = gx[j0, i1] * (xf - 1.0) + gy[j0, i1] * yf
    n01 = gx[j1, i0] * xf + gy[j1, i0] * (yf - 1.0)
    n11 = gx[j1, i1] * (xf - 1.0) + gy[j1, i1] * (yf - 1.0)

    u = _fade(xf)
    v = _fade(yf)
    top = n00 + u * (n10 - n00)
    bottom = n01 + u * (n11 - n01)
    out = top + v * (bottom - top)
    return (out * np.float32(1.41421356)).astype(np.float32)


def gradient_noise(
    shape: tuple[int, int],
    freq: float | tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Gradient noise on the regular grid for ``shape`` = (H, W)."""
    x, y = grid_coords(shape)
    return gradient_noise_at(x, y, freq, rng)


def fbm_at(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    freq: float | tuple[float, float] = 4.0,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    periodic: tuple[bool, bool] = (False, False),
) -> np.ndarray:
    """Sum ``gain^k * noise(freq * lacunarity^k)`` at arbitrary coordinates.

    The result is normalised by the total amplitude, so it stays roughly in
    [-1, 1] regardless of octave count.
    """
    fx, fy = _as_freq_pair(freq)
    total = np.zeros(np.shape(x), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    for _ in range(max(1, int(octaves))):
        total += np.float32(amp) * gradient_noise_at(x, y, (fx, fy), rng, periodic)
        norm += amp
        amp *= gain
        fx *= lacunarity
        fy *= lacunarity
    return (total / np.float32(max(norm, 1e-8))).astype(np.float32)


def fbm(
    shape: tuple[int, int],
    rng: np.random.Generator,
    freq: float | tuple[float, float] = 4.0,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """fBm on the regular grid for ``shape`` = (H, W)."""
    x, y = grid_coords(shape)
    return fbm_at(x, y, rng, freq, octaves, lacunarity, gain)


def ridged_at(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    freq: float | tuple[float, float] = 4.0,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Ridged multifractal ``1 - |fbm|`` in [0, 1] at arbitrary coordinates."""
    return (1.0 - np.abs(fbm_at(x, y, rng, freq, octaves, lacunarity, gain))).astype(
        np.float32
    )


def ridged(
    shape: tuple[int, int],
    rng: np.random.Generator,
    freq: float | tuple[float, float] = 4.0,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Ridged multifractal on the regular grid for ``shape`` = (H, W)."""
    x, y = grid_coords(shape)
    return ridged_at(x, y, rng, freq, octaves, lacunarity, gain)
