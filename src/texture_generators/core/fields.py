"""Scalar-field utilities: normalisation, remapping, smoothstep, height to normal."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    "blur",
    "draw_segments",
    "height_to_normal",
    "normalize01",
    "remap",
    "smoothstep",
]


def normalize01(a: np.ndarray) -> np.ndarray:
    """Rescale ``a`` to [0, 1]. Constant input maps to 0.5."""
    a = np.asarray(a, dtype=np.float32)
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < 1e-8:
        return np.full(a.shape, 0.5, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def remap(
    a: np.ndarray,
    in_lo: float,
    in_hi: float,
    out_lo: float = 0.0,
    out_hi: float = 1.0,
) -> np.ndarray:
    """Linearly map ``[in_lo, in_hi]`` onto ``[out_lo, out_hi]`` (unclamped)."""
    a = np.asarray(a, dtype=np.float32)
    span = in_hi - in_lo
    if abs(span) < 1e-8:
        return np.full(a.shape, out_lo, dtype=np.float32)
    t = (a - np.float32(in_lo)) / np.float32(span)
    return (np.float32(out_lo) + t * np.float32(out_hi - out_lo)).astype(np.float32)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep: 0 below ``edge0``, 1 above ``edge1``, smooth between."""
    x = np.asarray(x, dtype=np.float32)
    span = edge1 - edge0
    if abs(span) < 1e-8:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - np.float32(edge0)) / np.float32(span), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def height_to_normal(
    h: np.ndarray, strength: float = 1.0, periodic: bool = False
) -> np.ndarray:
    """Convert a height field to unit normals, shape (H, W, 3).

    Uses the standard derivative relation ``n ~ (-s*dH/du, -s*dH/dv, 1)``,
    with gradients from :func:`numpy.gradient` (central differences).

    ``periodic`` wraps those differences at the edges instead of falling back
    to one-sided ones. Use it when the height field itself is periodic --
    otherwise the edge rows get a different derivative from everywhere else,
    which shows up as a bright or dark rim and breaks tiling.
    """
    h = np.asarray(h, dtype=np.float32)
    if periodic:
        dhdv = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * np.float32(0.5)
        dhdu = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * np.float32(0.5)
    else:
        dhdv, dhdu = np.gradient(h)
    s = np.float32(strength)
    nx = -s * dhdu
    ny = -s * dhdv
    nz = np.ones_like(h)
    n = np.stack([nx, ny, nz], axis=-1)
    length = np.sqrt((n * n).sum(axis=-1, keepdims=True))
    return (n / np.maximum(length, 1e-8)).astype(np.float32)


def draw_segments(
    shape: tuple[int, int],
    segments,
    width: int = 1,
) -> np.ndarray:
    """Rasterise signed line segments onto a float field of ``shape`` (H, W).

    ``segments`` yields ``(x0, y0, x1, y1, value)`` in pixel coordinates. Uses
    a PIL "F" canvas, so the Python loop runs over segments (hundreds to a few
    thousand) rather than pixels. Overlapping segments overwrite rather than
    accumulate, which is what scratches and fibres want.
    """
    h, w = int(shape[0]), int(shape[1])
    canvas = Image.new("F", (w, h), 0.0)
    draw = ImageDraw.Draw(canvas)
    for x0, y0, x1, y1, value in segments:
        draw.line(
            (float(x0), float(y0), float(x1), float(y1)),
            fill=float(value),
            width=int(width),
        )
    return np.asarray(canvas, dtype=np.float32)


def blur(a: np.ndarray, radius: int = 1, passes: int = 1) -> np.ndarray:
    """Cheap separable box blur (wrap-around), used to soften scatter layers."""
    a = np.asarray(a, dtype=np.float32)
    if radius < 1:
        return a
    k = 2 * radius + 1
    out = a
    for _ in range(max(1, int(passes))):
        acc = np.zeros_like(out)
        for shift in range(-radius, radius + 1):
            acc += np.roll(out, shift, axis=1)
        out = acc / np.float32(k)
        acc = np.zeros_like(out)
        for shift in range(-radius, radius + 1):
            acc += np.roll(out, shift, axis=0)
        out = acc / np.float32(k)
    return out.astype(np.float32)
