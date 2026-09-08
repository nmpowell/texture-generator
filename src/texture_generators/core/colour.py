"""CIELAB <-> sRGB conversion, and the linear-light helpers that go with it.

Paper colour is specified in the trade as L*a*b* (plus ISO brightness and CIE
whiteness), so building the palette from Lab rather than from hand-picked RGB
is both more faithful and easier to reason about: whiteness is -b*, ageing is
+b*, kraft is a specific (L*, a*, b*) region.

The other materials in this library work directly in display values. Paper
does not: translucent fibres composite through one another, and compositing
in gamma space darkens overlaps incorrectly, so the paper pipeline works in
linear light and encodes to sRGB once at the end.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lab_to_linear_rgb",
    "lab_to_srgb",
    "linear_rgb_to_lab",
    "linear_to_srgb",
    "srgb_to_lab",
    "srgb_to_linear",
]

# sRGB primaries, D65 white point (IEC 61966-2-1).
_XYZ_TO_RGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)
_WHITE_D65 = np.asarray([0.95047, 1.00000, 1.08883], dtype=np.float64)


def _lab_f_inv(t: np.ndarray) -> np.ndarray:
    """Inverse of the CIELAB companding function."""
    delta = 6.0 / 29.0
    return np.where(t > delta, t**3, 3.0 * delta * delta * (t - 4.0 / 29.0))


def lab_to_linear_rgb(lab) -> np.ndarray:
    """Convert ``(L*, a*, b*)`` to linear sRGB in [0, 1] (clipped)."""
    lab = np.asarray(lab, dtype=np.float64)
    ell, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (ell + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz = np.stack([_lab_f_inv(fx), _lab_f_inv(fy), _lab_f_inv(fz)], axis=-1)
    xyz = xyz * _WHITE_D65
    rgb = xyz @ _XYZ_TO_RGB.T
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear light to sRGB display values, both in [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(
        x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055
    ).astype(np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB display values to linear light, both in [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(
        np.float32
    )


def lab_to_srgb(lab) -> np.ndarray:
    """Convert ``(L*, a*, b*)`` to sRGB display values in [0, 1]."""
    return linear_to_srgb(lab_to_linear_rgb(lab))


def _lab_f(t: np.ndarray) -> np.ndarray:
    """The CIELAB companding function."""
    delta = 6.0 / 29.0
    return np.where(t > delta**3, np.cbrt(t), t / (3.0 * delta * delta) + 4.0 / 29.0)


def linear_rgb_to_lab(rgb) -> np.ndarray:
    """Convert linear sRGB in [0, 1] to ``(L*, a*, b*)``.

    The inverse of :func:`lab_to_linear_rgb`, and the way a *rendered* colour is
    checked against the measured value it was built from: a difference of a few
    display levels means nothing on its own, whereas a dL* or a dC* is the same
    quantity the source measurement was quoted in.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = (rgb @ np.linalg.inv(_XYZ_TO_RGB).T) / _WHITE_D65
    f = _lab_f(xyz)
    return np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def srgb_to_lab(rgb) -> np.ndarray:
    """Convert sRGB display values in [0, 1] to ``(L*, a*, b*)``."""
    return linear_rgb_to_lab(srgb_to_linear(rgb))
