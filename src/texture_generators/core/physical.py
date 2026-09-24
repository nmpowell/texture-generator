"""Physical-coordinate helpers shared by sampled material representations."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np

__all__ = [
    "height_to_normal_physical",
    "pixel_centres",
    "validate_image_size",
    "validate_map_size",
    "validate_size_mm",
]


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_size_mm(size_mm: Sequence[float]) -> tuple[float, float]:
    """Validate and return physical ``(width, height)`` in millimetres."""
    if isinstance(size_mm, (str, bytes)) or len(size_mm) != 2:
        raise ValueError("size_mm must contain exactly (width, height)")
    width = _finite_real(size_mm[0], "size_mm[0]")
    height = _finite_real(size_mm[1], "size_mm[1]")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("size_mm dimensions must be positive")
    return width, height


def validate_image_size(size: Sequence[int]) -> tuple[int, int]:
    """Validate a positive raster size and return ``(width, height)``."""
    if isinstance(size, (str, bytes)) or len(size) != 2:
        raise ValueError("size must contain exactly (width, height)")
    values: list[int] = []
    for index, value in enumerate(size):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"size[{index}] must be an integer")
        values.append(int(value))
    if min(values) < 1:
        raise ValueError("image dimensions must be positive")
    return values[0], values[1]


def validate_map_size(size: Sequence[int]) -> tuple[int, int]:
    """Validate a material-map size and return ``(width, height)``.

    Maps smaller than three samples on either axis cannot support the declared
    central-difference contract.  Lower-level derivative helpers still handle
    such arrays by treating the undersampled axis as flat.
    """
    values = validate_image_size(size)
    if min(values) < 3:
        raise ValueError("material-map dimensions must each be at least 3")
    return values


def pixel_centres(
    size_mm: Sequence[float], *, size: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return broadcastable physical pixel-centre coordinates ``(x, y)``.

    ``size`` uses the public ``(width, height)`` convention.  The result has
    shapes ``(1, width)`` and ``(height, 1)`` and is expressed in millimetres.
    """
    width_mm, height_mm = validate_size_mm(size_mm)
    width, height = validate_image_size(size)
    x = width_mm * (np.arange(width, dtype=np.float64) + 0.5) / width
    y = height_mm * (np.arange(height, dtype=np.float64) + 0.5) / height
    return x[None, :], y[:, None]


def _axis_derivative(
    height: np.ndarray, spacing: float, axis: int, *, periodic: bool
) -> np.ndarray:
    if height.shape[axis] < 3:
        return np.zeros_like(height, dtype=np.float64)
    if periodic:
        return (np.roll(height, -1, axis=axis) - np.roll(height, 1, axis=axis)) / (
            2.0 * spacing
        )
    return np.gradient(height, spacing, axis=axis, edge_order=2)


def height_to_normal_physical(
    height_um: np.ndarray,
    size_mm: Sequence[float],
    *,
    periodic: bool = True,
) -> np.ndarray:
    """Derive signed canonical tangent-space normals from physical height.

    Raster ``u`` points right and ``v`` points down.  The returned right-handed
    frame is ``X=u``, ``Y=Ly-v``, ``+Z`` out of the sheet, hence
    ``normal ~ (-dh/du * 1e-3, +dh/dv * 1e-3, 1)``.  Pixel spacings are
    independently ``Lx/W`` and ``Ly/H``.
    """
    height = np.asarray(height_um)
    if height.ndim != 2:
        raise ValueError("height_um must be a two-dimensional array")
    if height.dtype.kind not in "fiu" or not np.all(np.isfinite(height)):
        raise ValueError("height_um must contain only finite numeric values")
    width_mm, height_mm = validate_size_mm(size_mm)
    rows, columns = height.shape
    if rows == 0 or columns == 0:
        raise ValueError("height_um axes must not be empty")

    working = height.astype(np.float64, copy=False)
    dh_du = _axis_derivative(working, width_mm / columns, axis=1, periodic=periodic)
    dh_dv = _axis_derivative(working, height_mm / rows, axis=0, periodic=periodic)
    p = 1e-3 * dh_du
    q = 1e-3 * dh_dv
    normal = np.stack((-p, q, np.ones_like(working)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal.astype(np.float32)
