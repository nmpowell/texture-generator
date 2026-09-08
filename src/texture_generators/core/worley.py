"""Worley (cellular) noise: F1/F2 distance fields from a jittered feature grid.

One feature point per grid cell, jittered inside the cell, with a 3x3
neighbourhood scan vectorised over cells (9 array ops, no Python pixel
loops). ``stretch`` scales the cell grid anisotropically to produce
elongated cells -- used for wood pores and moulded plastic stipple.
"""

from __future__ import annotations

import numpy as np

__all__ = ["worley", "worley_at"]


def worley_at(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    density: float = 16.0,
    metric: str = "euclidean",
    stretch: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(F1, F2)`` distances at arbitrary coordinates.

    ``density`` is the number of cells across the unit domain in x.
    ``stretch`` > 1 stretches cells along x (fewer cells in x than y).
    ``metric`` is ``"euclidean"``, ``"manhattan"`` or ``"chebyshev"``.
    Distances are in cell units, so F1 is roughly in [0, 1.2].
    """
    nx = max(
        1,
        round(density / max(stretch, 1e-6)),
    )
    ny = max(
        1,
        round(density),
    )

    xs = np.asarray(x, dtype=np.float32) * nx
    ys = np.asarray(y, dtype=np.float32) * ny
    xi = np.floor(xs).astype(np.int64)
    yi = np.floor(ys).astype(np.int64)
    xf = xs - xi
    yf = ys - yi

    # Size the feature grid to the cells actually visited, padded by one cell
    # each side for the 3x3 scan, so the pattern never repeats however far
    # the coordinates extend (a unit-wrapped grid band-repeats on tall
    # canvases). Feature point offset inside each cell is in [0, 1).
    x_lo = int(xi.min()) - 1
    y_lo = int(yi.min()) - 1
    cells_x = int(xi.max()) - x_lo + 2
    cells_y = int(yi.max()) - y_lo + 2
    jitter = rng.random((cells_y, cells_x, 2)).astype(np.float32)
    xi = xi - x_lo
    yi = yi - y_lo

    f1 = np.full(xs.shape, np.inf, dtype=np.float32)
    f2 = np.full(xs.shape, np.inf, dtype=np.float32)

    for dy in (-1, 0, 1):
        jj = yi + dy
        for dx in (-1, 0, 1):
            ii = xi + dx
            px = dx + jitter[jj, ii, 0] - xf
            py = dy + jitter[jj, ii, 1] - yf
            if metric == "manhattan":
                d = np.abs(px) + np.abs(py)
            elif metric == "chebyshev":
                d = np.maximum(np.abs(px), np.abs(py))
            else:
                d = np.sqrt(px * px + py * py)
            f2 = np.minimum(f2, np.maximum(f1, d))
            f1 = np.minimum(f1, d)

    return f1.astype(np.float32), f2.astype(np.float32)


def worley(
    shape: tuple[int, int],
    rng: np.random.Generator,
    density: float = 16.0,
    metric: str = "euclidean",
    stretch: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(F1, F2)`` distance fields on the grid for ``shape`` = (H, W)."""
    from .noise import grid_coords

    x, y = grid_coords(shape)
    return worley_at(x, y, rng, density, metric, stretch)
