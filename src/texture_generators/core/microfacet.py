"""Reference anisotropic GGX single-scattering evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["GGX_WIDTH_FLOOR", "evaluate_ggx", "openpbr_widths", "tangent_frame"]

GGX_WIDTH_FLOOR = 1.0e-4


def _unit(vector: ArrayLike, *, name: str) -> NDArray[np.float64]:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape == () or value.shape[-1] != 3:
        raise ValueError(f"{name} must have a final dimension of 3")
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(~np.isfinite(value)) or np.any(length <= 0.0):
        raise ValueError(f"{name} must contain finite, non-zero vectors")
    return value / length


def openpbr_widths(
    roughness: ArrayLike, anisotropy: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map OpenPBR perceptual controls to tangent/bitangent GGX widths."""
    r, a = np.broadcast_arrays(
        np.asarray(roughness, dtype=np.float64),
        np.asarray(anisotropy, dtype=np.float64),
    )
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(a)):
        raise ValueError("roughness and anisotropy must be finite")
    if np.any((r < 0.0) | (r > 1.0)) or np.any((a < 0.0) | (a > 1.0)):
        raise ValueError("roughness and anisotropy must lie in [0, 1]")
    alpha_t = r * r * np.sqrt(2.0 / (1.0 + (1.0 - a) ** 2))
    alpha_b = (1.0 - a) * alpha_t
    return np.maximum(alpha_t, GGX_WIDTH_FLOOR), np.maximum(alpha_b, GGX_WIDTH_FLOOR)


def _orthogonal_fallback(normal: NDArray[np.float64]) -> NDArray[np.float64]:
    index = np.argmin(np.abs(normal), axis=-1)
    basis = np.eye(3, dtype=np.float64)[index]
    projected = basis - np.sum(basis * normal, axis=-1, keepdims=True) * normal
    return projected / np.linalg.norm(projected, axis=-1, keepdims=True)


def tangent_frame(
    normal: ArrayLike, doubled_axis: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decode a clockwise raster doubled-angle axis into a surface frame.

    The desired tangent is ``(cos(theta), -sin(theta), 0)`` in the canonical
    right-handed frame. It is projected onto the signed normal and normalised;
    a deterministic least-aligned Cartesian axis handles degeneracy.
    """
    n = _unit(normal, name="normal")
    axis = np.asarray(doubled_axis, dtype=np.float64)
    if axis.shape == () or axis.shape[-1] != 2 or np.any(~np.isfinite(axis)):
        raise ValueError("doubled_axis must be finite with a final dimension of 2")
    n, axis3 = np.broadcast_arrays(
        n, np.pad(axis, [(0, 0)] * (axis.ndim - 1) + [(0, 1)])
    )
    axis = axis3[..., :2]
    theta = 0.5 * np.arctan2(axis[..., 1], axis[..., 0])
    desired = np.stack((np.cos(theta), -np.sin(theta), np.zeros_like(theta)), axis=-1)
    tangent = desired - np.sum(desired * n, axis=-1, keepdims=True) * n
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    fallback = _orthogonal_fallback(n)
    tangent = np.where(
        length > 1.0e-12, tangent / np.maximum(length, 1.0e-300), fallback
    )
    bitangent = np.cross(n, tangent)
    return tangent, bitangent


def evaluate_ggx(
    normal: ArrayLike,
    tangent: ArrayLike,
    wi: ArrayLike,
    wo: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
    fresnel: Callable[[NDArray[np.float64]], ArrayLike],
) -> NDArray[np.float64]:
    """Evaluate reciprocal height-correlated anisotropic GGX single scattering.

    Directions point away from the surface. Backfacing pairs return zero.
    ``fresnel`` receives ``dot(wi, h)`` and must return linear RGB.
    """
    n = _unit(normal, name="normal")
    incoming = _unit(wi, name="wi")
    outgoing = _unit(wo, name="wo")
    raw_tangent = _unit(tangent, name="tangent")
    n, incoming, outgoing, raw_tangent = np.broadcast_arrays(
        n, incoming, outgoing, raw_tangent
    )
    projected = raw_tangent - np.sum(raw_tangent * n, axis=-1, keepdims=True) * n
    projected_length = np.linalg.norm(projected, axis=-1, keepdims=True)
    t = np.where(
        projected_length > 1.0e-12,
        projected / np.maximum(projected_length, 1.0e-300),
        _orthogonal_fallback(n),
    )
    b = np.cross(n, t)

    at, ab = np.broadcast_arrays(
        np.asarray(alpha_t, dtype=np.float64), np.asarray(alpha_b, dtype=np.float64)
    )
    if (
        np.any(~np.isfinite(at))
        or np.any(~np.isfinite(ab))
        or np.any(at <= 0.0)
        or np.any(ab <= 0.0)
    ):
        raise ValueError("alpha_t and alpha_b must be finite and positive")
    shape = np.broadcast_shapes(n.shape[:-1], at.shape, ab.shape)
    n, incoming, outgoing, t, b = [
        np.broadcast_to(v, (*shape, 3)) for v in (n, incoming, outgoing, t, b)
    ]
    at, ab = np.broadcast_to(at, shape), np.broadcast_to(ab, shape)

    ni = np.sum(n * incoming, axis=-1)
    no = np.sum(n * outgoing, axis=-1)
    half_sum = incoming + outgoing
    half_length = np.linalg.norm(half_sum, axis=-1, keepdims=True)
    valid = (ni > 0.0) & (no > 0.0) & (half_length[..., 0] > 0.0)
    h = half_sum / np.maximum(half_length, 1.0e-300)
    hn = np.sum(h * n, axis=-1)
    ht = np.sum(h * t, axis=-1)
    hb = np.sum(h * b, axis=-1)
    valid &= hn > 0.0

    denominator = np.pi * at * ab * ((ht / at) ** 2 + (hb / ab) ** 2 + hn * hn) ** 2
    distribution = np.divide(
        1.0, denominator, out=np.zeros_like(denominator), where=valid
    )

    def smith_lambda(
        direction: NDArray[np.float64], cosine: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        vt = np.sum(direction * t, axis=-1)
        vb = np.sum(direction * b, axis=-1)
        projected2 = (at * vt) ** 2 + (ab * vb) ** 2
        root = np.sqrt(cosine * cosine + projected2)
        return np.divide(
            root - cosine,
            2.0 * cosine,
            out=np.full_like(cosine, np.inf),
            where=cosine > 0.0,
        )

    masking = 1.0 / (1.0 + smith_lambda(incoming, ni) + smith_lambda(outgoing, no))
    wh = np.clip(np.sum(incoming * h, axis=-1), 0.0, 1.0)
    f = np.asarray(fresnel(wh), dtype=np.float64)
    if f.shape == () or f.shape[-1] != 3:
        raise ValueError("fresnel must return RGB with a final dimension of 3")
    f = np.broadcast_to(f, (*shape, 3))
    scale = np.divide(
        distribution * masking,
        4.0 * ni * no,
        out=np.zeros_like(distribution),
        where=valid,
    )
    result = scale[..., None] * f
    return np.where(valid[..., None], result, 0.0)
