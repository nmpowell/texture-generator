"""Anisotropic directional-albedo compensation for GGX reflection.

The separable missing-energy shape is the Kulla--Conty approximation. Its
unit-Fresnel directional albedo is tabulated independently over both GGX widths,
view cosine and tangent-frame azimuth. This is an approximation to the
multi-bounce Smith model, not a claim of exact spectral multiple scattering.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from texture_generators.core.conductor import zinc_fresnel

__all__ = [
    "directional_albedo",
    "energy_resource_hashes",
    "evaluate_ggx_compensated",
    "schlick_directional_albedo",
    "schlick_directional_moment",
    "zinc_directional_albedo",
    "zinc_total_directional_albedo",
    "zinc_total_directional_moment",
]


@lru_cache(maxsize=1)
def energy_resource_hashes() -> Mapping[str, str]:
    """Verify and identify the installed renderer table and its provenance."""
    root = files("texture_generators").joinpath("data", "galvanised", "energy")
    manifest_bytes = root.joinpath("manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    table_hash = hashlib.sha256(
        root.joinpath("ggx_directional_albedo.npz").read_bytes()
    ).hexdigest()
    if manifest.get("table_sha256") != table_hash:
        raise RuntimeError("energy table checksum differs from its manifest")
    return MappingProxyType(
        {
            "ggx_directional_albedo.npz": table_hash,
            "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    )


@lru_cache(maxsize=1)
def _table() -> tuple[NDArray[np.float64], ...]:
    energy_resource_hashes()
    path = files("texture_generators").joinpath(
        "data", "galvanised", "energy", "ggx_directional_albedo.npz"
    )
    with path.open("rb") as stream, np.load(stream) as source:
        widths = source["widths"].astype(np.float64)
        cosines = source["cosines"].astype(np.float64)
        azimuths = source["azimuths"].astype(np.float64)
        albedo = source["albedo"].astype(np.float64)
        fifth = source["schlick_fifth"].astype(np.float64)
        zinc = source["zinc_albedo"].astype(np.float64)
        unit_moment = source["unit_moment"].astype(np.float64)
        fifth_moment = source["fifth_moment"].astype(np.float64)
        zinc_moment = source["zinc_moment"].astype(np.float64)
    # Integrate each bilinear cell exactly in azimuth and its linear cosine
    # interpolation times 2*mu, adding only a negligible [0, 0.001] endpoint.
    mu = np.r_[0.0, cosines]

    def mean(table: NDArray[np.float64]) -> NDArray[np.float64]:
        vector = table.ndim == 5
        values = np.concatenate((np.take(table, [0], axis=2), table), axis=2)
        mean_phi = np.trapezoid(values, azimuths, axis=3) / (np.pi / 2.0)
        e0 = np.take(mean_phi, np.arange(len(mu) - 1), axis=2)
        e1 = np.take(mean_phi, np.arange(1, len(mu)), axis=2)
        suffix = (1,) if vector else ()
        dm = np.diff(mu).reshape((1, 1, -1, *suffix))
        m0 = mu[:-1].reshape((1, 1, -1, *suffix))
        return np.sum(
            2.0
            * dm
            * (e0 * m0 + (e0 * dm + (e1 - e0) * m0) / 2.0 + (e1 - e0) * dm / 3.0),
            axis=2,
        )

    def z_mean(table: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.concatenate((table[:, :, :1, :], table), axis=2)
        mean_phi = np.trapezoid(values, azimuths, axis=3) / (np.pi / 2.0)
        e0, e1 = mean_phi[:, :, :-1], mean_phi[:, :, 1:]
        dm = np.diff(mu)[None, None, :]
        m0 = mu[:-1][None, None, :]
        delta = e1 - e0
        return np.sum(
            2
            * (
                e0 * (m0 * m0 * dm + m0 * dm * dm + dm**3 / 3)
                + delta * (m0 * m0 * dm / 2 + 2 * m0 * dm * dm / 3 + dm**3 / 4)
            ),
            axis=2,
        )

    average = mean(albedo)
    fifth_average = mean(fifth)
    zinc_average = mean(zinc)
    unit_z_average = z_mean(albedo)
    fifth_z_average = z_mean(fifth)
    unit_moment_average = np.zeros((*unit_moment.shape[:2], 3))
    fifth_moment_average = np.zeros_like(unit_moment_average)
    zinc_moment_average = np.zeros((*zinc_moment.shape[:2], 3, 3))
    arrays = (
        widths,
        cosines,
        azimuths,
        albedo,
        average,
        fifth,
        fifth_average,
        zinc,
        zinc_average,
        unit_moment,
        unit_moment_average,
        fifth_moment,
        fifth_moment_average,
        zinc_moment,
        zinc_moment_average,
        unit_z_average,
        fifth_z_average,
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _indices(
    grid: NDArray[np.float64], x: NDArray[np.float64], *, log: bool = False
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    coordinate = (
        np.log(np.clip(x, grid[0], grid[-1])) if log else np.clip(x, grid[0], grid[-1])
    )
    nodes = np.log(grid) if log else grid
    low = np.clip(
        np.searchsorted(nodes, coordinate, side="right") - 1, 0, len(grid) - 2
    )
    fraction = (coordinate - nodes[low]) / (nodes[low + 1] - nodes[low])
    return low, fraction


def _interpolate_4d(
    at: NDArray[np.float64],
    ab: NDArray[np.float64],
    mu: NDArray[np.float64],
    phi: NDArray[np.float64],
    table: NDArray[np.float64],
    averages: NDArray[np.float64],
    *,
    clamp: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    widths, cosines, azimuths = _table()[:3]
    ix, fx = _indices(widths, at, log=True)
    iy, fy = _indices(widths, ab, log=True)
    iz, fz = _indices(cosines, mu)
    iw, fw = _indices(azimuths, phi)
    trailing = table.shape[4:]
    result_shape = (*mu.shape, *trailing)
    value = np.zeros(result_shape, dtype=np.float64)
    average = np.zeros(result_shape, dtype=np.float64)
    for dx in (0, 1):
        wx = fx if dx else 1.0 - fx
        for dy in (0, 1):
            wxy = wx * (fy if dy else 1.0 - fy)
            average += (
                wxy.reshape((*wxy.shape, *(1 for _ in trailing)))
                * averages[ix + dx, iy + dy]
            )
            for dz in (0, 1):
                wxyz = wxy * (fz if dz else 1.0 - fz)
                for dw in (0, 1):
                    weight = wxyz * (fw if dw else 1.0 - fw)
                    value += (
                        weight.reshape((*weight.shape, *(1 for _ in trailing)))
                        * table[ix + dx, iy + dy, iz + dz, iw + dw]
                    )
    if clamp:
        return np.clip(value, 0.0, 1.0), np.clip(average, 0.0, 1.0)
    return value, average


def _directional_albedo(
    normal: ArrayLike,
    tangent: ArrayLike,
    direction: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
    table_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate GGX directional and cosine-weighted mean albedo."""
    n, t, d = [np.asarray(x, dtype=np.float64) for x in (normal, tangent, direction)]
    at, ab = [np.asarray(x, dtype=np.float64) for x in (alpha_t, alpha_b)]
    shape = np.broadcast_shapes(
        n.shape[:-1], t.shape[:-1], d.shape[:-1], at.shape, ab.shape
    )
    n, t, d = [np.broadcast_to(x, (*shape, 3)) for x in (n, t, d)]
    at, ab = [np.broadcast_to(x, shape) for x in (at, ab)]
    if any(
        np.any(~np.isfinite(x)) or np.any(np.linalg.norm(x, axis=-1) <= 0)
        for x in (n, t, d)
    ):
        raise ValueError("frame and direction vectors must be finite and nonzero")
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    t = t - np.sum(t * n, axis=-1, keepdims=True) * n
    if np.any(np.linalg.norm(t, axis=-1) <= 1e-12):
        raise ValueError("tangent must not be parallel to normal")
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    if (
        np.any(~np.isfinite(at))
        or np.any(~np.isfinite(ab))
        or np.any(at <= 0)
        or np.any(ab <= 0)
    ):
        raise ValueError("GGX widths must be finite and positive")
    mu = np.sum(n * d, axis=-1)
    b = np.cross(n, t)
    x = np.sum(d * t, axis=-1)
    y = np.sum(d * b, axis=-1)
    phi = np.arctan2(np.abs(y), np.abs(x))
    resources = _table()
    value, average = _interpolate_4d(
        at,
        ab,
        np.maximum(mu, 0),
        phi,
        resources[table_index],
        resources[table_index + 1],
        clamp=table_index < 9,
    )
    mask = (mu > 0).reshape((*mu.shape, *(1 for _ in resources[table_index].shape[4:])))
    return np.where(mask, value, 0.0), average


def directional_albedo(
    normal: ArrayLike,
    tangent: ArrayLike,
    direction: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit-Fresnel anisotropic GGX directional and mean albedo."""
    return _directional_albedo(normal, tangent, direction, alpha_t, alpha_b, 3)


def schlick_directional_albedo(
    normal: ArrayLike,
    tangent: ArrayLike,
    direction: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
    ior: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Schlick dielectric single-scattering directional and mean albedo."""
    if not np.isfinite(ior) or ior < 1.0:
        raise ValueError("ior must be finite and at least one")
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    e, avg = _directional_albedo(normal, tangent, direction, alpha_t, alpha_b, 3)
    fifth, fifth_avg = _directional_albedo(
        normal, tangent, direction, alpha_t, alpha_b, 5
    )
    return f0 * e + (1.0 - f0) * fifth, f0 * avg + (1.0 - f0) * fifth_avg


def zinc_directional_albedo(
    normal: ArrayLike,
    tangent: ArrayLike,
    direction: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """D65 zinc Fresnel weighted single-scattering albedo in RGB."""
    return _directional_albedo(normal, tangent, direction, alpha_t, alpha_b, 7)


def zinc_total_directional_albedo(
    normal: ArrayLike,
    tangent: ArrayLike,
    direction: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
) -> NDArray[np.float64]:
    """Analytic constant-environment response of the compensated zinc BRDF."""
    single, _ = zinc_directional_albedo(normal, tangent, direction, alpha_t, alpha_b)
    white, average = directional_albedo(normal, tangent, direction, alpha_t, alpha_b)
    fbar = _zinc_average()
    colour = (
        fbar
        * average[..., None]
        / np.maximum(1.0 - fbar * (1.0 - average[..., None]), 1e-12)
    )
    return single + (1.0 - white)[..., None] * colour


def _width_value(
    alpha_t: ArrayLike, alpha_b: ArrayLike, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    widths = _table()[0]
    at, ab = np.broadcast_arrays(
        np.asarray(alpha_t, dtype=np.float64), np.asarray(alpha_b, dtype=np.float64)
    )
    ix, fx = _indices(widths, at, log=True)
    iy, fy = _indices(widths, ab, log=True)
    return (
        (1 - fx) * (1 - fy) * values[ix, iy]
        + fx * (1 - fy) * values[ix + 1, iy]
        + (1 - fx) * fy * values[ix, iy + 1]
        + fx * fy * values[ix + 1, iy + 1]
    )


def _moment_signs(
    moment: NDArray[np.float64],
    normal: ArrayLike,
    tangent: ArrayLike,
    direction: ArrayLike,
) -> NDArray[np.float64]:
    n, t, d = [np.asarray(v, dtype=np.float64) for v in (normal, tangent, direction)]
    b = np.cross(n, t)
    sx = np.sign(np.sum(t * d, axis=-1))
    sy = np.sign(np.sum(b * d, axis=-1))
    signed = moment.copy()
    extra = moment.ndim - sx.ndim - 1
    signed[..., 0] *= sx.reshape((*sx.shape, *(1 for _ in range(extra))))
    signed[..., 1] *= sy.reshape((*sy.shape, *(1 for _ in range(extra))))
    return signed


def schlick_directional_moment(
    normal: ArrayLike,
    tangent: ArrayLike,
    direction: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
    ior: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """First local outgoing moment and missing-energy diffuse Z moment."""
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    unit, _ = _directional_albedo(normal, tangent, direction, alpha_t, alpha_b, 9)
    fifth, _ = _directional_albedo(normal, tangent, direction, alpha_t, alpha_b, 11)
    moment = _moment_signs(f0 * unit + (1 - f0) * fifth, normal, tangent, direction)
    resources = _table()
    avg = f0 * resources[4] + (1 - f0) * resources[6]
    z_avg = f0 * resources[15] + (1 - f0) * resources[16]
    missing_z = _width_value(
        alpha_t,
        alpha_b,
        np.divide(
            2.0 / 3.0 - z_avg,
            1.0 - avg,
            out=np.full_like(avg, 2.0 / 3.0),
            where=(1.0 - avg) > 1e-8,
        ),
    )
    return moment, missing_z


def zinc_total_directional_moment(
    normal: ArrayLike,
    tangent: ArrayLike,
    direction: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
) -> NDArray[np.float64]:
    """RGB first local outgoing moment of compensated zinc reflection."""
    single, _ = _directional_albedo(normal, tangent, direction, alpha_t, alpha_b, 13)
    result = _moment_signs(single, normal, tangent, direction)
    white, average = directional_albedo(normal, tangent, direction, alpha_t, alpha_b)
    fbar = _zinc_average()
    colour = (
        fbar
        * average[..., None]
        / np.maximum(1.0 - fbar * (1.0 - average[..., None]), 1e-12)
    )
    resources = _table()
    missing_z_grid = np.divide(
        2.0 / 3.0 - resources[15],
        1.0 - resources[4],
        out=np.full_like(resources[4], 2.0 / 3.0),
        where=(1.0 - resources[4]) > 1e-8,
    )
    missing_z = _width_value(alpha_t, alpha_b, missing_z_grid)
    result[..., 2] += (1.0 - white)[..., None] * colour * missing_z[..., None]
    return result


@lru_cache(maxsize=1)
def _zinc_average() -> NDArray[np.float64]:
    nodes, weights = np.polynomial.legendre.leggauss(96)
    mu = (nodes + 1.0) / 2.0
    result = np.sum(zinc_fresnel(mu) * (weights * mu)[:, None], axis=0)
    result.setflags(write=False)
    return result


def evaluate_ggx_compensated(
    normal: ArrayLike,
    tangent: ArrayLike,
    wi: ArrayLike,
    wo: ArrayLike,
    alpha_t: ArrayLike,
    alpha_b: ArrayLike,
    *,
    fresnel_average: ArrayLike | None = None,
    outgoing_albedo: ArrayLike | None = None,
    mean_albedo: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return the reciprocal coloured missing-energy lobe for GGX.

    Add this result to `evaluate_ggx` to obtain the approximate multi-bounce
    conductor BRDF. Default attenuation uses the pinned zinc Fresnel integral.
    For an ideal white conductor pass ``fresnel_average=np.ones(3)``.
    """
    ei, _ = directional_albedo(normal, tangent, wi, alpha_t, alpha_b)
    if (outgoing_albedo is None) != (mean_albedo is None):
        raise ValueError("outgoing_albedo and mean_albedo must be supplied together")
    if outgoing_albedo is None:
        eo, average = directional_albedo(normal, tangent, wo, alpha_t, alpha_b)
    else:
        eo = np.asarray(outgoing_albedo, dtype=np.float64)
        average = np.asarray(mean_albedo, dtype=np.float64)
        eo, average = np.broadcast_arrays(eo, average)
        if np.any(~np.isfinite(eo)) or np.any(~np.isfinite(average)):
            raise ValueError("prepared albedo terms must be finite")
    fbar = np.asarray(
        _zinc_average() if fresnel_average is None else fresnel_average,
        dtype=np.float64,
    )
    if (
        fbar.shape[-1] != 3
        or np.any(~np.isfinite(fbar))
        or np.any((fbar < 0) | (fbar > 1))
    ):
        raise ValueError("fresnel_average must contain RGB reflectances in [0, 1]")
    # Kulla--Conty geometric series for successive coloured bounces. Fbar=1
    # restores the full missing energy; absorbing conductors return less.
    colour = (
        fbar
        * average[..., None]
        / np.maximum(1.0 - fbar * (1.0 - average[..., None]), 1e-12)
    )
    geometry = (1.0 - ei) * (1.0 - eo) / (np.pi * np.maximum(1.0 - average, 1e-12))
    n = np.asarray(normal, dtype=np.float64)
    incoming = np.asarray(wi, dtype=np.float64)
    outgoing = np.asarray(wo, dtype=np.float64)
    valid = (np.sum(n * incoming, axis=-1) > 0.0) & (
        np.sum(n * outgoing, axis=-1) > 0.0
    )
    return np.where(valid[..., None], geometry[..., None] * colour, 0.0)
