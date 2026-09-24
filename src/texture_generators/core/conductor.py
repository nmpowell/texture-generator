"""Exact conductor Fresnel and pinned daylight-integrated zinc optics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["conductor_fresnel", "resource_hashes", "zinc_f0", "zinc_fresnel"]


def _resource(name: str):
    """Resolve the same pinned resources in source and installed packages."""
    package_root = files("texture_generators")
    installed = package_root.joinpath("data", "galvanised", "optics", name)
    if not installed.is_file():
        raise RuntimeError(f"missing packaged zinc optical resource: {name}")
    return installed


def conductor_fresnel(
    cos_theta: ArrayLike, n: ArrayLike, k: ArrayLike
) -> NDArray[np.float64]:
    """Return unpolarised air-to-conductor Fresnel reflectance.

    Inputs follow NumPy broadcasting. ``n + 1j*k`` is the complex refractive
    index. The square-root sign is selected so that the transmitted wave is
    attenuating (positive imaginary part under the ``exp(-i omega t)``
    convention), with positive real part as the lossless tie-break.
    """
    c, eta, extinction = np.broadcast_arrays(
        np.asarray(cos_theta, dtype=np.float64),
        np.asarray(n, dtype=np.float64),
        np.asarray(k, dtype=np.float64),
    )
    c = np.clip(c, 0.0, 1.0)
    m = eta.astype(np.complex128) + 1j * extinction
    q = np.sqrt(m * m - (1.0 - c * c))
    flip = (q.imag < 0.0) | ((q.imag == 0.0) & (q.real < 0.0))
    q = np.where(flip, -q, q)
    rs = (c - q) / (c + q)
    m2c = m * m * c
    rp = (m2c - q) / (m2c + q)
    result = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)
    return np.asarray(np.clip(result.real, 0.0, 1.0), dtype=np.float64)


@lru_cache(maxsize=1)
def _manifest() -> dict[str, object]:
    resource = _resource("manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _zinc_table() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    resource = _resource("zinc_fresnel_lut.csv")
    with resource.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cos_theta = np.asarray([float(row["cos_theta"]) for row in rows])
    rgb = np.asarray(
        [[float(row[channel]) for channel in ("r", "g", "b")] for row in rows]
    )
    if (
        cos_theta.ndim != 1
        or rgb.shape != (cos_theta.size, 3)
        or cos_theta.size < 2
        or np.any(np.diff(cos_theta) <= 0.0)
        or cos_theta[0] != 0.0
        or cos_theta[-1] != 1.0
    ):
        raise RuntimeError("invalid pinned zinc Fresnel LUT")
    cos_theta.setflags(write=False)
    rgb.setflags(write=False)
    return cos_theta, rgb


def zinc_fresnel(cos_theta: ArrayLike) -> NDArray[np.float64]:
    """Interpolate the pinned D65/CIE zinc Fresnel LUT in linear sRGB."""
    value = np.clip(np.asarray(cos_theta, dtype=np.float64), 0.0, 1.0)
    cosine, rgb = _zinc_table()
    return np.stack(
        [np.interp(value, cosine, rgb[:, channel]) for channel in range(3)],
        axis=-1,
    )


def zinc_f0() -> NDArray[np.float64]:
    """Return normal-incidence zinc reflectance in D65 linear sRGB."""
    result = np.asarray(_manifest()["zinc_f0_linear_srgb"], dtype=np.float64)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def resource_hashes() -> Mapping[str, str]:
    """Return immutable SHA-256 identities for source and derived resources."""
    manifest = _manifest()
    sources = cast(dict[str, str], manifest["source_sha256"])
    metadata = cast(dict[str, str], manifest["metadata_sha256"])
    derived = cast(dict[str, str], manifest["derived_sha256"])
    expected = dict(sources) | dict(metadata) | dict(derived)
    for name, checksum in expected.items():
        location = name if name in derived else f"sources/{name}"
        actual = hashlib.sha256(_resource(location).read_bytes()).hexdigest()
        if actual != checksum:
            raise RuntimeError(f"zinc optical resource checksum mismatch: {name}")
    manifest_f0 = np.asarray(manifest["zinc_f0_linear_srgb"], dtype=np.float64)
    if manifest_f0.shape != (3,) or not np.allclose(
        manifest_f0, zinc_fresnel(1.0), atol=1e-12, rtol=0.0
    ):
        raise RuntimeError("zinc manifest F0 differs from the pinned Fresnel table")
    identities = expected | {
        "manifest.json": hashlib.sha256(
            _resource("manifest.json").read_bytes()
        ).hexdigest()
    }
    return MappingProxyType(identities)
