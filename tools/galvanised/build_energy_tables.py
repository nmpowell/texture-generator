"""Build deterministic anisotropic GGX unit-Fresnel directional-albedo data.

Usage: python tools/galvanised/build_energy_tables.py

Visible-normal sampling follows Heitz, *Sampling the GGX Distribution of
Visible Normals*, JCGT 7(4), 2018. The tabulated quantity is
the single-bounce directional albedo; it is not a measured zinc property.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/texture_generators/data/galvanised/energy"
)
WIDTHS = np.asarray(
    [
        0.0001,
        0.015,
        0.035,
        0.07,
        0.1,
        0.12,
        0.16,
        0.2,
        0.27,
        0.35,
        0.45,
        0.5,
        0.55,
        0.7,
        0.8,
        0.95,
        1.1,
        1.25,
        1.415,
    ]
)
COSINES = np.asarray(
    [
        0.001,
        0.01,
        0.03,
        0.06,
        0.1,
        0.16,
        0.24,
        0.34,
        0.46,
        0.59,
        0.71,
        0.82,
        0.91,
        0.97,
        1.0,
    ]
)
AZIMUTHS = np.linspace(0.0, np.pi / 2.0, 10)
SAMPLES = 8192


def _radical_inverse(n: np.ndarray) -> np.ndarray:
    n = n.astype(np.uint32)
    n = (n << 16) | (n >> 16)
    n = ((n & 0x55555555) << 1) | ((n & 0xAAAAAAAA) >> 1)
    n = ((n & 0x33333333) << 2) | ((n & 0xCCCCCCCC) >> 2)
    n = ((n & 0x0F0F0F0F) << 4) | ((n & 0xF0F0F0F0) >> 4)
    n = ((n & 0x00FF00FF) << 8) | ((n & 0xFF00FF00) >> 8)
    return n.astype(np.float64) * (1.0 / 2**32)


def _albedo(
    alpha_t: float,
    alpha_b: float,
    wi: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    zinc_cos: np.ndarray,
    zinc_rgb: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = wi * (alpha_t, alpha_b, 1.0)
    v /= np.linalg.norm(v)
    lensq = v[0] ** 2 + v[1] ** 2
    t1 = (
        np.asarray([-v[1], v[0], 0.0]) / np.sqrt(lensq)
        if lensq > 1e-20
        else np.asarray([1.0, 0.0, 0.0])
    )
    t2 = np.cross(v, t1)
    radius = np.sqrt(u1)
    a = radius * np.cos(2.0 * np.pi * u2)
    b = radius * np.sin(2.0 * np.pi * u2)
    s = 0.5 * (1.0 + v[2])
    b = (1.0 - s) * np.sqrt(np.maximum(0.0, 1.0 - a * a)) + s * b
    z = np.sqrt(np.maximum(0.0, 1.0 - a * a - b * b))
    nh = a[:, None] * t1 + b[:, None] * t2 + z[:, None] * v
    m = nh * (alpha_t, alpha_b, 1.0)
    m /= np.linalg.norm(m, axis=1)[:, None]
    wo = 2.0 * np.sum(wi * m, axis=1)[:, None] * m - wi
    mu_o = wo[:, 2]
    lambda_i = 0.5 * (
        np.sqrt(1.0 + ((alpha_t * wi[0]) ** 2 + (alpha_b * wi[1]) ** 2) / wi[2] ** 2)
        - 1.0
    )
    safe_o = np.maximum(mu_o, 1e-30)
    lambda_o = 0.5 * (
        np.sqrt(
            1.0 + ((alpha_t * wo[:, 0]) ** 2 + (alpha_b * wo[:, 1]) ** 2) / safe_o**2
        )
        - 1.0
    )
    weight = np.where(mu_o > 0, (1.0 + lambda_i) / (1.0 + lambda_i + lambda_o), 0.0)
    half_cosine = np.clip(np.sum(wi * m, axis=1), 0.0, 1.0)
    fresnel = np.stack(
        [
            np.interp(half_cosine, zinc_cos, zinc_rgb[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    fifth = (1.0 - half_cosine) ** 5
    return (
        float(np.mean(weight)),
        float(np.mean(weight * fifth)),
        np.mean(weight[:, None] * fresnel, axis=0),
        np.mean(weight[:, None] * wo, axis=0),
        np.mean((weight * fifth)[:, None] * wo, axis=0),
        np.mean((weight[:, None] * fresnel)[:, :, None] * wo[:, None, :], axis=0),
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    sequence = np.arange(SAMPLES, dtype=np.uint32)
    u1 = (sequence.astype(np.float64) + 0.5) / SAMPLES
    u2 = _radical_inverse(sequence)
    source = ROOT.parent / "optics" / "zinc_fresnel_lut.csv"
    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    zinc_cos = np.asarray([float(row["cos_theta"]) for row in rows])
    zinc_rgb = np.asarray(
        [[float(row[channel]) for channel in ("r", "g", "b")] for row in rows]
    )
    table = np.empty(
        (len(WIDTHS), len(WIDTHS), len(COSINES), len(AZIMUTHS)), dtype=np.float32
    )
    fifth = np.empty_like(table)
    zinc = np.empty((*table.shape, 3), dtype=np.float32)
    unit_moment = np.empty((*table.shape, 3), dtype=np.float32)
    fifth_moment = np.empty_like(unit_moment)
    zinc_moment = np.empty((*table.shape, 3, 3), dtype=np.float32)
    for i, at in enumerate(WIDTHS):
        for j, ab in enumerate(WIDTHS):
            for k, mu in enumerate(COSINES):
                sine = np.sqrt(1.0 - mu * mu)
                for azimuth_index, phi in enumerate(AZIMUTHS):
                    wi = np.asarray([sine * np.cos(phi), sine * np.sin(phi), mu])
                    e, e5, ez, m1, m5, mz = _albedo(
                        float(at), float(ab), wi, u1, u2, zinc_cos, zinc_rgb
                    )
                    table[i, j, k, azimuth_index] = e
                    fifth[i, j, k, azimuth_index] = e5
                    zinc[i, j, k, azimuth_index] = ez
                    unit_moment[i, j, k, azimuth_index] = m1
                    fifth_moment[i, j, k, azimuth_index] = m5
                    zinc_moment[i, j, k, azimuth_index] = mz
        print(f"width {i + 1}/{len(WIDTHS)}", flush=True)
    path = ROOT / "ggx_directional_albedo.npz"
    np.savez_compressed(
        path,
        widths=WIDTHS,
        cosines=COSINES,
        azimuths=AZIMUTHS,
        albedo=table,
        schlick_fifth=fifth,
        zinc_albedo=zinc,
        unit_moment=unit_moment,
        fifth_moment=fifth_moment,
        zinc_moment=zinc_moment,
    )
    manifest = {
        "method": "Heitz 2018 GGX visible-normal QMC; height-correlated Smith G2/G1 ratio",
        "samples_per_direction": SAMPLES,
        "directions": "cos(theta) and first-quadrant azimuth; tangent and bitangent widths tabulated independently",
        "table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "quantity": "unit-Fresnel single-scattering directional hemispherical reflectance",
        "additional_quantities": [
            "Schlick (1-cos_half)^5 directional albedo",
            "pinned D65 zinc Fresnel directional albedo",
            "first outgoing directional moments of each single-scattering term",
        ],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
