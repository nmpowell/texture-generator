#!/usr/bin/env python3
"""Download, verify and convert the pinned zinc/CIE optical sources.

This development tool is the only networked part of the optics pipeline.
Runtime code reads the generated table through ``importlib.resources``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

SOURCES = {
    "Werner.yml": (
        "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/"
        "main/database/data/main/Zn/nk/Werner.yml",
        "c29023bed42520fbf3cec2272c9f31080429508073edf9124a0e76cbcaee6abf",
    ),
    "CIE_std_illum_D65.csv": (
        "https://files.cie.co.at/Publications-datasets/CIE_std_illum_D65.csv",
        "e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb",
    ),
    "CIE_xyz_1931_2deg.csv": (
        "https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv",
        "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1",
    ),
}
METADATA_URLS = {
    "CIE_std_illum_D65.csv_metadata_v2.json": (
        "https://files.cie.co.at/Publications-datasets/"
        "CIE_std_illum_D65.csv_metadata_v2.json"
    ),
    "CIE_xyz_1931_2deg.csv_metadata.json": (
        "https://files.cie.co.at/Publications-datasets/"
        "CIE_xyz_1931_2deg.csv_metadata.json"
    ),
}
XYZ_TO_RGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _obtain(name: str, url: str, source_dir: Path | None) -> bytes:
    if source_dir is not None:
        return (source_dir / name).read_bytes()
    request = urllib.request.Request(
        url, headers={"User-Agent": "texture-generator/optics-build"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _parse_werner(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    in_table = False
    for line in data.decode("utf-8").splitlines():
        if re.match(r"\s*data:\s*\|", line):
            in_table = True
            continue
        if in_table:
            match = re.fullmatch(
                r"\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*", line
            )
            if match is None:
                if rows:
                    break
                continue
            rows.append(tuple(map(float, match.groups())))
    if len(rows) < 3:
        raise ValueError("Werner.yml contains no usable tabulated nk data")
    table = np.asarray(rows, dtype=np.float64)
    return table[:, 0] * 1000.0, table[:, 1], table[:, 2]


def _numeric_csv(data: bytes, columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    for row in csv.reader(io.StringIO(data.decode("utf-8-sig"))):
        if len(row) < columns:
            continue
        try:
            rows.append([float(value) for value in row[:columns]])
        except ValueError:
            continue
    result = np.asarray(rows, dtype=np.float64)
    if result.shape[0] < 2:
        raise ValueError("official CIE CSV contains no usable numerical rows")
    return result


def _fresnel(cos_theta: np.ndarray, n: np.ndarray, k: np.ndarray) -> np.ndarray:
    c = cos_theta[..., None]
    m = n[None, :] + 1j * k[None, :]
    q = np.sqrt(m * m - (1.0 - c * c))
    q = np.where((q.imag < 0.0) | ((q.imag == 0.0) & (q.real < 0.0)), -q, q)
    rs = (c - q) / (c + q)
    rp = (m * m * c - q) / (m * m * c + q)
    return 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)


def build(output: Path, source_dir: Path | None) -> None:
    source_bytes: dict[str, bytes] = {}
    for name, (url, expected) in SOURCES.items():
        data = _obtain(name, url, source_dir)
        actual = _sha256(data)
        if actual != expected:
            raise ValueError(f"{name}: expected SHA-256 {expected}, got {actual}")
        source_bytes[name] = data
    metadata = {
        name: _obtain(name, url, source_dir) for name, url in METADATA_URLS.items()
    }

    source_output = output / "sources"
    source_output.mkdir(parents=True, exist_ok=True)
    for name, data in source_bytes.items() | metadata.items():
        (source_output / name).write_bytes(data)

    zinc_nm, source_n, source_k = _parse_werner(source_bytes["Werner.yml"])
    d65 = _numeric_csv(source_bytes["CIE_std_illum_D65.csv"], 2)
    observer = _numeric_csv(source_bytes["CIE_xyz_1931_2deg.csv"], 4)
    wavelength = np.arange(360.0, 831.0, dtype=np.float64)
    for name, grid in (
        ("Werner", zinc_nm),
        ("D65", d65[:, 0]),
        ("observer", observer[:, 0]),
    ):
        if grid[0] > wavelength[0] or grid[-1] < wavelength[-1]:
            raise ValueError(f"{name} data do not cover 360--830 nm")
    n = np.interp(wavelength, zinc_nm, source_n)
    k = np.interp(wavelength, zinc_nm, source_k)
    illuminant = np.interp(wavelength, d65[:, 0], d65[:, 1])
    cmf = np.stack(
        [np.interp(wavelength, observer[:, 0], observer[:, i]) for i in range(1, 4)],
        axis=-1,
    )
    normaliser = np.trapezoid(illuminant * cmf[:, 1], wavelength)
    cos_theta = np.linspace(0.0, 1.0, 1025, dtype=np.float64) ** 2
    spectral = _fresnel(cos_theta, n, k)
    xyz = np.stack(
        [
            np.trapezoid(spectral * illuminant * cmf[:, i], wavelength, axis=1)
            / normaliser
            for i in range(3)
        ],
        axis=-1,
    )
    rgb = xyz @ XYZ_TO_RGB.T
    white_xyz = np.asarray(
        [
            np.trapezoid(illuminant * cmf[:, i], wavelength) / normaliser
            for i in range(3)
        ]
    )
    white_rgb = white_xyz @ XYZ_TO_RGB.T

    output.mkdir(parents=True, exist_ok=True)
    table = output / "zinc_fresnel_lut.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("cos_theta", "r", "g", "b"))
        writer.writerows(np.column_stack((cos_theta, rgb)))
    manifest = {
        "schema": 1,
        "method": "D65/CIE 1931 2 degree, 360--830 nm at 1 nm, trapezoidal",
        "interpolation": "linear n and k in wavelength before Fresnel evaluation",
        "zinc_f0_linear_srgb": rgb[-1].tolist(),
        "perfect_reflector_linear_srgb": white_rgb.tolist(),
        "source_urls": {name: url for name, (url, _) in SOURCES.items()},
        "source_sha256": {name: digest for name, (_, digest) in SOURCES.items()},
        "metadata_urls": METADATA_URLS,
        "metadata_sha256": {name: _sha256(data) for name, data in metadata.items()},
        "derived_sha256": {table.name: _sha256(table.read_bytes())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/texture_generators/data/galvanised/optics"),
        help="resource directory",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="read already-downloaded, exactly named source files instead of networking",
    )
    args = parser.parse_args()
    build(args.output, args.source_dir)


if __name__ == "__main__":
    main()
