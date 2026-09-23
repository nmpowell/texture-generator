from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from texture_generators.core.conductor import (
    conductor_fresnel,
    resource_hashes,
    zinc_f0,
    zinc_fresnel,
)
from texture_generators.core.microfacet import (
    evaluate_ggx,
    openpbr_widths,
    tangent_frame,
)


def _constant_white(cos_theta: np.ndarray) -> np.ndarray:
    return np.ones((*np.shape(cos_theta), 3), dtype=np.float64)


def _direction(theta: float, phi: float) -> np.ndarray:
    return np.asarray(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def _independent_spectral_reference(
    cosine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate source bytes without using the runtime LUT or build script."""
    root = (
        Path(__file__).parents[1]
        / "src/texture_generators/data/galvanised/optics/sources"
    )
    rows = []
    for line in (root / "Werner.yml").read_text().splitlines():
        match = re.fullmatch(
            r"\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*", line
        )
        if match:
            rows.append(tuple(map(float, match.groups())))
    zinc = np.asarray(rows)
    d65 = np.loadtxt(root / "CIE_std_illum_D65.csv", delimiter=",")
    observer = np.loadtxt(root / "CIE_xyz_1931_2deg.csv", delimiter=",")
    wavelength = np.arange(360.0, 831.0)
    n = np.interp(wavelength, zinc[:, 0] * 1000.0, zinc[:, 1])
    k = np.interp(wavelength, zinc[:, 0] * 1000.0, zinc[:, 2])
    illuminant = np.interp(wavelength, d65[:, 0], d65[:, 1])
    cmf = np.stack(
        [np.interp(wavelength, observer[:, 0], observer[:, i]) for i in range(1, 4)],
        axis=-1,
    )
    c = np.asarray(cosine)[..., None]
    m = n + 1j * k
    q = np.sqrt(m * m - (1.0 - c * c))
    rs = (c - q) / (c + q)
    rp = (m * m * c - q) / (m * m * c + q)
    reflectance = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)
    normaliser = np.trapezoid(illuminant * cmf[:, 1], wavelength)
    xyz = np.stack(
        [
            np.trapezoid(
                reflectance * illuminant * cmf[:, channel], wavelength, axis=-1
            )
            / normaliser
            for channel in range(3)
        ],
        axis=-1,
    )
    matrix = np.asarray(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    white_xyz = np.asarray(
        [
            np.trapezoid(illuminant * cmf[:, channel], wavelength) / normaliser
            for channel in range(3)
        ]
    )
    return xyz @ matrix.T, white_xyz @ matrix.T


def test_conductor_normal_incidence_matches_analytic_source_nodes() -> None:
    # Independently transcribed representative Werner data nodes (nm, n, k).
    nodes = np.asarray(
        [
            [381.490, 0.5508, 3.2986],
            [413.281, 0.6029, 3.6376],
            [450.852, 0.6794, 4.0209],
            [495.937, 0.7858, 4.4636],
            [551.041, 0.9322, 4.9850],
            [619.921, 1.1356, 5.6116],
            [708.481, 1.4247, 6.3811],
        ]
    )
    n, k = nodes[:, 1], nodes[:, 2]
    expected = ((n - 1.0) ** 2 + k * k) / ((n + 1.0) ** 2 + k * k)
    np.testing.assert_allclose(conductor_fresnel(1.0, n, k), expected, atol=1e-14)


def test_conductor_broadcast_dense_grazing_is_finite_and_bounded() -> None:
    cosine = np.linspace(0.0, 1.0, 100_001)[:, None]
    result = conductor_fresnel(cosine, [0.5508, 0.9322], [3.2986, 4.9850])
    assert result.shape == (100_001, 2)
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0.0) & (result <= 1.0))
    np.testing.assert_array_equal(result[0], 1.0)


def test_zinc_f0_white_and_resource_identities() -> None:
    np.testing.assert_allclose(
        zinc_f0(), (0.87517970, 0.86888876, 0.85507598), atol=1e-8
    )
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "src/texture_generators/data/galvanised/optics/manifest.json"
        ).read_text()
    )
    _, white = _independent_spectral_reference(np.asarray([1.0]))
    assert np.max(np.abs(white - np.mean(white))) < 1e-3
    hashes = resource_hashes()
    assert set(hashes) == {
        "Werner.yml",
        "CIE_std_illum_D65.csv",
        "CIE_std_illum_D65.csv_metadata_v2.json",
        "CIE_xyz_1931_2deg.csv",
        "CIE_xyz_1931_2deg.csv_metadata.json",
        "zinc_fresnel_lut.csv",
        "manifest.json",
    }
    with pytest.raises(TypeError):
        hashes["Werner.yml"] = "changed"  # type: ignore[index]
    source_root = (
        Path(__file__).parents[1]
        / "src/texture_generators/data/galvanised/optics/sources"
    )
    for name, expected in manifest["source_sha256"].items():
        assert sha256((source_root / name).read_bytes()).hexdigest() == expected


def test_angular_lut_interpolation_against_independent_spectral_reference() -> None:
    cosine = np.linspace(0.0, 1.0, 2049) ** 1.7
    reference, _ = _independent_spectral_reference(cosine)
    actual = zinc_fresnel(cosine)
    assert np.max(np.abs(actual - reference)) < 1e-3


def test_openpbr_width_mapping_and_floor() -> None:
    alpha_t, alpha_b = openpbr_widths(0.5, 0.6)
    assert alpha_t == pytest.approx(0.25 * np.sqrt(2.0 / 1.16))
    assert alpha_b == pytest.approx(0.4 * alpha_t)
    floor_t, floor_b = openpbr_widths([0.0, 1.0], [0.0, 1.0])
    np.testing.assert_array_equal(floor_t >= 1e-4, True)
    np.testing.assert_array_equal(floor_b >= 1e-4, True)


def test_tangent_frame_clockwise_axis_projection_and_fallback() -> None:
    theta = np.deg2rad(35.0)
    doubled = [np.cos(2.0 * theta), np.sin(2.0 * theta)]
    normal = np.asarray([0.25, -0.1, 0.9630680142])
    normal /= np.linalg.norm(normal)
    tangent, bitangent = tangent_frame(normal, doubled)
    desired = np.asarray([np.cos(theta), -np.sin(theta), 0.0])
    desired -= np.dot(desired, normal) * normal
    desired /= np.linalg.norm(desired)
    np.testing.assert_allclose(tangent, desired, atol=1e-14)
    np.testing.assert_allclose(np.dot(tangent, normal), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.cross(normal, tangent), bitangent, atol=1e-14)

    fallback_t, fallback_b = tangent_frame([1.0, 0.0, 0.0], [1.0, 0.0])
    np.testing.assert_allclose(np.dot(fallback_t, [1.0, 0.0, 0.0]), 0.0)
    np.testing.assert_allclose(np.linalg.norm(fallback_b), 1.0)


def test_ggx_reciprocity_and_pi_tangent_symmetry() -> None:
    normal = np.asarray([0.0, 0.0, 1.0])
    tangent = np.asarray([0.8, 0.6, 0.0])
    wi = _direction(0.62, 0.31)
    wo = _direction(0.91, 2.24)
    forward = evaluate_ggx(normal, tangent, wi, wo, 0.24, 0.58, zinc_fresnel)
    reverse = evaluate_ggx(normal, tangent, wo, wi, 0.24, 0.58, zinc_fresnel)
    reversed_axis = evaluate_ggx(normal, -tangent, wi, wo, 0.24, 0.58, zinc_fresnel)
    np.testing.assert_allclose(forward, reverse, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(forward, reversed_axis, rtol=2e-14, atol=2e-14)


def test_isotropic_ggx_is_rotation_invariant() -> None:
    wi = _direction(0.47, 0.2)
    wo = _direction(0.81, 1.3)
    angle = 1.17
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    original = evaluate_ggx([0, 0, 1], [1, 0, 0], wi, wo, 0.37, 0.37, zinc_fresnel)
    rotated = evaluate_ggx(
        [0, 0, 1],
        rotation @ [1, 0, 0],
        rotation @ wi,
        rotation @ wo,
        0.37,
        0.37,
        zinc_fresnel,
    )
    np.testing.assert_allclose(original, rotated, rtol=2e-14, atol=2e-14)


def test_ggx_gates_backfaces_and_remains_finite_at_grazing() -> None:
    values = evaluate_ggx(
        [0, 0, 1],
        [1, 0, 0],
        [[0, 0, -1], [1.0, 0.0, 1e-14]],
        [[0, 0, 1], [-1.0, 0.0, 1e-14]],
        0.2,
        0.6,
        zinc_fresnel,
    )
    np.testing.assert_array_equal(values[0], 0.0)
    assert np.all(np.isfinite(values[1]))
    assert np.all(values[1] >= 0.0)


@pytest.mark.parametrize("roughness,anisotropy", [(0.15, 0.0), (0.45, 0.7), (0.9, 0.4)])
def test_single_scattering_integrated_energy_does_not_exceed_one(
    roughness: float, anisotropy: float
) -> None:
    # Tensor-product Gauss-Legendre in mu=cos(theta), uniform periodic azimuth.
    mu, weights = np.polynomial.legendre.leggauss(80)
    mu = 0.5 * (mu + 1.0)
    weights = 0.5 * weights
    phi = 2.0 * np.pi * np.arange(192) / 192.0
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing="ij")
    radius = np.sqrt(1.0 - mu_grid * mu_grid)
    wo = np.stack(
        (radius * np.cos(phi_grid), radius * np.sin(phi_grid), mu_grid), axis=-1
    )
    alpha_t, alpha_b = openpbr_widths(roughness, anisotropy)
    brdf = evaluate_ggx(
        [0, 0, 1],
        [1, 0, 0],
        _direction(0.63, 0.37),
        wo,
        alpha_t,
        alpha_b,
        _constant_white,
    )
    energy = np.sum(brdf * mu_grid[..., None] * weights[:, None, None], axis=(0, 1)) * (
        2.0 * np.pi / 192.0
    )
    assert np.all(energy <= 1.005)
