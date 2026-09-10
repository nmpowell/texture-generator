"""Tests for the W1 shading-core package of the wood-optics change.

Covers the design doc's ("wood-optics-design.md", section 4) numbered
validation rows that belong to the shared shading core rather than to
``materials/wood.py``: row 1 (axial sign), row 2 (coat/fibre separation,
the ``shade``-only half), row 4 (zero coverage / equal IOR, the
``shade``-only half -- the wood-specific ``FINISHES`` half is W3's), row 7
(analytic height), row 8 (fibre peak locus), row 9 (Snell), row 10 (ray
isolation), plus the band-limited-fbm behaviour introduced by D6 (no
numbered row of its own).

Untestable here, per the design doc's own list (section 4, last
paragraph): reciprocity (the view is fixed), the white-furnace check (no
integrator; ``conserve_energy``'s mean test is the analogue), Snell across
total internal reflection (only air-to-coat is modelled here), grazing
bump vs displacement (there is no displacement), angular hold-out and
reference convergence (no measured data or estimator), and Beer
monotonicity (no film-thickness field -- see design doc D6/5.4).
"""

from __future__ import annotations

import numpy as np
import pytest

from texture_generators.core.fields import height_to_normal
from texture_generators.core.noise import fbm_at
from texture_generators.core.shading import shade


def _box_blur_3px(a: np.ndarray) -> np.ndarray:
    """3-tap box blur (wrap-around), used only to split low from high frequency."""
    acc = np.zeros_like(a)
    for shift in (-1, 0, 1):
        acc += np.roll(a, shift, axis=1)
    acc = acc / 3.0
    out = np.zeros_like(acc)
    for shift in (-1, 0, 1):
        out += np.roll(acc, shift, axis=0)
    return out / 3.0


def test_normals_follow_physical_slopes_at_any_sampling() -> None:
    """A height field's normal must depend on its *physical* slope, not the grid pitch.

    Sampling the same plane sparsely and finely must give the same normal --
    the gradient has to be taken in the height field's own length units, not
    per texel -- and a sinusoid's normal has to match its calculus derivative,
    not just "some" gradient estimate.
    """
    expected_plane = np.asarray([-0.2, 0.0, 1.0], dtype=np.float64)
    expected_plane = expected_plane / np.linalg.norm(expected_plane)

    for spacing in (0.1, 0.4):
        size = 16
        x_mm = np.arange(size, dtype=np.float32) * np.float32(spacing)
        h = np.broadcast_to(0.2 * x_mm, (size, size)).astype(np.float32)

        normals = height_to_normal(h, strength=1.0, spacing=spacing)

        interior = normals[2:-2, 2:-2, :].astype(np.float64)
        assert np.abs(interior - expected_plane).max() < 1e-5, spacing

    spacing = 0.05
    size = 64
    x_mm = np.arange(size, dtype=np.float32) * np.float32(spacing)
    h = np.broadcast_to(0.05 * np.sin(2.0 * np.pi * x_mm / 2.0), (size, size)).astype(
        np.float32
    )

    normals = height_to_normal(h, strength=1.0, spacing=spacing)

    dhdx = 0.05 * (2.0 * np.pi / 2.0) * np.cos(2.0 * np.pi * x_mm / 2.0)
    expected_nx_row = (-dhdx) / np.sqrt(dhdx * dhdx + 1.0)
    expected_nx = np.broadcast_to(expected_nx_row, (size, size))

    interior = normals[2:-2, 2:-2, 0].astype(np.float64)
    assert np.abs(interior - expected_nx[2:-2, 2:-2]).max() < 2e-3


def test_surface_lobe_ignores_the_fibre_dip_while_the_fibre_lobe_follows_it() -> None:
    """The surface (Blinn-Phong) lobes must not see the fibre tangent's dip.

    Only the fibre lobe reads the tangent's out-of-plane component -- that is
    the whole point of having a separate lobe. So two boards that differ only
    in that dip must shade identically with the fibre lobe off, and must
    differ once it is on.
    """
    size = 10
    xx, yy = np.meshgrid(
        np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32)
    )
    height = 0.02 * np.sin(xx) + 0.015 * np.cos(yy)
    albedo = np.full((size, size, 3), 0.5, dtype=np.float32)

    tangent_flat = np.zeros((size, size, 3), dtype=np.float32)
    tangent_flat[..., 0] = 1.0
    dip = np.deg2rad(15.0)
    tangent_dipped = np.zeros((size, size, 3), dtype=np.float32)
    tangent_dipped[..., 0] = np.cos(dip)
    tangent_dipped[..., 2] = -np.sin(dip)

    surface_kwargs = dict(
        specular=0.25,
        shininess=40.0,
        aniso=0.4,
        aniso_dir=(1.0, 0.0),
        fibre_exponent=50.0,
    )

    surface_flat = shade(
        albedo, height, fibre_tangent=tangent_flat, fibre=0.0, **surface_kwargs
    )
    surface_dipped = shade(
        albedo, height, fibre_tangent=tangent_dipped, fibre=0.0, **surface_kwargs
    )
    assert np.array_equal(surface_flat, surface_dipped)

    full_flat = shade(
        albedo, height, fibre_tangent=tangent_flat, fibre=0.3, **surface_kwargs
    )
    full_dipped = shade(
        albedo, height, fibre_tangent=tangent_dipped, fibre=0.3, **surface_kwargs
    )
    fibre_part_flat = full_flat.astype(np.float64) - surface_flat.astype(np.float64)
    fibre_part_dipped = full_dipped.astype(np.float64) - surface_dipped.astype(
        np.float64
    )
    assert np.abs(fibre_part_flat - fibre_part_dipped).max() > 1.0 / 255.0


def test_coat_height_equal_to_the_substrate_changes_nothing() -> None:
    """A film that follows the wood exactly must render exactly like no film.

    ``coat_height`` only matters when it *differs* from the substrate; handing
    it back the same height field is the identity case and has to reproduce
    the bare-substrate render bit for bit -- otherwise the coat path is doing
    something other than "use this height instead".
    """
    size = 8
    xx, yy = np.meshgrid(
        np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32)
    )
    height = 0.02 * np.sin(xx) + 0.01 * yy
    albedo = np.full((size, size, 3), 0.45, dtype=np.float32)
    tangent = np.zeros((size, size, 3), dtype=np.float32)
    tangent[..., 0] = 1.0

    kwargs = dict(
        specular=0.3,
        shininess=35.0,
        aniso=0.3,
        aniso_dir=(1.0, 0.2),
        fibre_tangent=tangent,
        fibre=0.2,
    )

    base = shade(albedo, height, **kwargs)
    with_coat = shade(albedo, height, coat_height=height, **kwargs)
    assert np.array_equal(base, with_coat)


def test_fibre_ior_below_one_is_refused() -> None:
    """A coat cannot be optically thinner than air: ``fibre_ior`` is a lower bound of 1.

    Snell's construction here (``x, y`` divided by the index, ``z`` filled in
    from the unit-length constraint) only describes light entering a *denser*
    medium; below 1.0 it is not modelling a coat any more, and the mistake is
    much easier to catch here than downstream in a NaN.
    """
    albedo = np.zeros((2, 2, 3), dtype=np.float32)
    height = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="fibre_ior"):
        shade(albedo, height, fibre_ior=0.9)


def test_grazing_light_refracts_without_nan() -> None:
    """Snell's construction must not blow up as the light approaches the horizon.

    At 89 degrees from the normal the light is almost entirely tangential;
    dividing that tangential part by an index > 1 still has to leave a
    non-negative quantity under the square root for ``z``.
    """
    theta = np.deg2rad(89.0)
    light_dir = (float(np.sin(theta)), 0.0, float(np.cos(theta)))
    size = 6
    height = np.zeros((size, size), dtype=np.float32)
    albedo = np.full((size, size, 3), 0.4, dtype=np.float32)
    dip = np.deg2rad(20.0)
    tangent = np.zeros((size, size, 3), dtype=np.float32)
    tangent[..., 0] = np.cos(dip)
    tangent[..., 2] = -np.sin(dip)

    out = shade(
        albedo,
        height,
        light_dir=light_dir,
        fibre_tangent=tangent,
        fibre=0.5,
        fibre_exponent=40.0,
        fibre_ior=1.5,
    )

    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_refraction_moves_the_fibre_peak_toward_the_surface() -> None:
    """Refraction bends the light toward the normal, so the fibre lobe's peak dip must shrink.

    The lobe peaks where the (refracted) light's and the view's inclinations
    to the fibre axis are equal and opposite, i.e. where
    ``tan(dip) = s_x / (1 + s_z)`` for the refracted unit light ``s``. That is
    an oracle independent of ``_fibre_lobe``'s implementation -- it comes
    straight from the peak condition, not from running the code.
    """
    light_dir = np.asarray([-0.5, -0.55, 0.78], dtype=np.float64)
    light_dir = light_dir / np.linalg.norm(light_dir)

    def expected_peak_deg(fibre_ior: float) -> float:
        sx = light_dir[0] / fibre_ior
        sy = light_dir[1] / fibre_ior
        sz = np.sqrt(1.0 - sx * sx - sy * sy)
        return float(np.degrees(np.arctan(sx / (1.0 + sz))))

    dips = np.arange(-40.0, 40.0 + 1e-9, 0.02)
    # A 2x2 board rather than a literal 1x1: numpy's central-difference
    # gradient needs at least two samples per axis, and both give the same
    # "flat board" the spec asks for -- constant height, one uniform tangent.
    albedo = np.zeros((2, 2, 3), dtype=np.float32)
    height = np.zeros((2, 2), dtype=np.float32)

    for fibre_ior in (1.0, 1.5):
        luminance = []
        for dip_deg in dips:
            th = np.deg2rad(dip_deg)
            tangent = np.zeros((2, 2, 3), dtype=np.float32)
            tangent[..., 0] = np.cos(th)
            tangent[..., 2] = -np.sin(th)
            _, specular = shade(
                albedo,
                height,
                light_dir=tuple(light_dir),
                fibre_tangent=tangent,
                fibre=1.0,
                fibre_exponent=60.0,
                fibre_ior=fibre_ior,
                return_parts=True,
            )
            luminance.append(float(specular[0, 0].mean()))

        argmax_dip = float(dips[int(np.argmax(luminance))])
        expected = expected_peak_deg(fibre_ior)
        assert abs(argmax_dip - expected) < 0.25, (fibre_ior, argmax_dip, expected)


def test_ray_population_changes_only_the_flecks() -> None:
    """The ray lobe must only show up where ``ray_weight`` says a fleck is.

    Outside the mask the ray term contributes nothing -- ``w = 0`` there, so
    the fibre term reduces to exactly its old form -- and inside it, mixing
    in a lobe around a different axis has to change the render.
    """
    size = 12
    xx, yy = np.meshgrid(
        np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32)
    )
    height = 0.02 * np.sin(xx) + 0.01 * yy
    albedo = np.full((size, size, 3), 0.4, dtype=np.float32)
    tangent = np.zeros((size, size, 3), dtype=np.float32)
    tangent[..., 0] = 1.0
    ray_tangent = np.zeros((size, size, 3), dtype=np.float32)
    ray_tangent[..., 1] = 1.0
    mask = np.zeros((size, size), dtype=np.float32)
    mask[4:8, 3:9] = 1.0

    kwargs = dict(
        fibre_tangent=tangent,
        fibre=0.3,
        fibre_exponent=45.0,
        ray_tangent=ray_tangent,
        ray_gain=1.25,
    )

    with_rays = shade(albedo, height, ray_weight=mask, **kwargs)
    without_rays = shade(albedo, height, ray_weight=0.0, **kwargs)

    outside = mask < 0.5
    inside = mask >= 0.5
    assert np.array_equal(with_rays[outside], without_rays[outside])
    diff_inside = np.abs(
        with_rays[inside].astype(np.float64) - without_rays[inside].astype(np.float64)
    )
    assert diff_inside.max() > 1.0 / 255.0


def test_fibre_lobe_is_invariant_to_the_director_sign() -> None:
    """The fibre director has no arrowhead: ``u`` and ``-u`` describe the same fibre.

    Both ``_fibre_lobe`` (built from ``cos`` of a half angle) and the
    anisotropic surface lobe (built from squared tangent-plane projections)
    are even functions of the tangent, so reversing every director in the
    field -- the fibre tangent, its aniso_dir companion -- must leave the
    shaded image bit for bit unchanged, with specular, aniso, the fibre lobe
    and the ray population all switched on together.
    """
    size = 12
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dip = np.deg2rad(-25.0 + (50.0 / (size - 1)) * yy)
    azimuth = np.deg2rad((300.0 / (size - 1)) * xx)
    tx = np.cos(dip) * np.cos(azimuth)
    ty = np.cos(dip) * np.sin(azimuth)
    tz = -np.sin(dip)
    tangent = np.stack([tx, ty, tz], axis=-1).astype(np.float32)
    aniso_dir = np.stack([tx, ty], axis=-1).astype(np.float32)
    ray_tangent = np.stack([-ty, tx, np.zeros_like(tx)], axis=-1).astype(np.float32)
    ray_weight = np.where(xx >= size / 2, 1.0, 0.0).astype(np.float32)
    albedo = 0.3 + 0.4 * (xx / size)[..., None] * np.ones(3, dtype=np.float32)
    height = 0.05 * np.sin(2.0 * np.pi * xx / size) + 0.03 * yy

    kwargs = dict(
        specular=0.3,
        shininess=45.0,
        aniso=0.5,
        fibre=0.4,
        fibre_exponent=50.0,
        ray_weight=ray_weight,
        ray_gain=1.3,
    )

    positive = shade(
        albedo,
        height,
        fibre_tangent=tangent,
        aniso_dir=aniso_dir,
        ray_tangent=ray_tangent,
        **kwargs,
    )
    negative = shade(
        albedo,
        height,
        fibre_tangent=-tangent,
        aniso_dir=-aniso_dir,
        ray_tangent=ray_tangent,
        **kwargs,
    )

    assert np.array_equal(positive, negative)


def test_band_limited_fbm_drops_octaves_the_grid_cannot_carry() -> None:
    """An octave finer than the grid's Nyquist limit must be faded out, not aliased.

    A grid of 64 px can carry at most 32 cycles per unit; asking for octaves
    up to 200 cycles without a limit lets them alias into extra high-frequency
    energy, while ``max_freq`` should fade that content away.

    NOTE on the second half of this test: with ``freq=(4, 200)`` *every*
    octave's ``f_k = max(fx_k, fy_k)`` is already at or past 200 cycles/unit
    at octave 0 (200*2^k only grows), so all three octaves get the exact
    zero weight, and ``limited`` collapses to exactly zero -- there is no
    surviving low-frequency content left to compare between the two calls to
    1e-3, only the (correct) total absence of one. The "same total-amplitude
    normalisation" requirement -- that dropped energy is dropped rather than
    the divisor being rebalanced to compensate -- is instead checked with a
    second, lower-frequency pair chosen so that exactly one octave survives.
    """
    size = 64
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        np.linspace(0.0, 1.0, size, dtype=np.float32),
    )

    full = fbm_at(x, y, np.random.default_rng(3), freq=(4.0, 200.0), octaves=3)
    limited = fbm_at(
        x, y, np.random.default_rng(3), freq=(4.0, 200.0), octaves=3, max_freq=32.0
    )

    assert limited.shape == full.shape
    assert limited.dtype == full.dtype

    high_full = float((full - _box_blur_3px(full)).std())
    high_limited = float((limited - _box_blur_3px(limited)).std())
    assert high_limited < high_full
    # Every octave's frequency is at or beyond max_freq from the very first
    # octave, so the Hermite falloff is exactly 0.0 (not merely small) for
    # all three -- the band-limited call has to be exactly the zero field.
    assert np.array_equal(limited, np.zeros_like(limited))

    # Unweighted-norm check: freq=2.0 with max_freq=4.0 lets octave 0 (freq 2)
    # through at full weight and drops octaves 1 and 2 (freq 4, 8) entirely,
    # so the surviving term is octave 0's noise divided by the *three-octave*
    # amplitude sum (1 + 0.5 + 0.25 = 1.75) -- not by 1.0, which is what a
    # renormalising implementation would use once only one octave survives.
    rng_solo = np.random.default_rng(9)
    solo = fbm_at(x, y, rng_solo, freq=2.0, octaves=1)
    rng_stacked = np.random.default_rng(9)
    stacked = fbm_at(x, y, rng_stacked, freq=2.0, octaves=3, max_freq=4.0)
    three_octave_norm = np.float32(1.0 + 0.5 + 0.25)
    expected_stacked = solo / three_octave_norm
    assert np.abs(stacked - expected_stacked).max() < 1e-5


def test_shade_defaults_are_bit_identical_with_the_new_parameters() -> None:
    """Every parameter this package added must default to the pre-existing code path.

    That is the whole point of the "bit-identical" clause in each decision:
    nothing rendered before this package existed may shift by even one bit
    once the new parameters are threaded through, as long as callers do not
    ask for the new behaviour.
    """
    size = 9
    xx, yy = np.meshgrid(
        np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32)
    )
    height = 0.03 * np.sin(xx) + 0.02 * yy
    albedo = 0.4 + 0.1 * np.stack([xx, yy, xx + yy], axis=-1) / size
    tangent = np.zeros((size, size, 3), dtype=np.float32)
    dip = np.deg2rad(10.0)
    tangent[..., 0] = np.cos(dip)
    tangent[..., 2] = -np.sin(dip)

    kwargs = dict(
        specular=0.3,
        shininess=40.0,
        aniso=0.4,
        aniso_dir=(1.0, 0.3),
        specular2=0.15,
        shininess2=200.0,
        spec_tint=0.5,
        cavity=0.3,
        fibre_tangent=tangent,
        fibre=0.25,
        fibre_exponent=55.0,
        conserve_energy=True,
        normalise=True,
    )

    baseline = shade(albedo, height, **kwargs)
    extended = shade(
        albedo,
        height,
        height_spacing=1.0,
        coat_height=None,
        fibre_ior=1.0,
        ray_tangent=None,
        **kwargs,
    )
    assert np.array_equal(baseline, extended)

    h = np.asarray(
        [[0.0, 0.1, 0.4], [0.2, 0.5, 0.9], [0.3, 0.6, 1.2]], dtype=np.float32
    )
    assert np.array_equal(
        height_to_normal(h, 1.3, spacing=1.0), height_to_normal(h, 1.3)
    )

    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, 16, dtype=np.float32),
        np.linspace(0.0, 1.0, 16, dtype=np.float32),
    )
    assert np.array_equal(
        fbm_at(
            x, y, np.random.default_rng(7), freq=(3.0, 5.0), octaves=4, max_freq=None
        ),
        fbm_at(x, y, np.random.default_rng(7), freq=(3.0, 5.0), octaves=4),
    )
