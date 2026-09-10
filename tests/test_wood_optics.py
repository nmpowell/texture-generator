"""Tests for the wood-optics rendering change: coat, refraction, rays, physical
relief and linear-light compositing, plus the shared shading-core machinery
they are built on.

Each test is named for the row of the finished-wood validation matrix
(the material-development tests that go with a Marschner-style fibre-lobe
model: axial sign, coat/fibre separation, pore pooling and so on) that it
stands in for, and checks it against an oracle
independent of the implementation -- a hand-derived literal, a closed-form
formula, or an exact invariant -- never by calling the code under test to
produce its own "expected" answer. Rows covered here, and what stands in for
a measured reference in a fixed-view, 2-D texture generator:

* **Axial sign** (`test_fibre_lobe_is_invariant_to_the_director_sign`) -- a
  fibre direction has no arrowhead, so reversing every director in the field
  must render bit for bit unchanged.
* **Coat/fibre separation** (`test_surface_lobe_ignores_the_fibre_dip_while_the_fibre_lobe_follows_it`,
  `test_coat_height_equal_to_the_substrate_changes_nothing`) -- the surface
  (Blinn-Phong) lobes must not see the fibre tangent's out-of-plane dip, and a
  coat that exactly follows the substrate must shade identically to no coat.
* **Figure reversal** (`test_figure_still_moves_when_the_albedo_is_flat`) --
  curly figure has to live in the reflection: strip a curly board's albedo to
  its own flat mean and the luster still has to move and its across-grain
  peak still has to shift as the light swings.
* **Zero coverage / layer removal / equal IOR**
  (`test_fibre_ior_below_one_is_refused`,
  `test_finish_none_recovers_the_bare_board_exactly`) -- the bare ``"none"``
  finish must be the true zero-coverage case (no film, no tint, no Lab
  shift, no refractive boundary), and an out-of-range index is refused
  outright rather than silently misbehaving downstream.
* **Pore pooling** (`test_film_fills_a_pore_no_deeper_than_its_build`) -- a
  finish film pools in a pore but never deeper than the film it built, and
  self-levels the coat surface smoother than the substrate underneath it.
* **Beer analogue** (`test_finish_state_is_ordered_by_film_build`) -- no
  measured film-thickness/absorption field exists here (see "untestable"
  below), so the testable analogue is that the fitted finish state -- film
  build, fibre tint -- is internally ordered and every tint stays neutral
  energy.
* **Analytic height** (`test_normals_follow_physical_slopes_at_any_sampling`)
  -- a plane and a sinusoid sampled at two different pitches must give the
  same normal, matching the calculus derivative exactly, because the
  gradient is taken in the height field's own physical units and not per
  texel.
* **Fibre peak locus** (`test_refraction_moves_the_fibre_peak_toward_the_surface`)
  -- refracting the light into the coat has to move the fibre lobe's peak dip
  to exactly where ``tan(dip) = s_x / (1 + s_z)`` places it, for the
  refracted unit light ``s``.
* **Snell** (`test_grazing_light_refracts_without_nan`) -- the refraction
  construction must stay finite and unit-length as the light grazes the
  horizon.
* **Ray isolation** (`test_ray_population_changes_only_the_flecks`) -- mixing
  in a ray lobe must change the render only where the ray weight is nonzero.
* **Mip sweep** (`test_a_small_render_matches_a_downsampled_large_one`) -- a
  small direct render and a large render box-downsampled to the same size in
  linear light must agree in mean and in contrast; this stands in for the
  matrix's reference-convergence check with a self-consistency one instead
  (see "untestable" below).
* **Linear colour** (`test_wood_shades_in_linear_light`) -- a flat, lobe-less
  board must round-trip to its own albedo, and a two-slope height ramp's
  bright/dark contrast, decoded back to linear radiance, must match the
  underlying Lambertian dot products rather than their gamma-compressed
  shadow.
* **Analytic deformation** (`test_analytic_shear_transports_the_axis_by_the_inverse_jacobian`)
  -- for a warp with a closed-form Jacobian, the fibre tangent's in-plane
  direction must equal ``normalize(J_f^-1 e_u)`` computed straight from that
  Jacobian, not from the code under test.

Plus the band-limited-fbm behaviour introduced for physical relief
(`test_band_limited_fbm_drops_octaves_the_grid_cannot_carry`, no matrix
row of its own) and the cross-cutting defaults-are-inert acceptance
check (`test_shade_defaults_are_bit_identical_with_the_new_parameters`).

**Untestable here**, and why a fixed-view 2-D texture generator cannot
validate them: reciprocity (the view is fixed at +z, so there is no second
viewing angle to swap with the light); the white-furnace check (there is no
integrator to converge to unit reflectance under uniform illumination --
``conserve_energy``'s mean-preservation test is the nearest analogue); Snell
across total internal reflection (only the air-to-coat boundary is modelled,
never the denser-to-rarer direction that can totally internally reflect);
grazing bump vs. displacement parallax (there is no true displacement to
compare a bump against); angular hold-out and reference-path convergence
(there is no measured BRDF data or Monte Carlo estimator to hold against);
and Beer-law film absorption monotonicity (there is no film-thickness or
absorption-coefficient field -- the fitted CIELAB finish deltas and the
per-finish fibre tint are declared as an appearance approximation instead,
not a physical stand-in for one, and the film-build ordering above is the
closest testable analogue).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from texture_generators.core.colour import linear_to_srgb, srgb_to_linear
from texture_generators.core.fields import height_to_normal
from texture_generators.core.noise import fbm_at
from texture_generators.core.shading import shade
from texture_generators.materials import wood

LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _luminance(img: np.ndarray) -> np.ndarray:
    return (img @ LUMA).astype(np.float32)


def _box_downsample_linear(img: np.ndarray, factor: int) -> np.ndarray:
    """Box-downsample an sRGB image by ``factor``, averaging in linear light."""
    lin = srgb_to_linear(img)
    h, w, c = lin.shape
    down = lin.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3))
    return linear_to_srgb(down)


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


def test_a_small_render_matches_a_downsampled_large_one() -> None:
    """A direct small render must agree with a box-downsampled large one.

    Real anatomy looks the same whatever the sampling grid: a 500 px photo of
    a board and a 2000 px photo of the same board, box-downsampled in linear
    light, show the same mean colour and very nearly the same contrast. A
    height or albedo layer that is authored *at* the grid pitch -- rather
    than carrying physical content the grid merely resolves less of -- fails
    this, because the direct small render keeps re-drawing that content at its
    own coarse pitch instead of losing it the way a real downsample would.

    Seeds 4 and 9 (rather than the first two) are the regression guard here:
    the mip artifact this test targets is real but seed-dependent through the
    board's own random cut/fleck/knot draws (measured directly against
    ``main`` before this package: ratios from 1.09 to 1.55 across ten seeds,
    all biased above 1.0), and seeds 0/1 happen to land inside the [0.75,
    1.33] band on the unmodified code (1.09, 1.26) -- which would make the
    test pass vacuously before the fix it exists to guard.
    """
    for seed in (4, 9):
        rng_small = np.random.default_rng(seed)
        small = wood.generate(
            (256, 256),
            rng_small,
            "board",
            species="oak",
            finish="oil",
            colour_variation=0.0,
            knots=0.0,
            sapwood=0.0,
            mm_across=200.0,
        )
        rng_large = np.random.default_rng(seed)
        large = wood.generate(
            (1024, 1024),
            rng_large,
            "board",
            species="oak",
            finish="oil",
            colour_variation=0.0,
            knots=0.0,
            sapwood=0.0,
            mm_across=200.0,
        )
        down = _box_downsample_linear(large, 4)

        mean_err = np.abs(
            small.reshape(-1, 3).mean(axis=0) - down.reshape(-1, 3).mean(axis=0)
        )
        assert float(mean_err.max()) * 255.0 < 3.0, (seed, mean_err * 255.0)

        small_std = float(_luminance(small).std())
        down_std = float(_luminance(down).std())
        ratio = small_std / max(down_std, 1e-9)
        assert 0.75 <= ratio <= 1.33, (seed, small_std, down_std, ratio)


def test_wood_shades_in_linear_light() -> None:
    """``_shade_fields`` must composite in linear light, not display sRGB.

    Two independent checks. A flat, lobe-less board must render back to (very
    nearly) the albedo it was given: a spatially constant lighting field
    normalises to exactly 1 whichever space the multiply happens in, so this
    half holds either way and is not the discriminating one. The
    discriminating half is the diffuse contrast of a two-slope height ramp:
    its bright/dark ratio, decoded back to linear radiance, has to match the
    ratio of the underlying Lambertian dot products -- computed here straight
    from the light and normal geometry, independently of ``shade`` -- not the
    gamma-compressed ratio a display-space multiply produces.
    """
    size = 24
    px_per_mm = 4.0
    light_dir = (-0.5, -0.55, 0.78)

    # --- Part 1: flat board, no lobes, normal_strength=0 -> exact albedo ---
    albedo = np.full((size, size, 3), (0.62, 0.35, 0.20), dtype=np.float32)
    flat_fields = wood.BoardFields(
        albedo=albedo,
        height=np.zeros((size, size), dtype=np.float32),
        coat_height=np.zeros((size, size), dtype=np.float32),
        tangent=np.zeros((size, size, 3), dtype=np.float32),
        ray_tangent=np.zeros((size, size, 3), dtype=np.float32),
        ray_weight=np.zeros((size, size), dtype=np.float32),
        coat_gloss=np.zeros((size, size), dtype=np.float32),
        fibre_lustre=np.zeros((size, size), dtype=np.float32),
    )
    out_flat = wood._shade_fields(
        flat_fields,
        finish="none",
        rng=np.random.default_rng(0),
        params={"normal_strength": 0.0, "specular": 0.0},
        px_per_mm=px_per_mm,
        light_dir=light_dir,
    )
    err_255 = np.abs(out_flat.astype(np.float64) - albedo.astype(np.float64)) * 255.0
    assert float(err_255.max()) < 1.0, float(err_255.max())

    # --- Part 2: two-slope ramp, symmetric contrast in linear light -------
    m = 0.35  # dimensionless slope, mm of rise per mm of run
    x_mm = (np.arange(size, dtype=np.float32) / np.float32(px_per_mm))[None, :]
    top_band = np.arange(size)[:, None] < size // 2
    slope = np.where(top_band, np.float32(m), np.float32(-m)).astype(np.float32)
    height = (slope * x_mm).astype(np.float32)
    ramp_fields = wood.BoardFields(
        albedo=albedo,
        height=height,
        coat_height=height,
        tangent=np.zeros((size, size, 3), dtype=np.float32),
        ray_tangent=np.zeros((size, size, 3), dtype=np.float32),
        ray_weight=np.zeros((size, size), dtype=np.float32),
        coat_gloss=np.zeros((size, size), dtype=np.float32),
        fibre_lustre=np.zeros((size, size), dtype=np.float32),
    )
    out_ramp = wood._shade_fields(
        ramp_fields,
        finish="none",
        rng=np.random.default_rng(0),
        params={"normal_strength": 1.0, "specular": 0.0},
        px_per_mm=px_per_mm,
        light_dir=light_dir,
    )

    # Interior windows a few rows/columns clear of the band boundary (row
    # size//2) and the non-periodic array edges, where the height field is
    # exactly linear and the analytic normal below is exact.
    top = out_ramp[3 : size // 2 - 3, 3:-3, :]
    bottom = out_ramp[size // 2 + 3 : -3, 3:-3, :]

    def _analytic_ndl(dhdx: float) -> float:
        n = np.asarray([-1.0 * dhdx, 0.0, 1.0], dtype=np.float64)
        n = n / np.linalg.norm(n)
        light = np.asarray(light_dir, dtype=np.float64)
        light = light / np.linalg.norm(light)
        return float(np.clip(np.dot(n, light), 0.0, 1.0)), float(n[2])

    ambient = 0.62  # the constant _shade_fields itself passes to shade()
    ndl_top, nz_top = _analytic_ndl(m)
    ndl_bot, nz_bot = _analytic_ndl(-m)
    lit_top = ambient * (0.85 + 0.15 * nz_top) + (1.0 - ambient) * ndl_top
    lit_bot = ambient * (0.85 + 0.15 * nz_bot) + (1.0 - ambient) * ndl_bot
    expected_ratio = lit_top / lit_bot

    albedo_linear = float(srgb_to_linear(albedo[0, 0, 0:1])[0])
    top_lit = srgb_to_linear(top).astype(np.float64) / albedo_linear
    bottom_lit = srgb_to_linear(bottom).astype(np.float64) / albedo_linear
    measured_ratio = float(top_lit.mean() / bottom_lit.mean())

    assert abs(measured_ratio - expected_ratio) < 0.01, (measured_ratio, expected_ratio)


def test_finish_none_recovers_the_bare_board_exactly() -> None:
    """The bare ``"none"`` finish must be the true zero-coverage case.

    Every one of the new finish properties has to collapse to "no coat"
    for ``finish="none"``: no film, no fibre tint, and no Lab shift -- the
    board this finish describes was never coated at all.
    """
    none = wood.FINISHES["none"]
    assert none["film_um"] == (0.0, 0.0)
    assert none["fibre_tint"] == (1.0, 1.0, 1.0)
    assert none["ior"] == 1.0

    lab = np.asarray(wood.SPECIES["oak"]["lab"], dtype=np.float64)
    delta = np.asarray(
        [np.mean(none["dl"]), np.mean(none["dc"]), np.mean(none["db"])],
        dtype=np.float64,
    )
    assert np.array_equal(wood._finish_lab(lab, delta), lab)

    fields = wood._board_fields(
        (64, 64),
        np.random.default_rng(3),
        "oak",
        finish="none",
        knot_p=0.0,
        sap_p=0.0,
    )
    assert np.array_equal(fields.coat_height, fields.height)


def test_film_fills_a_pore_no_deeper_than_its_build() -> None:
    """A film pools in a pore, but never deeper than the film it built.

    ``coat_height`` is the substrate levelled by however much film sits above
    it, so the film's own contribution must stay within [0, film_max] -- and,
    because a film self-levels, the resulting coat surface has to be at least
    as smooth (lower gradient RMS) as the wood underneath it.
    """
    _film_lo, film_hi = wood.FINISHES["polyurethane"]["film_um"]
    film_max_mm = film_hi / 1000.0

    fields = wood._board_fields(
        (192, 192),
        np.random.default_rng(1),
        "oak",
        finish="polyurethane",
        knot_p=0.0,
        sap_p=0.0,
    )
    delta = (fields.coat_height - fields.height).astype(np.float64)
    assert float(delta.min()) >= -1e-6
    assert float(delta.max()) <= film_max_mm + 1e-6

    def _gradient_rms(h: np.ndarray) -> float:
        gy, gx = np.gradient(h.astype(np.float64))
        return float(np.sqrt((gx * gx + gy * gy).mean()))

    assert _gradient_rms(fields.coat_height) <= _gradient_rms(fields.height)


def test_finish_state_is_ordered_by_film_build() -> None:
    """Film build orders the finishes, and every fibre tint stays neutral energy.

    No thickness/absorption field exists here (a Beer-Lambert film term is
    deferred), so the Beer-film-
    absorption claim's testable analogue is the monotonic ordering of the
    fitted finish state across the finishes it applies to: no coat is
    thinner than bare wood, and the film thickens oil < acrylic <
    polyurethane.
    """

    def _film_mid(name: str) -> float:
        lo, hi = wood.FINISHES[name]["film_um"]
        return (lo + hi) / 2.0

    assert _film_mid("none") <= _film_mid("oil")
    assert _film_mid("oil") < _film_mid("acrylic") < _film_mid("polyurethane")

    # ``_fibre_colour`` normalises each finish's raw ``fibre_tint`` to unit
    # luminance: a constant, sub-unity *linear* albedo isolates that
    # normalisation without any channel clipping (a tint component can run
    # above 1.0 before normalisation, which a white albedo would clip and
    # break this check), so the output luminance must come back as exactly
    # ``sqrt(albedo)`` -- the tint's own luminance contributes a factor of 1.
    albedo_linear = np.full((1, 1, 3), 0.3, dtype=np.float32)
    expected_luma = float(np.sqrt(0.3))
    for name in wood.FINISHES:
        tint = wood._fibre_colour(albedo_linear, name)[0, 0]
        luma = float(0.2126 * tint[0] + 0.7152 * tint[1] + 0.0722 * tint[2])
        assert abs(luma - expected_luma) < 1e-4, (name, luma)
    assert wood.FINISHES["none"]["fibre_tint"] == (1.0, 1.0, 1.0)


def test_figure_still_moves_when_the_albedo_is_flat() -> None:
    """Curly figure has to live in the reflection, not the pigment.

    Replace a curly board's albedo with its own flat mean colour -- so the
    render's only remaining source of variation is the fibre-tangent field --
    and the across-grain luster still has to move with the light: adjacent
    light azimuths must differ by a visible amount, and the peak of the
    across-grain luminance profile has to shift rather than sit still, which a
    flat-print (pigment-only) figure could never do.
    """
    size = 192
    px_per_mm = size / 225.0
    finish = "polyurethane"  # the strongest fibre lobe of the four (SPECULAR)
    fields = wood._board_fields(
        (size, size),
        np.random.default_rng(1),
        "maple",
        finish=finish,
        figure="curly",
        knot_p=0.0,
        sap_p=0.0,
        px_per_mm=px_per_mm,
    )
    flat_mean = fields.albedo.reshape(-1, 3).mean(axis=0)
    flat_albedo = np.broadcast_to(flat_mean, fields.albedo.shape).astype(np.float32)
    flat_fields = dataclasses.replace(fields, albedo=flat_albedo)

    azimuths = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    # A low elevation swings the fibre lobe's dip term further per degree of
    # azimuth than a near-overhead light does.
    elevation = np.deg2rad(25.0)
    profiles = []
    for az in azimuths:
        light_dir = (
            float(np.cos(az) * np.cos(elevation)),
            float(np.sin(az) * np.cos(elevation)),
            float(np.sin(elevation)),
        )
        out = wood._shade_fields(
            flat_fields,
            finish=finish,
            rng=np.random.default_rng(0),
            params={},
            px_per_mm=px_per_mm,
            light_dir=light_dir,
        )
        # Collapsed along the grain (columns), so what survives is structure
        # that varies *across* it -- a figure measure, not a noise measure.
        profiles.append(_luminance(out).mean(axis=1))

    n = len(profiles)
    diffs = [
        float(np.abs(profiles[i] - profiles[(i + 1) % n]).mean()) for i in range(n)
    ]
    assert min(diffs) > 1.0 / 255.0, diffs

    peaks = {int(np.argmax(p)) for p in profiles}
    assert len(peaks) > 1, peaks


def test_analytic_shear_transports_the_axis_by_the_inverse_jacobian() -> None:
    """The in-plane fibre direction is exactly ``normalize(J_f^-1 e_u)``.

    ``wu = u + A*sin(k*v)``, ``wv = v``: a pure along-grain shear, chosen
    because its Jacobian ``J_f = [[1, A*k*cos(k*v)], [0, 1]]`` is triangular,
    so the closed form is exact rather than approximate. Applying the general
    2x2 inverse-Jacobian formula (not the fact that this particular warp makes
    it trivial) to ``e_u = (1, 0)`` gives ``(d, -c) / det`` with ``c = 0``
    identically for every ``v`` here (``wv`` has no ``u``-dependence at all),
    so the fibre direction has to come back as exactly ``(1, 0)`` -- pure
    along-grain, undeflected -- for every amplitude and wavenumber, which is
    an easy place for an implementation to leak the *other* Jacobian entry
    (``b = dwu/dv``) into the in-plane angle by mistake.
    """
    size = 96
    px_per_unit = float(size)
    x, y = np.meshgrid(
        np.arange(size, dtype=np.float32) / px_per_unit,
        np.arange(size, dtype=np.float32) / px_per_unit,
    )
    amp = 0.15
    k = 4.0 * np.pi
    wu = (x + amp * np.sin(k * y)).astype(np.float32)
    wv = y.copy()

    tangent = wood._fibre_tangents(
        x,
        y,
        wu,
        wv,
        np.random.default_rng(6),
        tilt=0.0,
        along_x=True,
        px_per_unit=px_per_unit,
        mm_per_unit=225.0,
    )

    # The general 2x2 inverse-Jacobian formula, evaluated from the warp's own
    # analytic derivatives -- not the shortcut that ``c == 0`` makes this
    # trivial.
    a = np.ones_like(y)
    b = amp * k * np.cos(k * y)
    c = np.zeros_like(y)
    d = np.ones_like(y)
    det = a * d - b * c
    jinv_eu = np.stack([d / det, -c / det], axis=-1)
    expected_angle = np.arctan2(jinv_eu[..., 1], jinv_eu[..., 0])

    actual_angle = np.arctan2(tangent[..., 1], tangent[..., 0])

    interior = np.s_[3:-3, 3:-3]
    assert np.abs(actual_angle[interior] - expected_angle[interior]).max() < 1e-4
