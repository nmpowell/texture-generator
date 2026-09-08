"""Tests for the physical primitives behind the paper generator.

These check the *statistics* rather than the pixels: a fibre network is only
worth anything if its coverage, curvature and orientation match what they
claim to, and those are all measurable. Kept small and fast.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from texture_generators import generate
from texture_generators.core.colour import lab_to_srgb, linear_to_srgb, srgb_to_linear
from texture_generators.core.fibres import (
    axial_von_mises,
    cluster_points,
    count_for_coverage,
    deposit,
)
from texture_generators.core.shading import iso_highpass
from texture_generators.core.spectral import band_field, matern_field
from texture_generators.materials.paper import VARIANTS, render_sheet


def _mean_coverage(step_px: float, seed: int = 0) -> float:
    """Mean deposited mass for a fixed fibre population at a given step size."""
    shape = (192, 192)
    return float(
        deposit(
            shape,
            np.random.default_rng(seed),
            count=600,
            length_px=24.0,
            width_px=0.6,
            length_sigma=0.0,
            kinks_per_length=0.0,
            step_px=step_px,
            max_steps=200,
        ).mean()
    )


def test_coverage_is_independent_of_step_size() -> None:
    """Deposited mass must not depend on how finely the polyline is sampled.

    Mass per sample is width * arclength, so halving the step doubles the
    sample count and halves each contribution. If this drifts, every
    coverage-derived parameter silently changes with render size.
    """
    coarse = _mean_coverage(3.0)
    fine = _mean_coverage(1.0)
    assert coarse == pytest.approx(fine, rel=0.05)


def test_count_for_coverage_hits_its_target() -> None:
    """The count from a target coverage really produces that many fibre layers."""
    shape = (256, 256)
    target = 4.0
    count = count_for_coverage(shape, target, length_px=32.0, width_px=0.8)
    mass = deposit(
        shape,
        np.random.default_rng(1),
        count=count,
        length_px=32.0,
        width_px=0.8,
        length_sigma=0.0,
        kinks_per_length=0.0,
    )
    # The end taper removes a fixed fraction of each fibre's mass, so this is
    # a "same order, right ballpark" check rather than an exact identity.
    assert 0.75 * target <= mass.mean() <= 1.05 * target


def test_axial_von_mises_matches_its_density() -> None:
    """Sampled orientations follow p(t) ~ exp(k cos 2(t - mu)), pi-periodic."""
    rng = np.random.default_rng(7)
    kappa, mu = 0.55, 0.7
    theta = axial_von_mises(rng, 400_000, mu, kappa)
    # Fold onto the axis, then compare the histogram with the density.
    folded = np.mod(theta - mu, np.pi)
    hist, edges = np.histogram(folded, bins=24, range=(0.0, np.pi), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    want = np.exp(kappa * np.cos(2.0 * centres))
    want = want / (want.mean() * np.pi)
    assert np.max(np.abs(hist - want)) < 0.05


def test_worm_like_chain_bends_but_does_not_coil() -> None:
    """Shorter persistence gives more end-to-end shortening, monotonically.

    Curl index is L/R - 1, so a stiffer fibre (larger persistence) must come
    out straighter. Straight fibres are the primary "scratch" cue, so this
    guards the thing the rewrite exists to fix.
    """
    ends = []
    for persistence in (0.4, 1.2, 6.0):
        spans = []
        for seed in range(12):
            mass = deposit(
                (256, 256),
                np.random.default_rng(seed),
                count=1,
                length_px=90.0,
                width_px=1.0,
                length_sigma=0.0,
                kappa=0.0,
                persistence=persistence,
                kinks_per_length=0.0,
                step_px=2.0,
                max_steps=200,
                positions=(np.asarray([128.0]), np.asarray([128.0])),
            )
            ys, xs = np.nonzero(mass > 0.01)
            spans.append(float(np.hypot(np.ptp(xs), np.ptp(ys))))
        ends.append(float(np.mean(spans)))
    assert ends[0] < ends[1] < ends[2]


def test_matern_field_is_seamless_and_normalised() -> None:
    """FFT-shaped noise tiles exactly and comes out zero-mean, unit-variance."""
    field = matern_field((96, 128), np.random.default_rng(4), corr_px=9.0, nu=0.5)
    assert field.shape == (96, 128)
    assert abs(float(field.mean())) < 1e-3
    assert float(field.std()) == pytest.approx(1.0, rel=1e-3)
    # Opposite edges are neighbours in a periodic field, so their difference
    # must be no larger than that of genuinely adjacent rows/columns.
    row_gap = np.abs(field[0] - field[-1]).mean()
    col_gap = np.abs(field[:, 0] - field[:, -1]).mean()
    inner = np.abs(field[1:] - field[:-1]).mean()
    assert row_gap < 2.0 * inner and col_gap < 2.0 * inner


def test_matern_smoothness_controls_feature_size() -> None:
    """A longer correlation length really does produce smoother output."""
    rng = np.random.default_rng(5)
    tight = matern_field((160, 160), rng, corr_px=2.0)
    broad = matern_field((160, 160), rng, corr_px=24.0)
    assert np.abs(np.diff(broad, axis=0)).mean() < np.abs(np.diff(tight, axis=0)).mean()


def test_band_field_peaks_where_asked() -> None:
    """The band-pass field's energy centres on the requested feature size."""
    rng = np.random.default_rng(6)
    field = band_field((128, 128), rng, centre_px=16.0, octaves_wide=0.6)
    power = np.abs(np.fft.rfft2(field)) ** 2
    ky = np.fft.fftfreq(128)[:, None]
    kx = np.fft.rfftfreq(128)[None, :]
    k = np.sqrt(ky**2 + kx**2)
    peak = float(k.ravel()[np.argmax(power.ravel())])
    assert 1.0 / peak == pytest.approx(16.0, rel=0.35)


def test_cluster_points_are_clustered_and_in_bounds() -> None:
    """Clustered placement raises the variance of a cell count over Poisson."""
    shape = (256, 256)
    rng = np.random.default_rng(8)
    xs_c, ys_c = cluster_points(shape, rng, 4000, clustered=0.8, parent_spacing_px=40.0)
    xs_u, ys_u = cluster_points(shape, rng, 4000, clustered=0.0)
    assert xs_c.min() >= 0 and xs_c.max() < shape[1]
    assert ys_c.min() >= 0 and ys_c.max() < shape[0]

    def spread(xs: np.ndarray, ys: np.ndarray) -> float:
        counts, _, _ = np.histogram2d(ys, xs, bins=16)
        return float(counts.var())

    assert spread(xs_c, ys_c) > 2.0 * spread(xs_u, ys_u)


def test_colour_round_trip() -> None:
    """sRGB <-> linear round-trips, and Lab lands where it should."""
    x = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-4)
    # L*=100 is the white point; a kraft-brown Lab must be a warm mid brown.
    assert np.allclose(lab_to_srgb(np.asarray([100.0, 0.0, 0.0])), 1.0, atol=1e-3)
    r, g, b = lab_to_srgb(np.asarray([57.0, 11.0, 25.0]))
    assert r > g > b and 0.5 < r < 0.8


@pytest.mark.parametrize("variant", VARIANTS)
def test_paper_output_stays_low_contrast(variant: str) -> None:
    """Rendered paper stays low-contrast and has no hard edges.

    This is a check on the *output*, not on the surface -- luminance carries
    albedo as well as shading, so it cannot stand in for surface slope. See
    :func:`test_surface_slope_matches_the_measured_target` for that.
    """
    arr = (
        np.asarray(generate("paper", size=192, seed=2, variant=variant), np.float32)
        / 255.0
    )
    lum = arr @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    assert lum.std() < 0.09
    # Adjacent pixels must not swing wildly: paper has no hard edges.
    assert np.abs(np.diff(lum, axis=0)).mean() < 0.02


def test_paper_scale_is_physical_not_pixel() -> None:
    """Doubling the render size shows more detail, not bigger features.

    Every paper feature is specified in millimetres, so at twice the sampling
    density the sub-millimetre roughness must span twice as many pixels. This
    is measured on the height field rather than the output: the output always
    carries some detail right down to the pixel scale (the finest band clamps
    at Nyquist at every resolution), which swamps a naive image-space check.
    """
    lengths = []
    for size in (384, 768):
        _, height_um, ppm = render_sheet(
            (size, size),
            np.random.default_rng(9),
            "kraft",
            creases=0.0,
            mm_across=16.0,
        )
        assert ppm == pytest.approx(size / 16.0)
        # Drop the waviness; what is left is the roughness band at an ISO 16610-21
        # cut-off of lc = 1 mm. Passing the cut-off to ``gaussian_blur`` as if it
        # were a sigma -- which is what this used to do -- asks for a 5.34 mm band
        # and leaves cockle inside the "roughness".
        roughness = iso_highpass(height_um, 1.0 * ppm)
        lengths.append(_corr_length(roughness, axis=1))
    assert lengths[1] > 1.6 * lengths[0], lengths


def _straight_fibre(step_px: float, width_px: float) -> np.ndarray:
    """One near-horizontal, near-straight fibre, for measuring its footprint."""
    return deposit(
        (64, 160),
        np.random.default_rng(0),
        count=1,
        length_px=80.0,
        width_px=width_px,
        length_sigma=0.0,
        theta_mu=0.0,
        kappa=400.0,
        persistence=1e9,
        kinks_per_length=0.0,
        step_px=step_px,
        max_steps=400,
        mass_jitter=0.0,
        positions=(np.asarray([20.0]), np.asarray([32.0])),
    )


def _corr_length(field: np.ndarray, axis: int) -> float:
    """1/e correlation length of ``field`` along ``axis``, in pixels."""
    field = field - field.mean()
    n = field.shape[axis]
    power = np.abs(np.fft.rfft(field, axis=axis)) ** 2
    auto = np.fft.irfft(power, n=n, axis=axis).mean(axis=1 - axis)
    return float(np.argmax(auto / auto[0] < np.exp(-1.0)))


@pytest.mark.parametrize("step_px", [0.5, 0.9, 2.5, 8.0])
def test_fibres_stay_connected_at_any_step(step_px: float) -> None:
    """A fibre must be a continuous ribbon, never a string of beads.

    Samples are splatted bilinearly, so they stop overlapping once they are
    more than a pixel apart; ``deposit`` clamps the step for exactly this
    reason. Without the clamp, every fibre in the sheet is dotted.
    """
    profile = _straight_fibre(step_px, 0.6).sum(axis=0)[30:90]
    assert profile.min() > 0.5 * profile.mean()


def test_fibre_footprint_widens_with_width() -> None:
    """Above a pixel, a wider fibre must be wider -- not merely more opaque.

    Below a pixel the thin-line rule applies and only alpha scales, but a 3 px
    fibre drawn as a 1 px line at triple opacity is narrow and hard-edged:
    the scribed look this module exists to avoid.
    """
    extents = []
    for width_px in (0.5, 3.0, 6.0):
        column = _straight_fibre(0.9, width_px)[:, 60]
        extents.append(int((column > column.max() * 0.05).sum()))
    assert extents[0] < extents[1] < extents[2]


def test_zero_coverage_asks_for_no_fibres() -> None:
    """Coverage of zero means zero fibres, not one."""
    assert count_for_coverage((128, 128), 0.0, 16.0, 0.5) == 0
    assert count_for_coverage((128, 128), -1.0, 16.0, 0.5) == 0


def test_deposit_rejects_short_position_arrays() -> None:
    """Supplying fewer positions than fibres is a bug, not a silent no-op."""
    with pytest.raises(ValueError):
        deposit(
            (32, 32),
            np.random.default_rng(0),
            count=10,
            length_px=8.0,
            width_px=0.5,
            positions=(np.zeros(3), np.zeros(3)),
        )


def test_anisotropy_elongates_along_the_named_angle() -> None:
    """``aniso`` must lengthen features along ``angle``, not across it.

    Attenuating a frequency band lengthens the structure that survives it, so
    the along-axis frequency is scaled *up*. Getting this backwards silently
    rotates every anisotropic field by 90 degrees.
    """
    field = matern_field(
        (512, 512), np.random.default_rng(1), corr_px=20.0, aniso=2.0, angle=0.0
    )
    assert _corr_length(field, axis=1) > 1.5 * _corr_length(field, axis=0)


# The Sq and slope calibration is stated at this sampling density. RMS slope
# is not a scale-free quantity: measured at finer lateral resolution a surface
# resolves steeper features and reports a larger slope, which is why real
# roughness figures always carry an instrument resolution with them.
REFERENCE_PX_PER_MM = 25.6


@pytest.mark.parametrize(
    "variant,target",
    [
        ("white", 0.080),
        ("kraft", 0.110),
        ("recycled", 0.085),
        ("newsprint", 0.105),
        # Wire marks, felt and deep fibre relief carry most of a laid sheet's
        # slope on their own, so its roughness blend was overweighted: it
        # measured 0.147 over eight seeds in a 10 mm window, outside the band.
        ("laid", 0.083),
        # An order of magnitude smoother, and the whole point of a coating: the
        # pigment fills the fibre relief instead of following it.
        ("coated", 0.028),
    ],
)
def test_surface_slope_matches_the_measured_target(variant: str, target: float) -> None:
    """RMS surface slope must hit paper's measured band at the reference sampling.

    This is the number that decides whether a render reads as paper or as
    tooled leather, and it is only meaningful on the height field itself.

    Uncoated stock measures 0.08-0.12 and coated 0.02-0.05, so the targets here
    are the mean of four seeds and the band is deliberately wider than the
    seed-to-seed spread: an individual sheet may sit outside it (Sq itself is a
    per-seed draw), the population may not.
    """
    size = 512
    slopes = []
    for seed in range(4):
        _, height_um, ppm = render_sheet(
            (size, size),
            np.random.default_rng(seed),
            variant,
            creases=0.0,
            mm_across=size / REFERENCE_PX_PER_MM,
        )
        dy, dx = np.gradient(height_um * ppm / 1000.0)
        slopes.append(float(np.sqrt((dx**2 + dy**2).mean())))
    measured = float(np.mean(slopes))
    assert target * 0.8 <= measured <= target * 1.25, measured


@pytest.mark.parametrize("variant", VARIANTS)
def test_surface_slope_stays_bounded_at_high_resolution(variant: str) -> None:
    """Resolving finer bands raises the slope, but it must not run away.

    Past about 20 degrees a surface stops reading as a sheet of paper and
    starts reading as hide, so stay well below that even when every band is
    fully resolved.
    """
    _, height_um, ppm = render_sheet(
        (768, 768), np.random.default_rng(0), variant, creases=0.0, mm_across=8.0
    )
    dy, dx = np.gradient(height_um * ppm / 1000.0)
    slope = float(np.sqrt((dx**2 + dy**2).mean()))
    assert np.degrees(np.arctan(slope)) < 20.0, slope


# The laid wires on a mould sit 0.9-1.3 mm apart. The count per tile is rounded
# to an integer (or the sheet seams), so the realised pitch quantises a little
# outside that band -- hence the slightly wider search window here.
_LAID_PITCH_MM = (0.90, 1.35)


def _spectral_line_strength(profile: np.ndarray, span_mm: float) -> float:
    """Strongest discrete line in the laid-pitch band, over its local background.

    A periodic wire mark is a *line* in the spectrum: energy in one bin, and
    ordinary paper roughness in its neighbours. Paper's own spectrum is steeply
    red, so the comparison has to be local -- against the median of the
    surrounding bins, not against the whole spectrum, or the low-frequency
    cockle would score as a "line" at every pitch.
    """
    profile = profile - profile.mean()
    power = np.abs(np.fft.rfft(profile)) ** 2
    lo = max(2, int(np.ceil(span_mm / _LAID_PITCH_MM[1])))
    hi = int(np.floor(span_mm / _LAID_PITCH_MM[0]))
    best = 0.0
    for n in range(lo, hi + 1):
        near = [
            power[m]
            for m in range(n - 6, n + 7)
            if 2 <= m < len(power) and abs(m - n) > 1
        ]
        best = max(best, float(power[n] / max(float(np.median(near)), 1e-30)))
    return best


def test_laid_height_carries_the_wire_pitch_and_plain_paper_does_not() -> None:
    """``laid`` must show a real periodic component at the laid-wire pitch.

    The marks live in the *height* field at a few micrometres, which is well
    under a percent of contrast in the render -- so this is measured where the
    feature actually is. Three claims at once: the line exists in ``laid``, it
    is absent from ``white`` (i.e. it is the wire and not an artefact of the
    measurement, which any red-noise field would otherwise trip), and it runs
    across the sheet rather than down it, which a transposed phase ramp would
    silently reverse.

    Averaging along the wires is what makes the line visible: the mark is
    coherent across the row and the rest of the sheet is not, so it survives the
    average while the roughness falls as 1/sqrt(width).
    """
    size = 384
    span_mm = size / REFERENCE_PX_PER_MM
    along, across, plain = [], [], []
    for seed in range(3):
        _, height_um, ppm = render_sheet(
            (size, size),
            np.random.default_rng(seed),
            "laid",
            creases=0.0,
            mm_across=span_mm,
        )
        assert ppm == pytest.approx(REFERENCE_PX_PER_MM)
        # Laid wires run across the sheet, so their pitch is measured down it.
        along.append(_spectral_line_strength(height_um.mean(axis=1), span_mm))
        across.append(_spectral_line_strength(height_um.mean(axis=0), span_mm))
        _, height_um, _ = render_sheet(
            (size, size),
            np.random.default_rng(seed),
            "white",
            creases=0.0,
            mm_across=span_mm,
        )
        plain.append(_spectral_line_strength(height_um.mean(axis=1), span_mm))

    # Medians, not means: the strength scales with the per-seed laid_um draw, so
    # one strong sheet must not be able to carry a weak one.
    assert float(np.median(along)) > 25.0, along
    assert float(np.median(plain)) < 12.0, plain
    assert float(np.median(along)) > 5.0 * float(np.median(plain)), (along, plain)
    # Only the chain wires vary across the sheet, and they are 20-30 mm apart.
    assert float(np.median(across)) < 0.5 * float(np.median(along)), across


# --- Height-distribution shape ----------------------------------------------
# Ssk/Sku are properties of the *roughness* band, so the waviness has to come off
# first, with the ISO 25178 S-filter: :func:`~..core.shading.iso_highpass` at a
# cut-off of lc = 1 mm, which is the same band ``sq_um`` calibrates. This was
# previously written as ``gaussian_blur(height_um, 1.0 * ppm)`` and described as a
# "1 mm high-pass", but the ISO 16610-21 filter's sd is 0.18739*lc, so that was a
# 5.34 mm cut-off -- and cockle, not roughness, was setting these numbers.
#
# Bands on the shape of the roughness band, at an ISO 16610-21 cut-off of
# ``lc = 1 mm``. The cut-off is half the specification: the same surface reports a
# different Ssk and Sku in every band, so a band without its cut-off asserts
# nothing. These were measured on this generator's output over seeds 0-3 and are
# regression guards on the mechanisms below, *not* a fit to instrument data.
#
# What keeps them honest is that they all sit inside the range real paper reports
# -- Ssk within [-1.3, +0.9] and Sku within 3.0-5.5, with uncoated stock positive
# and coated negative -- so a band cannot be widened to admit a measurement
# without leaving that range and failing on its face. Each is roughly the mean
# +/- 0.15 in Ssk and +/- 0.2 in Sku, which is several times the seed-to-seed
# spread (Sku varies by under 0.06 across seeds on five of the six).
_SHAPE_BANDS = {
    # variant: (Ssk band, Sku band, Sa/Sq band)
    # Machine-finished woodfree: mildly positive, barely leptokurtic.
    "white": ((0.35, 0.65), (3.02, 3.45), (0.785, 0.802)),
    # Uncalendered, coarse, shive-bearing: nothing has flattened its crowns.
    "kraft": ((0.42, 0.72), (3.32, 3.75), (0.779, 0.796)),
    "recycled": ((0.30, 0.60), (3.02, 3.42), (0.785, 0.802)),
    "newsprint": ((0.38, 0.68), (3.03, 3.45), (0.785, 0.802)),
    # The highest Ssk and Sku of the six, and the only stock with no nip in its
    # history: an isotropic long-fibre stack at coverage 10.5 puts its extremes in
    # sparse fibre crowns, and there is no calender to truncate them. Its Ssk band
    # runs up to 0.88 against paper's 0.9 ceiling, so this is the one stock where
    # the guard is nearly as tight as the physical limit.
    "laid": ((0.58, 0.88), (3.90, 4.40), (0.759, 0.776)),
    # A coating is a film over the fibre relief -- it bridges crowns and pools in
    # hollows -- so its extremes are voids and sags rather than crowns. The one
    # variant that measures Ssk < 0, and the sign is the assertion: a band that
    # straddled zero would pass on a surface with the wrong physics.
    "coated": ((-0.50, -0.22), (3.35, 3.80), (0.781, 0.798)),
}
_GAUSSIAN_SA_OVER_SQ = 0.79788


@functools.cache
def _roughness_shape(variant: str, seed: int) -> tuple[float, float, float]:
    """``(Ssk, Sku, Sa/Sq)`` of one sheet's roughness band. Cached: renders are slow."""
    size = 512
    _, height_um, ppm = render_sheet(
        (size, size),
        np.random.default_rng(seed),
        variant,
        creases=0.0,
        mm_across=size / REFERENCE_PX_PER_MM,
    )
    rough = np.asarray(iso_highpass(height_um, 1.0 * ppm), np.float64)
    c = rough - rough.mean()
    sd = float(c.std())
    return (
        float((c**3).mean() / sd**3),
        float((c**4).mean() / sd**4),
        float(np.abs(c).mean() / sd),
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_roughness_band_height_shape_is_paper_shaped(variant: str) -> None:
    """Ssk and Sku of the roughness band, at lc = 1 mm, must stay paper-shaped.

    A Gaussian random field gives Ssk 0 and Sku 3 for every grade, and that is
    wrong in a way that shows: at this sampling what is resolved is fibres, not
    pores, so a crown, a crossing or a shive is an *upward* extreme and uncoated
    stock measures Ssk > 0 with Sku > 3. Three mechanisms in
    :mod:`..materials.paper` produce that -- the soft maximum over the fibre
    stack, the two-slope skew warp, and the sparse asperity population -- and
    calendering pushes the other way, which is why ``coated`` is the one variant
    below zero.

    Sku is as load-bearing as Ssk here, in *both* directions. Sku < 3 is a
    platykurtic surface: tails lighter than a Gaussian's, which no real surface
    has, and which is what an over-eager calender clip produces by flattening the
    bulk of the upper half of the histogram rather than just the asperity peaks.
    Sku above ~5.5 is the opposite failure and almost always a single-pixel
    artefact rather than a surface: a splatting or warp bug, not roughness.
    """
    ssk_band, sku_band, saq_band = _SHAPE_BANDS[variant]
    shape = [_roughness_shape(variant, seed) for seed in range(4)]
    ssk, sku, saq = (float(np.mean([s[i] for s in shape])) for i in range(3))
    assert ssk_band[0] <= ssk <= ssk_band[1], ssk
    assert sku_band[0] <= sku <= sku_band[1], sku
    assert saq_band[0] <= saq <= saq_band[1], saq


def test_sa_over_sq_is_not_pinned_to_the_gaussian_value() -> None:
    """Sa/Sq must vary by grade, which is only possible if the marginal is not normal.

    Sa/Sq is exactly sqrt(2/pi) = 0.79788 for *any* Gaussian surface whatever its
    spectrum, so a generator built from Gaussian random fields reports that one
    number for all six stocks. The spread across the six is the assertion worth
    making: a per-variant "differs from 0.79788" test would be a tripwire, since
    ``white`` and ``coated`` legitimately sit within 0.004 of it -- a
    lightly-skewed surface can have a near-Gaussian Sa/Sq while its Ssk is not
    near zero at all.
    """
    ratios = {
        variant: float(np.mean([_roughness_shape(variant, s)[2] for s in range(4)]))
        for variant in VARIANTS
    }
    assert max(ratios.values()) - min(ratios.values()) > 0.015, ratios
    # And the skewest grades are the ones that depart, in the direction a heavy
    # upper tail implies: mass concentrated near the mean, so Sa/Sq falls.
    for variant in ("kraft", "laid"):
        assert ratios[variant] < _GAUSSIAN_SA_OVER_SQ - 0.008, ratios


# --- Formation --------------------------------------------------------------
# Skewness of local grammage is meaningless without an aperture, and the standard
# is 1 mm diameter -- the classical beta-radiography formation aperture (Fund.
# Res. Symp. 1973, DOI 10.15376/frc.1973.1.7).
_FORMATION_APERTURE_MM = 1.0
# For a gamma variate CV = 1/sqrt(k) and gamma1 = 2/sqrt(k), so gamma1 = 2*CV
# exactly; clumping beyond the random-fibre case raises the ratio towards 3-4.
# That whole range is the assertion, because it is the one that says the
# deposition really is compound-Poisson and not a warped Gaussian.
_GAMMA_RATIO_BAND = (1.8, 4.0)


def _aperture_mean(field: np.ndarray, diameter_px: float) -> np.ndarray:
    """Mean of ``field`` over a disc aperture, periodically.

    A disc, not a Gaussian: the aperture a formation measurement quotes is a
    physical hole. The convolution is done in Fourier space because every field
    in the paper path is periodic, so a wrapped convolution is exact rather than
    an approximation at the edges.
    """
    h, w = field.shape
    yy = np.fft.fftfreq(h) * h
    xx = np.fft.fftfreq(w) * w
    d = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    # Antialias the rim over one pixel, or the aperture area quantises.
    k = np.clip(0.5 * float(diameter_px) + 0.5 - d, 0.0, 1.0)
    k /= k.sum()
    return np.fft.irfft2(np.fft.rfft2(field) * np.fft.rfft2(k), s=(h, w))


@pytest.mark.parametrize("variant", VARIANTS)
def test_formation_is_positively_skewed_and_gamma_like(variant: str) -> None:
    """Local grammage must be positively skewed, with gamma's gamma1/CV ratio.

    Grammage is a positive random variable with a hard floor at zero and no upper
    bound -- a hole is bounded below, a floc is not -- so its skewness is always
    positive, which is what beta-radiography of handsheets reports (Miami Univ.
    M.S. thesis 2009: grammage distributions positively skewed, softwood more than
    hardwood). Nothing here imposes that: mass comes out of an actual
    compound-Poisson fibre deposition, so the *shape* is a prediction of the
    model and this test is what checks the prediction.

    The ratio is the load-bearing assertion. The absolute level is higher than a
    real 1 mm formation index -- see :data:`_FORMATION_SKEW_BANDS`.
    """
    lo, hi = _FORMATION_SKEW_BANDS[variant]
    skews, cvs = [], []
    size = 512
    for seed in range(4):
        out: dict = {}
        _, _, ppm = render_sheet(
            (size, size),
            np.random.default_rng(seed),
            variant,
            out=out,
            creases=0.0,
            mm_across=size / REFERENCE_PX_PER_MM,
        )
        g = np.asarray(
            _aperture_mean(out["mass"], _FORMATION_APERTURE_MM * ppm), np.float64
        )
        c = g - g.mean()
        skews.append(float((c**3).mean() / float(g.std()) ** 3))
        cvs.append(float(g.std() / g.mean()))
    skew, cv = float(np.mean(skews)), float(np.mean(cvs))
    assert skew > 0.0, skew
    assert lo <= skew <= hi, skew
    ratio = skew / cv
    assert _GAMMA_RATIO_BAND[0] <= ratio <= _GAMMA_RATIO_BAND[1], (skew, cv, ratio)


# Measured bands, and they sit **above** the +0.10 to +0.45 that a 1 mm
# beta-radiography aperture reports for real stock. That is not a skew fault: it
# is the flocculation. gamma1 = 2*CV ties the two together, and this generator's
# floc process (``clustered``, ``floc_spacing_mm``, ``floc_spread_mm``) is tuned
# for visible mottle at a 12-44 mm crop, so it lands a 1 mm formation CV of
# 25-34% against the 4-14% real paper measures -- and carries the skewness up
# with it, in the correct proportion. Depositing the same furnish *uniformly*
# measures gamma1 +0.07 at CV 4.8% (ratio 1.5), i.e. squarely in the published
# band, so the deposition model itself is right and the floc strength is what
# would have to change. Bringing these into the physical band therefore means
# retuning the flocculation, which is a *contrast* change to every stock rather
# than a distribution-shape fix, and is deliberately not done here.
_FORMATION_SKEW_BANDS = {
    "white": (0.45, 1.05),
    "kraft": (0.35, 0.90),
    "recycled": (0.50, 1.10),
    "newsprint": (0.30, 0.95),
    "laid": (0.45, 1.05),
    "coated": (0.45, 1.05),
}


# --- Fibre-scale detail -----------------------------------------------------


def _along_fibre_profile(**kwargs) -> np.ndarray:
    """Mass per unit length along one long, straight, horizontal fibre."""
    mass = deposit(
        (48, 512),
        np.random.default_rng(kwargs.pop("seed", 0)),
        count=1,
        length_px=440.0,
        width_px=0.7,
        length_sigma=0.0,
        theta_mu=0.0,
        kappa=400.0,
        persistence=1e9,
        kinks_per_length=0.0,
        step_px=0.9,
        max_steps=600,
        mass_jitter=0.0,
        positions=(np.asarray([30.0]), np.asarray([24.0])),
        **kwargs,
    )
    return mass.sum(axis=0)


def _autocorr_at(profile: np.ndarray, lag: int) -> float:
    """Normalised autocorrelation of ``profile`` at ``lag`` samples."""
    p = profile - profile.mean()
    return float((p[:-lag] * p[lag:]).mean() / (p * p).mean())


def test_fibre_width_profile_is_correlated_not_white() -> None:
    """Along-fibre width must vary over collapse domains, not per pixel.

    Projected width varies because collapse is *intermittent* along a fibre --
    uncollapsed ~20-30 um, collapsed ~25-45 um, alternating over domains a few
    hundred micrometres long. White noise at the same amplitude is a different
    thing entirely and looks it: a beaded fibre rather than a fibre that is fat
    in places. So the assertion is on the correlation, at a 200 um lag against
    the 350 um correlation length the generator uses (both figures unverified;
    see ``_WIDTH_CORR_UM``).
    """
    ppm = REFERENCE_PX_PER_MM
    lag = max(2, round(0.200 * ppm))
    interior = slice(60, 420)
    correlated = _along_fibre_profile(
        width_cv=0.20, width_corr_px=0.350 * ppm, taper_px=8.0
    )[interior]
    white = _along_fibre_profile(width_cv=0.20, width_corr_px=0.9, taper_px=8.0)[
        interior
    ]
    assert _autocorr_at(correlated, lag) > 0.4, _autocorr_at(correlated, lag)
    assert _autocorr_at(white, lag) < 0.15, _autocorr_at(white, lag)


def test_native_fibre_end_stops_at_a_blunt_tip() -> None:
    """A native tracheid end tapers to 30-60% of mid-width, not to zero.

    A softwood tracheid end is tapered, closed and bluntly pointed: it narrows
    over its last 150-500 um to a tip that is a third to two thirds of mid-fibre
    width and then stops. A symmetric smoothstep to zero -- which is what this
    module used to do -- is a brush stroke, not a fibre. A *cut* end is the other
    population and does not taper at all, so the two are checked together: they
    are what makes the end model two-population rather than one shape.
    """
    ppm = REFERENCE_PX_PER_MM
    taper_px = 0.300 * ppm
    native, cut = [], []
    for seed in range(8):
        for bucket, cut_frac in ((native, 0.0), (cut, 1.0)):
            p = _along_fibre_profile(seed=seed, taper_px=taper_px, cut_ends=cut_frac)
            mid = float(np.median(p[150:350]))
            # The last sample lands about a pixel short of the nominal end, and
            # bilinear splatting halves whatever falls in the final column, so
            # measure a pixel inside the tip rather than on it.
            last = int(np.max(np.nonzero(p > 0.02 * mid)[0]))
            bucket.append(float(p[last - 1] / mid))
    assert 0.30 <= float(np.mean(native)) <= 0.65, native
    # A transverse fracture keeps its width right up to the break.
    assert float(np.mean(cut)) > 0.85, cut


def test_fibrillation_adds_mass_outside_the_fibre_edge() -> None:
    """The fibrillation index must land as coverage *beside* the fibre, not inside it.

    Fibrils are 50 nm - 1 um wide, so they are sub-pixel at every size this
    library renders and cannot be drawn as geometry -- the fringe is an opacity
    contribution hugging the wall instead. Two things to check: it carries the
    index's worth of extra mass (so the parameter means what it says), and it
    goes *outside* the fibre, widening its footprint rather than darkening its
    centre.
    """
    common = dict(taper_px=8.0, cut_ends=0.0)
    plain = _along_fibre_profile(**common)
    fringed = _along_fibre_profile(fibrillation=0.06, fringe_px=0.20, **common)
    assert fringed.sum() == pytest.approx(1.06 * plain.sum(), rel=0.02)

    def off_axis(**kwargs) -> float:
        """Share of a dead-straight fibre's mass that lands off its own row."""
        mass = deposit(
            (48, 512),
            np.random.default_rng(0),
            count=1,
            length_px=440.0,
            width_px=0.7,
            length_sigma=0.0,
            theta_mu=0.0,
            # Straight to a fifth of a pixel over the whole fibre, so "off its
            # own row" means the fringe and not the fibre's own wander.
            kappa=1e6,
            persistence=1e9,
            kinks_per_length=0.0,
            step_px=0.9,
            max_steps=600,
            mass_jitter=0.0,
            positions=(np.asarray([30.0]), np.asarray([24.0])),
            **kwargs,
        )
        column = mass[:, 100:400].sum(axis=1)
        return float((column.sum() - column[24]) / column.sum())

    # A fringe reaching a pixel and a half, so the effect is unambiguous at one
    # pixel of resolution; in a real render it reaches 5 um, i.e. 0.1-0.5 px.
    # A bare fibre still puts a few percent off its row -- even at this kappa it
    # drifts a fifth of a pixel over 440 -- so compare against that, not zero.
    bare = off_axis(**common)
    fringed = off_axis(fibrillation=0.9, fringe_px=1.6, **common)
    assert bare < 0.08, bare
    assert fringed > 3.0 * bare, (bare, fringed)


@pytest.mark.parametrize("variant", VARIANTS)
def test_flat_paper_tiles(variant: str) -> None:
    """Without creases, every field in the paper path is periodic, so it tiles.

    Opposite edges are neighbours in a tiled layout, so their difference must
    be no larger than that of genuinely adjacent rows.
    """
    arr = np.asarray(
        generate("paper", size=192, seed=4, variant=variant, creases=0.0), np.float32
    ).mean(axis=-1)
    interior = np.abs(np.diff(arr, axis=0)).mean()
    assert np.abs(arr[0] - arr[-1]).mean() < 1.4 * interior
    assert np.abs(arr[:, 0] - arr[:, -1]).mean() < 1.4 * interior
