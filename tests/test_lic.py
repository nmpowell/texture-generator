"""Tests for Line Integral Convolution and the direction field it runs on.

Statistical, like the rest of the paper suite: what matters is that the
orientation distribution is the one claimed, that the smear runs *along* the
field rather than across it or at 90 degrees to it, and that the kernel really
covers the length it was asked for. A LIC that is subtly wrong still looks like
noise, so every check here is a measurement rather than an eyeball.
"""

from __future__ import annotations

import numpy as np
import pytest

from texture_generators.core.lic import direction_field, felt, lic, slope_blur
from texture_generators.core.spectral import matern_field, resize_periodic


def _corr_length(field: np.ndarray, axis: int) -> float:
    """1/e correlation length of ``field`` along ``axis``, in pixels."""
    field = field - field.mean()
    n = field.shape[axis]
    power = np.abs(np.fft.rfft(field, axis=axis)) ** 2
    auto = np.fft.irfft(power, n=n, axis=axis).mean(axis=1 - axis)
    return float(np.argmax(auto / auto[0] < np.exp(-1.0)))


def _white(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_direction_field_matches_the_axial_von_mises_density() -> None:
    """Headings follow p(t) ~ exp(k cos 2(t - mu)), pi-periodic.

    Same check as
    :func:`test_paper_physics.test_axial_von_mises_matches_its_density`, but the
    field is spatially correlated, so a single realisation holds only
    ``(size / wander_px) ** 2`` effectively independent samples. Hence the short
    correlation length and the average over seeds: with the default 40 px there
    are a couple of hundred independent draws in a 512 px tile and the histogram
    error is 30 times this tolerance for sampling reasons alone.
    """
    kappa, mu = 0.55, 0.7
    hists = []
    for seed in range(6):
        theta = direction_field(
            (512, 512), np.random.default_rng(seed), mu=mu, kappa=kappa, wander_px=2.0
        )
        folded = np.mod(theta - mu, np.pi)
        hist, edges = np.histogram(folded, bins=24, range=(0.0, np.pi), density=True)
        hists.append(hist)
    centres = 0.5 * (edges[:-1] + edges[1:])
    want = np.exp(kappa * np.cos(2.0 * centres))
    want = want / (want.mean() * np.pi)
    assert np.max(np.abs(np.mean(hists, axis=0) - want)) < 0.02


def test_direction_field_is_spatially_smooth() -> None:
    """Neighbouring headings must agree far better than unrelated ones.

    This is what separates a field LIC can advect along from a pile of
    independent per-pixel draws, which would smear nothing at all.
    """
    theta = direction_field((256, 256), np.random.default_rng(2), wander_px=40.0)
    neighbour = float(np.abs(np.diff(theta, axis=1)).mean())
    shuffled = np.random.default_rng(3).permutation(theta.ravel()).reshape(theta.shape)
    random_pair = float(np.abs(theta - shuffled).mean())
    assert random_pair > 8.0 * neighbour


def test_direction_field_is_seamless() -> None:
    """Warping a periodic field is monotone, so the result still tiles."""
    theta = direction_field((96, 128), np.random.default_rng(4), wander_px=12.0)
    inner = float(np.abs(np.diff(theta, axis=0)).mean())
    assert float(np.abs(theta[0] - theta[-1]).mean()) < 2.0 * inner
    assert float(np.abs(theta[:, 0] - theta[:, -1]).mean()) < 2.0 * inner


@pytest.mark.parametrize("angle", [0.0, np.pi / 2])
def test_lic_smears_along_the_field_not_across_it(angle: float) -> None:
    """A constant field at 0 must stretch along x, and at pi/2 along y.

    The single most valuable test here: a transposed gather or a swapped
    sin/cos still produces a plausible-looking fibrous field, just one rotated
    90 degrees from the direction it was asked for.
    """
    shape = (256, 256)
    out = lic(_white(shape), np.full(shape, angle, np.float32), 32.0)
    along = _corr_length(out, 1 if angle == 0.0 else 0)
    across = _corr_length(out, 0 if angle == 0.0 else 1)
    assert along > 8.0 * across, (along, across)


def test_lic_correlation_length_tracks_the_kernel_length() -> None:
    """A longer kernel gives a proportionally longer correlation length."""
    shape = (256, 256)
    flat = np.zeros(shape, dtype=np.float32)
    lengths = [
        _corr_length(lic(_white(shape), flat, length), 1)
        for length in (8.0, 16.0, 32.0, 64.0)
    ]
    assert lengths[0] < lengths[1] < lengths[2] < lengths[3]
    # Roughly two-thirds of the kernel length, and it must scale with it.
    assert lengths[3] == pytest.approx(2.0 * lengths[2], rel=0.15)


def test_lic_treats_the_field_as_an_axis_not_a_vector() -> None:
    """Naming a fibre theta or theta+pi must give the same smear.

    Both halves of the axial contract are on trial here: the doubled-angle
    interpolation (so a pi jump is not a discontinuity) and the sign flip
    against the current heading (so the walk continues instead of reversing).
    A random half of the pixels get their axis renamed, which is a no-op
    physically and a pi-jump minefield numerically.
    """
    shape = (256, 256)
    noise = _white(shape)
    theta = direction_field(shape, np.random.default_rng(1), kappa=0.35)
    renamed = (theta + np.pi * (np.random.default_rng(9).random(shape) < 0.5)).astype(
        np.float32
    )
    base = lic(noise, theta, 48.0)
    assert np.abs(lic(noise, renamed, 48.0) - base).max() < 1e-3 * float(base.std())


@pytest.mark.parametrize("mu", [0.0, np.pi / 2, np.pi / 4])
def test_lic_averages_its_whole_kernel(mu: float) -> None:
    """The smear must average every sample on the streamline, at any orientation.

    A streamline that folds back on itself revisits pixels it has already
    averaged, so it reduces the variance of a white-noise input by less than
    the ``2 * steps + 1`` samples it took would imply -- and by an amount that
    depends on which way the field happens to point. That is the signature of a
    collapsed LIC, and it is far easier to measure than it is to see.
    """
    shape = (256, 256)
    length, step = 48.0, 1.0
    samples = 2 * int(np.ceil(0.5 * length / step)) + 1
    theta = direction_field(shape, np.random.default_rng(1), mu=mu, kappa=0.35)
    out = lic(_white(shape), theta, length, step_px=step)
    # Bilinear sampling smooths a little on its own, so the true reduction beats
    # 1/sqrt(N); it must not fall short of it.
    assert float(out.std()) < 1.0 / np.sqrt(samples)


def test_capping_the_steps_keeps_the_kernel_length() -> None:
    """With ``max_steps`` reached the step grows, so the span is unchanged.

    Measured on an impulse rather than on noise: a coarser step samples the
    same span at wider intervals, which shortens a noise field's lag-1
    correlation without shortening the kernel. The impulse shows the span
    itself.
    """
    shape = (128, 128)
    delta = np.zeros(shape, dtype=np.float32)
    delta[64, 64] = 1.0
    flat = np.zeros(shape, dtype=np.float32)
    spans = []
    for max_steps in (64, 16, 8):
        out = lic(delta, flat, 64.0, max_steps=max_steps)
        _, xs = np.nonzero(out > 1e-7)
        spans.append(int(np.ptp(xs)) + 1)
    assert spans == [65, 65, 65], spans


def test_felt_is_normalised_and_tiles() -> None:
    """Zero mean, unit variance, and seamless -- it goes into a tiling texture."""
    field = felt((192, 192), np.random.default_rng(0), length_px=40.0)
    assert field.shape == (192, 192)
    assert field.dtype == np.float32
    assert abs(float(field.mean())) < 1e-3
    assert float(field.std()) == pytest.approx(1.0, rel=1e-3)
    inner = float(np.abs(np.diff(field, axis=0)).mean())
    assert float(np.abs(field[0] - field[-1]).mean()) < 2.0 * inner
    assert float(np.abs(field[:, 0] - field[:, -1]).mean()) < 2.0 * inner


def test_felt_blends_its_octaves_at_the_stated_weights() -> None:
    """The second octave carries the variance the 1.0 : 0.45 recipe implies.

    ``octaves=1`` is exactly the first octave of ``octaves=2`` from the same
    seed, and the two octaves are independent, so their correlation pins the
    weight ratio: 1 / sqrt(1 + 0.45^2). Equal weights would read 0.707.
    """
    shape = (256, 256)
    one = felt(shape, np.random.default_rng(0), length_px=48.0, octaves=1)
    two = felt(shape, np.random.default_rng(0), length_px=48.0, octaves=2)
    assert float(one.std()) == pytest.approx(1.0, rel=1e-3)
    assert np.corrcoef(one.ravel(), two.ravel())[0, 1] == pytest.approx(
        1.0 / np.sqrt(1.0 + 0.45**2), abs=0.01
    )


def _orientation_spread(field: np.ndarray, bins: int = 18) -> float:
    """Variance of the gradient-orientation histogram, 0 for a perfectly flat one.

    Filaments run across their own gradient, so the gradient orientation
    histogram *is* the orientation histogram, rotated. Weighting by gradient
    energy keeps flat regions from voting. The result is scaled by ``bins**2``
    so it is comparable across bin counts: 0 means every orientation is equally
    represented (a felt), large means the energy piles into a few bins (a comb).
    """
    dy, dx = np.gradient(field.astype(np.float64))
    theta = np.mod(np.arctan2(dy, dx), np.pi)
    hist, _ = np.histogram(
        theta, bins=bins, range=(0.0, np.pi), weights=dx * dx + dy * dy
    )
    hist = hist / max(float(hist.sum()), 1e-12)
    return float(hist.var()) * bins**2


def test_stacking_layers_decorrelates_orientation() -> None:
    """More layers must flatten the orientation histogram -- comb towards felt.

    A single LIC field gives every filament in a neighbourhood the *same*
    heading, because they all follow one direction field: locally that is
    brushed hair, and it is the single thing that stops LIC reading as paper.
    Summing independently-oriented layers is the fix, so the orientation
    histogram must measurably flatten as ``layers`` rises -- and keep flattening,
    not just differ.
    """
    spreads = []
    for layers in (1, 2, 4, 6):
        spreads.append(
            float(
                np.mean(
                    [
                        _orientation_spread(
                            felt(
                                (192, 192),
                                np.random.default_rng(seed),
                                length_px=36.0,
                                kappa=0.45,
                                layers=layers,
                            )
                        )
                        for seed in range(3)
                    ]
                )
            )
        )
    assert spreads == sorted(spreads, reverse=True), spreads
    # Six layers is a felt, not a comb: an order of magnitude flatter.
    assert spreads[-1] < 0.2 * spreads[0], spreads


def test_one_layer_is_the_default_and_unchanged() -> None:
    """``layers=1`` is exactly the old single-field behaviour, draw for draw."""
    a = felt((96, 96), np.random.default_rng(3), length_px=20.0)
    b = felt((96, 96), np.random.default_rng(3), length_px=20.0, layers=1)
    assert np.array_equal(a, b)


def test_slope_blur_of_a_constant_field_is_that_constant() -> None:
    """A smear averages: it conserves the mean and invents no structure."""
    shape = (96, 96)
    flat = np.full(shape, 0.375, dtype=np.float32)
    out = slope_blur(flat, direction_field(shape, np.random.default_rng(2)), 24.0)
    assert np.allclose(out, 0.375, atol=1e-6)


def test_slope_blur_reduces_variance_along_the_grain() -> None:
    """Dragging a field along the grain smooths it there, not across it."""
    shape = (256, 256)
    out = slope_blur(_white(shape), np.zeros(shape, dtype=np.float32), 24.0)
    assert _corr_length(out, 1) > 8.0 * _corr_length(out, 0)


@pytest.mark.parametrize("length_px", [0.0, -5.0])
def test_a_kernel_of_no_length_is_a_no_op(length_px: float) -> None:
    """Nothing to convolve along means the input back, not a blank field."""
    field = _white((32, 32))
    angles = direction_field((32, 32), np.random.default_rng(0))
    assert np.array_equal(lic(field, angles, length_px), field)
    assert np.array_equal(slope_blur(field, angles, length_px), field)


def test_zero_concentration_still_gives_a_usable_field() -> None:
    """kappa = 0 is uniform axial -- a valid sheet with no machine direction."""
    theta = direction_field((128, 128), np.random.default_rng(1), kappa=0.0)
    assert np.isfinite(theta).all()
    assert -0.5 * np.pi - 1e-5 <= float(theta.min())
    assert float(theta.max()) <= 0.5 * np.pi + 1e-5
    field = felt((128, 128), np.random.default_rng(1), length_px=24.0, kappa=0.0)
    assert np.isfinite(field).all()
    assert float(field.std()) == pytest.approx(1.0, rel=1e-3)


def test_everything_flows_from_the_passed_generator() -> None:
    """Same seed, same field; and the global numpy state is never touched."""
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    first = felt((64, 64), np.random.default_rng(5), length_px=12.0)
    after = np.random.random()
    second = felt((64, 64), np.random.default_rng(5), length_px=12.0)
    assert np.array_equal(first, second)
    assert before == after


def test_mismatched_shapes_are_rejected() -> None:
    """A texture and a direction field of different sizes is a bug, not a crop."""
    with pytest.raises(ValueError):
        lic(np.zeros((8, 8), np.float32), np.zeros((8, 9), np.float32), 8.0)


def test_non_square_shapes_wrap_on_both_axes() -> None:
    """The wrap is per-axis, so a non-square tile must still come out clean."""
    field = felt((96, 160), np.random.default_rng(7), length_px=24.0)
    assert field.shape == (96, 160)
    assert np.isfinite(field).all()
    assert float(field.std()) == pytest.approx(1.0, rel=1e-3)


# --- resize_periodic ------------------------------------------------------
#
# The felt stack is computed on a capped canvas and resampled up, so this
# resize sits directly under the expensive path: if it is not amplitude- and
# seam-preserving, every high-resolution sheet inherits the error.


def _seam_vs_neighbour(field: np.ndarray) -> tuple[float, float]:
    """RMS opposite-edge difference and RMS adjacent-line difference.

    A field that tiles has no discontinuity at the wrap, so the first should be
    no larger than the second -- which is the natural step between any two
    neighbouring lines of the same field.
    """
    seam = np.sqrt(
        np.mean((field[0] - field[-1]) ** 2)
        + np.mean((field[:, 0] - field[:, -1]) ** 2)
    )
    neigh = np.sqrt(
        np.mean(np.diff(field, axis=0) ** 2) + np.mean(np.diff(field, axis=1) ** 2)
    )
    return float(seam), float(neigh)


def test_resize_periodic_round_trips_through_a_larger_canvas() -> None:
    """Up then back down returns the original to float error, not merely close.

    Upsampling only adds zeroed high-frequency coefficients and downsampling
    discards exactly those, so the whole original band survives -- including the
    Nyquist row and column, which are split on the way up and folded back on the
    way down. Nothing here is approximate, so the tolerance is float32's.
    """
    field = matern_field((64, 64), np.random.default_rng(0), corr_px=6.0)
    back = resize_periodic(resize_periodic(field, (160, 160)), (64, 64))
    assert back.shape == field.shape
    assert np.abs(back - field).max() < 1e-5 * float(field.std())


def _alternating(shape: tuple[int, int], axis: int) -> np.ndarray:
    """A +1/-1 checker along ``axis`` only: pure Nyquist content on that axis."""
    h, w = shape
    line = np.where(np.arange(shape[axis]) % 2 == 0, 1.0, -1.0).astype(np.float32)
    return np.tile(line[:, None], (1, w)) if axis == 0 else np.tile(line, (h, 1))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dst", [(8, 16), (16, 8), (8, 32), (32, 8), (24, 24)])
def test_resize_periodic_preserves_nyquist_amplitude(
    axis: int, dst: tuple[int, int]
) -> None:
    """An alternating field must stay in [-1, 1] whichever axis carries it.

    On an even-length axis the Nyquist bin stands for a conjugate *pair* aliased
    onto one another, so it contributes its amplitude once. Copied into a wider
    spectrum unsplit it gains a partner and contributes twice, which doubled an
    alternating-column field to -2..2 -- the row axis went the same way whenever
    its Nyquist content sat off column zero.
    """
    field = _alternating((8, 8), axis)
    out = resize_periodic(field, dst)
    assert float(out.min()) == pytest.approx(-1.0, abs=1e-5)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_resize_periodic_round_trips_an_alternating_field(axis: int) -> None:
    """Down again from any of those canvases returns the field exactly."""
    field = _alternating((8, 8), axis)
    for mid in ((16, 16), (32, 8), (8, 32), (24, 40)):
        back = resize_periodic(resize_periodic(field, mid), (8, 8))
        assert np.abs(back - field).max() < 1e-5, mid


@pytest.mark.parametrize(
    "mid", [(160, 160), (128, 128), (192, 96), (96, 192), (101, 167)]
)
def test_resize_periodic_round_trip_is_exact_with_nyquist_content(
    mid: tuple[int, int],
) -> None:
    """The round trip holds for a field deliberately loaded at both Nyquists.

    A Matern field's Nyquist bins are essentially empty, which is what made the
    unsplit version look harmless (0.2 % of the spread). Give the field real
    energy at the Nyquist row, the Nyquist column and their corner and the error
    was over 13 % of its own standard deviation; it should be float error.
    """
    rng = np.random.default_rng(5)
    y, x = np.indices((64, 64))
    field = (
        matern_field((64, 64), rng, corr_px=6.0)
        + 0.4 * _alternating((64, 64), 1)
        + 0.3 * _alternating((64, 64), 0)
        + 0.2 * (-1.0) ** (y + x)
        # Nyquist rows and columns modulated *across* the other axis: the case
        # a plain checker misses, because a checker puts all of its energy in
        # column zero, where irfft2's own symmetrisation hides the error.
        + 0.3 * (-1.0) ** y * np.cos(2.0 * np.pi * x / 16.0)
        + 0.3 * (-1.0) ** x * np.cos(2.0 * np.pi * y / 16.0)
    ).astype(np.float32)
    back = resize_periodic(resize_periodic(field, mid), (64, 64))
    assert np.abs(back - field).max() < 1e-5 * float(field.std())


def test_resize_periodic_preserves_mean_and_spread() -> None:
    """Amplitude, not just shape: the scale factor undoes irfft2's 1/N.

    Downsampling drops the band above the new Nyquist, so the standard deviation
    falls a little; for a Matern field with a correlation length of several
    pixels that band holds only a percent or two of the variance.
    """
    field = matern_field((128, 96), np.random.default_rng(1), corr_px=8.0) + 3.0
    for shape in ((256, 192), (64, 48)):
        out = resize_periodic(field, shape)
        assert out.shape == shape
        assert float(out.mean()) == pytest.approx(float(field.mean()), abs=1e-4)
        assert float(out.std()) == pytest.approx(float(field.std()), rel=0.04)


def test_resize_periodic_result_still_tiles() -> None:
    """The reason for spectral resampling rather than bilinear: paper tiles."""
    field = matern_field((96, 96), np.random.default_rng(2), corr_px=7.0)
    for shape in ((256, 256), (48, 48), (200, 96)):
        seam, neigh = _seam_vs_neighbour(resize_periodic(field, shape))
        assert seam <= 1.2 * neigh, (shape, seam, neigh)


@pytest.mark.parametrize(
    "src,dst",
    [
        ((64, 64), (100, 37)),  # odd target, non-square
        ((64, 64), (63, 65)),  # odd both ways, one axis up one down
        ((50, 30), (128, 96)),  # non-square up
        ((51, 31), (17, 9)),  # odd down
        ((32, 32), (32, 32)),  # no-op
    ],
)
def test_resize_periodic_handles_odd_and_non_square_shapes(
    src: tuple[int, int], dst: tuple[int, int]
) -> None:
    """Any shape in, any shape out, and always real, finite and float32."""
    field = matern_field(src, np.random.default_rng(3), corr_px=5.0)
    out = resize_periodic(field, dst)
    assert out.shape == dst
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # No wild overshoot: band-limited resampling can ring a little, not a lot.
    assert float(np.abs(out).max()) < 2.0 * float(np.abs(field).max()) + 1e-3
