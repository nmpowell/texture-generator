"""Spectral field synthesis: noise built by shaping white noise in Fourier space.

fBm sums octaves, so it has no characteristic scale: every fBm field is
self-similar and looks like generic cloud. Real materials often do have a
characteristic scale -- paper formation peaks at a floc size of a few
millimetres and falls away on both sides of it -- and reproducing that needs
a filter with a knee, not a power law.

Filtering white noise by a prescribed amplitude spectrum gives exact control
of the result's power spectrum (the spectrum of a convolution is the product
of the spectra, and white noise is flat), and because the FFT is periodic the
field it produces tiles seamlessly.

The workhorse here is the Matern / von Karman family

    S(k) = (1 + (2*pi*lambda*k)^2) ^ -(nu + 1)

whose real-space autocorrelation is exponential, exp(-r/lambda), at
``nu = 0.5`` -- which is what beta-radiographic measurements of paper
grammage actually report.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "band_field",
    "freq_grid",
    "matern_field",
    "resize_periodic",
    "shaped_noise",
]


def freq_grid(
    shape: tuple[int, int],
    px_per_unit: float = 1.0,
    aniso: float = 1.0,
    angle: float = 0.0,
) -> np.ndarray:
    """Radial frequency magnitude for ``shape``, in cycles per unit.

    ``aniso`` > 1 stretches features along ``angle`` (radians) by *scaling up*
    that axis' frequency before the radial magnitude is taken, so the filter
    attenuates that band harder and the surviving structure is longer along
    ``angle``. A value of 2 roughly doubles the correlation length there.
    ``px_per_unit`` converts pixel frequencies into physical ones -- pass
    pixels per millimetre to work in mm.
    """
    h, w = int(shape[0]), int(shape[1])
    fy = np.fft.fftfreq(h, d=1.0 / max(px_per_unit, 1e-8)).astype(np.float32)
    fx = np.fft.rfftfreq(w, d=1.0 / max(px_per_unit, 1e-8)).astype(np.float32)
    kx, ky = np.meshgrid(fx, fy, indexing="xy")
    if abs(aniso - 1.0) > 1e-6 or abs(angle) > 1e-6:
        ca, sa = np.float32(np.cos(angle)), np.float32(np.sin(angle))
        along = kx * ca + ky * sa
        across = -kx * sa + ky * ca
        # Multiply, do not divide: attenuating the along-axis frequency band
        # *lengthens* features along that axis. Dividing broadens the spectrum
        # there and shortens them -- the opposite of the contract.
        along = along * np.float32(max(aniso, 1e-6))
        return np.sqrt(along * along + across * across).astype(np.float32)
    return np.sqrt(kx * kx + ky * ky).astype(np.float32)


def shaped_noise(
    shape: tuple[int, int],
    rng: np.random.Generator,
    amplitude: np.ndarray,
) -> np.ndarray:
    """White noise filtered by ``amplitude`` (an rfft2-shaped |H(k)| array).

    Returns a zero-mean, unit-variance float32 field that tiles seamlessly.
    """
    h, w = int(shape[0]), int(shape[1])
    spec = np.fft.rfft2(rng.standard_normal((h, w)))
    spec = spec * amplitude
    spec[0, 0] = 0.0
    out = np.fft.irfft2(spec, s=(h, w)).astype(np.float32)
    sd = float(out.std())
    if sd < 1e-8:
        return np.zeros((h, w), dtype=np.float32)
    return (out / np.float32(sd)).astype(np.float32)


def matern_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    corr_px: float,
    nu: float = 0.5,
    aniso: float = 1.0,
    angle: float = 0.0,
) -> np.ndarray:
    """Seamless Matern / von Karman random field, unit variance.

    Args:
        shape: ``(H, W)``.
        rng: seeded generator.
        corr_px: correlation length in pixels -- the scale at which structure
            lives. Energy rolls off above ``1 / corr_px``.
        nu: Matern smoothness. 0.5 gives an exponential autocorrelation
            (rough, clumpy); larger values give smoother, rounder blobs.
        aniso: > 1 elongates features along ``angle``.
        angle: elongation direction in radians.
    """
    k = freq_grid(shape, px_per_unit=1.0, aniso=aniso, angle=angle)
    amp = np.power(
        1.0 + (2.0 * np.pi * float(corr_px) * k) ** 2, -(float(nu) + 1.0) * 0.5
    ).astype(np.float32)
    return shaped_noise(shape, rng, amp)


def band_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    centre_px: float,
    octaves_wide: float = 1.0,
    aniso: float = 1.0,
    angle: float = 0.0,
) -> np.ndarray:
    """Seamless band-pass field peaking at a feature size of ``centre_px``.

    A log-Gaussian bump on the radial frequency axis: unlike fBm this has a
    genuine characteristic scale, which is what "tooth" and "grain" need --
    they are a band, not a spectrum tail.
    """
    k = freq_grid(shape, px_per_unit=1.0, aniso=aniso, angle=angle)
    k0 = 1.0 / max(float(centre_px), 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        logk = np.log2(np.maximum(k, 1e-12) / k0)
    sigma = max(float(octaves_wide), 1e-3)
    amp = np.exp(-0.5 * (logk / sigma) ** 2).astype(np.float32)
    amp[k <= 0.0] = 0.0
    return shaped_noise(shape, rng, amp)


def resize_periodic(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a periodic field to ``shape`` by zero-padding its spectrum.

    Upsampling is *exact* band-limited resampling for a field that is periodic
    on its own extent: the result is the field's own trigonometric interpolant
    evaluated on the new grid, so nothing is lost and up-then-back-down is an
    identity to floating-point tolerance. Downsampling is exact for content
    below the new Nyquist and discards the rest -- that band is genuinely
    unrepresentable on the smaller grid, not an error in the method. Either way
    no interpolation kernel of its own is introduced, and -- the reason it is
    used here -- the result still tiles seamlessly, because a truncated or
    zero-padded Fourier series is still periodic on the new extent. Paper is
    generated as a tile, so any resize applied inside the pipeline has to
    preserve that.

    ``rfft2``'s row axis is a full ``fftfreq`` layout (non-negative frequencies
    at the start, negative ones at the end) while the column axis is
    ``rfftfreq`` (non-negative only), so the rows are copied from both ends of
    the array and the columns only from the front.

    Both axes need their Nyquist bin handled, and the layouts make it look
    different on each. On an even-length axis the bin at ``N // 2`` carries the
    frequencies ``+N/2`` and ``-N/2`` aliased onto one another, so it stands for
    a conjugate *pair* while contributing its amplitude once. Move it somewhere
    it is no longer Nyquist and it becomes an ordinary coefficient contributing
    twice, so it has to be split back into halves; move an ordinary pair onto a
    new Nyquist bin and the two have to be folded together.

    For the column axis the split is a plain halving, because the column at
    ``w0 // 2`` is self-conjugate down the rows and its ``-N/2`` half is exactly
    what the reconstruction's implicit conjugate column supplies. For the row
    axis both halves are stored explicitly, so the split writes the second one
    into the row the padding left empty. Getting the rows wrong is quieter than
    getting the columns wrong -- ``irfft2`` discards the imaginary part of the
    self-conjugate columns, which happens to symmetrise a mishandled row
    Nyquist for free whenever its energy sits in column 0 (an alternating-*row*
    field, say) -- but off that column it silently loses amplitude.

    Handles both up- and down-sampling, non-square and odd sizes.
    """
    arr = np.asarray(field)
    h0, w0 = int(arr.shape[0]), int(arr.shape[1])
    h1, w1 = int(shape[0]), int(shape[1])
    if (h1, w1) == (h0, w0):
        return arr.astype(np.float32, copy=True)
    if h1 < 1 or w1 < 1:
        raise ValueError(f"resize_periodic: bad target shape {shape!r}")

    spec = np.fft.rfft2(arr)
    new_spec = np.zeros((h1, w1 // 2 + 1), dtype=spec.dtype)

    hm = min(h0, h1)
    n_top = hm // 2 + 1  # rows 0 .. hm//2 hold the non-negative frequencies
    n_bot = (hm - 1) // 2  # the remaining rows hold the negative ones
    n_col = min(w0 // 2, w1 // 2) + 1

    new_spec[:n_top, :n_col] = spec[:n_top, :n_col]
    if n_bot:
        new_spec[h1 - n_bot :, :n_col] = spec[h0 - n_bot :, :n_col]

    # A Nyquist bin on an even-length axis stands for a conjugate *pair* of
    # frequencies, +N/2 and -N/2, aliased onto one another. Carried across
    # unchanged into an extent where it is no longer Nyquist it would contribute
    # twice, so it has to be split back into its two halves -- and, going the
    # other way, two source bins that land on the target's Nyquist have to be
    # folded together. Rows are handled before columns, because the row fold
    # still needs a row of the untouched source spectrum.
    r_max = n_top - 1  # highest non-negative row frequency carried across
    if h0 % 2 == 0 and r_max == h0 // 2 and not (h1 % 2 == 0 and r_max == h1 // 2):
        new_spec[r_max, :n_col] *= 0.5
        new_spec[h1 - r_max, :n_col] = new_spec[r_max, :n_col]
    elif h1 % 2 == 0 and r_max == h1 // 2 and not (h0 % 2 == 0 and r_max == h0 // 2):
        new_spec[r_max, :n_col] += spec[h0 - r_max, :n_col]

    c_max = n_col - 1  # highest column frequency carried across
    if w0 % 2 == 0 and c_max == w0 // 2 and not (w1 % 2 == 0 and c_max == w1 // 2):
        # The source's Nyquist column is self-conjugate down the rows, so
        # halving it in place *is* the split: the -N/2 half is what the
        # reconstruction's implicit conjugate column now supplies.
        new_spec[:, c_max] *= 0.5
    elif w1 % 2 == 0 and c_max == w1 // 2 and not (w0 % 2 == 0 and c_max == w0 // 2):
        # Becoming Nyquist costs the column its implicit conjugate partner, so
        # fold that partner in explicitly to keep it self-conjugate.
        mirror = np.conj(new_spec[-np.arange(h1) % h1, c_max])
        new_spec[:, c_max] += mirror

    # irfft2 normalises by the output size, so undo the new size and reapply the
    # old one to leave amplitudes (and hence mean and variance) unchanged.
    out = np.fft.irfft2(new_spec, s=(h1, w1)) * (h1 * w1) / (h0 * w0)
    return out.astype(np.float32)
