"""Line Integral Convolution: smear noise along the streamlines of a direction field.

Drawing individual fibres (:mod:`.fibres`) is right only while a fibre is
resolvable. Below that -- fines, mechanical pulp, the felt seen *through* a
coating -- a fibre is a fraction of a pixel wide and there are millions of
them, so depositing them one by one is both wasteful and wrong: what the eye
actually receives is a dense, space-filling weave of coherent filaments with no
individually traceable member.

LIC (Cabral & Leedom, SIGGRAPH '93) produces exactly that. Convolving white
noise along the streamlines of a direction field correlates pixels *along* the
flow and leaves them independent *across* it, which is the defining signature
of a fibrous surface. Substance Designer's ``Slope Blur`` and the Photoshop
"noise then motion blur" trick are both approximations of the same integral.

Two things matter and are easy to get wrong:

* **Direction is an axis, not an arrow.** Fibre orientation is theta ~ theta+pi,
  so the field is interpolated in the doubled angle, and a streamline sampling
  it ahead resolves the leftover sign ambiguity against its own heading. Skip
  that and streamlines fold back on themselves, the integral collapses to a
  local average, and the result is a mushy blotch that looks superficially like
  noise but has no filaments in it.
* **The field must be smooth to advect along.** Per-pixel independent angle
  draws smear nothing, so the heading field is a correlated Gaussian field
  warped to the right marginal, not a pile of independent samples.

Everything wraps: sampling is modulo the array shape, so the fields tile.
"""

from __future__ import annotations

import numpy as np

from .spectral import matern_field

__all__ = ["direction_field", "felt", "lic", "slope_blur"]

# Points on the tabulated inverse CDF of the axial von Mises. The density has
# one period over the tabulated half-turn, so a few hundred points resolve it
# to far better than the angular precision LIC can act on.
_ICDF_POINTS = 512


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorised.

    ``scipy.special.erf`` is unavailable here and ``math.erf`` is scalar-only,
    so this is the tanh-cubic logistic approximation of Phi -- worst-case
    absolute error around 2e-4, which is orders of magnitude below the
    histogram tolerance any orientation statistic is checked to.
    """
    z = z.astype(np.float32, copy=False)
    return 0.5 * (
        1.0 + np.tanh(np.float32(0.7978845608) * (z + np.float32(0.044715) * z**3))
    )


def direction_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    mu: float = 0.0,
    kappa: float = 0.35,
    wander_px: float = 40.0,
) -> np.ndarray:
    """Per-pixel fibre heading in radians, shape (H, W).

    Marginally the headings follow the axial von Mises p(t) ~ exp(k cos 2(t-mu))
    that :func:`..fibres.axial_von_mises` samples pointwise, but LIC needs a
    field it can *advect along*: independent per-pixel draws have no streamlines
    and would smear nothing. So the field is a smooth Gaussian field warped to
    the right marginal -- Gaussian, to uniform through the normal CDF, to
    angle through the numerically tabulated inverse CDF of the target density.
    Warping is monotone, so it preserves both the spatial correlation and the
    periodicity of the underlying field.

    Args:
        shape: ``(H, W)``.
        rng: seeded generator.
        mu: preferred axis (machine direction) in radians.
        kappa: axial von Mises concentration. Paper measures 0.2--0.7; at or
            below 1e-6 the distribution degrades to uniform axial.
        wander_px: correlation length of the heading field in pixels -- how far
            a filament travels before the sheet's local grain has changed.
    """
    h, w = int(shape[0]), int(shape[1])
    # nu = 1.5 rather than the module default of 0.5: an exponentially
    # correlated field is nowhere differentiable, and its heading kinks at
    # every pixel, which frays the streamlines LIC integrates along.
    z = matern_field((h, w), rng, corr_px=max(float(wander_px), 1e-3), nu=1.5)
    u = _normal_cdf(z)

    if float(kappa) <= 1e-6:
        # Uniform over one axial period: theta and theta+pi are one fibre.
        return (np.float32(mu - 0.5 * np.pi) + u * np.float32(np.pi)).astype(np.float32)

    t = np.linspace(mu - 0.5 * np.pi, mu + 0.5 * np.pi, _ICDF_POINTS, dtype=np.float64)
    dens = np.exp(float(kappa) * np.cos(2.0 * (t - float(mu))))
    # Trapezoidal cumulative, so the tabulated CDF starts at exactly 0 and ends
    # at exactly 1 and stays strictly increasing (the density never vanishes).
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]))))
    cdf /= cdf[-1]
    return np.interp(u, cdf, t).astype(np.float32)


def _taps(
    shape: tuple[int, int], x: np.ndarray, y: np.ndarray
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Flat indices and weights of the four cells around each ``(x, y)``, wrapping.

    This is the bilinear *gather* -- the transpose of the bilinear scatter in
    :func:`..fibres._splat`. Indices and weights are returned separately from
    the fetch so that the three fields sampled at the same positions (the
    texture and the two components of the doubled-angle direction) share one
    index computation, which is a third of the cost of a step.
    """
    h, w = int(shape[0]), int(shape[1])
    # x - w*floor(x/w) rather than np.mod: same result on a periodic field and
    # an order of magnitude cheaper in float32, and this runs once per step per
    # walk so it is squarely on the hot path.
    xc = x - np.float32(w) * np.floor(x * np.float32(1.0 / w))
    yc = y - np.float32(h) * np.floor(y * np.float32(1.0 / h))
    # Take the fractional part off the float floor, not off the integer cast:
    # float32 minus an integer array promotes to float64 in numpy, which would
    # silently run every weight and every gather below at double width.
    fx = xc - np.floor(xc)
    fy = yc - np.floor(yc)
    # Cast to intp, not to a narrower integer: fancy indexing with anything
    # else pays for a conversion to intp on every one of the eight gathers.
    # A position a hair below the edge reduces to exactly w in float32, so the
    # integer part still needs wrapping -- otherwise it indexes off the end.
    x0 = xc.astype(np.intp) % w
    y0 = yc.astype(np.intp) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    r0 = y0 * w
    r1 = y1 * w
    gx = np.float32(1.0) - fx
    gy = np.float32(1.0) - fy
    return (r0 + x0, r0 + x1, r1 + x0, r1 + x1), (gx * gy, fx * gy, gx * fy, fx * fy)


def _gather(
    flat: np.ndarray, idx: tuple[np.ndarray, ...], wts: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Bilinear sample of a flattened field at pre-computed taps.

    ``flat[idx]`` rather than ``np.take(flat, idx)``: identical result, but
    measurably faster here, and this is the hot inner operation.
    """
    out = flat[idx[0]] * wts[0]
    for i in range(1, 4):
        out += flat[idx[i]] * wts[i]
    return out


def lic(
    texture: np.ndarray,
    angles: np.ndarray,
    length_px: float,
    *,
    step_px: float = 1.0,
    max_steps: int = 64,
) -> np.ndarray:
    """Smear ``texture`` along the streamlines of ``angles``. Returns (H, W).

    Each pixel's streamline is integrated ``length_px / 2`` forwards and the
    same distance backwards, so the kernel spans ``length_px`` centred on the
    pixel, and the samples along it are averaged with a **box kernel** (equal
    weights; a tent or Hanning kernel would soften the filament ends slightly
    but costs an extra multiply per step for no visible gain at these lengths).

    ``angles`` is treated as an *axis*, not a vector, in both places that care.
    The angle sampled ahead of the walk is converted to a unit vector and
    negated when it opposes the current heading; without that, a streamline
    reverses wherever the field describes the same fibre the other way round,
    and the integral collapses into a local average -- the classic mushy,
    blotchy LIC failure that has no filaments in it. And the field is
    interpolated in the *doubled* angle, so t and t+pi are literally the same
    sample; interpolating the raw angle instead swings through 90 degrees at
    every pi jump, which puts a one-pixel seam of cross-grain smear along every
    branch cut of, say, an ``arctan2`` gradient direction.

    Cost is ``O(H * W * length_px / step_px)``, so ``max_steps`` caps the steps
    taken per direction. When ``length_px / (2 * step_px)`` exceeds that cap the
    step grows instead of the count: the kernel still spans ``length_px``, but
    the streamline is sampled more coarsely, and coarsely has two costs. The
    walk starts to cut corners on a curved field, and -- because each sample is
    a point sample, not a line integral over the step -- neighbouring pixels
    sample interleaved sets rather than overlapping ones, so a white-noise input
    comes out correlated at multiples of the step and much less so between them.
    Keep the step at or below about 1 px whenever the input carries pixel-scale
    detail; raise ``max_steps`` rather than the step to lengthen the kernel.

    Args:
        texture: field to convolve, ``(H, W)``.
        angles: per-pixel heading in radians, ``(H, W)``; same shape as
            ``texture``. Any branch is fine -- the result is unchanged by adding
            pi to any subset of pixels.
        length_px: total kernel length in pixels, half either side of each
            pixel. Zero or less returns ``texture`` unchanged.
        step_px: nominal arclength per integration step.
        max_steps: cap on steps per direction.
    """
    tex = np.ascontiguousarray(texture, dtype=np.float32)
    ang = np.ascontiguousarray(angles, dtype=np.float32)
    if tex.shape != ang.shape:
        raise ValueError(f"texture {tex.shape} and angles {ang.shape} must match")
    h, w = tex.shape
    reach = 0.5 * float(length_px)
    step = max(float(step_px), 1e-3)
    if reach < 0.5 * step or float(length_px) <= 0.0:
        # Shorter than a single step: there is nothing to convolve along.
        return tex.copy()

    steps = int(np.ceil(reach / step))
    if steps > int(max_steps):
        steps = max(int(max_steps), 1)
        step = reach / steps

    # Flat throughout: every operation is elementwise over pixels, and 1-D
    # arrays keep the indexing arithmetic to a single term.
    xs = np.tile(np.arange(w, dtype=np.float32), h)
    ys = np.repeat(np.arange(h, dtype=np.float32), w)
    tex_flat = tex.reshape(-1)
    seed_x = np.cos(ang.reshape(-1))
    seed_y = np.sin(ang.reshape(-1))
    # The doubled angle is the axial representation: it is what makes t and
    # t+pi the same value, so interpolating it interpolates the fibre axis
    # rather than the arbitrary arrow chosen to name it.
    cos2 = seed_x * seed_x - seed_y * seed_y
    sin2 = np.float32(2.0) * seed_x * seed_y
    ds = np.float32(step)

    acc = tex_flat.copy()
    for sign in (np.float32(1.0), np.float32(-1.0)):
        # Forward and backward walks from the same seed axis; the backward one
        # simply starts out facing the other way along it.
        px, py = xs.copy(), ys.copy()
        hx, hy = seed_x * sign, seed_y * sign
        for _ in range(steps):
            px += hx * ds
            py += hy * ds
            idx, wts = _taps((h, w), px, py)
            acc += _gather(tex_flat, idx, wts)
            gc = _gather(cos2, idx, wts)
            gs = _gather(sin2, idx, wts)
            # Halve the interpolated doubled angle without any trig:
            # |cos t| = sqrt((1 + cos 2t) / 2), |sin t| = sqrt((1 - cos 2t) / 2),
            # and sin 2t carries their relative sign. Interpolation shortens the
            # vector, so normalise cos 2t first; the magnitude form has no
            # degenerate direction, unlike dividing by 1 + cos 2t.
            c2 = gc / np.maximum(np.sqrt(gc * gc + gs * gs), np.float32(1e-12))
            # Rounding can put the ratio a hair outside [-1, 1], and sqrt of a
            # negative would poison the whole walk with NaN.
            np.clip(c2, np.float32(-1.0), np.float32(1.0), out=c2)
            ax = np.sqrt(np.float32(0.5) * (np.float32(1.0) + c2))
            ay = np.sqrt(np.float32(0.5) * (np.float32(1.0) - c2))
            np.negative(ay, out=ay, where=gs < 0.0)
            # Halving leaves the overall sign undetermined -- which is exactly
            # the axial ambiguity. Resolve it by continuing the walk rather than
            # turning it back on itself.
            flip = np.where(ax * hx + ay * hy < 0.0, np.float32(-1.0), np.float32(1.0))
            hx = ax * flip
            hy = ay * flip

    acc /= np.float32(2 * steps + 1)
    return acc.reshape(h, w)


def felt(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    length_px: float,
    mu: float = 0.0,
    kappa: float = 0.35,
    wander_px: float = 40.0,
    octaves: int = 2,
    layers: int = 1,
    max_steps: int = 64,
) -> np.ndarray:
    """A fibrous felt field: LIC of white noise, zero-mean, unit-variance.

    Two octaves, a third of the length apart at relative weights 1.0 and 0.45:
    a single LIC length gives filaments that are all the same length, which
    reads as combed rather than felted. Real furnish is a mixture of long
    fibres and fines, and the short octave is that second population -- so it
    gets its own white noise rather than a second smear of the first, but the
    *same* direction field, because both populations lie in the same grain.

    Each octave is normalised to unit variance before the blend: a box average
    over length L has variance proportional to 1/L, so raw amplitudes would let
    the short octave dominate and the stated weights would mean nothing.

    ``layers`` is the difference between combed hair and paper. One direction
    field gives *every* filament in a neighbourhood the same orientation,
    because they all follow the same streamlines -- which is what a brushed or
    combed surface looks like, not what a felt looks like. Real fibres
    criss-cross, so stacking several LIC fields with **independent** direction
    fields is what decorrelates local orientation: three layers already read as
    a mat, five or six as a convincing felt. No amount of parameter tuning
    fixes a single layer.

    Cost is ``O(H * W * layers * length_px / step_px)`` -- linear in both
    ``layers`` and ``length_px``, so budget their *product*, not either alone.

    Args:
        shape: ``(H, W)``.
        rng: seeded generator.
        length_px: filament length in pixels (the long octave).
        mu: preferred axis in radians.
        kappa: axial von Mises concentration; see :func:`direction_field`.
        wander_px: correlation length of the heading field in pixels.
        octaves: 1 for the long filaments only, 2 for the recipe above. Higher
            values keep dividing by 3 and scaling the weight by 0.45.
        layers: number of independently-oriented felt stacks to sum, each
            drawing its own direction field from ``rng``. 1 (the default) is a
            single combed grain.
        max_steps: cap on steps per direction; see :func:`lic`.
    """
    h, w = int(shape[0]), int(shape[1])

    out = np.zeros((h, w), dtype=np.float32)
    for _ in range(max(int(layers), 1)):
        angles = direction_field(
            (h, w), rng, mu=mu, kappa=kappa, wander_px=max(float(wander_px), 1e-3)
        )
        for octave in range(max(int(octaves), 1)):
            noise = rng.standard_normal((h, w)).astype(np.float32)
            band = lic(
                noise, angles, float(length_px) / 3.0**octave, max_steps=int(max_steps)
            )
            sd = float(band.std())
            if sd > 1e-8:
                out += band * np.float32(0.45**octave / sd)

    out -= np.float32(out.mean())
    sd = float(out.std())
    if sd < 1e-8:
        return out
    return (out / np.float32(sd)).astype(np.float32)


def slope_blur(
    field: np.ndarray,
    angles: np.ndarray,
    length_px: float,
    *,
    step_px: float = 1.0,
    max_steps: int = 64,
) -> np.ndarray:
    """Directional smear of an arbitrary field along ``angles`` (LIC applied to
    an existing field rather than to noise).

    This is Substance Designer's ``Slope Blur`` in its anisotropic-blur sense:
    the same integral as :func:`lic`, named for what it is used for. Reach for
    it to drag an existing map -- a height field, a coating mask, a stain --
    along the grain, which is how a real fibre network pulls whatever sits on
    it into alignment. The kernel averages, so the mean survives and no
    structure is invented that was not already in ``field``.

    Args:
        field: field to smear, ``(H, W)``.
        angles: per-pixel heading in radians, ``(H, W)``.
        length_px: total kernel length in pixels; zero or less is a no-op.
        step_px: nominal arclength per integration step.
        max_steps: cap on steps per direction; see :func:`lic`.
    """
    return lic(field, angles, length_px, step_px=step_px, max_steps=max_steps)
