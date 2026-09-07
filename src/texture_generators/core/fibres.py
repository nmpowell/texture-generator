"""Stochastic fibre networks: deposit many curved fibres and accumulate mass.

A sheet of paper is a *stack* of fibres, not a background with lines drawn on
it. Total fibre length per unit area is grammage / coarseness, so 80 g/m^2
office paper carries roughly 440 mm of fibre per mm^2 -- with a 30 um fibre
that is about 13 layers of coverage. Drawing a few hundred opaque segments on
an empty ground gives coverage well below 1, which is why sparse line marks
read as scratches rather than as felt: they are isolated, they are opaque,
and each one is several times too wide.

This module fixes all three at once:

* **Count from coverage.** :func:`count_for_coverage` turns a target coverage
  into a fibre count, so the density is a physical quantity rather than a
  magic number.
* **Sub-pixel width.** A fibre 30 um wide is 0.15--1.0 px across at any
  sensible capture scale, so it must contribute *partial* coverage.
  Accumulating bilinearly-splatted mass does that exactly; rasterising a
  1-px-wide opaque line cannot.
* **Curvature.** Fibre centrelines follow a 2-D worm-like chain -- heading
  performs a random walk with variance ``2*ds/persistence`` per step -- plus
  a Poisson process of sharp kinks. Measured curl indices put the persistence
  length at roughly 0.7--2 fibre lengths, which bends a fibre visibly without
  coiling it.

Everything is vectorised across fibres: one ``(F, S)`` array of headings, one
``cumsum``, one ``bincount`` per bilinear corner. Coordinates wrap, so the
fields tile.

Three further facts shape what a *single* fibre looks like, and all three are
sub-pixel, which is what decides how they must be implemented:

* **External fibrillation is opacity, not geometry.** Refining peels microfibril
  bundles off the outer wall; they are 50 nm - 1 um wide (mode 0.1-0.5 um), and
  whole peeled S1 ribbons only reach 1-5 um. At 25.6 px/mm a pixel is 39 um and
  even a 2000 px render of a 20 mm crop is 10 um per pixel, so *every fibril is
  far narrower than one pixel at every size this library renders*. Drawing them
  as polylines would put sub-pixel objects on the raster as pixel-scale marks --
  the same category error that made early procedural paper read as scratches --
  so they are a coverage contribution instead: :func:`deposit` splats a soft
  anisotropic fringe just outside each fibre's own edge, which composites
  through the same Beer-Lambert ``over`` as the fibre and comes out as a hazy
  fibre edge and inter-fibre webbing. It is parameterised the way a fibre
  analyser reports it, as a **fibrillation index** -- fibril projected area over
  parent-fibre projected area.
* **A fibre end is not a point.** Two populations: a *native* tracheid end
  tapers over its last 150-500 um to a blunt tip 30-60% of mid-fibre width and
  stops there, and a *cut* end (refining; 20-50% of ends) does not taper at all
  -- the S2 fails along its microfibril helix, so the break frays into a brush
  of 3-10 ribbons reaching 20-100 um past it. The ribbons are 0.5-3 um wide, so
  they are a fanned opacity plume rather than drawn strands, for the reason
  above.
* **Projected width varies along a fibre**, because collapse is intermittent:
  an uncollapsed softwood tracheid presents ~20-30 um and a collapsed one
  ~25-45 um (a flattened ribbon is wider than the tube it was), and thick-walled
  latewood resists collapse where thin-walled earlywood gives way. The variation
  is therefore *correlated over hundreds of micrometres*, not white noise, which
  is the visually load-bearing part.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "axial_von_mises",
    "cluster_points",
    "count_for_coverage",
    "deposit",
]

# Splatting more than this many points at once costs more in temporaries than
# the chunking overhead saves.
_CHUNK = 1 << 21

# Native end taper length as a fraction of fibre length, for callers that do not
# state one in pixels: 0.22 of a 1.2 mm fibre is ~260 um, mid-band of the
# 150-500 um a softwood tracheid tapers over.
# UNVERIFIED: the 150-500 um and the 30-60% tip below are domain knowledge --
# no source in the research behind this module reports them.
_TAPER_FRAC = 0.22
_TIP_FRAC = (0.30, 0.60)

# Microfibril angle of the wall layer a fibril was peeled from, in degrees to the
# fibre axis. **The layer angles are textbook** -- S1 is an outer crossed helix
# at 50-90 degrees and is removed first by refining, S2 is the bulk at 5-20
# degrees (20-50 in juvenile or earlywood) and is exposed as refining goes on.
# **The two-mode mixture and its weights are inference from those angles, not a
# measured distribution.**
_S1_DEG = (45.0, 75.0)
_S2_DEG = (10.0, 25.0)
_S1_WEIGHT = 0.3
# Once free and dried a fibril lies back along the fibre, so the angle that
# shows in a micrograph is smaller than the angle it left the wall at: the
# reported apparent range is 10-40 degrees, which this factor lands the mixture
# in. Also inference.
_LAY_BACK = 0.55

# Cut-end brush: ribbons per end, and samples along each ribbon. The brush is
# 3-10 ribbons; six is the middle of that, and four samples over a 20-100 um
# reach is finer than a pixel at any size rendered here.
_BRUSH_RIBBONS = 6
_BRUSH_SAMPLES = 4
# Projected area of one brush relative to its parent fibre, per unit of
# fibrillation index: six 60 um ribbons at 1.5 um wide is ~540 um^2 against a
# 1.15 mm x 25 um fibre's ~29,000 um^2, i.e. ~2% at the 0.035 fibrillation of a
# normal printing furnish. Tying it to the index rather than fixing it is the
# physical statement that refining frays ends and peels fibrils together.
#
# The brush comes *out of* the index rather than on top of it: a fibre analyser
# measures fibril area projecting from the fibre and does not care whether a
# given fibril is rooted on the wall or on a frayed end, so both populations
# share one budget and :func:`deposit` gives the wall fringe whatever the ends
# leave. Adding the brush on top instead -- which is what this used to do --
# meant ``fibrillation = 0.060`` actually deposited 9.49% of extra mass, so the
# key did not mean what its name and docstring said.
_BRUSH_AREA_PER_INDEX = 0.6

# Standard deviation of smoothstep-interpolated unit-variance knots. For
# h(u) = 3u^2 - 2u^3 the interpolant has variance 1 - 2E[h] + 2E[h^2] = 0.742857
# averaged over the cell, so dividing by this makes the profile unit-variance
# without a reduction pass over the array.
_SMOOTHSTEP_SD = 0.861892


def count_for_coverage(
    shape: tuple[int, int],
    coverage: float,
    length_px: float,
    width_px: float,
) -> int:
    """How many fibres of this size are needed to cover the canvas ``coverage`` times.

    Coverage is the mean number of fibre layers over a point: area of canvas
    times coverage, divided by the area of one fibre.
    """
    if float(coverage) <= 0.0:
        return 0
    h, w = int(shape[0]), int(shape[1])
    per_fibre = max(float(length_px) * float(width_px), 1e-6)
    return max(
        1,
        round(h * w * float(coverage) / per_fibre),
    )


def axial_von_mises(
    rng: np.random.Generator, size: int, mu: float, kappa: float
) -> np.ndarray:
    """Sample ``size`` orientations from the axial distribution p(t) ~ exp(k cos 2(t-mu)).

    Fibre direction is an axis, not an arrow: theta and theta+pi are the same
    fibre. Sampling a von Mises in the doubled angle and halving it gives
    exactly that density. ``kappa`` is small for paper -- measured fibre
    orientation distributions have a peak-to-trough ratio of only 1.5--4,
    i.e. kappa between 0.2 and 0.7. Large kappa produces parallel hatching,
    which no paper looks like.
    """
    if kappa <= 1e-6:
        return rng.uniform(0.0, np.pi, size=size).astype(np.float32)
    doubled = rng.vonmises(0.0, float(kappa), size=size)
    return (float(mu) + 0.5 * doubled).astype(np.float32)


def cluster_points(
    shape: tuple[int, int],
    rng: np.random.Generator,
    count: int,
    *,
    clustered: float = 0.6,
    parent_spacing_px: float = 60.0,
    spread_px: float = 18.0,
    aniso: float = 1.0,
    angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions from a Matern cluster (Neyman-Scott) process mixed with uniform.

    Fibres arrive at the wire in flocs, and flocculation is what formation
    measures. Generating clustered *positions* makes formation emerge from
    the fibre layer instead of being painted on top as an independent cloud,
    so the two agree by construction.

    ``clustered`` is the fraction of fibres drawn around parents; the rest are
    uniform. Pure clustering looks blotchy, so keep it below ~0.75.
    """
    h, w = int(shape[0]), int(shape[1])
    n = int(count)
    n_clustered = round(n * float(np.clip(clustered, 0.0, 1.0)))
    n_uniform = n - n_clustered

    xs = [rng.uniform(0.0, w, size=n_uniform).astype(np.float32)]
    ys = [rng.uniform(0.0, h, size=n_uniform).astype(np.float32)]

    if n_clustered > 0:
        n_parents = max(
            1,
            round(h * w / max(parent_spacing_px**2, 1.0)),
        )
        px = rng.uniform(0.0, w, size=n_parents).astype(np.float32)
        py = rng.uniform(0.0, h, size=n_parents).astype(np.float32)
        which = rng.integers(0, n_parents, size=n_clustered)
        # Flocs are sheared along the machine direction, so the offspring
        # cloud is an ellipse rather than a disc.
        ox = rng.normal(0.0, spread_px * float(aniso), size=n_clustered)
        oy = rng.normal(0.0, spread_px, size=n_clustered)
        ca, sa = np.cos(angle), np.sin(angle)
        xs.append((px[which] + ox * ca - oy * sa).astype(np.float32))
        ys.append((py[which] + ox * sa + oy * ca).astype(np.float32))

    return (
        np.concatenate(xs).astype(np.float32) % np.float32(w),
        np.concatenate(ys).astype(np.float32) % np.float32(h),
    )


def _splat(
    shape: tuple[int, int], x: np.ndarray, y: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Bilinearly accumulate ``weight`` at fractional positions, wrapping at the edges.

    ``np.bincount`` is used rather than ``np.add.at``: both accumulate
    duplicates correctly, but bincount is roughly an order of magnitude
    faster, which matters at tens of thousands of fibres.
    """
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros(h * w, dtype=np.float64)
    total = x.size
    for start in range(0, total, _CHUNK):
        stop = min(start + _CHUNK, total)
        xc = np.mod(x[start:stop], w)
        yc = np.mod(y[start:stop], h)
        wt = weight[start:stop]
        x0 = np.floor(xc).astype(np.int64)
        y0 = np.floor(yc).astype(np.int64)
        fx = (xc - x0).astype(np.float64)
        fy = (yc - y0).astype(np.float64)
        x0 %= w
        y0 %= h
        x1 = (x0 + 1) % w
        y1 = (y0 + 1) % h
        rows0 = y0 * w
        rows1 = y1 * w
        for idx, frac in (
            (rows0 + x0, (1.0 - fx) * (1.0 - fy)),
            (rows0 + x1, fx * (1.0 - fy)),
            (rows1 + x0, (1.0 - fx) * fy),
            (rows1 + x1, fx * fy),
        ):
            out += np.bincount(idx, weights=frac * wt, minlength=h * w)
    return out.reshape(h, w).astype(np.float32)


def _correlated_profile(
    rng: np.random.Generator, n: int, steps: int, corr_steps: float
) -> np.ndarray:
    """``(n, steps)`` zero-mean unit-variance noise, correlated along axis 1.

    Along-fibre width variation is generated by intermittent collapse, so it is
    correlated over a *collapse domain* -- hundreds of micrometres -- and white
    noise at that amplitude would read as a beaded fibre instead of a fibre that
    is fat in places. Independent knots every ``corr_steps`` samples, smoothstep
    interpolated between them, give that correlation for one gather and two
    multiplies per sample; an FFT along the step axis would cost several times
    as much for a profile nothing measures the spectrum of.
    """
    corr = max(float(corr_steps), 1.0)
    knots = int(np.ceil(steps / corr)) + 2
    k = rng.standard_normal((n, knots)).astype(np.float32)
    pos = np.arange(steps, dtype=np.float32) / np.float32(corr)
    i0 = np.floor(pos).astype(np.int64)
    u = pos - i0.astype(np.float32)
    hu = (u * u * (3.0 - 2.0 * u))[None, :]
    prof = k[:, i0] * (np.float32(1.0) - hu) + k[:, i0 + 1] * hu
    return (prof / np.float32(_SMOOTHSTEP_SD)).astype(np.float32)


def _fibril_angles(rng: np.random.Generator, size: int) -> np.ndarray:
    """Apparent divergence angles of external fibrils, in radians.

    A peeled fibril leaves the wall at the microfibril angle of the layer it came
    from, and those layer angles are textbook (see :data:`_S1_DEG`). The mixture
    weights and the dried lay-back factor are inference from them rather than a
    measured distribution -- flagged where they are defined.
    """
    s1 = rng.random(size) < _S1_WEIGHT
    lo = np.where(s1, _S1_DEG[0], _S2_DEG[0])
    hi = np.where(s1, _S1_DEG[1], _S2_DEG[1])
    deg = lo + (hi - lo) * rng.random(size)
    return np.deg2rad(deg * _LAY_BACK).astype(np.float32)


def deposit(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    count: int,
    length_px: float,
    width_px: float,
    length_sigma: float = 0.6,
    theta_mu: float = 0.0,
    kappa: float = 0.3,
    persistence: float = 1.2,
    kinks_per_length: float = 3.0,
    kink_deg: float = 25.0,
    step_px: float = 2.5,
    max_steps: int = 96,
    positions: tuple[np.ndarray, np.ndarray] | None = None,
    mass_jitter: float = 0.35,
    width_cv: float = 0.0,
    width_corr_px: float = 0.0,
    taper_px: float | None = None,
    cut_ends: float = 0.0,
    fibrillation: float = 0.0,
    fringe_px: float = 0.0,
    brush_px: float = 0.0,
) -> np.ndarray:
    """Deposit ``count`` curved fibres and return the accumulated mass field.

    The returned field is in coverage units -- its mean is the mean number of
    fibre layers over a pixel. The end taper removes a fixed fraction of each
    fibre's mass, so a count from :func:`count_for_coverage` lands somewhat below
    its nominal coverage rather than exactly on it -- how far below depends on
    ``taper_px`` and ``cut_ends``, since a cut end does not taper at all.

    Args:
        shape: ``(H, W)``.
        rng: seeded generator.
        count: number of fibres; see :func:`count_for_coverage`.
        length_px: median fibre length in pixels (lognormal median).
        width_px: median fibre width in pixels. Values below 1 are the normal
            case and are handled exactly -- the fibre contributes partial
            coverage rather than a thin opaque line.
        length_sigma: lognormal sigma of length. 0.6 matches measured pulp
            length distributions; 0.7+ for mixed recycled furnish.
        theta_mu: machine direction in radians.
        kappa: axial von Mises concentration; see :func:`axial_von_mises`.
        persistence: persistence length as a multiple of fibre length. 0.7--2
            spans never-dried chemical pulp through recycled.
        kinks_per_length: expected sharp kinks per fibre length.
        kink_deg: standard deviation of the kink angle in degrees.
        step_px: arclength per polyline step. **Must be <= 1 px**: samples are
            splatted bilinearly, so a footprint spans one pixel either side of
            the sample and consecutive samples stop overlapping above that.
            At 2.5 px a "fibre" is a string of disconnected beads. Values
            above 1.0 are clamped.
        max_steps: cap on steps per fibre; lengths are clipped to match.
        positions: optional ``(x, y)`` start points, e.g. from
            :func:`cluster_points`. Uniform if omitted.
        mass_jitter: lognormal sigma on per-fibre mass (coarseness varies).
        width_cv: coefficient of variation of projected width *along* one fibre,
            from intermittent collapse. 0.15-0.25 is the plausible range and
            **is unverified**: analysers report a *population* width
            distribution, which is a different quantity. 0 disables it.
        width_corr_px: correlation length of that profile -- the size of a
            collapse domain, 200-500 um, also unverified. Both must be set for
            the profile to apply; white-noise width variation is wrong.
        taper_px: length of the *native* end taper. Defaults to
            :data:`_TAPER_FRAC` of the fibre's own length. The taper runs to a
            blunt tip of 30-60% of mid-fibre width, **not** to zero: a softwood
            tracheid end is tapered, closed and bluntly pointed.
        cut_ends: fraction of ends that are refining cuts rather than native
            tips, 0.2-0.5 in a refined furnish. A cut end does not taper, and
            frays into a brush if ``brush_px`` is set.
        fibrillation: fibrillation index -- fibril projected area over parent
            fibre projected area, so this fraction of each fibre's mass is
            splatted again as external fibril. 0.01 unrefined sack kraft, 0.035 a
            normal printing furnish, 0.06 heavily worked recycled. It is the
            **total** budget: the cut-end brushes take their share of it (see
            :data:`_BRUSH_AREA_PER_INDEX`) and the wall fringe gets the rest, so
            the extra mass deposited is the index and not more.
        fringe_px: how far a fibril reaches from the wall, i.e. the fringe band
            width, 2-8 um. Needed as well as ``fibrillation`` for any fringe.
        brush_px: how far a cut end's brush reaches past the break, 20-100 um.
            Needs ``cut_ends`` and ``fibrillation`` as well.
    """
    h, w = int(shape[0]), int(shape[1])
    n = int(count)
    if n <= 0:
        return np.zeros((h, w), dtype=np.float32)

    # Bilinear splatting spreads a sample one pixel either side, so samples
    # must be at most a pixel apart or the fibre breaks into beads.
    step = float(np.clip(step_px, 0.25, 1.0))
    lengths = rng.lognormal(np.log(max(float(length_px), 1e-3)), float(length_sigma), n)
    lengths = np.clip(lengths, step, step * max_steps).astype(np.float32)

    steps = int(np.ceil(float(lengths.max()) / step))
    steps = int(np.clip(steps, 2, max_steps))

    if positions is None:
        x0 = rng.uniform(0.0, w, size=n).astype(np.float32)
        y0 = rng.uniform(0.0, h, size=n).astype(np.float32)
    else:
        x0 = np.asarray(positions[0], dtype=np.float32)[:n]
        y0 = np.asarray(positions[1], dtype=np.float32)[:n]
        if x0.size < n or y0.size < n:
            raise ValueError(
                f"positions supply {min(x0.size, y0.size)} points for {n} fibres"
            )

    theta0 = axial_von_mises(rng, n, theta_mu, kappa)

    # Worm-like chain: heading variance per step is 2*ds/lp, so the
    # tangent-tangent correlation decays as exp(-s/lp).
    lp = np.maximum(lengths * np.float32(max(persistence, 1e-3)), step)
    turn_sd = np.sqrt(2.0 * step / lp).astype(np.float32)
    dtheta = rng.standard_normal((n, steps)).astype(np.float32) * turn_sd[:, None]

    # Sharp kinks on top of the smooth bend: this is what separates a paper
    # fibre from a hair.
    if kinks_per_length > 0.0:
        p_kink = np.clip(
            float(kinks_per_length) * step / np.maximum(lengths, 1e-3), 0.0, 1.0
        )
        hits = rng.random((n, steps)).astype(np.float32) < p_kink[:, None]
        kick = np.abs(
            rng.standard_normal((n, steps)).astype(np.float32)
            * np.float32(np.deg2rad(kink_deg))
        )
        sign = np.where(rng.random((n, steps)) < 0.5, -1.0, 1.0).astype(np.float32)
        dtheta = dtheta + np.where(hits, kick * sign, 0.0).astype(np.float32)

    theta = theta0[:, None] + np.cumsum(dtheta, axis=1)
    xs = x0[:, None] + np.cumsum(np.cos(theta) * np.float32(step), axis=1)
    ys = y0[:, None] + np.cumsum(np.sin(theta) * np.float32(step), axis=1)

    # Mask samples past each fibre's own length, and taper the last step so
    # length is honoured to sub-step precision.
    arclen = (np.arange(1, steps + 1, dtype=np.float32) * np.float32(step))[None, :]
    live = np.clip((lengths[:, None] - arclen) / np.float32(step) + 1.0, 0.0, 1.0)

    # --- Ends: two populations -------------------------------------------
    # A native tracheid end tapers over its last 150-500 um to a blunt tip
    # 30-60% of mid-fibre width and stops; it does *not* run out to zero, which
    # is what a symmetric smoothstep over a sixth of the fibre used to do here
    # and is why every fibre used to fade out at both ends like a brush stroke.
    # A cut end is a transverse fracture, so it keeps full width right up to the
    # break -- and gets a brush instead, further down.
    taper_len = max(
        float(taper_px) if taper_px else _TAPER_FRAC * float(length_px), step
    )
    tip = rng.uniform(_TIP_FRAC[0], _TIP_FRAC[1], size=n).astype(np.float32)[:, None]
    cut = rng.random((n, 2)) < float(cut_ends)

    def _blunt(d: np.ndarray, is_cut: np.ndarray) -> np.ndarray:
        s = np.clip(d / np.float32(taper_len), 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)
        return np.where(is_cut[:, None], np.float32(1.0), tip + (1.0 - tip) * s)

    taper = np.minimum(
        _blunt(arclen, cut[:, 0]),
        _blunt(lengths[:, None] - arclen, cut[:, 1]),
    ).astype(np.float32)

    coarseness = rng.lognormal(0.0, float(mass_jitter), n).astype(np.float32)
    # Mass per sample = width * arclength, so the pixel-integrated coverage
    # of one fibre is width_px * length_px regardless of step size.
    weight = live * taper * (coarseness[:, None] * np.float32(float(width_px) * step))

    # --- Along-fibre width variation --------------------------------------
    # Collapse is intermittent along a fibre, so projected width is too: an
    # uncollapsed tracheid presents ~20-30 um and a collapsed one ~25-45 um,
    # alternating over collapse domains of a few hundred micrometres. Width
    # enters as mass (the fibre is sub-pixel, so wider means denser, not
    # broader) and as footprint (above a pixel it really is broader).
    wmul = None
    if width_cv > 1e-6 and width_corr_px > 1e-6:
        wmul = np.clip(
            1.0
            + np.float32(width_cv)
            * _correlated_profile(rng, n, steps, float(width_corr_px) / step),
            0.25,
            None,
        ).astype(np.float32)
        weight = weight * wmul

    # Below a pixel, a fibre is a sub-unit-coverage mark and the bilinear
    # footprint is already wider than it is: scale opacity, keep the footprint.
    # Above a pixel it is a real ribbon and must actually be that wide, so lay
    # down parallel strands across the local normal, splitting the mass
    # between them. Without this, a 3 px fibre at 2000 px renders as a 1 px
    # line at triple opacity -- narrow, hard-edged, and exactly the scribed
    # look the rewrite exists to remove.
    strands = int(np.clip(round(float(width_px)), 1, 4))
    weight = weight / np.float32(strands)

    # Every fibre is walked out to the longest fibre's step count and the
    # surplus masked off, so roughly half the samples carry no weight. Drop
    # them before splatting rather than paying four bincounts for zeros.
    keep = (weight > 1e-6).ravel()
    xs_f = xs.ravel()[keep]
    ys_f = ys.ravel()[keep]
    wt_f = weight.ravel()[keep].astype(np.float64)
    tx_f = np.cos(theta).ravel()[keep]
    ty_f = np.sin(theta).ravel()[keep]
    nx_f = -ty_f
    ny_f = tx_f
    # Where the fibre is locally wider, its footprint is wider too.
    wm_f = np.float32(1.0) if wmul is None else wmul.ravel()[keep]

    mass = np.zeros((h, w), dtype=np.float32)
    for k in range(strands):
        offset = np.float32(
            (k - (strands - 1) * 0.5) * (float(width_px) / max(strands, 1))
        )
        mass += _splat(
            (h, w), xs_f + offset * wm_f * nx_f, ys_f + offset * wm_f * ny_f, wt_f
        )

    # --- External fibrillation: a sub-pixel opacity fringe ------------------
    # Fibrils are 50 nm - 1 um wide, so at 10-39 um per pixel they can only ever
    # be *coverage just outside the fibre wall*, never geometry. Splatting
    # ``fibrillation`` of each sample's mass one fibril-reach out along the local
    # normal puts exactly that there: bilinear splatting spreads it into the
    # neighbouring pixel, which is the softened hazy edge and the inter-fibre
    # webbing that fibrillation actually looks like at these scales. One splat,
    # with the side drawn per sample rather than one splat per side -- there are
    # hundreds of samples per pixel, so both sides fill in either way and the
    # deposition pass is the dominant cost of the whole generator.
    # The index is the *total* external-fibril area, and the cut-end brushes below
    # claim a share of it: 2 ends x P(cut) x _BRUSH_AREA_PER_INDEX per unit index.
    # The wall fringe gets the remainder, so the two together deposit exactly
    # ``fibrillation`` of each fibre's mass however hard the furnish was refined.
    brush_share = min(
        2.0 * float(np.clip(cut_ends, 0.0, 1.0)) * _BRUSH_AREA_PER_INDEX, 1.0
    )
    fringe_index = float(fibrillation) * (1.0 - brush_share)
    if fringe_index > 1e-6 and fringe_px > 1e-6 and xs_f.size:
        m = xs_f.size
        phi = _fibril_angles(rng, m)
        # Position along the fibril, so the fringe is a band rather than a line.
        u = rng.random(m).astype(np.float32) * np.float32(fringe_px)
        side = np.where(rng.random(m) < 0.5, np.float32(-1.0), np.float32(1.0))
        # A fibril rooted at the wall reaches sideways by its length times the
        # sine of its divergence angle and lies back along the fibre by the
        # cosine -- so the S1/S2 angle mixture is what sets how far the fringe
        # spreads, and a low-angle S2 fibril hugs the fibre as it should.
        lat = np.float32(0.5 * float(width_px)) * wm_f + side * u * np.sin(phi)
        lon = u * np.cos(phi)
        mass += _splat(
            (h, w),
            xs_f + lat * nx_f + lon * tx_f,
            ys_f + lat * ny_f + lon * ty_f,
            wt_f * fringe_index,
        )

    # --- Cut-end brushes ---------------------------------------------------
    # The S2 fails along its microfibril helix, so a refining cut frays into a
    # brush of 3-10 ribbons reaching 20-100 um past the break. At 0.5-3 um wide
    # those are sub-pixel too, so the brush is a fanned plume of coverage; the
    # ribbons are coherent along their length (they are ribbons, not dust), which
    # is why this walks each one out rather than scattering points in the fan.
    if fibrillation > 1e-6 and brush_px > 1e-6 and cut_ends > 1e-6 and cut.any():
        last = np.clip(np.ceil(lengths / step).astype(np.int64) - 1, 0, steps - 1)
        # Per-fibre mass, so a heavy fibre gets a heavy brush.
        fibre_mass = weight.sum(axis=1)
        px, py, pth, pw = [], [], [], []
        for end, (idx, flip) in enumerate(
            ((last, 0.0), (np.zeros(n, np.int64), np.pi))
        ):
            sel = np.nonzero(cut[:, end])[0]
            if not sel.size:
                continue
            px.append(xs[sel, idx[sel]])
            py.append(ys[sel, idx[sel]])
            pth.append(theta[sel, idx[sel]] + np.float32(flip))
            pw.append(fibre_mass[sel])
        ex = np.concatenate(px)
        ey = np.concatenate(py)
        eth = np.concatenate(pth)
        ew = np.concatenate(pw).astype(np.float64)
        e = ex.size
        # One splayed direction per ribbon, from the same microfibril-angle
        # mixture the fringe uses -- the brush and the fringe are the same
        # mechanism at the end of the fibre rather than along it.
        fan = _fibril_angles(rng, e * _BRUSH_RIBBONS).reshape(e, _BRUSH_RIBBONS)
        fan = fan * np.where(
            rng.random((e, _BRUSH_RIBBONS)) < 0.5, np.float32(-1.0), np.float32(1.0)
        )
        ang = eth[:, None] + fan
        # Ribbon lengths vary, and mass thins out towards the tips.
        reach = np.float32(brush_px) * (
            0.35 + 0.65 * rng.random((e, _BRUSH_RIBBONS)).astype(np.float32)
        )
        r = (np.arange(1, _BRUSH_SAMPLES + 1, dtype=np.float32) / _BRUSH_SAMPLES)[
            None, None, :
        ]
        rr = reach[..., None] * r
        bx = ex[:, None, None] + rr * np.cos(ang)[..., None]
        by = ey[:, None, None] + rr * np.sin(ang)[..., None]
        # Total brush mass per end is a fixed share of the parent fibre, scaled
        # by the fibrillation index: refining frays ends and peels fibrils
        # together, so one parameter drives both.
        share = float(_BRUSH_AREA_PER_INDEX * fibrillation) / (
            _BRUSH_RIBBONS * _BRUSH_SAMPLES
        )
        # 1.5 - 0.8r averages to 1 over the four samples, so the share above is
        # the whole brush's mass and the taper only redistributes it.
        bw = np.broadcast_to((ew * share)[:, None, None], bx.shape) * np.broadcast_to(
            1.5 - 0.8 * r, bx.shape
        )
        mass += _splat((h, w), bx.ravel(), by.ravel(), bw.ravel())
    return mass
