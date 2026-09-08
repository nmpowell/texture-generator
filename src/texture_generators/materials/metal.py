"""Metal textures: brushed, radial and polished sheet metal, plain or filmed.

Algorithm: brushing is fbm evaluated with a severely anisotropic frequency
(a few cells along the brush direction, hundreds across it) in a randomly
rotated frame, so the streaks run in an arbitrary direction. A sparse layer of
1px scratch segments, aligned within a few degrees of the brush, is added to
the same height field that drives the normals, and the streaks also modulate
albedo by a couple of percent -- one field, three channels of evidence.
Shading uses an anisotropic specular lobe plus a broad low-frequency
highlight band, which is what makes a flat image read as a lit metal sheet.

The ``heat_tinted``, ``oil_film`` and ``anodised_titanium`` variants add a
transparent interference film over the metal -- tempering oxide, an oil slick,
anodised titanium -- whose colour comes out of the spectrally integrated table in
:mod:`..core.film`. That path is entered only when a film is requested; the three
finishes above are untouched by it, down to the byte.

Fine parallel grooves are also a diffraction grating, so ``iridescence`` (default
0, off) adds the first order's colour inside the scratches of the grooved
finishes: one wavelength per pixel out of the grating equation, gated by how wide
the light source is. See the groove-diffraction section below.

``engine_turned`` (jewelling, damaskeening) is the ``radial`` recipe evaluated in
a per-cell local frame: a lattice of overlapping swirl marks, composited
last-one-wins so each disc cuts a crescent out of the one behind it. Same
byte-for-byte promise -- it adds a builder and touches nothing above.
"""

from __future__ import annotations

import numpy as np

from ..core.colour import linear_to_srgb, srgb_to_linear
from ..core.fields import draw_segments, normalize01, smoothstep
from ..core.film import SYSTEMS, film_tint, wavelength_rgb
from ..core.noise import fbm, fbm_at, grid_coords
from ..core.shading import shade, tangent_frame
from ..core.warp import double_warp, rotate

VARIANTS = [
    "brushed",
    "radial",
    "polished",
    "heat_tinted",
    "oil_film",
    "anodised_titanium",
    "engine_turned",
]

PALETTES = {
    "aluminium": (0.87, 0.88, 0.89),
    "steel": (0.75, 0.76, 0.77),
    "dark_steel": (0.52, 0.53, 0.55),
    "brass": (0.85, 0.70, 0.35),
    "copper": (0.90, 0.62, 0.48),
}


def _brush_angle_radians(brush_angle: float | None) -> float | None:
    """Validate degrees and return the equivalent direction in radians."""
    numeric_types = (int, float, np.integer, np.floating)
    if brush_angle is None:
        return None
    if isinstance(brush_angle, bool) or not isinstance(brush_angle, numeric_types):
        raise ValueError("brush_angle must be a finite number of degrees")
    angle = float(brush_angle)
    if not np.isfinite(angle):
        raise ValueError("brush_angle must be a finite number of degrees")
    return float(np.deg2rad(angle % 360.0))


def _pick_palette(rng: np.random.Generator) -> np.ndarray:
    """Choose a base metal colour and jitter each channel by +/-0.03.

    The jitter is mostly a shared brightness shift with a small per-channel
    tint on top: independent per-channel jitter of the greys reads as a colour
    cast, which metal should not have.
    """
    name = str(rng.choice(sorted(PALETTES)))
    base = np.asarray(PALETTES[name], dtype=np.float32)
    shift = np.float32(rng.uniform(-0.02, 0.02))
    tint = rng.uniform(-0.01, 0.01, size=3).astype(np.float32)
    return np.clip(base + shift + tint, 0.05, 1.0)


def _streak_freq(px_across: int, extent: float, rng: np.random.Generator) -> float:
    """Across-brush frequency giving ~2-5 px per streak cell at this size.

    Expressed in pixels rather than as a fixed frequency so the brushing stays
    the same visual fineness at any output resolution (and stays clear of the
    Nyquist beading that a fixed 150-400 cell frequency shows below ~512px).
    """
    streak_px = float(rng.uniform(2.2, 5.0))
    return float(px_across / (streak_px * max(extent, 1e-6)))


def _highlight_band(
    shape: tuple[int, int],
    rng: np.random.Generator,
    strength: float = 0.16,
    softness: float = 0.35,
    angle: float | None = None,
    contrast: float = 0.45,
    power: float = 1.0,
) -> np.ndarray:
    """Broad soft lighting band across the sheet (multiplicative, mean ~1).

    ``contrast`` scales how far the off-band side drops below 1: sheets whose
    tonal read depends on the band (brushed) push it towards 1 so the body has
    real shadow to shine out of.
    """
    x, y = grid_coords(shape)
    ang = float(rng.uniform(0.0, np.pi)) if angle is None else float(angle)
    proj = x * np.float32(np.cos(ang)) + y * np.float32(np.sin(ang))
    lo, hi = float(proj.min()), float(proj.max())
    centre = lo + (hi - lo) * float(rng.uniform(0.25, 0.75))
    sigma = max((hi - lo) * softness, 1e-4)
    band = np.exp(-(((proj - np.float32(centre)) / np.float32(sigma)) ** 2))
    if power != 1.0:
        band = band ** np.float32(power)
    return (1.0 - strength * contrast + strength * band).astype(np.float32)


def _env_gradient(
    shape: tuple[int, int], rng: np.random.Generator, angle: float
) -> np.ndarray:
    """Reflected-room gradient: bright cool 'sky' half, dark warm 'ground' half.

    A mirror's dominant cue in a flat photograph is what it reflects, and
    interiors reflect as two halves with a soft horizon between them -- not
    as a symmetric spotlight ridge.
    """
    x, y = grid_coords(shape)
    proj = x * np.float32(np.cos(angle)) + y * np.float32(np.sin(angle))
    proj = normalize01(proj)
    horizon = float(rng.uniform(0.30, 0.70))
    soft = float(rng.uniform(0.08, 0.22))
    t = smoothstep(horizon - soft, horizon + soft, proj)[..., None]
    sky = np.asarray([1.06, 1.08, 1.11], dtype=np.float32)
    ground = np.asarray([0.90, 0.88, 0.85], dtype=np.float32)
    return (ground[None, None, :] + (sky - ground)[None, None, :] * t).astype(
        np.float32
    )


def _soften(field: np.ndarray) -> np.ndarray:
    """3x3 smoothing so 1px scratch rasters read as hairlines, not dashes."""
    out = field * np.float32(0.4)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            out = out + np.roll(np.roll(field, dy, axis=0), dx, axis=1) * np.float32(
                0.075
            )
    return out.astype(np.float32)


def _scratches(
    shape: tuple[int, int],
    rng: np.random.Generator,
    count: int,
    angle: float,
    spread: float = 0.052,
    length_frac: tuple[float, float] = (0.10, 0.60),
    amp: float = 0.35,
) -> np.ndarray:
    """Straight 1px scratch grooves aligned within ``spread`` rad of ``angle``.

    Each groove is rasterised as a lit/shadowed dipole -- the same segment
    twice, offset +-0.7px perpendicular with opposite signs -- which is what
    makes a hairline read as incised rather than drawn on.
    """
    h, w = int(shape[0]), int(shape[1])
    segs = []
    for _ in range(int(count)):
        theta = angle + float(rng.normal(0.0, spread))
        # Log-uniform lengths: abrasion produces many short nicks for every
        # long drag, which uniform lengths flatten into sameness.
        lo, hi = length_frac
        length = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) * w
        x0 = float(rng.uniform(-0.05, 1.05)) * w
        y0 = float(rng.uniform(-0.05, 1.05)) * h
        value = float(rng.uniform(-1.0, 1.0)) * amp
        ca, sa = float(np.cos(theta)), float(np.sin(theta))
        x1, y1 = x0 + length * ca, y0 + length * sa
        nx, ny = -sa * 0.7, ca * 0.7
        segs.append((x0 + nx, y0 + ny, x1 + nx, y1 + ny, value))
        segs.append((x0 - nx, y0 - ny, x1 - nx, y1 - ny, -value * 0.8))
    return _soften(draw_segments((h, w), segs, width=1))


def _arc_scratches(
    shape: tuple[int, int],
    rng: np.random.Generator,
    count: int,
    centre: tuple[float, float],
    amp: float = 0.35,
) -> np.ndarray:
    """Concentric arc scratches about ``centre`` (pixel coords), as polylines.

    Radii are area-uniform (sqrt draw) so scuffs are not crowded at the
    centre, arc LENGTH is heavy-tailed in pixels (constant angular span made
    small arcs a few px and rim arcs hundreds -- backwards), a few arcs run
    the full turned revolution, and each is a radial lit/shadowed dipole.
    """
    h, w = int(shape[0]), int(shape[1])
    cx, cy = centre
    max_r = float(np.hypot(max(cx, w - cx), max(cy, h - cy)))
    segs = []
    for _ in range(int(count)):
        radius = float(np.sqrt(rng.uniform(0.0009, 1.0))) * max_r
        if rng.random() < 0.08:
            span = 2.0 * np.pi
        else:
            arc_px = 0.05 * max_r * float((1.0 - rng.random()) ** (-1.0 / 1.5))
            arc_px = min(arc_px, 1.2 * max_r)
            span = float(np.clip(arc_px / max(radius, 2.0), 0.05, 2.0 * np.pi))
        start = float(rng.uniform(0.0, 2.0 * np.pi))
        value = float(rng.uniform(-1.0, 1.0)) * amp
        steps = min(max(4, int(span * radius / 3.0)), 1400)
        thetas = np.linspace(start, start + span, steps)
        # Wobble the radius along the arc so it is not compass-drawn.
        rr = radius * (
            1.0
            + 0.004 * np.sin(thetas * rng.uniform(2.0, 7.0) + rng.uniform(0.0, 6.28))
        )
        for rad, sign in ((rr + 0.7, value), (rr - 0.7, -value * 0.8)):
            xs = cx + rad * np.cos(thetas)
            ys = cy + rad * np.sin(thetas)
            for i in range(steps - 1):
                segs.append((xs[i], ys[i], xs[i + 1], ys[i + 1], sign))
    return _soften(draw_segments((h, w), segs, width=1))


def _glints(
    shape: tuple[int, int],
    rng: np.random.Generator,
    count: int,
    angle: float = 0.0,
    spread: float = 0.06,
    length_px: tuple[float, float] = (1.5, 6.0),
    centre: tuple[float, float] | None = None,
) -> np.ndarray:
    """Tiny bright flecks stretched along the brush: the sparkle of cut metal.

    Individual groove facets catch the light as near-point highlights. With
    ``centre`` set, each fleck aligns tangentially to the circles about it
    (spun finishes sparkle along their arcs).
    """
    h, w = int(shape[0]), int(shape[1])
    segs = []
    for _ in range(int(count)):
        x0 = float(rng.uniform(0.0, w))
        y0 = float(rng.uniform(0.0, h))
        if centre is not None:
            theta = float(
                np.arctan2(y0 - centre[1], x0 - centre[0])
                + np.pi * 0.5
                + rng.normal(0.0, spread)
            )
        else:
            theta = angle + float(rng.normal(0.0, spread))
        length = float(np.exp(rng.uniform(np.log(length_px[0]), np.log(length_px[1]))))
        # Heavily skewed brightness: most facets barely flash, a few sparkle.
        value = float(0.15 + 0.85 * rng.random() ** 4)
        segs.append(
            (
                x0,
                y0,
                x0 + length * np.cos(theta),
                y0 + length * np.sin(theta),
                value,
            )
        )
    return _soften(draw_segments((h, w), segs, width=1))


def _add_glints(
    rgb: np.ndarray,
    glints: np.ndarray,
    band: np.ndarray,
    base: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Additively composite glints, concentrated where the sheen band is lit.

    The fleck light is tinted by the metal colour like every other reflection
    off a conductor.
    """
    focus = normalize01(band)
    vis = glints * (0.3 + 0.7 * focus) * np.float32(gain)
    tint = (0.35 + 0.65 * base)[None, None, :]
    return rgb + vis[..., None] * tint


# ---------------------------------------------------------------------------
# Groove diffraction: the rainbow in a scratch.
#
# Fine parallel grooves are not only a set of anisotropic microfacets -- they are
# a diffraction grating, and therefore dispersive. For a groove pitch ``d``, an
# incoming direction ``L`` and an outgoing ``V``, the first order satisfies the
# grating equation projected PERPENDICULAR to the groove:
#
#     lambda = d * (dot(V, g) - dot(L, g))
#
# with ``g`` the in-plane unit vector across the groove. This shader has one
# fixed light and one fixed view, so rather than smearing K wavelengths over the
# frame that equation is INVERTED: per pixel there is exactly one wavelength that
# diffracts into the eye, and the only question left is whether it is visible.
# Three things gate it -- the wavelength has to land in 380-730 nm, there has to
# be a groove at that pixel at all (the scratch mask), and the source has to be
# small enough that the first order's own angular width is not washed out by it.
#
# The pitch field is SUB-PIXEL and adds no geometry: at 2000 px over a 100 mm
# plate one pixel is 50 um, so a 1-30 um grating is 2-50x finer than a pixel. It
# is the burnishing the grit left inside the groove, not the visible scratch.
#
# Defaults off, and off means untouched: with ``iridescence=0`` none of this runs
# and none of it draws from ``rng``.
# ---------------------------------------------------------------------------

# Light rig shared by the three grooved finishes. The diffraction pass needs the
# same L that shade() is given, or it computes a wavelength for a light that is
# not there.
_SHEET_LIGHT = (-0.42, -0.48, 0.77)

GROOVE_PITCH_UM = (1.0, 30.0)  # UNVERIFIED (derived): grit burnishing pitch
SUN_ANGULAR_RADIUS_DEG = 0.53  # VERIFIED: the sun subtends ~0.53 deg diameter

# UNVERIFIED (derived): angular width of the 380-730 nm first order against
# pitch. Reads as -- 1.0 um: violently wide rainbow; 1.6 um: the classic CD fan;
# 5 um: tight fringes flanking the highlight; 10 um: faint colour edging; 30 um:
# sun-lit only; 50 um: desaturation rather than colour.
_SPREAD_PITCH_UM = (1.0, 1.6, 5.0, 10.0, 30.0, 50.0)
_SPREAD_DEG = (21.0, 11.4, 3.4, 1.7, 0.6, 0.34)


def _groove_pitch_um(
    shape: tuple[int, int],
    rng: np.random.Generator,
    pitch_um: tuple[float, float] = GROOVE_PITCH_UM,
) -> np.ndarray:
    """Log-normal sub-pixel groove pitch in micrometres, varying slowly.

    Grit leaves a distribution of pitches, not one pitch, and the distribution of
    an abrasive's spacing is multiplicative -- so a low-frequency fbm is mapped
    through an exponential and truncated to ``pitch_um``, with the range spanning
    +/-2 sd. Slowly, because what the eye reads is bands of colour across a plate
    rather than per-pixel confetti.
    """
    lo, hi = float(pitch_um[0]), float(pitch_um[1])
    if not 0.0 < lo <= hi:
        raise ValueError(f"groove_pitch_um must satisfy 0 < lo <= hi; got {pitch_um!r}")
    field = fbm(shape, rng, freq=float(rng.uniform(0.8, 1.8)), octaves=3, gain=0.5)
    sd = float(field.std())
    if sd > 1e-6:
        field = field / np.float32(sd)
    mu = np.float32(0.5 * (np.log(lo) + np.log(hi)))
    sigma = np.float32(0.25 * np.log(hi / lo))
    return np.clip(np.exp(mu + sigma * field), lo, hi).astype(np.float32)


def _groove_spread_deg(pitch_um: np.ndarray) -> np.ndarray:
    """Angular width of the visible first order, interpolated in log-log.

    The tabulated widths above go as ~1/d over most of the range, so the
    interpolation is done on the logarithms of both; outside 1-50 um it clamps to
    the end values, which is harmless because the gate has saturated by then.
    """
    d = np.maximum(np.asarray(pitch_um, dtype=np.float32), 1e-6)
    return np.exp(
        np.interp(np.log(d), np.log(_SPREAD_PITCH_UM), np.log(_SPREAD_DEG))
    ).astype(np.float32)


def _source_gate(spread_deg: np.ndarray, source_deg: float) -> np.ndarray:
    """How much of the first order's colour survives a source of that width.

    A source of angular radius ``s`` smears every wavelength's diffracted
    direction by ``s``, so the spectrum survives only while its own fan is wider
    than the source: the ratio that matters is ``source / spread``, and the
    colour dies as the source grows past the fan. That is the direction the
    visibility figures demand -- a 0.53 deg sun colours grooves out to ~30 um
    while a 10 deg softbox manages nothing above ~2 um -- and it is the opposite
    of a gate written ``exp(-(spread / source)**2)``, which would make the
    softbox the better rainbow lamp and would rule out the 1 um grating whose
    21 deg fan is the most colourful case there is.
    """
    s = np.float32(max(float(source_deg), 1e-4))
    return np.exp(-((s / np.maximum(spread_deg, np.float32(1e-4))) ** 2)).astype(
        np.float32
    )


def _iridescence(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    aniso_dir,
    scratches: np.ndarray,
    light_dir: tuple[float, float, float],
    weight: float,
    source_angular_radius: float = SUN_ANGULAR_RADIUS_DEG,
    groove_pitch_um: tuple[float, float] = GROOVE_PITCH_UM,
) -> np.ndarray:
    """Linear-light RGB of the first diffraction order, per pixel.

    Args:
        aniso_dir: the groove tangent field -- exactly what :func:`shade` is
            given for its anisotropic lobe (an angle for a brushed sheet,
            per-pixel tangential vectors for a spun or turned one).
        scratches: the finish's signed scratch field; its magnitude is the mask,
            so colour appears only where a groove was actually drawn.
        light_dir: the same ``L`` handed to :func:`shade`.
        weight: user weight; 0 returns zeros.

    Returns:
        (H, W, 3) float32 addition **in linear light**, zero wherever the
        diffracted wavelength falls outside the visible band, the source is too
        broad, or there is no groove.
    """
    tangent, _ = tangent_frame(aniso_dir, shape)
    # g: in-plane, across the groove. The view is +z everywhere in core.shading,
    # so dot(V, g) is identically zero for an in-plane g and the whole first
    # order comes out of the light's own across-groove component.
    gx = -tangent[..., 1]
    gy = tangent[..., 0]
    light = np.asarray(light_dir, dtype=np.float64)
    light = light / max(float(np.linalg.norm(light)), 1e-8)
    # The +1 and -1 orders are both there, and the sign of g is a free choice
    # between two equally valid across-groove directions, so what the geometry
    # fixes is the magnitude.
    proj = np.abs(
        np.float32(0.0) - (np.float32(light[0]) * gx + np.float32(light[1]) * gy)
    )

    d_um = _groove_pitch_um(shape, rng, groove_pitch_um)
    lam_nm = (d_um * np.float32(1000.0) * proj).astype(np.float32)
    visible = (lam_nm >= np.float32(380.0)) & (lam_nm <= np.float32(730.0))

    gate = _source_gate(_groove_spread_deg(d_um), source_angular_radius)

    mask = np.abs(np.asarray(scratches, dtype=np.float32))
    peak = float(mask.max())
    if peak > 1e-8:
        mask = mask / np.float32(peak)

    amp = np.where(visible, np.float32(weight) * mask * gate, np.float32(0.0))
    colour = wavelength_rgb(np.clip(lam_nm, 380.0, 730.0))
    return (colour * amp[..., None]).astype(np.float32)


def _composite_iridescence(parts, irid: np.ndarray) -> np.ndarray:
    """Add diffraction colour to :func:`shade`'s specular part, in linear light.

    Diffraction is a property of the REFLECTION, so it goes on the specular term
    and never on the diffuse body -- and the sum is done in linear light, as all
    of core.film's colour work is, because adding energies is the only sum that
    means anything. Pixels the diffraction misses are passed straight through
    rather than round-tripped through the transfer function, so a groove-free
    pixel stays bit-identical to the same render with the effect off.
    """
    diffuse, specular = parts
    active = irid.max(axis=-1) > 0.0
    lit = linear_to_srgb(srgb_to_linear(specular) + irid)
    specular = np.where(active[..., None], lit, specular)
    return np.clip(diffuse + specular, 0.0, 1.0).astype(np.float32)


def _irid_rng(rng: np.random.Generator) -> np.random.Generator:
    """A child generator for the pitch field that leaves ``rng``'s stream alone.

    Spawning rather than drawing means switching the effect on changes only the
    diffraction, not the glints and bands that come after it -- which is what
    makes an on/off pair of renders a controlled comparison.
    """
    return rng.spawn(1)[0]


def _brushed(
    shape: tuple[int, int],
    rng: np.random.Generator,
    base: np.ndarray,
    *,
    irid: dict | None = None,
    brush_angle: float | None = None,
) -> np.ndarray:
    """Render a linear brushed finish; ``brush_angle`` is in radians."""
    h, w = shape
    aspect = h / max(w, 1)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    if brush_angle is not None:
        angle = brush_angle
    x, y = grid_coords((h, w))
    ru, rv = rotate((x, y), -angle, (0.5, aspect * 0.5))

    f_across = _streak_freq(h, aspect, rng)
    fine = fbm_at(
        ru, rv, rng, freq=(float(rng.uniform(1.0, 3.0)), f_across), octaves=3, gain=0.55
    )
    coarse = fbm_at(
        ru,
        rv,
        rng,
        freq=(float(rng.uniform(1.0, 2.5)), f_across * 0.12),
        octaves=2,
        gain=0.6,
    )
    # Grit lines start and stop: modulate the fine streaks by a slow field
    # that varies ALONG the brush, so lines fade in and out over their run
    # instead of scoring dead-uniform edge to edge.
    breaks = fbm_at(ru, rv, rng, freq=(6.0, max(f_across * 0.02, 1.0)), octaves=2)
    # Keep the coarse layer subtle: too much of it reads as wood grain.
    fine = fine * (0.62 + 0.38 * normalize01(breaks))
    streaks = (fine * 0.85 + coarse * 0.35).astype(np.float32)

    scratches = _scratches(
        (h, w),
        rng,
        int(rng.integers(40, 151) * (h * w) / (512.0 * 512.0)),
        angle,
        amp=float(rng.uniform(0.13, 0.26)),
    )

    # Streak contrast is kept low: real brushing shows mostly in the broad
    # sheen, with the individual lines only a few percent of luminance.
    height = streaks * np.float32(0.35) + scratches
    # Darken the sheet a little so the sheen band has headroom to shine into:
    # full-brightness aluminium under ambient + specular clips to paper white.
    body = base * np.float32(rng.uniform(0.80, 0.93))
    albedo = body[None, None, :] * (1.0 + np.float32(0.05) * streaks[..., None])
    albedo = albedo * (1.0 + np.float32(0.03) * scratches[..., None])

    # Pass-correlated roughness: where one sweep of the belt overlapped the
    # last (the coarse bands), the grit ran coarser -- a dimmer, broader
    # lobe. That gloss banding is how a machine-brushed sheet photographs.
    rough = normalize01(coarse)
    spec0 = float(rng.uniform(0.22, 0.42))
    shin0 = float(rng.uniform(18.0, 45.0))
    rgb = shade(
        np.clip(albedo, 0.0, 1.0),
        normalize01(height),
        light_dir=_SHEET_LIGHT,
        specular=(spec0 * (1.0 - 0.4 * rough)).astype(np.float32),
        shininess=(8.0 + shin0 * (1.0 - 0.55 * rough)).astype(np.float32),
        normal_strength=float(rng.uniform(1.5, 3.0)),
        aniso_dir=angle,
        aniso=0.85,
        ambient=0.54,
        spec_tint=0.85,
        return_parts=irid is not None,
    )
    if irid is not None:
        rgb = _composite_iridescence(
            rgb,
            _iridescence(
                (h, w),
                _irid_rng(rng),
                aniso_dir=angle,
                scratches=scratches,
                light_dir=_SHEET_LIGHT,
                **irid,
            ),
        )
    # The band's long axis must run PERPENDICULAR to the brushing: grooves
    # smear a point light across themselves (anisotropic microfacets spread
    # slopes across the groove direction), the way a light on a brushed
    # fridge door streaks vertically over horizontal grain.
    # _highlight_band(angle=a) varies along a, so its stripe lies along
    # a+pi/2; passing the brush angle itself puts the stripe across it.
    band = _highlight_band(
        (h, w),
        rng,
        strength=float(rng.uniform(0.3, 0.45)),
        angle=angle,
        contrast=0.9,
    )
    rgb = rgb * band[..., None]
    scale = (h * w) / float(512 * 512)
    glints = _glints((h, w), rng, count=int(rng.integers(90, 320) * scale), angle=angle)
    rgb = _add_glints(rgb, glints, band, base, gain=float(rng.uniform(0.10, 0.22)))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _radial(
    shape: tuple[int, int],
    rng: np.random.Generator,
    base: np.ndarray,
    *,
    irid: dict | None = None,
) -> np.ndarray:
    """Circular (spun) brushing about a roughly central point."""
    h, w = shape
    aspect = h / max(w, 1)
    x, y = grid_coords((h, w))
    cx = float(rng.uniform(0.35, 0.65))
    cy = float(rng.uniform(0.35, 0.65)) * aspect
    dx = x - np.float32(cx)
    dy = y - np.float32(cy)
    radius = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    theta = np.arctan2(dy, dx).astype(np.float32)
    # theta/2pi spans exactly one lattice period, so the noise is seamless
    # across the +/-pi branch cut.
    tnorm = (theta / np.float32(2.0 * np.pi) + 0.5).astype(np.float32)

    f_radial = _streak_freq(h, aspect, rng) * 0.5
    fine = fbm_at(
        tnorm,
        radius,
        rng,
        freq=(float(rng.integers(2, 5)), f_radial),
        octaves=3,
        gain=0.55,
        periodic=(True, False),
    )
    coarse = fbm_at(
        tnorm,
        radius,
        rng,
        freq=(float(rng.integers(1, 4)), f_radial * 0.1),
        octaves=2,
        gain=0.6,
        periodic=(True, False),
    )
    streaks = (fine * 0.75 + coarse * 0.5).astype(np.float32)

    # Angular detail's arc length collapses to zero at the spindle centre --
    # fade it out so the middle does not shimmer into a pinwheel.
    r0 = float(rng.uniform(0.05, 0.12))
    centre_fade = smoothstep(0.0, r0, radius)
    streaks = streaks * centre_fade

    scratches = _arc_scratches(
        (h, w),
        rng,
        int(rng.integers(25, 70)),
        (cx * w, (cy / max(aspect, 1e-6)) * h),
        amp=float(rng.uniform(0.14, 0.26)),
    )

    height = streaks * np.float32(0.5) + scratches

    # Faint Archimedean tool-feed spiral under the stochastic streaks: the
    # constant-feed groove a facing cut leaves. Seam-safe: the theta term
    # steps the phase by exactly one turn across the branch cut.
    pitch = float(rng.uniform(3.0, 8.0))
    sphase = np.mod(radius * np.float32(w / pitch) - tnorm, 1.0)
    sdist = np.minimum(sphase, 1.0 - sphase)
    sigma = float(rng.uniform(0.06, 0.13))
    groove = np.exp(-0.5 * (sdist / np.float32(sigma)) ** 2)
    dropout = 0.4 + 0.6 * normalize01(coarse)
    height = height - groove * dropout * np.float32(rng.uniform(0.06, 0.14))

    # Machining witness at the centre: a few tight rings and a centre-drill
    # dimple explain the smooth core a spun part always has.
    if rng.random() < 0.65:
        # Keep the ring period >= ~4px at any output size or it aliases.
        ring_pitch = max(float(rng.uniform(0.008, 0.016)), 4.0 / w)
        rings = np.sin(radius * np.float32(2.0 * np.pi / ring_pitch))
        mark = rings * np.exp(-((radius / np.float32(rng.uniform(0.018, 0.035))) ** 2))
        dimple = -np.exp(-((radius / np.float32(0.006)) ** 2)) * np.float32(
            rng.uniform(0.6, 1.2)
        )
        height = height + np.float32(0.5) * mark + dimple
    albedo = base[None, None, :] * (1.0 + np.float32(0.04) * streaks[..., None])
    albedo = albedo * (1.0 + np.float32(0.03) * scratches[..., None])

    # Anisotropy direction is tangent to the circles, so it varies per pixel.
    tan_x = -dy
    tan_y = dx
    rgb = shade(
        np.clip(albedo, 0.0, 1.0),
        normalize01(height),
        light_dir=_SHEET_LIGHT,
        specular=float(rng.uniform(0.5, 0.9)),
        shininess=float(rng.uniform(30.0, 70.0)),
        normal_strength=float(rng.uniform(1.5, 3.5)),
        aniso_dir=(tan_x, tan_y),
        aniso=0.85,
        ambient=0.58,
        spec_tint=0.85,
        return_parts=irid is not None,
    )
    if irid is not None:
        rgb = _composite_iridescence(
            rgb,
            _iridescence(
                (h, w),
                _irid_rng(rng),
                aniso_dir=(tan_x, tan_y),
                scratches=scratches,
                light_dir=_SHEET_LIGHT,
                **irid,
            ),
        )
    # A straight sheen band contradicts circular grooves. The real signature
    # of a spun face under one light is a bow-tie: two opposed bright spokes
    # through the centre, where the radial direction points at the light
    # azimuth. Skewed so it is not mirror-perfect.
    phi = float(np.arctan2(-0.48, -0.42) + rng.uniform(-0.35, 0.35))
    k = float(rng.uniform(3.0, 9.0))
    s = float(rng.uniform(0.18, 0.32))
    lobe = np.abs(np.cos(theta - np.float32(phi))) ** np.float32(k)
    skew = 1.0 + 0.25 * np.cos(theta - np.float32(phi))
    fall = 0.75 + 0.25 * np.exp(-radius * np.float32(1.2))
    band = (1.0 - s * 0.45 + s * lobe * skew * fall).astype(np.float32)
    rgb = rgb * band[..., None]
    scale = (h * w) / float(512 * 512)
    centre_px = (cx * w, (cy / max(aspect, 1e-6)) * h)
    glints = _glints(
        (h, w), rng, count=int(rng.integers(60, 220) * scale), centre=centre_px
    )
    rgb = _add_glints(rgb, glints, band, base, gain=float(rng.uniform(0.08, 0.18)))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _polished(
    shape: tuple[int, int],
    rng: np.random.Generator,
    base: np.ndarray,
    *,
    irid: dict | None = None,
) -> np.ndarray:
    """Mirror finish: restraint is the realism -- no visible grain.

    ``irid`` is accepted and ignored. A mirror has no grating: its handful of
    stray hairlines run at scattered angles with no common tangent, and there is
    no ``aniso_dir`` here for a groove-diffraction pass to invert.
    """
    h, w = shape
    undulation = fbm((h, w), rng, freq=float(rng.uniform(1.5, 3.0)), octaves=2)
    smudge = fbm((h, w), rng, freq=float(rng.uniform(0.8, 1.8)), octaves=2, gain=0.4)
    hairlines = _scratches(
        (h, w),
        rng,
        int(rng.integers(2, 11)),
        float(rng.uniform(0.0, np.pi)),
        spread=0.9,
        length_frac=(0.15, 0.75),
        amp=float(rng.uniform(0.03, 0.07)),
    )
    # A handful of sub-pixel pits: each height dip buys a bright/dark rim
    # pair for free from the normals.
    n_pits = int(rng.integers(5, 31))
    pits = np.zeros((h, w), dtype=np.float32)
    pits[rng.integers(0, h, n_pits), rng.integers(0, w, n_pits)] = -rng.uniform(
        0.5, 1.0, n_pits
    ).astype(np.float32)
    pits = _soften(pits)

    height = (
        undulation * np.float32(0.35)
        + smudge * np.float32(0.25)
        + hairlines
        + pits * np.float32(0.8)
    )
    body = base * np.float32(rng.uniform(0.86, 0.98))
    albedo = body[None, None, :] * (
        1.0
        + np.float32(0.04) * smudge[..., None]
        + np.float32(0.008) * undulation[..., None]
    )

    # Greasy patches broaden and dim the reflection -- the smudges live in
    # the gloss, not just the tint.
    spec0 = float(rng.uniform(0.5, 0.8))
    spec_map = (spec0 * (1.0 - 0.4 * normalize01(smudge))).astype(np.float32)
    rgb = shade(
        np.clip(albedo, 0.0, 1.0),
        normalize01(height),
        light_dir=(-0.3, -0.35, 0.89),
        specular=spec_map,
        shininess=float(rng.uniform(90.0, 200.0)),
        normal_strength=float(rng.uniform(2.0, 4.0)),
        ambient=0.58,
        spec_tint=0.9,
    )
    # What a mirror shows is what it reflects: a room gradient (bright cool
    # half, dark warm half) under a crisp-edged reflected-light streak.
    diag = float(rng.uniform(0.5, 1.1))
    env = _env_gradient((h, w), rng, angle=diag + np.pi / 2)
    broad = _highlight_band((h, w), rng, strength=0.18, softness=0.6, angle=diag)
    tight = _highlight_band(
        (h, w),
        rng,
        strength=float(rng.uniform(0.26, 0.4)),
        softness=float(rng.uniform(0.14, 0.26)),
        angle=diag + float(rng.uniform(-0.15, 0.15)),
        power=3.0,
    )
    rgb = rgb * env * broad[..., None] * tight[..., None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Engine turning (jewelling, damaskeening). Everything here is UNVERIFIED
# domain knowledge -- no primary source was consulted for the numbers below:
#   - an abrasive cup or dowel is spun and stepped across the plate, leaving a
#     disc of concentric/radial abrasion 3-20 mm across;
#   - the step between centres is 0.4-0.7 of a diameter, so consecutive discs
#     overlap by 30-60%;
#   - the layout is a regular grid, or offset ("brick") rows;
#   - each disc is cut AFTER its neighbour and therefore occludes it, so what
#     survives of the disc behind is a crescent. That last-one-wins ordering is
#     the whole look, and it is the only reason this is not just tiled _radial.
# Metal takes no physical scale, so the plate is assumed to be this wide and
# the disc size is a fixed fraction of the render: a 1024px plate is the same
# plate photographed larger, not a more finely turned one.
# ---------------------------------------------------------------------------

ENGINE_TURN_PLATE_MM = 120.0  # UNVERIFIED: assumed plate width across the render
ENGINE_TURN_DIAMETER_MM = (5.0, 16.0)  # UNVERIFIED
ENGINE_TURN_STEP_FRAC = (0.40, 0.70)  # UNVERIFIED: step / diameter


def engine_turn_lattice(
    rng: np.random.Generator, width_px: int
) -> tuple[float, float, bool]:
    """Draw the disc lattice: ``(diameter_px, step_px, brick_rows)``.

    Public because the lattice *is* the measurable claim this variant makes:
    the tests reproduce these draws to know which spatial frequency to look
    for, rather than guessing it back out of the pixels.
    """
    lo, hi = ENGINE_TURN_DIAMETER_MM
    diam_px = float(rng.uniform(lo, hi)) / ENGINE_TURN_PLATE_MM * float(width_px)
    step_px = diam_px * float(rng.uniform(*ENGINE_TURN_STEP_FRAC))
    return diam_px, step_px, bool(rng.random() < 0.5)


def _brick_offset(rows: np.ndarray, step_px: float, brick: bool) -> np.ndarray:
    """Half-step x offset applied to odd rows when the layout is brick-laid."""
    if not brick:
        return np.float32(0.0)
    return np.where(rows % 2 != 0, np.float32(0.5 * step_px), np.float32(0.0)).astype(
        np.float32
    )


def _cell_random(ci: np.ndarray, cj: np.ndarray, salt: int, stream: int) -> np.ndarray:
    """Per-cell uniform in [0, 1) hashed from the integer cell indices.

    A hash rather than an rng draw because which disc owns a pixel is only
    known *per pixel*: the disc's private values have to be recoverable from
    its cell index alone, with no per-cell Python loop.
    """
    v = (ci.astype(np.int64) * 73856093) ^ (cj.astype(np.int64) * 19349663)
    v = (v ^ np.int64((salt + 0x9E3779B1 * stream) & 0x7FFFFFFF)) & 0x7FFFFFFF
    v = (v * 0x27D4EB2D) & 0x7FFFFFFF
    v = v ^ (v >> 13)
    v = (v * 0x27D4EB2D) & 0x7FFFFFFF
    return ((v & 0xFFFFFF).astype(np.float32) / np.float32(0x1000000)).astype(
        np.float32
    )


def engine_turn_frame(
    shape: tuple[int, int], radius_px: float, step_px: float, brick: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local frame of the *winning* disc at every pixel.

    Returns ``(lx, ly, cell_i, cell_j)``: pixel offsets from that disc's centre
    and its integer cell index. The discs are cut in raster order (rows top to
    bottom, left to right within a row) and each occludes the one before, so
    the winner is the greatest ``(j, i)`` whose disc reaches the pixel -- found
    by sweeping candidate cells in that same order and letting the last write
    stand. +/-2 cells is enough: the step is at least 0.4 diameters, so a disc
    can reach at most 1.25 steps.
    """
    h, w = int(shape[0]), int(shape[1])
    x, y = grid_coords((h, w))
    # Pixel coordinates: grid_coords' x spans [0, 1) across the width, and y
    # keeps the same scale down the height, so both scale by the width.
    px = (x * np.float32(w)).astype(np.float32)
    py = (y * np.float32(w)).astype(np.float32)
    step = np.float32(step_px)
    r2 = np.float32(radius_px * radius_px)

    row0 = np.floor(py / step).astype(np.int64)
    best_j = row0
    best_i = np.floor((px - _brick_offset(row0, step_px, brick)) / step).astype(
        np.int64
    )

    for dj in range(-2, 3):
        rj = row0 + dj
        roff = _brick_offset(rj, step_px, brick)
        dy = py - (rj.astype(np.float32) + np.float32(0.5)) * step
        dy2 = dy * dy
        col0 = np.floor((px - roff) / step).astype(np.int64)
        for di in range(-2, 3):
            ii = col0 + di
            dx = px - ((ii.astype(np.float32) + np.float32(0.5)) * step + roff)
            inside = (dx * dx + dy2) <= r2
            best_i = np.where(inside, ii, best_i)
            best_j = np.where(inside, rj, best_j)

    roff = _brick_offset(best_j, step_px, brick)
    lx = px - ((best_i.astype(np.float32) + np.float32(0.5)) * step + roff)
    ly = py - ((best_j.astype(np.float32) + np.float32(0.5)) * step)
    return lx.astype(np.float32), ly.astype(np.float32), best_i, best_j


def _engine_turned(
    shape: tuple[int, int],
    rng: np.random.Generator,
    base: np.ndarray,
    *,
    irid: dict | None = None,
) -> np.ndarray:
    """Engine turning: a lattice of overlapping circular swirl marks.

    Structurally this is the ``radial`` recipe evaluated in a per-cell frame.
    Every disc is its own little spun face: local polar coordinates about the
    winning disc's centre drive the same anisotropic-streak-in-polar noise, and
    the anisotropy handed to :func:`shade` is tangent to *that* disc's circles,
    so each swirl carries its own two-spoke highlight. The plate then glitters
    as a lattice of small suns rather than under one broad band, which is what
    a turned firewall or watch plate actually does.
    """
    h, w = shape
    diam_px, step_px, brick = engine_turn_lattice(rng, w)
    radius_px = 0.5 * diam_px
    salt = int(rng.integers(0, 2**31 - 1))

    lx, ly, ci, cj = engine_turn_frame((h, w), radius_px, step_px, brick)
    lr = np.sqrt(lx * lx + ly * ly).astype(np.float32)
    theta = np.arctan2(ly, lx).astype(np.float32)

    # Per-disc phase and radial noise offset, so neighbours are not clones of
    # one another: the phase spins the pattern, the offset samples a different
    # stretch of the noise lattice radially.
    phase = _cell_random(ci, cj, salt, 0)
    roffset = _cell_random(ci, cj, salt, 1)
    burnish = _cell_random(ci, cj, salt, 2)

    # theta/2pi spans exactly one lattice period, so the angular noise is
    # seamless across each disc's +/-pi branch cut.
    tnorm = np.mod(
        theta / np.float32(2.0 * np.pi) + np.float32(0.5) + phase, np.float32(1.0)
    ).astype(np.float32)
    r_coord = (lr / np.float32(w) + roffset).astype(np.float32)

    aspect = h / max(w, 1)
    f_radial = _streak_freq(h, aspect, rng)
    fine = fbm_at(
        tnorm,
        r_coord,
        rng,
        # Only a couple of cells AROUND each disc against ~150 across its
        # radius: the grit tracks have to run circumferentially, and an angular
        # frequency anywhere near the radial one turns each mark into a fan of
        # spokes instead of a set of fine rings.
        freq=(float(rng.integers(1, 4)), f_radial),
        octaves=3,
        gain=0.55,
        periodic=(True, False),
    )
    coarse = fbm_at(
        tnorm,
        r_coord,
        rng,
        freq=(float(rng.integers(1, 4)), f_radial * 0.12),
        octaves=2,
        gain=0.6,
        periodic=(True, False),
    )
    streaks = (fine * 0.75 + coarse * 0.5).astype(np.float32)
    # Angular arc length collapses at each pivot, and beyond the cup's radius
    # (the slivers no disc reaches) the ground was never turned at all.
    streaks = streaks * smoothstep(0.0, 0.16 * radius_px, lr)
    streaks = streaks * (1.0 - smoothstep(radius_px, 1.12 * radius_px, lr))

    # Constant-feed spiral groove inside each mark, as in _radial: the cup is
    # spinning while it is pressed, so the abrasion is a spiral not rings.
    pitch_px = max(radius_px / float(rng.uniform(6.0, 14.0)), 2.2)
    sphase = np.mod(lr / np.float32(pitch_px) - tnorm, 1.0)
    sdist = np.minimum(sphase, 1.0 - sphase)
    sigma = float(rng.uniform(0.10, 0.20))
    groove = np.exp(-0.5 * (sdist / np.float32(sigma)) ** 2).astype(np.float32)
    dropout = (0.4 + 0.6 * normalize01(coarse)).astype(np.float32)

    # The lip at each disc's own rim is what draws the crescents: a disc's rim
    # is only on screen where no later disc covered it, so this ring survives
    # as an arc, not a circle.
    lip = np.exp(
        -0.5 * ((radius_px - lr) / np.float32(max(0.07 * radius_px, 0.8))) ** 2
    ).astype(np.float32)
    pip = -np.exp(-((lr / np.float32(max(0.06 * radius_px, 0.8))) ** 2)).astype(
        np.float32
    )

    scale = (h * w) / float(512 * 512)
    scratches = _scratches(
        (h, w),
        rng,
        int(rng.integers(8, 40) * scale),
        float(rng.uniform(0.0, np.pi)),
        spread=0.8,
        amp=float(rng.uniform(0.05, 0.12)),
    )

    # Kept deliberately shallow. Turning removes a few microns; the marks are a
    # gloss pattern, not relief, and a strong lip plus a deep pip turns the
    # lattice into embossed fish scales.
    height = (
        streaks * np.float32(0.45)
        - groove * dropout * np.float32(rng.uniform(0.08, 0.16))
        + lip * np.float32(rng.uniform(0.06, 0.13))
        + pip * np.float32(rng.uniform(0.12, 0.26))
        + scratches
    )

    body = base * np.float32(rng.uniform(0.86, 0.97))
    albedo = body[None, None, :] * (1.0 + np.float32(0.04) * streaks[..., None])
    albedo = albedo * (1.0 + np.float32(0.03) * scratches[..., None])
    # Each pass burnishes a shade differently, so the lattice is not one tone.
    albedo = albedo * (np.float32(0.985) + np.float32(0.03) * burnish)[..., None]

    rgb = shade(
        np.clip(albedo, 0.0, 1.0),
        normalize01(height),
        light_dir=_SHEET_LIGHT,
        specular=float(rng.uniform(0.5, 0.9)),
        shininess=float(rng.uniform(30.0, 70.0)),
        normal_strength=float(rng.uniform(1.0, 2.0)),
        # Tangent to each disc's own circles: per-pixel, and rotating with the
        # cell rather than with the plate.
        aniso_dir=(-ly, lx),
        aniso=0.85,
        ambient=0.58,
        spec_tint=0.85,
        return_parts=irid is not None,
    )
    if irid is not None:
        rgb = _composite_iridescence(
            rgb,
            _iridescence(
                (h, w),
                _irid_rng(rng),
                aniso_dir=(-ly, lx),
                scratches=scratches,
                light_dir=_SHEET_LIGHT,
                **irid,
            ),
        )
    # _radial's bow-tie, but one per disc: two opposed bright spokes through
    # each pivot, where the local radial direction points at the light azimuth.
    phi = float(np.arctan2(-0.48, -0.42) + rng.uniform(-0.35, 0.35))
    k = float(rng.uniform(3.0, 8.0))
    s = float(rng.uniform(0.20, 0.34))
    ang = theta - np.float32(phi)
    lobe = np.abs(np.cos(ang)) ** np.float32(k)
    skew = 1.0 + 0.25 * np.cos(ang)
    disc_band = (1.0 - s * 0.45 + s * lobe * skew).astype(np.float32)
    # One broad sheen over the whole plate on top, so the lattice sits in a
    # lighting environment instead of repeating dead-uniformly to the edges.
    sheen = _highlight_band(
        (h, w),
        rng,
        strength=float(rng.uniform(0.18, 0.30)),
        softness=float(rng.uniform(0.35, 0.60)),
    )
    band = (disc_band * sheen).astype(np.float32)
    rgb = rgb * band[..., None]
    glints = _glints(
        (h, w),
        rng,
        count=int(rng.integers(80, 260) * scale),
        angle=float(rng.uniform(0.0, np.pi)),
        spread=1.2,
    )
    rgb = _add_glints(rgb, glints, band, base, gain=float(rng.uniform(0.08, 0.18)))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


_BUILDERS = {
    "brushed": _brushed,
    "radial": _radial,
    "polished": _polished,
    "engine_turned": _engine_turned,
}


# ---------------------------------------------------------------------------
# Thin-film interference. Everything below is reached only when a film is
# actually requested, so the three finishes above are bit-for-bit unaffected.
# ---------------------------------------------------------------------------

# Which optical system each film variant sits in, the thickness range of its
# field in **nanometres**, and how the metal under it is finished. Thickness
# ranges: see core.film for what is VERIFIED and what is not.
#   - heat_tinted: a tempering/heat-affected zone, 40-150 nm of oxide.
#   - oil_film: 100 nm - 2 um of oil or fuel. The top of that range is the
#     point -- above ~1 um the fringes are finer than the eye's colour matching
#     functions and the spill goes pearly grey in the middle while its feathered
#     edges stay saturated. That desaturation is not faked; it falls out of the
#     spectral integration in core.film.
#   - anodised_titanium: 15-250 nm of barrier oxide on Ti or Nb.
FILM_VARIANTS = {
    "heat_tinted": {"system": "oxide", "nm": (38.0, 150.0), "field": "weld"},
    "oil_film": {"system": "oil", "nm": (100.0, 2000.0), "field": "spill"},
    "anodised_titanium": {"system": "titania", "nm": (15.0, 250.0), "field": "patch"},
}

_FILM_GLOSS = {
    "heat_tinted": "hairline",
    "oil_film": "brushed",
    "anodised_titanium": "satin",
    "brushed": "brushed",
    "radial": "satin",
    "polished": "satin",
    "engine_turned": "satin",
}


def _thickness_weld(
    shape: tuple[int, int], rng: np.random.Generator, lo: float, hi: float
) -> np.ndarray:
    """Heat-affected zone: oxide thickest on a line, thinning away from it.

    Tempering colour follows the heat, and heat into a weld bead or along a
    ground edge falls off from a line -- so the colour ladder appears as
    *bands* parallel to it, which is the reading the eye recognises. The
    falloff is banded rather than a clean ramp because pass overlap and
    convection make the isotherms lumpy.
    """
    h, w = shape
    x, y = grid_coords((h, w))
    ang = float(rng.uniform(0.0, np.pi))
    proj = x * np.float32(np.cos(ang)) + y * np.float32(np.sin(ang))
    proj = normalize01(proj)
    seam = float(rng.uniform(0.15, 0.85))
    width = float(rng.uniform(0.14, 0.34))
    dist = np.abs(proj - np.float32(seam)) / np.float32(width)
    # Exponent below 2 keeps a wide warm shoulder, so several rungs of the
    # ladder are on screen at once instead of one hot line.
    core = np.exp(-np.power(dist, np.float32(1.35)))
    wobble = normalize01(fbm((h, w), rng, freq=float(rng.uniform(1.2, 3.0)), octaves=3))
    field = np.clip(core * (0.80 + 0.34 * wobble), 0.0, 1.0)
    return (np.float32(lo) + np.float32(hi - lo) * field).astype(np.float32)


def _thickness_spill(
    shape: tuple[int, int], rng: np.random.Generator, lo: float, hi: float
) -> np.ndarray:
    """Oil spill: a warped blob, deep in the middle and feathering to nothing.

    Warped rather than plain fbm because a spill spreading over a surface
    tension gradient makes those long drawn-out swirls, and the swirls are
    where the eye reads a slick rather than a stain.
    """
    h, w = shape
    coords = grid_coords((h, w))
    wu, wv = double_warp(coords, rng, amp=float(rng.uniform(0.18, 0.34)), freq=1.8)
    body = normalize01(fbm_at(wu, wv, rng, freq=1.5, octaves=4, gain=0.55))
    pool = normalize01(fbm_at(wu, wv, rng, freq=0.9, octaves=2))
    # Multiplying the two gives a spill with genuinely thin *edges*: the
    # rainbow lives there, and it needs the field to reach the bottom of the
    # range rather than hovering mid-scale. Adding them instead leaves a floor
    # of a few hundred nm everywhere and the slick comes out uniformly grey.
    field = np.clip(np.power(body * pool, np.float32(0.75)), 0.0, 1.0)
    return (np.float32(lo) + np.float32(hi - lo) * field).astype(np.float32)


def _thickness_patch(
    shape: tuple[int, int], rng: np.random.Generator, lo: float, hi: float
) -> np.ndarray:
    """Anodising: smooth broad variation from current density over the part."""
    h, w = shape
    coarse = fbm((h, w), rng, freq=float(rng.uniform(1.0, 2.2)), octaves=3, gain=0.5)
    drift = fbm((h, w), rng, freq=float(rng.uniform(0.6, 1.2)), octaves=2)
    field = normalize01(coarse * np.float32(0.7) + drift * np.float32(0.5))
    return (np.float32(lo) + np.float32(hi - lo) * field).astype(np.float32)


_THICKNESS_FIELDS = {
    "weld": _thickness_weld,
    "spill": _thickness_spill,
    "patch": _thickness_patch,
}


def _film_spec(variant: str, film) -> dict:
    """Normalise the ``film`` argument into a full spec dict.

    ``film`` may be ``True`` (take the variant's preset), a system name from
    :data:`core.film.SYSTEMS`, or a dict overriding any of ``system``, ``nm``
    and ``field``.
    """
    spec = dict(FILM_VARIANTS.get(variant, FILM_VARIANTS["heat_tinted"]))
    if isinstance(film, str):
        spec["system"] = film
    elif isinstance(film, dict):
        spec.update(film)
    elif film is not True:
        raise TypeError(
            f"film must be None, True, a system name or a dict; got {type(film).__name__}"
        )
    if spec["system"] not in SYSTEMS:
        raise ValueError(
            f"unknown film system {spec['system']!r}; choose from {sorted(SYSTEMS)}"
        )
    if spec["field"] not in _THICKNESS_FIELDS:
        raise ValueError(
            f"unknown film field {spec['field']!r}; choose from {sorted(_THICKNESS_FIELDS)}"
        )
    return spec


def _filmed(
    shape: tuple[int, int],
    rng: np.random.Generator,
    base: np.ndarray,
    variant: str,
    spec: dict,
    *,
    brush_angle: float | None = None,
) -> np.ndarray:
    """Sheet metal under a transparent interference film.

    ``brush_angle`` controls the brushed features in radians.

    The film tints the **reflection** -- the specular lobe and the broad
    reflected-environment term -- and leaves the diffuse body alone, because
    interference happens to light that bounced off the two film surfaces, not
    to light that came back out of the metal. The tint is applied in linear
    light (interference is a statement about energy, and multiplying gamma
    values by it is simply the wrong sum), then encoded back so the rest of
    metal's display-value pipeline is untouched.
    """
    h, w = shape
    aspect = h / max(w, 1)
    gloss = _FILM_GLOSS.get(variant, "satin")

    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    if brush_angle is not None:
        angle = brush_angle
    x, y = grid_coords((h, w))
    ru, rv = rotate((x, y), -angle, (0.5, aspect * 0.5))
    f_across = _streak_freq(h, aspect, rng)
    if gloss == "brushed":
        streaks = fbm_at(ru, rv, rng, freq=(2.0, f_across), octaves=3, gain=0.55)
        shin, spec_w, nstr, aniso = 42.0, 0.40, 2.2, 0.85
    elif gloss == "hairline":
        streaks = fbm_at(ru, rv, rng, freq=(1.5, f_across * 0.35), octaves=3, gain=0.5)
        shin, spec_w, nstr, aniso = 75.0, 0.46, 1.8, 0.7
    else:  # satin
        streaks = fbm_at(ru, rv, rng, freq=(2.0, f_across * 0.08), octaves=3, gain=0.5)
        shin, spec_w, nstr, aniso = 120.0, 0.52, 1.5, 0.4

    scale = (h * w) / float(512 * 512)
    scratches = _scratches(
        (h, w),
        rng,
        int(rng.integers(6, 40) * scale),
        angle,
        amp=float(rng.uniform(0.05, 0.14)),
    )
    height = streaks * np.float32(0.30) + scratches
    # Darker body than the unfilmed finishes, and a correspondingly stronger
    # reflected term below. A filmed part reads as filmed only in proportion to
    # how much of its brightness is *reflection*: leave metal's usual bright
    # diffuse in place and the interference tint washes out to a few units of
    # chroma, which is to say to nothing.
    body = base * np.float32(rng.uniform(0.52, 0.66))
    albedo = body[None, None, :] * (1.0 + np.float32(0.04) * streaks[..., None])

    diffuse, specular = shade(
        np.clip(albedo, 0.0, 1.0),
        normalize01(height),
        light_dir=(-0.38, -0.44, 0.81),
        specular=float(rng.uniform(0.85, 1.15)) * spec_w,
        shininess=float(rng.uniform(0.8, 1.25)) * shin,
        normal_strength=float(rng.uniform(0.85, 1.2)) * nstr,
        aniso_dir=angle,
        aniso=aniso,
        ambient=0.46,
        spec_tint=0.85,
        return_parts=True,
    )

    # The reflected environment, kept separate from the body so the film can
    # colour it. Without this the film would only show inside the hot lobe,
    # whereas a tempered part is coloured over its whole face.
    diag = float(rng.uniform(0.4, 1.2))
    env = _env_gradient((h, w), rng, angle=diag + np.pi / 2)
    broad = _highlight_band(
        (h, w), rng, strength=0.30, softness=float(rng.uniform(0.4, 0.7)), angle=diag
    )
    tint_metal = (0.35 + 0.65 * base)[None, None, :]
    refl = (
        np.float32(rng.uniform(0.26, 0.36)) * env * broad[..., None] * tint_metal
    ).astype(np.float32)

    # Incidence angle: a flat sheet's normal barely moves, but which part of
    # the room each part of it reflects does -- and interference colour is a
    # function of the angle, which is why a tempered curve shifts hue across
    # its width. A smooth tilt field stands in for that.
    tilt = float(rng.uniform(0.25, 0.60))
    ramp = normalize01(x * np.float32(np.cos(diag)) + y * np.float32(np.sin(diag)))
    jitter = normalize01(fbm((h, w), rng, freq=float(rng.uniform(1.5, 3.5)), octaves=2))
    cos_theta = np.clip(
        1.0 - np.float32(tilt) * (0.75 * ramp + 0.25 * jitter), 0.18, 1.0
    ).astype(np.float32)

    thickness = _THICKNESS_FIELDS[spec["field"]](
        (h, w), rng, float(spec["nm"][0]), float(spec["nm"][1])
    )
    tint = film_tint(spec["system"], thickness, cos_theta)

    reflected = np.clip(specular + refl, 0.0, 1.0)
    reflected = linear_to_srgb(srgb_to_linear(reflected) * tint)

    rgb = diffuse + reflected
    # Hold the sheet inside display range without crushing its colour. Three
    # additive terms (body, lobe, reflection) have independent peaks, so some
    # palette and band draws overshoot badly -- and clipping is exactly where
    # interference colour dies, because a clipped pixel is white no matter what
    # tint it was handed. Scaling to a high percentile trades a few blown
    # specular pixels for a seventh of the image.
    peak = float(np.percentile(rgb.max(axis=-1), 99.5))
    if peak > 0.97:
        rgb = rgb * np.float32(0.97 / peak)
    glints = _glints((h, w), rng, count=int(rng.integers(40, 160) * scale), angle=angle)
    rgb = _add_glints(rgb, glints, broad, base, gain=float(rng.uniform(0.06, 0.14)))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def generate(
    shape: tuple[int, int],
    rng: np.random.Generator,
    variant: str = "brushed",
    film=None,
    *,
    brush_angle: float | None = None,
    iridescence: float = 0.0,
    source_angular_radius: float = SUN_ANGULAR_RADIUS_DEG,
    groove_pitch_um: tuple[float, float] = GROOVE_PITCH_UM,
    **params,
) -> np.ndarray:
    """Generate a metal texture as float32 (H, W, 3) in [0, 1].

    Args:
        shape: ``(height, width)`` in pixels.
        rng: seeded generator; the whole texture comes out of it.
        variant: one of :data:`VARIANTS`.
        film: transparent interference film on the metal. ``None`` (the
            default) means the variant decides -- which for ``brushed``,
            ``radial`` and ``polished`` means no film and no change of any kind,
            including no extra draw from ``rng``. Otherwise ``True`` for the
            variant's preset, a system name (``"oxide"``, ``"oil"``,
            ``"titania"``), or a dict overriding ``system``, ``nm`` or
            ``field``.
        brush_angle: brushed direction in clockwise image-coordinate degrees,
            where 0 is horizontal and 90 is vertical. Finite values wrap modulo
            360. ``None`` retains the seeded random direction. This option is
            supported only for the ``brushed`` variant.
        iridescence: weight of the groove-diffraction rainbow inside the
            scratches. **Defaults to 0**, which is off: the grooved finishes then
            render exactly as they always did, down to the byte. Applies to
            ``brushed``, ``radial`` and ``engine_turned`` -- the finishes that
            have a groove tangent for the grating equation to work with.
        source_angular_radius: angular radius of the light source in **degrees**.
            The default is sun-like, and the effect is strongly sensitive to it:
            a broad source smears the first order over itself and the colour goes
            away. See :func:`_source_gate`.
        groove_pitch_um: ``(lo, hi)`` bounds on the sub-pixel grating pitch in
            micrometres. Fine pitches fan the spectrum widely and survive a
            broader source; coarse ones need a near-point light.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown metal variant {variant!r}; choose from {VARIANTS}")
    if brush_angle is not None and variant != "brushed":
        raise ValueError("brush_angle is supported only for the brushed metal variant")
    angle_radians = _brush_angle_radians(brush_angle)
    base = _pick_palette(rng)
    h, w = int(shape[0]), int(shape[1])

    irid = (
        None
        if float(iridescence) <= 0.0
        else {
            "weight": float(iridescence),
            "source_angular_radius": float(source_angular_radius),
            "groove_pitch_um": groove_pitch_um,
        }
    )

    if film is None:
        film = True if variant in FILM_VARIANTS else None
    # Short-circuit before anything below can draw from rng: the classic
    # finishes must consume exactly the random stream they always did.
    if film is None:
        if variant == "brushed":
            return _brushed((h, w), rng, base, irid=irid, brush_angle=angle_radians)
        return _BUILDERS[variant]((h, w), rng, base, irid=irid)

    return _filmed(
        (h, w),
        rng,
        base,
        variant,
        _film_spec(variant, film),
        brush_angle=angle_radians,
    )
