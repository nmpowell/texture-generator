"""Wood textures: plain-sawn board faces and multi-plank panels.

Algorithm: growth rings come from a **grown ring-width series** -- an AR(1)
log-normal process in millimetres (:func:`_ring_widths`), cumulated into ring
boundaries and looked up by radius -- rather than from a constant spacing, so
no two rings are the same width. ``r`` is the distance to a pith placed far
off-canvas, so the rings read as gentle arcs rather than bullseyes, and the
coordinates are domain-warped before the ring lookup, which is what makes the
figure organic rather than mechanical. Within each ring the profile is a
**sawtooth** (:func:`_ring_sawtooth`): a smooth earlywood -> latewood climb and
an abrupt step at the ring boundary, with its polarity and sharpness set by the
species' porosity class. Fine grain streaks (severely anisotropic fbm), axial
vessel streaks (:func:`_pore_streaks`) and an optional knot are layered on top.
Albedo, height and the shading normals all derive from the same fields.

Those two -- evenly ruled ring widths and a symmetric soft ring ramp -- are the
signatures of a colour-ramp lookup, and between them they are what makes
procedural wood read as printed laminate however good its colour is.

Colour is specified in **CIELAB** (:data:`SPECIES`), not as hand-picked RGB,
because that is how wood colour is measured and published -- and because three
of the things wood colour actually does are single-axis moves in Lab and
nothing of the kind in RGB:

* **Finishing** deepens and saturates (L* down, C* up together;
  :data:`FINISHES`). Raw wood's air/cell-wall interface, n 1.00 -> 1.53,
  scatters light straight back as a white veil; a film or a penetrating oil
  index-matches that interface away, so the same pigment is seen through less
  scattering. An RGB multiply darkens *without* saturating, which is why tinted
  wood palettes never look finished.
* **Latewood** is a dL* (plus a small da*/db*) from the species value, not a
  second free colour -- it is the same wood, grown denser.
* **Board-to-board spread** is a small Lab offset per board
  (:data:`BOARD_LAB_SPREAD`). A floor of identically coloured boards is
  impossible, and it is the loudest "printed" tell at assembly scale.

Two consequences for the pipeline. The mean of the assembled albedo is pinned
back onto the specified colour (:func:`_recentre`), because the species value
*is* the board's mean and everything else here -- rings, drift, pores,
oxidation -- is variation about it. And the shading pass is asked to normalise
its lighting to its own mean (``normalise=True`` in
:func:`..core.shading.shade`, the same option
:func:`..core.shading.shade_translucent` has for paper), because otherwise it
darkens the render by an arbitrary amount and the measurement is lost.

**Chatoyance** is the last of the flat-print tells, and the only one that is not
in the albedo at all. Wood's fibres are aligned, so it reflects anisotropically
and its luster shifts with angle; figured wood shimmers because the fibre
direction varies across the face. Two lobes carry it (:data:`SPECULAR`): a sharp
surface lobe off the finish film, and a soft, separately-coloured fibre lobe
underneath it (:func:`..core.shading._fibre_lobe`). Both ride on a per-pixel
fibre-tangent field which is **not** a new noise field -- it is the gradient of
the same domain warp that bends the rings (:func:`_fibre_tangents`), following
Liu et al. 2016, so grain and figure agree by construction. The one thing added
on top is :data:`FIGURE_P`: an occasional curly or ribbon draw, which is nearly
free once the tangents exist because curl is just a bigger out-of-plane dip.

**Ray fleck** is the one figure that is not a variation of the fibre field but a
second *tissue*. Ray parenchyma is ~17% of hardwood xylem and runs radially, so
a quartersawn board -- now an explicit cut rather than an anonymous branch,
:data:`CUTS` -- exposes it as broad lens-shaped sheets: the tiger-oak flash
(:data:`RAY_FLECK_COVERAGE`). It is drawn as geometry, because at 0.2-0.8 mm
wide it is the one piece of anatomy here that the grid can actually resolve, and
its albedo contrast is deliberately small. What makes it read is that the ray's
fibres run *across* the board's: :func:`_ray_tangents` builds a second axis at
the fibre's own in-plane angle plus 90 degrees and no dip, and the ray lobe is a
**mixture**, not a rotation -- :func:`..core.shading.shade` blends the fibre and
ray lobes' *responses* per pixel by ``ray_weight``, rather than averaging the
two tangent fields into one, which would invent a diagonal fibre direction that
exists nowhere in the wood.

**A finish is two separate states**, not one. The CIELAB deltas above are a
*fitted appearance approximation* to what a finish does to the eye -- declared
as such (:func:`_finish_lab`) -- and carry no physical darkening; the finish's
actual optics are explicit state layered on top of the substrate instead: a
per-board film build in microns (:data:`FINISHES`'s ``film_um``) that
self-levels over the wood's relief into its own ``coat_lift`` above it and can
pool a pore no deeper than it is built, a refractive index (``ior``) that bends the
light reaching the fibre and ray lobes before they see it, and a fibre-lobe
tint (``fibre_tint``) for the light that has passed through the film. Nothing
physical is stacked on the CIELAB fit; the two live side by side.

**Shading happens in linear light** (:func:`_shade_fields`). The
display-referred sRGB albedo is decoded before
:func:`..core.shading.shade` and the clipped result re-encoded after, because
Lambert shading and lobe addition are linear-light operations and compositing
them directly in sRGB gamma-shapes the contrast: the *mean* survives either
way, since the lighting is mean-normalised and the albedo mean is pinned (see
above), but the contrast does not.

**Relief is carried in real millimetres, unnormalised.** The fine grain
streaks are no longer clamped to the sampling grid, so a texel's height
derivative describes the board's own physical slope rather than a slope that
depends on how many pixels the board happens to be rendered at --
``height_spacing`` (:func:`..core.shading.shade`) is what makes that
resolution-independence hold. ``normal_strength`` is now an exaggeration
factor on that physical slope (1.0 renders the true slope) rather than a knob
on an arbitrary normalised range, which is why its 1.1-2.2 default draw is
kept: a board this flat in true millimetres still needs a little
exaggeration to read as relief on a flat texture map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

import numpy as np

from ..core.colour import lab_to_srgb, linear_to_srgb, srgb_to_linear
from ..core.fields import normalize01, smoothstep
from ..core.noise import fbm_at, grid_coords
from ..core.shading import gaussian_blur, shade
from ..core.warp import rotate, warp

VARIANTS = ["board", "planks"]


# Heartwood colour as CIE L*a*b* (D65, 10 degree, as wood colour is reported).
#
#   lab   nominal heartwood colour: the mid of the published range.
#   half  half-width of that range, i.e. the *population* spread of the species.
#         Per-board draws are clamped inside it.
#   late  latewood offset (dL*, da*, db*) from the species value. Negative dL*:
#         latewood is the denser, darker band.
#   sap   sapwood colour, or None where sapwood is not visually distinct.
#         Cherry and walnut have dramatic splits -- walnut sapwood is nearly
#         white against a L* 40 heartwood -- and a pale band along one edge is
#         a strong authenticity cue.
#   sap_p probability that a board carries a sapwood band, 12-30% by species.
#         An occasional draw, not every board: sapwood is normally cut away.
#         Note this is per BOARD, so a panel of n planks shows a sapwood band
#         somewhere with probability 1 - (1 - sap_p)^n -- which for walnut at
#         0.3 over 5 planks is 5 panels in 6, and is why it felt like every
#         other seed even at the intended per-board rate.
#   aged  colour after ~12 months of light exposure, where the species moves
#         enough to be worth modelling (cherry darkens and reddens markedly).
#
# The L*a*b* figures are typical published values for the species, quoted here
# as the mid and half-width of the ranges they are reported over. They are
# **UNVERIFIED**: individual species figures are domain knowledge, not
# measurements taken for this library. ``late``, ``sap_p`` and the ranges the
# ageing interpolates over are reasoned, not published at all.
class SpeciesSpec(TypedDict):
    lab: tuple[float, float, float]
    half: tuple[float, float, float]
    late: tuple[float, float, float]
    sap: tuple[float, float, float] | None
    sap_p: float
    aged: NotRequired[tuple[float, float, float]]


SPECIES: dict[str, SpeciesSpec] = {
    # Southern yellow pine: earlywood is the species value, and its latewood is
    # the hardest dark line of anything here -- that contrast is the softwood.
    "pine": {
        "lab": (76.0, 7.5, 27.0),
        "half": (4.0, 1.5, 3.0),
        "late": (-18.0, 1.0, 1.5),
        "sap": None,
        "sap_p": 0.0,
    },
    # Hard maple: the palest wood in common use, and nearly achromatic in a*.
    "maple": {
        "lab": (81.5, 4.5, 19.5),
        "half": (3.5, 1.5, 2.5),
        "late": (-5.0, 0.8, 1.5),
        "sap": None,
        "sap_p": 0.0,
    },
    # White ash: ring-porous like oak but paler, and its sapwood is about the
    # same colour as its heartwood, so there is no band to draw.
    "ash": {
        "lab": (76.0, 6.5, 20.5),
        "half": (4.0, 1.5, 2.5),
        "late": (-11.0, 1.0, 1.5),
        "sap": None,
        "sap_p": 0.0,
    },
    "oak": {
        "lab": (65.0, 7.5, 21.5),
        "half": (5.0, 1.5, 2.5),
        "late": (-9.0, 1.0, 1.5),
        "sap": (78.0, 5.0, 20.0),
        "sap_p": 0.12,
    },
    "cherry": {
        "lab": (64.0, 13.0, 22.5),
        "half": (4.0, 2.0, 2.5),
        "late": (-5.0, 1.0, 1.5),
        # Nearly white against the heartwood: the most dramatic split here.
        "sap": (83.0, 6.0, 18.0),
        "sap_p": 0.26,
        "aged": (48.5, 17.0, 27.5),
    },
    "walnut": {
        "lab": (40.0, 7.0, 15.0),
        "half": (5.0, 2.0, 3.0),
        "late": (-6.0, 1.0, 1.5),
        "sap": (78.5, 6.0, 18.0),
        "sap_p": 0.28,
    },
    "sapele": {
        "lab": (45.0, 14.5, 21.0),
        "half": (5.0, 2.5, 3.0),
        "late": (-7.0, 1.0, 1.5),
        "sap": (74.0, 9.0, 20.0),
        "sap_p": 0.15,
    },
    # Swietenia, i.e. genuine mahogany -- a mid brown with a red cast, not the
    # crimson a hand-picked RGB palette always makes of it.
    "mahogany": {
        "lab": (50.0, 14.0, 23.0),
        "half": (5.0, 2.0, 3.0),
        "late": (-6.0, 1.0, 1.5),
        "sap": (76.0, 8.0, 20.0),
        "sap_p": 0.12,
    },
}

# Board-to-board Lab spread within one panel, as half-widths of (L*, a*, b*).
# The single cheapest thing that stops a multi-plank panel reading as a repeated
# print. Clamped into the species' published ``half`` range, so it never leaves
# the species.
#
# The a*/b* pair is **not** drawn as two independent numbers, for the same
# reason :func:`_finish_lab` moves chroma along the hue angle: board to board,
# wood varies in how much extractive is in the cell wall, which is a *chroma*
# difference, not a hue rotation. Drawn independently, a low b* against a high
# a* rotates the hue by 15-20 degrees -- enough to take walnut out of the brown
# family and render one board in a floor distinctly grey-purple. So the (a*, b*)
# half-widths above set the half-width of the chroma move, and the hue is left
# to :data:`BOARD_HUE_SPREAD`: a couple of degrees, not twenty.
BOARD_LAB_SPREAD = (4.0, 1.5, 2.5)
BOARD_HUE_SPREAD = 0.6


# A finish is a change of *scattering*, not a tint: it index-matches away the
# air/cell-wall interface that veils raw wood, so colour deepens (L* down) and
# saturates (C* up) at the same time, with a warm shift on top from the film's
# own colour. Applied in CIELAB for exactly that reason -- an RGB multiply can
# only do the first half.
#
#   dl/dc/db    ranges of dL*, dC* (chroma) and db* the finish adds.
#   p           share of boards. Almost all wood one sees is finished, so bare
#               wood is the rare case here rather than the default.
#   fibre_tint  colour of the fibre lobe (:func:`_fibre_colour`), normalised to
#               unit luminance: this light has passed through the film, so it
#               carries the film's own cast on top of the wood's pigment.
#               UNVERIFIED: the existing amber kept for oil and polyurethane
#               (an oil-based finish warms the light it passes, film or not);
#               neutral for bare wood; near-neutral for the non-yellowing
#               waterborne acrylic.
#   film_um     dry film build, in microns, as a (lo, hi) range -- the coat's
#               own optical state, separate from the CIELAB appearance
#               fit above: how thick a continuous film sits over the wood,
#               which is what a pore can pool under and a surface lobe can
#               reflect off independently of the substrate. UNVERIFIED: none
#               and oil (0, 0) -- a penetrating finish leaves no continuous
#               film to build; acrylic (25, 60) and polyurethane (50, 110) are
#               trade figures for 2-3 coats at ~25-40 um dry film each.
#   ior         refractive index of that film, for refracting the light before
#               the fibre and ray lobes (:func:`..core.shading.shade`'s
#               ``fibre_ior``). ``1.0`` where there is no continuous film to
#               refract through (``none``, ``oil``); ``1.50`` for the two film
#               finishes is **UNVERIFIED**: the conventional generic-finish
#               value of rendering practice, not a measured cured-film index. Measured
#               priors, recorded for provenance and not used: liquid linseed
#               oil 1.478-1.483, cured linseed ~1.57, liquid tung oil
#               1.518-1.522, shellac ~1.516, OpenPBR's own default 1.60.
#
# Deltas are typical measured magnitudes per finish class and are **UNVERIFIED**.
class FinishSpec(TypedDict):
    dl: tuple[float, float]
    dc: tuple[float, float]
    db: tuple[float, float]
    p: float
    fibre_tint: tuple[float, float, float]
    film_um: tuple[float, float]
    ior: float


FINISHES: dict[str, FinishSpec] = {
    "none": {
        "dl": (0.0, 0.0),
        "dc": (0.0, 0.0),
        "db": (0.0, 0.0),
        "p": 0.06,
        "fibre_tint": (1.0, 1.0, 1.0),
        "film_um": (0.0, 0.0),
        "ior": 1.0,
    },
    # Penetrating oil wets the cell wall itself, so it is the strongest move.
    "oil": {
        "dl": (-10.0, -4.0),
        "dc": (5.0, 12.0),
        "db": (3.0, 8.0),
        "p": 0.30,
        "fibre_tint": (1.06, 1.00, 0.88),
        "film_um": (0.0, 0.0),
        "ior": 1.0,
    },
    "polyurethane": {
        "dl": (-8.0, -3.0),
        "dc": (4.0, 10.0),
        "db": (4.0, 10.0),
        "p": 0.40,
        "fibre_tint": (1.06, 1.00, 0.88),
        "film_um": (50.0, 110.0),
        "ior": 1.50,
    },
    # Waterborne acrylic is the "non-yellowing" one, and barely moves the wood.
    "acrylic": {
        "dl": (-5.0, -2.0),
        "dc": (2.0, 6.0),
        "db": (1.0, 3.0),
        "p": 0.24,
        "fibre_tint": (1.02, 1.00, 0.96),
        "film_um": (25.0, 60.0),
        "ior": 1.50,
    },
}

# The film's self-levelling length, in millimetres: the blur radius
# :func:`_board_fields` uses to build the coat's own (smoother) surface over
# the substrate. UNVERIFIED: surface-tension levelling of a ~50 um film.
COAT_LEVEL_MM = 0.75

# --- Chatoyance: two specular lobes, driven by the finish --------------------
#
# Wood's fibres are aligned, so it reflects **anisotropically** and its luster
# shifts with angle. That is chatoyance, and layering two lobes is what makes a
# polished panel look deep rather than printed:
#
#   * a sharp surface lobe -- a dielectric at n ~ 1.5, white, so ``F0`` 0.04
#     applies whether that surface is the finish film or, bare, the cell wall
#     itself, and it is only mildly anisotropic (whatever grain telegraphs
#     through);
#   * a soft fibre lobe *underneath* it, aligned to the local fibre tangent, which
#     sweeps as the light moves. Marschner's separately-coloured component: this
#     light has been through pigment, so it is neither white nor fully saturated.
#
# Per finish class:
#
#   alpha_along/alpha_across  GGX-style roughness along and across the grain. The
#                             pair *is* the surface lobe's anisotropy; ``aniso``
#                             below is how it is handed to
#                             :func:`..core.shading.shade`, whose model splits one
#                             base exponent into an along/across pair itself, so
#                             the mean of the two alphas sets the base and
#                             ``aniso`` sets the ratio.
#   aniso                     0..1 anisotropy of the surface lobe.
#   fibre_exponent            longitudinal exponent of the fibre lobe.
#   fibre_weight             weight of the fibre lobe, at unbroken luster.
#
# The trend across the four rows is the thing to preserve: as the film gets
# glossier the surface lobe **sharpens and becomes more isotropic**, while the
# fibre lobe stays anisotropic and gets **stronger** -- an oiled or varnished face
# index-matches the surface veil away, so more of what one sees is the fibre.
#
# Every number here is an **UNVERIFIED** practical fitting value, not a
# measurement: the mapping of a finish class onto a roughness pair is domain
# knowledge, and the ``F0``-to-lobe-weight normalisation below is a fit to this
# renderer rather than a physical one.
SPECULAR = {
    # Raw, 180 grit: a broad low sheen and almost no fibre lobe -- the surface
    # veil this finish does not remove is what hides it.
    "none": dict(
        alpha_along=(0.55, 0.75),
        alpha_across=(0.60, 0.80),
        aniso=(0.05, 0.15),
        fibre_exponent=(15.0, 30.0),
        fibre_weight=(0.02, 0.05),
    ),
    # Penetrating oil / hardwax: the most *anisotropic* of the four, and the one
    # that first makes figure visible.
    "oil": dict(
        alpha_along=(0.25, 0.40),
        alpha_across=(0.35, 0.55),
        aniso=(0.30, 0.50),
        fibre_exponent=(30.0, 60.0),
        fibre_weight=(0.10, 0.20),
    ),
    # Waterborne acrylic stands in for satin varnish.
    "acrylic": dict(
        alpha_along=(0.18, 0.30),
        alpha_across=(0.22, 0.35),
        aniso=(0.15, 0.30),
        fibre_exponent=(40.0, 80.0),
        fibre_weight=(0.10, 0.18),
    ),
    # Polyurethane stands in for gloss varnish: the sharpest, most isotropic
    # surface lobe here, over the strongest fibre lobe.
    "polyurethane": dict(
        alpha_along=(0.04, 0.12),
        alpha_across=(0.05, 0.14),
        aniso=(0.05, 0.15),
        fibre_exponent=(60.0, 120.0),
        fibre_weight=(0.12, 0.25),
    ),
}

# Normal-incidence reflectance at n ~ 1.5 -- both a generic clear finish film
# AND, bare, the cell wall's own air interface, so this one conventional F0
# applies to bare and finished wood alike:
# ``F0 = ((n - 1)/(n + 1))^2 = 0.04``. VERIFIED (Schlick / Fresnel at n 1.5).
FINISH_F0 = 0.04

# Out-of-plane dip of the fibres below the sawn face, in degrees. All
# **UNVERIFIED** -- these are working figures for what the eye reads as each
# kind of figure, not measurements:
#
#   plain flatsawn, straight grain  0-3 deg, slowly varying
#   normal grain deviation          +/-2-6 deg, wavelength 30-150 mm
#   curly / fiddleback              +/-10-25 deg, quasi-sinusoidal, 3-15 mm
#                                   wavelength ACROSS the grain
#   ribbon / interlocked (sapele)   +/-3-8 deg, sign alternating, 5-25 mm bands
#   around a knot                   10-40 deg, within 10-30 mm of it
#
# The deviation field's *wavelength* is not a free parameter here: it comes from
# the board's existing domain warp (1.2-2.5 cycles per unit, i.e. 90-190 mm on a
# 225 mm board), which already sits inside the 30-150 mm band. That is the point
# of deriving the tangents from the distortion rather than from a second field.
#
# ...and it is also why the *dip* sigma is well under the 2-6 deg of total grain
# deviation above. At 90-190 mm the dip field is nearly the lowest frequency the
# board carries, so whatever amplitude it gets lands as broad soft patches in the
# fibre lobe. At 2-6 deg those patches read as uneven lighting or water staining
# on a floor rather than as luster -- worst on pine, ash and the plank variant.
# 1-3 deg is the working figure that keeps the luster sweeping along the grain
# without the damp-patch look; the rest of the 2-6 deg is in-plane, where the
# deflection term above already carries it.
GRAIN_DIP_SIGMA_DEG = (1.0, 3.0)
CURLY_DIP_DEG = (10.0, 25.0)
CURLY_WAVELENGTH_MM = (3.0, 15.0)
RIBBON_DIP_DEG = (3.0, 8.0)
RIBBON_BAND_MM = (5.0, 25.0)
KNOT_DIP_DEG = (10.0, 40.0)

# Figure is a property of the log, so it is drawn per panel and not per board --
# and it is an occasional draw, not a default. Curly maple and ribbon-striped
# sapele are the two the trade names after.
FIGURE_P = {
    "maple": {"curly": 0.25},
    "sapele": {"curly": 0.10, "ribbon": 0.35},
}
FIGURES = ["plain", "curly", "ribbon"]

# Fine-streak relief, in millimetres: the streak fbm has a std of ~0.25, so
# this amplitude puts its RMS at ~8 um. UNVERIFIED: the order of Ra 3-8 um /
# Rz 30-60 um for P120-P180 sanded hardwood, from the sanded-wood roughness
# literature (Gurau, Pro Ligno 10(3) 2014) that we have not read ourselves.
STREAK_RELIEF_MM = 0.03
# Broad machining-waviness relief, in millimetres. UNVERIFIED: machining
# waviness.
WAVINESS_MM = 0.05

# Latewood relief, in millimetres, *positive = latewood proud*. Sanding abrades
# the soft earlywood faster than the dense latewood, so a sanded face carries
# each latewood band slightly raised, with the abrupt drop at the ring boundary
# where next year's earlywood starts. UNVERIFIED: 10-30 um is woodworking
# knowledge of differential sanding on a P120-P180 face, not a profilometry
# figure. (Before the height field was physical this term sat at -0.06 in
# arbitrary units, recessed, and was swamped by the grid-relative streak noise;
# once the slopes were real it drew a bright embossed edge along every ring.)
LATEWOOD_RELIEF_MM = 0.03

# Out-of-plane angle between a vessel axis and the sawn face. A flatsawn board
# is never cut exactly parallel to the fibre, and |N(0, 3 deg)| is the working
# distribution; it is what makes exposed vessel length strongly right-skewed.
# (Domain knowledge, not a measured figure.)
PORE_ANGLE_SIGMA_DEG = 3.0

# Lateral softening of the pore mask, in pixels. Sub-pixel by construction: it
# rounds the streak's walls without spreading the streak. See :func:`_pore_streaks`.
_PORE_EDGE_PX = 0.55


# Vessel-pore anatomy and ring hue casts per species. Oak is ring-porous
# (rows of coarse vessels crowd the earlywood right after each latewood
# line); walnut is semi-ring-porous, its pore size tapering across the ring;
# mahogany/cherry carry diffuse pores at constant size; pine has no vessels at
# all, which is exactly what makes it read as a softwood.
#
# ``ew_dia_um``/``lw_dia_um`` (early/latewood vessel diameters) are VERIFIED
# measurements. ``*_density`` (vessels per mm^2), ``streak_mm`` (across-grain
# streak width), ``*_depth_um`` (trough depth), ``band_mm``, ``taper`` and
# ``dl_star`` are domain knowledge, not measurements -- oak's 60-200 um
# earlywood trough and walnut's 40-100 um are the anchors, cherry's and
# mahogany's are scaled from their vessel diameters.
#
# Hue shifts are small per-channel multipliers: earlywood runs lighter and
# yellower, latewood darker and browner-red.
class AnatomySpec(TypedDict):
    pore_class: str
    rings: tuple[int, int]
    line: float
    ew_dia_um: NotRequired[tuple[float, float]]
    lw_dia_um: NotRequired[tuple[float, float]]
    ew_density: NotRequired[tuple[float, float]]
    lw_density: NotRequired[tuple[float, float]]
    streak_mm: NotRequired[tuple[float, float]]
    ew_depth_um: NotRequired[tuple[float, float]]
    lw_depth_um: NotRequired[tuple[float, float]]
    dl_star: NotRequired[tuple[float, float]]
    band_mm: NotRequired[tuple[float, float]]
    taper: NotRequired[tuple[float, float]]


ANATOMY: dict[str, AnatomySpec] = {
    "pine": dict(pore_class="softwood", rings=(10, 22), line=1.3),
    "oak": dict(
        pore_class="ring-porous",
        rings=(8, 18),
        line=1.0,
        ew_dia_um=(200.0, 300.0),
        lw_dia_um=(20.0, 50.0),
        ew_density=(4.0, 10.0),
        lw_density=(20.0, 60.0),
        streak_mm=(0.20, 0.30),
        ew_depth_um=(60.0, 200.0),
        lw_depth_um=(10.0, 30.0),
        # 2-4 crowded rows of earlywood vessels. Roughly CONSTANT width
        # whatever the ring width: a wide ring means more latewood, not a
        # wider pore band.
        band_mm=(0.5, 1.5),
        dl_star=(20.0, 30.0),
    ),
    "walnut": dict(
        pore_class="semi-ring-porous",
        rings=(14, 26),
        line=0.75,
        ew_dia_um=(135.0, 215.0),
        lw_dia_um=(60.0, 100.0),
        ew_density=(10.0, 25.0),
        lw_density=(10.0, 25.0),
        streak_mm=(0.15, 0.20),
        ew_depth_um=(40.0, 100.0),
        lw_depth_um=(20.0, 45.0),
        # Pore size tapers smoothly over the first 30-60% of the ring.
        taper=(0.30, 0.60),
        dl_star=(15.0, 25.0),
    ),
    "mahogany": dict(
        pore_class="diffuse-porous",
        rings=(14, 28),
        line=0.6,
        ew_dia_um=(100.0, 200.0),
        lw_dia_um=(100.0, 200.0),
        ew_density=(4.0, 10.0),
        lw_density=(4.0, 10.0),
        streak_mm=(0.15, 0.15),
        ew_depth_um=(30.0, 80.0),
        lw_depth_um=(30.0, 80.0),
        dl_star=(15.0, 25.0),
    ),
    "cherry": dict(
        pore_class="diffuse-porous",
        rings=(16, 30),
        line=0.55,
        ew_dia_um=(50.0, 80.0),
        lw_dia_um=(50.0, 80.0),
        ew_density=(90.0, 140.0),
        lw_density=(90.0, 140.0),
        streak_mm=(0.06, 0.06),
        ew_depth_um=(10.0, 30.0),
        lw_depth_um=(10.0, 30.0),
        dl_star=(15.0, 22.0),
    ),
    # Hard maple: diffuse-porous with the finest vessels of the set, so its
    # pores are pure sub-pixel wash at any sane render size -- which is exactly
    # why maple reads as a smooth, almost featureless face.
    "maple": dict(
        pore_class="diffuse-porous",
        rings=(14, 28),
        line=0.5,
        ew_dia_um=(30.0, 60.0),
        lw_dia_um=(30.0, 60.0),
        ew_density=(100.0, 200.0),
        lw_density=(100.0, 200.0),
        streak_mm=(0.05, 0.05),
        ew_depth_um=(8.0, 20.0),
        lw_depth_um=(8.0, 20.0),
        dl_star=(15.0, 20.0),
    ),
    # White ash: ring-porous like oak, and its earlywood vessels are the
    # coarsest here -- a rank ash ring is visibly open to the eye.
    "ash": dict(
        pore_class="ring-porous",
        rings=(6, 14),
        line=1.0,
        ew_dia_um=(200.0, 350.0),
        lw_dia_um=(20.0, 60.0),
        ew_density=(3.0, 8.0),
        lw_density=(15.0, 45.0),
        streak_mm=(0.22, 0.34),
        ew_depth_um=(70.0, 220.0),
        lw_depth_um=(10.0, 30.0),
        band_mm=(0.5, 1.5),
        dl_star=(20.0, 30.0),
    ),
    "sapele": dict(
        pore_class="diffuse-porous",
        rings=(14, 28),
        line=0.6,
        ew_dia_um=(100.0, 180.0),
        lw_dia_um=(100.0, 180.0),
        ew_density=(4.0, 10.0),
        lw_density=(4.0, 10.0),
        streak_mm=(0.14, 0.14),
        ew_depth_um=(30.0, 80.0),
        lw_depth_um=(30.0, 80.0),
        dl_star=(15.0, 25.0),
    ),
}
EARLY_SHIFT = {
    "pine": (0.030, 0.018, -0.010),
    "oak": (0.020, 0.010, -0.010),
    "walnut": (0.015, 0.010, 0.000),
    "mahogany": (0.020, 0.005, -0.005),
    "cherry": (0.025, 0.010, -0.005),
    "maple": (0.018, 0.010, -0.004),
    "ash": (0.022, 0.012, -0.008),
    "sapele": (0.020, 0.006, -0.005),
}
LATE_SHIFT = {
    "pine": (0.010, -0.012, -0.022),
    "oak": (0.000, -0.015, -0.020),
    "walnut": (0.005, -0.010, -0.015),
    "mahogany": (0.012, -0.010, -0.012),
    "cherry": (0.008, -0.012, -0.016),
    "maple": (0.004, -0.008, -0.012),
    "ash": (0.000, -0.014, -0.018),
    "sapele": (0.010, -0.010, -0.012),
}


# Ring widths are a grown series, not a ruler. Dendrochronology models a
# ring-width series as **AR(1) in the log of the width**, and three things are
# going on at once -- all three are needed before the rings stop reading as
# evenly ruled lines:
#
#   * widths are LOG-NORMAL, so the wide rings sit further from the mean than
#     the narrow ones. ``mean_mm`` is the series mean for furniture/flooring
#     stock; :data:`RING_WIDTH_LIMITS_MM` is the full range of usable timber.
#   * successive rings are CORRELATED -- a good year follows a good year. The
#     lag-1 autocorrelation of the *raw* series is ~0.7 (:data:`RING_ACF1`), of
#     which most is the age trend; the AR(1) coefficient of the detrended
#     series is 0.3-0.4 (:data:`RING_AR1`). White noise reads as noise.
#   * rings NARROW OUTWARD from the pith (the age trend), fast at first and then
#     levelling off: a similar volume of wood laid down each year on an
#     ever-longer circumference is an ever-thinner ring.
#
# ``sigma`` is the standard deviation of the **raw** log widths -- trend and
# year-to-year term together, which is the spread you get by taking logs of a
# measured series and nothing else -- and it is the one "how dramatic is this
# board" knob. Real series sit at 0.20-0.35, so that is the band these ranges
# cover, ring-porous hardwoods at the quiet end and stressed conifers at the
# loud one. It is deliberately the raw spread and not the detrended index: the
# trend's variance is part of what the eye sees down a board, so specifying the
# index leaves the total unspecified and the series comes out ~1.5x too
# dispersed. The index follows from it at ``1/sqrt(1 + k)``, ~0.68x here.
#
# Mean sensitivity, the standard measure of how variable a series is (the mean
# relative change between adjacent rings), then follows from the AR part alone:
# ``E|dlog| = sqrt(2/pi) * sqrt(2 * (1 - phi)) * sd_ar``, i.e. ~0.6 sigma at
# these phi, which puts these ranges at 0.16-0.20 -- inside the 0.15-0.35 real
# series show, at the quiet end of it, which is where furniture stock belongs.
#
# These are the published ranges for the *statistics*; the split of the variance
# between trend and AR term is derived (see :func:`_ring_widths`), not measured.
RING_STATS = {
    "ring-porous": dict(mean_mm=(1.8, 3.5), sigma=(0.22, 0.30)),
    "semi-ring-porous": dict(mean_mm=(1.5, 3.2), sigma=(0.24, 0.32)),
    "diffuse-porous": dict(mean_mm=(1.5, 3.2), sigma=(0.24, 0.32)),
    "softwood": dict(mean_mm=(1.8, 3.5), sigma=(0.28, 0.35)),
}
# AR(1) coefficient of the *detrended* log-width series, and the lag-1
# autocorrelation of the raw series the trend and the AR term together must
# reproduce.
RING_AR1 = (0.30, 0.40)
RING_ACF1 = 0.70
RING_WIDTH_LIMITS_MM = (0.4, 8.0)


# A ring is not a symmetric ramp. It is a **sawtooth**: density climbs smoothly
# from the open earlywood to the dense latewood across the ring and then falls
# off a cliff at the ring boundary, where next spring's earlywood starts.
# (Hafidi & Wilkie, *Computer Graphics Forum* 44(2), 2025, arrive at the same
# asymmetric-and-discontinuous profile for the same reason.)
#
# Polarity and sharpness are set by the porosity class, not by taste:
#
#   lw_frac     latewood as a share of the ring.
#   width_mm    the ring widths, in mm, over which ``lw_frac`` sweeps that range.
#               A ring-porous species' earlywood pore band is a roughly fixed
#               width whatever the ring, so a WIDE ring means more latewood, not
#               more earlywood. ``None`` where the share does not track width.
#   trans_mm    width of the earlywood -> latewood transition in mm of ring. Oak
#               and ash switch inside a third of a millimetre, which is why
#               their ring boundary is a visible line.
#   trans_frac  ...or as a share of the ring, where the transition is gradual
#               enough to scale with the ring rather than sit at a fixed width.
#
# Overall ring contrast stays on ``ANATOMY[...]["line"]`` (capped at 1), which
# is already per species: diffuse-porous woods have weak ring contrast however
# sharp the transition, and pine's latewood is the hardest line here.
#
# The shares and transition widths are domain knowledge for the porosity
# classes; the within-zone density climbs are reasoned, not measured.
class RingProfileSpec(TypedDict):
    lw_frac: tuple[float, float]
    width_mm: tuple[float, float] | None
    trans_mm: tuple[float, float] | None
    trans_frac: tuple[float, float] | None


RING_PROFILE: dict[str, RingProfileSpec] = {
    "ring-porous": dict(
        lw_frac=(0.50, 0.85),
        width_mm=(1.0, 4.0),
        trans_mm=(0.10, 0.30),
        trans_frac=None,
    ),
    "semi-ring-porous": dict(
        lw_frac=(0.40, 0.70),
        width_mm=None,
        trans_mm=None,
        trans_frac=(0.30, 0.60),
    ),
    "diffuse-porous": dict(
        lw_frac=(0.35, 0.60),
        width_mm=None,
        trans_mm=None,
        trans_frac=(0.60, 0.95),
    ),
    # Pine's latewood is a narrow dense dark band, abrupt at its outer edge.
    "softwood": dict(
        lw_frac=(0.15, 0.30),
        width_mm=None,
        trans_mm=None,
        trans_frac=(0.10, 0.22),
    ),
}

# Sapwood: a pale band along ONE edge of the board, capped to the outer 10-35%
# of its width, with a wavy edge feathered over a few millimetres. Boards are
# normally cut to exclude it, so it is an occasional draw (``sap_p`` per
# species, 12-30%) -- and where it lands it is a *colour change in the same
# wood*, so the grain, rings and pores all have to read through it.
SAPWOOD_COVERAGE = (0.10, 0.35)
SAPWOOD_WAVE_MM = (2.0, 7.0)
SAPWOOD_FEATHER_MM = (2.0, 5.0)

# --- The cut ------------------------------------------------------------------
#
# How the board was sawn out of the log, which is the single biggest control on
# what its face looks like -- and, until now, an anonymous ``rng.random()``:
#
#   cathedral    flatsawn through or near the pith, modelled as an AXIS dipping
#                below the face, so the ring cones graze it and their traces
#                widen into nested pointed crowns.
#   flatsawn     the ordinary tangential board: a distant in-plane pith, gentle
#                arcs.
#   quartersawn  the saw plane runs *radially*, so the rings meet the face
#                nearly at right angles and read as near-straight parallel
#                lines. Geometrically that is a very distant pith, which is what
#                the old third branch already was -- named here because the
#                ray fleck below only exists on this cut.
#
# Shares are the trade's rough mix of what a board off a saw looks like, not a
# measurement (**UNVERIFIED**), and they are the shares this generator has drawn
# all along.
CUTS = ["cathedral", "flatsawn", "quartersawn"]
CUT_P = {"cathedral": 0.35, "flatsawn": 0.45, "quartersawn": 0.20}
# Pith distance in units of the across-grain reference extent, per cut. Further
# out is straighter: at 3-6x the board's width the rings are near-parallel.
PITH_DIST = {"flatsawn": (1.0, 3.0), "quartersawn": (3.0, 6.0)}

# --- Ray fleck: the figure of a quartersawn face -------------------------------
#
# Ray parenchyma is **~17% of hardwood xylem** by volume (VERIFIED; some species
# over 30%), and it is the one tissue this generator did not represent at all.
# Rays run *radially*, at right angles to the axial fibres, so on a flatsawn
# face the saw crosses them and they are invisible at this scale -- but a
# quartersawn cut runs along them and exposes each one as a broad flat sheet.
# That is **ray fleck**: the shimmering figure of "tiger oak", and the single
# most recognisable "this is real oak" cue there is.
#
# Two things make it read, and the second is the one that matters:
#
# * **Geometry.** Lens-shaped flecks, tapering to a point at both ends, scattered
#   along the grain -- 2-25 mm long with a mode at 5-8 mm and a long tail to
#   50 mm+, and 0.2-0.8 mm wide (occasionally 1.5), i.e. an aspect ratio of
#   10:1 to 40:1. Unlike the vessels and the fine uniseriate rays these are
#   *resolvable* at a sane render size, so they are drawn as geometry rather than
#   as a noise field. The length distribution is right-skewed: a uniform-length
#   population reads as a pattern however well the shape is drawn.
# * **A 90 degree in-plane tangent rotation.** The ray's own fibre direction is
#   perpendicular to the surrounding wood's, so the fleck sits in the anisotropic
#   fibre lobe at right angles to everything around it. *That* is why ray fleck
#   flashes light-then-dark as the light or the viewer moves: it is chatoyance,
#   not pigment. The albedo contrast is deliberately modest (dL* 3-8) and mostly
#   pale -- flecks that are simply painted lighter read as chalk dashes.
#
# Lengths are VERIFIED ranges for oak ray height; the log-normal fitted through
# them, the widths' dependence on aspect and the dL* are reasoned.
FLECK_LENGTH_MM = (2.0, 55.0)
FLECK_MODE_MM = (5.0, 8.0)
FLECK_LOG_SIGMA = 0.60
FLECK_ASPECT = (10.0, 40.0)
FLECK_WIDTH_MM = (0.2, 1.5)
FLECK_DL_STAR = (3.0, 8.0)
# Share of flecks that are *paler* than the wood around them. Ray tissue is
# mostly the lighter of the two, but not always, and a population that is all
# one sign loses the flicker.
FLECK_PALE_P = 0.75
# A 0.5 mm fleck is 1.1 px at 512 px across a 225 mm board, so most of this
# population is narrower than the grid. Floored at this width and faded in
# proportion, exactly as the sub-pixel vessels are (:func:`_pore_streaks`): the
# integrated lightness change is conserved, so a floored fleck is drawn wider
# and correspondingly fainter rather than as a hard one-pixel line.
FLECK_MIN_WIDTH_PX = 1.3
# Areal coverage of the *quartersawn* face by fleck, per species. True
# quartersawn oak runs 10-25%; commercial "quartersawn" (mixed ray angle) 3-12%.
# Oak is the exemplar because it is a two-ray-size species whose broad
# multiseriate rays are over 1 mm wide and several centimetres high; hard maple
# is the other, and subtler. Everything else here has narrow rays only, so its
# fleck is faint to absent -- and pine's uniseriate rays are invisible at any
# size, which is part of what makes it read as a softwood.
RAY_FLECK_COVERAGE = {
    "oak": (0.10, 0.25),
    "maple": (0.02, 0.06),
    "ash": (0.004, 0.015),
    "cherry": (0.003, 0.012),
    "walnut": (0.003, 0.012),
    "sapele": (0.004, 0.014),
    "mahogany": (0.004, 0.014),
    "pine": (0.0, 0.0),
}

# Gain on the ray lobe relative to the ordinary fibre lobe it is mixed with
# inside a fleck (:func:`..core.shading.shade`'s ``ray_gain``) -- moved
# here from the old gloss sheen factor the ray-fleck rotation used to carry.
# UNVERIFIED: the existing fleck sheen factor, not a fitted value.
RAY_LOBE_GAIN = 1.25


def _ring_widths(
    n: int,
    rng: np.random.Generator,
    *,
    mean_mm: float,
    sigma: float,
    phi: float | None = None,
) -> np.ndarray:
    """One board's ring-width series in millimetres, pith outward.

    An AR(1) process in log-width plus a negative-exponential age trend, which
    between them give the three properties a real series has: log-normal widths,
    a lag-1 autocorrelation of :data:`RING_ACF1`, and rings that narrow outward.
    See :data:`RING_STATS` for where the numbers come from.

    The variance split is derived rather than chosen. A smooth monotone trend
    has an autocorrelation of ~1 at lag 1, so for a variance ratio
    ``k = var(trend) / var(AR)`` the sum's lag-1 autocorrelation is
    ``(k + phi) / (k + 1)``; solving that for :data:`RING_ACF1` gives
    ``k = (acf - phi) / (1 - acf)``. That is why ``phi`` here is the 0.3-0.4 of
    a *detrended* series and not the 0.7 of a raw one -- the trend is modelled
    explicitly, so it must not be counted twice.

    Args:
        n: number of rings to grow.
        mean_mm: arithmetic mean of the returned widths, in mm.
        sigma: standard deviation of the *raw* log widths, trend and AR term
            together (see :data:`RING_STATS`). The detrended index comes out at
            ``1/sqrt(1 + k)`` of it, about 0.68x at these phi.
        phi: AR(1) coefficient; drawn from :data:`RING_AR1` if omitted.

    Returns:
        (n,) float64 widths in millimetres, clipped to
        :data:`RING_WIDTH_LIMITS_MM`.
    """
    n = max(int(n), 4)
    phi = float(rng.uniform(*RING_AR1)) if phi is None else float(phi)
    phi = float(np.clip(phi, 0.0, 0.95))
    sigma = max(float(sigma), 0.0)

    # ``sigma`` is the spread of the *raw* log widths, so it is the sum that has
    # to hit it and the split divides it: var(trend) + var(AR) = sigma**2 with
    # var(trend)/var(AR) = k. Applied as the AR spread instead (what this did)
    # the trend's variance lands on top of it and the raw series comes out ~1.5x
    # over-dispersed -- softwoods at std(log w) 0.43 against a 0.20-0.35 spec.
    k = max((RING_ACF1 - phi) / max(1.0 - RING_ACF1, 1e-6), 0.0)
    sd_ar = sigma / np.sqrt(1.0 + k)
    sd_trend = sd_ar * np.sqrt(k)

    # Age trend: the classic juvenile -> mature negative exponential, scaled to
    # the variance the split asked for rather than to a fixed amplitude, so the
    # statistics hold whatever the board's ring count.
    t = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    shape = -np.expm1(-float(rng.uniform(2.0, 4.0)) * t)
    spread = float(shape.std())
    trend = -shape * (sd_trend / spread) if spread > 1e-9 else np.zeros(n)

    # AR(1), started from its own stationary distribution so the innermost rings
    # are not systematically closer to the mean than the rest.
    innov = np.sqrt(max(1.0 - phi * phi, 1e-9)) * sd_ar
    z = rng.standard_normal(n)
    e = np.empty(n, dtype=np.float64)
    e[0] = z[0] * sd_ar
    for i in range(1, n):
        e[i] = phi * e[i - 1] + innov * z[i]

    w = np.exp(trend + e)
    # The mean is a specified quantity, so pin it. A multiplicative rescale is an
    # additive shift in log space, so it leaves both sigma and the
    # autocorrelation exactly where they were.
    w *= float(mean_mm) / max(float(w.mean()), 1e-9)
    return np.clip(w, *RING_WIDTH_LIMITS_MM)


def _ring_sawtooth(
    g: np.ndarray,
    width_mm: np.ndarray,
    pore_class: str,
    rng: np.random.Generator,
    jitter: np.ndarray,
    trans_floor: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The within-ring density profile: 0 in open earlywood, 1 in dense latewood.

    ``g`` is the position within the ring (0 just after a boundary, 1 at the
    next one) and ``width_mm`` the local *radial* ring width, so the latewood
    share can track it (see :data:`RING_PROFILE`). ``jitter`` is a per-ring
    random in roughly [-1, 1]: no two years lay down the same latewood.

    The profile is deliberately **discontinuous** -- it reaches 1 as ``g`` -> 1
    and the next ring starts back at 0 -- because that step is what a ring
    boundary is, and a symmetric ramp cannot produce it. Within each zone the
    density still climbs outward: earlywood cells thicken their walls as the
    season goes on, and the latewood is densest right at the boundary.

    ``trans_floor`` is a lower bound on the transition width, in the same
    within-ring units, and it is not optional once the profile is a step: pass
    the local ring-phase gradient per pixel. A step narrower than the sampling
    grid dithers rather than resolving, and the ring boundary is a *contour of a
    noisy field* (the phase carries the domain warp, the wiggle and any knot's
    pull), so wherever that noise has pixel-scale energy the contour breaks up
    into a crawling stipple that reads as a rendering fault, not as wood.

    Returns ``(density, latewood_weight)``. The second is the smooth 0->1
    earlywood/latewood indicator, used where a *zone* rather than a density is
    wanted (hue shift, sheen).
    """
    spec = RING_PROFILE[pore_class]
    lo, hi = spec["lw_frac"]
    if spec["width_mm"] is not None:
        w_lo, w_hi = spec["width_mm"]
        t = np.clip(
            (width_mm - np.float32(w_lo)) / np.float32(max(w_hi - w_lo, 1e-6)), 0.0, 1.0
        )
        lw = np.float32(lo) + np.float32(hi - lo) * t
    else:
        lw = np.full(g.shape, float(rng.uniform(lo, hi)), dtype=np.float32)
    lw = np.clip(lw * (np.float32(1.0) + np.float32(0.15) * jitter), 0.08, 0.92)

    if spec["trans_mm"] is not None:
        # A fixed width in mm, so a wide ring gets a *sharper* boundary in
        # ring-fraction terms -- which is what makes oak's ring boundary a line.
        trans = np.float32(float(rng.uniform(*spec["trans_mm"]))) / np.maximum(
            width_mm, np.float32(0.2)
        )
    else:
        # Every :data:`RING_PROFILE` entry sets exactly one of ``trans_mm``
        # and ``trans_frac``; the assert lets mypy narrow the latter here
        # without weakening the shared ``RingProfileSpec`` type.
        assert spec["trans_frac"] is not None
        trans = np.full(
            g.shape, float(rng.uniform(*spec["trans_frac"])), dtype=np.float32
        )
    trans = np.clip(np.maximum(trans, trans_floor), 0.01, 0.95)

    start = np.float32(1.0) - lw
    t = np.clip((g - start) / trans + np.float32(0.5), 0.0, 1.0)
    late_w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    early = np.float32(0.18) * np.clip(
        g / np.maximum(start, np.float32(1e-3)), 0.0, 1.0
    )
    late = np.float32(0.80) + np.float32(0.20) * np.clip(
        (g - start) / np.maximum(lw, np.float32(1e-3)), 0.0, 1.0
    )
    density = (early + (late - early) * late_w).astype(np.float32)
    return density, late_w


def _pick_finish(rng: np.random.Generator) -> str:
    """Draw a finish class by its share of real boards."""
    names = sorted(FINISHES)
    p = np.asarray([FINISHES[n]["p"] for n in names], dtype=np.float64)
    return str(rng.choice(names, p=p / p.sum()))


def _finish_delta(finish: str, rng: np.random.Generator) -> np.ndarray:
    """Draw one finish's ``(dL*, dC*, db*)``.

    Drawn once per board and then applied to every colour that board carries
    (heartwood *and* sapwood), because one board went through one finishing
    schedule -- the same can of varnish, the same number of coats.
    """
    spec = FINISHES[finish]
    return np.asarray(
        [
            rng.uniform(*spec["dl"]),
            rng.uniform(*spec["dc"]),
            rng.uniform(*spec["db"]),
        ],
        dtype=np.float64,
    )


def _finish_lab(lab, delta) -> np.ndarray:
    """Apply a finish's ``(dL*, dC*, db*)`` to a Lab colour.

    This is a fitted CIELAB *appearance* approximation to what a finish does
    to wood colour, not a physical scattering or absorption model -- the
    optics of the coat itself (film build, refractive index, fibre tint) are
    separate, explicit state, so nothing here is a stand-in for them
    and no physical darkening is stacked on top of this fit.

    The chroma term is the point: ``dC*`` moves the (a*, b*) pair *along its own
    hue angle*, which deepens the colour the way index-matching the cell wall
    does. Adding it to a* and b* separately would rotate the hue instead, and
    multiplying in RGB cannot saturate at all.

    ``dC*`` and ``db*`` are **not independent** for wood, and treating them as
    two separate additions is a factor-of-two error. Wood chroma is ~90% b*
    (oak: a* 7.5, b* 21.5, C* 22.8), so a dC* of +8.5 *is* a db* of +8 on its
    own. Applying the table's db* on top of it lands both totals near +13 --
    outside both of the ranges they came from. So dC* is honoured as the total
    chroma move, and db* as a *floor* on the warm shift: the film's own amber
    cast, which only shows where the chroma gain has not already delivered it.
    """
    lab = np.asarray(lab, dtype=np.float64).copy()
    dl, dc, db = (float(x) for x in np.asarray(delta, dtype=np.float64))
    lab[0] = lab[0] + dl
    b_floor = lab[2] + db
    chroma = float(np.hypot(lab[1], lab[2]))
    if chroma > 1e-6:
        scale = max(chroma + dc, 0.0) / chroma
        lab[1] *= scale
        lab[2] *= scale
    lab[2] = max(lab[2], b_floor)
    return lab


def _board_lab(
    species: str,
    rng: np.random.Generator,
    variation: float = 1.0,
    age: float = 0.0,
) -> np.ndarray:
    """This board's unfinished heartwood colour in Lab.

    Every board gets its own offset (:data:`BOARD_LAB_SPREAD`), clamped inside
    the species' published range so it stays the same species. ``variation``
    scales that spread so a caller can ask for the nominal species colour
    exactly.

    ``age`` (0 = fresh, 1 = the species' aged value) is passed *in* rather than
    drawn here, because ageing is a property of the panel and not of the board:
    boards laid at the same time have seen the same light for the same number of
    months, so a cherry floor darkens as a whole. Drawn per board it reads as
    boards from four different decades nailed down side by side.
    """
    spec = SPECIES[species]
    lab = np.asarray(spec["lab"], dtype=np.float64)
    if variation <= 0.0:
        return lab
    aged = spec.get("aged")
    if aged is not None:
        # Cherry darkens and reddens visibly within months of daylight, and the
        # aged colour is a long way from the fresh one -- so where on that
        # journey the panel sits matters more than any other colour draw here.
        t = float(np.clip(age, 0.0, 1.0)) * float(np.clip(variation, 0.0, 1.0))
        lab = lab + t * (np.asarray(aged, dtype=np.float64) - lab)
    half = np.asarray(spec["half"], dtype=np.float64)
    spread = np.asarray(BOARD_LAB_SPREAD, dtype=np.float64)
    # One draw along the species' own hue angle (chroma) and a small one across
    # it (hue), rather than three independent per-channel draws -- see
    # :data:`BOARD_LAB_SPREAD`.
    hue = np.arctan2(lab[2], lab[1])
    dc = float(np.hypot(spread[1], spread[2])) * rng.uniform(-1.0, 1.0)
    dh = BOARD_HUE_SPREAD * rng.uniform(-1.0, 1.0)
    off = variation * np.asarray(
        [
            spread[0] * rng.uniform(-1.0, 1.0),
            dc * np.cos(hue) - dh * np.sin(hue),
            dc * np.sin(hue) + dh * np.cos(hue),
        ],
        dtype=np.float64,
    )
    return np.clip(lab + off, lab - half, lab + half)


def _recentre(colour: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pin the assembled albedo's mean onto the specified colour.

    The species L*a*b* value *is* the board's mean face colour, and everything
    layered on top of it here -- the ring profile, the tonal drift, the pores,
    the oxidation gradient -- is variation *about* that mean, all of it with a
    non-zero average. Left uncorrected those biases stack up one way (every one
    of them darkens), which is half of why the hand-picked palette this replaced
    rendered so far below the measured colour. Same argument as the lighting
    normalisation in :func:`..core.shading.shade`, applied to the albedo.
    """
    mean = colour.reshape(-1, 3).mean(axis=0)
    gain = np.clip(
        np.asarray(target, dtype=np.float32) / np.maximum(mean, np.float32(1e-4)),
        0.2,
        5.0,
    )
    return np.clip(colour * gain[None, None, :], 0.0, 1.0).astype(np.float32)


def _darken_lstar(colour: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """Darken display-referred ``colour`` by ``dl`` in CIE L*, holding hue.

    Finish pools in an open pore, so a pore is a *lightness* drop measured in
    L*, not an arbitrary RGB multiply. Above the companding toe
    ``L* = 116*(Y/Yn)^(1/3) - 16``, so scaling the tristimulus Y by
    ``((L* - dl + 16) / (L* + 16))^3`` lands on the target lightness; applying
    that one scale to all three channels leaves the hue where it was.

    The round trip through linear light is not optional. ``colour`` carries sRGB
    display values, and Y is a *linear* quantity: run on the display values as
    though they were linear, this formula asks for dL* 15-30 and delivers 33-38,
    because the scale it computes is then applied through the encoding's own
    2.2-ish gamma on top. That was the whole reason the pore streaks read as
    near-black ink flecks rather than as soft grooves.
    """
    lin = srgb_to_linear(colour)
    y = np.clip(
        lin[..., 0] * np.float32(0.2126)
        + lin[..., 1] * np.float32(0.7152)
        + lin[..., 2] * np.float32(0.0722),
        1e-4,
        1.0,
    )
    ell = np.float32(116.0) * np.cbrt(y) - np.float32(16.0)
    scale = ((np.maximum(ell - dl, 0.0) + 16.0) / (ell + 16.0)) ** 3
    return linear_to_srgb(lin * scale[..., None])


def _pore_streaks(
    wu: np.ndarray,
    wv: np.ndarray,
    g: np.ndarray,
    rng: np.random.Generator,
    species: str,
    *,
    px_per_mm: float,
    mm_per_unit: float,
    px_per_unit: float,
    spacing: float | np.ndarray,
    along_axis: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Axial vessel streaks for one board face -- the anatomy a laminate lacks.

    On a flatsawn face you never see vessels as dots: the saw plane cuts each
    vessel tube lengthwise, so they read as axial grooves. For a vessel of
    diameter ``d`` whose axis leaves the face at out-of-plane angle ``theta``
    the exposed groove is

        L ~ d / tan(theta)

    which is derived geometry, not a literature figure. With
    ``theta ~ |N(0, 3 deg)|`` this is strongly right-skewed: a few very long
    streaks among many short ones, which no uniform dash pattern fakes. Note
    what *not* to use -- a vessel is many vessel elements fused end to end and
    runs decimetres, so sizing streaks from element length (200-600 um) is the
    classic procedural mistake and gives streaks ~10x too short.

    The areal pore fraction on the sawn face is the transverse vessel area
    fraction: ``n * sin(theta)`` axes cross unit face area and each covers
    ``d * L = d^2 / tan(theta)``, so coverage ``= n * d^2 * cos(theta) ~
    n * d^2`` -- independent of theta. That is what sets the threshold.

    ``wu``/``wv`` are the *warped* along/across-grain coordinates, so streaks
    follow the figure rather than a global axis, and ``along_axis`` says which
    array axis ``wu`` runs down. ``g`` is the position within the ring (0 at the
    earlywood start, 1 at the ring boundary) and ``spacing`` the ring width in
    the same units as ``wu``/``wv`` -- **per pixel**, since ring widths vary
    (:func:`_ring_widths`), or a scalar where they do not. Per-pixel is what
    keeps a ring-porous species' pore band a fixed width in a wide ring instead
    of stretching it.

    Returns ``(mask, delta_lstar, depth_mm, trough_mm)``, all zero for
    softwoods. ``depth_mm`` is ``mask``-weighted (the per-pixel *coverage*
    depth a shading pass adds straight to height); ``trough_mm`` is the
    unweighted per-pixel physical trough depth *where a vessel is present*
    -- the earlywood/latewood-interpolated ``depth_um`` a coat film would
    have to fill, independent of how much of a texel the vessel happens to
    cover.
    """
    anat = ANATOMY[species]
    if anat["pore_class"] == "softwood":
        # Pine has no vessels at all. That absence is the whole point.
        zeros = np.zeros_like(g)
        return zeros, zeros, zeros, zeros

    # Where in the ring the coarse vessels sit.
    if anat["pore_class"] == "ring-porous":
        # Found in mm from the ring start, NOT as a fraction of the ring, so
        # the band keeps its width when the ring is wide.
        pos_mm = g * (np.asarray(spacing, dtype=np.float32) * np.float32(mm_per_unit))
        band_mm = float(rng.uniform(*anat["band_mm"]))
        ew = 1.0 - smoothstep(band_mm * 0.55, band_mm, pos_mm)
    elif anat["pore_class"] == "semi-ring-porous":
        ew = 1.0 - smoothstep(0.0, float(rng.uniform(*anat["taper"])), g)
    else:
        ew = np.ones_like(g)
    ew = ew.astype(np.float32)

    d_ew = float(rng.uniform(*anat["ew_dia_um"])) / 1000.0
    d_lw = float(rng.uniform(*anat["lw_dia_um"])) / 1000.0
    d_mm = np.float32(d_lw) + np.float32(d_ew - d_lw) * ew
    n_ew = float(rng.uniform(*anat["ew_density"]))
    n_lw = float(rng.uniform(*anat["lw_density"]))
    density = np.float32(n_lw) + np.float32(n_ew - n_lw) * ew
    cov = np.clip(density * np.float32(np.pi / 4.0) * d_mm * d_mm, 0.0, 0.5)

    # Across-grain: one lattice cell per streak width, clamped so a cell never
    # falls below 2 px. Nothing sets the along:across anisotropy directly -- it
    # falls out as L/d, which for these species is the 20-60:1 that stretched
    # grain noise is usually hand-tuned to.
    streak_mm = float(rng.uniform(*anat["streak_mm"]))
    f_across = min(mm_per_unit / streak_mm, px_per_unit / 2.0)

    # Sub-pixel anatomy. At 512 px across a 225 mm board a 0.25 mm oak pore is
    # 0.6 px and a 0.06 mm cherry pore is 0.14 px, so most vessels cannot be
    # resolved at their own width: a vessel narrower than a grid cell is
    # drawn AT the cell's own width and faded in proportion, exactly as the
    # sub-pixel ray fleck already is (:func:`_ray_fleck`). ``fade`` is the
    # share of a drawn cell a real vessel actually covers, so thresholding at
    # ``draw = cov / fade`` draws the vessel COUNT the anatomy asked for, each
    # one wider and fainter than it truly is -- and the area is conserved by
    # construction *while* ``cov <= fade``, i.e. while at most one vessel's
    # worth of coverage falls in a cell. Once a species' true areal fraction
    # exceeds what one narrow-vessel-per-cell can carry even at full contrast,
    # ``draw`` saturates at 1: every pixel there already holds *several*
    # vessels, which is not a streak pattern any more but the physically
    # correct case for a uniform pedestal (``wash``, below). That pedestal
    # is what the earlier resolved/wash split also had; what has gone is its
    # ``sqrt`` contrast exaggeration and the full-contrast, artificially
    # fattened streaks it drew at coarse resolutions where a single vessel
    # is still what is being drawn.
    cell_px = px_per_unit / f_across
    fade = float(np.clip(streak_mm * px_per_mm / cell_px, 0.0, 1.0))
    draw = np.clip(cov / np.float32(max(fade, 1e-6)), 0.0, 1.0)
    # Per-pixel coverage the fade-capped streaks below cannot carry once
    # ``draw`` has saturated at 1 -- zero everywhere ``cov <= fade``.
    wash = np.clip(cov - np.float32(fade), 0.0, None).astype(np.float32)

    # One theta per streak: a field that varies across the grain at the streak
    # pitch but is near-constant along it. fbm is roughly Gaussian, so
    # |lane| / std(lane) is roughly half-normal -- exactly the theta we want.
    lane = fbm_at(wu, wv, rng, freq=(2.0, f_across), octaves=1)
    sd = max(float(lane.std()), 1e-6)
    theta = np.clip(
        np.abs(lane) * np.float32(PORE_ANGLE_SIGMA_DEG / sd), 0.12, 60.0
    ).astype(np.float32)
    length_mm = np.clip(d_mm / np.tan(np.radians(theta)), 0.15, 80.0).astype(np.float32)

    # A streak is an excursion of the noise below a threshold, and an excursion
    # is shorter than the noise's own correlation length -- the more so the
    # rarer it is. Measured on this basis (single octave), the mean run length
    # is 0.32/0.50/0.72 of the correlation length at coverage 0.02/0.10/0.35,
    # which 0.19 + 0.90*sqrt(cov) fits to a few percent. Divide it out so the
    # *visible* streak is the length the geometry asked for. (Empirical
    # calibration of this noise, not a physical relation.)
    ratio = np.clip(np.float32(0.19) + np.float32(0.90) * np.sqrt(draw), 0.2, 0.9)
    lam_mm = np.clip(length_mm / ratio, max(2.5 / px_per_mm, 0.5), 240.0)

    # Stretched noise whose along-grain frequency varies per streak. fbm_at
    # samples arbitrary coordinates, so the along-grain coordinate becomes a
    # *phase*: integrate the per-streak frequency along the grain, and the
    # local frequency of the result is exactly the one the geometry asked for.
    # (Multiplying the coordinate by a per-streak scale instead looks
    # equivalent but is not -- it adds a d(scale)/du term to the local
    # frequency, which for these fields is an order of magnitude larger than
    # the frequency itself and chops every streak down to a few pixels.)
    f_along = np.float32(mm_per_unit) / lam_mm
    du = np.gradient(wu, axis=along_axis).astype(np.float32)
    phase_u: np.ndarray = np.cumsum(f_along * du, axis=along_axis, dtype=np.float32)
    # Every lane's phase integral starts at 0, which would line the first
    # streak of each lane up down one edge. Offset each lane, using a slice
    # that is constant along the grain so it adds nothing to the frequency.
    edge = (slice(None), slice(0, 1)) if along_axis == 1 else (slice(0, 1), slice(None))
    phase_u = phase_u + fbm_at(
        np.zeros_like(wv[edge]), wv[edge], rng, freq=(1.0, f_across), octaves=1
    ) * np.float32(8.0)
    field = fbm_at(phase_u, wv, rng, freq=(1.0, f_across), octaves=1)

    # Threshold to the coverage target. fbm is not uniform, so flatten it
    # through its own empirical CDF first -- then "keep the lowest cov" really
    # keeps a fraction cov of the area, per pixel's local target.
    edges = np.linspace(0.0, 1.0, 129, dtype=np.float32)
    qs = np.quantile(field, edges).astype(np.float32)
    qs = qs + np.arange(qs.size, dtype=np.float32) * np.float32(1e-6)
    uni = np.interp(field, qs, edges).astype(np.float32)
    soft = np.maximum(draw * np.float32(0.35), np.float32(0.004))
    t = np.clip((draw - uni) / soft + np.float32(0.5), 0.0, 1.0)
    streaks = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    # A vessel is a rounded groove, not a slot: its walls turn over within the
    # pore's own diameter, which is sub-pixel here (see above). A sub-pixel blur
    # is what carries that -- a hard-edged mask reads as an ink fleck lying on
    # the surface rather than as a trough cut into it. It is a *blur*, so the
    # ``draw``-fraction area is conserved exactly.
    streaks = gaussian_blur(streaks, _PORE_EDGE_PX)
    # Per-pixel vessel coverage: ``streaks`` at up to full (drawn-cell)
    # contrast, scaled down by the share of that cell a real vessel covers,
    # plus the ``wash`` pedestal for whatever coverage the fade-capped
    # streaks cannot carry -- so the mean of ``mask`` is ``cov`` again, by
    # construction, in both regimes: ``cov <= fade`` gives
    # ``draw * fade == cov`` from the streaks alone (``wash`` is zero there);
    # ``cov > fade`` gives ``fade + (cov - fade) == cov`` from the
    # fade-saturated streaks plus the wash.
    mask = np.clip(streaks * np.float32(fade) + wash, 0.0, 1.0).astype(np.float32)

    # An open pore is a trough that catches the light, so it goes into height
    # as well as albedo -- that pairing is what stops it reading as print.
    depth_ew = float(rng.uniform(*anat["ew_depth_um"]))
    depth_lw = float(rng.uniform(*anat["lw_depth_um"]))
    depth_um = np.float32(depth_lw) + np.float32(depth_ew - depth_lw) * ew
    # The physical trough depth where a vessel is actually present, i.e. NOT
    # scaled by ``mask``'s area fraction -- a coat film pools against this
    # depth, not against the coverage-weighted one, since "how deep is the
    # vessel a texel happens to expose" does not depend on how much of that
    # texel the vessel covers.
    trough_mm = (depth_um / np.float32(1000.0)).astype(np.float32)
    depth_mm = (mask * depth_um / np.float32(1000.0)).astype(np.float32)

    # The dL* is a *contrast* -- what an open pore is worth against the wood
    # beside it -- and ``mask`` cannot carry it directly: it is scaled down as
    # an AREA by ``fade`` (plus the wash pedestal in the saturated regime),
    # which is right for coverage, gloss and depth and wrong for the contrast
    # a single vessel actually has. So the drop is taken off a peak-normalised
    # core (the sub-pixel blur above conserves a streak's area while knocking
    # the peak off a one-pixel core), faded and washed the same way as the
    # mask, which restores the contrast without touching the area the other
    # three consumers want.
    peak = float(np.quantile(streaks, 0.999))
    core: np.ndarray = streaks / np.float32(max(peak, 0.25)) if peak > 1e-4 else streaks
    delta_l = (
        np.clip(core * np.float32(fade) + wash, 0.0, 1.0)
        * np.float32(rng.uniform(*anat["dl_star"]))
    ).astype(np.float32)
    return mask, delta_l, depth_mm, trough_mm


def _fleck_lengths_mm(
    n: int, rng: np.random.Generator, *, mode_mm: float
) -> np.ndarray:
    """Draw ``n`` ray-fleck lengths in mm: log-normal, right-skewed.

    A ray's height on the face is its height in the tree, and that population is
    strongly right-skewed -- most rays are 5-8 mm high and a few run to 50 mm and
    beyond. So the mode is the parameter worth naming, not the mean: log-normal
    with mode ``exp(mu - sigma^2)``, hence ``mu = log(mode) + sigma^2``. At
    :data:`FLECK_LOG_SIGMA` that puts the median about 1.4x the mode and the 90th
    percentile about 3x it, which lands the bulk in the 2-25 mm the anatomy
    reports with a tail into the exceptional flecks.

    Drawn as one batch rather than per fleck because the caller sizes its
    population *from* the drawn areas (see :func:`_ray_fleck`).
    """
    sigma = float(FLECK_LOG_SIGMA)
    mu = float(np.log(max(mode_mm, 1e-3))) + sigma * sigma
    return np.clip(rng.lognormal(mu, sigma, size=int(n)), *FLECK_LENGTH_MM)


def _ray_fleck(
    shape: tuple[int, int],
    rng: np.random.Generator,
    species: str,
    phi: np.ndarray,
    *,
    tilt: float,
    along_x: bool,
    px_per_mm: float,
    coverage: float | None = None,
) -> dict:
    """Scatter lens-shaped ray flecks over a quartersawn face.

    See :data:`RAY_FLECK_COVERAGE` for what this is and why it only exists on
    this cut. Each fleck is an ellipse-in-profile -- half-width
    ``(W/2) * sqrt(1 - (2a/L)^2)`` along its own axis, so it tapers to a point at
    both ends rather than ending in a dash -- laid down along the **local fibre
    direction** ``phi`` at its centre, not along a global axis, so a fleck
    population follows the board's figure the way the grain does.

    The count is not a knob: it follows from the coverage target. Overlaps are
    accounted for, because at 10-25% they are not negligible -- for a Poisson
    scatter the union covers ``1 - exp(-sum_area / area)``, so the areas have to
    add up to ``-log(1 - coverage)`` times the face, not to ``coverage`` times
    it. Centres are drawn over a canvas *padded* by half the longest fleck so the
    density is uniform right to the edge instead of thinning out over the last
    centimetre.

    ``phi`` is the grain-frame in-plane fibre angle (:func:`_fibre_frame`), and
    ``tilt``/``along_x`` are what turn it back into an image-space direction.

    Returns ``{"mask", "dl", "length_mm", "width_mm"}``: the 0..1 coverage mask,
    the *signed* dL* the fleck asks for at each pixel (positive = paler than the
    wood around it), and the geometry of the flecks that actually landed on the
    face, for measurement.
    """
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.float32)
    dl = np.zeros((h, w), dtype=np.float32)
    lo, hi = RAY_FLECK_COVERAGE.get(species, (0.0, 0.0))
    cov = float(rng.uniform(lo, hi)) if coverage is None else float(coverage)
    if cov <= 0.0 or h < 4 or w < 4:
        z = np.zeros(0, dtype=np.float32)
        return {"mask": mask, "dl": dl, "length_mm": z, "width_mm": z}

    # One population, sized by cumulating its own drawn areas until the coverage
    # target is met -- which is why the lengths come as a pool rather than one at
    # a time.
    pool = 6000
    length_mm = _fleck_lengths_mm(pool, rng, mode_mm=float(rng.uniform(*FLECK_MODE_MM)))
    # Width is not an independent draw: the aspect ratio is the reported quantity
    # (10:1 to 40:1), so a long fleck is a wider one. The clip is the anatomy's
    # own 0.2-1.5 mm.
    aspect = rng.uniform(FLECK_ASPECT[0], FLECK_ASPECT[1], size=pool)
    width_mm = np.clip(length_mm / aspect, *FLECK_WIDTH_MM)
    length_px = (length_mm * px_per_mm).astype(np.float64)
    half_true = width_mm * px_per_mm * 0.5
    half_px = np.maximum(half_true, FLECK_MIN_WIDTH_PX * 0.5)
    # Sub-pixel flecks are drawn at the floor width and faded in proportion, so
    # the *integrated* lightness change is the one the geometry asked for.
    fade = np.clip(half_true / half_px, 0.0, 1.0)

    pad = 0.5 * float(length_px.max()) + 2.0
    padded = (h + 2.0 * pad) * (w + 2.0 * pad)
    target = -np.log(max(1.0 - cov, 1e-3)) * padded
    area = (np.pi / 2.0) * length_px * half_px
    n = min(int(np.searchsorted(np.cumsum(area), target)) + 1, pool)

    rows = rng.uniform(-pad, h + pad, size=n)
    cols = rng.uniform(-pad, w + pad, size=n)
    dl_star = rng.uniform(*FLECK_DL_STAR, size=n) * np.where(
        rng.random(n) < FLECK_PALE_P, 1.0, -1.0
    )

    # Grain-frame angle -> image-space direction, the same way
    # :func:`_fibre_tangents` backs its tangents out: undo the tilt rotation,
    # then the axis swap. Rows follow y and columns follow x.
    ri = np.clip(rows.astype(np.int64), 0, h - 1)
    ci = np.clip(cols.astype(np.int64), 0, w - 1)
    ph = np.asarray(phi, dtype=np.float64)[ri, ci]
    ca, sa = float(np.cos(tilt)), float(np.sin(tilt))
    du, dv = np.cos(ph), np.sin(ph)
    a_, b_ = du * ca + dv * sa, -du * sa + dv * ca
    dir_c, dir_r = (a_, b_) if along_x else (b_, a_)

    kept_len: list[float] = []
    kept_wid: list[float] = []
    for i in range(n):
        lp, hp = float(length_px[i]), float(half_px[i])
        dr, dc = float(dir_r[i]), float(dir_c[i])
        r0, c0 = float(rows[i]), float(cols[i])
        ext_r = 0.5 * lp * abs(dr) + hp * abs(dc) + 1.0
        ext_c = 0.5 * lp * abs(dc) + hp * abs(dr) + 1.0
        rlo, rhi = int(np.floor(r0 - ext_r)), int(np.ceil(r0 + ext_r)) + 1
        clo, chi = int(np.floor(c0 - ext_c)), int(np.ceil(c0 + ext_c)) + 1
        rlo, rhi = max(rlo, 0), min(rhi, h)
        clo, chi = max(clo, 0), min(chi, w)
        if rhi <= rlo or chi <= clo:
            continue
        rr = np.arange(rlo, rhi, dtype=np.float32)[:, None] - np.float32(r0)
        cc = np.arange(clo, chi, dtype=np.float32)[None, :] - np.float32(c0)
        # Along / across the fleck's own axis.
        along = rr * np.float32(dr) + cc * np.float32(dc)
        across = -rr * np.float32(dc) + cc * np.float32(dr)
        t = np.clip(np.float32(1.0) - (along * np.float32(2.0 / lp)) ** 2, 0.0, 1.0)
        half = np.float32(hp) * np.sqrt(t)
        # One pixel of antialiasing on the edge: these are 1-6 px wide, so a hard
        # mask reads as aliased dashes.
        shp = np.clip(half - np.abs(across) + np.float32(0.5), 0.0, 1.0)
        if not shp.any():
            continue
        box = mask[rlo:rhi, clo:chi]
        take = shp > box
        box_dl = dl[rlo:rhi, clo:chi]
        box_dl[take] = (shp * np.float32(dl_star[i] * fade[i]))[take]
        np.maximum(box, shp, out=box)
        kept_len.append(float(length_mm[i]))
        kept_wid.append(float(width_mm[i]))

    return {
        "mask": mask,
        "dl": dl,
        "length_mm": np.asarray(kept_len, dtype=np.float32),
        "width_mm": np.asarray(kept_wid, dtype=np.float32),
    }


def _fibre_frame(
    u: np.ndarray,
    v: np.ndarray,
    wu: np.ndarray,
    wv: np.ndarray,
    *,
    along_x: bool,
    px_per_unit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The two derivatives of the board's distortion the fibre field is built on.

    Split out of :func:`_fibre_tangents` because it draws **no random numbers**:
    the ray fleck needs the in-plane fibre angle to orient itself
    (:func:`_ray_fleck`), and it has to have it *before* the tangents are
    built, while the tangents stay the last thing every *earlier* layer's
    draw keeps its sequence against (the film build after them draws ``rng``
    too, but only once that whole earlier sequence -- fleck, tangents -- is
    fixed).

    Returns ``(phi, ddu_dv)``: the in-plane fibre angle in the grain frame, and
    the cross-grain shear that becomes the out-of-plane dip.
    """
    ax_along = 1 if along_x else 0
    ax_across = 0 if along_x else 1
    dist_u = (wu - u).astype(np.float32)
    dist_v = (wv - v).astype(np.float32)
    # Coordinate units, not pixels: one unit is ``px_per_unit`` pixels.
    ddv_du = np.gradient(dist_v, axis=ax_along).astype(np.float32) * np.float32(
        px_per_unit
    )
    ddv_dv = np.gradient(dist_v, axis=ax_across).astype(np.float32) * np.float32(
        px_per_unit
    )
    ddu_dv = np.gradient(dist_u, axis=ax_across).astype(np.float32) * np.float32(
        px_per_unit
    )
    # ``wv = v + dist_v``, so ``grad(wv) = (d(dist_v)/du, 1 + d(dist_v)/dv)`` and
    # the fibre runs at right angles to it -- along the ring, not up its normal.
    phi = np.arctan2(-ddv_du, np.float32(1.0) + ddv_dv).astype(np.float32)
    return phi, ddu_dv


def _grain_frame_to_image(
    tu: np.ndarray, tv: np.ndarray, tz: np.ndarray, *, tilt: float, along_x: bool
) -> np.ndarray:
    """Back a ``(tu, tv, tz)`` grain-frame vector out into image ``(x, y, z)``.

    ``u, v`` were rotated by ``+tilt`` about the board centre, so a vector's
    components go the other way; then undo the axis swap that ``along_x=False``
    made. Shared by :func:`_fibre_tangents` and :func:`_ray_tangents` so the two
    axes are backed out of the same frame identically.
    """
    ca, sa = np.float32(np.cos(tilt)), np.float32(np.sin(tilt))
    a = tu * ca + tv * sa
    b = -tu * sa + tv * ca
    tx, ty = (a, b) if along_x else (b, a)
    return np.stack([tx, ty, tz], axis=-1).astype(np.float32)


def _ray_tangents(phi: np.ndarray, *, tilt: float, along_x: bool) -> np.ndarray:
    """Per-pixel unit ray-fleck axis: crosswise to the fibre, no dip.

    Ray parenchyma runs *radially*, at right angles to the axial fibres
    (:data:`RAY_FLECK_COVERAGE`), so its in-plane angle is the fibre's own
    grain-frame angle ``phi`` (:func:`_fibre_frame`) turned a further 90
    degrees, with zero out-of-plane dip -- there is no evidence for ray relief
    Backed out of the grain frame with the same tilt/axis-swap
    :func:`_fibre_tangents` uses, via :func:`_grain_frame_to_image`, so the two
    axes agree by construction. Draws no random numbers.
    """
    phi_ray = (phi + np.float32(0.5 * np.pi)).astype(np.float32)
    tu = np.cos(phi_ray).astype(np.float32)
    tv = np.sin(phi_ray).astype(np.float32)
    tz = np.zeros_like(tu)
    return _grain_frame_to_image(tu, tv, tz, tilt=tilt, along_x=along_x)


def _fibre_tangents(
    u: np.ndarray,
    v: np.ndarray,
    wu: np.ndarray,
    wv: np.ndarray,
    rng: np.random.Generator,
    *,
    tilt: float,
    along_x: bool,
    px_per_unit: float,
    mm_per_unit: float,
    figure: str = "plain",
    knot: dict | None = None,
) -> np.ndarray:
    """Per-pixel unit fibre tangents, taken from the board's own distortion field.

    Liu, Dong, Hasan & Marschner, *Simulating the Structure and Texture of Solid
    Wood*, ACM TOG 35(6), 2016 (VERIFIED,
    <https://www.cs.cornell.edu/projects/wood/>): the fibre directions an
    anisotropic specular needs **follow from the growth distortions**. So there is
    deliberately no second noise field here for "figure". ``(wu - u, wv - v)`` is
    the domain warp that bent the rings; its gradient bends the fibres, and the
    two therefore agree by construction rather than by tuning. A separate figure
    field would drift out of register with the rings, which is the tell that says
    "two textures multiplied together".

    Three terms, all off the one distortion:

    * **In-plane deflection.** The fibre runs *along* the ring, i.e. along a
      contour of the warped across-grain coordinate ``wv``, so its direction is
      perpendicular to that coordinate's gradient:
      ``theta = atan2(-d(wv)/du, d(wv)/dv)``. Both terms matter. Dropping the
      ``d(wv)/dv`` one assumes the warp does not stretch across the grain, and
      the minus sign is the difference between following the ring and running up
      its normal -- a sign error there survives every unit-length and variation
      check while reflecting the whole grain field about the ring boundary,
      which is why :mod:`..tests.test_wood_chatoyance` now pins
      ``t . grad(wv) == 0`` directly.
    * **Out-of-plane dip.** The cross-grain shear of the same field,
      ``d(dist_u)/dv``, scaled to :data:`GRAIN_DIP_SIGMA_DEG`. This is a
      **modelling assumption** and worth naming as one: the warp is 2D, so it
      carries only the in-plane trace of a 3D shear, and this takes that trace as
      a stand-in for the out-of-plane part. What it buys is the right spatial
      statistics for free -- same wavelength band, same registration with the
      rings.
    * **Figure**, where the board has any: a quasi-sinusoid across the grain
      (:data:`CURLY_DIP_DEG`) or sign-alternating bands
      (:data:`RIBBON_DIP_DEG`), both riding on the *warped* across-grain
      coordinate so the curl follows the figure rather than ruling straight
      lines over it. Plus the knot's own steep dip.

    Ray fleck is no longer a fourth term here: it is its own axis
    (:func:`_ray_tangents`), since it is a second tissue rather than a
    variation of this one (:data:`RAY_FLECK_COVERAGE`).

    Returns an (H, W, 3) unit-length field in image ``(x, y, z)``, z negative
    where the fibre dips below the face.
    """
    # ``u`` runs down one array axis and ``v`` the other, up to the board's +/-6
    # degree tilt, which is small enough to ignore when picking the axis.
    phi, ddu_dv = _fibre_frame(u, v, wu, wv, along_x=along_x, px_per_unit=px_per_unit)

    sd = max(float(ddu_dv.std()), 1e-8)
    dip_deg = ddu_dv * np.float32(float(rng.uniform(*GRAIN_DIP_SIGMA_DEG)) / sd)
    dip_deg = np.clip(dip_deg, -12.0, 12.0).astype(np.float32)

    if figure in ("curly", "ribbon"):
        if figure == "curly":
            lam_mm = float(rng.uniform(*CURLY_WAVELENGTH_MM))
            amp = float(rng.uniform(*CURLY_DIP_DEG))
        else:
            lam_mm = float(rng.uniform(*RIBBON_BAND_MM))
            amp = float(rng.uniform(*RIBBON_DIP_DEG))
        # Never ask for a wave the grid cannot carry: 3 mm at 512 px across a
        # 225 mm board is under 7 px per cycle, and below that a sinusoid
        # aliases into a stipple rather than reading as figure.
        lam = max(lam_mm / mm_per_unit, 8.0 / px_per_unit)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        # Quasi-sinusoidal, not sinusoidal: real fiddleback wanders in both
        # phase and amplitude, and a pure sine reads as corrugated iron.
        jitter = fbm_at(u, v, rng, freq=(2.0, 2.5), octaves=2)
        wave = np.sin(
            np.float32(2.0 * np.pi) * wv / np.float32(lam)
            + np.float32(phase)
            + np.float32(1.4) * jitter
        ).astype(np.float32)
        if figure == "ribbon":
            # Interlocked grain alternates in *sign* between bands rather than
            # sweeping smoothly through them, so square the sinusoid off.
            wave = np.tanh(np.float32(2.5) * wave).astype(np.float32)
        depth = np.clip(
            0.75 + 0.5 * fbm_at(u, v, rng, freq=(1.5, 1.5), octaves=2), 0.3, 1.4
        ).astype(np.float32)
        dip_deg = dip_deg + np.float32(amp) * depth * wave

    if knot is not None:
        # Fibres sweep steeply up into a branch, and that is why a knot flares
        # in the reflection long before its colour arrives.
        dip_deg = (
            dip_deg
            + np.float32(
                float(rng.uniform(*KNOT_DIP_DEG))
                * (1.0 if rng.random() < 0.5 else -1.0)
            )
            * knot["dip"]
        )

    theta = np.deg2rad(dip_deg).astype(np.float32)
    cos_t = np.cos(theta)
    tu = (cos_t * np.cos(phi)).astype(np.float32)
    tv = (cos_t * np.sin(phi)).astype(np.float32)
    tz = (-np.sin(theta)).astype(np.float32)

    return _grain_frame_to_image(tu, tv, tz, tilt=tilt, along_x=along_x)


@dataclass(frozen=True)
class BoardFields:
    """The anatomy of one sawn board face, before it is shaded.

    The seam between the board's anatomy and :func:`_shade_fields`'s lighting
    call: everything a renderer needs to know about the wood itself, and
    nothing about how it is lit.

    Attributes:
        albedo: (H, W, 3) float32 sRGB display-referred colour, in [0, 1].
        height: (H, W) float32 substrate relief, in millimetres, not
            normalised: physical slopes are what :func:`_shade_fields`
            asks :func:`..core.shading.shade` for.
        coat_lift: (H, W) float32 millimetres, >= 0: how far the finish
            film's own (smoother, self-levelled) surface sits above
            ``height``. ``height + coat_lift`` is the coat's own surface,
            which the surface specular lobes are shaded off; the diffuse
            term, cavity and the fibre/ray lobes stay on ``height`` itself.
        tangent: (H, W, 3) float32 unit fibre-tangent axis, in image (x, y, z)
            with z negative where the fibre dips below the face
            (:func:`_fibre_tangents`).
        ray_tangent: (H, W, 3) float32 unit ray-fleck axis, crosswise to the
            fibre and with no dip (:func:`_ray_tangents`).
        ray_weight: (H, W) float32 in [0, 1], the ray-fleck coverage mask
            (:func:`_ray_fleck`); zero everywhere off a quartersawn face.
        coat_gloss: (H, W) float32 sheen modulation for the surface (coat)
            specular lobes -- breaks where a pore is still open under the
            film and where the ray fleck's own sheen adds.
        fibre_lustre: (H, W) float32 sheen modulation for the fibre lobe --
            no fleck factor, since the ray lobe carries its own gain.
    """

    albedo: np.ndarray
    height: np.ndarray
    coat_lift: np.ndarray
    tangent: np.ndarray
    ray_tangent: np.ndarray
    ray_weight: np.ndarray
    coat_gloss: np.ndarray
    fibre_lustre: np.ndarray


def _board_fields(
    shape: tuple[int, int],
    rng: np.random.Generator,
    species: str,
    *,
    finish: str = "none",
    variation: float = 1.0,
    age: float = 0.0,
    sap_p: float | None = None,
    knot_p: float | None = None,
    ring_sigma: float | None = None,
    figure: str = "plain",
    cut: str | None = None,
    along_x: bool = True,
    ref_across: float | None = None,
    px_per_mm: float | None = None,
) -> BoardFields:
    """Build the :class:`BoardFields` for one sawn board face.

    Colour is drawn here rather than passed in, and that is deliberate: it is
    **one board's** colour, so a multi-plank panel gets a fresh Lab offset and a
    fresh finish draw per strip (see :data:`BOARD_LAB_SPREAD`). ``variation``
    scales that spread; 0 gives the species' nominal Lab exactly, which is what
    makes the rendered colour testable against a published figure. ``age`` is
    the exception and comes from the caller, because it belongs to the panel
    rather than to the board -- see :func:`_board_lab`.

    ``along_x`` selects the grain direction (the panel's long axis); the
    board's own aspect ratio is respected so plank strips still work.
    ``ref_across`` is the across-grain extent that ring spacing and knot size
    are measured against -- pass the whole panel's extent for plank strips so
    a narrow plank shows proportionally fewer rings.

    ``sap_p``/``knot_p`` override the species' sapwood probability and the
    board's knot probability; both exist so the two features can be turned off
    for measurement, since each is a deliberate *departure* from the board's
    mean colour and would otherwise be averaged into it.

    ``ring_sigma`` overrides the standard deviation of the raw log ring-width
    series (:data:`RING_STATS`), which is the "how dramatic is this board" knob:
    mean sensitivity follows from it.

    ``px_per_mm`` is the physical scale, and is used *only* to size the vessel
    pore layer (:func:`_pore_streaks`); the ring geometry above stays in
    normalised units. Defaults to a 225 mm board, mid of the usual range.

    ``figure`` is one of :data:`FIGURES` and reaches only the fibre tangents
    (:func:`_fibre_tangents`) -- curl is a *reflection* effect, so it is invisible
    in the albedo and appears when the panel is shaded.

    ``cut`` is one of :data:`CUTS`, drawn by :data:`CUT_P` otherwise. It sets the
    ring geometry, and it gates the ray fleck: a quartersawn face is the only one
    that exposes ray tissue as sheets (:data:`RAY_FLECK_COVERAGE`).

    ``tangent`` is an (H, W, 3) unit fibre-tangent field, not the single grain
    angle this used to hand back: the whole point of deriving it from the
    distortion is that it varies per pixel. See :class:`BoardFields` for the
    other fields.
    """
    h, w = int(shape[0]), int(shape[1])
    x, y = grid_coords((h, w))
    aspect = h / max(w, 1)

    # --- This board's colour, in CIELAB -----------------------------------
    # One finish draw per board, applied to every colour the board carries.
    fin = _finish_delta(finish, rng)
    raw_lab = _board_lab(species, rng, variation, age)
    lab = _finish_lab(raw_lab, fin)
    # Earlywood is the species value; latewood is the same wood grown denser,
    # so it is a dL* (plus a touch of a*/b*) from it and not a free colour.
    light = lab_to_srgb(lab)
    dark = lab_to_srgb(_finish_lab(raw_lab + np.asarray(SPECIES[species]["late"]), fin))
    target = light.astype(np.float32)

    # grid_coords puts 1.0 of either coordinate at ``w`` pixels (y spans
    # [0, h/w) over h rows), so one unit is the same physical length on both
    # axes whatever the board's aspect ratio.
    px_per_unit = float(w)
    if px_per_mm is None:
        px_per_mm = px_per_unit / 225.0
    mm_per_unit = px_per_unit / float(px_per_mm)

    # Work in (along-grain, across-grain) coordinates.
    if along_x:
        u, v = x, y
        along_extent, across_extent = 1.0, aspect
    else:
        u, v = y, x
        along_extent, across_extent = aspect, 1.0

    ref = float(ref_across) if ref_across else across_extent

    # Small random tilt of the grain direction (+/- 6 degrees).
    tilt = float(rng.uniform(-0.105, 0.105))
    u, v = rotate((u, v), tilt, (along_extent * 0.5, across_extent * 0.5))

    # Domain warp the ring coordinates: the source of organic figure. Kept
    # low-amplitude -- real plain-sawn arcs are flattish, with the organic
    # character coming from the higher-frequency wiggle added to the phase.
    wu, wv = warp(
        (u, v),
        rng,
        amp=float(rng.uniform(0.01, 0.025)) * ref,
        freq=(float(rng.uniform(1.2, 2.5)), float(rng.uniform(1.2, 2.5))),
        octaves=3,
    )

    # Three cuts (:data:`CUTS`). "Cathedral" models the pith as an AXIS dipping
    # below the face along the grain: where the ring cones graze the board plane
    # their traces widen into the nested pointed crowns of a real plain-sawn face
    # (a 2D point-pith cannot produce that widening -- |grad r| is 1
    # everywhere). The other two keep the classic in-plane distant pith, near for
    # flatsawn's gentle arcs and far for quartersawn's near-straight parallel
    # rings. The draw happens either way, so forcing ``cut`` for a measurement
    # does not shift every other random number on the board.
    side = 1.0 if rng.random() < 0.5 else -1.0
    draw = rng.random()
    if cut is None:
        cut = (
            "cathedral"
            if draw < CUT_P["cathedral"]
            else (
                "flatsawn"
                if draw < CUT_P["cathedral"] + CUT_P["flatsawn"]
                else "quartersawn"
            )
        )
    if cut not in CUT_P:
        raise ValueError(f"unknown wood cut {cut!r}; choose from {CUTS}")
    if cut == "cathedral":
        pith_v = across_extent * float(rng.uniform(0.15, 0.85))
        apex_u = along_extent * float(rng.uniform(0.1, 0.9))
        d0 = float(rng.uniform(0.02, 0.12)) * ref
        # Shallow dip: crown ovals stretch ~1/slope along the grain, and
        # real cathedrals run 3-6x longer than wide.
        slope = float(rng.uniform(0.15, 0.4)) * side
        depth = d0 + slope * (wu - apex_u)
        r = np.sqrt((wv - pith_v) ** 2 + depth * depth)
    else:
        pith_dist = float(rng.uniform(*PITH_DIST[cut])) * ref
        pith_u = along_extent * float(rng.uniform(-0.6, 1.6))
        pith_v = across_extent * 0.5 + side * (pith_dist + across_extent * 0.5)
        r = np.sqrt((wu - pith_u) ** 2 + (wv - pith_v) ** 2)

    anatomy = ANATOMY[species]
    # ``rings`` is the count visible across the board, i.e. the *mean apparent*
    # ring spacing. A flatsawn face cuts the ring cones at a shallow angle, so
    # every ring trace on the face is wider than the radial ring width by
    # 1/sin(that angle) -- which is why 2 mm rings show as 10-25 mm bands. One
    # such scale factor per board, below, from the drawn series' own mean.
    n_rings = float(rng.uniform(*anatomy["rings"]))
    spacing = ref / n_rings

    # Optional knot: bend the rings radially around it and darken an ellipse.
    # Narrow plank strips rarely get one -- a knot squeezed into a thin strip
    # caps down to a smudge, and three planks each with a knot reads as
    # wallpaper repeat.
    knot = None
    if knot_p is None:
        knot_p = 0.4 if across_extent >= 0.3 * ref else 0.12
    if rng.random() < float(knot_p):
        # Use the warped coordinates so the knot's own rings wobble too.
        knot = _knot(
            wu, wv, rng, along_extent, across_extent, spacing, ref, px_per_unit
        )
        r = r + knot["pull"]

    # --- Ring widths: a grown series, not a ruler -------------------------
    # Grow a ring-width series in millimetres, cumulate it into ring boundary
    # radii, and read the ring coordinate off it by interpolation. That last
    # step is the one that matters: dividing by a constant spacing (what this
    # did) makes every ring the same width however much noise is added to the
    # phase, and evenly ruled rings are one of the two loudest "printed
    # laminate" tells there is.
    stats = RING_STATS[anatomy["pore_class"]]
    mean_mm = float(rng.uniform(*stats["mean_mm"]))
    sigma = (
        float(rng.uniform(*stats["sigma"])) if ring_sigma is None else float(ring_sigma)
    )
    r0 = float(r.min())
    span = float(r.max()) - r0
    # Grow generously: enough rings to cross the board's whole radius span with
    # room to spare, since a short series would leave a ringless patch in one
    # corner. The guard below covers the tail case.
    n_grow = int(np.ceil(span / max(spacing, 1e-6) * 1.6)) + 16
    widths_mm = _ring_widths(n_grow, rng, mean_mm=mean_mm, sigma=sigma)
    # Radial mm -> apparent units on the sawn face (see ``n_rings`` above): one
    # scale per board, set so the *mean* apparent spacing is the one the
    # species' ring count asks for. The variation about it is the series'.
    unit_per_mm = spacing / max(float(widths_mm.mean()), 1e-9)
    widths = (widths_mm * unit_per_mm).astype(np.float32)
    bounds = np.concatenate([[0.0], np.cumsum(widths, dtype=np.float64)])
    while bounds[-1] < span:  # vanishingly rare; keep the distribution
        widths = np.concatenate([widths, widths])
        widths_mm = np.concatenate([widths_mm, widths_mm])
        bounds = np.concatenate([[0.0], np.cumsum(widths, dtype=np.float64)])

    # Radius -> continuous ring coordinate: integer part is the ring index,
    # fraction the position within that ring.
    ring_u = np.interp(
        r - np.float32(r0), bounds, np.arange(bounds.size, dtype=np.float64)
    ).astype(np.float32)

    # Ring phase: low-frequency drift + fine high-frequency wiggle (the
    # "nervous" line quality of real grain). No width jitter here any more --
    # the widths are now grown rather than faked with noise on the phase.
    beta = float(rng.uniform(0.12, 0.3))
    phase_noise = fbm_at(
        u,
        v,
        rng,
        freq=(float(rng.uniform(1.5, 3.5)), float(rng.uniform(1.5, 4.0))),
        octaves=2,
        gain=0.45,
    )
    wiggle = fbm_at(
        u,
        v,
        rng,
        freq=(float(rng.uniform(4.0, 8.0)), float(rng.uniform(5.0, 10.0))),
        octaves=2,
        gain=0.5,
    )
    # The phase noise is a *lateral displacement of the ring boundary*, so it has
    # to be scaled by the local phase gradient. Added straight to the phase it
    # displaces the boundary by noise / |grad phase| instead, which diverges
    # exactly where the ring traces are widest on the face -- at a crown apex,
    # where |grad| goes to zero -- and turns the boundary into a fractal
    # coastline rather than the sweeping arc a real cathedral figure draws. This
    # factor is 1 where the rings sit at their mean spacing and falls away as
    # they stretch out, which is the same fixed physical wander either way.
    ring_grad = (
        np.abs(np.gradient(ring_u, axis=0)) + np.abs(np.gradient(ring_u, axis=1))
    ) * np.float32(px_per_unit)
    lateral = np.clip(ring_grad * np.float32(spacing), 0.0, 1.0).astype(np.float32)
    phase = ring_u + lateral * (
        np.float32(beta) * phase_noise + np.float32(rng.uniform(0.015, 0.04)) * wiggle
    )
    g = np.mod(phase, 1.0).astype(np.float32)

    # Per-ring randomness, piecewise constant across each ring: real trees
    # grow a different latewood every year. The index switches at the ring
    # boundary, which is where the profile steps anyway.
    ring_idx = np.floor(phase)
    idx = np.clip(ring_idx, 0.0, float(widths.size - 1)).astype(np.int32)
    # This pixel's own ring, radially (the physical width the profile keys off)
    # and as it appears on the face (what the pore band is measured against).
    ring_mm = np.asarray(widths_mm, dtype=np.float32)[idx]
    ring_units = widths[idx]
    ring_r1 = fbm_at(
        ring_idx * np.float32(0.37),
        np.zeros_like(ring_idx),
        rng,
        freq=(1.0, 1.0),
        octaves=1,
    )
    ring_r2 = fbm_at(
        ring_idx * np.float32(0.53) + np.float32(7.3),
        np.zeros_like(ring_idx),
        rng,
        freq=(1.0, 1.0),
        octaves=1,
    )

    # Rings read as colour, not corrugation. The profile is the asymmetric,
    # discontinuous sawtooth of :func:`_ring_sawtooth` -- an abrupt step at the
    # ring boundary rather than the symmetric soft ramp that gives a colour-ramp
    # lookup away -- with its strength fading in and out along the board like
    # real grain and varying ring to ring (ring_r1/ring_r2).
    # The step cannot be sharper than the pixel grid carries: floor the
    # transition at the ring phase's own per-pixel gradient. |d(phase)| is also
    # large wherever the phase has pixel-scale noise in it, which is what stops
    # the boundary contour dithering near a knot.
    dphase = np.abs(np.gradient(phase, axis=0)) + np.abs(np.gradient(phase, axis=1))
    density, late_w = _ring_sawtooth(
        g,
        ring_mm,
        anatomy["pore_class"],
        rng,
        ring_r1,
        trans_floor=(np.float32(1.6) * dphase).astype(np.float32),
    )
    fade = fbm_at(u, v, rng, freq=(2.5, 4.5), octaves=2)
    # Both terms are *variation about* full strength, so both are centred on 1.
    # Centred on 0.80 and 0.84 instead (what this did) they only ever subtract,
    # and their product took a third off every ring in the panel before the
    # species' own line weight was applied at all -- which is where pine's and
    # oak's ring definition went. The board still fades in and out along the
    # grain and still varies ring to ring; it just no longer does so downhill.
    amp = np.clip(1.00 + 0.32 * fade, 0.45, 1.35) * np.clip(
        1.00 + 0.34 * ring_r2, 0.45, 1.35
    )
    # Species line weight: pine's latewood is the hardest dark line here,
    # cherry's is barely there. Capped at 1, because the profile now spans the
    # whole earlywood -> latewood colour difference the species publishes.
    amp = (amp * np.float32(min(float(anatomy["line"]), 1.0))).astype(np.float32)
    profile = np.clip(density * amp, 0.0, 1.0).astype(np.float32)
    # The latewood *zone*, for the things that want a zone rather than a
    # density: the hue shift, the sheen and the little relief there is.
    band = np.clip(late_w * amp, 0.0, 1.0).astype(np.float32)

    # Fine grain streaks: fbm stretched 30-80x along the grain. The
    # across-grain frequency is physical now, not clamped to the grid --
    # ``fbm_at``'s ``max_freq`` band-limits it at the sampling Nyquist
    # instead, fading out whatever octave the grid cannot carry rather than
    # re-authoring it at the grid's own pitch (see :func:`_shade_fields` for
    # why that re-authoring was the mip-stability bug).
    f_along = float(rng.uniform(3.0, 7.0))
    stretch = float(rng.uniform(30.0, 80.0))
    f_across = f_along * stretch
    streaks = fbm_at(
        u,
        v,
        rng,
        freq=(f_along, f_across),
        octaves=3,
        gain=0.6,
        max_freq=0.5 * px_per_unit,
    )

    # Axial vessel streaks, sized from real anatomy at the board's physical
    # scale. Stretched along the *warped* coordinates so they follow the
    # figure. This is the one feature printed laminate physically cannot have.
    pores, pore_dl, pore_depth_mm, pore_trough_mm = _pore_streaks(
        wu,
        wv,
        g,
        rng,
        species,
        px_per_mm=float(px_per_mm),
        mm_per_unit=mm_per_unit,
        px_per_unit=px_per_unit,
        # Per-pixel, so the earlywood pore band keeps its width when the ring is
        # wide -- a wide ring gets more latewood, not a wider band of pores.
        spacing=ring_units,
        along_axis=1 if along_x else 0,
    )

    # Ray fleck, on a quartersawn face only: broad sheets of ray tissue lying in
    # the cut. Oriented along the board's own fibre angle, which is available
    # here without touching ``rng`` (:func:`_fibre_frame`), so the tangents can
    # stay the last random draw on the board.
    phi_fibre, _ = _fibre_frame(u, v, wu, wv, along_x=along_x, px_per_unit=px_per_unit)
    if cut == "quartersawn":
        fleck = _ray_fleck(
            (h, w),
            rng,
            species,
            phi_fibre,
            tilt=tilt,
            along_x=along_x,
            px_per_mm=float(px_per_mm),
        )
    else:
        zeros = np.zeros((h, w), dtype=np.float32)
        fleck = {"mask": zeros, "dl": zeros}

    # Broad tonal drift so the board is not uniformly coloured.
    drift = fbm_at(u, v, rng, freq=(1.5, 2.0), octaves=3) * np.float32(0.055)

    colour = light[None, None, :] + (dark - light)[None, None, :] * profile[..., None]
    colour = colour * (1.0 + streaks[..., None] * np.float32(0.12) + drift[..., None])
    # Within-ring hue drift: earlywood lighter/yellower, latewood browner.
    early_w = (1.0 - late_w).astype(np.float32)
    e_shift = np.asarray(EARLY_SHIFT[species], dtype=np.float32)
    l_shift = np.asarray(LATE_SHIFT[species], dtype=np.float32)
    colour = colour * (
        1.0
        + early_w[..., None] * e_shift[None, None, :]
        + band[..., None] * l_shift[None, None, :]
    )
    # Open pores hold pooled finish: a lightness drop of dL* 15-30.
    colour = _darken_lstar(colour, pore_dl)
    # Ray fleck carries only a small dL* -- and it is *signed*, mostly pale but
    # not always. The flash is in the tangent rotation below, not here: a fleck
    # drawn with a pore's contrast reads as chalk. Applied before
    # :func:`_recentre`, since at 10-25% coverage a one-sided dL* would otherwise
    # move the board's whole mean colour off the species value.
    colour = _darken_lstar(colour, -fleck["dl"])
    # Slow oxidation/UV drift: aged timber warms and darkens towards one
    # side rather than sitting flat in tone corner to corner.
    a_ox = float(rng.uniform(0.0, 2.0 * np.pi))
    ox = normalize01(u * np.float32(np.cos(a_ox)) + v * np.float32(np.sin(a_ox)))
    ox_vec = np.asarray([-0.010, -0.032, -0.058], dtype=np.float32)
    colour = colour * (1.0 + ox[..., None] * ox_vec[None, None, :])

    # Everything above is variation about the board's colour, so pin the mean
    # back onto it. Sapwood and knots go on *after* this, because both are
    # departures from the heartwood the species value describes rather than
    # variation within it.
    colour = _recentre(colour, target)

    sap = SPECIES[species]["sap"]
    if sap_p is None:
        sap_p = float(SPECIES[species]["sap_p"])
    if sap is not None and float(sap_p) > 0.0 and rng.random() < float(sap_p):
        # Sapwood is the outer, still-living wood: it never took the heartwood
        # extractives, so on cherry and walnut it is nearly white against a dark
        # heart. Boards are normally cut to exclude it, so a band along one edge
        # is an occasional draw -- and a strong authenticity cue when it lands.
        sap_srgb = lab_to_srgb(_finish_lab(np.asarray(sap, dtype=np.float64), fin))
        # It is a band along ONE EDGE, so it is capped to the outer 10-35% of the
        # board's width (:data:`SAPWOOD_COVERAGE`). Uncapped it covers a large
        # central area and stops reading as timber at all -- a pale band on an
        # edge is sapwood, a pale wash across the middle is a blown highlight.
        frac = float(rng.uniform(*SAPWOOD_COVERAGE)) * across_extent
        near = v if rng.random() < 0.5 else (across_extent - v)
        # The heart/sap boundary is fairly sharp in real timber but never a
        # step: a few millimetres of wander (it follows the rings, so it is not
        # a ruled line) and a few millimetres of feather. Both in mm, so they
        # do not scale with the board -- and the feather is floored at a couple
        # of pixels so it survives a small render.
        wave = float(rng.uniform(*SAPWOOD_WAVE_MM)) / mm_per_unit
        feather = max(
            float(rng.uniform(*SAPWOOD_FEATHER_MM)) / mm_per_unit, 2.0 / px_per_unit
        )
        edge = near + np.float32(wave) * fbm_at(
            u, v, rng, freq=(float(rng.uniform(1.5, 3.5)), 0.4), octaves=2
        )
        t = np.clip((np.float32(frac) - edge) / np.float32(feather) + 0.5, 0.0, 1.0)
        sap_w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        # Multiplicative, so the grain, pores and rings carry on through the
        # band instead of it reading as a pale stripe painted over them: it is a
        # colour change in the same wood, not a different board.
        gain = (sap_srgb / np.maximum(target, np.float32(1e-3))).astype(np.float32)
        colour = colour * (1.0 + sap_w[..., None] * (gain[None, None, :] - 1.0))

    # Sawn wood is nearly flat: rings are almost entirely a colour effect,
    # with the proud latewood, the fine streaks and the pore troughs carrying
    # what little relief there is. Height is carried directly in millimetres,
    # with no ``normalize01`` anywhere in the pipeline -- see
    # :data:`LATEWOOD_RELIEF_MM`, :data:`STREAK_RELIEF_MM` and
    # :data:`WAVINESS_MM` for the amplitudes below, and :func:`_pore_streaks`
    # for ``pore_depth_mm``, already in millimetres.
    height = (
        band * np.float32(LATEWOOD_RELIEF_MM)
        + streaks * np.float32(STREAK_RELIEF_MM)
        - pore_depth_mm
        + fbm_at(u, v, rng, freq=(1.5, 2.0), octaves=2) * np.float32(WAVINESS_MM)
    )

    if knot is not None:
        # Knots are a colour of their own (dark red-brown), not just a
        # shadow: blend towards it rather than multiplying down to black.
        knot_rgb = (dark * np.float32(0.55))[None, None, :]
        k = np.clip(knot["dark"], 0.0, 1.0)[..., None]
        colour = colour * (1.0 - k * np.float32(0.85)) + knot_rgb * k * np.float32(0.85)
        height = height + knot["bump"]

    # Fibre tangents next-to-last (the film build below is the true last
    # draw), so that every draw above keeps the sequence it had before this
    # layer existed.
    tangent = _fibre_tangents(
        u,
        v,
        wu,
        wv,
        rng,
        tilt=tilt,
        along_x=along_x,
        px_per_unit=px_per_unit,
        mm_per_unit=mm_per_unit,
        figure=figure,
        knot=knot,
    )
    # The ray axis is its own field now: crosswise to the fibre, no dip, gated
    # by the same mask that used to rotate the fibre tangent in place. Built
    # from ``phi_fibre`` above, which is why that stays available even off the
    # quartersawn cut -- ``ray_weight`` is simply zero there.
    ray_tangent = _ray_tangents(phi_fibre, tilt=tilt, along_x=along_x)

    # Coat geometry: a real film self-levels, so its own surface is the
    # substrate blurred over the film's own levelling length and then capped
    # to how much film there actually is -- a pore pools finish up to
    # ``film_mm`` deep and no deeper. Drawn last, after the tangents (though
    # the film build itself is a draw too, so it is the true last one, not
    # the tangents -- see :func:`_fibre_frame`), so every earlier draw keeps
    # the sequence it had before this layer existed. ``coat_lift`` is a
    # *lift*, not a second absolute surface: ``mode="edge"`` blurs the
    # substrate against its own edges rather than its opposite side, since a
    # board face is not tileable, and the blur is skipped entirely where
    # there is no film to level.
    film_mm = float(rng.uniform(*FINISHES[finish]["film_um"])) / 1000.0
    if film_mm > 0.0:
        coat_lift = np.clip(
            gaussian_blur(height, float(COAT_LEVEL_MM * px_per_mm), mode="edge")
            - height,
            0.0,
            film_mm,
        ).astype(np.float32)
    else:
        coat_lift = np.zeros_like(height)

    # Latewood is denser and takes a polish; the latewood zone is a broad band
    # rather than a line, so the coefficients are the ~2:1 sheen ratio between
    # the two zones, not a spike on top of a base. Open pores break the sheen
    # -- but only the part of the trough the film has NOT filled: a pore the
    # coat has pooled level with supports a continuous film across it.
    knot_dark = (
        np.clip(knot["dark"], 0.0, 1.0) if knot is not None else np.zeros_like(band)
    )
    # Off the *trough* (the vessel's own physical depth where one is present),
    # not the mask-weighted ``pore_depth_mm`` -- the fraction of a vessel left
    # open under the film must not depend on how much of a texel it covers.
    has_pore = pore_trough_mm > 0.0
    open_frac = np.where(
        has_pore,
        np.clip(pore_trough_mm - np.float32(film_mm), 0.0, None)
        / np.where(has_pore, pore_trough_mm, np.float32(1.0)),
        np.float32(0.0),
    ).astype(np.float32)
    coat_gloss = (
        (0.45 + 0.5 * band)
        * (1.0 - 0.55 * pores * open_frac)
        # Exposed ray tissue is a smooth flat sheet of thin-walled cells and
        # takes a better polish than the fibre around it.
        * (1.0 + 0.25 * np.clip(fleck["mask"], 0.0, 1.0))
        * (1.0 - 0.5 * knot_dark)
    ).astype(np.float32)
    # The fibre lobe's own lustre carries no fleck factor: the ray lobe now
    # carries its own gain instead of a gloss multiplier.
    fibre_lustre = (
        (0.45 + 0.5 * band) * (1.0 - 0.55 * pores) * (1.0 - 0.5 * knot_dark)
    ).astype(np.float32)

    return BoardFields(
        albedo=np.clip(colour, 0.0, 1.0).astype(np.float32),
        height=height.astype(np.float32),
        coat_lift=coat_lift,
        tangent=tangent,
        ray_tangent=ray_tangent,
        ray_weight=fleck["mask"],
        coat_gloss=coat_gloss,
        fibre_lustre=fibre_lustre,
    )


def _knot(
    u: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
    along_extent: float,
    across_extent: float,
    spacing: float,
    ref_across: float,
    px_per_unit: float,
) -> dict:
    """Radial ring distortion plus a dark elliptical knot with its own rings."""
    ku = along_extent * float(rng.uniform(0.15, 0.85))
    kv = across_extent * float(rng.uniform(0.15, 0.85))
    radius = float(rng.uniform(0.4, 0.9)) * ref_across * 0.24
    # Never let a knot outgrow the board it sits on (plank strips are narrow).
    radius = min(radius, 0.38 * min(along_extent, across_extent))
    # Knots are slightly elongated along the grain.
    elong = float(rng.uniform(1.2, 2.2))
    du = (u - ku) / elong
    dv = v - kv
    d = np.sqrt(du * du + dv * dv)
    # Wobble the radial distance so the knot's own rings are irregular
    # rather than compass-drawn circles. The frequency is clamped against the
    # *resolution*, not a constant: a small knot asks for 60+ cycles per unit,
    # which is a handful of pixels, and this wobble goes into ``pull``, i.e.
    # into the ring phase. Pixel-scale noise in the ring phase makes the ring
    # boundary -- a step, and a piecewise-constant per-ring random -- flicker
    # between one ring and the next, which reads as a crawling stipple along
    # every ring near the knot.
    wobble_freq = min(9.0 / max(radius, 1e-6), 0.08 * px_per_unit)
    d = d * (
        1.0 + np.float32(0.07) * fbm_at(u, v, rng, freq=(wobble_freq,) * 2, octaves=2)
    )

    falloff = np.exp(-((d / np.float32(radius * 2.2)) ** 2)).astype(np.float32)
    pull = falloff * np.float32(spacing * rng.uniform(1.5, 4.0))

    # A real knot is a small solid dark branch stub, circled by its own
    # rings out to a distinct dark rim -- flat darkening over the whole
    # influence ellipse reads as a scorch mark, not a knot.
    stub = smoothstep(radius * 0.58, radius * 0.4, d)
    centre = smoothstep(radius * 0.4, radius * 0.12, d)
    ring_zone = smoothstep(radius * 1.02, radius * 0.55, d)
    rim = np.exp(-(((d - np.float32(radius)) / np.float32(radius * 0.09)) ** 2))
    inner_g = np.mod(d / np.float32(radius * rng.uniform(0.16, 0.28)), 1.0)
    inner = smoothstep(0.5, 0.8, inner_g) * (1.0 - smoothstep(0.92, 1.0, inner_g))
    dark = (
        stub * np.float32(0.7)
        + centre * np.float32(0.25)
        + ring_zone * inner * np.float32(0.4)
        + ring_zone * np.float32(0.06)
        + rim * np.float32(0.45)
    )
    bump = stub * np.float32(rng.uniform(-0.25, -0.08))
    return {
        "pull": pull,
        "dark": dark.astype(np.float32),
        "bump": bump.astype(np.float32),
        # Fibre dip is concentrated far tighter than the ring pull -- 10-30 mm,
        # i.e. about one knot radius, against the 2.2 radii the pull reaches.
        "dip": np.exp(-((d / np.float32(radius * 0.9)) ** 2)).astype(np.float32),
    }


class _BoardColour(TypedDict):
    """Keyword colour params shared by :func:`_board_fields` and :func:`_planks`."""

    finish: str
    variation: float
    age: float
    sap_p: float | None
    knot_p: float | None
    ring_sigma: float | None
    figure: str
    cut: str | None


def generate(
    shape: tuple[int, int],
    rng: np.random.Generator,
    variant: str = "board",
    **params,
) -> np.ndarray:
    """Generate a wood texture as float32 (H, W, 3) sRGB in [0, 1].

    ``mm_across`` is the physical width of the board in millimetres (default: a
    real board's 150-300 mm). It exists so the vessel pore layer can be sized
    in microns; the ring geometry is still in normalised units.

    Colour params, all optional:

    * ``species`` -- a key of :data:`SPECIES`; random otherwise.
    * ``finish`` -- a key of :data:`FINISHES`, drawn by share of real boards
      otherwise. One class per panel (a floor is finished in place), with its
      magnitude re-drawn per board.
    * ``colour_variation`` -- scales the per-board Lab spread and the ageing
      draw. 0 renders the species' nominal published colour.
    * ``age`` -- 0 (fresh) to 1 (the species' aged colour), for the species that
      publish one; drawn per *panel* otherwise, since boards laid together age
      together.
    * ``sapwood``, ``knots`` -- probability overrides, 0 to disable.
    * ``ring_sigma`` -- standard deviation of the raw log ring-width series, i.e.
      how variable the ring widths are (:data:`RING_STATS`, 0.20-0.35); drawn
      per species otherwise. Mean sensitivity follows from it, so this is the
      single "how dramatic is this board" control.
    * ``figure`` -- one of :data:`FIGURES`. Drawn per *panel* otherwise, and
      rarely: curl belongs to the log (:data:`FIGURE_P`). It is a reflection
      effect only, so it does not touch the albedo.
    * ``cut`` -- one of :data:`CUTS`, drawn by :data:`CUT_P` otherwise. Forcing
      ``"quartersawn"`` is how to ask for ray fleck
      (:data:`RAY_FLECK_COVERAGE`), which no other cut shows.
    * ``light_dir`` -- direction towards the light, for asking the same panel
      what it looks like from a second angle. Chatoyance is by definition a
      thing that only shows up when this moves.
    * ``normal_strength`` -- exaggeration factor on the board's *physical*
      slopes: 1.0 renders the true slope the millimetre-scale height
      field describes, and the 1.1-2.2 default draw is kept because a texture
      map this flat needs a little exaggeration to read as relief at all, not
      because the underlying slopes are wrong.
    """
    h, w = int(shape[0]), int(shape[1])
    species = str(params.get("species") or rng.choice(sorted(SPECIES)))
    if species not in SPECIES:
        raise ValueError(
            f"unknown wood species {species!r}; choose from {sorted(SPECIES)}"
        )
    finish = params.get("finish")
    finish = _pick_finish(rng) if finish is None else str(finish)
    if finish not in FINISHES:
        raise ValueError(
            f"unknown wood finish {finish!r}; choose from {sorted(FINISHES)}"
        )
    # One ageing draw for the whole panel: cherry's fresh and aged values are 16
    # L* apart, so drawing this per board would put a 12-month board next to a
    # day-old one in the same floor. The per-board spread stays per board.
    age = params.get("age")
    age = float(rng.random()) ** 0.8 if age is None else float(age)
    # Figure belongs to the log, so it is one draw for the whole panel: five
    # planks off one curly maple board are all curly. An occasional draw, and
    # only for the species the trade names it after (:data:`FIGURE_P`).
    figure = params.get("figure")
    figure = _pick_figure(species, rng) if figure is None else str(figure)
    if figure not in FIGURES:
        raise ValueError(f"unknown wood figure {figure!r}; choose from {FIGURES}")
    cut = params.get("cut")
    cut = None if cut is None else str(cut)
    if cut is not None and cut not in CUT_P:
        raise ValueError(f"unknown wood cut {cut!r}; choose from {CUTS}")
    colour: _BoardColour = dict(
        finish=finish,
        variation=float(params.get("colour_variation", 1.0)),
        age=age,
        sap_p=params.get("sapwood"),
        knot_p=params.get("knots"),
        ring_sigma=params.get("ring_sigma"),
        figure=figure,
        cut=cut,
    )
    along_x = w >= h

    mm_across = float(params.get("mm_across", rng.uniform(150.0, 300.0)))
    if mm_across <= 0.0:
        raise ValueError(f"mm_across must be positive, got {mm_across}")
    px_per_mm = w / mm_across

    if variant == "board":
        fields = _board_fields(
            (h, w), rng, species, along_x=along_x, px_per_mm=px_per_mm, **colour
        )
    elif variant == "planks":
        fields = _planks(
            (h, w), rng, species, along_x=along_x, px_per_mm=px_per_mm, **colour
        )
    else:
        raise ValueError(f"unknown wood variant {variant!r}; choose from {VARIANTS}")

    light_dir = (-0.5, -0.55, 0.78)
    if "light_dir" in params:
        lx, ly, lz = (float(c) for c in params["light_dir"])
        light_dir = (lx, ly, lz)
    return _shade_fields(
        fields,
        finish=finish,
        rng=rng,
        params=params,
        px_per_mm=px_per_mm,
        light_dir=light_dir,
    )


# Recess depth, in millimetres, at which the ``cavity`` share of the shading
# darkening is fully lost, forwarded to :func:`..core.shading.shade` as
# ``cavity_depth``. UNVERIFIED: the depth of a large earlywood vessel trough
# (oak/ash run to 0.20 mm, :data:`ANATOMY`'s ``ew_depth_um``) -- a recess
# deeper than this (a plank gap, :data:`PLANK_GAP_DEPTH_MM`) shadows no more
# than one already does, rather than washing out every shallower recess'
# share the way normalising against the field's own single deepest feature
# used to.
CAVITY_DEPTH_MM = 0.15


def _shade_fields(
    fields: BoardFields,
    *,
    finish: str,
    rng: np.random.Generator,
    params: dict,
    px_per_mm: float,
    light_dir: tuple[float, float, float],
) -> np.ndarray:
    """Light a board's anatomy: the seam between :class:`BoardFields` and :func:`..core.shading.shade`.

    Everything :func:`generate` used to do once the fields existed lives here
    now: the two specular lobes, the ring-correlated lustre, and the ``shade``
    call itself -- ``height`` arrives in millimetres and is passed straight
    through, unnormalised, with ``px_per_mm`` converting it to the
    physical ``height_spacing`` :func:`..core.shading.shade` needs. This is
    the place a test replaces ``fields.albedo`` with a flat colour to check
    that figure still lives in the reflection once the pigment cannot carry it
    (:data:`~texture_generators.materials.wood`).

    Shades in **linear light**: ``fields.albedo`` is display-referred
    sRGB, decoded before ``shade`` and the clipped result re-encoded after --
    Lambert shading and lobe addition are linear-light operations, and
    compositing them directly in sRGB is a gamma error (see
    :func:`~texture_generators.materials.wood`, and the module docstring's
    linear-light paragraph).
    """
    albedo = fields.albedo
    albedo_linear = srgb_to_linear(albedo)
    height = fields.height
    lobes = _specular_lobes(finish, rng)
    # Ring-correlated sheen: the dense latewood lines catch the light while
    # pores and knots break it, so the rings live in the reflection too. It
    # modulates *both* lobes -- an open pore breaks the film and the fibre under
    # it alike -- but the fibre weight is a table value, so that one is
    # normalised to its own mean and the table number stays the mean weight.
    spec_base = float(params.get("specular", lobes["surface_weight"]))
    lustre = np.clip(
        fields.fibre_lustre / max(float(fields.fibre_lustre.mean()), 1e-6), 0.0, 2.0
    ).astype(np.float32)

    lit = shade(
        albedo_linear,
        height,
        light_dir=light_dir,
        specular=(spec_base * fields.coat_gloss).astype(np.float32),
        shininess=float(params.get("shininess", lobes["shininess"])),
        normal_strength=float(params.get("normal_strength", rng.uniform(1.1, 2.2))),
        height_spacing=1.0 / px_per_mm,
        coat_height=(
            None if not fields.coat_lift.any() else fields.height + fields.coat_lift
        ),
        fibre_ior=float(FINISHES[finish]["ior"]),
        # Only the in-plane part: this lobe is the finish film reflecting off the
        # surface, and it is the *fibre* lobe below that cares about the dip.
        aniso_dir=fields.tangent[..., :2],
        aniso=lobes["aniso"],
        fibre_tangent=fields.tangent,
        fibre=(np.float32(lobes["fibre_weight"]) * lustre),
        fibre_exponent=lobes["fibre_exponent"],
        fibre_colour=_fibre_colour(albedo_linear, finish),
        # The ray population is its own axis mixed in by weight, not a
        # rotation of the fibre tangent: a fleck's fibres run crosswise to the
        # grain, they do not run diagonally to it.
        ray_tangent=fields.ray_tangent,
        ray_weight=fields.ray_weight,
        ray_gain=RAY_LOBE_GAIN,
        ambient=0.62,
        cavity=0.12,
        cavity_depth=CAVITY_DEPTH_MM,
        # The albedo is a measured colour, so the lighting must average to 1 or
        # the render is a species' colour times an arbitrary constant. Without
        # this every board came out 20-25% dark in every channel.
        normalise=True,
        # ...and for the same reason the specular has to come *out* of the
        # diffuse rather than be piled on top of it: the fibre lobe runs to 0.2,
        # which is 50 levels of pure addition on a measured colour.
        conserve_energy=True,
    )
    return linear_to_srgb(lit)


def _pick_figure(species: str, rng: np.random.Generator) -> str:
    """Draw a figure class for one panel: usually ``"plain"``."""
    table = FIGURE_P.get(species)
    if not table:
        return "plain"
    draw = float(rng.random())
    acc = 0.0
    for name in sorted(table):
        acc += float(table[name])
        if draw < acc:
            return name
    return "plain"


def _specular_lobes(finish: str, rng: np.random.Generator) -> dict:
    """Draw this panel's two specular lobes from its finish (:data:`SPECULAR`).

    The roughness pair arrives as GGX-style alphas and the renderer wants a
    Blinn-Phong exponent, so ``n = 2 / alpha**2 - 2`` -- the standard equivalence,
    from matching the two lobes' widths at half maximum.

    The lobe *weight* does not follow from ``F0`` alone, because
    :func:`..core.shading.shade` is not energy-normalised: the same 4% reflectance
    spread over a broad lobe and concentrated into a sharp one has to peak at very
    different heights. ``F0 * (1 + 0.25 * sqrt(n))`` is a fit to this renderer
    (**UNVERIFIED**), and it lands raw wood at 0.06 and gloss varnish at 0.20 --
    which is the 0.06-0.18 range this generator was hand-tuned to before the
    finish drove it, now ordered by finish instead of drawn at random.
    """
    spec = SPECULAR[finish]
    alpha = 0.5 * (
        float(rng.uniform(*spec["alpha_along"]))
        + float(rng.uniform(*spec["alpha_across"]))
    )
    shininess = float(np.clip(2.0 / max(alpha * alpha, 1e-6) - 2.0, 4.0, 400.0))
    return {
        "shininess": shininess,
        "aniso": float(rng.uniform(*spec["aniso"])),
        "surface_weight": float(
            np.clip(FINISH_F0 * (1.0 + 0.25 * np.sqrt(shininess)), 0.04, 0.35)
        ),
        "fibre_exponent": float(rng.uniform(*spec["fibre_exponent"])),
        "fibre_weight": float(rng.uniform(*spec["fibre_weight"])),
    }


def _fibre_colour(albedo_linear: np.ndarray, finish: str) -> np.ndarray:
    """Colour of the fibre lobe: ``sqrt(albedo_linear)``, tinted by the finish.

    Marschner's separately-coloured specular component. This light did not bounce
    off the surface -- it went into the wood, off a fibre and back out, so it
    carries the pigment but only once rather than to saturation: white would read
    as a dusty film lying on top, fully-saturated as a coloured light. The square
    root is the single-pass-through-pigment argument, and it is a
    **linear-light** argument -- it only holds when ``albedo_linear`` is the
    wood's linear reflectance, not its sRGB-encoded display value. Practical
    guidance rather than a derived exponent (**UNVERIFIED**); the finish's own
    ``fibre_tint`` (:data:`FINISHES`) adds the film's cast on top.
    """
    tint = np.asarray(FINISHES[finish]["fibre_tint"], dtype=np.float32)
    # Unit luminance, so this tints the lobe without changing its strength.
    tint = tint / np.float32(
        max(float(0.2126 * tint[0] + 0.7152 * tint[1] + 0.0722 * tint[2]), 1e-6)
    )
    return np.clip(np.sqrt(np.clip(albedo_linear, 0.0, 1.0)) * tint, 0.0, 1.0).astype(
        np.float32
    )


# Plank gap and bevel, in millimetres -- the seam between adjacent boards in
# a multi-plank panel. UNVERIFIED (woodworking practice, not a measurement):
#
#   PLANK_GAP_MM        the fitted gap, drawn per panel: a tight, deliberate
#                        joint, not a construction tolerance.
#   PLANK_GAP_DEPTH_MM  the groove's depth (the old -0.8 literal, now a
#                        physical unit rather than a bare number).
#   PLANK_BEVEL_MM      an arrissed edge either side of the gap: the eased
#                        corner every real plank edge carries, sloping the
#                        groove's depth back up to the board face rather than
#                        stopping it in a hard step.
PLANK_GAP_MM = (0.5, 1.5)
PLANK_GAP_DEPTH_MM = 0.8
PLANK_BEVEL_MM = 1.0


def _planks(
    shape: tuple[int, int],
    rng: np.random.Generator,
    species: str,
    *,
    finish: str = "none",
    variation: float = 1.0,
    age: float = 0.0,
    sap_p: float | None = None,
    knot_p: float | None = None,
    ring_sigma: float | None = None,
    figure: str = "plain",
    cut: str | None = None,
    along_x: bool = True,
    px_per_mm: float | None = None,
) -> BoardFields:
    """Split the canvas across the grain into 3-6 boards with gap lines.

    Each strip is a separate board, so each gets its own Lab offset, its own
    finish magnitude and its own sapwood draw -- three random numbers and a
    couple more, for the single loudest "printed" tell there is at this scale.
    ``age`` is shared across the strips, being a property of the panel.

    Per-strip fields are pasted into panel-sized arrays and returned as one
    :class:`BoardFields`; the gap-line edits below darken ``albedo``, cut
    ``height`` (the coat lift carries through unedited, by construction --
    ``height + coat_lift`` follows the groove because ``height`` now does)
    and scale both gloss fields as they always have, and do not touch either
    tangent field.
    """
    h, w = int(shape[0]), int(shape[1])
    if px_per_mm is None:
        px_per_mm = float(w) / 225.0
    albedo = np.zeros((h, w, 3), dtype=np.float32)
    height = np.zeros((h, w), dtype=np.float32)
    coat_lift = np.zeros((h, w), dtype=np.float32)
    coat_gloss = np.ones((h, w), dtype=np.float32)
    fibre_lustre = np.ones((h, w), dtype=np.float32)

    # Planks are cut across the grain-perpendicular axis.
    n = int(rng.integers(3, 7))
    span = h if along_x else w
    n = max(1, min(n, max(1, span // 12)))
    edges = _split_edges(span, n, rng)
    # Ring spacing and knots are sized against the whole panel, not the strip.
    panel_ref = (h / max(w, 1)) if along_x else 1.0

    # Per-pixel fibre and ray tangents so each plank's chatoyance follows its
    # own grain rather than whichever plank was built last. Both default to
    # the +x axis where no strip sets them, like the fibre tangent always has;
    # ray_weight defaults to zero, i.e. no fleck, which is also the default
    # off a quartersawn face.
    tangent = np.zeros((h, w, 3), dtype=np.float32)
    tangent[..., 0] = 1.0
    ray_tangent = np.zeros((h, w, 3), dtype=np.float32)
    ray_tangent[..., 0] = 1.0
    ray_weight = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        sub_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        sub_shape = (hi - lo, w) if along_x else (h, hi - lo)
        fields = _board_fields(
            sub_shape,
            sub_rng,
            species,
            finish=finish,
            variation=variation,
            age=age,
            sap_p=sap_p,
            knot_p=knot_p,
            ring_sigma=ring_sigma,
            figure=figure,
            cut=cut,
            along_x=along_x,
            ref_across=panel_ref,
            px_per_mm=px_per_mm,
        )
        sl = (slice(lo, hi), slice(None)) if along_x else (slice(None), slice(lo, hi))
        albedo[sl] = fields.albedo
        height[sl] = fields.height
        coat_lift[sl] = fields.coat_lift
        coat_gloss[sl] = fields.coat_gloss
        fibre_lustre[sl] = fields.fibre_lustre
        tangent[sl] = fields.tangent
        ray_tangent[sl] = fields.ray_tangent
        ray_weight[sl] = fields.ray_weight

    # Dark gap lines with an arrissed bevel on each plank edge, all in
    # physical millimetres rather than a fixed pixel count -- the gap and
    # bevel keep the same *size on the board* whatever resolution the panel
    # is rendered at, just as the rest of the relief does (module docstring).
    # One rng call for the gap width, exactly where the old pixel draw was.
    gap_mm = float(rng.uniform(*PLANK_GAP_MM))
    gap_px = max(1, round(gap_mm * px_per_mm))
    bevel_px = max(1, round(PLANK_BEVEL_MM * px_per_mm))
    for e in edges[1:-1]:
        lo = max(0, e - gap_px)
        hi = min(span, e)
        sl = (slice(lo, hi), slice(None)) if along_x else (slice(None), slice(lo, hi))
        albedo[sl] *= np.float32(0.28)
        coat_gloss[sl] *= np.float32(0.2)
        fibre_lustre[sl] *= np.float32(0.2)
        height[sl] -= np.float32(PLANK_GAP_DEPTH_MM)
        # An arrissed edge, not a raised lip: a linear ramp from near the
        # groove's own depth back up to the board face, ``bevel_px`` rows
        # outward on each side of the gap.
        for k in range(bevel_px):
            depth = np.float32(PLANK_GAP_DEPTH_MM) * (1.0 - (k + 1) / (bevel_px + 1))
            r_lo, r_hi = lo - 1 - k, hi + k
            if r_lo >= 0:
                sl_lo = (
                    (slice(r_lo, r_lo + 1), slice(None))
                    if along_x
                    else (slice(None), slice(r_lo, r_lo + 1))
                )
                height[sl_lo] -= depth
            if r_hi < span:
                sl_hi = (
                    (slice(r_hi, r_hi + 1), slice(None))
                    if along_x
                    else (slice(None), slice(r_hi, r_hi + 1))
                )
                height[sl_hi] -= depth

    return BoardFields(
        albedo=albedo,
        height=height,
        coat_lift=coat_lift,
        tangent=tangent,
        ray_tangent=ray_tangent,
        ray_weight=ray_weight,
        coat_gloss=coat_gloss,
        fibre_lustre=fibre_lustre,
    )


def _split_edges(span: int, n: int, rng: np.random.Generator) -> list[int]:
    """Random-ish plank boundaries covering ``span`` in ``n`` pieces."""
    weights = rng.uniform(0.75, 1.25, size=n)
    weights = weights / weights.sum()
    edges = [0]
    acc = 0.0
    for i in range(n - 1):
        acc += float(weights[i])
        edges.append(max(edges[-1] + 4, min(span - 4 * (n - i - 1), int(acc * span))))
    edges.append(span)
    return edges
