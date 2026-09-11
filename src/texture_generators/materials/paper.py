"""Paper textures: six stocks, each built as an actual fibre stack.

Paper is not a tinted ground with lines drawn on it -- it is a felt five to
fifteen fibre-layers deep, seen through its own scattering. Three physical
facts drive the whole design here, and getting any of them wrong is what makes
procedural paper look like scratched card:

1. **Coverage.** Total fibre length per unit area is grammage / coarseness, so
   80 g/m^2 office paper carries ~440 mm of fibre per mm^2 -- about 13 layers.
   A few hundred marks on an empty ground is coverage well under 1: isolated
   strokes, which the eye reads as scratches.
2. **Sub-pixel width.** A 30 um fibre is 0.15-1.0 px wide at any sensible
   capture scale, so it can only ever contribute *partial* coverage. An opaque
   1 px line is several times too wide and far too dark.
3. **Translucency.** Light enters the sheet, spreads sideways over a
   point-spread radius, and leaves. The radius is per stock (``psf_um`` in
   :data:`STOCKS`), 25-250 um across the six: 25 for ``coated``, 90 for
   ``white``, 130-150 for ``recycled``, ``laid`` and ``kraft``, and 250 for
   unfilled ``newsprint``. Height detail finer than that
   radius produces almost no diffuse shading, while the surface sheen keeps all
   of it. That is why the shading pass uses two normals -- a blurred one for
   diffuse, a sharp one for sheen -- and why formation, which is obvious when
   you hold a sheet to a window, is nearly invisible flat-lit: reflectance
   saturates with grammage, so a +/-8% mass swing is only +/-1-2% in reflected
   light.

So the pipeline is: deposit curved fibres in depth slabs, accumulate mass
additively (mass is extensive) *and* composite the slabs optically with
Beer-Lambert alpha-over (light is not), blurring each slab by its depth. The
mass field gives formation and relief; the composite gives colour. Everything
runs in linear light and is encoded to sRGB once at the end.

A fourth fact governs the *shape* of the height distribution, which a Gaussian
random field gets wrong:

4. **Paper's height histogram is not symmetric, and the sign of its asymmetry
   is a grade property.** At this sampling (25-40 um/px) what is resolved is
   *fibres*, not pores, and a fibre crown, a fibre crossing or a shive stands
   **above** the sheet plane -- so the upper tail is the heavy one and
   uncoated stock measures Ssk > 0, Sku > 3. Two mechanisms produce that here,
   both in :func:`render_sheet`: the surface is a *maximum* over the fibre
   stack rather than an average of it (:func:`_stack_top`), and the height in a
   column follows the fibre mass stacked there, whose marginal is
   gamma/Pearson-III rather than normal (:func:`_skew_warp`). Calendering
   then works the other way: the nip is carried by the tallest asperities, so
   it crushes peaks and leaves valleys (:func:`_calender_clip`), driving Ssk
   back down and, at gloss-calendering loads, negative.

Colour comes from CIELAB values measured on real stock rather than hand-picked
RGB, so whiteness is -b*, ageing is +b*, and kraft sits where kraft actually
sits.

Three structures sit on top of that shared pipeline, each switched on per stock
by a key in :data:`STOCKS`:

* **Felt** (``felt``). Below the resolvable-fibre scale -- fines, mechanical
  pulp, the weave seen through a coating -- there is nothing to gain from
  drawing fibres one at a time, so a LIC field (:mod:`..core.lic`) supplies that
  texture and drives mass, albedo and height together.
* **Wire marks** (``wire``). A laid mould presses laid and chain wires into the
  wet web, which thins it: watermark physics, so the marks belong in the
  *height* field and are only 1-3% in reflection.
* **Coating** (``coating``). A pigment coat fills the fibre relief instead of
  following it, so the same network comes out an order of magnitude smoother,
  crisper (a much shorter point spread) and glossier, with coat-weight mottle
  in the sheen.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

import numpy as np

from ..core.colour import lab_to_linear_rgb, linear_to_srgb
from ..core.fibres import cluster_points, count_for_coverage, deposit
from ..core.fields import normalize01, smoothstep
from ..core.lic import felt
from ..core.noise import fbm_at, grid_coords
from ..core.shading import gaussian_blur, shade_translucent
from ..core.spectral import band_field, matern_field, resize_periodic
from ..core.warp import warp
from ..core.worley import worley_at

VARIANTS = ["white", "kraft", "recycled", "newsprint", "laid", "coated"]

# Per-variant physical profile. Fibre dimensions are length-weighted averages
# from pulp-testing practice; coverage is grammage / (coarseness * width);
# psf_um is the lateral optical spread that causes optical dot gain, which is
# also the radius over which diffuse shading is smeared.
#
# Four keys describe structure the drawn fibre network cannot carry:
#   tooth_mm  centre of the roughness band that dominates the height field.
#   rough_w   its weights, as (tooth, floc, surface, micro). The blend is
#             renormalised to unit variance, so only the ratios matter -- they
#             set the *spectrum* of the surface and hence its RMS slope, while
#             sq_um sets the amplitude.
#   felt      LIC felt parameters (:func:`_felt_stack`), or None for no felt.
#             This is the sub-resolution fibre texture: a fines population at
#             20-120 um (``fines_mm``) plus, where the stock sets it, a second
#             longer one at fibre length (``fibre_mm``) standing in for grain
#             too faint to be worth depositing strand by strand. Neither draws
#             the resolvable fibres, which core.fibres deposits individually.
#   md        machine-direction strength, 0-1. A Fourdrinier sheet is drained
#             while it moves, so its flocs, its wet streaks and its tooth are
#             all elongated along the machine direction; a sheet made by hand
#             on a mould has no machine direction at all, and 0 removes every
#             one of those elongations together.
#   wire      laid-mould wire marks (:func:`_wire_marks`), or None.
#   coating   pigment-coating overrides (mottle), or None for uncoated.
#
# One key describes the *fibre*, not the sheet:
#   fibrillation  external-fibrillation index -- fibril projected area over
#               parent-fibre projected area, as a fibre analyser reports it. It
#               is a refining measure, so it also sets the share of ends that are
#               refining cuts (:func:`_cut_end_fraction`). Fibrils are 50 nm -
#               1 um wide, i.e. sub-pixel at every size rendered here, so this
#               buys an opacity fringe on the fibre edge rather than geometry;
#               see :func:`..core.fibres.deposit`. Index values are typical
#               instrument outputs and are UNVERIFIED.
#
# Two more set the *shape* of the height histogram rather than its spectrum, both
# measured on the roughness band at an ISO 16610-21 cut-off of lc = 1 mm (a
# roughness statistic without its cut-off is meaningless; see
# :func:`..core.shading.iso_highpass`):
#   stack_skew  strength of the two-slope asymmetry the fibre stack imposes on
#               the roughness band (:func:`_skew_warp`). Zero leaves the band
#               symmetric, which is what a pigment coating does -- the coat is a
#               film, not a fibre top.
#   asperity    signed *depth* of the sparse extreme-value population
#               (:func:`_asperities`), in the same units as ``rough_w``: positive
#               where protruding fibres and shives are the extremes, negative
#               where coating voids and sags are. It saturates, so this is a
#               height and not a variance weight.
#   calender_k  calendering load, as the strength of the one-sided softplus clip
#               in :func:`_calender_clip`: 0 uncalendered, ~0.45 machine-
#               finished, ~0.8 supercalendered or gloss-calendered, and 1 the
#               ceiling the operator stays monotone up to. It also
#               drives the gloss and the calender blackening, because all three
#               are consequences of the same local densification.


class FeltSpec(TypedDict):
    """Sub-resolution LIC felt parameters. See :func:`_felt_stack`."""

    fines_mm: float
    fibre_mm: float
    layers: int
    mass: float
    albedo: float
    height: float


class WireSpec(TypedDict):
    """Laid-mould wire-mark parameters. See :func:`_wire_marks`."""

    laid_pitch_mm: tuple[float, float]
    laid_um: tuple[float, float]
    chain_pitch_mm: tuple[float, float]
    chain_um: tuple[float, float]


class CoatingSpec(TypedDict):
    """Pigment-coating overrides applied on top of the base fibre network."""

    mottle_mm: tuple[float, float]
    gloss: float
    albedo: float


class Stock(TypedDict):
    """Per-variant physical profile. See the table comment above and :data:`STOCKS`."""

    mm_across: tuple[float, float]
    lab: tuple[float, float, float]
    lab_jitter: tuple[float, float, float]
    fibre_mm: float
    fibre_sigma: float
    width_um: float
    coverage: float
    kappa: tuple[float, float]
    persistence: float
    kinks: float
    tone_spread: float
    extinction: float
    floc_spacing_mm: float
    floc_spread_mm: float
    clustered: float
    fibrillation: float
    psf_um: float
    sheen: tuple[float, float]
    sheen_exp: tuple[float, float]
    sq_um: float
    cockle_um: tuple[float, float]
    roughness_deg: float
    calender_k: float
    stack_skew: float
    asperity: float
    crease_chance: float
    md: float
    tooth_mm: float
    rough_w: tuple[float, float, float, float]
    felt: FeltSpec | None
    wire: WireSpec | None
    coating: CoatingSpec | None
    # Only ``coated`` sets this; every other stock uses the default of 1.0
    # applied where it is read, in :func:`render_sheet`.
    crease_scale: NotRequired[float]


STOCKS: dict[str, Stock] = {
    # Uncoated woodfree copier stock: short hardwood-rich furnish, heavily
    # filled, calendered, optically brightened.
    "white": {
        "mm_across": (12.0, 22.0),
        "lab": (95.5, 0.5, -1.2),
        "lab_jitter": (1.0, 0.4, 1.4),
        "fibre_mm": 1.15,
        "fibre_sigma": 0.6,
        "width_um": 21.0,
        "coverage": 9.0,
        "kappa": (0.20, 0.40),
        "persistence": 1.4,
        "kinks": 1.5,
        "tone_spread": 1.6,
        "extinction": 0.55,
        "floc_spacing_mm": 1.4,
        "floc_spread_mm": 0.30,
        "clustered": 0.45,
        # Moderately refined printing/writing furnish.
        "fibrillation": 0.035,
        "psf_um": 90.0,
        "sheen": (0.036, 0.055),
        "sheen_exp": (12.0, 18.0),
        "sq_um": 5.7,
        "cockle_um": (22.0, 70.0),
        "roughness_deg": 18.0,
        "calender_k": 0.45,
        "stack_skew": 0.26,
        "asperity": 1.20,
        "crease_chance": 0.12,
        "md": 1.0,
        "tooth_mm": 0.35,
        "rough_w": (0.68, 0.24, 0.22, 0.08),
        "felt": None,
        "wire": None,
        "coating": None,
    },
    # Unbleached sack kraft: long, coarse, uncollapsed softwood fibre, no
    # filler, no bleaching -- the fibre network is fully exposed and carries
    # shives.
    "kraft": {
        "mm_across": (16.0, 30.0),
        "lab": (59.0, 9.5, 24.0),
        "lab_jitter": (2.8, 1.1, 2.0),
        "fibre_mm": 2.5,
        "fibre_sigma": 0.6,
        "width_um": 31.0,
        "coverage": 7.0,
        "kappa": (0.35, 0.55),
        "persistence": 1.1,
        "kinks": 3.0,
        "tone_spread": 4.2,
        "extinction": 0.45,
        "floc_spacing_mm": 1.9,
        "floc_spread_mm": 0.45,
        "clustered": 0.50,
        # Barely refined sack kraft, high freeness: the fibre wall is
        # still intact, so almost no fringe.
        "fibrillation": 0.010,
        "psf_um": 150.0,
        "sheen": (0.036, 0.058),
        "sheen_exp": (9.0, 14.0),
        "sq_um": 7.9,
        "cockle_um": (35.0, 120.0),
        "roughness_deg": 24.0,
        "calender_k": 0.05,
        "stack_skew": 0.22,
        "asperity": 0.0,
        "crease_chance": 0.30,
        "md": 1.0,
        "tooth_mm": 0.35,
        "rough_w": (0.68, 0.24, 0.22, 0.08),
        "felt": None,
        "wire": None,
        "coating": None,
    },
    # Deinked recycled: mixed short furnish, high fines, residual ink at three
    # scales, warm low-chroma grey (never neutral).
    "recycled": {
        "mm_across": (13.0, 24.0),
        "lab": (77.5, 1.6, 5.0),
        "lab_jitter": (2.2, 0.7, 1.2),
        "fibre_mm": 1.25,
        "fibre_sigma": 0.72,
        "width_um": 25.0,
        "coverage": 8.0,
        "kappa": (0.10, 0.30),
        "persistence": 0.9,
        "kinks": 3.5,
        "tone_spread": 2.8,
        "extinction": 0.50,
        "floc_spacing_mm": 1.6,
        "floc_spread_mm": 0.35,
        "clustered": 0.50,
        # Heavily worked through several life cycles, fines-rich.
        "fibrillation": 0.060,
        "psf_um": 130.0,
        "sheen": (0.028, 0.042),
        "sheen_exp": (9.0, 14.0),
        "sq_um": 5.85,
        "cockle_um": (30.0, 100.0),
        "roughness_deg": 22.0,
        "calender_k": 0.35,
        "stack_skew": 0.20,
        "asperity": 1.60,
        "crease_chance": 0.25,
        "md": 1.0,
        "tooth_mm": 0.35,
        "rough_w": (0.68, 0.24, 0.22, 0.08),
        "felt": None,
        "wire": None,
        "coating": None,
    },
    # Mechanical (groundwood/TMP) newsprint: 25-40% fines, and fibre that is
    # coarse, stiff and uncollapsed rather than ribbon-flat. Both facts point
    # the same way -- the sub-75 um debris fills the space between the fibres,
    # so the sheet reads as a *felt* rather than as a network of separable
    # fibres. That is what the LIC layer is for, and newsprint carries by far
    # the largest felt weights here. Low grammage (40-49 g/m^2) over a coarse
    # furnish also means low coverage and only ~90% opacity, so it is the most
    # translucent and hence the softest stock in the set.
    "newsprint": {
        "mm_across": (14.0, 26.0),
        "lab": (82.0, 1.5, 9.5),
        "lab_jitter": (2.0, 0.6, 1.5),
        "fibre_mm": 1.7,
        "fibre_sigma": 0.65,
        "width_um": 32.0,
        # 45 g/m^2 at 0.32 mg/m coarseness is ~140 mm of fibre per mm^2, which
        # at 32 um wide is only ~4.5 layers: half of what woodfree carries.
        "coverage": 5.0,
        "kappa": (0.30, 0.50),
        # Stiff, uncollapsed mechanical fibre: it kinks (it was ground, not
        # cooked) but it does not curl into ribbons the way chemical pulp does.
        "persistence": 1.6,
        "kinks": 2.4,
        "tone_spread": 3.4,
        "extinction": 0.40,
        "floc_spacing_mm": 1.7,
        "floc_spread_mm": 0.42,
        "clustered": 0.55,
        # Groundwood is fines-rich but it was never refined, so the
        # fines are debris rather than peeled fibril bundles.
        "fibrillation": 0.030,
        # 200-300 um, the largest lateral point spread of any stock here.
        "psf_um": 250.0,
        # 3-8 GU at 75 degrees: the flattest sheen in the set.
        "sheen": (0.019, 0.030),
        "sheen_exp": (7.0, 11.0),
        "sq_um": 5.95,
        "cockle_um": (30.0, 110.0),
        "roughness_deg": 25.0,
        "calender_k": 0.35,
        "stack_skew": 0.30,
        "asperity": 1.40,
        "crease_chance": 0.35,
        "md": 1.0,
        "tooth_mm": 0.32,
        # The felt supplies what the tooth band would otherwise have to, so the
        # spectral terms below it all come down.
        "rough_w": (0.58, 0.20, 0.16, 0.05),
        "felt": {
            "fines_mm": 0.22,
            "fibre_mm": 1.7,
            "layers": 6,
            "mass": 0.26,
            "albedo": 0.055,
            "height": 0.29,
        },
        "wire": None,
        "coating": None,
    },
    # Handmade / antique laid paper: a rag furnish couched off a laid mould, so
    # the wire pattern is pressed into the wet sheet (see :func:`_wire_marks`)
    # and there is no machine direction whatsoever -- a hand-shaken mould drains
    # in place, so the sheet is isotropic and its flocs are round. Uncalendered,
    # so the roughest stock here, and cream because there is no optical
    # brightener within two centuries of it.
    "laid": {
        # A crop from the middle of a large sheet, wide enough to show a chain
        # line: the deckle edge is explicitly out of scope.
        "mm_across": (26.0, 44.0),
        "lab": (91.0, 1.5, 8.0),
        "lab_jitter": (1.1, 0.4, 1.0),
        # Long, barely refined rag/linen fibre, and a heavy sheet.
        "fibre_mm": 1.9,
        "fibre_sigma": 0.7,
        "width_um": 24.0,
        "coverage": 10.5,
        # 0-0.05: isotropic. This is the defining statistic of a handmade sheet.
        "kappa": (0.0, 0.05),
        "persistence": 1.2,
        "kinks": 2.0,
        "tone_spread": 2.4,
        "extinction": 0.55,
        # A hand-shaken mould flocs at a larger scale than a headbox does.
        "floc_spacing_mm": 2.3,
        "floc_spread_mm": 0.55,
        "clustered": 0.55,
        # Hand-formed rag, lightly worked.
        "fibrillation": 0.020,
        "psf_um": 140.0,
        "sheen": (0.024, 0.038),
        "sheen_exp": (8.0, 13.0),
        # Uncalendered and hand-couched: rough, and prone to cockle -- but the
        # wire marks, the felt and the deep fibre relief already carry most of
        # this sheet's slope, so the roughness blend on top of them has to be
        # smaller than the bare Sq figure for handmade stock would suggest, or
        # the sheet measures steeper than paper does.
        "sq_um": 4.75,
        "cockle_um": (40.0, 130.0),
        "roughness_deg": 25.0,
        "calender_k": 0.0,
        # The highest warp of the six, and it has to be: uncalendered, so nothing
        # has flattened the fibre crowns, and at an ISO cut-off of lc = 1 mm the
        # laid pitch (~1.1 mm) sits astride the band edge, where the surviving
        # part of a symmetric sinusoid dilutes the marginal's skew. Both push the
        # same way, so this reads about the Ssk of the other uncalendered stock
        # despite the larger number.
        "stack_skew": 0.48,
        # Deliberately no asperity population, even though an uncalendered
        # hand-couched sheet has nothing to press protruding fibre back into the
        # plane -- so this is the one place the model is knowingly conservative.
        # It is a shape decision, not a stability one: this stock's Sku is already
        # the highest of the six (4.5 at lc = 1 mm) because it is the only stock
        # whose band is not truncated by a nip, and it needs no help to reach
        # paper's 3-5. The older reason given here -- that the population made Sku
        # "run away" (5.5 at 0.06, 28 at 0.14) -- was an artefact of the version
        # of :func:`_asperities` that normalised a sparse field to unit variance,
        # and no longer applies now that the population saturates.
        "asperity": 0.0,
        "crease_chance": 0.20,
        "md": 0.0,
        "tooth_mm": 0.42,
        "rough_w": (0.66, 0.24, 0.20, 0.06),
        "felt": {
            "fines_mm": 0.26,
            "fibre_mm": 1.9,
            "layers": 6,
            "mass": 0.18,
            "albedo": 0.030,
            "height": 0.20,
        },
        "wire": {
            # 20-28 wires/inch, and the chain stitching every 20-30 mm.
            "laid_pitch_mm": (0.95, 1.30),
            # The upper half of the 3-10 um / 8-25 um bands the marks really
            # span. The bottom of each is physical but renders as nothing: a
            # sheet that drew 3 um of laid line is a plain cream sheet in
            # reflected light, and three seeds in four landing there makes a
            # variant called ``laid`` that does not show laid lines. The full
            # range belongs to laid paper as a population; this stock is the
            # half of it that is worth rendering. The wire's reflectance
            # modulation comes out at 1.2-1.9% RMS over eight seeds -- still
            # the 1-3% a watermark reads at in reflected light, so this buys
            # consistency and not a printed pattern.
            "laid_um": (6.5, 10.0),
            "chain_pitch_mm": (20.0, 30.0),
            "chain_um": (16.5, 25.0),
        },
        "coating": None,
    },
    # Matte/silk coated stock. Coating is not a different generator: it is the
    # same fibre network with the fibre layer's *height* contribution suppressed
    # by ~80% and its *albedo* contribution by ~95% -- the pigment fills the
    # inter-fibre valleys and hides the furnish -- everything below ~50 um gone,
    # roughness down by an order of magnitude (Sq 0.5-2.5 um against 3-9), and
    # one new field: a 1-5 mm coating mottle that lives in the **gloss**. That
    # mottle is the single most identifiable coated cue, and it is nearly
    # invisible in colour, which is why it is applied at 1% to the albedo and
    # 55% to the sheen.
    "coated": {
        "mm_across": (12.0, 22.0),
        "lab": (95.0, 0.5, -2.0),
        "lab_jitter": (0.7, 0.25, 0.8),
        # The base sheet is ordinary woodfree stock; only what is on top of it
        # differs, so the furnish is white's furnish.
        "fibre_mm": 1.15,
        "fibre_sigma": 0.6,
        "width_um": 21.0,
        "coverage": 9.0,
        "kappa": (0.20, 0.40),
        "persistence": 1.4,
        "kinks": 1.5,
        # The 95% albedo suppression: per-fibre tone spread all but vanishes
        # under the coating, which is why coated stock looks so even.
        "tone_spread": 0.14,
        "extinction": 0.55,
        "floc_spacing_mm": 1.4,
        "floc_spread_mm": 0.30,
        "clustered": 0.45,
        # White's furnish, under the coat.
        "fibrillation": 0.035,
        # 10-40 um, against 90-250 for uncoated: the pigment scatters at the
        # surface rather than through the sheet, so the diffuse normal is barely
        # blurred and coated paper looks crisper.
        "psf_um": 25.0,
        # 10-25 GU matte, 30-50 GU silk at 75 degrees, against 4-10 uncoated --
        # and a much *tighter* lobe, so the exponent goes up with the weight.
        "sheen": (0.085, 0.175),
        "sheen_exp": (30.0, 48.0),
        "sq_um": 1.87,
        # Calendered and stiffer, so it cockles far less than an uncoated sheet.
        "cockle_um": (12.0, 45.0),
        "roughness_deg": 9.0,
        "calender_k": 0.80,
        # The only negative warp of the six. A pigment coat is a film, not a fibre
        # top: it bridges the crowns and pools in the hollows, so its marginal is
        # the fibre stack's mirrored -- a flat plateau at the blade's level with
        # excursions downward, where an uncoated sheet has a plateau with
        # excursions up. Zero would be defensible for a *film* in isolation, but
        # the band underneath it is still the soft-max over a fibre stack, which
        # arrives positively skewed; this is what cancels that and leaves the sign
        # the coat's own geometry implies.
        "stack_skew": -0.22,
        # The only negative population of the six, and the sole source of this
        # stock's negative Ssk. A coated surface's extremes are *voids* -- pinholes
        # where the blade starved a hollow, and sags over the substrate's pores --
        # and it is deep because there is nothing else left: the gloss calender has
        # truncated the upper tail, so the lower tail is the whole of the shape.
        "asperity": -2.80,
        "crease_chance": 0.10,
        # A coating is a brittle film over a smooth substrate, and the crease
        # amplitude in :func:`_creases` is tuned for uncoated stock, where a fold
        # buckles the whole fibre network. At full strength the Voronoi network
        # reads as crazed ceramic glaze rather than paper, so a third of it.
        "crease_scale": 0.33,
        "md": 1.0,
        # Coated topography is bimodal -- coat-weight variation above 1 mm and
        # pigment micro-texture at 0.05-0.3 mm, with the fibre tooth between them
        # filled in -- so this single band sits between the two rather than on
        # either, which is what lands the measured slope.
        "tooth_mm": 0.55,
        # The 80% height suppression, and no micro band at all: sub-50 um relief
        # is exactly what the coating fills.
        "rough_w": (0.68, 0.05, 0.04, 0.0),
        # A faint felt, still visible through a matte coating if you look for it.
        "felt": {
            "fines_mm": 0.12,
            "fibre_mm": 0.0,
            "layers": 5,
            "mass": 0.0,
            "albedo": 0.006,
            "height": 0.05,
        },
        "wire": None,
        "coating": {"mottle_mm": (1.4, 4.0), "gloss": 0.55, "albedo": 0.010},
    },
}

_SLABS = 3
_TONES = 2

# --- Fibre-scale detail, shared by every stock -------------------------------
# How far a peeled fibril reaches from the wall, in micrometres: the fringe band
# width. Textbook fibril widths are 50 nm - 1 um (whole peeled S1 ribbons 1-5 um),
# so at 10-39 um per pixel the fringe is a *sub-pixel* opacity contribution at
# every size rendered here and is splatted as one (see :func:`..core.fibres.deposit`).
_FRINGE_UM = 5.0
# How far a cut end's brush of frayed ribbons reaches past the break, 20-100 um.
_BRUSH_UM = 55.0
# Native end taper length. 150-500 um for a softwood tracheid; unverified.
_TAPER_UM = 300.0
# Along-fibre projected-width variation from intermittent collapse: CV and the
# correlation length of a collapse domain. **Both figures are UNVERIFIED** --
# fibre analysers report a *population* width distribution, which is a different
# quantity and cannot be substituted for the variation along one fibre.
_WIDTH_CV = 0.20
_WIDTH_CORR_UM = 350.0


def _cut_end_fraction(fibrillation: float) -> float:
    """Share of fibre ends that are refining cuts rather than native tips.

    Order 20-50%, rising with refining intensity -- and refining is also what
    peels fibrils, so one number drives both rather than two being authored
    independently. Anchored at the unrefined and heavily-worked ends of the
    fibrillation table. **The 20-50% range is a mechanism-backed guess, not a
    measurement.**
    """
    t = (float(fibrillation) - 0.010) / (0.060 - 0.010)
    return float(np.clip(0.20 + 0.30 * t, 0.20, 0.50))


# LIC costs O(H * W * layers * length_px / step), so the *product* of layers and
# kernel length is the thing to budget, not either alone. One 384x384 felt at
# length_px 48 measures 0.18 s, which is 0.1 s per (layer x pixel-of-length) at
# 2000x2000 -- so 150 buys about 15 s at that size and far less below it, and
# anything much larger dwarfs the fibre deposition that this is only a detail
# on. The kernel is also clamped in pixels: fines are a fixed *physical* size,
# and letting their kernel grow with resolution without bound is what turns a
# 2000 px render from seconds into a minute.
_FELT_BUDGET_PX = 150.0
_FELT_MAX_PX = 64.0
# Pixel budget for the felt canvas, ~1024x1024. See :func:`_felt_stack`.
_FELT_MAX_AREA_PX = 1_100_000.0


# How many fibre layers are close enough to the surface to be the one a probe
# touches. Coverage is 5-13 layers, but only the top two or three are within a
# fibre diameter of the plane, so those are the ones that compete for the
# surface; deeper layers are always buried by them. Matching :data:`_SLABS` is
# not a coincidence -- it is the same depth partition.
_CROWN_LAYERS = 3
# Softness of the maximum in :func:`_stack_top`, in units of the layer sd. A
# hard max creases along every watershed between layers; a fibre lying across
# another bends over it instead, so the knee is rounded at roughly the fibre's
# own bending scale. Fitted, not measured.
_STACK_SOFTNESS = 0.22
# Fraction of the surface the sparse extreme-value population occupies -- a few
# percent, which is the order the visible protruding-fibre, shive and coating-sag
# counts run at. Quoted as an area because that is how such counts are reported;
# see :func:`_asperities` for why a fixed sd threshold was the wrong parameter.
# Fitted, not measured.
_ASPERITY_AREA = 0.035
# Lateral extent of one asperity, in micrometres: a protruding fibre crown is a
# fibre width or two, a coating sag a few tens of micrometres. Below this there
# is nothing an asperity could be, so the population is smoothed to it.
_ASPERITY_SIZE_UM = 45.0
# Half-width of the skew warp's knee, in sd of the band it is applied to. Sets
# how much of the histogram the two slopes bend across; ~0.7 sd bends over the
# bulk and leaves both tails linear. Fitted.
_SKEW_KNEE_SD = 0.7
# Onset and shoulder width of the calender clip, in units of the band's Sq.
# There is no published bearing-area curve for paper to fit these from (nor for
# ``calender_k``), so they are fitted to the Ssk/Sku targets and are priors, not
# measurements.
#
# The onset has to sit out in the *tail*, because a nip is carried by the tallest
# asperities and nothing else. An onset of 0.75 Sq -- which is what this was --
# is the 77th percentile of the histogram, i.e. the bulk, and clipping from there
# does two wrong things at once. It flattens most of the upper half of the
# distribution, which drove the four calendered stocks *platykurtic* (measured
# Sku 2.74-2.94 at lc = 1 mm, against paper's 3-5: lighter tails than a Gaussian,
# which no real surface has). And it leaves whatever rare spikes are in the field
# as the only surviving structure above the knee, so they take over the moments.
# 2.0 Sq is about the 98th percentile: a couple of percent of the area, which is
# the order of the real contact area in a nip.
_CLIP_ONSET_SQ = 2.0
_CLIP_SHOULDER_SQ = 0.30
# Densification is read off the clip as a fraction of the band's Sq, so that a
# pixel pushed down by most of an Sq counts as fully worked.
_DENSIFY_SQ = 0.8
# Linear-reflectance loss and relative gloss gain of a fully densified spot.
# 4.5% of Y at L* 95 is dL* about -1.5, the low end of the -1 to -4 that
# calender blackening is reported at: it is a defect, not a feature.
_BLACKENING = 0.045
_CALENDER_GLOSS = 0.9


def _stack_top(layers: list[np.ndarray], softness: float) -> np.ndarray:
    """Soft maximum over a fibre stack's layer surfaces, zero-mean unit-variance.

    The height a probe reads in a column is set by whichever fibre reaches
    highest there, not by the average of the stack -- so the surface is a
    **maximum** over layers, and a maximum of random variables is positively
    skewed even when every layer is symmetric. That is one of the two reasons
    uncoated paper measures Ssk > 0 at fibre bandwidth: the extreme values are
    all protrusions (crowns, crossings, shives), because a pore cannot be
    deeper than the layer below it.

    The previous formulation, ``1 - exp(-1.3 * mass)``, did the opposite: at the
    ~3 layers of coverage a slab carries it saturates, so the field sits pinned
    near its ceiling with occasional dips and measures Ssk about -3.8. Saturating
    the surface is right for *opacity* and wrong for *height*.
    """
    stack = np.stack([np.asarray(f, dtype=np.float32) for f in layers], axis=0)
    # Work in the stack's own units so ``softness`` means the same thing whatever
    # the layers carry.
    stack = stack / np.float32(float(stack.std()) + 1e-6)
    top = stack.max(axis=0)
    if softness > 1e-4 and stack.shape[0] > 1:
        # log-sum-exp taken about the max, which is what keeps the exponential
        # from overflowing however far apart the layers are.
        top = top + np.float32(softness) * np.log(
            np.exp((stack - top) / np.float32(softness)).sum(axis=0)
        )
    top = top - np.float32(float(top.mean()))
    return (top / np.float32(float(top.std()) + 1e-6)).astype(np.float32)


def _skew_warp(field: np.ndarray, strength: float) -> np.ndarray:
    """Give a symmetric unit-variance field paper's positively skewed marginal.

    How high the surface stands in a column follows how much fibre is stacked
    there, and local grammage is a compound-Poisson sum of fibre masses: gamma
    in the continuum limit, Pearson III (shifted gamma) when fitted, which is
    what transmission measurements of paperboard report (*Fibers* 12(12):113,
    2024, DOI 10.3390/fib12120113 -- the distribution *family* is measured; the
    strength here is not).

    **What the shape of the warp has to respect.** The obvious way to reach a
    gamma-like marginal from a Gaussian is the exponential map ``expm1(s*z)/s``,
    which is what this used to be, and it is wrong here for a reason that does
    not show up in the algebra: its skewness figure (``gamma1 ~ 3*s``) is the
    skewness it produces *from a Gaussian input*, and the field it is handed is
    not Gaussian. :func:`_stack_top` and the felt already leave a heavy upper
    tail in the blend, and an exponential map does not reshape a tail, it
    **multiplies** it -- a 5-sd pixel comes out at ``exp(5s)/s``. At the 0.60
    ``stack_skew`` that ``laid`` needed, the roughness band measured Sku 26.9
    against paper's 3-5, all of it from that multiplication (ablating the warp
    alone took it to 3.06).

    So the warp is asymptotically **linear** instead: two slopes, ``1 + s``
    above the median and ``1 - s`` below it, joined by a smooth knee a fraction
    of an Sq wide (:data:`_SKEW_KNEE_SD`). That is also the better physical
    statement. The gamma marginal belongs to *mass*; height is mass seen
    through the sheet's own compressibility, which saturates, so the asymmetry
    between the two halves of the histogram is a gain difference and not an
    exponential blow-up. Ssk comes out at the same cost -- ``gamma1`` is about
    ``2.4*s`` for small ``s`` -- while Sku stays within a few percent of 3.

    Monotone (``|s| < 1`` is enforced), so it moves no feature laterally and
    leaves the field's periodicity -- and hence the tiling -- intact.
    """
    if abs(strength) <= 1e-6:
        return field
    # |s| >= 1 would make the warp non-monotone below the knee, which folds the
    # lower tail back on itself and invents extrema the surface never had.
    s = np.float32(float(np.clip(strength, -0.95, 0.95)))
    b = np.float32(_SKEW_KNEE_SD)
    z = np.asarray(field, dtype=np.float32)
    w = z + s * (np.sqrt(z * z + b * b) - b)
    w = w - np.float32(float(w.mean()))
    return (w / np.float32(float(w.std()) + 1e-6)).astype(np.float32)


def _asperities(
    crowns: np.ndarray, sign: float, area: float, size_px: float
) -> np.ndarray:
    """The sparse extreme-value population of the surface, saturating at unit depth.

    Sku > 3 is a statement about *rare* events, and a smooth monotone warp of a
    whole field cannot produce one without also moving the bulk of the histogram
    -- which is why :func:`_skew_warp` alone buys Ssk more cheaply than it buys
    Sku. The rare events in paper are a population, not a tail shape: single
    protruding fibres, fibre crossings and shives on an uncoated sheet (``sign``
    +1, they stand proud of the plane), and coating micro-voids and sags over the
    base sheet's own depressions on a coated one (``sign`` -1, the coat bridges
    the crowns and settles into the hollows). Taking the exceedance of the crown
    field above a high quantile isolates exactly that population and nothing else.

    Two things about *how* it is isolated are load-bearing, and both were wrong
    before:

    * **It saturates.** A protruding fibre stands one fibre diameter proud of
      the plane and a coating sag is as deep as the coat is thick -- neither is
      unbounded, so the population has a *depth*, and ``area`` of the surface is
      at it. The previous version took a raw exceedance and divided by its own
      standard deviation, which is a near-zero number for a sparse field, so the
      handful of most extreme pixels came out at tens of Sq: single-pixel spikes,
      not asperities. On ``coated`` that alone was the whole of a measured Sku of
      9.35 (ablating it took the band to 2.65), and it was invisible at the
      5.34 mm cut-off the tests used to measure at, because a filter that wide
      leaves enough cockle in the band to average the spikes away.
    * **It is thresholded by area, not by sd.** ``crowns`` is a soft maximum, so
      it is skewed, so a fixed 1.6-sd threshold picks out a few percent of the
      area on the positive side and a few *tenths* of a percent on the negative
      one. An area fraction is the physical parameter anyway -- a dirt or
      protruding-fibre count is quoted per unit area -- and it makes the
      positive and negative populations the same size.

    ``size_px`` is the asperity's own lateral extent: a fibre width for a crown,
    a few tens of micrometres for a coating sag. Below it there is nothing for
    the population to be, so a single pixel over the threshold is measurement
    noise in the crown field rather than an asperity.

    Returns a zero-mean field whose extreme reaches ``+/-1``, so the caller's
    weight is the population's depth in whatever units it is blending in.
    """
    s = np.float32(sign) * np.asarray(crowns, dtype=np.float32)
    area = float(np.clip(area, 1e-4, 0.5))
    thr = float(np.quantile(s, 1.0 - area))
    # The deepest fifth of the population sits at full depth; the rest ramps.
    top = float(np.quantile(s, 1.0 - 0.2 * area))
    span = max(top - thr, 1e-4)
    e = np.clip((s - np.float32(thr)) / np.float32(span), 0.0, 1.0)
    e = (e * e * (3.0 - 2.0 * e)).astype(np.float32)
    if size_px > 1e-2:
        e = gaussian_blur(e, size_px)
    peak = float(e.max())
    if peak < 1e-6:
        return np.zeros_like(s)
    e = e / np.float32(peak)
    return (np.float32(sign) * (e - np.float32(float(e.mean())))).astype(np.float32)


def _calender_clip(height_um: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided softplus clip of the roughness band. Returns ``(clipped, push)``.

    A calender nip is two rolls far smoother than the sheet, so contact -- and
    with it the compressive strain -- is established at the tallest asperities
    first. Paper is enormously compressible through its thickness and nearly
    incompressible in-plane, and there is no lateral mass transport beyond a
    fibre width, so the peaks are crushed while the valleys survive as valleys:
    a pore stays a pore. That makes the operator one-sided.

    The knee has to be smooth. A hard ``min(z, z0)`` piles every crushed pixel
    onto one level -- a delta spike in the histogram and a discontinuity in
    slope -- whereas the deformation is viscoelastic and recovers partially, so
    the real transition is gradual. A softplus is the cheapest shape with that
    property.

    ``push`` is how far each pixel was driven down, which **is** the local
    densification: the same map therefore drives the height loss here, the gloss
    rise and the opacity loss (calender blackening) in :func:`render_sheet`.
    Deriving all three from one map is the point -- the earlier code picked its
    glossy freckles from the floc field and never flattened them, which cannot
    happen in a nip.

    ``k`` is clamped to 1. The derivative of ``z - k*shoulder*softplus(t)`` is
    ``1 - k*sigmoid(t)``, which is positive for every ``t`` only while ``k <= 1``;
    above that the operator turns a tall asperity into a *hole*, reversing the
    order of two heights and inventing a valley where the nip found a peak. The
    configured loads are all <= 0.8, so this is a guard on the parameter and not
    a change to any stock.
    """
    k = float(np.clip(k, 0.0, 1.0))
    sq = float(height_um.std()) + 1e-6
    z0 = float(height_um.mean()) + _CLIP_ONSET_SQ * sq
    shoulder = _CLIP_SHOULDER_SQ * sq
    t = (np.asarray(height_um, dtype=np.float32) - np.float32(z0)) / np.float32(
        shoulder
    )
    # log1p(exp(t)) tends to t for large t, so evaluate it that way rather than
    # letting the exponential overflow on a tall outlier.
    soft = np.where(t > 20.0, t, np.log1p(np.exp(np.minimum(t, np.float32(20.0)))))
    push = np.float32(k * shoulder) * soft.astype(np.float32)
    return (height_um - push).astype(np.float32), push


def _base_lab(variant: str, rng: np.random.Generator) -> np.ndarray:
    """Per-seed base colour in CIELAB, drawn around the stock's nominal value.

    White stock also samples its optical-brightener load: b* runs from about
    +3 (unbrightened natural white, or a cream book paper) to -7 (a heavily
    brightened premium sheet, which photographs blue-white in daylight).
    """
    stock = STOCKS[variant]
    lab = np.asarray(stock["lab"], dtype=np.float64).copy()
    jit = np.asarray(stock["lab_jitter"], dtype=np.float64)
    lab = lab + rng.normal(0.0, jit)
    if variant == "white":
        roll = rng.random()
        if roll < 0.22:
            # Natural white / cream: no OBA, lignin-free but unbrightened.
            lab[1] += rng.uniform(0.0, 1.0)
            lab[2] = rng.uniform(2.0, 8.0)
            lab[0] -= rng.uniform(0.5, 3.5)
        elif roll < 0.35:
            # Premium high-white: heavy OBA load.
            lab[2] = rng.uniform(-5.0, -3.0)
    elif variant == "newsprint":
        # Mechanical pulp keeps its lignin, and lignin photo-oxidises to
        # quinone chromophores within days of daylight: b* runs from the fresh
        # +9.5 up towards +13, and L* falls as it goes.
        age = float(rng.random()) ** 1.6
        lab[2] += 3.2 * age
        lab[0] -= 1.2 * age
    elif variant == "laid":
        # No optical brightener existed, so the whole range is on the warm side
        # of neutral: from a distinctly cream rag sheet towards the natural
        # white of a well-washed one, never past it.
        t = float(rng.random())
        lab = lab + t * (np.asarray([93.5, 0.8, 2.5]) - np.asarray([91.0, 1.5, 8.0]))
    return lab


def _fines_and_flocs(
    shape: tuple[int, int],
    rng: np.random.Generator,
    ppm: float,
    machine_dir: float,
    md: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fines, wet_streak)`` -- the sub-resolution and long-range mass fields.

    Fines (ray cells and refining debris under ~75 um) are 20-40% of the
    furnish but are smaller than a pixel at any sensible scale, so they belong
    in the spectrum rather than in drawn geometry: a band around one to two
    pixels is exactly what they contribute.

    Wet streaks are the 5-30 mm excess-variance tail of the formation
    spectrum, elongated along the machine direction -- so a sheet with no
    machine direction (``md = 0``) has round patches instead of streaks.
    """
    fines = band_field(shape, rng, centre_px=max(1.4, 0.05 * ppm), octaves_wide=1.1)
    streak = matern_field(
        shape,
        rng,
        corr_px=max(6.0, 7.0 * ppm),
        nu=0.8,
        aniso=1.0 + float(md),
        angle=machine_dir,
    )
    return fines, streak


def _felt_stack(
    shape: tuple[int, int],
    rng: np.random.Generator,
    spec: FeltSpec,
    ppm: float,
    machine_dir: float,
    kappa: float,
) -> np.ndarray:
    """Sub-resolution fibre texture as multi-layer LIC felt, unit variance.

    This is the layer *below* the scale at which a fibre is worth drawing.
    ``core.fibres`` deposits everything the render can resolve; what is left is
    fines and inter-fibre weave at 20-120 um -- millions of members, none
    individually traceable -- and that is exactly what LIC of white noise along
    a direction field produces.

    Two things decide whether it works:

    * **Layers.** One direction field gives every filament in a neighbourhood
      the same heading, which is combed hair, not felt. Real fibres criss-cross,
      so each felt is a stack of independently-oriented LIC fields (see
      :func:`..core.lic.felt`). Three reads as a mat, five or six as a felt.
    * **Length in millimetres, not pixels.** ``fines_mm`` is the fines scale
      (0.05-0.30 mm), ``fibre_mm`` an optional second, longer population at
      fibre length. Both convert through ``ppm``, so the felt is the same
      physical texture at any render size, and the layer count is then whatever
      the cost budget allows at that size.

    Above ~1 megapixel the stack is computed on a proportionally smaller canvas
    and resampled up, because LIC costs ``O(H*W*layers*length_px/step)`` and both
    the pixel count (quadratically) and ``length_px`` (linearly) grow with the
    sampling density -- so the true cost goes as the *cube* of resolution, and a
    2000 px sheet does ~25x the LIC work of a 512 px one for no visible gain.

    This scaling is an approximation, not a lossless transform. The resize
    itself is exact -- :func:`..core.spectral.resize_periodic` reconstructs the
    reduced canvas' own band-limited interpolant and keeps the tile seamless --
    but the reduced canvas cannot *hold* anything above its own Nyquist, so
    everything finer is simply never generated (which is also why the variance
    has to be renormalised below). ``length_px`` is separately clamped to
    ``_FELT_MAX_PX``, so at high resolution the filaments are shorter than their
    physical length as well. Both are acceptable here for the same reason: at
    2000 px the filaments are already tens of pixels wide, so the felt's own
    structure is far coarser than the band being dropped, and the genuinely
    pixel-scale detail in a sheet comes from the fibre deposition layer rather
    than from this one.
    """
    h, w = int(shape[0]), int(shape[1])

    if h * w > _FELT_MAX_AREA_PX:
        # Scale ppm by the same factor as the canvas so ``length_px`` and
        # ``wander_px`` shrink with it and the *physical* felt is unchanged;
        # resize_periodic keeps the result seamless, which a bilinear upsample of
        # a tile would not.
        f = float(np.sqrt(_FELT_MAX_AREA_PX / float(h * w)))
        rh = max(
            2,
            2 * round(h * f / 2.0),
        )
        rw = max(
            2,
            2 * round(w * f / 2.0),
        )
        small = _felt_stack((rh, rw), rng, spec, ppm * f, machine_dir, kappa)
        big = resize_periodic(small, (h, w))
        sd = float(big.std())
        if sd < 1e-8:
            return np.zeros((h, w), dtype=np.float32)
        # Resampling drops the band above the small canvas' Nyquist, so restore
        # the unit variance this function promises.
        return (big / np.float32(sd)).astype(np.float32)

    out = np.zeros((h, w), dtype=np.float32)
    hit = False
    # Two populations sharing one cost budget, two thirds of it to the fines:
    # they are the layer that must not be thin, because a felt of *short*
    # filaments needs many orientations before it stops looking combed, while
    # the long population is a broad grain that reads fine at two or three.
    # The long one is also a single octave -- it *is* the coarse fibre, and a
    # third-length sub-octave of it would only duplicate the fines felt.
    populations: tuple[
        tuple[Literal["fines_mm", "fibre_mm"], float, float, int], ...
    ] = (
        ("fines_mm", 100.0, 1.0, 2),
        ("fibre_mm", 50.0, 0.62, 1),
    )
    for key, share, weight, octaves in populations:
        scale_mm = float(spec.get(key, 0.0))
        if scale_mm <= 0.0:
            continue
        budget = share * _FELT_BUDGET_PX / 150.0
        length_px = float(np.clip(scale_mm * ppm, 2.5, _FELT_MAX_PX))
        layers = int(np.clip(budget // length_px, 2, int(spec["layers"])))
        # Two layers is the floor -- below that there is no felt, only a comb --
        # so at high resolution the kernel gives way instead of the layer count.
        # Both clamps trade physical length for orientation variety, which is
        # the right way round: length is what the eye forgives.
        length_px = min(length_px, budget / layers)
        out += np.float32(weight) * felt(
            (h, w),
            rng,
            length_px=length_px,
            mu=machine_dir,
            kappa=kappa,
            # The grain wanders over a floc, not over a fixed pixel count: the
            # local machine direction is set by where the flocs went.
            wander_px=max(5.0, 0.55 * ppm),
            octaves=octaves,
            layers=layers,
            max_steps=48,
        )
        hit = True
    if not hit:
        return out
    return (out / np.float32(float(out.std()) + 1e-6)).astype(np.float32)


def wire_counts(
    shape: tuple[int, int], ppm: float, laid_pitch_mm: float, chain_pitch_mm: float
) -> tuple[int, int]:
    """Periods per tile for a target laid and chain pitch, and nothing else.

    Every periodic feature must fit a **whole** number of periods into the tile
    or the sheet seams, so the pitch is not a free parameter: it is whatever the
    nearest integer count implies. Parameterise by count, report the pitch.
    """
    mm_down = shape[0] / max(ppm, 1e-6)
    mm_across = shape[1] / max(ppm, 1e-6)
    return (
        max(
            1,
            round(mm_down / max(laid_pitch_mm, 1e-6)),
        ),
        max(
            1,
            round(mm_across / max(chain_pitch_mm, 1e-6)),
        ),
    )


def _wire_marks(
    shape: tuple[int, int],
    rng: np.random.Generator,
    spec: WireSpec,
    ppm: float,
    formation: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Laid-mould wire marks in micrometres, plus the pitches they came out at.

    A laid mould carries closely spaced *laid* wires running along its length
    and widely spaced *chain* wires stitched around them at right angles. Both
    press the wet web, and the compressed areas end up **thinner** -- which is
    the watermark mechanism, and why the marks are so much stronger in
    transmission than in reflection. Three things follow, and all three are the
    difference between a watermark and a printed pattern:

    * The marks belong in the **height** field, at 3-10 um for laid lines and
      8-25 um for chain lines (the chain wires sit proud on the outside of the
      mould, so they press harder and read bolder). In reflected light that is
      only 1-3% contrast, visible when the light rakes and not otherwise. Put
      them in the albedo at 10% instead and you get a print.
    * They are modulated by **formation**: the wire can only displace fibre
      where there is fibre to displace, so the mark fades over the thin patches.
      An unmodulated pattern reads as a screen overlay laid on top of the sheet.
    * The wires **sag and the web drifts**, so the phase wanders by a percent
      or two of the pitch. Perfectly straight lines read as printed.

    Returns ``(height_um, laid_pitch_mm, chain_pitch_mm)``.
    """
    h, w = int(shape[0]), int(shape[1])
    n_laid, n_chain = wire_counts(
        (h, w),
        ppm,
        float(rng.uniform(*spec["laid_pitch_mm"])),
        float(rng.uniform(*spec["chain_pitch_mm"])),
    )
    u = (np.arange(w, dtype=np.float32) / np.float32(w))[None, :]
    v = (np.arange(h, dtype=np.float32) / np.float32(h))[:, None]
    two_pi = np.float32(2.0 * np.pi)

    # Lateral wander, as a phase offset in radians of the wire's own cycle. A
    # Matern field is periodic, so this keeps the tile seamless where a plain
    # ramp or an aperiodic noise would break it; 0.12 cycles is a ~2% pitch
    # jitter accumulated over the sheet, which is the measured order.
    wander = matern_field(
        (h, w), rng, corr_px=max(8.0, 0.35 * min(h, w)), nu=1.5, aniso=1.0
    )
    # Only the *count* has to be an integer for the tile to close; a constant
    # phase is free. Without these the whole wire grid registers to the top-left
    # corner of every render -- a laid line exactly on row 0 and, since a 20 mm
    # crop fits only one chain period, a chain line exactly on column 0.
    laid_offset, chain_offset = rng.uniform(0.0, 2.0 * np.pi, size=2)
    laid_phase = (
        two_pi * np.float32(n_laid) * v
        + np.float32(laid_offset)
        + np.float32(0.12 * 2.0 * np.pi) * wander
        # A low integer harmonic of the *other* axis: the wires are stitched to
        # the chains, so they bow slightly between them, and an integer number
        # of bows keeps the tile periodic.
        + np.float32(0.05 * 2.0 * np.pi)
        * np.sin(two_pi * np.float32(max(n_chain, 1)) * u)
    )
    chain_phase = (
        two_pi * np.float32(n_chain) * u
        + np.float32(chain_offset)
        + np.float32(0.10 * 2.0 * np.pi)
        * matern_field(
            (h, w), rng, corr_px=max(8.0, 0.5 * min(h, w)), nu=1.5, aniso=1.0
        )
    )

    # The wire only marks where there is fibre to displace.
    mod = np.float32(0.6) + np.float32(0.8) * normalize01(formation)

    # A wire is a rounded ridge pressed into the web, so the imprint is a
    # narrowed cosine trough rather than a sinusoid: 1 under the wire, 0 between.
    laid = (np.float32(0.5) + np.float32(0.5) * np.cos(laid_phase)) ** np.float32(1.6)
    cc = np.cos(chain_phase)
    chain = (np.float32(0.5) + np.float32(0.5) * cc) ** np.float32(2.6)
    # Pulp displaced by the proud chain wire piles just off it: an antisymmetric
    # lip, which is what makes a chain line catch a raking light at all.
    lip = (
        np.float32(0.30)
        * np.sin(chain_phase)
        * (np.float32(0.5) + np.float32(0.5) * cc) ** np.float32(1.4)
    )

    laid_um = float(rng.uniform(*spec["laid_um"]))
    chain_um = float(rng.uniform(*spec["chain_um"]))
    marks = mod * (
        np.float32(-laid_um) * laid + np.float32(chain_um) * (lip - chain)
    ).astype(np.float32)
    # Zero-mean: the wire thins the sheet locally, and a constant offset in a
    # height field is invisible anyway, but it would otherwise inflate Sq.
    marks = marks - np.float32(float(marks.mean()))
    return marks.astype(np.float32), h / ppm / n_laid, w / ppm / n_chain


def _creases(
    shape: tuple[int, int], rng: np.random.Generator, ppm: float
) -> np.ndarray:
    """A crumple network: sharp ridges bounding soft facets, roughly in [-1, 1].

    Crumpled paper breaks into flat polygonal facets separated by sharp,
    nearly straight folds that meet at vertices. That is a Voronoi diagram,
    not a noise field: taking the cell boundaries of a warped Worley pattern
    gives creases with the right straightness and the right junctions, where
    the zero set of a smooth noise field only ever gives rounded puddles.
    """
    x, y = grid_coords(shape)
    xw, yw = warp((x, y), rng, amp=0.10, freq=1.8, octaves=3)
    # Facets are 5-14 mm across, so the cell count follows the capture scale
    # rather than the pixel count: a closer crop shows fewer, larger facets.
    mm_across = shape[1] / max(ppm, 1e-3)
    facet_mm = float(rng.uniform(5.0, 14.0))
    density = float(np.clip(mm_across / facet_mm, 1.5, 12.0))
    f1, f2 = worley_at(
        xw, yw, rng, density=density, stretch=float(rng.uniform(0.7, 1.4))
    )
    # Sharp fold lines where two cells meet, plus a gentle dome per facet so
    # the flats between folds are not dead level.
    fold = np.exp(-(f2 - f1) / np.float32(rng.uniform(0.018, 0.038)))
    # Crumpled paper facets are *flat* -- the sheet is inextensible, so it
    # folds rather than stretching. Doming the cells is what makes a Voronoi
    # crumple read as quilted leather instead of paper.
    facet = (0.5 - f1) * np.float32(0.08)
    # Most folds are ridges; a minority are valleys, as in a real crumple.
    sign = np.where(fbm_at(x, y, rng, freq=3.5, octaves=2) > -0.15, 1.0, -0.8)
    return (fold * sign + facet).astype(np.float32)


def _specks(
    shape: tuple[int, int],
    rng: np.random.Generator,
    count: int,
    radius_px: tuple[float, float],
    rng_aspect: tuple[float, float] = (1.0, 2.2),
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Irregular dark inclusions: a coverage mask in [0, 1].

    Sizes are heavy-tailed -- many tiny specks, a few large shards -- because
    dirt counts (TAPPI T437, in mm^2/m^2) are dominated by area from rare large
    particles while the *count* is dominated by small ones.

    ``weights`` is an optional (H, W) placement density: residual ink adheres
    to fines, so specks are not uniformly scattered but clustered wherever
    fines collect. Uniformly scattered dots are the thing that reads as
    digital dirt.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    n = int(count)
    if n <= 0:
        return mask
    lo, hi = radius_px
    # Pareto-ish: most near lo, a tail towards hi.
    radii = lo * (hi / max(lo, 1e-3)) ** rng.power(0.35, size=n)
    angles = rng.uniform(0.0, np.pi, size=n)
    aspects = rng.uniform(*rng_aspect, size=n)
    if weights is None:
        cx = rng.uniform(0.0, w, size=n)
        cy = rng.uniform(0.0, h, size=n)
    else:
        prob = np.clip(np.asarray(weights, dtype=np.float64).ravel(), 1e-6, None)
        prob = prob / prob.sum()
        flat = rng.choice(prob.size, size=n, p=prob)
        cy = (flat // w).astype(np.float64) + rng.random(n)
        cx = (flat % w).astype(np.float64) + rng.random(n)
    strengths = rng.uniform(0.45, 1.0, size=n)
    for i in range(n):
        r = float(radii[i])
        a = float(aspects[i])
        half = int(np.ceil(r * a)) + 1
        y0, y1 = int(cy[i]) - half, int(cy[i]) + half + 1
        x0, x1 = int(cx[i]) - half, int(cx[i]) + half + 1
        yy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy[i]
        xx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx[i]
        ca, sa = np.cos(angles[i]), np.sin(angles[i])
        xr = (ca * xx + sa * yy) / (r * a)
        yr = (-sa * xx + ca * yy) / r
        d = np.sqrt(xr * xr + yr * yr)
        blob = strengths[i] * smoothstep(1.15, 0.55, d)
        rows = np.arange(y0, y1) % h
        cols = np.arange(x0, x1) % w
        patch = mask[np.ix_(rows, cols)]
        mask[np.ix_(rows, cols)] = np.maximum(patch, blob)
    return mask


def generate(
    shape: tuple[int, int],
    rng: np.random.Generator,
    variant: str = "white",
    **params,
) -> np.ndarray:
    """Generate a paper texture as float32 (H, W, 3) sRGB in [0, 1]."""
    lit, _height_um, _ppm = render_sheet(shape, rng, variant, **params)
    return np.clip(linear_to_srgb(lit), 0.0, 1.0).astype(np.float32)


def render_sheet(
    shape: tuple[int, int],
    rng: np.random.Generator,
    variant: str = "white",
    *,
    out: dict | None = None,
    **params,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Render a sheet, returning ``(linear_rgb, height_um, px_per_mm)``.

    Split out from :func:`generate` so the surface can be measured rather than
    inferred from the encoded pixels: roughness (Sq) and RMS slope are the two
    numbers that decide whether paper reads as paper or as tooled leather, and
    both live in the height field, not in the output.

    ``out``, if given, is a dict the intermediate fields are written into. The
    height field is already returned, but *mass* is not, and mass is the thing
    formation is defined on: local grammage is what a beta-radiograph measures,
    and its statistics (CV and skewness through a stated aperture) are the only
    way to check the deposition against formation measurements. Two keys:

    * ``mass`` -- deposited fibre coverage, in mean-fibre-layers per pixel.
      Extensive and non-negative; this is grammage up to a constant.
    * ``formation`` -- that mass low-passed to the floc band (0.35 mm) and
      normalised to zero mean and unit variance, which is the field that drives
      relief and modulates the wire marks.

    Nothing else in the render depends on ``out``, so passing it changes no
    pixel; the signature and return value are unchanged for callers that do not.
    """
    if variant not in STOCKS:
        raise ValueError(f"unknown paper variant {variant!r}; choose from {VARIANTS}")
    stock = STOCKS[variant]
    h, w = int(shape[0]), int(shape[1])

    # --- Capture scale ---------------------------------------------------
    # Everything downstream is expressed in millimetres and converted here,
    # so a 2000 px render of the same sheet shows more detail rather than
    # bigger features.
    mm_across = float(params.get("mm_across", rng.uniform(*stock["mm_across"])))
    ppm = w / max(mm_across, 1e-3)
    um = ppm / 1000.0  # micrometres -> pixel-equivalent height units

    machine_dir = float(rng.uniform(0.0, np.pi))
    kappa = float(rng.uniform(*stock["kappa"]))
    md = float(stock["md"])

    # --- Fibre stack ------------------------------------------------------
    fibre_px = stock["fibre_mm"] * ppm
    width_px = stock["width_um"] * 1e-3 * ppm
    coverage = float(stock["coverage"]) * float(rng.uniform(0.85, 1.15))
    total = count_for_coverage((h, w), coverage, fibre_px, width_px)

    # One clustered point set for the whole stack, so flocs line up through
    # the thickness instead of each slab inventing its own.
    xs, ys = cluster_points(
        (h, w),
        rng,
        total,
        clustered=float(stock["clustered"]),
        parent_spacing_px=stock["floc_spacing_mm"] * ppm,
        spread_px=stock["floc_spread_mm"] * ppm,
        aniso=1.0 + 0.6 * md,
        angle=machine_dir,
    )
    order = rng.permutation(total)
    xs, ys = xs[order], ys[order]

    # Fibre-scale detail, all specified in micrometres and converted here so it
    # is the same physical structure at any render size. The fringe and the brush
    # are sub-pixel at every size, which is the whole reason they are opacity
    # rather than drawn strands.
    fibrillation = float(stock["fibrillation"])
    fringe_px = _FRINGE_UM * 1e-3 * ppm
    brush_px = _BRUSH_UM * 1e-3 * ppm
    taper_px = _TAPER_UM * 1e-3 * ppm
    width_corr_px = _WIDTH_CORR_UM * 1e-3 * ppm
    cut_ends = _cut_end_fraction(fibrillation)

    per_group = max(1, total // (_SLABS * _TONES))
    # Samples are splatted bilinearly, so they must stay within a pixel of one
    # another or the fibre breaks into beads. This is the cost that buys a
    # continuous ribbon, and it is the dominant cost of the whole generator.
    step_px = 0.9
    slab_masses: list[np.ndarray] = []
    group_masses: list[list[np.ndarray]] = []
    for slab in range(_SLABS):
        tones = []
        for tone in range(_TONES):
            idx = (slab * _TONES + tone) * per_group
            sel = slice(idx, idx + per_group)
            tones.append(
                deposit(
                    (h, w),
                    rng,
                    count=per_group,
                    length_px=fibre_px,
                    width_px=width_px,
                    length_sigma=float(stock["fibre_sigma"]),
                    theta_mu=machine_dir,
                    kappa=kappa,
                    persistence=float(stock["persistence"]),
                    kinks_per_length=float(stock["kinks"]),
                    step_px=step_px,
                    max_steps=int(np.clip(fibre_px / step_px * 2.2, 16, 900)),
                    positions=(xs[sel], ys[sel]),
                    width_cv=_WIDTH_CV,
                    width_corr_px=width_corr_px,
                    taper_px=taper_px,
                    cut_ends=cut_ends,
                    fibrillation=fibrillation,
                    fringe_px=fringe_px,
                    brush_px=brush_px,
                )
            )
        group_masses.append(tones)
        slab_masses.append(tones[0] + tones[1])

    mass = np.sum(slab_masses, axis=0)
    fines, streak = _fines_and_flocs((h, w), rng, ppm, machine_dir, md)
    # Fines and wet streaks ride on the fibre mass rather than being
    # independent layers: debris collects where fibre collects.
    mass = mass * (1.0 + 0.16 * fines + 0.10 * streak)

    # --- Sub-resolution felt ---------------------------------------------
    # Only for stocks that specify one, and only after every draw the other
    # stocks make, so adding it leaves their pixels untouched.
    felt_field: np.ndarray | None = None
    felt_spec = stock["felt"]
    if felt_spec is not None:
        felt_field = _felt_stack((h, w), rng, felt_spec, ppm, machine_dir, kappa)
        # Fines occupy volume, so they add mass -- but mass is a coverage and
        # cannot go negative, and a unit-variance field has a tail past -3.
        mass = mass * np.maximum(
            1.0 + np.float32(felt_spec["mass"]) * felt_field, np.float32(0.0)
        )

    # --- Optical composite: Beer-Lambert alpha-over, back to front --------
    base_lab = _base_lab(variant, rng)
    psf_px = float(stock["psf_um"] * 1e-3 * ppm)
    # Pores between fibres look deeper into the sheet, so they are darker and
    # very slightly warmer.
    substrate = lab_to_linear_rgb(base_lab + np.asarray([-6.0, 0.4, 0.8]))
    rgb = np.broadcast_to(substrate, (h, w, 3)).astype(np.float32).copy()

    tone_spread = float(stock["tone_spread"])
    # Extinction per unit coverage. A single cellulose fibre is *translucent*
    # -- it transmits most of what falls on it -- so one layer must not be one
    # opaque stroke. Tie tau to the fibre, not to the slab count: at 0.5 per
    # layer of coverage each group contributes ~40% and the six groups reach
    # 96% together, which is what makes the result read as a stack seen
    # through itself rather than as the topmost layer painted over the rest.
    tau = float(params.get("extinction", stock["extinction"]))
    for slab in range(_SLABS):
        depth = (_SLABS - 1 - slab) / max(_SLABS - 1, 1)  # 1 = deepest
        # Only *buried* fibres are seen through the scattering pulp. Light
        # reflecting off a surface fibre leaves before it can diffuse, so the
        # top slab stays sharp regardless of the point-spread radius --
        # scaling every slab by the PSF washes the surface out at high
        # resolution, where the fibre is several pixels wide.
        blur_px = 0.35 + 2.0 * psf_px * depth**1.4
        for tone in rng.permutation(_TONES):
            m = group_masses[slab][int(tone)]
            alpha = 1.0 - np.exp(-np.float32(tau) * m)
            alpha = gaussian_blur(alpha, blur_px)
            dl = tone_spread * (1.0 if tone else -1.0) * float(rng.uniform(0.6, 1.3))
            colour = lab_to_linear_rgb(
                base_lab
                + np.asarray(
                    [
                        dl,
                        rng.normal(0.0, tone_spread * 0.18),
                        rng.normal(0.0, tone_spread * 0.35),
                    ]
                )
            )
            a = alpha[..., None]
            rgb = a * colour + (1.0 - a) * rgb

    # Fines and filler are finer than the drawn fibres, so they show up as a
    # multiplicative grain rather than as geometry. It is the *same* field that
    # thickened the mass above: fines occupy volume and scatter light, and
    # splitting them into two independent textures is what makes procedural
    # grain read as sensor noise. Keep it to a couple of percent.
    rgb = rgb * (1.0 + np.float32(rng.uniform(0.010, 0.022)) * fines[..., None])
    if felt_field is not None and felt_spec is not None:
        # The felt is *oriented* debris, so unlike the isotropic fines band it
        # reads as grain with a direction to it -- the thing that separates a
        # mechanical-pulp sheet from a noisy one. Still only a few percent:
        # reflectance saturates with grammage.
        rgb = rgb * (
            1.0 + np.float32(felt_spec["albedo"]) * felt_field[..., None]
        ).astype(np.float32)

    # --- Coating mottle ---------------------------------------------------
    # Uneven coat weight and uneven coating gloss at 1-5 mm: the classic coated
    # defect, and the single most identifiable coated cue. It is a *gloss*
    # feature, so it goes into the sheen at full strength and into the albedo at
    # 1% -- put it in the colour instead and the sheet reads as blotchy paper
    # rather than as coated paper.
    mottle: np.ndarray | None = None
    coating = stock["coating"]
    if coating is not None:
        mottle = matern_field(
            (h, w),
            rng,
            corr_px=max(3.0, float(rng.uniform(*coating["mottle_mm"])) * ppm),
            nu=1.2,
        )
        rgb = rgb * (1.0 + np.float32(coating["albedo"]) * mottle[..., None]).astype(
            np.float32
        )

    # --- Dark inclusions --------------------------------------------------
    area_mm2 = mm_across * (h / w) * mm_across
    if variant == "kraft":
        # Shives: incompletely cooked fibre bundles, 0.5-3 mm long and much
        # darker than the felt around them.
        shive_count = int(rng.integers(25, 90) * area_mm2 / 400.0) + 4
        shives = deposit(
            (h, w),
            rng,
            count=shive_count,
            length_px=0.55 * ppm,
            width_px=max(0.055 * ppm, 0.7),
            length_sigma=0.55,
            theta_mu=machine_dir,
            kappa=kappa * 1.5,
            persistence=2.5,
            kinks_per_length=0.8,
            step_px=step_px,
        )
        # A shive sits *in* the sheet, so it is seen through a layer of pulp:
        # softened and de-contrasted, never a crisp dark stroke on the surface.
        shives = gaussian_blur(shives, max(0.6, psf_px * 0.7))
        a = np.clip(shives * 0.75, 0.0, 0.55)[..., None]
        dark = lab_to_linear_rgb(base_lab + np.asarray([-15.0, 1.5, 1.0]))
        rgb = a * dark + (1.0 - a) * rgb
    elif variant == "recycled":
        # Three ink populations. The sub-visible one is what actually makes
        # deinked stock grey; the drawable specks are comparatively rare, and
        # making them the only ink layer is what produces "digital dots".
        ink_haze = normalize01(
            matern_field((h, w), rng, corr_px=max(3.0, 0.5 * ppm), nu=0.5)
        )
        rgb = rgb * (1.0 - np.float32(rng.uniform(0.015, 0.040)) * ink_haze[..., None])

        speck_n = int(rng.uniform(400.0, 1600.0) * area_mm2 / 400.0)
        specks = _specks(
            (h, w),
            rng,
            speck_n,
            (0.3, max(0.45, 0.11 * ppm)),
            weights=ink_haze**1.6 + 0.25,
        )
        specks = gaussian_blur(specks, max(0.4, psf_px * 0.35))
        a = (specks * float(rng.uniform(0.35, 0.65)))[..., None]
        ink = lab_to_linear_rgb(base_lab + np.asarray([-34.0, 0.0, -2.0]))
        rgb = a * ink + (1.0 - a) * rgb

        # Coloured contraries from printed waste: rare, and fibre-shaped
        # because they are dyed fibres, not blobs.
        fleck_n = max(1, int(rng.uniform(2.0, 9.0) * area_mm2 / 400.0))
        flecks = deposit(
            (h, w),
            rng,
            count=fleck_n,
            length_px=0.42 * ppm,
            width_px=max(0.045 * ppm, 0.6),
            length_sigma=0.6,
            kappa=0.0,
            persistence=1.0,
            kinks_per_length=2.0,
            step_px=step_px,
        )
        hue = rng.uniform(-28.0, 28.0, size=2)
        a = np.clip(gaussian_blur(flecks, 0.5) * 0.9, 0.0, 0.6)[..., None]
        col = lab_to_linear_rgb(
            base_lab + np.asarray([-30.0, float(hue[0]), float(hue[1])])
        )
        rgb = a * col + (1.0 - a) * rgb

    # --- Height -----------------------------------------------------------
    # Built in micrometres and converted to pixel-equivalent units so that
    # gradients are true surface slopes at any render size. ``sq_um``
    # calibrates the *roughness* band only -- cockle and creases are waviness
    # and sit outside it, exactly as Sq is defined in surface metrology. The
    # weights below are then set so RMS slope lands in the measured 2-7 degree
    # band, which is the number that actually decides whether it reads as
    # paper.
    floc = gaussian_blur(mass, max(1.0, 0.35 * ppm))
    floc = (floc - float(floc.mean())) / (float(floc.std()) + 1e-6)
    if out is not None:
        out["mass"] = mass
        out["formation"] = floc
    # Fibre crowns and crossings: the top of the stack in each column, taken as
    # a soft maximum over the depth slabs rather than as a saturating film of the
    # topmost one (see :func:`_stack_top` for why the old form was backwards).
    surface = _stack_top(slab_masses, _STACK_SOFTNESS)
    # The perceptually load-bearing band is 0.2-1.0 mm -- the "tooth" you feel
    # with a fingernail. A single high-frequency octave at 40-120 um sits in
    # the sensor-noise band instead and reads as digital grain, which is what
    # the previous version did.
    #
    # One draw per near-surface fibre layer, combined by the same maximum: each
    # layer of the felt has its own tooth, and the sheet's tooth is the highest
    # of them. This costs a few extra FFTs and is what puts the extreme-value
    # asymmetry in the band that carries most of the variance -- putting it only
    # in ``surface``, at a weight of 0.22, would dilute it by a factor of forty.
    tooth = _stack_top(
        [
            matern_field(
                (h, w),
                rng,
                corr_px=max(2.0, stock["tooth_mm"] * ppm),
                nu=0.35,
                aniso=1.0 + 0.15 * md,
                angle=machine_dir,
            )
            for _ in range(_CROWN_LAYERS)
        ],
        _STACK_SOFTNESS,
    )
    micro = band_field((h, w), rng, centre_px=max(1.6, 0.055 * ppm), octaves_wide=0.9)

    # Weights set so the 0.2-1.0 mm tooth band carries most of the variance,
    # as measured surfaces do. The very high-frequency terms (fibre crowns,
    # micro grain) are what drive RMS *slope*, so they stay small: real paper
    # slopes are only 2-7 degrees, and overshooting them is precisely what
    # makes procedural paper read as leather. Coated stock is the same weights
    # with the two fibre terms cut and the micro term gone entirely, which is
    # what an 80% height suppression and a 50 um low-pass amount to.
    w_tooth, w_floc, w_surface, w_micro = stock["rough_w"]
    rough = (
        tooth * np.float32(w_tooth)
        + floc * np.float32(w_floc)
        + surface * np.float32(w_surface)
        + micro * np.float32(w_micro)
    )
    asperity = float(stock["asperity"])
    if abs(asperity) > 1e-6:
        # ``asperity`` is the population's *depth*, in the same units as
        # ``rough_w`` above -- it saturates, so this is a height and not a
        # variance weight.
        rough = rough + _asperities(
            surface,
            np.sign(asperity),
            _ASPERITY_AREA,
            _ASPERITY_SIZE_UM * 1e-3 * ppm,
        ) * np.float32(abs(asperity))
    if felt_field is not None and felt_spec is not None:
        # Fines pack into the inter-fibre valleys and stand slightly proud of
        # them, so the felt is relief as well as mass and albedo.
        rough = rough + felt_field * np.float32(felt_spec["height"])
    rough = rough / np.float32(float(rough.std()) + 1e-6)
    # The blend of several fields is close to Gaussian by the central limit
    # theorem however skewed its parts are, so the stack's marginal is imposed
    # here, on the assembled band, rather than being left to survive the blend.
    rough = _skew_warp(rough, float(stock["stack_skew"]))
    sq_target = float(stock["sq_um"] * rng.uniform(0.85, 1.2))
    height_um = rough * np.float32(sq_target)

    # --- Calendering ------------------------------------------------------
    # Peaks crushed, valleys kept, and the same map used for all three of the
    # nip's consequences. It applies to the roughness band alone: calendering is
    # micro-topography and a nip does not iron out a fold or a cockle blister,
    # which is why this sits above the waviness terms rather than below them.
    calender_k = float(stock["calender_k"])
    if calender_k > 1e-3:
        height_um, push_um = _calender_clip(height_um, calender_k)
        # The clip removes variance, and how much depends on the seed's own draw,
        # so rescale onto the Sq target instead of trusting the operator's output
        # scale. The densification map is rescaled with it to stay in micrometres.
        gain = np.float32(sq_target / (float(height_um.std()) + 1e-6))
        height_um = height_um * gain
        push_um = push_um * gain
    else:
        push_um = np.zeros_like(height_um)
    densified = np.clip(push_um / np.float32(_DENSIFY_SQ * sq_target), 0.0, 1.0).astype(
        np.float32
    )

    # Wire marks are waviness, not roughness: they sit outside the Sq-calibrated
    # band, alongside cockle, which is why they are added in micrometres here
    # rather than blended into ``rough`` above.
    # The pitches actually used come back for reporting; :func:`wire_counts` is
    # the public way to ask what they will be for a given crop.
    if stock["wire"] is not None:
        marks, _laid_mm, _chain_mm = _wire_marks((h, w), rng, stock["wire"], ppm, floc)
        height_um = height_um + marks

    cockle = matern_field(
        (h, w),
        rng,
        corr_px=max(8.0, float(rng.uniform(5.0, 13.0)) * ppm),
        # Cockle is a buckling mode, so it is bending-stiffness limited: the
        # elastic cost of curvature goes as k^4, which suppresses short
        # wavelengths far harder than the nu = 1.2 this used to use. That matters
        # for more than looks -- at nu = 1.2 a 22-70 um cockle leaked enough
        # sub-millimetre energy to supply 85-93% of the variance in the 1 mm
        # roughness band, so the band measured its own waviness rather than the
        # roughness the band exists to isolate.
        nu=3.0,
        # Cockle is moisture-driven and follows the sheet's own grain, so a
        # sheet without a grain direction cockles into round hollows.
        aniso=1.0 + (float(rng.uniform(1.0, 1.8)) - 1.0) * md,
        angle=machine_dir,
    )
    height_um = height_um + cockle * np.float32(rng.uniform(*stock["cockle_um"]))

    # Creases are an event, not a default: most sheets are flat, some are
    # cockled, a few have been folded. Amplitude is in micrometres against a
    # 5-10 um Sq, so even 60 um is already a strong fold.
    crease_amount = params.get("creases")
    if crease_amount is None:
        crease_amount = (
            float(rng.uniform(0.35, 1.0))
            if rng.random() < float(stock["crease_chance"])
            else 0.0
        )
    crease_amount = float(crease_amount)
    if crease_amount > 0.02:
        # Per-stock scale: how far a fold carries depends on the stock, not just
        # on how hard it was folded (see ``crease_scale`` in :data:`STOCKS`).
        height_um = height_um + _creases((h, w), rng, ppm) * np.float32(
            crease_amount
            * rng.uniform(35.0, 130.0)
            * float(stock.get("crease_scale", 1.0))
        )

    height = (height_um * np.float32(um)).astype(np.float32)

    # --- Calender blackening ----------------------------------------------
    # A crushed spot is locally densified, and densification collapses the
    # fibre-air interfaces that do the scattering: the light-scattering
    # coefficient falls, so the spot turns marginally translucent and reflects
    # less. It is glossier *and* darker at the same time -- the classic
    # "calender blackening" defect (AIC Conservation Wiki, *Calendered*:
    # over-high nip load on moist paper costs opacity). Magnitudes are reasoned
    # from the mechanism, not measured.
    if calender_k > 1e-3:
        rgb = rgb * (1.0 - np.float32(_BLACKENING) * densified[..., None])

    # --- Gloss ------------------------------------------------------------
    # The same map again: the spots the nip flattened are the only near-specular
    # micro-facets on the sheet, which is what the "shiny freckling" of
    # machine-finished stock is, and the one structural feature that shows up
    # better in reflection than in transmission. Centred on its own mean so the
    # stock's sheen range still means what it says.
    sheen = float(rng.uniform(*stock["sheen"]))
    sheen_field = np.float32(sheen) * (
        1.0
        + np.float32(_CALENDER_GLOSS)
        * (densified - np.float32(float(densified.mean())))
    )
    if mottle is not None and coating is not None:
        # The mottle belongs here, at a weight an order of magnitude above what
        # it gets in the albedo. Floored rather than clipped symmetrically: a
        # dull patch of coating is still coating, and never a matte hole.
        sheen_field = sheen_field * np.maximum(
            1.0 + np.float32(coating["gloss"]) * mottle, np.float32(0.25)
        )

    lit = shade_translucent(
        rgb,
        height,
        light_dir=(-0.5, -0.62, float(rng.uniform(0.5, 0.72))),
        diffuse_sigma=max(0.8, psf_px),
        diffuse_strength=float(params.get("normal_strength", 1.0)),
        # Height is in real units, so the sharp normal takes physical
        # strength; exaggerating it here would undo the slope calibration.
        spec_strength=float(rng.uniform(1.0, 1.35)),
        wrap=float(rng.uniform(0.35, 0.5)),
        roughness_deg=float(stock["roughness_deg"]),
        sheen=sheen_field,
        sheen_exponent=float(rng.uniform(*stock["sheen_exp"])),
        ambient=float(params.get("ambient", 0.62)),
        occlusion=0.22,
    )
    return lit, height_um, ppm
