"""Tests for ray fleck: the figure a quartersawn hardwood face carries.

Ray fleck is two claims, and they are tested separately because only the first
one is visible in a still frame:

* a *population* of lens-shaped sheets, with the length distribution, the width
  and the areal coverage the anatomy reports -- and gated hard, so a species or
  a cut that should show nothing shows nothing;
* a **90 degree in-plane rotation of the fibre tangent** inside each fleck,
  which is the whole reason ray fleck flashes rather than sitting there as pale
  dashes. That one is checked on the tangent field directly, since it is a
  behaviour of the reflection and not a colour.

The coverage measures here read the *drawn* mask rather than the geometric area
the sampler asked for: most of this population is narrower than a pixel at any
sane render size, so what the eye gets is the floored-and-faded mask (see
:data:`~texture_generators.materials.wood.FLECK_MIN_WIDTH_PX`), and that is the
thing worth pinning.
"""

from __future__ import annotations

import numpy as np

from texture_generators.core.noise import grid_coords
from texture_generators.core.warp import warp
from texture_generators.materials import wood
from texture_generators.materials.wood import (
    CUTS,
    FLECK_ASPECT,
    FLECK_LENGTH_MM,
    FLECK_WIDTH_MM,
    RAY_FLECK_COVERAGE,
    _board_fields,
    _fibre_frame,
    _fibre_tangents,
    _fleck_lengths_mm,
    _ray_fleck,
)

SIZE = 384
PX_PER_MM = SIZE / 225.0


def _fleck(species: str, seed: int, *, size: int = SIZE, **kw) -> dict:
    """One fleck population on a flat (unwarped) face, so geometry is testable."""
    phi = np.zeros((size, size), dtype=np.float32)
    return _ray_fleck(
        (size, size),
        np.random.default_rng(seed),
        species,
        phi,
        tilt=0.0,
        along_x=True,
        px_per_mm=size / 225.0,
        **kw,
    )


def _coverage(fleck: dict) -> float:
    return float((fleck["mask"] > 0.5).mean())


def _cross_grain_fraction(species: str, cut: str, seed: int = 0) -> float:
    """Share of the board whose in-plane fibre tangent runs *across* the grain.

    The fleck rotation is the only thing on a board that can turn the tangent
    more than 45 degrees off the grain axis -- the deflection term is a few
    degrees and the tilt is capped at six -- so on an ``along_x`` board this is a
    fleck detector that goes through the whole of :func:`_board_fields` rather
    than reaching into the fleck layer.
    """
    _, _, tangent, _ = _board_fields(
        (SIZE, SIZE),
        np.random.default_rng(seed),
        species,
        finish="oil",
        knot_p=0.0,
        sap_p=0.0,
        cut=cut,
        along_x=True,
        px_per_mm=PX_PER_MM,
    )
    return float((np.abs(tangent[..., 1]) > np.abs(tangent[..., 0])).mean())


# --- The population ----------------------------------------------------------


def test_quartersawn_oak_fleck_coverage_is_in_the_reported_band() -> None:
    """10-25% of a true quartersawn oak face, per :data:`RAY_FLECK_COVERAGE`.

    Oak is the exemplar: it is a two-ray-size species, and its broad
    multiseriate rays are what make the fleck. Sampling slack either side, since
    a finite scatter of a right-skewed population does not land on its own mean.
    """
    covs = [_coverage(_fleck("oak", seed)) for seed in range(8)]
    assert all(0.07 <= c <= 0.30 for c in covs), covs
    assert 0.10 <= float(np.mean(covs)) <= 0.25, covs


def test_fleck_is_gated_on_the_species_two_ray_sizes() -> None:
    """Oak dramatic, hard maple present but subtler, everything else faint.

    Ray *fleck* needs broad multiseriate rays, not just rays: the narrow-rayed
    species here have as much ray tissue and almost no figure to show for it.
    """
    covs = {
        sp: float(np.mean([_coverage(_fleck(sp, s)) for s in range(4)]))
        for sp in ("oak", "maple", "ash", "walnut")
    }
    assert covs["oak"] > 2.5 * covs["maple"] > 0.0, covs
    assert covs["maple"] > covs["ash"], covs
    assert covs["ash"] < 0.02 and covs["walnut"] < 0.02, covs


def test_pine_shows_no_fleck_at_all() -> None:
    """A softwood's rays are uniseriate: invisible at any render size.

    That absence is the same point the missing vessels make -- it is part of what
    reads as softwood.
    """
    assert RAY_FLECK_COVERAGE["pine"] == (0.0, 0.0)
    for seed in range(3):
        f = _fleck("pine", seed)
        assert float(f["mask"].max()) == 0.0
        assert f["length_mm"].size == 0
        assert _cross_grain_fraction("pine", "quartersawn", seed) == 0.0


def test_only_the_quartersawn_cut_shows_fleck() -> None:
    """Flatsawn and cathedral faces cut *across* the rays, so they show none.

    Measured through :func:`_board_fields` on the tangent field, i.e. on the
    thing the renderer actually consumes: a quartersawn oak board is
    unmistakable and the other two cuts are identically zero.
    """
    for seed in range(3):
        assert _cross_grain_fraction("oak", "quartersawn", seed) > 0.05
        assert _cross_grain_fraction("oak", "flatsawn", seed) == 0.0
        assert _cross_grain_fraction("oak", "cathedral", seed) == 0.0


def test_fleck_length_is_right_skewed() -> None:
    """Mode 5-8 mm, bulk under 25 mm, and a tail to the exceptional flecks.

    A uniform-length population reads as a pattern however well each fleck is
    drawn, so the skew is the property worth pinning: mean above median above
    mode, and a few percent of the population past 25 mm.
    """
    lengths = _fleck_lengths_mm(20000, np.random.default_rng(2), mode_mm=6.5)
    mean, median = float(lengths.mean()), float(np.median(lengths))
    skew = float(((lengths - mean) ** 3).mean() / lengths.std() ** 3)
    assert mean > median > 6.5, (mean, median)
    assert skew > 1.0, skew
    assert 0.01 < float((lengths > 25.0).mean()) < 0.12
    assert float(lengths.max()) > 40.0
    assert FLECK_LENGTH_MM[0] <= float(lengths.min())
    assert float(lengths.max()) <= FLECK_LENGTH_MM[1]


def test_fleck_width_and_aspect_stay_in_the_anatomy_band() -> None:
    """0.2-0.8 mm wide for the bulk, and 10:1 to 40:1 in aspect.

    Width is not an independent draw -- it comes from the aspect ratio, which is
    the reported quantity -- so the clip on width must not be able to take the
    aspect outside the band. It cannot, and this is the check: the narrowest
    fleck is 2 mm long, which at the 0.2 mm floor is exactly 10:1.
    """
    f = _fleck("oak", 0)
    lengths, widths = f["length_mm"], f["width_mm"]
    assert widths.size > 200
    assert FLECK_WIDTH_MM[0] <= float(widths.min())
    assert float(widths.max()) <= FLECK_WIDTH_MM[1]
    assert 0.2 <= float(np.median(widths)) <= 0.8
    aspect = lengths / widths
    assert FLECK_ASPECT[0] - 0.01 <= float(aspect.min())
    assert float(aspect.max()) <= FLECK_ASPECT[1] + 0.01


def test_fleck_albedo_contrast_stays_modest_and_signed() -> None:
    """dL* 3-8, and mostly -- but not only -- paler than the wood around it.

    The flash is the tangent rotation, not the pigment. A fleck drawn with a
    pore's contrast reads as chalk, and a population that is all one sign loses
    the flicker.
    """
    f = _fleck("oak", 0)
    core = f["mask"] > 0.9
    dl = f["dl"][core]
    assert float(np.median(np.abs(dl))) <= 8.0
    assert float(np.percentile(np.abs(dl), 95)) <= 8.5
    assert 0.5 < float((dl > 0).mean()) < 0.95


# --- The part that makes it flash --------------------------------------------


def test_the_tangent_is_rotated_ninety_degrees_inside_a_fleck() -> None:
    """Rays run radially, so the ray's fibre is at right angles to the wood's.

    Rendered from the same seed with and without the fleck mask, so the only
    difference between the two tangent fields is the rotation. Inside a fleck it
    is 90 degrees; outside it is exactly nothing, which is the other half of the
    claim -- the fleck must not perturb the grain field around it.
    """
    x, y = grid_coords((SIZE, SIZE))
    wu, wv = warp(
        (x, y), np.random.default_rng(0), amp=0.02, freq=(2.0, 2.0), octaves=3
    )
    phi, _ = _fibre_frame(x, y, wu, wv, along_x=True, px_per_unit=SIZE)
    mask = _ray_fleck(
        (SIZE, SIZE),
        np.random.default_rng(1),
        "oak",
        phi,
        tilt=0.0,
        along_x=True,
        px_per_mm=PX_PER_MM,
    )["mask"]

    def tangents(fleck):
        return _fibre_tangents(
            x,
            y,
            wu,
            wv,
            np.random.default_rng(5),
            tilt=0.0,
            along_x=True,
            px_per_unit=SIZE,
            mm_per_unit=225.0,
            fleck=fleck,
        )

    plain, flecked = tangents(None), tangents(mask)
    delta = np.degrees(
        np.abs(
            np.arctan2(flecked[..., 1], flecked[..., 0])
            - np.arctan2(plain[..., 1], plain[..., 0])
        )
    )
    core = mask > 0.99
    assert int(core.sum()) > 1000
    assert 88.0 <= float(np.median(delta[core])) <= 92.0
    assert float(np.abs(delta[mask <= 0.0]).max()) == 0.0
    # A rotation, so the field is still unit length -- the property every other
    # consumer of the tangents relies on.
    assert np.allclose(np.linalg.norm(flecked, axis=-1), 1.0, atol=2e-3)


def test_the_fleck_flashes_the_opposite_way_to_the_wood_around_it() -> None:
    """Move the light, and fleck and wood go in *opposite* directions.

    This is the behaviour the whole feature is for, and a still frame cannot show
    it: the fleck's albedo contrast is a few L*, so if it only ever sat that much
    paler it would be pale dashes. Because its tangent is crosswise, the same
    light that brightens the wood's fibre lobe darkens the fleck's -- which is
    what "chatoyance, not pigment" means, and it is worth an order of magnitude
    more luminance than the albedo term.
    """
    lum = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    captured: dict = {}
    plain = wood._ray_fleck

    def spy(*args, **kwargs):
        out = plain(*args, **kwargs)
        captured["mask"] = out["mask"]
        return out

    def render(light):
        return wood.generate(
            (256, 256),
            np.random.default_rng(11),
            "board",
            species="oak",
            cut="quartersawn",
            finish="oil",
            knots=0.0,
            sapwood=0.0,
            light_dir=light,
        )

    wood._ray_fleck = spy
    try:
        first = render((-0.5, -0.55, 0.78))
        mask = captured["mask"].copy()
        second = render((-0.75, 0.1, 0.65))
    finally:
        wood._ray_fleck = plain

    core, wood_only = mask > 0.9, mask <= 0.0
    delta = (first @ lum) - (second @ lum)
    on_fleck, on_wood = float(delta[core].mean()), float(delta[wood_only].mean())
    assert on_fleck * on_wood < 0.0, (on_fleck, on_wood)
    assert abs(on_fleck - on_wood) > 0.02, (on_fleck, on_wood)


# --- The cut ------------------------------------------------------------------


def test_every_cut_renders_in_both_variants() -> None:
    for cut in CUTS:
        for variant in wood.VARIANTS:
            img = wood.generate(
                (128, 128),
                np.random.default_rng(4),
                variant,
                species="oak",
                cut=cut,
            )
            assert img.shape == (128, 128, 3)
            assert np.isfinite(img).all()
            assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_an_unknown_cut_is_refused() -> None:
    for bad in ("riftsawn", "quarter"):
        try:
            wood.generate((64, 64), np.random.default_rng(0), "board", cut=bad)
        except ValueError as exc:
            assert "cut" in str(exc)
        else:  # pragma: no cover - the point of the test
            raise AssertionError(f"cut {bad!r} was accepted")


def test_the_cut_is_a_ring_geometry_and_not_only_a_fleck_gate() -> None:
    """The three cuts are three different faces off the same seed.

    Quartersawn's pith sits entirely beyond flatsawn's -- that is the whole of
    what makes its rings near-straight and parallel rather than arcs -- so the
    two ranges must not overlap, and the same board asked for each of the three
    cuts must come out visibly different. Rendered on pine, which carries no
    fleck, so this measures the ring geometry alone.
    """
    assert wood.PITH_DIST["quartersawn"][0] >= wood.PITH_DIST["flatsawn"][1]
    assert set(wood.CUT_P) == set(CUTS)
    assert abs(sum(wood.CUT_P.values()) - 1.0) < 1e-9

    faces = {}
    for cut in CUTS:
        albedo, _, _, _ = _board_fields(
            (SIZE, SIZE),
            np.random.default_rng(3),
            "pine",
            finish="oil",
            knot_p=0.0,
            sap_p=0.0,
            cut=cut,
            along_x=True,
            px_per_mm=PX_PER_MM,
        )
        faces[cut] = albedo[..., 1]
    for a, b in (("cathedral", "flatsawn"), ("flatsawn", "quartersawn")):
        assert float(np.abs(faces[a] - faces[b]).mean()) > 0.01, (a, b)
