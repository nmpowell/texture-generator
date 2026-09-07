"""Tests for wood colour: the species value, the finish, and per-board spread.

Wood colour is specified as measured CIELAB (:data:`..materials.wood.SPECIES`),
so it is checkable in a way a hand-picked RGB palette never was: render a board
and the mean must come back as the colour it was asked for. Everything here is
measured in Lab rather than in display levels, because that is the unit the
source figures are quoted in -- a "few RGB levels" is meaningless on its own,
while a dL* or a dC* is the same quantity as the measurement.
"""

from __future__ import annotations

import numpy as np

from texture_generators.core.colour import lab_to_srgb, srgb_to_lab
from texture_generators.core.shading import shade
from texture_generators.materials import wood
from texture_generators.materials.wood import (
    FINISHES,
    SPECIES,
    _darken_lstar,
    _finish_delta,
    _finish_lab,
)

# A "clear heartwood" board: no finish, no board-to-board offset, and neither of
# the two features that are deliberate *departures* from the species colour
# (sapwood, knots), so what is left is the species value itself.
CLEAR = dict(finish="none", colour_variation=0.0, sapwood=0.0, knots=0.0)


def _mean_srgb(size: int = 192, seed: int = 0, variant: str = "board", **params):
    img = wood.generate((size, size), np.random.default_rng(seed), variant, **params)
    return img.reshape(-1, 3).mean(axis=0)


def _strip_means(img: np.ndarray) -> np.ndarray:
    """Split a ``planks`` render at its gap lines; mean sRGB of each strip.

    Gap lines are the one feature that darkens the albedo outright (to 28%), so
    a row mean well below the median row is a gap and everything between two
    gaps is one board. Two rows are dropped either side to clear the bevel.
    """
    row = img.mean(axis=(1, 2))
    gap = row < 0.72 * float(np.median(row))
    out, start = [], 0
    for i in range(1, len(row) + 1):
        if i == len(row) or gap[i]:
            if not gap[start] and i - start >= 6:
                out.append(img[start + 2 : i - 2].reshape(-1, 3).mean(axis=0))
            start = i + 1
        elif gap[start]:
            start = i
    return np.asarray(out)


def test_rendered_mean_is_the_published_species_colour() -> None:
    """A clear board must render, on average, the L*a*b* it was specified as.

    This is the whole point of specifying colour as a measurement. It needs two
    things to hold at once: the albedo's mean has to be the species value
    (``_recentre``) *and* the shading pass has to normalise its lighting
    (``normalise=True``), because a light rig that averages to 0.8 turns a
    measured colour into 80% of a measured colour. Before either was in place
    every species rendered 25-60% below its target -- cherry at 40% of its
    target lightness -- and the blue channel was crushed hardest of the three,
    so the palette was over-saturated as well as dark.
    """
    for species in sorted(SPECIES):
        target = lab_to_srgb(np.asarray(SPECIES[species]["lab"], dtype=np.float64))
        for seed in (0, 1):
            mean = _mean_srgb(seed=seed, species=species, **CLEAR)
            err = float(np.abs(mean - target).max()) * 255.0
            assert err < 4.0, f"{species} seed {seed}: {err:.1f}/255 from target"


def test_pine_and_walnut_stay_far_apart() -> None:
    """The species must not be interchangeable once rendered.

    A palette can be faithful in the mean and still collapse in practice if the
    render pulls everything towards one tone. Pine and walnut are the extremes
    of this set, 36 L* apart as published; a render that loses most of that has
    lost the species distinction whatever its mean says.
    """
    pine = srgb_to_lab(_mean_srgb(species="pine", **CLEAR))
    walnut = srgb_to_lab(_mean_srgb(species="walnut", **CLEAR))
    assert pine[0] - walnut[0] > 30.0


def test_finish_takes_lightness_down_and_chroma_up() -> None:
    """A finish deepens *and* saturates -- the thing an RGB multiply cannot do.

    Index-matching the air/cell-wall interface removes a white surface veil, so
    the same pigment is seen through less scattering: L* falls and C* rises
    together. A multiply in RGB would take both down.
    """
    for species in ("oak", "walnut", "maple"):
        base = srgb_to_lab(_mean_srgb(species=species, **CLEAR))
        base_c = float(np.hypot(base[1], base[2]))
        for finish in ("oil", "polyurethane", "acrylic"):
            lab = srgb_to_lab(
                _mean_srgb(species=species, **{**CLEAR, "finish": finish})
            )
            dl = lab[0] - base[0]
            dc = float(np.hypot(lab[1], lab[2])) - base_c
            assert dl < -1.5, f"{species}/{finish}: dL* {dl:.2f} not a deepening"
            assert dc > 1.5, f"{species}/{finish}: dC* {dc:.2f} not a saturation"
            # Inside the ranges the finish classes are specified over, with a
            # little room for the chroma the warm shift adds on top.
            assert -11.0 < dl < 0.0
            assert dc < 13.0

    # And the strength ranks by finish class: oil wets the cell wall itself,
    # waterborne acrylic barely touches it.
    oil = srgb_to_lab(_mean_srgb(species="oak", **{**CLEAR, "finish": "oil"}))
    acr = srgb_to_lab(_mean_srgb(species="oak", **{**CLEAR, "finish": "acrylic"}))
    assert oil[0] < acr[0]


def test_finish_holds_the_hue_it_deepens() -> None:
    """dC* moves (a*, b*) along their own hue angle, so mahogany stays brown.

    Adding the chroma to a* and b* separately would rotate the hue, and adding
    it to a* alone is how procedural mahogany ends up fire-engine crimson.
    """
    for species in ("mahogany", "cherry", "oak"):
        lab = np.asarray(SPECIES[species]["lab"], dtype=np.float64)
        rng = np.random.default_rng(3)
        fin = _finish_lab(lab, _finish_delta("oil", rng))
        h0 = float(np.degrees(np.arctan2(lab[2], lab[1])))
        h1 = float(np.degrees(np.arctan2(fin[2], fin[1])))
        assert abs(h1 - h0) < 8.0, f"{species}: hue moved {h1 - h0:.1f} degrees"
        assert np.hypot(fin[1], fin[2]) > np.hypot(lab[1], lab[2])


def test_each_plank_gets_its_own_colour() -> None:
    """No two boards in a panel may share a colour.

    A floor of identically coloured boards is impossible, and it is the loudest
    "printed" tell at assembly scale. Each strip draws its own Lab offset, so
    the strip means must separate; with the spread turned off they must collapse
    back onto one colour, which is what shows the spread is doing the work
    rather than the ring figure or the tonal drift.
    """
    spreads = {}
    for variation in (0.0, 1.0):
        worst = 0.0
        for seed in range(4):
            img = wood.generate(
                (256, 256),
                np.random.default_rng(seed),
                "planks",
                species="oak",
                **{**CLEAR, "colour_variation": variation},
            )
            strips = _strip_means(img)
            assert len(strips) >= 3, f"only {len(strips)} strips found"
            spread = strips.max(axis=0) - strips.min(axis=0)
            worst = max(worst, float(spread.max()))
        spreads[variation] = worst * 255.0
    assert spreads[0.0] < 3.0, f"boards differ without a spread: {spreads[0.0]:.1f}/255"
    assert spreads[1.0] > 6.0, f"boards barely differ: {spreads[1.0]:.1f}/255"


def test_cherry_ages_as_a_panel_not_as_boards() -> None:
    """Boards laid together have aged together, so ageing is a panel property.

    Cherry's fresh and 12-month values are ~16 L* apart -- far outside the +/-4
    board-to-board spread -- so drawing the ageing per board would put a
    day-old board next to a year-old one in the same floor. The per-board spread
    must still be there, which is what the bounds are for.
    """
    for age in (0.0, 1.0):
        img = wood.generate(
            (256, 256),
            np.random.default_rng(2),
            "planks",
            species="cherry",
            finish="none",
            sapwood=0.0,
            knots=0.0,
            age=age,
        )
        strips = srgb_to_lab(_strip_means(img))
        assert len(strips) >= 3
        spread = float(strips[:, 0].max() - strips[:, 0].min())
        assert 0.5 < spread < 10.0, f"age {age}: strip L* spread {spread:.1f}"

    # And the two ends of the journey are a long way apart. (Measured on the
    # board colour rather than the render, because ``colour_variation=0`` -- the
    # only way to pin the render -- is also what turns the ageing off.)
    fresh = wood._board_lab("cherry", np.random.default_rng(0), 1.0, 0.0)
    old = wood._board_lab("cherry", np.random.default_rng(0), 1.0, 1.0)
    assert fresh[0] - old[0] > 10.0
    assert np.hypot(old[1], old[2]) > np.hypot(fresh[1], fresh[2])


def test_sapwood_band_is_dramatically_paler_than_the_heartwood() -> None:
    """Walnut and cherry sapwood is nearly white against their heartwood.

    Boards are normally cut to exclude it, so it is an occasional draw -- but
    where it lands it has to be the real split (walnut: L* ~78 sapwood against
    L* 40 heartwood) rather than a gentle lightening of one edge.
    """
    for species in ("walnut", "cherry"):
        heart = float(np.asarray(SPECIES[species]["lab"])[0])
        img = wood.generate(
            (256, 256),
            np.random.default_rng(0),
            "board",
            species=species,
            **{**CLEAR, "sapwood": 1.0},
        )
        bands = srgb_to_lab(img.reshape(32, -1, 3).mean(axis=1))
        assert bands[:, 0].max() - heart > 8.0
        # And it is a band, not a wash over the whole face: most of the board is
        # still heartwood.
        assert float(np.median(bands[:, 0])) - heart < 6.0


def test_pore_darkening_lands_on_the_lstar_it_asks_for() -> None:
    """An open pore is a dL* of 15-30, and must not arrive as 2x that.

    ``_darken_lstar`` scales the tristimulus Y, which is a *linear* quantity.
    Run on sRGB display values as though they were linear -- which is what this
    did -- the requested drop is delivered through the encoding's gamma on top,
    so a nominal dL* 30 measured 43 on a light face and the streaks read as
    near-black ink flecks instead of soft grooves.
    """
    for base in ((0.72, 0.60, 0.47), (0.45, 0.35, 0.27)):
        colour = np.asarray(base, dtype=np.float32)[None, None, :]
        before = float(srgb_to_lab(colour.reshape(3))[0])
        for dl in (15.0, 22.0, 30.0):
            got = _darken_lstar(colour, np.full((1, 1), dl, dtype=np.float32))
            after = float(srgb_to_lab(got.reshape(3))[0])
            assert abs((before - after) - dl) < 0.6


def test_pore_dl_star_stays_in_the_specified_band() -> None:
    """No species may ask for more than the dL* 15-30 an open pore is worth."""
    for species, anat in wood.ANATOMY.items():
        if anat["pore_class"] == "softwood":
            continue
        lo, hi = anat["dl_star"]
        assert 15.0 <= lo <= hi <= 30.0, f"{species}: dl_star {(lo, hi)}"


def test_shade_normalise_is_opt_in() -> None:
    """The lighting normalisation must not touch the materials tuned without it.

    Metal and plastic hand-tuned their albedo against this light rig *including*
    whatever it averages to, so renormalising it would shift every one of them.
    Wood asks for it because its albedo is a measured colour.
    """
    rng = np.random.default_rng(0)
    albedo = np.full((32, 32, 3), 0.5, dtype=np.float32)
    height = rng.normal(0.0, 0.3, size=(32, 32)).astype(np.float32)

    plain = shade(albedo, height, ambient=0.6)
    again = shade(albedo, height, ambient=0.6, normalise=False)
    assert np.array_equal(plain, again)

    normed = shade(albedo, height, ambient=0.6, normalise=True)
    assert not np.allclose(plain, normed)
    # With a flat albedo and no specular, normalising makes the render's mean the
    # albedo exactly -- that is the property wood needs.
    assert abs(float(normed.mean()) - 0.5) < 1e-3
    assert float(plain.mean()) < 0.5


def test_finish_shares_are_a_distribution_over_finished_wood() -> None:
    """Almost all wood one sees is finished, so bare wood is the rare draw."""
    total = sum(f["p"] for f in FINISHES.values())
    assert abs(total - 1.0) < 1e-9
    assert FINISHES["none"]["p"] < 0.1
