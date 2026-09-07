"""Thin-film interference on metal: the table, the ladder, and the promise.

The promise is that the film is purely **additive**: ``brushed``, ``radial`` and
``polished`` render byte-for-byte as they did before the film existed. The
mechanism for keeping it is the ``film=None`` short-circuit in
:func:`metal.generate`, which returns before any of the film code can draw from
the rng -- so the test that matters most here is the one that pins the classic
variants' bytes against a stored digest.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from texture_generators import generate
from texture_generators.core import film
from texture_generators.core.colour import linear_rgb_to_lab
from texture_generators.materials import metal

CLASSIC = ("brushed", "radial", "polished")
FILMED = ("heat_tinted", "oil_film", "anodised_titanium")

# Digests of the three unfilmed variants, captured *before* the film existed.
# If one of these moves, the additive promise is broken -- do not re-bless them.
GOLDEN = {
    "brushed-1-64": "9ac175ae4fdad3e7",
    "radial-1-64": "9eb45e911b1e6c82",
    "polished-1-64": "3223d36389c8c622",
}


def _digest(variant: str, seed: int, size: int) -> str:
    arr = np.asarray(
        generate("metal", size=size, seed=seed, variant=variant), dtype=np.uint8
    )
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _hue_chroma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CIELAB hue angle in degrees and chroma for an array of linear RGB."""
    lab = linear_rgb_to_lab(np.clip(rgb, 0.0, 1.0))
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return hue, np.hypot(lab[..., 1], lab[..., 2])


def _sweep_colours(system: str = "oxide", lo: float = 80.0, hi: float = 400.0):
    """Luminance-normalised film colour over a thickness sweep, at normal incidence.

    Normalising to constant luminance is what isolates the *hue* the film adds
    from how bright the underlying metal happens to be, which is the quantity
    the published sequence is quoted in.
    """
    nm = np.linspace(lo, hi, 400)
    tint = film.film_tint(system, nm, np.ones_like(nm))
    lum = tint @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return nm, tint * (0.35 / np.maximum(lum, 1e-6))[:, None]


# --------------------------------------------------------------------------
# The additive promise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", CLASSIC)
def test_unfilmed_variants_are_byte_identical(variant: str) -> None:
    key = f"{variant}-1-64"
    assert _digest(variant, 1, 64) == GOLDEN[key], (
        f"{variant} changed; the film must not perturb the classic finishes"
    )


@pytest.mark.parametrize("variant", CLASSIC)
def test_film_none_draws_nothing_extra(variant: str) -> None:
    """Passing ``film=None`` explicitly is the same stream as omitting it."""
    a = metal.generate((48, 48), np.random.default_rng(7), variant)
    b = metal.generate((48, 48), np.random.default_rng(7), variant, film=None)
    assert np.array_equal(a, b)


def test_new_variants_are_registered() -> None:
    assert list(metal.VARIANTS[:3]) == list(CLASSIC)
    for name in FILMED:
        assert name in metal.VARIANTS
        assert name in metal.FILM_VARIANTS


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


def test_table_shape_and_neutral_reference() -> None:
    for name in film.SYSTEMS:
        table = film.table_for(name)
        assert table.shape == (film.N_THICKNESS, film.N_ANGLE, 3)
        assert np.all(np.isfinite(table))
    # Zero thickness is the reference the tint divides by, so it must be
    # exactly neutral -- otherwise a film-free pixel would be tinted.
    tint = film.film_tint("oxide", np.zeros((4, 4), dtype=np.float32), 1.0)
    assert np.allclose(tint, 1.0, atol=1e-5)


def test_reflectance_is_periodic_in_inverse_lambda_not_a_ramp() -> None:
    """A thickness sweep must *cycle* through hues, not ramp monotonically.

    This is the aliasing failure the spectral integration exists to avoid: three
    RGB samples of an oscillation whose period shrinks with thickness produce a
    confident monotone ramp. A real film comes back round.
    """
    _nm, rgb = _sweep_colours("oxide", 0.0, 450.0)
    hue, chroma = _hue_chroma(rgb)
    coloured = chroma > 4.0
    # Unwrapped hue must travel more than a full turn over the sweep.
    travel = np.abs(np.diff(np.unwrap(np.radians(hue[coloured]))))
    assert np.degrees(travel.sum()) > 360.0

    # And each channel must be non-monotone: it goes up and it comes back down.
    for ch in range(3):
        d = np.diff(rgb[:, ch])
        assert (d > 0).any() and (d < 0).any()


def test_thick_film_desaturates() -> None:
    """Above ~1 um the fringes outrun the observer and the colour washes out.

    Earned by the spectral integration, not asserted: at 2 um the fringe spacing
    is ~52 nm against colour matching functions ~100 nm wide.
    """
    _, thin = _sweep_colours("oil", 150.0, 450.0)
    _, thick = _sweep_colours("oil", 1500.0, 2100.0)
    _, c_thin = _hue_chroma(thin)
    _, c_thick = _hue_chroma(thick)
    assert c_thick.mean() < 0.5 * c_thin.mean()


def test_verified_tempering_sequence_over_80_to_400_nm() -> None:
    """VERIFIED: brown -> blue -> gold -> purple -> green over 80-400 nm.

    "Colors changing from brown, blue, gold, purple to green are obtained, in
    this sequence, as the interference film thickness increases from 80 nm to
    400 nm" -- Surface & Coatings Technology, S0257897206003495.

    Checked as a *subsequence* of nearest-anchor labels: the model passes through
    intermediate hues between the five named ones (it is a continuous sweep, and
    the paper named landmarks, not a partition), so what is asserted is that the
    five appear and appear in that order.
    """
    anchors = {
        "brown": 55.0,
        "gold": 90.0,
        "green": 140.0,
        "blue": 235.0,
        "purple": 310.0,
    }
    _nm, rgb = _sweep_colours("oxide", 80.0, 400.0)
    hue, chroma = _hue_chroma(rgb)

    labels = []
    for h, c in zip(hue, chroma, strict=False):
        if c < 4.0:
            continue
        name = min(anchors, key=lambda k: abs((h - anchors[k] + 180.0) % 360.0 - 180.0))
        if not labels or labels[-1] != name:
            labels.append(name)

    want = ["brown", "blue", "gold", "purple", "green"]
    it = iter(labels)
    assert all(w in it for w in want), (
        f"expected {want} as a subsequence of the measured ladder, got {labels}"
    )


def test_angle_axis_shifts_the_colour() -> None:
    """Interference colour is a function of incidence angle, so it must move."""
    nm = np.full((32,), 120.0, dtype=np.float32)
    face_on = film.film_tint("oxide", nm, np.ones_like(nm))
    grazing = film.film_tint("oxide", nm, np.full_like(nm, 0.25))
    h0, _ = _hue_chroma(face_on)
    h1, _ = _hue_chroma(grazing)
    assert abs((h0[0] - h1[0] + 180.0) % 360.0 - 180.0) > 15.0


def test_aluminium_anodising_is_not_modelled() -> None:
    """Coloured anodised aluminium is dye in a porous 5-25 um film, not
    interference. It must not appear as a film system."""
    assert not any("alumin" in k for k in film.SYSTEMS)
    assert not any("alumin" in v for v in metal.FILM_VARIANTS)


# --------------------------------------------------------------------------
# The variants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", FILMED)
def test_filmed_variants_render_in_range(variant: str) -> None:
    out = metal.generate((96, 96), np.random.default_rng(3), variant)
    assert out.shape == (96, 96, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("variant", FILMED)
def test_filmed_variants_are_deterministic(variant: str) -> None:
    a = metal.generate((64, 64), np.random.default_rng(11), variant)
    b = metal.generate((64, 64), np.random.default_rng(11), variant)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("variant", FILMED)
def test_filmed_variants_carry_more_chroma_than_bare_metal(variant: str) -> None:
    """The point of the feature: filmed metal is coloured, bare metal is not."""
    filmed = metal.generate((160, 160), np.random.default_rng(5), variant)
    bare = metal.generate((160, 160), np.random.default_rng(5), "polished")
    _, c_filmed = _hue_chroma(filmed)
    _, c_bare = _hue_chroma(bare)
    assert np.median(c_filmed) > 2.0 * max(np.median(c_bare), 0.5)


def test_thickness_fields_stay_in_their_stated_range() -> None:
    for variant, spec in metal.FILM_VARIANTS.items():
        lo, hi = spec["nm"]
        field = metal._THICKNESS_FIELDS[spec["field"]](
            (96, 96), np.random.default_rng(2), lo, hi
        )
        assert field.min() >= lo - 1e-3, variant
        assert field.max() <= hi + 1e-3, variant
        # A field that does not vary cannot show a ladder.
        assert field.max() - field.min() > 0.25 * (hi - lo), variant


def test_oil_film_reaches_both_the_rainbow_and_the_grey() -> None:
    """A spill is coloured at its thin edges and neutral in the middle."""
    lo, hi = metal.FILM_VARIANTS["oil_film"]["nm"]
    field = metal._THICKNESS_FIELDS["spill"](
        (256, 256), np.random.default_rng(4), lo, hi
    )
    assert (field < 500.0).mean() > 0.05, "no thin, saturated edge"
    assert (field > 1000.0).mean() > 0.05, "no thick, pearly middle"


def test_film_override_and_errors() -> None:
    a = metal.generate((48, 48), np.random.default_rng(1), "polished", film="oxide")
    b = metal.generate((48, 48), np.random.default_rng(1), "polished")
    assert not np.array_equal(a, b)

    named = metal.generate((48, 48), np.random.default_rng(1), "heat_tinted")
    override = metal.generate(
        (48, 48), np.random.default_rng(1), "heat_tinted", film={"nm": (400.0, 450.0)}
    )
    assert not np.array_equal(named, override)

    with pytest.raises(ValueError):
        metal.generate((16, 16), np.random.default_rng(1), "heat_tinted", film="rust")
    with pytest.raises(ValueError):
        metal.generate((16, 16), np.random.default_rng(1), "nope")
    with pytest.raises(TypeError):
        metal.generate((16, 16), np.random.default_rng(1), "heat_tinted", film=3.5)


def test_shade_return_parts_recomposes_exactly() -> None:
    """``shade`` with ``return_parts`` must still add up to ``shade`` without it."""
    from texture_generators.core.shading import shade

    rng = np.random.default_rng(0)
    albedo = rng.uniform(0.2, 0.8, size=(32, 32, 3)).astype(np.float32)
    height = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
    for kwargs in ({"specular": 0.0}, {"specular": 0.6, "shininess": 40.0}):
        whole = shade(albedo, height, **kwargs)
        diffuse, spec = shade(albedo, height, return_parts=True, **kwargs)
        assert np.allclose(whole, np.clip(diffuse + spec, 0.0, 1.0), atol=1e-6)
