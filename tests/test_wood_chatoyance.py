"""Tests for wood chatoyance: the fibre-tangent field and the two specular lobes.

Chatoyance is not a look, it is a *behaviour*: the luster has to move when the
light does, and it has to move differently along the grain than across it. So
nearly everything here renders the same board twice under two lights and compares
the pair -- a single render cannot tell an anisotropic lobe from a painted-on
streak, which is exactly why the flat-print tell survived so long.

The tangent field is checked for the property the derivation is supposed to
guarantee (unit length, and varying per pixel because the distortion varies), not
for particular values: the values belong to the warp, and the warp is already
tested where it lives.
"""

from __future__ import annotations

import numpy as np

from texture_generators.core.shading import _fibre_lobe, shade
from texture_generators.materials import wood
from texture_generators.materials.wood import FIGURES, SPECULAR, _specular_lobes

LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _lum(img: np.ndarray) -> np.ndarray:
    return (img @ LUMA).astype(np.float32)


def _render(seed: int = 0, size: int = 256, **params) -> np.ndarray:
    base = dict(species="maple", finish="polyurethane", knots=0.0, sapwood=0.0)
    return wood.generate(
        (size, size), np.random.default_rng(seed), "board", **{**base, **params}
    )


def _across_grain_energy(
    img: np.ndarray, band_px: tuple[float, float] = (4.0, 40.0)
) -> float:
    """Power in the across-grain luminance profile at fiddleback wavelengths.

    Collapsing along the grain first is what makes this a *figure* measure rather
    than a noise measure: rings, streaks and pores all run along the grain, so the
    mean over that axis keeps only structure that is banded across it.
    """
    prof = _lum(img).mean(axis=1)
    prof = prof - prof.mean()
    power = np.abs(np.fft.rfft(prof)) ** 2
    k = np.arange(power.size)
    n = prof.size
    keep = (k >= n / band_px[1]) & (k <= n / band_px[0])
    return float(power[keep].sum())


# --- The tangent field -------------------------------------------------------


def test_fibre_tangents_are_unit_length_and_vary_across_the_board() -> None:
    """Unit vectors, and not the same unit vector everywhere.

    A constant tangent field is what the generator effectively had before (one
    grain angle per board), and it cannot produce figure: the whole argument for
    taking the field off the growth distortion is that the distortion varies.
    """
    for variant in ("board", "planks"):
        builder = wood._board_fields if variant == "board" else wood._planks
        _, _, tangent, _ = builder(
            (192, 192), np.random.default_rng(4), "maple", finish="oil"
        )
        assert tangent.shape == (192, 192, 3)
        norm = np.sqrt((tangent * tangent).sum(axis=-1))
        assert np.abs(norm - 1.0).max() < 1e-4, f"{variant}: not unit length"
        # Every component has to move, the out-of-plane one included -- that is
        # the component the surface tangent frame cannot see and the fibre lobe
        # lives on.
        for i, name in enumerate("xyz"):
            assert tangent[..., i].std() > 1e-4, f"{variant}: constant t{name}"


def test_the_tangent_field_is_the_warp_not_a_second_noise_field() -> None:
    """Two boards that share a warp must share a tangent field.

    Liu et al.'s point is that the fibre directions *follow* from the growth
    distortions. Concretely that means the field carries no randomness of its own
    beyond its amplitude: hand :func:`_fibre_tangents` the same coordinates twice
    and the in-plane part comes back identical, because it is a gradient of the
    warp and nothing else.
    """
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, 128, dtype=np.float32),
        np.linspace(0.0, 1.0, 128, dtype=np.float32),
    )
    wu, wv = x + 0.02 * np.sin(6.0 * y), y + 0.02 * np.sin(5.0 * x)
    kw = dict(tilt=0.0, along_x=True, px_per_unit=128.0, mm_per_unit=225.0)
    a = wood._fibre_tangents(x, y, wu, wv, np.random.default_rng(0), **kw)
    b = wood._fibre_tangents(x, y, wu, wv, np.random.default_rng(1), **kw)
    # The in-plane *direction* is a pure function of the warp -- no rng anywhere
    # in it. (Only the dip's amplitude is drawn, which foreshortens both in-plane
    # components together and so leaves the angle alone.)
    angle_a = np.arctan2(a[..., 1], a[..., 0])
    angle_b = np.arctan2(b[..., 1], b[..., 0])
    assert np.abs(angle_a - angle_b).max() < 1e-5
    # A warp that bends across the grain must deflect the fibres in plane.
    assert np.abs(angle_a).max() > 1e-3

    # ...and it must deflect them ALONG the ring, not up its normal. Unit length
    # and variation both survive a sign or a dropped term on the deformation
    # gradient, so pin the direction itself: the fibre follows a contour of the
    # warped across-grain coordinate, i.e. t . grad(wv) == 0. The buggy form
    # ``atan(d(wv)/du)`` is the ring *normal* and fails this by ~2 * d(wv)/du.
    gu = np.gradient(wv, axis=1) * 128.0  # d(wv)/du, u running down axis 1
    gv = np.gradient(wv, axis=0) * 128.0
    dot = a[..., 0] * gu + a[..., 1] * gv
    scale = np.hypot(a[..., 0], a[..., 1]) * np.maximum(np.hypot(gu, gv), 1e-6)
    assert np.abs(dot / scale).max() < 1e-3, "fibre runs up the ring normal"
    # Along the grain, not backwards along it.
    assert a[..., 0].min() > 0.0


# --- The fibre lobe ---------------------------------------------------------


def test_fibre_lobe_is_anisotropic_about_the_tangent() -> None:
    """The lobe must answer to the light's angle *along* the fibre, not across it.

    A fibre reflects into a cone about its own axis, so rotating the light in the
    plane containing the axis sweeps straight through the cone, while rotating it
    about the axis leaves the cone condition untouched. That asymmetry is the
    definition of anisotropy, and an isotropic lobe would score the same either
    way.
    """
    t = np.zeros((1, 1, 3), dtype=np.float32)
    t[..., 0] = 1.0  # fibre along +x
    ndl = np.ones((1, 1), dtype=np.float32)

    def lobe(lx: float, ly: float) -> float:
        lz = float(np.sqrt(max(1.0 - lx * lx - ly * ly, 1e-6)))
        light = np.asarray([lx, ly, lz], dtype=np.float32)
        return float(_fibre_lobe(t, light, ndl, 40.0)[0, 0])

    along = [lobe(lx, 0.0) for lx in np.linspace(-0.6, 0.6, 13)]
    across = [lobe(0.0, ly) for ly in np.linspace(-0.6, 0.6, 13)]
    assert max(along) - min(along) > 0.5, "swinging along the fibre did nothing"
    assert max(across) - min(across) < 1e-4, "swinging about the fibre changed it"


def test_fibre_dip_is_what_the_surface_tangent_frame_cannot_see() -> None:
    """A dip of +/-20 degrees must swing the fibre lobe over most of its range.

    This is the reason the fibre lobe exists rather than being folded into the
    anisotropic Blinn-Phong lobe already in :func:`shade`: on a face-on surface
    the half-vector's tangent-plane projection has no out-of-plane component, so
    the dip cancels out of that lobe exactly. Curly figure *is* the dip.
    """
    ndl = np.ones((1, 1), dtype=np.float32)
    light = np.asarray([-0.463, -0.509, 0.723], dtype=np.float32)

    def lobe(dip_deg: float) -> float:
        th = np.deg2rad(dip_deg)
        t = np.asarray([[[np.cos(th), 0.0, -np.sin(th)]]], dtype=np.float32)
        return float(_fibre_lobe(t, light, ndl, 40.0)[0, 0])

    values = [lobe(d) for d in (-20.0, -10.0, 0.0, 10.0, 20.0)]
    assert max(values) - min(values) > 0.6, f"dip barely moved the lobe: {values}"


# --- The two lobes, per finish ----------------------------------------------


def test_finish_sharpens_and_isotropises_the_surface_lobe_but_strengthens_the_fibre() -> (
    None
):
    """The trend :data:`SPECULAR` exists to encode, measured on the drawn values.

    As the film gets glossier the surface lobe sharpens and becomes more
    isotropic (it is a film, not wood), while the fibre lobe -- which is the wood
    seen through an index-matched interface -- gets stronger. Losing either half
    of that is how a gloss finish ends up reading as a plastic laminate.
    """
    order = ["none", "oil", "acrylic", "polyurethane"]
    drawn = {
        f: [_specular_lobes(f, np.random.default_rng(s)) for s in range(64)]
        for f in order
    }
    med = {
        f: {k: float(np.median([d[k] for d in v])) for k in v[0]}
        for f, v in drawn.items()
    }

    sharpness = [med[f]["shininess"] for f in order]
    assert sharpness == sorted(sharpness), f"surface lobe not sharpening: {sharpness}"
    fibre = [med[f]["fibre_weight"] for f in order]
    assert fibre[0] < fibre[1] and fibre[1] < fibre[-1], (
        f"fibre lobe not gaining: {fibre}"
    )
    # Raw wood and a gloss film are both near-isotropic; oil is the anisotropic one.
    assert med["oil"]["aniso"] > 2.0 * med["none"]["aniso"]
    assert med["oil"]["aniso"] > 2.0 * med["polyurethane"]["aniso"]
    # A dielectric film reflects 4% at normal incidence, so no lobe weight here
    # has any business being large.
    assert all(med[f]["surface_weight"] < 0.25 for f in order)


def test_every_finish_declares_both_lobes() -> None:
    """No finish may fall back on a default: the table is the whole model."""
    assert set(SPECULAR) == set(wood.FINISHES)
    keys = {"alpha_along", "alpha_across", "aniso", "fibre_exponent", "fibre_weight"}
    for finish, spec in SPECULAR.items():
        assert set(spec) == keys, finish
        for name, (lo, hi) in spec.items():
            assert 0.0 <= lo <= hi, f"{finish}/{name}: {lo}..{hi}"
        # Roughness is lower along the grain than across it -- that inequality is
        # the anisotropy, and reversing it would streak the highlight the wrong way.
        assert spec["alpha_along"][0] <= spec["alpha_across"][0]
        assert spec["alpha_along"][1] <= spec["alpha_across"][1]


# --- The behaviour: luster that moves with the light ------------------------


def test_rotating_the_light_moves_the_luster_differently_along_and_across() -> None:
    """Swing the light along the grain, then across it: the two must not match.

    For an isotropic surface these two swings are the same swing -- the response
    depends only on the angle to the normal. A fibre lobe aligned to the grain
    breaks that, and by a lot: which way the light comes relative to the grain is
    the single thing chatoyance is about.
    """
    for figure in ("plain", "curly"):
        along, across = [], []
        for seed in range(3):
            kw = dict(seed=seed, figure=figure)
            a = np.abs(
                _lum(_render(light_dir=(-0.6, 0.0, 0.8), **kw))
                - _lum(_render(light_dir=(0.6, 0.0, 0.8), **kw))
            ).mean()
            b = np.abs(
                _lum(_render(light_dir=(0.0, -0.6, 0.8), **kw))
                - _lum(_render(light_dir=(0.0, 0.6, 0.8), **kw))
            ).mean()
            along.append(float(a))
            across.append(float(b))
        ratio = float(np.mean(along) / max(np.mean(across), 1e-9))
        assert not 0.67 < ratio < 1.5, (
            f"{figure}: light swings along and across the grain agree to "
            f"{ratio:.2f}, which is what an isotropic lobe would give"
        )


def test_curly_oscillates_across_the_grain_more_than_plain() -> None:
    """Fiddleback is a banded oscillation across the grain, and must measure as one.

    The bands are in the *reflection*, not the albedo, so this only appears once
    the board is shaded -- which is why it comes almost free off the tangent
    field and needs no figure texture of its own.
    """
    for species in ("maple", "sapele"):
        scores = {}
        for figure in ("plain", "curly"):
            scores[figure] = float(
                np.mean(
                    [
                        _across_grain_energy(
                            _render(
                                seed=s, species=species, finish="oil", figure=figure
                            )
                        )
                        for s in range(4)
                    ]
                )
            )
        assert scores["curly"] > 2.5 * scores["plain"], f"{species}: {scores}"


def test_figure_is_an_occasional_draw_and_only_on_the_species_that_have_it() -> None:
    """Curl belongs to the log, so it is drawn per panel and it is rare.

    Every board figured is as wrong as none: figured stock is selected timber, and
    a floor of it reads as a rendering setting rather than as wood.
    """
    for species in sorted(wood.SPECIES):
        draws = [
            wood._pick_figure(species, np.random.default_rng(s)) for s in range(200)
        ]
        plain = draws.count("plain") / len(draws)
        if species in wood.FIGURE_P:
            assert 0.3 < plain < 0.95, f"{species}: {plain:.2f} plain"
        else:
            assert plain == 1.0, f"{species}: figured without a FIGURE_P entry"
        assert set(draws) <= set(FIGURES)


def test_both_variants_render_every_figure() -> None:
    """No figure may break a variant, and none may blow the exposure."""
    for variant in ("board", "planks"):
        for figure in FIGURES:
            img = wood.generate(
                (192, 192),
                np.random.default_rng(11),
                variant,
                species="sapele",
                finish="oil",
                figure=figure,
            )
            assert img.shape == (192, 192, 3)
            assert np.isfinite(img).all()
            # Chatoyance adds light; if it clipped, the measurement is gone.
            assert float((img > 0.999).mean()) < 0.001, f"{variant}/{figure} clipping"


def test_specular_energy_comes_out_of_the_diffuse() -> None:
    """``conserve_energy`` must hold the render's mean, per channel.

    The species colour is a *measurement*, and a fibre lobe running to 0.2 is 50
    display levels of pure addition on top of it. Light reflected at or just under
    the surface never reached the pigment, so the diffuse is what pays for it --
    otherwise every finish silently lightens and desaturates the board it was
    supposed to deepen.
    """
    rng = np.random.default_rng(2)
    albedo = np.full((64, 64, 3), 0.45, dtype=np.float32)
    albedo[..., 2] = 0.3
    height = rng.standard_normal((64, 64)).astype(np.float32) * 0.1
    tangent = np.zeros((64, 64, 3), dtype=np.float32)
    tangent[..., 0] = 1.0
    kw = dict(
        specular=0.1, shininess=40.0, fibre_tangent=tangent, fibre=0.2, normalise=True
    )
    loud = shade(albedo, height, **kw)
    quiet = shade(albedo, height, normalise=True)
    paid = shade(albedo, height, conserve_energy=True, **kw)
    lit = np.asarray([float(loud[..., c].mean()) for c in range(3)])
    ref = np.asarray([float(quiet[..., c].mean()) for c in range(3)])
    fair = np.asarray([float(paid[..., c].mean()) for c in range(3)])
    assert (lit - ref).min() > 0.01, "the lobes added nothing to measure"
    assert np.abs(fair - ref).max() < 0.005, f"mean not held: {fair} vs {ref}"


def test_the_new_lobe_is_off_by_default() -> None:
    """Metal, plastic and paper must see exactly the shading they saw before.

    Their albedos are hand-tuned *including* this rig's mean, so a lobe that
    switched itself on -- or an energy correction that did -- would shift every one
    of them. The guard is that the defaults are inert to the bit.
    """
    rng = np.random.default_rng(5)
    albedo = rng.uniform(0.2, 0.8, (48, 48, 3)).astype(np.float32)
    height = rng.standard_normal((48, 48)).astype(np.float32)
    tangent = np.zeros((48, 48, 3), dtype=np.float32)
    tangent[..., 2] = 1.0
    base = shade(
        albedo, height, specular=0.3, shininess=60.0, specular2=0.2, spec_tint=0.7
    )
    same = shade(
        albedo,
        height,
        specular=0.3,
        shininess=60.0,
        specular2=0.2,
        spec_tint=0.7,
        fibre_tangent=tangent,
        fibre=0.0,
    )
    assert np.array_equal(base, same)
