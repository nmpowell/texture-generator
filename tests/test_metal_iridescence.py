"""Groove diffraction: the rainbow a scratch shows, and when it does not.

The claims under test are the ones that make this diffraction rather than a
tinted highlight: the colour lives only where there is a groove, a finer grating
fans the spectrum wider and so survives a wider source, and a source broad enough
to smear the first order over itself removes the colour entirely. Plus the
promise the whole effect is built around -- off by default, and off means the
classic finishes are untouched to the last bit.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from texture_generators.core.film import wavelength_rgb
from texture_generators.materials import metal

# A light straight down +x makes the geometry exact rather than approximate:
# g = (-sin a, cos a, 0) for a groove at angle ``a``, the view contributes
# nothing (it is +z), so dot(V, g) - dot(L, g) is just -sin a and the diffracted
# wavelength is d * |sin a|.
LIGHT_X = (1.0, 0.0, 0.0)
SHAPE = (96, 96)


def _chroma(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel saturation as max-minus-min channel; 0 for any grey."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return (rgb.max(axis=-1) - rgb.min(axis=-1)).astype(np.float32)


def _angle_for(pitch_um: float, lam_nm: float) -> float:
    """Groove angle that puts the first order of ``pitch_um`` on ``lam_nm``."""
    proj = lam_nm / (pitch_um * 1000.0)
    assert proj <= 1.0, f"{lam_nm} nm is unreachable at {pitch_um} um"
    return float(np.arcsin(proj))


def _irid(
    pitch_um: float,
    *,
    lam_nm: float = 550.0,
    source_deg: float = metal.SUN_ANGULAR_RADIUS_DEG,
    weight: float = 1.0,
    scratches: np.ndarray | None = None,
) -> np.ndarray:
    """Diffraction term for a constant pitch, aimed at one wavelength.

    Holding the wavelength fixed while the pitch sweeps is what isolates the
    source-width gate: every pitch is then equally in-band, so the only thing
    left that can vary is the width of its fan.
    """
    if scratches is None:
        scratches = np.ones(SHAPE, dtype=np.float32)
    return metal._iridescence(
        SHAPE,
        np.random.default_rng(0),
        aniso_dir=_angle_for(pitch_um, lam_nm),
        scratches=scratches,
        light_dir=LIGHT_X,
        weight=weight,
        source_angular_radius=source_deg,
        groove_pitch_um=(pitch_um, pitch_um),
    )


def test_colour_appears_only_where_there_is_a_groove() -> None:
    """The scratch mask gates the colour: no groove, no grating, no rainbow."""
    scratches = np.zeros(SHAPE, dtype=np.float32)
    scratches[20:30, :] = 0.4  # one band of grooves
    scratches[60, :] = -0.4  # and one drawn with the opposite sign

    irid = _irid(1.6, scratches=scratches)
    lit = irid.max(axis=-1) > 0.0

    assert lit[20:30, :].all(), "grooved rows got no diffraction colour"
    assert lit[60, :].all(), "a negative-signed groove is still a groove"
    assert not lit[0:20, :].any(), "colour bled outside the grooves"
    assert not lit[31:60, :].any(), "colour bled outside the grooves"
    assert _chroma(irid)[20:30].mean() > 0.1, "diffraction colour is not coloured"
    assert _chroma(irid)[0:20].max() == 0.0


def test_finer_pitch_fans_wider_and_so_keeps_more_colour() -> None:
    """Chroma falls monotonically with pitch under a fixed sun-like source.

    1 um fans 380-730 nm over ~21 degrees and shrugs the sun off; 30 um fans it
    over ~0.6 degrees, which a 0.53 degree source is already eating into.
    """
    pitches = [1.0, 1.6, 5.0, 10.0, 30.0]
    chroma = [float(_chroma(_irid(p)).mean()) for p in pitches]

    assert all(a > b for a, b in pairwise(chroma)), (
        f"chroma should fall as the pitch coarsens, got {dict(zip(pitches, chroma, strict=False))}"
    )
    assert chroma[0] > 2.0 * chroma[-1], (
        "1 um should show substantially more colour than 30 um, "
        f"got {chroma[0]:.4f} vs {chroma[-1]:.4f}"
    )


def test_a_broad_source_suppresses_the_rainbow() -> None:
    """A 10 degree softbox smears the first order over itself; the sun does not."""
    sunlit = float(_chroma(_irid(5.0, source_deg=0.53)).mean())
    softbox = float(_chroma(_irid(5.0, source_deg=10.0)).mean())

    assert sunlit > 0.1, "a sun-like source should show colour at 5 um"
    assert softbox < 0.01 * sunlit, (
        f"a 10 degree source should suppress a 5 um grating, "
        f"got {softbox:.6f} vs {sunlit:.4f}"
    )
    # And the fine end is where a broad source can still find colour: 1 um fans
    # 21 degrees, which is wider than the softbox itself.
    assert float(_chroma(_irid(1.0, source_deg=10.0)).mean()) > 0.1


def test_wavelengths_outside_the_visible_band_contribute_nothing() -> None:
    """The first order is only visible while it lands in 380-730 nm."""
    # Aim the geometry at 550 nm for a 1 um pitch, then keep the geometry and
    # coarsen the grating 10x: the order moves to 5.5 um, which the eye has no
    # receptor for.
    angle = _angle_for(1.0, 550.0)
    out = metal._iridescence(
        SHAPE,
        np.random.default_rng(0),
        aniso_dir=angle,
        scratches=np.ones(SHAPE, dtype=np.float32),
        light_dir=LIGHT_X,
        weight=1.0,
        groove_pitch_um=(10.0, 10.0),
    )
    assert out.max() == 0.0


def test_the_spectral_lookup_is_the_observer_not_a_hue_ramp() -> None:
    """Landmark wavelengths come back the colour the CIE table says they are."""
    r, g, b = (wavelength_rgb(nm) for nm in (630.0, 530.0, 460.0))
    assert r.argmax() == 0 and g.argmax() == 1 and b.argmax() == 2
    # 555 nm is the luminous peak, and the locus is outside sRGB, so clipping
    # leaves it positive in every channel-argmax sense but never negative.
    assert wavelength_rgb(np.array([380.0, 555.0, 730.0])).min() >= 0.0


def test_off_by_default_and_bit_identical_when_off() -> None:
    """The default is no diffraction, and asking for none changes nothing."""
    for variant in ("brushed", "radial", "polished", "engine_turned"):
        default = metal.generate((128, 128), np.random.default_rng(11), variant)
        explicit = metal.generate(
            (128, 128), np.random.default_rng(11), variant, iridescence=0.0
        )
        assert np.array_equal(default, explicit), f"{variant} moved with the effect off"


def test_iridescence_raises_chroma_inside_the_scratches_only() -> None:
    """End to end: switching it on colours grooves and leaves the sheet alone.

    The scratch field is captured off the real render rather than rebuilt, so the
    "inside" set is the finish's own grooves and the containment check is not
    circular.
    """
    for variant in ("brushed", "radial", "engine_turned"):
        seen: dict[str, np.ndarray] = {}
        real = metal._iridescence

        def spy(*args, **kwargs):
            seen["scratches"] = np.asarray(  # noqa: B023  # called before loop advances
                kwargs["scratches"], dtype=np.float32
            )
            return real(*args, **kwargs)  # noqa: B023  # called before loop advances

        off = metal.generate((256, 256), np.random.default_rng(3), variant)
        metal._iridescence = spy
        try:
            on = metal.generate(
                (256, 256), np.random.default_rng(3), variant, iridescence=1.0
            )
        finally:
            metal._iridescence = real

        grooved = np.abs(seen["scratches"]) > 0.0
        changed = (off != on).any(axis=-1)

        assert changed.any(), f"{variant} showed no diffraction at all"
        assert changed.mean() < 0.10, (
            f"{variant} coloured {changed.mean():.1%} of pixels"
        )
        assert not (changed & ~grooved).any(), f"{variant} coloured un-grooved pixels"
        assert np.array_equal(off[~grooved], on[~grooved]), (
            f"{variant} moved outside its grooves"
        )
        assert _chroma(on)[changed].mean() > 3.0 * _chroma(off)[changed].mean(), (
            f"{variant} gained no chroma where it diffracted"
        )


def test_source_width_gate_reaches_a_real_render() -> None:
    """A softbox kills a coarse grating and barely dents a fine one.

    Both cases in one variant: a spun face carries every groove orientation, so
    a single render has pixels in band at 10-30 um and pixels in band at
    1-1.6 um. Measured as the chroma the effect *adds*, not the chroma on screen
    -- a brass or copper sheet already has a couple of tenths of chroma, and that
    offset would swamp the comparison.
    """

    def gains(pitch_um: tuple[float, float]) -> tuple[float, float, int]:
        args = ((256, 256),)
        off = metal.generate(*args, np.random.default_rng(3), "radial")
        kw = dict(iridescence=1.0, groove_pitch_um=pitch_um)
        sun = metal.generate(*args, np.random.default_rng(3), "radial", **kw)
        softbox = metal.generate(
            *args,
            np.random.default_rng(3),
            "radial",
            source_angular_radius=10.0,
            **kw,
        )
        lit = (off != sun).any(axis=-1)
        base = _chroma(off)[lit]
        return (
            float((_chroma(sun)[lit] - base).mean()),
            float((_chroma(softbox)[lit] - base).mean()),
            int(lit.sum()),
        )

    coarse_sun, coarse_softbox, coarse_n = gains((10.0, 30.0))
    fine_sun, fine_softbox, fine_n = gains((1.0, 1.6))

    assert coarse_n > 50 and fine_n > 50, "the pitch bands put nothing in band"
    assert coarse_sun > 0.05 and fine_sun > 0.05, "no colour under a sun-like source"
    assert coarse_softbox < 0.05 * coarse_sun, (
        f"a 10 degree source should wash out a 10-30 um grating, "
        f"got {coarse_softbox:.4f} vs {coarse_sun:.4f}"
    )
    assert fine_softbox > 0.5 * fine_sun, (
        f"a 1-1.6 um grating fans wider than a 10 degree source and should "
        f"survive it, got {fine_softbox:.4f} vs {fine_sun:.4f}"
    )
