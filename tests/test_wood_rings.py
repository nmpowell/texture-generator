"""Tests for wood ring geometry: the width series and the within-ring profile.

Two things gave the old ring layer away as a colour-ramp lookup, and both are
*statistical* claims that can be measured rather than eyeballed:

* ring widths came from a constant spacing, so every ring was the same width.
  A real series is an autocorrelated log-normal one, and the numbers it is
  described by -- lag-1 autocorrelation, the standard deviation of the raw log
  widths, mean sensitivity, the mean width in millimetres -- are the standard
  dendrochronological measures, so they are checked as such here.
* the profile across a ring was a symmetric soft ramp. A real ring is a
  sawtooth: a smooth earlywood -> latewood climb and an abrupt step back at the
  ring boundary. So the test is for *asymmetry* and for the size of that step.
"""

from __future__ import annotations

import numpy as np

from texture_generators.materials import wood
from texture_generators.materials.wood import (
    ANATOMY,
    RING_ACF1,
    RING_PROFILE,
    RING_STATS,
    RING_WIDTH_LIMITS_MM,
    _ring_sawtooth,
    _ring_widths,
)

# The bands the series statistics are specified over (see RING_STATS).
MEAN_WIDTH_MM = (1.5, 3.5)
SENSITIVITY = (0.15, 0.35)


def _acf1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of ``x`` about its own mean."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    return float((x[:-1] * x[1:]).sum() / max((x * x).sum(), 1e-12))


def _sensitivity(w: np.ndarray) -> float:
    """Mean sensitivity: the mean relative change between adjacent rings."""
    w = np.asarray(w, dtype=np.float64)
    return float(np.abs(2.0 * (w[1:] - w[:-1]) / (w[1:] + w[:-1])).mean())


def _index_sd(w: np.ndarray) -> float:
    """Standard deviation of the log-width *index*, i.e. after detrending.

    Detrended the way a dendrochronologist would: fit a smooth curve through the
    log widths and take the residual. A cubic is enough to absorb the age trend
    without eating the year-to-year signal, and it deliberately does not know
    which curve :func:`_ring_widths` used.
    """
    lw = np.log(np.asarray(w, dtype=np.float64))
    t = np.arange(lw.size, dtype=np.float64) / max(lw.size - 1, 1)
    return float((lw - np.polyval(np.polyfit(t, lw, 3), t)).std())


def _series(pore_class: str, n_boards: int = 120, seed: int = 0) -> list[np.ndarray]:
    """Draw ``n_boards`` ring-width series for one porosity class."""
    stats = RING_STATS[pore_class]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boards):
        out.append(
            _ring_widths(
                int(rng.integers(30, 140)),
                rng,
                mean_mm=float(rng.uniform(*stats["mean_mm"])),
                sigma=float(rng.uniform(*stats["sigma"])),
            )
        )
    return out


def test_ring_widths_are_an_autocorrelated_log_normal_series() -> None:
    """Each of the four statistics a ring-width series is described by.

    Lag-1 autocorrelation ~0.7 on the raw series, the raw log-width spread
    inside the class's own band, mean sensitivity 0.15-0.35, and a mean width
    in the 1.5-3.5 mm of furniture and flooring stock. All four have to hold at
    once: the autocorrelation is what a white-noise series fails, and mean
    sensitivity is what a series with the autocorrelation but no year-to-year
    amplitude fails.
    """
    for pore_class, stats in RING_STATS.items():
        series = _series(pore_class)
        acf = float(np.mean([_acf1(np.log(w)) for w in series]))
        assert abs(acf - RING_ACF1) < 0.15, f"{pore_class}: lag-1 acf {acf:.3f}"

        # sigma is the spread of the RAW log widths (see RING_STATS), so that is
        # what is measured here: trend and year-to-year term together, logs taken
        # and nothing else done. Specifying the detrended index instead leaves
        # the total free, and the trend's variance then lands on top of it -- the
        # over-dispersion this checks against.
        raw = np.asarray([np.log(w).std() for w in series])
        lo, hi = stats["sigma"]
        assert 0.20 <= lo <= hi <= 0.35, f"{pore_class}: sigma {(lo, hi)} off spec"
        assert lo * 0.92 < raw.mean() < hi * 1.05, (
            f"{pore_class}: raw sd {raw.mean():.3f}"
        )

        # ...and the detrended index is the AR part of that split, ~0.68x of it.
        idx = np.asarray([_index_sd(w) for w in series])
        assert 0.55 < idx.mean() / raw.mean() < 0.80, (
            f"{pore_class}: index/raw {idx.mean() / raw.mean():.3f}"
        )

        sens = np.asarray([_sensitivity(w) for w in series])
        assert SENSITIVITY[0] < sens.mean() < SENSITIVITY[1], (
            f"{pore_class}: mean sensitivity {sens.mean():.3f}"
        )

        means = np.asarray([w.mean() for w in series])
        assert MEAN_WIDTH_MM[0] < means.mean() < MEAN_WIDTH_MM[1]
        widths = np.concatenate(series)
        assert RING_WIDTH_LIMITS_MM[0] <= widths.min()
        assert widths.max() <= RING_WIDTH_LIMITS_MM[1]


def test_ring_widths_are_not_white_noise_and_not_a_ruler() -> None:
    """The two failure modes either side of a grown series.

    A ruler (the constant spacing this replaced) has no variation at all; white
    noise has variation but no memory, so a good year is as likely to follow a
    bad one as another good one. Shuffling the series destroys exactly the
    property being claimed and nothing else, which is what makes it the control.
    """
    rng = np.random.default_rng(11)
    w = _ring_widths(120, rng, mean_mm=2.5, sigma=0.25)

    assert float(np.std(w) / np.mean(w)) > 0.15, "widths barely vary: still a ruler"

    shuffled = np.array(w)
    rng.shuffle(shuffled)
    assert _acf1(np.log(w)) > 0.45
    assert abs(_acf1(np.log(shuffled))) < 0.25
    # ...and the same widths in a different order, so every other statistic here
    # is identical. Only the memory is gone.
    assert np.isclose(np.sort(w), np.sort(shuffled)).all()


def test_rings_narrow_outward_from_the_pith() -> None:
    """The age trend: a similar volume of wood on an ever-longer circumference.

    Checked as inner third against outer third, averaged over boards, because a
    single series' AR term can easily swamp the trend over a short run.
    """
    for pore_class in RING_STATS:
        ratios = []
        for w in _series(pore_class, n_boards=60, seed=5):
            third = max(w.size // 3, 2)
            ratios.append(float(w[:third].mean() / w[-third:].mean()))
        assert np.mean(ratios) > 1.3, f"{pore_class}: no age trend"


def test_ring_sigma_is_the_dramatic_board_knob() -> None:
    """Mean sensitivity follows sigma, and sigma 0 is the even-ruled degenerate.

    ``sigma`` is the one control worth exposing, so it has to be monotone in the
    thing it claims to control -- and at 0 it has to collapse back to a constant
    spacing, which is both the sanity check and the way to render the old
    behaviour for comparison.
    """
    rng = np.random.default_rng(2)
    even = _ring_widths(80, rng, mean_mm=2.5, sigma=0.0)
    assert float(even.std()) < 1e-9
    assert abs(float(even.mean()) - 2.5) < 1e-9

    got = []
    for sigma in (0.10, 0.20, 0.30, 0.40):
        rng = np.random.default_rng(4)
        runs = [
            _sensitivity(_ring_widths(120, rng, mean_mm=2.5, sigma=sigma))
            for _ in range(40)
        ]
        got.append(float(np.mean(runs)))
    assert got == sorted(got), f"sensitivity not monotone in sigma: {got}"
    # The relation is ~0.6 sigma (see RING_STATS: sigma is the raw spread, and
    # sensitivity sees only the AR share of it), so it is a usable knob and not a
    # nearly flat one.
    assert 0.45 < (got[-1] - got[0]) / (0.40 - 0.10) < 0.85


def _profile(pore_class: str, width_mm: float, seed: int = 0, n: int = 512):
    """The within-ring profile across one ring, sampled at ``n`` points."""
    g = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return _ring_sawtooth(
        g,
        np.full(n, width_mm, dtype=np.float32),
        pore_class,
        np.random.default_rng(seed),
        np.zeros(n, dtype=np.float32),
    )


def test_ring_profile_is_an_asymmetric_discontinuous_sawtooth() -> None:
    """The profile must be a sawtooth, for every porosity class.

    Three properties, and a symmetric soft ramp -- the thing that reads as a
    printed colour ramp -- fails all three: the density is at its maximum at the
    ring boundary rather than in the middle, it steps back to ~0 across that
    boundary, and its mass sits in the outer half of the ring.
    """
    for pore_class in RING_PROFILE:
        density, late_w = _profile(pore_class, 2.5, seed=3)

        assert density[-1] > 0.95, f"{pore_class}: not densest at the boundary"
        assert density[0] < 0.08, f"{pore_class}: earlywood does not start open"
        # The step across the boundary, which is the whole point.
        assert density[-1] - density[0] > 0.9, f"{pore_class}: no step at the boundary"
        # Monotone across the ring: density only ever climbs within a year.
        assert float(np.diff(density).min()) > -1e-6, f"{pore_class}: not monotone"
        # Asymmetric: the centroid of the density sits past the mid-ring.
        g = np.linspace(0.0, 1.0, density.size)
        centroid = float((g * density).sum() / density.sum())
        assert centroid > 0.55, f"{pore_class}: profile is symmetric ({centroid:.2f})"

        # And the latewood share is the one the class publishes.
        share = float(late_w.mean())
        lo, hi = RING_PROFILE[pore_class]["lw_frac"]
        assert lo - 0.15 < share < hi + 0.15, (
            f"{pore_class}: latewood share {share:.2f}"
        )


def test_transition_sharpness_separates_the_porosity_classes() -> None:
    """Ring-porous switches inside a third of a millimetre; diffuse never does.

    That contrast is the porosity class, so if these collapse together the
    profile has stopped being species-specific. Measured over the middle of the
    rise (25-75% of the range), which is the part that lies inside the
    transition itself rather than in either zone's own gentle climb, then scaled
    back up: a smoothstep covers its middle half in 0.414 of its full width.
    """

    def rise_mm(pore_class: str, width_mm: float = 2.5) -> float:
        density, _ = _profile(pore_class, width_mm, seed=7, n=4096)
        lo, hi = float(density.min()), float(density.max())
        band = (density > lo + 0.25 * (hi - lo)) & (density < lo + 0.75 * (hi - lo))
        return float(band.mean()) * width_mm / 0.414

    ring = rise_mm("ring-porous")
    semi = rise_mm("semi-ring-porous")
    diffuse = rise_mm("diffuse-porous")
    soft = rise_mm("softwood")

    # Oak and ash switch over a fraction of a millimetre -- visible as a line.
    assert ring < 0.45, f"ring-porous transition {ring:.2f} mm is not abrupt"
    # Walnut and cherry take 30-60% of the ring over it.
    assert 0.6 < semi < 1.8, f"semi-ring-porous transition {semi:.2f} mm"
    assert diffuse > semi, "diffuse-porous should be the most gradual"
    assert soft < semi, "pine's latewood band is abrupt at its outer edge"


def test_ring_porous_latewood_share_rises_with_ring_width() -> None:
    """A wide ring-porous ring means more latewood, not more earlywood.

    Its earlywood is a roughly fixed-width band of coarse vessels whatever the
    year, so everything a good year adds goes on the latewood side. That is the
    same fact the pore layer relies on, seen from the colour side.
    """
    narrow, _ = _profile("ring-porous", 1.0, seed=1)
    wide, _ = _profile("ring-porous", 4.0, seed=1)
    assert wide.mean() > narrow.mean() + 0.15
    # The classes that do not track ring width must not drift with it.
    for pore_class in ("semi-ring-porous", "diffuse-porous", "softwood"):
        a, _ = _profile(pore_class, 1.0, seed=1)
        b, _ = _profile(pore_class, 4.0, seed=1)
        assert abs(a.mean() - b.mean()) < 0.02, f"{pore_class} tracks ring width"


def test_variable_ring_widths_reach_the_render() -> None:
    """The series has to be wired to the pixels, not just to a unit test.

    ``ring_sigma=0`` collapses the series to the constant spacing this replaced,
    so rendering the same seed either side of that is the end-to-end check --
    and pine, whose latewood is the hardest line here, is where it shows.
    """
    common = dict(species="pine", finish="none", colour_variation=0.0, knots=0.0)
    even = wood.generate(
        (192, 192), np.random.default_rng(0), "board", ring_sigma=0.0, **common
    )
    grown = wood.generate(
        (192, 192), np.random.default_rng(0), "board", ring_sigma=0.35, **common
    )
    assert not np.allclose(even, grown)
    # Same board otherwise: the mean is still the species colour either way.
    assert float(np.abs(even.mean(axis=(0, 1)) - grown.mean(axis=(0, 1))).max()) < 0.02


def test_every_species_declares_ring_statistics_and_a_profile() -> None:
    """No species may fall back on a default: the tables are keyed by pore class."""
    for species, anat in ANATOMY.items():
        pore_class = anat["pore_class"]
        assert pore_class in RING_STATS, f"{species}: no ring statistics"
        assert pore_class in RING_PROFILE, f"{species}: no ring profile"
        lo, hi = RING_STATS[pore_class]["sigma"]
        assert 0.15 <= lo <= hi <= 0.35, f"{pore_class}: sigma {(lo, hi)}"
        lo, hi = RING_STATS[pore_class]["mean_mm"]
        assert MEAN_WIDTH_MM[0] <= lo <= hi <= MEAN_WIDTH_MM[1]


# Band split for the ring:pore balance below, in millimetres of board. The ring
# arcs on a flatsawn face are 10-30 mm apart, so everything coarser than a few
# millimetres is ring figure and everything finer than a millimetre is pores,
# grain and the latewood line.
RING_BAND_MM = 3.0
PORE_BAND_MM = 1.0
BOARD_MM = 225.0
