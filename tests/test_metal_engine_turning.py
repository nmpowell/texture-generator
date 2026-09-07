"""Engine turning (jewelling / damaskeening): the lattice, the ordering, the frames.

Three claims are worth pinning, because all three are things the render could
lose silently while still looking busy:

1. the swirl marks sit on a lattice whose **period is the step between disc
   centres**, and that step scales with the render size rather than drifting
   into a finer or coarser finish as the output grows;
2. the discs are ordered **last-one-wins** -- a pixel is owned by the last-cut
   disc that reaches it, never an earlier one -- which is the only reason the
   marks come out as crescents instead of as tiled circles;
3. the anisotropy direction is **per disc**: it turns through a full revolution
   inside each mark and jumps at the seam between marks, so the plate glitters
   as a lattice of small suns rather than under one broad band.

The disc geometry itself (3-20 mm across, stepped 0.4-0.7 of a diameter) is
UNVERIFIED domain knowledge; these tests pin the numbers the code claims, not
the numbers a workshop would measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from texture_generators import generate
from texture_generators.materials import metal

SIZE = 512
SEEDS = (0, 1, 2, 3, 5, 6, 7)


def _lattice(seed: int, size: int = SIZE) -> tuple[float, float, bool]:
    """Reproduce the lattice ``metal.generate`` will draw for ``seed``.

    ``generate`` seeds one Generator, draws the palette, then draws the
    lattice, so replaying those two calls in order gives the same numbers the
    render used.
    """
    rng = np.random.default_rng(seed)
    metal._pick_palette(rng)
    return metal.engine_turn_lattice(rng, size)


def _luma(seed: int, size: int = SIZE) -> np.ndarray:
    img = generate("metal", size=size, seed=seed, variant="engine_turned")
    return np.asarray(img, dtype=np.float64).mean(axis=-1) / 255.0


def _fft_period(luma: np.ndarray, axis: int) -> tuple[float, float]:
    """Dominant spatial period along ``axis``, and its power over the median.

    Magnitudes are averaged over the *transverse* lines rather than the lines
    themselves being averaged first: brick-laid rows are offset by half a step,
    so averaging the signal would cancel the very peak being looked for, while
    averaging magnitudes is blind to that phase.
    """
    a = luma.T if axis == 0 else luma
    a = a - a.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(a, axis=1)).mean(axis=0)
    # The step is 8-48 px at 512 px wide, so the fundamental lives here. Below
    # k=6 is the broad sheen band, which is not lattice structure.
    lo, hi = 6, min(120, mag.size - 1)
    k = lo + int(np.argmax(mag[lo : hi + 1]))
    return a.shape[1] / k, float(mag[k] / np.median(mag[lo : hi + 1]))


def _nominal_cell(
    shape: tuple[int, int], step_px: float, brick: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The cell each pixel geometrically falls in, ignoring occlusion."""
    h, w = shape
    rows = np.floor(np.arange(h, dtype=np.float32) / np.float32(step_px)).astype(
        np.int64
    )
    nj = np.repeat(rows[:, None], w, axis=1)
    off = metal._brick_offset(nj, step_px, brick) + np.zeros((h, w), dtype=np.float32)
    x = np.arange(w, dtype=np.float32)[None, :]
    ni = np.floor((x - off) / np.float32(step_px)).astype(np.int64)
    return ni, nj


# --- the variant exists and is purely an addition ---------------------------


def test_registered_without_disturbing_the_classics() -> None:
    assert "engine_turned" in metal.VARIANTS
    assert list(metal.VARIANTS[:3]) == ["brushed", "radial", "polished"]
    assert "engine_turned" in metal._BUILDERS
    # No film: the variant must take the unfilmed short-circuit in generate().
    assert "engine_turned" not in metal.FILM_VARIANTS


@pytest.mark.parametrize("seed", (1, 4))
def test_renders_in_range(seed: int) -> None:
    luma = _luma(seed, size=128)
    assert luma.shape == (128, 128)
    assert 0.15 < luma.mean() < 0.9


# --- 1. the lattice period is the step, and it scales with the render -------


@pytest.mark.parametrize("seed", SEEDS)
def test_lattice_period_is_the_step(seed: int) -> None:
    diam, step, brick = _lattice(seed)
    lo, hi = metal.ENGINE_TURN_DIAMETER_MM
    frac = diam / SIZE * metal.ENGINE_TURN_PLATE_MM
    assert lo <= frac <= hi
    assert 0.4 <= step / diam <= 0.7

    luma = _luma(seed)
    period_x, power_x = _fft_period(luma, axis=1)
    assert power_x > 4.0, "no lattice peak at all"
    assert abs(period_x - step) / step < 0.06, (period_x, step)

    # Across the rows the fundamental is the step for a plain grid; brick rows
    # repeat every *two* rows, so twice the step is equally correct there.
    period_y, power_y = _fft_period(luma, axis=0)
    assert power_y > 4.0
    ratio = period_y / step
    assert min(abs(ratio - 1.0), abs(ratio - 2.0)) < 0.12, (period_y, step, brick)


def test_disc_size_scales_with_the_render() -> None:
    """Twice the pixels is the same plate photographed larger, not finer work."""
    for seed in SEEDS:
        small = _lattice(seed, 256)
        large = _lattice(seed, 512)
        assert large[2] == small[2]
        for a, b in zip(large[:2], small[:2], strict=False):
            assert a == pytest.approx(2.0 * b, rel=1e-6)


@pytest.mark.parametrize("seed", (0, 3, 6))
def test_measured_period_scales_with_the_render(seed: int) -> None:
    p256, _ = _fft_period(_luma(seed, 256), axis=1)
    p512, _ = _fft_period(_luma(seed, 512), axis=1)
    assert p512 / p256 == pytest.approx(2.0, rel=0.08)


# --- 2. last-one-wins depth ordering ---------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_ordering_is_last_one_wins(seed: int) -> None:
    diam, step, brick = _lattice(seed)
    _, _, ci, cj = metal.engine_turn_frame((SIZE, SIZE), 0.5 * diam, step, brick)
    ni, nj = _nominal_cell((SIZE, SIZE), step, brick)

    # A cell's own disc always reaches its own pixels (the corner is at most
    # 0.707 steps out, and the radius is at least 0.71 steps), so the winner
    # can only ever be a *later* disc -- never an earlier one.
    earlier = (cj < nj) | ((cj == nj) & (ci < ni))
    assert not earlier.any()

    # And it frequently is a later one: 30-60% overlap means a large share of
    # the plate belongs to a disc cut after the one centred nearest.
    occluded = float(((ci != ni) | (cj != nj)).mean())
    assert 0.2 < occluded < 0.95, occluded


@pytest.mark.parametrize("seed", SEEDS)
def test_occlusion_bites_the_later_side(seed: int) -> None:
    """The bite is taken out of the side the *next* discs come from.

    That asymmetry is the signature of the ordering, and it is what a
    nearest-centre (Voronoi) assignment would not produce: there the surviving
    region would be a symmetric cell centred on its own pivot. Here every mark
    keeps its up-left side and loses its down-right side to the discs cut after
    it, so the mean local offset over the whole plate is negative in both axes.
    """
    diam, step, brick = _lattice(seed)
    lx, ly, ci, cj = metal.engine_turn_frame((SIZE, SIZE), 0.5 * diam, step, brick)
    radius = 0.5 * diam
    assert float(lx.mean()) / radius < -0.15
    assert float(ly.mean()) / radius < -0.20

    # And no mark survives whole: the overlap costs every one of them area.
    _, areas = np.unique(cj * 100000 + ci, return_counts=True)
    full = np.pi * radius * radius
    assert float(areas.max()) < 0.75 * full
    interior = areas[areas > 0.1 * full]
    assert interior.size > 20
    assert 0.15 * full < float(interior.mean()) < 0.7 * full


# --- 3. the anisotropy direction is per disc -------------------------------


@pytest.mark.parametrize("seed", (0, 3, 5))
def test_anisotropy_turns_within_each_disc_and_jumps_between_them(seed: int) -> None:
    diam, step, brick = _lattice(seed)
    lx, ly, ci, cj = metal.engine_turn_frame((SIZE, SIZE), 0.5 * diam, step, brick)
    # The direction handed to shade() is the local tangent (-ly, lx).
    ang = np.arctan2(lx, -ly)

    key = cj * 100000 + ci
    uniq, counts = np.unique(key, return_counts=True)
    busiest = uniq[np.argsort(-counts)[:12]]
    spans = [np.ptp(ang[key == u]) for u in busiest]
    assert float(np.median(spans)) > 4.0, spans

    dang = np.abs(
        np.arctan2(np.sin(ang[:, 1:] - ang[:, :-1]), np.cos(ang[:, 1:] - ang[:, :-1]))
    )
    seam = (ci[:, 1:] != ci[:, :-1]) | (cj[:, 1:] != cj[:, :-1])
    assert float(np.median(dang[seam])) > 0.5
    assert float(np.median(dang[~seam])) < 0.15


def test_neighbouring_discs_get_different_phases() -> None:
    """Per-cell phase: adjacent marks must not be clones of one another."""
    i, j = np.meshgrid(np.arange(-20, 20), np.arange(-20, 20))
    phase = metal._cell_random(i, j, 12345, 0)
    assert phase.min() >= 0.0 and phase.max() < 1.0
    # Deterministic, and independent of the neighbours and of the stream.
    assert np.array_equal(phase, metal._cell_random(i, j, 12345, 0))
    assert not np.allclose(phase, metal._cell_random(i, j, 12345, 1))
    assert np.median(np.abs(np.diff(phase, axis=1))) > 0.2
    assert np.median(np.abs(np.diff(phase, axis=0))) > 0.2
    # Roughly uniform, so no disc population is favoured.
    assert 0.4 < float(phase.mean()) < 0.6
    assert float(phase.std()) > 0.24
