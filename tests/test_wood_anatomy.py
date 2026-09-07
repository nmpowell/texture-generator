"""Tests for the vessel-pore anatomy behind the wood generator.

Checks the *statistics* of the pore streak layer rather than the pixels: the
layer is only worth anything if the pore area fraction it lays down matches the
species anatomy it claims to model, and that is measurable.
"""

from __future__ import annotations

import numpy as np

from texture_generators.core.noise import grid_coords
from texture_generators.materials.wood import ANATOMY, _pore_streaks

BOARD_MM = 225.0


def _pore_fields(
    species: str, seed: int = 0, size: int = 256, n_rings: float = 12.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run :func:`_pore_streaks` over a plain ring stack for one species."""
    rng = np.random.default_rng(seed)
    u, v = grid_coords((size, size))
    spacing = 1.0 / n_rings
    g = np.mod(v / np.float32(spacing), 1.0).astype(np.float32)
    return _pore_streaks(
        u,
        v,
        g,
        rng,
        species,
        px_per_mm=size / BOARD_MM,
        mm_per_unit=BOARD_MM,
        px_per_unit=float(size),
        spacing=spacing,
        along_axis=1,
    )


def _coverage(species: str, seeds: int = 3) -> float:
    """Mean pore area fraction over the whole face."""
    return float(np.mean([_pore_fields(species, s)[0].mean() for s in range(seeds)]))


def test_pore_coverage_follows_the_species_pore_class() -> None:
    """Pore area fraction must separate the three pore classes, and pine gets none.

    Coverage on a sawn face is the transverse vessel area fraction ``n * d^2``
    (see :func:`_pore_streaks`), so it is set by anatomy, not by taste. Oak
    crowds its coarse vessels into a narrow earlywood band and so covers little
    of the *face*; cherry spreads fine ones over all of it. If these collapse
    together the layer has stopped being species-specific and every board is
    reading as the same wood again.
    """
    oak = _coverage("oak")
    cherry = _coverage("cherry")

    assert ANATOMY["oak"]["pore_class"] == "ring-porous"
    assert ANATOMY["cherry"]["pore_class"] == "diffuse-porous"
    # Diffuse pores cover several times more of the face than a banded ring-
    # porous species, whose pores are confined to ~1 mm of each ring.
    assert cherry > 3.0 * oak
    assert 0.01 < oak < 0.15
    assert 0.15 < cherry < 0.45

    # Pine is a softwood: no vessels at all, so no streaks, no troughs and no
    # darkening. That absence is what makes it read as a softwood.
    mask, delta_l, depth_mm = _pore_fields("pine")
    assert ANATOMY["pine"]["pore_class"] == "softwood"
    assert float(mask.max()) == 0.0
    assert float(delta_l.max()) == 0.0
    assert float(depth_mm.max()) == 0.0


def test_ring_porous_pores_stay_in_a_narrow_earlywood_band() -> None:
    """Oak's coarse pores crowd the first millimetre or so of each ring.

    The band's width is set in millimetres, not as a fraction of the ring, so a
    wide ring means more latewood rather than a wider pore band. Checked at two
    ring widths: the in-band coverage must not fall when the rings widen.
    """
    covs = {}
    for n_rings in (8.0, 24.0):
        mask, _, depth_mm = _pore_fields("oak", seed=1, n_rings=n_rings)
        ring_mm = BOARD_MM / n_rings
        _, v = grid_coords((mask.shape[0], mask.shape[1]))
        pos_mm = np.mod(v * np.float32(n_rings), 1.0) * np.float32(ring_mm)
        band = pos_mm < np.float32(1.5)
        covs[n_rings] = (float(mask[band].mean()), float(mask[~band].mean()))
        # And the band is where the relief is: oak's earlywood troughs run
        # 60-200 um against 10-30 um in the latewood, over more of the area.
        assert depth_mm[band].mean() > 4.0 * depth_mm[~band].mean()

    for in_band, outside in covs.values():
        assert in_band > 4.0 * outside
    assert covs[24.0][0] > 0.6 * covs[8.0][0]
