"""Independent midpoint-height checks for the LOD diagnostic tool."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from texture_generators.materials.galvanised import build_state, sample_state
from texture_generators.materials.galvanised_config import GalvanisedConfig

_CHECKER_PATH = Path(__file__).resolve().parents[1] / "tools/galvanised/check_lod.py"
_SPEC = spec_from_file_location("galvanised_check_lod", _CHECKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_lod = module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_lod)


def test_independent_four_midpoints_are_phase_aligned_and_tile_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lx, ly = 7.0, 5.0
    width, height = 7, 5
    state = SimpleNamespace(config=SimpleNamespace(size_mm=(lx, ly)))

    def analytic_points(
        _state: object, x: np.ndarray, y: np.ndarray
    ) -> dict[str, np.ndarray]:
        return {
            "height_um": 3.0 * np.sin(2 * np.pi * 3 * x / lx)
            + 2.0 * np.cos(2 * np.pi * 2 * y / ly)
        }

    monkeypatch.setattr(check_lod, "sample_points", analytic_points)
    four = check_lod._height(state, (width, height), chunk_rows=5, quadrature=4)
    expected = np.empty_like(four)
    for row in range(height):
        for col in range(width):
            samples = [
                analytic_points(
                    state,
                    np.asarray(lx * (col + (sx + 0.5) / 4) / width),
                    np.asarray(ly * (row + (sy + 0.5) / 4) / height),
                )["height_um"]
                for sy in range(4)
                for sx in range(4)
            ]
            expected[row, col] = np.mean(samples)
    np.testing.assert_array_equal(four, expected)
    monkeypatch.setattr(check_lod, "_HEIGHT_SAMPLE_LIMIT", 32)
    tiled = check_lod._height(state, (width, height), chunk_rows=5, quadrature=4)
    np.testing.assert_array_equal(tiled, four)
    two = check_lod._height(state, (width, height), chunk_rows=5, quadrature=2)
    assert np.max(np.abs(four - two)) > 0.01


@pytest.mark.parametrize("representation,quadrature", [("rich", 2), ("single_lobe", 4)])
def test_checker_height_matches_production_sampler(
    representation: str,
    quadrature: int,
) -> None:
    config = GalvanisedConfig.from_mapping(
        {"representation": representation, "size_mm": (6.0, 5.0)}
    )
    state = build_state(config, seed=4, size=(7, 5))
    assert check_lod._height_quadrature(state) == quadrature
    direct = check_lod._height(state, (7, 5), chunk_rows=2, quadrature=quadrature)
    sampled = sample_state(state, size=(7, 5), maps=["height_um"])
    np.testing.assert_array_equal(direct, sampled["height_um"])


def test_height_only_report_records_four_by_four_and_independent_crosscheck() -> None:
    report = check_lod.measure(
        seed=4,
        preset="regular",
        size_mm=(6.0, 5.0),
        resolutions=[(3, 3), (6, 6)],
        render_size=(3, 3),
        height_only=True,
    )
    assert report["check_version"] == "3"
    assert report["height_sampling"]["quadrature_per_axis"] == 4
    assert report["height_sampler_crosscheck_size"] == [3, 3]
    assert report["height_sampler_crosscheck_rms_um"] == 0.0
    assert not report["fail_flags"]["height_sampler_crosscheck_failed"]
