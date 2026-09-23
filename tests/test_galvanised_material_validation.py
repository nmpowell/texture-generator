"""Bounded validation retains the rich material's exact public contract."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from texture_generators.core import material as material_module
from texture_generators.core.material import LOBE_DTYPE, MaterialMaps


@pytest.fixture(autouse=True)
def tiny_validation_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(material_module, "_SPATIAL_BLOCK_PIXELS", 9)
    monkeypatch.setattr(material_module, "_RECORD_BLOCK", 7)
    monkeypatch.setattr(material_module, "_PIXEL_BUCKET", 5)
    monkeypatch.setattr(material_module, "_BUCKET_READ_BLOCK", 6)
    monkeypatch.setattr(material_module, "_MAX_OPEN_BUCKET_FILES", 2)


def _mixed_records() -> np.ndarray:
    count = 6 * 7
    records = np.zeros(count * 3, dtype=LOBE_DTYPE)
    records["pixel_index"] = np.repeat(np.arange(count, dtype=np.uint32), 3)
    records["material_id"] = np.tile(np.array([0, 0, 1], dtype=np.uint8), count)
    records["weight"] = np.tile(np.array([0.25, 0.25, 0.5], dtype=np.float32), count)
    records["normal_ts"] = (0, 0, 1)
    records["tangent_ts"] = (1, 0, 0)
    records["alpha_t"] = 0.2
    records["alpha_b"] = 0.3
    records["optical_parameter_index"] = records["material_id"]
    return records


def _maps(records: np.ndarray, *, all_coverages: bool = True) -> MaterialMaps:
    arrays: dict[str, np.ndarray] = {
        "height_um": np.zeros((6, 7), dtype=np.float32),
        "metallic": np.full((6, 7), 0.5, dtype=np.float32),
    }
    if all_coverages:
        arrays["patina_coverage"] = np.full((6, 7), 0.5, dtype=np.float32)
        arrays["white_stain_coverage"] = np.zeros((6, 7), dtype=np.float32)
    return MaterialMaps(arrays, {"representation": "rich"}, records)


def test_arbitrary_lobe_order_and_partial_coverage_remain_valid() -> None:
    records = _mixed_records()
    shuffled = records[np.random.default_rng(42).permutation(len(records))]
    _maps(shuffled).validate()
    _maps(shuffled, all_coverages=False).validate()
    _maps(records[::-1], all_coverages=False).validate()


def test_missing_pixel_and_five_same_material_lobes_fail_across_blocks() -> None:
    records = _mixed_records()
    missing = records[records["pixel_index"] != 19]
    with pytest.raises(ValueError, match="every pixel"):
        _maps(missing)

    extra = np.repeat(records[records["pixel_index"] == 19][0:1], 3)
    five = np.concatenate((records, extra))
    five["weight"][five["pixel_index"] == 19] = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]
    shuffled = five[np.random.default_rng(7).permutation(len(five))]
    with pytest.raises(ValueError, match="at most four lobes"):
        _maps(shuffled)


def test_readonly_noncontiguous_mapped_channels_and_tolerance(tmp_path: Path) -> None:
    path = tmp_path / "wide.npy"
    wide = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(6, 14))
    wide[:, ::2] = 0.5
    narrow = wide[:, ::2]
    narrow.setflags(write=False)
    records = _mixed_records()
    material = MaterialMaps(
        {"metallic": narrow}, {"representation": "rich"}, records[::-1]
    )
    assert np.shares_memory(material["metallic"], wide)
    material.validate()

    # The reduction uses the existing absolute tolerance for float32 weights.
    near = np.zeros(6 * 7 * 2, dtype=LOBE_DTYPE)
    near["pixel_index"] = np.repeat(np.arange(42, dtype=np.uint32), 2)
    near["weight"] = 0.5
    near["weight"][::2] += np.float32(4e-7)
    near["normal_ts"] = (0, 0, 1)
    near["tangent_ts"] = (1, 0, 0)
    near["alpha_t"] = 0.2
    near["alpha_b"] = 0.3
    MaterialMaps(
        {"height_um": np.zeros((6, 7), dtype=np.float32)},
        {"representation": "rich"},
        near[::-1],
    )


def test_rejects_invalid_values_in_later_spatial_block() -> None:
    height = np.zeros((6, 7), dtype=np.float32)
    height[-1, -1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        MaterialMaps({"height_um": height}, {})
    distance = np.zeros((6, 7), dtype=np.float32)
    distance[-1, -1] = np.inf
    MaterialMaps({"boundary_distance_mm": distance}, {})
    invalid_distance = distance.copy()
    invalid_distance[-1, -1] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        MaterialMaps({"boundary_distance_mm": invalid_distance}, {})


def test_scratch_directory_is_removed_after_reduction_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tempfile.TemporaryDirectory
    created: list[Path] = []

    def tracked_directory(
        *args: object, **kwargs: object
    ) -> tempfile.TemporaryDirectory[str]:
        directory = original(*args, dir=tmp_path, **kwargs)
        created.append(Path(directory.name))
        return directory

    monkeypatch.setattr(
        material_module.tempfile, "TemporaryDirectory", tracked_directory
    )
    missing = _mixed_records()
    missing = missing[missing["pixel_index"] != 31]
    with pytest.raises(ValueError, match="every pixel"):
        _maps(missing)
    assert created and all(not path.exists() for path in created)


def test_optical_table_indices_and_material_identity_remain_checked() -> None:
    optics = [
        {"material_id": 0, "kind": "conductor", "roughness": 0.3, "anisotropy": 0.5},
        {
            "material_id": 1,
            "kind": "dielectric",
            "roughness": 0.7,
            "anisotropy": 0.0,
            "ior": 1.5,
            "diffuse_color_linear": [0.4, 0.4, 0.4],
        },
    ]
    arrays = {"height_um": np.zeros((6, 7), dtype=np.float32)}
    records = _mixed_records()
    MaterialMaps(
        arrays, {"representation": "rich", "optical_parameters": optics}, records
    )

    invalid_index = records.copy()
    invalid_index["optical_parameter_index"][8] = 2
    with pytest.raises(ValueError, match="outside the table"):
        MaterialMaps(
            arrays,
            {"representation": "rich", "optical_parameters": optics},
            invalid_index,
        )
    wrong_identity = records.copy()
    wrong_identity["optical_parameter_index"][8] = 0
    with pytest.raises(ValueError, match="differs from its optical parameter"):
        MaterialMaps(
            arrays,
            {"representation": "rich", "optical_parameters": optics},
            wrong_identity,
        )
