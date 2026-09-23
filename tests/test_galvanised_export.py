"""Serialization, integrity, and replay contract tests for galvanised maps."""

from __future__ import annotations

import builtins
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from texture_generators.core.conductor import resource_hashes
from texture_generators.core.material import LOBE_DTYPE, MaterialMaps
from texture_generators.export import (
    export_material,
    load_material,
    replay_material,
    validate_export_profile,
)
from texture_generators.materials.galvanised import GENERATOR_VERSION
from texture_generators.materials.galvanised_config import GalvanisedConfig


def _fixture(*, rich: bool = True) -> MaterialMaps:
    shape = (3, 4)
    normal = np.zeros((*shape, 3), dtype=np.float32)
    normal[..., 0] = -0.00002
    normal[..., 2] = np.sqrt(np.float32(1.0) - normal[..., 0] ** 2)
    arrays = {
        "height_um": np.arange(12, dtype=np.float32).reshape(shape) - 6,
        "normal_ts": normal,
        "base_color_linear": np.full((*shape, 3), 0.8, dtype=np.float32),
        "grain_id": np.arange(12, dtype=np.uint32).reshape(shape),
        "metallic": np.ones(shape, dtype=np.float32),
    }
    lobes = None
    if rich:
        lobes = np.zeros(24, dtype=LOBE_DTYPE)
        lobes["pixel_index"] = np.repeat(np.arange(12, dtype=np.uint32), 2)
        lobes["weight"] = 0.5
        lobes["normal_ts"][0::2] = (0.0001, 0, np.sqrt(1 - 0.0001**2))
        lobes["normal_ts"][1::2] = (-0.0001, 0, np.sqrt(1 - 0.0001**2))
        lobes["tangent_ts"] = (0, 1, 0)
        lobes["alpha_t"][0::2] = 0.1
        lobes["alpha_t"][1::2] = 0.2
        lobes["alpha_b"][0::2] = 0.2
        lobes["alpha_b"][1::2] = 0.1
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "config": GalvanisedConfig(
            size_mm=(4, 3), representation="rich" if rich else "single_lobe"
        ).to_mapping(),
        "material_key": 123456,
        "representation": "rich" if rich else "single_lobe",
        "optical_parameters": [
            {
                "material_id": 0,
                "kind": "conductor",
                "roughness": 0.3,
                "anisotropy": 0.55,
            }
        ],
        "resource_hashes": dict(resource_hashes()),
        "seed": 42,
        "size_mm": [4, 3],
    }
    return MaterialMaps(arrays, metadata, lobes)


def test_lossless_selected_channels_and_rich_records_roundtrip(tmp_path: Path) -> None:
    original = _fixture()
    manifest_path = export_material(original, tmp_path / "bundle")
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["maps"]) == set(original)
    assert manifest["maps"]["grain_id"]["path"] == "diagnostics/grain_id.npy"
    assert manifest["lobes"]["path"] == "layers/lobes/records.npy"
    loaded = load_material(manifest_path, mmap_mode="r")
    assert tuple(loaded) == tuple(original)
    for name in original:
        np.testing.assert_array_equal(loaded[name], original[name])
        assert not loaded[name].flags.writeable
    assert loaded["height_um"].min() < 0
    assert loaded["grain_id"].dtype == np.uint32
    assert loaded.lobes is not None
    np.testing.assert_array_equal(loaded.lobes, original.lobes)
    assert loaded.metadata["config"] == original.metadata["config"]


def test_refuses_overwrite_and_preserves_old_bundle_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from texture_generators import export as export_module

    old = _fixture()
    path = tmp_path / "bundle"
    export_material(old, path)
    before = (path / "material.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_material(old, path)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected interrupted write")

    monkeypatch.setattr(export_module, "_write_npy", fail_write)
    with pytest.raises(OSError, match="interrupted"):
        export_material(old, path, overwrite=True)
    assert (path / "material.json").read_bytes() == before
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_overwrite_publish_failure_restores_old_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bundle"
    export_material(_fixture(), path)
    before = (path / "material.json").read_bytes()
    original_rename = Path.rename

    def fail_publish(source: Path, target: Path) -> Path:
        if source.name.startswith(".bundle.staging-") and target == path:
            raise OSError("injected publish failure")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        export_material(_fixture(), path, overwrite=True)
    assert (path / "material.json").read_bytes() == before
    assert not list(tmp_path.glob(".bundle.old-*"))


def test_explicit_overwrite_replaces_bundle(tmp_path: Path) -> None:
    path = tmp_path / "bundle"
    original = _fixture(rich=False)
    export_material(original, path)
    replacement = _fixture(rich=True)
    export_material(replacement, path, overwrite=True)
    loaded = load_material(path)
    assert loaded.lobes is not None
    np.testing.assert_array_equal(loaded.lobes, replacement.lobes)
    assert not list(tmp_path.glob(".bundle.old-*"))


def test_checksum_and_path_escape_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bundle"
    manifest_path = export_material(_fixture(), path)
    (path / "height_um.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_material(path)

    manifest = json.loads(manifest_path.read_text())
    manifest["maps"]["height_um"]["path"] = "../outside.npy"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe manifest path"):
        load_material(path)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bundle"
    manifest_path = export_material(_fixture(), path)
    outside = tmp_path / "outside.npy"
    (path / "height_um.npy").rename(outside)
    (path / "height_um.npy").symlink_to(outside)
    manifest = json.loads(manifest_path.read_text())
    manifest["maps"]["height_um"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes bundle"):
        load_material(path)


def test_unknown_profiles_fail_preflight() -> None:
    with pytest.raises(ValueError, match="unsupported export profile"):
        validate_export_profile("jpeg")


def test_tiff_dependency_fails_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_tifffile(name: str, *args: object, **kwargs: object) -> object:
        if name == "tifffile":
            raise ImportError("simulated missing optional package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tifffile)
    with pytest.raises(RuntimeError, match="requires the optional tifffile"):
        validate_export_profile("tiff")


def test_tiff_exact_float32_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("tifffile")
    original = _fixture(rich=False)
    export_material(original, tmp_path / "tiff", profile="tiff")
    loaded = load_material(tmp_path / "tiff")
    for name in original:
        np.testing.assert_array_equal(loaded[name], original[name])
    mapped = load_material(tmp_path / "tiff", mmap_mode="r")
    assert isinstance(mapped["normal_ts"].base, np.memmap)
    checker = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "tools"
            / "galvanised"
            / "check_export.py"
        )
    )
    checked = checker["check_bundle"](tmp_path / "tiff")
    assert "normal_ts" in checked
    decoded = checker["decode_float_tiff"](tmp_path / "tiff" / "normal_ts.tif")
    np.testing.assert_array_equal(decoded, original["normal_ts"])


def test_replay_uses_stored_key_and_resolved_config(tmp_path: Path) -> None:
    from texture_generators.materials.galvanised import build_state, sample_state

    config = GalvanisedConfig(size_mm=(12.0, 9.0), representation="single_lobe")
    state = build_state(config, material_key=98765, seed=42, size=(6, 5))
    original = sample_state(state, (6, 5), maps=("height_um", "grain_id"))
    export_material(original, tmp_path / "replay")
    replayed = replay_material(tmp_path / "replay")
    assert replayed.metadata["material_key"] == 98765
    assert replayed.metadata["seed"] == 42
    for name in original:
        np.testing.assert_array_equal(replayed[name], original[name])
    resized = replay_material(tmp_path / "replay", size=(8, 6), maps=("height_um",))
    assert resized.size == (8, 6)
    assert tuple(resized) == ("height_um",)
    assert resized.metadata["material_key"] == 98765


def test_replay_rejects_changed_optical_resources(tmp_path: Path) -> None:
    manifest_path = export_material(_fixture(rich=False), tmp_path / "bundle")
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["resource_hashes"] = {}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="optical resource hashes"):
        replay_material(tmp_path / "bundle")


@pytest.mark.parametrize("preset", ["regular", "wet_storage"])
@pytest.mark.parametrize("selected", [None, ("height_um",)])
def test_streamed_surface_bundle_and_replay(
    tmp_path: Path, preset: str, selected: tuple[str, ...] | None
) -> None:
    from texture_generators.materials.galvanised import build_state, sample_state

    state = build_state(
        GalvanisedConfig(preset=preset, size_mm=(12.0, 9.0)),
        material_key=12345,
        seed=42,
        size=(16, 12),
    )
    sampled = sample_state(
        state, (16, 12), maps=selected, output_dir=tmp_path / "sampled"
    )
    manifest = export_material(sampled, tmp_path / "bundle")
    loaded = load_material(manifest, mmap_mode="r")
    assert tuple(loaded) == tuple(sampled)
    assert all(isinstance(loaded[name].base, np.memmap) for name in loaded)
    for name in sampled:
        np.testing.assert_array_equal(loaded[name], sampled[name])
    assert loaded.lobes is not None and sampled.lobes is not None
    np.testing.assert_array_equal(loaded.lobes, sampled.lobes)
    replayed = replay_material(manifest)
    for name in sampled:
        np.testing.assert_array_equal(replayed[name], sampled[name])
    resized = replay_material(manifest, size=(20, 15))
    assert resized.size == (20, 15)
    assert tuple(resized) == tuple(sampled)
