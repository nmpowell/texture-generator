"""Stored recipes may not silently replay with different morphology resources."""

import json

import pytest

from texture_generators import export_material, generate_maps, replay_material
from texture_generators.core import conductor, energy_compensation


@pytest.mark.parametrize("alteration", ["missing", "changed", "extra"])
@pytest.mark.parametrize("kind", ["surface", "renderer"])
def test_replay_rejects_a_different_resource_identity(tmp_path, alteration, kind):
    maps = generate_maps(
        "metal",
        size=(6, 5),
        variant="galvanised",
        seed=42,
        galvanised={"size_mm": (12, 10)},
    )
    path = export_material(maps, tmp_path / "bundle")
    manifest = json.loads(path.read_text())
    field = f"{kind}_resource_hashes"
    hashes = manifest["metadata"][field]
    if alteration == "missing":
        del manifest["metadata"][field]
    elif alteration == "changed":
        hashes[next(iter(hashes))] = "0" * 64
    else:
        hashes["unexpected_resource.json"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"{kind} resource hashes"):
        replay_material(path)


def test_energy_table_identity_rejects_changed_packaged_bytes(tmp_path, monkeypatch):
    root = tmp_path / "data" / "galvanised" / "energy"
    root.mkdir(parents=True)
    original = energy_compensation.files("texture_generators").joinpath(
        "data", "galvanised", "energy"
    )
    (root / "manifest.json").write_bytes(
        original.joinpath("manifest.json").read_bytes()
    )
    (root / "ggx_directional_albedo.npz").write_bytes(b"changed table")
    with monkeypatch.context() as patch:
        patch.setattr(energy_compensation, "files", lambda package: tmp_path)
        energy_compensation.energy_resource_hashes.cache_clear()
        with pytest.raises(RuntimeError, match="energy table checksum"):
            energy_compensation.energy_resource_hashes()
    energy_compensation.energy_resource_hashes.cache_clear()


def test_optical_identity_rejects_changed_packaged_source(tmp_path, monkeypatch):
    original = conductor._resource
    changed = tmp_path / "Werner.yml"
    changed.write_bytes(original("sources/Werner.yml").read_bytes() + b"\n# changed\n")
    with monkeypatch.context() as patch:
        patch.setattr(
            conductor,
            "_resource",
            lambda name: changed if name == "sources/Werner.yml" else original(name),
        )
        conductor.resource_hashes.cache_clear()
        with pytest.raises(RuntimeError, match=r"checksum mismatch: Werner\.yml"):
            conductor.resource_hashes()
    conductor.resource_hashes.cache_clear()


def test_optical_identity_rejects_inconsistent_manifest_f0(monkeypatch):
    changed = dict(conductor._manifest())
    changed["zinc_f0_linear_srgb"] = [0.1, 0.2, 0.3]
    with monkeypatch.context() as patch:
        patch.setattr(conductor, "_manifest", lambda: changed)
        conductor.resource_hashes.cache_clear()
        with pytest.raises(RuntimeError, match="manifest F0 differs"):
            conductor.resource_hashes()
    conductor.resource_hashes.cache_clear()
