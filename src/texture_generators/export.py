"""Validated, reproducible galvanised material bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from texture_generators.core.material import (
    CANONICAL_CHANNELS,
    LOBE_DTYPE,
    MaterialMaps,
)
from texture_generators.materials.galvanised import GENERATOR_VERSION

SCHEMA_VERSION = 1
_PROFILES = {"lossless", "tiff"}
_DIAGNOSTICS = {"grain_id", "orientation_id"}


def validate_export_profile(profile: str) -> None:
    """Reject unsupported profiles and missing optional I/O before sampling."""
    if profile not in _PROFILES:
        raise ValueError(
            f"unsupported export profile {profile!r}; choose lossless or tiff"
        )
    if profile == "tiff":
        try:
            import tifffile  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "TIFF export requires the optional tifffile package "
                "(install texture-generator[export])"
            ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "export metadata must contain only finite JSON data"
        ) from error


def _validated_metadata(maps: MaterialMaps) -> dict[str, Any]:
    metadata = _json_copy(dict(maps.metadata))
    required = {
        "generator_version",
        "config",
        "material_key",
        "representation",
        "optical_parameters",
        "resource_hashes",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(
            f"material metadata missing export provenance: {sorted(missing)!r}"
        )
    if metadata["generator_version"] != GENERATOR_VERSION:
        raise ValueError(
            f"unsupported generator_version: {metadata['generator_version']!r}"
        )
    if "schema_version" in metadata and (
        isinstance(metadata["schema_version"], bool)
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported metadata schema_version")
    if isinstance(metadata["material_key"], bool) or not isinstance(
        metadata["material_key"], int
    ):
        raise ValueError("metadata material_key must be an integer")
    if not 0 <= metadata["material_key"] < 1 << 64:
        raise ValueError("metadata material_key must be unsigned 64-bit")
    if not isinstance(metadata["config"], dict):
        raise ValueError("metadata config must be a resolved mapping")
    from texture_generators.materials.galvanised_config import GalvanisedConfig

    config = GalvanisedConfig.from_resolved_mapping(metadata["config"])
    if not isinstance(metadata["optical_parameters"], list):
        raise ValueError("metadata optical_parameters must be a list")
    if metadata["representation"] == "rich" and maps.lobes is None:
        raise ValueError("rich representation requires lobe records")
    if metadata["representation"] != config.representation:
        raise ValueError("metadata representation does not match resolved config")
    if not isinstance(metadata["resource_hashes"], dict):
        raise ValueError("metadata resource_hashes must be a mapping")
    metadata["schema_version"] = SCHEMA_VERSION
    return metadata


def _map_path(name: str, profile: str, array: np.ndarray) -> Path:
    directory = Path("diagnostics") if name in _DIAGNOSTICS else Path()
    suffix = ".tif" if profile == "tiff" and array.dtype == np.float32 else ".npy"
    return directory / f"{name}{suffix}"


def _write_npy(path: Path, array: np.ndarray) -> None:
    target = np.lib.format.open_memmap(
        path, mode="w+", dtype=array.dtype, shape=array.shape
    )
    np.copyto(target, array)
    target.flush()
    del target


def _write_tiff(path: Path, array: np.ndarray) -> None:
    import tifffile

    # IEEE float, no image-dependent normalization, display transfer or compression.
    tifffile.imwrite(
        path,
        array,
        photometric="rgb" if array.ndim == 3 and array.shape[-1] == 3 else "minisblack",
        planarconfig="contig",
        metadata=None,
        compression=None,
        byteorder="<",
    )


def _descriptor(
    path: Path, root: Path, array: np.ndarray, *, name: str
) -> dict[str, Any]:
    canonical = CANONICAL_CHANNELS[name]
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "units": canonical.units,
        "colour_space": "linear" if name == "base_color_linear" else "data",
    }


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    temporary = root / "material.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.rename(root / "material.json")


def _replaceable(destination: Path) -> bool:
    """Whether ``overwrite=True`` may replace ``destination``: a bundle or empty dir."""
    if destination.is_symlink() or not destination.is_dir():
        return False
    return (destination / "material.json").is_file() or not any(destination.iterdir())


def _publish(stage: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"material bundle already exists: {destination}")
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.old-", dir=destination.parent)
        )
        backup.rmdir()
        destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            backup.rename(destination)
            raise
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    else:
        stage.rename(destination)


def export_material(
    maps: MaterialMaps,
    path: str | os.PathLike[str],
    *,
    profile: str = "lossless",
    overwrite: bool = False,
    preview: Image.Image | None = None,
) -> Path:
    """Write a complete sibling-staged bundle and return its manifest path."""
    validate_export_profile(profile)
    if not isinstance(maps, MaterialMaps):
        raise TypeError("maps must be MaterialMaps")
    maps.validate()
    metadata = _validated_metadata(maps)
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"material bundle already exists: {destination}")
    if (destination.exists() or destination.is_symlink()) and not _replaceable(
        destination
    ):
        raise FileExistsError(
            f"refusing to overwrite {destination}: not a material bundle or empty directory"
        )
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"bundle parent does not exist: {destination.parent}")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "profile": profile,
            "metadata": metadata,
            "maps": {},
        }
        for name, array in maps.items():
            relative = _map_path(name, profile, array)
            output = stage / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix == ".tif":
                _write_tiff(output, array)
            else:
                _write_npy(output, array)
            manifest["maps"][name] = _descriptor(output, stage, array, name=name)
        if maps.lobes is not None:
            lobes_path = stage / "layers" / "lobes" / "records.npy"
            lobes_path.parent.mkdir(parents=True, exist_ok=True)
            _write_npy(lobes_path, maps.lobes)
            manifest["lobes"] = {
                "path": lobes_path.relative_to(stage).as_posix(),
                "sha256": _sha256(lobes_path),
                "shape": list(maps.lobes.shape),
                "dtype": _json_copy(LOBE_DTYPE.descr),
            }
        if preview is not None:
            if not isinstance(preview, Image.Image):
                raise TypeError("preview must be a PIL Image")
            preview_path = stage / "preview.png"
            preview.convert("RGB").save(preview_path)
            manifest["preview"] = {
                "path": "preview.png",
                "sha256": _sha256(preview_path),
                "colour_space": "sRGB",
            }
        # Validate data on disk before the completion marker exists.
        _read_bundle(stage, manifest, mmap_mode="r")
        _write_manifest(stage, manifest)
        _publish(stage, destination, overwrite)
        return destination / "material.json"
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _safe_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("manifest path must be a non-empty relative POSIX path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.split("/")
    ):
        raise ValueError(f"unsafe manifest path: {relative!r}")
    path = root / candidate
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"manifest path escapes bundle or is missing: {relative!r}")
    return path


def _verified_file(root: Path, descriptor: Mapping[str, Any]) -> Path:
    path = _safe_file(root, descriptor.get("path"))
    checksum = descriptor.get("sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or _sha256(path) != checksum
    ):
        raise ValueError(f"checksum mismatch for {descriptor.get('path')!r}")
    return path


def _load_array(path: Path, *, mmap_mode: Literal["r"] | None) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    if path.suffix == ".tif":
        validate_export_profile("tiff")
        import tifffile

        if mmap_mode == "r":
            try:
                return tifffile.memmap(path, mode="r")
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"TIFF map is not uncompressed, contiguous, memory-mappable float data: {path}"
                ) from error
        return tifffile.imread(path)
    raise ValueError(f"unsupported material map format: {path.suffix}")


def _read_bundle(
    root: Path, manifest: Mapping[str, Any], *, mmap_mode: Literal["r"] | None
) -> MaterialMaps:
    if (
        isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported material manifest schema_version")
    profile = manifest.get("profile")
    if not isinstance(profile, str):
        raise ValueError("material manifest profile must be a string")
    validate_export_profile(profile)
    metadata = manifest.get("metadata")
    descriptors = manifest.get("maps")
    if (
        not isinstance(metadata, dict)
        or not isinstance(descriptors, dict)
        or not descriptors
    ):
        raise ValueError("material manifest requires metadata and maps")
    if metadata.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("unsupported material generator_version")
    arrays: dict[str, np.ndarray] = {}
    for name, descriptor in descriptors.items():
        if name not in CANONICAL_CHANNELS or not isinstance(descriptor, dict):
            raise ValueError(f"invalid map descriptor: {name!r}")
        path = _verified_file(root, descriptor)
        expected_path = _map_path(
            name, profile, np.empty((), dtype=CANONICAL_CHANNELS[name].dtype)
        ).as_posix()
        if descriptor["path"] != expected_path:
            raise ValueError(f"unexpected format for {name!r}")
        array = _load_array(path, mmap_mode=mmap_mode)
        canonical = CANONICAL_CHANNELS[name]
        if (
            descriptor.get("shape") != list(array.shape)
            or descriptor.get("dtype") != array.dtype.str
            or array.dtype != canonical.dtype
            or descriptor.get("units") != canonical.units
            or descriptor.get("colour_space")
            != ("linear" if name == "base_color_linear" else "data")
        ):
            raise ValueError(f"map descriptor mismatch for {name!r}")
        arrays[name] = array
    lobes = None
    if "lobes" in manifest:
        lobe_descriptor = manifest["lobes"]
        if not isinstance(lobe_descriptor, dict):
            raise ValueError("invalid lobes descriptor")
        lobe_path = _verified_file(root, lobe_descriptor)
        if lobe_descriptor["path"] != "layers/lobes/records.npy":
            raise ValueError("lobe records require NPY")
        lobes = np.load(lobe_path, allow_pickle=False, mmap_mode=mmap_mode)
        if (
            lobe_descriptor.get("shape") != list(lobes.shape)
            or lobe_descriptor.get("dtype") != _json_copy(LOBE_DTYPE.descr)
            or lobes.dtype != LOBE_DTYPE
        ):
            raise ValueError("lobe descriptor mismatch")
    if metadata.get("representation") == "rich" and lobes is None:
        raise ValueError("rich material bundle is missing lobe records")
    if "preview" in manifest:
        preview = manifest["preview"]
        if not isinstance(preview, dict) or preview.get("colour_space") != "sRGB":
            raise ValueError("invalid preview descriptor")
        if preview.get("path") != "preview.png":
            raise ValueError("preview must use preview.png")
        _verified_file(root, preview)
    result = MaterialMaps(arrays, metadata, lobes)
    _validated_metadata(result)
    return result


def load_material(
    path: str | os.PathLike[str], *, mmap_mode: str | None = None
) -> MaterialMaps:
    """Validate and load a material bundle, optionally memory-mapping NPY arrays."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path /= "material.json"
    if manifest_path.name != "material.json":
        raise ValueError("material path must be a bundle directory or material.json")
    if mmap_mode not in (None, "r"):
        raise ValueError("mmap_mode must be None or 'r'")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("material manifest must be a JSON object")
    return _read_bundle(
        manifest_path.parent, manifest, mmap_mode="r" if mmap_mode == "r" else None
    )


def replay_material(
    path: str | os.PathLike[str],
    *,
    size: tuple[int, int] | None = None,
    maps: tuple[str, ...] | None = None,
) -> MaterialMaps:
    """Resample the stored resolved recipe with its concrete material key."""
    stored = load_material(path, mmap_mode="r")
    metadata = stored.metadata
    from texture_generators.core.conductor import resource_hashes
    from texture_generators.core.energy_compensation import energy_resource_hashes
    from texture_generators.materials.galvanised import (
        build_state,
        sample_state,
        surface_resource_hashes,
    )
    from texture_generators.materials.galvanised_config import GalvanisedConfig

    recorded_hashes = metadata.get("resource_hashes")
    if not isinstance(recorded_hashes, dict) or recorded_hashes != dict(
        resource_hashes()
    ):
        raise ValueError(
            "material optical resource hashes do not match this installation"
        )
    if metadata.get("surface_resource_hashes") != surface_resource_hashes():
        raise ValueError(
            "material surface resource hashes do not match this installation"
        )
    if metadata.get("renderer_resource_hashes") != dict(energy_resource_hashes()):
        raise ValueError(
            "material renderer resource hashes do not match this installation"
        )
    config = GalvanisedConfig.from_resolved_mapping(metadata["config"])
    output_size = stored.size if size is None else size
    selected = tuple(stored.keys()) if maps is None else maps
    state = build_state(
        config,
        material_key=metadata["material_key"],
        seed=metadata.get("seed"),
        size=output_size,
    )
    return sample_state(state, output_size, maps=selected)
