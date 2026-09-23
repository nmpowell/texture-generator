"""Independently decode the uncompressed float TIFFs in a galvanised bundle.

This deliberately uses only Python's standard library and NumPy, rather than
the optional tifffile writer/reader. It checks the TIFF storage contract as
well as the manifest checksums and expected array dimensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_float_tiff(path: Path) -> np.ndarray:
    """Read classic, little-endian, contiguous, uncompressed IEEE float TIFF."""
    data = path.read_bytes()
    if data[:4] != b"II*\x00":
        raise ValueError("expected classic little-endian TIFF")
    ifd_offset = struct.unpack_from("<I", data, 4)[0]
    count = struct.unpack_from("<H", data, ifd_offset)[0]
    tags: dict[int, list[int]] = {}
    type_sizes = {3: 2, 4: 4}
    for index in range(count):
        entry = ifd_offset + 2 + index * 12
        tag, typ, length, value = struct.unpack_from("<HHII", data, entry)
        if typ not in type_sizes:
            continue
        byte_count = type_sizes[typ] * length
        start = entry + 8 if byte_count <= 4 else value
        code = "H" if typ == 3 else "I"
        tags[tag] = list(struct.unpack_from(f"<{length}{code}", data, start))

    def one(tag: int, default: int | None = None) -> int:
        values = tags.get(tag)
        if values is None:
            if default is not None:
                return default
            raise ValueError(f"missing TIFF tag {tag}")
        if len(values) != 1:
            raise ValueError(f"TIFF tag {tag} should be scalar")
        return values[0]

    width, height = one(256), one(257)
    components = one(277, 1)
    if (
        one(259, 1) != 1
        or one(284, 1) != 1
        or tags.get(339) not in ([3], [3] * components)
    ):
        raise ValueError("TIFF must have uncompressed contiguous IEEE float samples")
    if tags.get(258) != [32] * components:
        raise ValueError("TIFF samples must be float32")
    offsets, byte_counts = tags.get(273), tags.get(279)
    if not offsets or not byte_counts or len(offsets) != len(byte_counts):
        raise ValueError("invalid TIFF strip table")
    raw = b"".join(
        data[offset : offset + size]
        for offset, size in zip(offsets, byte_counts, strict=True)
    )
    expected = width * height * components * 4
    if len(raw) != expected:
        raise ValueError("TIFF strip byte count does not match dimensions")
    shape = (height, width) if components == 1 else (height, width, components)
    return np.frombuffer(raw, dtype="<f4").reshape(shape)


def check_bundle(path: Path) -> list[str]:
    """Independently check each TIFF channel and return checked map names."""
    root = path if path.is_dir() else path.parent
    manifest = json.loads((root / "material.json").read_text(encoding="utf-8"))
    if manifest["profile"] != "tiff":
        raise ValueError("independent TIFF check requires a TIFF profile bundle")
    checked = []
    for name, descriptor in manifest["maps"].items():
        relative = descriptor["path"]
        if not relative.endswith(".tif"):
            continue
        if relative.startswith("/") or any(
            part in ("", ".", "..") for part in relative.split("/")
        ):
            raise ValueError("unsafe TIFF path")
        source = root / relative
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError("TIFF path escapes bundle")
        if _sha256(source) != descriptor["sha256"]:
            raise ValueError(f"TIFF checksum mismatch: {name}")
        decoded = decode_float_tiff(source)
        if list(decoded.shape) != descriptor["shape"] or decoded.dtype != np.float32:
            raise ValueError(f"TIFF array descriptor mismatch: {name}")
        checked.append(name)
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    names = check_bundle(args.bundle)
    print(f"Checked {len(names)} float TIFF maps: {', '.join(names)}")


if __name__ == "__main__":
    main()
