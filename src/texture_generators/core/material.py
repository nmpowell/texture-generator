"""Validated sampled-material map and angular-lobe contracts."""

from __future__ import annotations

import copy
import json
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "CANONICAL_CHANNELS",
    "CHANNEL_DEPENDENCIES",
    "CHANNEL_DESCRIPTORS",
    "DEFAULT_CHANNELS",
    "DEFAULT_MAPS",
    "LOBE_DTYPE",
    "MAP_DEPENDENCIES",
    "MATERIAL_IDS",
    "ChannelDescriptor",
    "MaterialMaps",
    "validate_optical_parameters",
    "validate_sampling_request",
]


@dataclass(frozen=True)
class ChannelDescriptor:
    """Machine-readable contract for one canonical material channel."""

    dtype: np.dtype[Any]
    components: int
    units: str
    meaning: str
    value_range: tuple[float, float] | None = None


def _descriptor(
    dtype: np.dtype[Any] | type[np.generic],
    components: int,
    units: str,
    meaning: str,
    value_range: tuple[float, float] | None = None,
) -> ChannelDescriptor:
    return ChannelDescriptor(np.dtype(dtype), components, units, meaning, value_range)


CANONICAL_CHANNELS: Mapping[str, ChannelDescriptor] = MappingProxyType(
    {
        "height_um": _descriptor(np.float32, 1, "um", "signed final surface height"),
        "normal_ts": _descriptor(
            np.float32,
            3,
            "unit_vector",
            "canonical signed tangent-space normal",
            (-1.0, 1.0),
        ),
        "base_color_linear": _descriptor(
            np.float32,
            3,
            "linear_reflectance",
            "conductor F0 or dielectric diffuse colour",
            (0.0, 1.0),
        ),
        "roughness": _descriptor(
            np.float32, 1, "perceptual", "unresolved microfacet roughness", (0.0, 1.0)
        ),
        "anisotropy": _descriptor(
            np.float32, 1, "openpbr", "directional lobe inequality", (0.0, 1.0)
        ),
        "anisotropy_axis": _descriptor(
            np.float32,
            2,
            "doubled_angle_unit_axis",
            "clockwise raster-frame unoriented axis",
            (-1.0, 1.0),
        ),
        "metallic": _descriptor(
            np.float32, 1, "coverage", "visible exposed-zinc fraction", (0.0, 1.0)
        ),
        "patina_coverage": _descriptor(
            np.float32, 1, "coverage", "visible opaque patina fraction", (0.0, 1.0)
        ),
        "white_stain_coverage": _descriptor(
            np.float32, 1, "coverage", "visible wet-storage-stain fraction", (0.0, 1.0)
        ),
        "grain_id": _descriptor(
            np.uint32, 1, "identifier", "centre-sample crystallographic-domain ID"
        ),
        "boundary_distance_mm": _descriptor(
            np.float32,
            1,
            "mm",
            "distance to sampled grain boundary",
            (0.0, float("inf")),
        ),
        "substrate_height_um": _descriptor(
            np.float32, 1, "um", "substrate height before deposits"
        ),
        "deposit_height_um": _descriptor(
            np.float32, 1, "um", "weathering deposit height", (0.0, float("inf"))
        ),
        "orientation_id": _descriptor(
            np.uint32, 1, "identifier", "centre-sample orientation record ID"
        ),
        "finite_band_slope": _descriptor(
            np.float32, 2, "dimensionless", "resolved finite-band raster slopes"
        ),
    }
)
CHANNEL_DESCRIPTORS = CANONICAL_CHANNELS

CHANNEL_DEPENDENCIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "normal_ts": ("height_um",),
        "base_color_linear": ("metallic", "patina_coverage", "white_stain_coverage"),
        "metallic": ("patina_coverage", "white_stain_coverage"),
        "roughness": ("height_um", "patina_coverage", "white_stain_coverage"),
        "anisotropy": ("anisotropy_axis", "patina_coverage", "white_stain_coverage"),
    }
)
MAP_DEPENDENCIES = CHANNEL_DEPENDENCIES

DEFAULT_MAPS = (
    "height_um",
    "normal_ts",
    "base_color_linear",
    "roughness",
    "anisotropy",
    "anisotropy_axis",
    "metallic",
    "patina_coverage",
    "white_stain_coverage",
)
DEFAULT_CHANNELS = DEFAULT_MAPS


def validate_sampling_request(
    maps: Sequence[str] | None, chunk_size: int
) -> tuple[str, ...]:
    """Reject invalid selectors and tile sizes before constructing a state."""
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    selected = tuple(DEFAULT_MAPS if maps is None else maps)
    if (
        not selected
        or any(
            not isinstance(name, str) or name not in CANONICAL_CHANNELS
            for name in selected
        )
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("maps must contain distinct declared channel names")
    return selected


LOBE_DTYPE = np.dtype(
    [
        ("pixel_index", np.uint32),
        ("material_id", np.uint8),
        ("weight", np.float32),
        ("normal_ts", np.float32, (3,)),
        ("tangent_ts", np.float32, (3,)),
        ("alpha_t", np.float32),
        ("alpha_b", np.float32),
        ("optical_parameter_index", np.uint32),
    ]
)

MATERIAL_IDS: Mapping[str, int] = MappingProxyType(
    {"zinc": 0, "patina": 1, "white_stain": 2}
)
_COVERAGE_FOR_ID = {
    MATERIAL_IDS["zinc"]: "metallic",
    MATERIAL_IDS["patina"]: "patina_coverage",
    MATERIAL_IDS["white_stain"]: "white_stain_coverage",
}

_SPATIAL_BLOCK_PIXELS = 65_536
_RECORD_BLOCK = 131_072
_PIXEL_BUCKET = 131_072
_BUCKET_READ_BLOCK = 131_072
_MAX_OPEN_BUCKET_FILES = 32
_COMPACT_LOBE_DTYPE = np.dtype(
    [("pixel_index", "<u4"), ("material_id", "u1"), ("weight", "<f4")],
    align=False,
)


def _spatial_slices(shape: tuple[int, int]) -> Iterator[tuple[slice, slice]]:
    """Yield bounded views without flattening non-contiguous arrays."""
    height, width = shape
    if width <= _SPATIAL_BLOCK_PIXELS:
        rows = max(1, _SPATIAL_BLOCK_PIXELS // width)
        for y0 in range(0, height, rows):
            yield slice(y0, min(height, y0 + rows)), slice(0, width)
    else:
        for y in range(height):
            for x0 in range(0, width, _SPATIAL_BLOCK_PIXELS):
                yield slice(y, y + 1), slice(x0, min(width, x0 + _SPATIAL_BLOCK_PIXELS))


def _raster_window(array: np.ndarray, start: int, stop: int, width: int) -> np.ndarray:
    """Copy at most one pixel bucket from any 2D raster layout."""
    output = np.empty(stop - start, dtype=array.dtype)
    position = start
    while position < stop:
        row, column = divmod(position, width)
        count = min(stop - position, width - column)
        output[position - start : position - start + count] = array[
            row, column : column + count
        ]
        position += count
    return output


def validate_optical_parameters(parameters: object) -> None:
    """Check the optical table before a bundle is accepted or rendered."""
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("optical_parameters must be a non-empty list")
    for entry in parameters:
        if not isinstance(entry, Mapping):
            raise ValueError("optical parameters must be mappings")
        identifier = entry.get("material_id")
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, Integral)
            or identifier not in _COVERAGE_FOR_ID
        ):
            raise ValueError("optical material_id must identify zinc, patina or stain")
        expected_kind = "conductor" if identifier == 0 else "dielectric"
        if entry.get("kind") != expected_kind:
            raise ValueError("optical material_id/kind mismatch")
        for name in ("roughness", "anisotropy"):
            value = entry.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"optical {name} must be finite in [0,1]")
        if expected_kind == "dielectric":
            ior = entry.get("ior")
            if (
                isinstance(ior, bool)
                or not isinstance(ior, Real)
                or not np.isfinite(float(ior))
                or float(ior) < 1.0
            ):
                raise ValueError("dielectric optical ior must be finite and >=1")
        for name in ("diffuse_color_linear", "f0_linear_srgb"):
            if name not in entry:
                if name == "diffuse_color_linear" and expected_kind == "dielectric":
                    raise ValueError("dielectric optics require diffuse_color_linear")
                continue
            colour = np.asarray(entry[name])
            if (
                colour.shape != (3,)
                or colour.dtype.kind not in "fiu"
                or np.any(~np.isfinite(colour))
                or np.any((colour < 0.0) | (colour > 1.0))
            ):
                raise ValueError(f"optical {name} must be finite RGB in [0,1]")


def _readonly_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not array.flags.owndata and array.flags.writeable:
        array = np.array(array, copy=True)
    if array.flags.writeable:
        array.setflags(write=False)
    return array


def _json_safe(metadata: Mapping[str, Any]) -> dict[str, Any]:
    try:
        copied = copy.deepcopy(dict(metadata))
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be self-contained JSON-safe data") from error
    return copied


def _complete_metadata(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    result = _json_safe(metadata)
    unknown = set(arrays) - set(CANONICAL_CHANNELS)
    if unknown:
        raise ValueError(f"unknown material channels: {sorted(unknown)!r}")
    units = {name: CANONICAL_CHANNELS[name].units for name in arrays}
    dtypes = {name: CANONICAL_CHANNELS[name].dtype.name for name in arrays}
    for field, canonical in (("map_units", units), ("map_dtypes", dtypes)):
        supplied = result.get(field)
        if supplied is not None and supplied != canonical:
            raise ValueError(f"metadata {field} does not match selected channels")
        result[field] = canonical
    first = next(iter(arrays.values()))
    if first.ndim < 2:
        raise ValueError("material channels must have at least HxW dimensions")
    size = [first.shape[1], first.shape[0]]
    if "size" in result:
        try:
            supplied_size = list(result["size"])
        except TypeError as error:
            raise ValueError("metadata size must be a two-element sequence") from error
        if supplied_size != size:
            raise ValueError("metadata size does not match selected channels")
    result["size"] = size
    result.setdefault("coordinate_frame", "raster u right, v down; X=u, Y=Ly-v, +Z=out")
    result.setdefault(
        "normal_convention",
        "signed unit vector (-1e-3*dh/du, +1e-3*dh/dv, 1), normalised",
    )
    result.setdefault("linear_map_data", True)
    return result


@dataclass(frozen=True)
class MaterialMaps(Mapping[str, np.ndarray]):
    """Frozen, mapping-like sampled channels with optional rich lobe records."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    lobes: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {name: _readonly_array(value) for name, value in self.arrays.items()}
        if not arrays:
            raise ValueError("MaterialMaps requires at least one selected channel")
        metadata = _complete_metadata(self.metadata, arrays)
        lobes = None if self.lobes is None else _readonly_array(self.lobes)
        object.__setattr__(self, "arrays", MappingProxyType(arrays))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "lobes", lobes)
        self.validate()

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def __len__(self) -> int:
        return len(self.arrays)

    @property
    def size(self) -> tuple[int, int]:
        """Return sampled ``(width, height)``."""
        if not self.arrays:
            raise ValueError("MaterialMaps requires at least one channel")
        shape = next(iter(self.arrays.values())).shape
        return shape[1], shape[0]

    def validate(self) -> None:
        """Validate all selected channels, metadata, and rich lobe records."""
        if not self.arrays:
            raise ValueError("MaterialMaps requires at least one selected channel")
        spatial_shape: tuple[int, int] | None = None
        for name, array in self.arrays.items():
            if name not in CANONICAL_CHANNELS:
                raise ValueError(f"unknown material channel: {name!r}")
            if array.dtype.hasobject:
                raise TypeError(f"channel {name!r} must not use an object dtype")
            descriptor = CANONICAL_CHANNELS[name]
            expected_ndim = 2 if descriptor.components == 1 else 3
            if array.ndim != expected_ndim or (
                descriptor.components != 1 and array.shape[-1] != descriptor.components
            ):
                suffix = (
                    "" if descriptor.components == 1 else f"x{descriptor.components}"
                )
                raise ValueError(f"channel {name!r} must have shape HxW{suffix}")
            current_shape = (array.shape[0], array.shape[1])
            if min(current_shape) < 3:
                raise ValueError("material-map dimensions must each be at least 3")
            if spatial_shape is None:
                spatial_shape = current_shape
            elif current_shape != spatial_shape:
                raise ValueError("all material channels must share one HxW shape")
            if array.dtype != descriptor.dtype:
                raise TypeError(
                    f"channel {name!r} must use {descriptor.dtype}, got {array.dtype}"
                )
            for ys, xs in _spatial_slices(current_shape):
                block = array[ys, xs]
                if array.dtype.kind == "f":
                    valid = np.isfinite(block)
                    if name == "boundary_distance_mm":
                        # A one-grain torus has no inter-ID boundary.
                        valid |= np.isposinf(block)
                    if not np.all(valid):
                        raise ValueError(f"channel {name!r} contains non-finite values")
                if descriptor.value_range is not None:
                    lower, upper = descriptor.value_range
                    if np.any(block < lower) or np.any(block > upper):
                        raise ValueError(
                            f"channel {name!r} lies outside [{lower}, {upper}]"
                        )

        self._validate_vectors()
        self._validate_coverages()
        if "optical_parameters" in self.metadata:
            validate_optical_parameters(self.metadata["optical_parameters"])
        representation = self.metadata.get("representation")
        if representation == "rich" and self.lobes is None:
            raise ValueError("representation='rich' requires lobe records")
        if representation not in (None, "rich", "single_lobe"):
            raise ValueError("metadata representation must be 'rich' or 'single_lobe'")
        if self.lobes is not None:
            assert spatial_shape is not None
            self._validate_lobes(spatial_shape)

    def _validate_vectors(self) -> None:
        if "normal_ts" in self.arrays:
            normals = self.arrays["normal_ts"]
            for ys, xs in _spatial_slices(normals.shape[:2]):
                block = normals[ys, xs]
                lengths = np.linalg.norm(block, axis=-1)
                if not np.allclose(lengths, 1.0, atol=1e-5, rtol=0.0):
                    raise ValueError("normal_ts vectors must be unit length")
                if np.any(block[..., 2] <= 0.0):
                    raise ValueError(
                        "normal_ts vectors must lie in the positive hemisphere"
                    )
        if "anisotropy_axis" in self.arrays:
            axes = self.arrays["anisotropy_axis"]
            for ys, xs in _spatial_slices(axes.shape[:2]):
                block = axes[ys, xs]
                lengths = np.linalg.norm(block, axis=-1)
                valid = np.isclose(lengths, 0.0, atol=1e-6) | np.isclose(
                    lengths, 1.0, atol=1e-5
                )
                if not np.all(valid):
                    raise ValueError("anisotropy_axis must contain unit axes or zero")
                if "anisotropy" in self.arrays:
                    active = self.arrays["anisotropy"][ys, xs] > 1e-6
                    if np.any(active != (lengths > 0.5)):
                        raise ValueError(
                            "anisotropy_axis must be unit exactly where anisotropy is non-zero"
                        )

    def _validate_coverages(self) -> None:
        names = ("metallic", "patina_coverage", "white_stain_coverage")
        if all(name in self.arrays for name in names):
            for ys, xs in _spatial_slices(self.arrays[names[0]].shape):
                total = sum(
                    (self.arrays[name][ys, xs] for name in names),
                    start=np.float32(0.0),
                )
                if not np.allclose(total, 1.0, atol=1e-6, rtol=0.0):
                    raise ValueError("visible material coverages must sum to one")

    def _validate_lobes(self, spatial_shape: tuple[int, int]) -> None:
        assert self.lobes is not None
        records = self.lobes
        if records.dtype != LOBE_DTYPE or records.ndim != 1:
            raise TypeError("lobes must be a one-dimensional array with LOBE_DTYPE")
        if records.size == 0:
            raise ValueError("rich lobe records must not be empty")
        pixel_count = spatial_shape[0] * spatial_shape[1]
        if pixel_count > 1 << 32:
            raise ValueError("lobe pixel_index cannot address more than 2^32 pixels")
        parameters = self.metadata.get("optical_parameters")
        identities = (
            None
            if parameters is None
            else np.asarray(
                [entry["material_id"] for entry in parameters], dtype=np.uint8
            )
        )
        self._validate_lobe_buckets(records, spatial_shape, pixel_count, identities)

    def _validate_lobe_buckets(
        self,
        records: np.ndarray,
        spatial_shape: tuple[int, int],
        pixel_count: int,
        identities: np.ndarray | None,
    ) -> None:
        """Validate arbitrary record order with bounded memory and disk scratch."""
        from typing import BinaryIO

        bucket_count = (pixel_count + _PIXEL_BUCKET - 1) // _PIXEL_BUCKET
        with tempfile.TemporaryDirectory(prefix="texture-material-lobes-") as directory:
            root = Path(directory)
            open_files: OrderedDict[int, BinaryIO] = OrderedDict()
            try:
                for start in range(0, len(records), _RECORD_BLOCK):
                    block = records[start : start + _RECORD_BLOCK]
                    self._validate_lobe_fields(block, pixel_count, identities)
                    bucket_ids = block["pixel_index"] // _PIXEL_BUCKET
                    order = np.argsort(bucket_ids, kind="stable")
                    sorted_buckets = bucket_ids[order]
                    compact = np.empty(len(block), dtype=_COMPACT_LOBE_DTYPE)
                    for name in _COMPACT_LOBE_DTYPE.names or ():
                        compact[name] = block[name][order]
                    boundaries = np.concatenate(
                        ([0], np.flatnonzero(np.diff(sorted_buckets)) + 1, [len(block)])
                    )
                    for first, last in pairwise(boundaries):
                        bucket = int(sorted_buckets[first])
                        stream = open_files.pop(bucket, None)
                        if stream is None:
                            if len(open_files) >= _MAX_OPEN_BUCKET_FILES:
                                _, oldest = open_files.popitem(last=False)
                                oldest.close()
                            stream = (root / f"{bucket:08d}.bin").open("ab")
                        open_files[bucket] = stream
                        compact[first:last].tofile(stream)
            finally:
                for stream in open_files.values():
                    stream.close()

            width = spatial_shape[1]
            for bucket in range(bucket_count):
                first_pixel = bucket * _PIXEL_BUCKET
                stop_pixel = min(pixel_count, first_pixel + _PIXEL_BUCKET)
                length = stop_pixel - first_pixel
                total = np.zeros(length, dtype=np.float64)
                material_total = np.zeros(
                    (len(_COVERAGE_FOR_ID), length), dtype=np.float64
                )
                counts = np.zeros((len(_COVERAGE_FOR_ID), length), dtype=np.int64)
                bucket_path = root / f"{bucket:08d}.bin"
                if bucket_path.exists():
                    with bucket_path.open("rb") as stream:
                        while True:
                            compact = np.fromfile(
                                stream,
                                dtype=_COMPACT_LOBE_DTYPE,
                                count=_BUCKET_READ_BLOCK,
                            )
                            if not len(compact):
                                break
                            local = (
                                compact["pixel_index"].astype(np.int64) - first_pixel
                            )
                            material = compact["material_id"].astype(np.int64)
                            weights = compact["weight"]
                            np.add.at(total, local, weights)
                            np.add.at(material_total, (material, local), weights)
                            np.add.at(counts, (material, local), 1)
                if not np.allclose(total, 1.0, atol=1e-6, rtol=0.0):
                    raise ValueError("lobe weights in every pixel must sum to one")
                if np.any(counts > 4):
                    raise ValueError(
                        "at most four lobes per visible material and pixel are allowed"
                    )
                for identifier, coverage_name in _COVERAGE_FOR_ID.items():
                    if coverage_name not in self.arrays:
                        continue
                    expected = _raster_window(
                        self.arrays[coverage_name], first_pixel, stop_pixel, width
                    )
                    if not np.allclose(
                        material_total[identifier], expected, atol=1e-6, rtol=0.0
                    ):
                        raise ValueError(
                            f"lobe weights for material {identifier} do not match {coverage_name}"
                        )

    @staticmethod
    def _validate_lobe_fields(
        records: np.ndarray, pixel_count: int, identities: np.ndarray | None
    ) -> None:
        pixel_index = records["pixel_index"]
        material_id = records["material_id"]
        weights = records["weight"]
        if np.any(pixel_index >= pixel_count):
            raise ValueError("lobe pixel_index lies outside the sampled maps")
        if np.any(material_id >= len(_COVERAGE_FOR_ID)):
            raise ValueError("lobe material_id is not declared")
        if identities is not None:
            indices = records["optical_parameter_index"]
            if np.any(indices >= len(identities)):
                raise ValueError("lobe optical_parameter_index lies outside the table")
            if np.any(identities[indices] != material_id):
                raise ValueError("lobe material_id differs from its optical parameter")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("lobe weights must be finite and positive")
        normals = records["normal_ts"]
        tangents = records["tangent_ts"]
        if not np.all(np.isfinite(normals)) or not np.all(np.isfinite(tangents)):
            raise ValueError("lobe frames must be finite")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError("lobe normals must be unit length")
        if np.any(normals[:, 2] <= 0.0):
            raise ValueError("lobe normals must lie in the positive hemisphere")
        if not np.allclose(np.linalg.norm(tangents, axis=1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError("lobe tangents must be unit length")
        if not np.allclose(
            np.sum(normals * tangents, axis=1), 0.0, atol=1e-5, rtol=0.0
        ):
            raise ValueError("lobe tangents must be orthogonal to their normals")
        for width_name in ("alpha_t", "alpha_b"):
            widths = records[width_name]
            if (
                not np.all(np.isfinite(widths))
                or np.any(widths <= 0.0)
                or np.any(widths > np.sqrt(2.0) + 1e-7)
            ):
                raise ValueError(f"lobe {width_name} must lie in (0, sqrt(2)]")
