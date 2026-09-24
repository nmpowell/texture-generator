"""Strict immutable configuration contracts for galvanised metal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np

from texture_generators.core.physical import validate_size_mm

__all__ = [
    "PRESET_REVISION",
    "PRESET_VERSION",
    "SCHEMA_VERSION",
    "GalvanisedConfig",
    "Placement",
    "Preset",
    "PreviewConfig",
    "Quality",
    "Representation",
]

SCHEMA_VERSION = 1
PRESET_REVISION = "2026-09-23.release-1"
PRESET_VERSION = PRESET_REVISION

Preset = Literal[
    "regular",
    "minimised",
    "weathered",
    "wet_storage",
    "inconspicuous",
    "batch",
]
Quality = Literal["draft", "production", "reference"]
Representation = Literal["rich", "single_lobe"]
Placement = Literal["poisson", "uniform"]
Rig = Literal["studio", "oblique", "overcast", "grazing"]

_UNSET = object()
_INITIAL_PRESETS = {"regular", "minimised", "weathered", "wet_storage"}
_EXPERIMENTAL_PRESETS = {"inconspicuous", "batch"}

_SCHEMA_DEFAULTS: dict[str, Any] = {
    "size_mm": None,
    "spangle_diameter_mm": 8.0,
    "spangle_cv": 0.28,
    "placement": "poisson",
    "texture_strength": 0.45,
    "crystal_tilt_concentration": 4.0,
    "dendrite_relief_um": 1.0,
    "trunk_relief_um": 2.0,
    "boundary_depth_um": 1.0,
    "micro_relief_um": 0.4,
    "roughness": 0.30,
    "anisotropy": 0.70,
    "exposure": 0.0,
    "wetness": 0.0,
    "confinement": 0.0,
    "salt_exposure": 0.0,
    "white_stain": 0.0,
    "weather_seed": None,
    "dross_per_cm2": 0.0,
    "runs_per_cm2": 0.0,
    "gravity_angle_deg": 90.0,
    "quality": "production",
    "representation": "rich",
    "memory_budget_mb": 512,
}

_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "regular": {},
    "minimised": {
        "spangle_diameter_mm": 1.5,
        "spangle_cv": 0.20,
        "dendrite_relief_um": 0.6,
        "trunk_relief_um": 1.2,
        "micro_relief_um": 0.25,
    },
    "weathered": {
        "exposure": 3.0,
        "wetness": 0.35,
        "white_stain": 0.05,
    },
    "wet_storage": {
        "exposure": 1.8,
        "wetness": 0.95,
        "confinement": 0.90,
        "salt_exposure": 0.25,
        "white_stain": 0.90,
    },
    # Accepted for forward integration, but deliberately labelled experimental.
    "inconspicuous": {
        "spangle_diameter_mm": 0.4,
        "spangle_cv": 0.15,
        "dendrite_relief_um": 0.5,
        "trunk_relief_um": 1.0,
        "micro_relief_um": 0.15,
    },
    "batch": {
        "placement": "uniform",
        "dross_per_cm2": 0.25,
        "runs_per_cm2": 0.12,
        "dendrite_relief_um": 3.0,
        "trunk_relief_um": 6.0,
    },
}

_VALUE_FIELDS = tuple(_SCHEMA_DEFAULTS)
_RESOLVED_FIELDS = ("preset", *_VALUE_FIELDS)


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded(value: object, name: str, lower: float, upper: float) -> float:
    result = _real(value, name)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return result


def _non_negative(value: object, name: str) -> float:
    result = _real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: object, name: str) -> float:
    result = _real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _uint64_or_none(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if not 0 <= result < 1 << 64:
        raise ValueError(f"{name} must be in the unsigned 64-bit range")
    return result


def _choice(value: object, name: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")
    return value


def _validate_values(
    values: Mapping[str, Any], *, replay: bool = False
) -> dict[str, Any]:
    result = dict(values)
    preset = result["preset"]
    if replay:
        if not isinstance(preset, str) or not preset:
            raise ValueError("preset must be a non-empty string")
    else:
        result["preset"] = _choice(preset, "preset", set(_PRESET_DEFAULTS))

    size_mm = result["size_mm"]
    result["size_mm"] = None if size_mm is None else validate_size_mm(size_mm)
    result["spangle_diameter_mm"] = _positive(
        result["spangle_diameter_mm"], "spangle_diameter_mm"
    )
    result["spangle_cv"] = _bounded(result["spangle_cv"], "spangle_cv", 0.0, 1.0)
    result["placement"] = _choice(
        result["placement"], "placement", {"poisson", "uniform"}
    )
    result["texture_strength"] = _bounded(
        result["texture_strength"], "texture_strength", 0.0, 1.0
    )
    result["crystal_tilt_concentration"] = _non_negative(
        result["crystal_tilt_concentration"], "crystal_tilt_concentration"
    )
    for name in (
        "dendrite_relief_um",
        "trunk_relief_um",
        "boundary_depth_um",
        "micro_relief_um",
        "exposure",
        "dross_per_cm2",
        "runs_per_cm2",
    ):
        result[name] = _non_negative(result[name], name)
    for name in (
        "roughness",
        "anisotropy",
        "wetness",
        "confinement",
        "salt_exposure",
        "white_stain",
    ):
        result[name] = _bounded(result[name], name, 0.0, 1.0)
    result["weather_seed"] = _uint64_or_none(result["weather_seed"], "weather_seed")
    result["gravity_angle_deg"] = _real(
        result["gravity_angle_deg"], "gravity_angle_deg"
    )
    result["quality"] = _choice(
        result["quality"], "quality", {"draft", "production", "reference"}
    )
    result["representation"] = _choice(
        result["representation"], "representation", {"rich", "single_lobe"}
    )
    budget = result["memory_budget_mb"]
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise TypeError("memory_budget_mb must be an integer")
    result["memory_budget_mb"] = int(budget)
    if result["memory_budget_mb"] <= 0:
        raise ValueError("memory_budget_mb must be positive")
    return result


def _warnings(values: Mapping[str, Any], explicit: set[str]) -> tuple[str, ...]:
    messages = [
        "Preset values are physically grounded authoring defaults, not metrological calibration."
    ]
    preset = values["preset"]
    if preset in _EXPERIMENTAL_PRESETS:
        messages.append(
            f"Preset {preset!r} is experimental and is not an initial calibrated preset."
        )
    diameter = cast(float, values["spangle_diameter_mm"])
    calibrated_range = {
        "regular": (5.0, 15.0),
        "weathered": (5.0, 15.0),
        "wet_storage": (5.0, 15.0),
        "minimised": (0.8, 3.0),
        "inconspicuous": (0.1, 0.8),
    }.get(cast(str, preset))
    if (
        calibrated_range is not None
        and not calibrated_range[0] <= diameter <= calibrated_range[1]
    ):
        messages.append(
            "spangle_diameter_mm is outside the preset's initial visual fitting range."
        )
    if values["representation"] == "single_lobe":
        messages.append(
            "single_lobe is an explicitly approximate representation of mixed material response."
        )
    if preset != "batch" and (
        cast(float, values["dross_per_cm2"]) > 0.0
        or cast(float, values["runs_per_cm2"]) > 0.0
    ):
        messages.append(
            "Batch-style dross or drainage was explicitly enabled outside the experimental batch preset."
        )
    if "spangle_cv" in explicit:
        messages.append(
            "spangle_cv is the target realised equivalent-diameter CV, not a power-weight CV."
        )
    return tuple(messages)


@dataclass(frozen=True, init=False)
class GalvanisedConfig:
    """Resolved galvanised material inputs, independent of preview lighting.

    Constructor resolution order is schema defaults, preset defaults, explicit
    overrides, then validation.  Use :meth:`from_resolved_mapping` when
    replaying a stored recipe so later preset changes cannot alter it.
    """

    preset: str
    size_mm: tuple[float, float] | None
    spangle_diameter_mm: float
    spangle_cv: float
    placement: Placement
    texture_strength: float
    crystal_tilt_concentration: float
    dendrite_relief_um: float
    trunk_relief_um: float
    boundary_depth_um: float
    micro_relief_um: float
    roughness: float
    anisotropy: float
    exposure: float
    wetness: float
    confinement: float
    salt_exposure: float
    white_stain: float
    weather_seed: int | None
    dross_per_cm2: float
    runs_per_cm2: float
    gravity_angle_deg: float
    quality: Quality
    representation: Representation
    memory_budget_mb: int
    schema_version: int
    preset_revision: str
    warnings: tuple[str, ...]

    def __init__(
        self,
        *,
        preset: Preset = "regular",
        size_mm: tuple[float, float] | object | None = _UNSET,
        spangle_diameter_mm: float | object = _UNSET,
        spangle_cv: float | object = _UNSET,
        placement: Placement | object = _UNSET,
        texture_strength: float | object = _UNSET,
        crystal_tilt_concentration: float | object = _UNSET,
        dendrite_relief_um: float | object = _UNSET,
        trunk_relief_um: float | object = _UNSET,
        boundary_depth_um: float | object = _UNSET,
        micro_relief_um: float | object = _UNSET,
        roughness: float | object = _UNSET,
        anisotropy: float | object = _UNSET,
        exposure: float | object = _UNSET,
        wetness: float | object = _UNSET,
        confinement: float | object = _UNSET,
        salt_exposure: float | object = _UNSET,
        white_stain: float | object = _UNSET,
        weather_seed: int | object | None = _UNSET,
        dross_per_cm2: float | object = _UNSET,
        runs_per_cm2: float | object = _UNSET,
        gravity_angle_deg: float | object = _UNSET,
        quality: Quality | object = _UNSET,
        representation: Representation | object = _UNSET,
        memory_budget_mb: int | object = _UNSET,
    ) -> None:
        _choice(preset, "preset", set(_PRESET_DEFAULTS))
        supplied = locals()
        explicit = {name for name in _VALUE_FIELDS if supplied[name] is not _UNSET}
        values = {"preset": preset, **_SCHEMA_DEFAULTS, **_PRESET_DEFAULTS[preset]}
        values.update({name: supplied[name] for name in explicit})
        validated = _validate_values(values)
        self._assign(
            validated,
            schema_version=SCHEMA_VERSION,
            preset_revision=PRESET_REVISION,
            warnings=_warnings(validated, explicit),
        )

    def _assign(
        self,
        values: Mapping[str, Any],
        *,
        schema_version: int,
        preset_revision: str,
        warnings: tuple[str, ...],
    ) -> None:
        for name in _RESOLVED_FIELDS:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "preset_revision", preset_revision)
        object.__setattr__(self, "warnings", warnings)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> GalvanisedConfig:
        """Construct from strict user overrides; unknown keys are errors."""
        unknown = set(mapping) - set(_RESOLVED_FIELDS)
        if unknown:
            raise ValueError(f"unknown GalvanisedConfig keys: {sorted(unknown)!r}")
        return cls(**dict(mapping))

    @classmethod
    def from_resolved_mapping(cls, mapping: Mapping[str, Any]) -> GalvanisedConfig:
        """Replay a complete resolved mapping without consulting preset defaults."""
        administrative = {"schema_version", "preset_revision", "warnings"}
        unknown = set(mapping) - set(_RESOLVED_FIELDS) - administrative
        missing = set(_RESOLVED_FIELDS) - set(mapping)
        if unknown or missing:
            raise ValueError(
                f"resolved config keys mismatch; missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
            )
        schema_version = mapping.get("schema_version", SCHEMA_VERSION)
        if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {schema_version!r}")
        preset_revision = mapping.get("preset_revision", PRESET_REVISION)
        if not isinstance(preset_revision, str) or not preset_revision:
            raise ValueError("preset_revision must be a non-empty string")
        warning_values = mapping.get("warnings", ())
        if (
            not isinstance(warning_values, Sequence)
            or isinstance(warning_values, (str, bytes))
            or not all(isinstance(item, str) for item in warning_values)
        ):
            raise TypeError("warnings must be a sequence of strings")
        values = _validate_values(
            {name: mapping[name] for name in _RESOLVED_FIELDS}, replay=True
        )
        instance = object.__new__(cls)
        instance._assign(
            values,
            schema_version=SCHEMA_VERSION,
            preset_revision=preset_revision,
            warnings=tuple(warning_values),
        )
        return instance

    def to_mapping(self) -> dict[str, Any]:
        """Return the complete JSON-safe resolved recipe for stable replay."""
        result = {name: getattr(self, name) for name in _RESOLVED_FIELDS}
        result.update(
            schema_version=self.schema_version,
            preset_revision=self.preset_revision,
            warnings=list(self.warnings),
        )
        if self.size_mm is not None:
            result["size_mm"] = list(self.size_mm)
        return result

    def resolve(self, *, size: tuple[int, int]) -> GalvanisedConfig:
        """Resolve physical extent from image ``(width, height)`` if necessary."""
        if self.size_mm is not None:
            return self
        width, height = size
        for name, value in (("size[0]", width), ("size[1]", height)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        resolved_size = (100.0, 100.0 * int(height) / int(width))
        mapping = self.to_mapping()
        mapping["size_mm"] = resolved_size
        return self.from_resolved_mapping(mapping)

    @property
    def preset_version(self) -> str:
        """Compatibility spelling for the immutable preset revision."""
        return self.preset_revision

    @property
    def is_experimental(self) -> bool:
        """Whether the selected preset is accepted only as experimental."""
        return self.preset in _EXPERIMENTAL_PRESETS


@dataclass(frozen=True)
class PreviewConfig:
    """Strict preview-only light, view, exposure, and sampling controls."""

    rig: Rig = "studio"
    exposure_stops: float = 0.0
    normal_strength: float = 1.0
    light_azimuth_deg: float = 315.0
    # Fixed 24-degree polar tilt, azimuth 135 degrees. This observes the
    # default studio source slightly off its mirror direction, where lobe
    # widths are visible. Normal incidence remains an explicit (0, 0, 1).
    view: tuple[float, float, float] = (
        -0.28760623847595074,
        0.28760623847595074,
        0.9135454576426009,
    )
    render_samples: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rig",
            _choice(self.rig, "rig", {"studio", "oblique", "overcast", "grazing"}),
        )
        object.__setattr__(
            self, "exposure_stops", _real(self.exposure_stops, "exposure_stops")
        )
        object.__setattr__(
            self,
            "normal_strength",
            _non_negative(self.normal_strength, "normal_strength"),
        )
        object.__setattr__(
            self,
            "light_azimuth_deg",
            _real(self.light_azimuth_deg, "light_azimuth_deg"),
        )
        if len(self.view) != 3:
            raise ValueError("view must contain exactly three components")
        view = tuple(
            _real(value, f"view[{index}]") for index, value in enumerate(self.view)
        )
        if view[2] <= 0.0 or not np.isclose(
            np.linalg.norm(view), 1.0, atol=1e-6, rtol=0.0
        ):
            raise ValueError("view must be a unit vector in the positive hemisphere")
        object.__setattr__(self, "view", view)
        if isinstance(self.render_samples, bool) or not isinstance(
            self.render_samples, Integral
        ):
            raise TypeError("render_samples must be an integer")
        if self.render_samples <= 0:
            raise ValueError("render_samples must be positive")
        object.__setattr__(self, "render_samples", int(self.render_samples))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> PreviewConfig:
        """Construct from a mapping while rejecting every unknown input."""
        allowed = {
            "rig",
            "exposure_stops",
            "normal_strength",
            "light_azimuth_deg",
            "view",
            "render_samples",
        }
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError(f"unknown PreviewConfig keys: {sorted(unknown)!r}")
        return cls(**dict(mapping))
