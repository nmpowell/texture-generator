"""Optional material-map capability alongside the existing RGB protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .core.material import MaterialMaps, validate_sampling_request
from .core.physical import validate_map_size
from .materials.galvanised_config import GalvanisedConfig

MAP_CAPABILITIES = frozenset({("metal", "galvanised")})


def resolve_galvanised_config(
    galvanised: GalvanisedConfig | Mapping[str, Any] | None = None,
    galvanised_preset: str | None = None,
) -> GalvanisedConfig:
    """Resolve the two mutually exclusive recipe entry points strictly."""
    if galvanised is not None and galvanised_preset is not None:
        raise ValueError("galvanised and galvanised_preset are mutually exclusive")
    if isinstance(galvanised, GalvanisedConfig):
        return galvanised
    if isinstance(galvanised, Mapping):
        return GalvanisedConfig.from_mapping(galvanised)
    if galvanised is not None:
        raise TypeError("galvanised must be a GalvanisedConfig or a mapping")
    return GalvanisedConfig.from_mapping(
        {} if galvanised_preset is None else {"preset": galvanised_preset}
    )


def generate_maps(
    material: str,
    size: int | tuple[int, int] = 512,
    seed: int | None = None,
    variant: str | None = None,
    *,
    galvanised: GalvanisedConfig | Mapping[str, Any] | None = None,
    galvanised_preset: str | None = None,
    maps: Sequence[str] | None = None,
    chunk_size: int = 128,
    output_dir: str | Path | None = None,
) -> MaterialMaps:
    """Sample physical material channels; currently supports metal/galvanised.

    Sizes are ``(width, height)`` and physical sizes live in the recipe. Lighting
    belongs to ``render_material``. ``output_dir`` enables mapped arrays for
    large selected outputs; it is working storage, not an export bundle.
    """
    if (material, variant) not in MAP_CAPABILITIES:
        raise ValueError(
            "material maps require material='metal' and explicit variant='galvanised'"
        )
    dimensions = (size, size) if isinstance(size, (int, np.integer)) else size
    width, height = validate_map_size(dimensions)
    selected = validate_sampling_request(maps, chunk_size)
    config = resolve_galvanised_config(galvanised, galvanised_preset)
    from .materials.galvanised import build_state, sample_state

    state = build_state(config, seed=seed, size=(width, height))
    return sample_state(
        state,
        size=(width, height),
        maps=selected,
        chunk_size=chunk_size,
        output_dir=output_dir,
    )
