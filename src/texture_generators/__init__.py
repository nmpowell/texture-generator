"""Procedural texture generator: random but realistic metal/plastic/wood/paper PNGs.

Public API::

    from texture_generators import generate, generate_array, MATERIALS, sample_sheet

    img = generate("wood", size=512, seed=42)
    img = generate("metal", size=(640, 480), variant="brushed")
    arr = generate_array("wood", size=512, seed=42)  # float32 (H, W, 3) in [0, 1]
    sheet = sample_sheet(size=256, seed=7)

Every texture is produced from a single seeded ``np.random.Generator``, so the
same seed always yields byte-identical output.
"""

from __future__ import annotations

from importlib.metadata import version as _distribution_version

import numpy as np
from PIL import Image, ImageDraw

from .materials import MATERIALS, Material

__version__ = _distribution_version("texture-generator")

__all__ = [
    "MATERIALS",
    "VARIANTS_BY_MATERIAL",
    "Material",
    "__version__",
    "all_pairs",
    "generate",
    "generate_array",
    "resolve_variant",
    "sample_sheet",
    "to_image",
    "variants",
]

VARIANTS_BY_MATERIAL = {name: list(mod.VARIANTS) for name, mod in MATERIALS.items()}


def variants(material: str) -> list[str]:
    """Variant names for ``material``."""
    return list(_module(material).VARIANTS)


def all_pairs() -> list[tuple[str, str]]:
    """Every ``(material, variant)`` combination, in registry order."""
    return [(m, v) for m, mod in MATERIALS.items() for v in mod.VARIANTS]


def _module(material: str):
    """Look up a material module, raising a helpful ValueError."""
    try:
        return MATERIALS[material]
    except KeyError:
        valid = ", ".join(sorted(MATERIALS))
        raise ValueError(
            f"unknown material {material!r}; choose from: {valid}"
        ) from None


def _parse_size(size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalise ``size`` to ``(height, width)`` in pixels."""
    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError("size tuple must be (width, height)")
        w, h = int(size[0]), int(size[1])
    else:
        w = h = int(size)
    if w < 2 or h < 2:
        raise ValueError(f"size must be at least 2x2, got {w}x{h}")
    return h, w


def _pick_variant(mod: Material, rng: np.random.Generator) -> str:
    """Draw one of ``mod``'s variants from ``rng``.

    The single place the default variant is chosen, so a caller can learn which
    variant a seed selects without duplicating (or perturbing) the draw.
    """
    return str(mod.VARIANTS[int(rng.integers(0, len(mod.VARIANTS)))])


def resolve_variant(
    material: str,
    seed: int | None = None,
    variant: str | None = None,
) -> str:
    """Name the variant a render will use, without rendering it.

    ``variant`` is returned as given, once validated. With ``variant=None`` and
    a concrete ``seed``, the return value is the variant
    ``generate(material, seed=seed)`` will render, so a caller can name it (in a
    report or a filename) before rendering it. With ``seed=None`` this simply
    draws one such variant from fresh entropy: each call is independent, and it
    predicts nothing about a separate unseeded render.

    Raises:
        ValueError: on an unknown material or variant.
    """
    mod = _module(material)
    if variant is None:
        return _pick_variant(mod, np.random.default_rng(seed))
    if variant not in mod.VARIANTS:
        valid = ", ".join(mod.VARIANTS)
        raise ValueError(
            f"unknown {material} variant {variant!r}; choose from: {valid}"
        )
    return variant


def to_image(rgb: np.ndarray) -> Image.Image:
    """Convert a float32 (H, W, 3) array in [0, 1] to a PIL RGB image."""
    arr = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def generate_array(
    material: str,
    size: int | tuple[int, int] = 512,
    seed: int | None = None,
    variant: str | None = None,
    **params,
) -> np.ndarray:
    """Generate a texture as a float32 ``(H, W, 3)`` array in ``[0, 1]``.

    Same contract as :func:`generate` but without the PIL conversion, for
    callers that want the raw pixels.

    Args:
        material: one of :data:`MATERIALS` (``metal``, ``plastic``, ``wood``,
            ``paper``).
        size: ``N`` for a square texture or ``(width, height)``.
        seed: integer seed; ``None`` draws fresh entropy.
        variant: variant name, or ``None`` to pick one with the seeded rng.
        **params: forwarded to the material module (e.g. ``specular``).

    Raises:
        ValueError: on an unknown material or variant.
    """
    mod = _module(material)
    h, w = _parse_size(size)
    rng = np.random.default_rng(seed)

    if variant is None:
        variant = _pick_variant(mod, rng)
    elif variant not in mod.VARIANTS:
        valid = ", ".join(mod.VARIANTS)
        raise ValueError(
            f"unknown {material} variant {variant!r}; choose from: {valid}"
        )

    rgb = mod.generate((h, w), rng, variant, **params)
    if rgb.shape[:2] != (h, w):
        raise RuntimeError(
            f"{material}/{variant} produced {rgb.shape[1::-1]}, expected {(w, h)}"
        )
    return rgb


def generate(
    material: str,
    size: int | tuple[int, int] = 512,
    seed: int | None = None,
    variant: str | None = None,
    **params,
) -> Image.Image:
    """Generate a texture and return it as a PIL RGB image.

    Args:
        material: one of :data:`MATERIALS` (``metal``, ``plastic``, ``wood``,
            ``paper``).
        size: ``N`` for a square texture or ``(width, height)``.
        seed: integer seed; ``None`` draws fresh entropy.
        variant: variant name, or ``None`` to pick one with the seeded rng.
        **params: forwarded to the material module (e.g. ``specular``).

    Raises:
        ValueError: on an unknown material or variant.
    """
    return to_image(generate_array(material, size, seed, variant, **params))


def sample_sheet(
    size: int | tuple[int, int] = 256,
    seed: int | None = None,
    *,
    columns: int | None = None,
    **params,
) -> Image.Image:
    """Contact sheet with one labelled tile per material x variant.

    ``size`` is the per-tile size: ``N`` for square tiles or
    ``(width, height)``.
    """
    pairs = all_pairs()
    rng = np.random.default_rng(seed)
    cols = columns or min(4, len(pairs))
    rows = (len(pairs) + cols - 1) // cols

    tile_h, tile_w = _parse_size(size)
    pad = 6
    label_h = max(14, tile_h // 16)
    cell_w = tile_w + pad
    cell_h = tile_h + label_h + pad
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)

    for i, (material, variant) in enumerate(pairs):
        tile_seed = int(rng.integers(0, 2**31 - 1))
        tile = generate(material, size, seed=tile_seed, variant=variant, **params)
        cx = pad + (i % cols) * cell_w
        cy = pad + (i // cols) * cell_h
        sheet.paste(tile, (cx, cy))
        draw.text(
            (cx + 2, cy + tile_h + 2),
            f"{material}/{variant} #{tile_seed % 10000}",
            fill=(225, 225, 225),
        )
    return sheet
