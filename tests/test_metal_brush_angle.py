"""Brush direction controls for the public metal texture API."""

from __future__ import annotations

import numpy as np
import pytest

from texture_generators import generate, generate_array, to_image
from texture_generators.materials import metal


def _directional_gradient_energy(rgb: np.ndarray) -> tuple[float, float, float, float]:
    """Return x, y, down-right, and down-left luminance gradient energy."""
    luminance = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    across_x = float(np.mean(np.diff(luminance, n=2, axis=1) ** 2))
    across_y = float(np.mean(np.diff(luminance, n=2, axis=0) ** 2))
    down_right = float(np.mean((luminance[1:, 1:] - luminance[:-1, :-1]) ** 2))
    down_left = float(np.mean((luminance[1:, :-1] - luminance[:-1, 1:]) ** 2))
    return across_x, across_y, down_right, down_left


@pytest.mark.parametrize(
    ("angle", "size", "along_index", "across_index"),
    [
        (0.0, 192, 0, 1),
        (90.0, 192, 1, 0),
        (45.0, (240, 128), 2, 3),
        (45.0, (128, 240), 2, 3),
        (135.0, (240, 128), 3, 2),
        (135.0, (128, 240), 3, 2),
    ],
    ids=[
        "horizontal",
        "vertical",
        "down-right-landscape",
        "down-right-portrait",
        "down-left-landscape",
        "down-left-portrait",
    ],
)
def test_brush_angle_orients_the_fine_signal(
    angle: float,
    size: int | tuple[int, int],
    along_index: int,
    across_index: int,
) -> None:
    """Fine grooves vary most strongly across their requested direction."""
    brushed = generate_array(
        "metal", size=size, seed=13, variant="brushed", brush_angle=angle
    )

    energy = _directional_gradient_energy(brushed)

    assert energy[across_index] > 1.5 * energy[along_index]


@pytest.mark.parametrize("wrapped_angle", [-315.0, 360_000_000_000_045.0])
def test_brush_angle_wraps_around_a_full_turn(wrapped_angle: float) -> None:
    """Angles separated by one turn produce byte-identical arrays."""
    canonical = generate_array(
        "metal", size=(96, 64), seed=7, variant="brushed", brush_angle=45.0
    )

    wrapped = generate_array(
        "metal",
        size=(96, 64),
        seed=7,
        variant="brushed",
        brush_angle=wrapped_angle,
    )

    assert np.array_equal(wrapped, canonical)


@pytest.mark.parametrize(
    "invalid_angle",
    [float("nan"), float("inf"), float("-inf"), "45", object()],
    ids=["nan", "positive-infinity", "negative-infinity", "string", "object"],
)
def test_brush_angle_requires_a_finite_number(invalid_angle: object) -> None:
    """Undefined and non-numeric directions are rejected at the metal boundary."""
    with pytest.raises(ValueError, match=r"brush_angle.*finite"):
        generate_array(
            "metal", size=32, seed=7, variant="brushed", brush_angle=invalid_angle
        )


@pytest.mark.parametrize(
    "material,variant",
    [("metal", None), ("metal", "radial"), ("wood", "board")],
    ids=["implicit-metal-variant", "radial-metal", "other-material"],
)
def test_public_brush_angle_requires_an_explicit_brushed_variant(
    material: str, variant: str | None
) -> None:
    """The convenience API never applies an angle to a seeded variant choice."""
    with pytest.raises(ValueError, match=r"brush_angle.*explicit.*brushed"):
        generate_array(material, size=32, seed=7, variant=variant, brush_angle=45.0)


@pytest.mark.parametrize("variant", ["radial", "polished", "heat_tinted"])
def test_direct_metal_angle_rejects_a_non_brushed_variant(variant: str) -> None:
    """The material module does not apply a linear direction to radial metal."""
    with pytest.raises(ValueError, match=r"brush_angle.*brushed"):
        metal.generate((32, 32), np.random.default_rng(7), variant, brush_angle=45.0)


@pytest.mark.parametrize(
    ("angle", "size", "along_index", "across_index", "film"),
    [
        (0.0, 192, 0, 1, True),
        (90.0, 192, 1, 0, "oxide"),
        (45.0, (240, 128), 2, 3, {"system": "oil"}),
        (45.0, (128, 240), 2, 3, True),
        (135.0, (240, 128), 3, 2, "oxide"),
        (135.0, (128, 240), 3, 2, {"system": "oil"}),
    ],
    ids=[
        "horizontal-preset",
        "vertical-named",
        "down-right-landscape-mapping",
        "down-right-portrait-preset",
        "down-left-landscape-named",
        "down-left-portrait-mapping",
    ],
)
def test_brush_angle_orients_a_filmed_brushed_surface(
    angle: float,
    size: int | tuple[int, int],
    along_index: int,
    across_index: int,
    film: bool | str | dict[str, str],
) -> None:
    """A transparent film retains the requested direction underneath it."""
    brushed = generate_array(
        "metal",
        size=size,
        seed=13,
        variant="brushed",
        brush_angle=angle,
        film=film,
    )

    energy = _directional_gradient_energy(brushed)

    assert energy[across_index] > 1.5 * energy[along_index]


@pytest.mark.parametrize("film", [None, True], ids=["bare", "filmed"])
def test_explicit_none_preserves_the_array_and_rng_state(film: bool | None) -> None:
    """Explicit ``None`` is byte-for-byte equivalent to omitting the option."""
    omitted_rng = np.random.default_rng(23)
    explicit_none_rng = np.random.default_rng(23)
    omitted = metal.generate((64, 96), omitted_rng, "brushed", film=film)
    explicit_none = metal.generate(
        (64, 96), explicit_none_rng, "brushed", film=film, brush_angle=None
    )

    omitted_tail = omitted_rng.integers(0, 2**31, size=8)
    explicit_none_tail = explicit_none_rng.integers(0, 2**31, size=8)

    assert np.array_equal(explicit_none, omitted)
    assert np.array_equal(explicit_none_tail, omitted_tail)


def test_generate_matches_array_for_a_fixed_brush_angle() -> None:
    """PIL and array entry points render the same controlled brush recipe."""
    recipe = {
        "material": "metal",
        "size": (96, 64),
        "seed": 13,
        "variant": "brushed",
        "brush_angle": 45.0,
        "film": True,
    }

    array = generate_array(**recipe)
    image = generate(**recipe)

    assert image.tobytes() == to_image(array).tobytes()


def test_same_full_brush_recipe_is_repeatable() -> None:
    """Repeating every brush and diffraction input produces the same array."""
    recipe = {
        "material": "metal",
        "size": (96, 64),
        "seed": 13,
        "variant": "brushed",
        "brush_angle": 90.0,
        "iridescence": 1.0,
        "source_angular_radius": 0.53,
        "groove_pitch_um": (1.0, 1.0),
    }

    first = generate_array(**recipe)
    second = generate_array(**recipe)

    assert np.array_equal(second, first)


@pytest.mark.parametrize(
    ("angle", "matches_unlit"),
    [(45.0, True), (0.0, False), (90.0, False)],
    ids=["out-of-band", "horizontal-visible", "vertical-visible"],
)
def test_unfilmed_diffraction_uses_the_brush_angle(
    angle: float, matches_unlit: bool
) -> None:
    """A fixed-pitch grating diffracts only at visible light projections."""
    recipe = {
        "material": "metal",
        "size": 256,
        "seed": 13,
        "variant": "brushed",
        "brush_angle": angle,
        "groove_pitch_um": (1.0, 1.0),
    }

    unlit = generate_array(**recipe, iridescence=0.0)
    diffracting = generate_array(**recipe, iridescence=1.0)

    assert np.array_equal(diffracting, unlit) is matches_unlit
