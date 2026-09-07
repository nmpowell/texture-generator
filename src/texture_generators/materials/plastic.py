"""Plastic textures: glossy, matte and moulded-texture surfaces.

Algorithm: the defining feature of moulded plastic is that the albedo is
almost flat (only ~2% low-frequency mottling -- strong albedo noise makes
plastic read as stone) and all the character comes from the microstructure:
band-passed fbm "orange peel" for the height field, plus a Worley stipple for
the moulded-texture variant. The variants differ mostly in the specular
response, which is where gloss actually lives.
"""

from __future__ import annotations

import numpy as np

from ..core.fields import normalize01, smoothstep
from ..core.noise import fbm, fbm_at, grid_coords
from ..core.shading import shade
from ..core.worley import worley

VARIANTS = ["glossy", "matte", "textured"]

# Source: a sample palette of the 100 most common background colours measured
# from photos of real device housings, in rank order.
# fmt: off
PALETTE = [
    "#d7d9d6", "#d7d5c9", "#c8c9c7", "#e7e4d9", "#b8b9b7",
    "#c8c5b8", "#e8e8e5", "#b8b5a9", "#a9a9a7", "#c6baa8",
    "#a8a599", "#a79a88", "#fbfcfa", "#d6c9b7", "#b7aa98",
    "#e5dac8", "#999a97", "#999489", "#888478", "#867969",
    "#bbc4c8", "#978978", "#c4bdb5", "#797977", "#898987",
    "#cad4d7", "#77746a", "#b6aca6", "#d3cdc5", "#abb4b7",
    "#76695a", "#d7c5aa", "#dce2db", "#686967", "#a59d95",
    "#968c84", "#dbe4e6", "#696458", "#766c65", "#e6d6b9",
    "#665d56", "#857c76", "#ccd2cb", "#595958", "#e9e3ca",
    "#f6f6ea", "#e2ded6", "#665848", "#b9a58a", "#b6bcc4",
    "#c6b598", "#39342a", "#a8adb4", "#9ba3a6", "#97856a",
    "#9ca297", "#dcd2b9", "#474645", "#c9d8e6", "#a48b76",
    "#b8c8d5", "#99937a", "#c6ccd4", "#c8c2ad", "#969ca4",
    "#5a544a", "#7b8486", "#a6957c", "#8a9296", "#574839",
    "#46382b", "#373636", "#d4bca4", "#b49c85", "#bdc3ba",
    "#d5b898", "#e6c9a8", "#b5987a", "#a9b8c5", "#c3ad96",
    "#47433b", "#544c45", "#aba28c", "#272625", "#c7a689",
    "#f6d9b8", "#9aa7b4", "#877659", "#777d84", "#f3e8d8",
    "#bab29b", "#898e94", "#846c57", "#adb1aa", "#d5dde3",
    "#987858", "#676d75", "#755b4a", "#8a826b", "#947c66",
]
# fmt: on


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    """'#rrggbb' to a (r, g, b) tuple of floats in [0, 1]."""
    v = value.lstrip("#")
    return (int(v[0:2], 16) / 255.0, int(v[2:4], 16) / 255.0, int(v[4:6], 16) / 255.0)


_PALETTE_RGB = np.asarray([_hex_to_rgb(c) for c in PALETTE], dtype=np.float32)


def _base_colour(rng: np.random.Generator) -> np.ndarray:
    """Plastic base colour: 85% from the surveyed palette, 15% free HSV.

    Real device housings cluster tightly in off-white/grey/warm-neutral
    territory, so preferring the surveyed colours reads far more plausible
    than uniform HSV; the 15% HSV tail keeps saturated novelty colours
    possible. A small gain-plus-per-channel jitter turns the 100 fixed
    entries back into a continuum.
    """
    if rng.random() < 0.85:
        colour = _PALETTE_RGB[int(rng.integers(0, _PALETTE_RGB.shape[0]))]
        gain = np.float32(rng.uniform(0.94, 1.06))
        jitter = rng.normal(0.0, 0.008, size=3).astype(np.float32)
        return np.clip(colour * gain + jitter, 0.02, 1.0).astype(np.float32)
    if rng.random() < 0.10:
        value = float(rng.uniform(0.12, 0.85))
        return np.full(3, value, dtype=np.float32)
    hue = float(rng.uniform(0.0, 1.0))
    sat = float(rng.uniform(0.25, 0.9))
    val = float(rng.uniform(0.35, 0.9))
    return _hsv_to_rgb(hue, sat, val)


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    """Single-colour HSV to RGB (h, s, v in [0, 1])."""
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    return np.asarray(table[i], dtype=np.float32)


def _orange_peel(
    shape: tuple[int, int], rng: np.random.Generator, scale: float = 1.0
) -> np.ndarray:
    """Band-passed fbm: the shallow dimpling of a moulded plastic surface."""
    h, w = shape
    freq = float(w) / float(rng.uniform(14.0, 30.0)) * scale
    peel = fbm((h, w), rng, freq=freq, octaves=int(rng.integers(3, 5)), gain=0.5)
    # Band-pass: subtract a smoother version so only the mid scale survives.
    coarse = fbm((h, w), rng, freq=freq * 0.25, octaves=2)
    return (peel - 0.45 * coarse).astype(np.float32)


def _vignette(shape: tuple[int, int], strength: float = 0.08) -> np.ndarray:
    """Subtle darkening towards the corners."""
    x, y = grid_coords(shape)
    aspect = shape[0] / max(shape[1], 1)
    dx = (x - 0.5) * 2.0
    dy = (y - aspect * 0.5) * 2.0 / max(aspect, 1e-6)
    r2 = np.clip(dx * dx + dy * dy, 0.0, 2.0) * 0.5
    return (1.0 - np.float32(strength) * r2).astype(np.float32)


def generate(
    shape: tuple[int, int],
    rng: np.random.Generator,
    variant: str = "glossy",
    **params,
) -> np.ndarray:
    """Generate a plastic texture as float32 (H, W, 3) in [0, 1]."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown plastic variant {variant!r}; choose from {VARIANTS}")
    h, w = int(shape[0]), int(shape[1])
    base = _base_colour(rng)

    # Glossy needs a broader, shallower peel: fine dimples under a tight
    # specular lobe turn into sparkle, which reads as flake paint, not plastic.
    peel = _orange_peel((h, w), rng, scale=0.4 if variant == "glossy" else 1.0)
    # Albedo is nearly flat: only a couple of percent of low-frequency drift.
    mottle = fbm((h, w), rng, freq=float(rng.uniform(1.5, 4.0)), octaves=3)
    lum = 1.0 + np.float32(rng.uniform(0.012, 0.022)) * mottle
    albedo = base[None, None, :] * lum[..., None]

    height = peel * np.float32(0.5)
    aniso_dir = None
    aniso = 0.0

    specular2 = 0.0
    shininess2 = 300.0
    cavity = 0.0
    light_dir = (-0.33, -0.38, 0.86)
    if variant == "glossy":
        specular = float(rng.uniform(0.5, 0.8))
        shininess = float(rng.uniform(60.0, 120.0))
        normal_strength = float(rng.uniform(0.5, 1.1))
        ambient = 0.6
    elif variant == "matte":
        specular = float(rng.uniform(0.1, 0.2))
        shininess = float(rng.uniform(6.0, 10.0))
        normal_strength = float(rng.uniform(1.5, 3.0))
        ambient = 0.66
        # A matte moulding takes its finish from a bead-blasted tool:
        # shallow overlapping craters (a cellular distribution), not white
        # noise -- pure pixel grain reads as camera sensor noise.
        f1c, _ = worley((h, w), rng, density=float(w) / float(rng.uniform(2.8, 4.5)))
        craters = (1.0 - np.power(smoothstep(0.0, 0.55, f1c), 0.7)).astype(np.float32)
        grain = fbm((h, w), rng, freq=float(w) / 2.2, octaves=1)
        albedo = albedo * (1.0 + np.float32(0.010) * (craters - 0.5)[..., None])
        albedo = albedo * (1.0 + np.float32(0.006) * grain[..., None])
        height = height + craters * np.float32(0.22) + grain * np.float32(0.06)
    else:  # textured
        specular = float(rng.uniform(0.18, 0.35))
        shininess = float(rng.uniform(14.0, 30.0))
        normal_strength = float(rng.uniform(1.4, 2.6))
        ambient = 0.66
        # Moulded stipple: dense small domes that stay separated (F1 is in
        # cell units, so a 0.42 outer edge keeps neighbouring domes from
        # merging into worm clumps), plus a sparser half-density layer so the
        # bump sizes mix like real moulded texture.
        density = float(w) / float(rng.uniform(3.0, 6.5))
        f1, _ = worley(
            (h, w), rng, density=density, stretch=float(rng.uniform(0.95, 1.1))
        )
        bumps = np.power(smoothstep(0.42, 0.06, f1), 0.8)
        f1b, _ = worley((h, w), rng, density=density * 0.45)
        bumps_b = np.power(smoothstep(0.5, 0.1, f1b), 0.8)
        stipple = (bumps * 0.7 + bumps_b * 0.45).astype(np.float32)
        height = height * 0.35 + stipple
        albedo = albedo * (1.0 + np.float32(0.015) * (stipple - 0.5)[..., None])
        # Seat the domes: the hollows between them collect a little shadow.
        cavity = float(rng.uniform(0.15, 0.3))

    # Injection-moulding history, on ~a third of parts: faint curved flow
    # bands radiating from a gate, sometimes with one weld line where two
    # flow fronts met. These are GLOSS features -- they live in the
    # specular weight, with only a whisper of tint.
    spec_field = float(params.get("specular", specular))
    if rng.random() < 0.35:
        x, y = grid_coords((h, w))
        aspect = h / max(w, 1)
        fgx = float(rng.uniform(-0.15, 1.15))
        fgy = float(rng.uniform(-0.15, 1.15)) * aspect
        dg = np.sqrt((x - fgx) ** 2 + (y - fgy) ** 2)
        # Normalised angle on a theta-periodic lattice: raw atan2 through
        # aperiodic noise would seam at the branch cut when the gate sits
        # on-canvas.
        tg = (np.arctan2(y - fgy, x - fgx) / np.float32(2.0 * np.pi) + 0.5).astype(
            np.float32
        )
        flow = fbm_at(
            tg,
            dg,
            rng,
            freq=(2.0, float(w) / 26.0),
            octaves=2,
            periodic=(True, False),
        )
        # Stipple already dominates a textured part, so its flow history
        # shows at half strength.
        flow_amp = 0.15 if variant == "textured" else 0.30
        gloss = 1.0 - flow_amp * normalize01(flow)
        if rng.random() < 0.25:
            wa = float(rng.uniform(0.0, np.pi))
            wc = float(rng.uniform(0.25, 0.75))
            wd = np.abs(x * np.float32(np.cos(wa)) + y * np.float32(np.sin(wa)) - wc)
            gloss = gloss * (1.0 - 0.5 * np.exp(-((wd / np.float32(0.004)) ** 2)))
            albedo = albedo * (
                1.0 - 0.008 * np.exp(-((wd / np.float32(0.004)) ** 2))[..., None]
            )
        spec_field = (spec_field * gloss).astype(np.float32)

    rgb = shade(
        np.clip(albedo, 0.0, 1.0),
        normalize01(height),
        light_dir=light_dir,
        specular=spec_field,
        shininess=float(params.get("shininess", shininess)),
        normal_strength=float(params.get("normal_strength", normal_strength)),
        aniso_dir=aniso_dir,
        aniso=aniso,
        ambient=ambient,
        specular2=specular2,
        shininess2=shininess2,
        cavity=cavity,
    )
    if variant == "matte":
        # Matte ABS is not Lambertian: it keeps a wide soft sheen band.
        x, y = grid_coords((h, w))
        ang = float(rng.uniform(0.0, np.pi))
        proj = x * np.float32(np.cos(ang)) + y * np.float32(np.sin(ang))
        span = float(proj.max() - proj.min()) or 1.0
        centre = float(proj.min()) + span * float(rng.uniform(0.3, 0.7))
        fill = np.exp(-(((proj - np.float32(centre)) / np.float32(span * 0.45)) ** 2))
        rgb = rgb * (0.96 + np.float32(0.07) * fill)[..., None]

    if variant == "glossy":
        # A flat-topped reflected-light streak (quartic falloff, so it has
        # edges) over a broad soft fill band. The streak's coordinate is
        # distorted by the peel field: a glossy panel looks smooth everywhere
        # EXCEPT that its reflections wobble -- that wobble is the gloss cue,
        # where shading the peel directly just reads as leather.
        x, y = grid_coords((h, w))
        ang = float(rng.uniform(0.0, np.pi))
        proj = x * np.float32(np.cos(ang)) + y * np.float32(np.sin(ang))
        span = float(proj.max() - proj.min()) or 1.0
        centre = float(proj.min()) + span * float(rng.uniform(0.3, 0.7))
        # Kept below ~1/4 of the streak width: stronger wobble shreds the
        # streak's edges into smoke.
        wobble = peel * np.float32(span * rng.uniform(0.02, 0.045))
        pd = proj + wobble
        fill = np.exp(-(((proj - np.float32(centre)) / np.float32(span * 0.45)) ** 2))
        streak_w = np.float32(span * float(rng.uniform(0.10, 0.18)))
        streak = np.exp(-(((pd - np.float32(centre)) / streak_w) ** 4))
        # Fill is multiplicative (soft uneven room light); the streak is
        # ADDITIVE white -- a dielectric reflection does not scale with
        # albedo, which is why gloss shows most on dark plastic.
        rgb = rgb * (0.92 + np.float32(0.10) * fill)[..., None]
        rgb = rgb + (streak * np.float32(rng.uniform(0.07, 0.16)))[..., None]
    rgb = rgb * _vignette((h, w), strength=float(rng.uniform(0.04, 0.12)))[..., None]

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
