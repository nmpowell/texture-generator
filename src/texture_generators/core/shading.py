"""Directional-light shading: Lambert diffuse plus (optionally anisotropic) specular.

This is deliberately *not* a 3D renderer. The goal is a flat texture map that
reads as a material, so the ambient floor is high (~0.55 by default) and the
diffuse term only embosses the height field rather than dominating it.

The anisotropic specular follows the Ward / Ashikhmin-Shirley idea: project
the half-vector onto the surface tangent frame and blend two effective
Blinn-Phong exponents according to the along- and across-tangent components,
which elongates the highlight lobe perpendicular to ``aniso_dir``.

On top of that there is an optional **fibre** lobe (:func:`_fibre_lobe`), for
materials that reflect off aligned fibres rather than off their surface: wood.
It takes a full 3D per-pixel tangent, because the out-of-plane component is what
the surface-tangent frame above structurally cannot see, and that component is
what makes figured wood shimmer as the light moves.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .fields import height_to_normal

__all__ = [
    "ISO_SIGMA_PER_CUTOFF",
    "as_rgb",
    "gaussian_blur",
    "iso_highpass",
    "shade",
    "shade_translucent",
    "tangent_frame",
]

# ISO 16610-21 specifies the (areal) Gaussian filter's weighting function as
#
#     s(x) = 1/(a*lc) * exp(-pi * (x / (a*lc))^2),   a = sqrt(ln2/pi) = 0.46972
#
# with ``a`` chosen so that transmission is exactly 50% at the cut-off
# wavelength ``lc``. That function is a Gaussian of standard deviation
#
#     sigma = a * lc / sqrt(2*pi) = 0.18739 * lc     <=>     lc = 5.336 * sigma
#
# The factor matters because it is a factor of five: passing a *cut-off* to a
# routine that wants a *sigma* (or the reverse) mislabels the band by 5.3x, and
# a roughness statistic quoted against the wrong band is not a roughness
# statistic. Always state the cut-off, never the sigma.
ISO_SIGMA_PER_CUTOFF = 0.18739


def as_rgb(a: np.ndarray) -> np.ndarray:
    """Promote a (H, W) field to (H, W, 3); pass (H, W, 3) through."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        return np.repeat(a[..., None], 3, axis=-1)
    return a


def _unit(v: tuple[float, float, float]) -> np.ndarray:
    """Normalise a 3-vector."""
    arr = np.asarray(v, dtype=np.float32)
    return (arr / max(float(np.linalg.norm(arr)), 1e-8)).astype(np.float32)


def tangent_frame(
    aniso_dir: float | Sequence[float] | Sequence[np.ndarray] | np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build tangent/bitangent fields from an angle, a 2-vector, or per-pixel arrays.

    Accepts a scalar angle in radians, a ``(dx, dy)`` pair, a pair of (H, W)
    arrays, or an (H, W, 2) array. Returns two arrays broadcastable to
    (H, W, 3).
    """
    tx: np.ndarray
    ty: np.ndarray
    if isinstance(aniso_dir, (int, float, np.floating)):
        tx = np.asarray(np.cos(float(aniso_dir)), dtype=np.float32)
        ty = np.asarray(np.sin(float(aniso_dir)), dtype=np.float32)
    else:
        arr = aniso_dir
        if isinstance(arr, (tuple, list)):
            tx, ty = arr[0], arr[1]
        else:
            arr = np.asarray(arr, dtype=np.float32)
            tx, ty = arr[..., 0], arr[..., 1]
        tx = np.asarray(tx, dtype=np.float32)
        ty = np.asarray(ty, dtype=np.float32)

    norm = np.sqrt(np.asarray(tx) ** 2 + np.asarray(ty) ** 2)
    norm = np.maximum(norm, 1e-8)
    tx = tx / norm
    ty = ty / norm
    zeros = np.zeros_like(np.asarray(tx, dtype=np.float32))
    tangent = np.stack([tx + zeros, ty + zeros, zeros], axis=-1).astype(np.float32)
    bitangent = np.stack([-ty + zeros, tx + zeros, zeros], axis=-1).astype(np.float32)
    if tangent.ndim == 1:
        tangent = tangent.reshape(1, 1, 3)
        bitangent = bitangent.reshape(1, 1, 3)
    return tangent, bitangent


def shade(
    albedo: np.ndarray,
    height: np.ndarray,
    light_dir: tuple[float, float, float] = (-0.55, -0.55, 0.75),
    *,
    specular: float | np.ndarray = 0.0,
    shininess: float | np.ndarray = 32.0,
    normal_strength: float = 1.0,
    aniso_dir=None,
    aniso: float = 0.0,
    ambient: float = 0.55,
    spec_tint: float = 0.0,
    specular2: float | np.ndarray = 0.0,
    shininess2: float = 300.0,
    fibre_tangent: np.ndarray | None = None,
    fibre: float | np.ndarray = 0.0,
    fibre_exponent: float = 40.0,
    fibre_colour=None,
    cavity: float = 0.0,
    cavity_depth: float | None = None,
    normalise: bool = False,
    conserve_energy: bool = False,
    return_parts: bool = False,
    height_spacing: float = 1.0,
    coat_height: np.ndarray | None = None,
    fibre_ior: float = 1.0,
    ray_tangent: np.ndarray | None = None,
    ray_weight: float | np.ndarray = 0.0,
    ray_gain: float = 1.0,
):
    """Shade an albedo/height pair into an RGB array in [0, 1].

    Args:
        albedo: (H, W, 3) or (H, W) base colour in [0, 1].
        height: (H, W) height field; only its gradients matter.
        light_dir: direction *towards* the light; default upper-left.
        specular: Blinn-Phong specular weight (0 disables it). A scalar, or an
            (H, W) array for spatially varying gloss (e.g. latewood vs
            earlywood).
        shininess: specular exponent (higher = tighter highlight). Scalar or
            (H, W) array for spatially varying roughness.
        normal_strength: multiplier on the height gradients.
        aniso_dir: brush/grain direction as an angle, a ``(dx, dy)`` pair or
            per-pixel arrays; required when ``aniso`` > 0.
        aniso: 0..1 anisotropy of the specular lobe.
        ambient: ambient floor, keeping the texture bright and flat-ish.
        spec_tint: 0..1 blend of the highlight towards the albedo colour.
            Conductors tint their reflections (a copper highlight is copper);
            dielectrics like plastic reflect white and keep this at 0.
        specular2: weight of a second, tighter isotropic lobe (a cheap
            clearcoat: glossy plastic shows a broad body sheen plus a sharp
            hot reflection). Scalar or (H, W).
        shininess2: exponent of the second lobe.
        fibre_tangent: (H, W, 3) unit vectors along a **fibre** axis, for
            materials whose reflection comes off aligned fibres rather than off
            the surface alone -- wood's chatoyance is the case this exists for.
            The third component is the point: a fibre dipping below the surface
            reflects into a different cone than one lying in it, and that is
            what makes figured wood shimmer. See ``fibre`` below.
        fibre: weight of the fibre lobe (0 disables it). Scalar or (H, W).
        fibre_exponent: longitudinal exponent of the fibre lobe.
        fibre_colour: (3,) or (H, W, 3) colour of the fibre lobe. This light has
            passed through pigment before it left, so unlike a surface highlight
            it is neither white nor fully saturated; ``None`` means white.
        cavity: 0..1 strength of concavity darkening derived from the height
            field -- recesses collect less light, which grounds relief that
            normals alone leave floating.
        cavity_depth: physical recess depth (in ``height``'s own units) at
            which ``cavity`` reaches its full darkening, forwarded to
            :func:`_cavity_shadow` as ``depth_scale``. Without it every
            render's darkest recess is *always* fully dark, whatever its
            physical depth -- a single deep feature (a plank gap) then
            dominates and every shallower recess reads relative to it rather
            than to its own depth. ``None`` (the default) keeps that
            peak-normalised path exactly.
        normalise: divide the diffuse+ambient lighting by its own mean, so the
            shaded mean is the albedo it was given rather than the albedo times
            whatever this light rig happens to average to. Needed only where the
            albedo is a *measured* colour -- wood specifies its species in
            CIELAB, and without this the shading pass silently darkened every
            board by 20-25% and the measurement was lost. **Defaults off**: the
            materials that hand-tuned their albedo against this rig (metal,
            plastic) are calibrated including its mean, and renormalising them
            would shift every one of them. :func:`shade_translucent` has the
            same option for paper, and defaults it on for the same reason.
        conserve_energy: give the specular lobes' mean back out of the diffuse,
            per channel. ``normalise`` pins the *diffuse* mean onto the albedo,
            which is the right thing when the specular is a whisper -- but a
            strong lobe (wood's fibre lobe runs to 0.2) is then pure addition,
            and a measured colour plus 0.05 is not that colour. Light reflected
            at or just under the surface never reached the pigment, so the
            diffuse is what should pay for it. **Defaults off**, again because
            metal and plastic are calibrated as they stand.
        return_parts: return ``(diffuse, specular)`` **unclipped** instead of
            the composited image, so a caller can modify one term before adding
            them. This exists for thin-film interference, which is a property of
            the *reflection* and must not touch the diffuse body -- and which
            needs the two separated before the final clip, because a difference
            of two clipped renders loses exactly the bright highlight pixels the
            film colour lives in. ``shade(...)`` is
            ``clip(sum(shade(..., return_parts=True)))``.
        height_spacing: distance between adjacent height samples, in the
            height field's own length units (see
            :func:`~texture_generators.core.fields.height_to_normal`).
            Forwarded to every normal computed inside this function, so a
            board's relief keeps the same normals whatever resolution it is
            rendered at. ``1.0`` reproduces the historical per-texel
            gradient exactly.
        coat_height: (H, W) height of a separate coat surface sitting above
            ``height``, for a film that has its own (smoother) relief -- a
            varnish self-levels, but the wood grain underneath does not. When
            given, the *surface* lobes (``specular``/``specular2``, the
            anisotropic tangent-plane projection, the micro-rim Schlick term
            and the ``ndl > 0`` gate of those lobes) use normals from
            ``coat_height`` instead of ``height``. The diffuse term,
            ``cavity`` and the fibre/ray lobes always use the substrate
            normals from ``height`` -- light that reaches the fibres or gets
            absorbed by the pigment has already crossed the film, so it is
            the wood's own relief that scatters it, not the film's.
            ``None`` (the default) uses ``height`` for both, which is
            bit-identical to not having a coat at all.
        fibre_ior: refractive index of a flat coat sitting over the fibres,
            for the fibre and ray lobes only. Before those lobes are
            evaluated the unit light direction is refracted into the coat
            (tangential components divided by ``fibre_ior``, ``z`` filled in
            from the unit-length constraint); the view stays at +z and
            refracts to itself, so only the light moves. ``1.0`` (the
            default, no boundary) takes the exact pre-existing code path.
            Must be at least 1.0 -- air cannot be denser than the coat.
        ray_tangent: (H, W, 3) unit vectors along a **ray** axis -- wood's
            medullary rays, a second fibre population running roughly
            crosswise to the grain. Rays are a distinct population, not the
            same fibres rotated: mixing the two lobes' *weights* per pixel
            (see ``ray_weight``) keeps that, where blending the two *axes*
            together first would invent diagonal fibres that exist nowhere
            in the wood. ``None`` (the default) omits the ray population.
        ray_weight: 0..1 fraction of the fibre response, per pixel, that
            comes from the ray population instead of the ordinary fibre
            axis -- typically a fleck mask. A scalar or an (H, W) array.
            The fibre term becomes
            ``fibre * ((1 - w) * lobe(fibre_tangent) + w * ray_gain *
            lobe(ray_tangent))``, with the same exponent, colour and
            (refracted) light for both. ``ray_tangent is None`` or an
            all-zero ``ray_weight`` takes the exact pre-existing
            single-population code path.
        ray_gain: gain on the ray lobe relative to the fibre lobe -- rays
            typically read slightly brighter than the surrounding fibres.

    Raises:
        ValueError: if ``fibre_ior`` is less than 1.0.

    Returns:
        (H, W, 3) float32 RGB clipped to [0, 1], or -- with ``return_parts`` --
        a ``(diffuse, specular)`` pair of unclipped (H, W, 3) float32 arrays.
    """
    if fibre_ior < 1.0:
        raise ValueError(f"fibre_ior must be >= 1.0 (got fibre_ior={fibre_ior})")

    alb = as_rgb(albedo)
    h = np.asarray(height, dtype=np.float32)
    normals = height_to_normal(h, normal_strength, spacing=height_spacing)
    if coat_height is None:
        # Bit-identical to today: no second normal computation at all.
        coat_normals = normals
    else:
        coat_normals = height_to_normal(
            np.asarray(coat_height, dtype=np.float32),
            normal_strength,
            spacing=height_spacing,
        )

    light = _unit(light_dir)
    ndl = np.clip((normals * light).sum(axis=-1), 0.0, 1.0)
    # Directional ambient: up-facing microfacets see more sky, so the ambient
    # floor itself carries a whisper of the relief.
    amb = np.float32(ambient) * (0.85 + 0.15 * np.clip(normals[..., 2], 0.0, 1.0))
    lit = amb + np.float32(1.0 - ambient) * ndl
    if cavity > 0.0:
        lit = lit * _cavity_shadow(h, np.float32(cavity), depth_scale=cavity_depth)
    if normalise:
        # Only the diffuse path: the specular below is an addition on top of the
        # surface's own reflectance, not part of it, so scaling it here would
        # make the highlight depend on how dark the rest of the render came out.
        lit = lit / np.float32(max(float(lit.mean()), 1e-6))
    out = alb * lit[..., None]

    spec_rgb = None

    spec_w = np.asarray(specular, dtype=np.float32)
    spec2_w = np.asarray(specular2, dtype=np.float32)
    if float(spec_w.max(initial=0.0)) > 0.0 or float(spec2_w.max(initial=0.0)) > 0.0:
        view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        half = _unit(tuple((light + view).tolist()))
        ndh = np.clip((coat_normals * half).sum(axis=-1), 0.0, 1.0)

        shin = np.asarray(shininess, dtype=np.float32)
        if aniso > 0.0 and aniso_dir is not None:
            tangent, bitangent = tangent_frame(aniso_dir, h.shape)
            # Tangent-plane component of the half-vector, which varies with
            # the local (coat) normal and therefore across the texture.
            proj = half.reshape(1, 1, 3) - ndh[..., None] * coat_normals
            along = (proj * tangent).sum(axis=-1)
            across = (proj * bitangent).sum(axis=-1)
            a2 = along * along
            b2 = across * across
            exp_along = shin * np.float32(1.0 + 8.0 * aniso)
            exp_across = np.maximum(
                np.float32(6.0), shin * np.float32(1.0 - 0.85 * aniso)
            )
            expo = (exp_along * a2 + exp_across * b2) / (a2 + b2 + 1e-8)
        else:
            expo = shin

        spec = spec_w * np.power(ndh, expo)
        if float(spec2_w.max(initial=0.0)) > 0.0:
            spec = spec + spec2_w * np.power(ndh, np.float32(shininess2))
        # ``coat_height is None`` means ``coat_normals is normals`` (same array),
        # so this is exactly the ``ndl`` already computed above -- reuse it
        # rather than reducing the whole frame a second time.
        ndl_coat = (
            ndl
            if coat_height is None
            else np.clip((coat_normals * light).sum(axis=-1), 0.0, 1.0)
        )
        spec = np.where(ndl_coat > 0.0, spec, 0.0).astype(np.float32)
        # Schlick term on the MICRO-geometry: steep feature edges (scratch
        # lips, dimple walls) whiten and brighten, which is where grazing
        # reflection actually lands on a face-on flat sheet. Uses the coat
        # normal, like the rest of this block: it is the film's surface that
        # grazes, not the substrate's.
        fres = np.clip(
            np.power(1.0 - np.clip(coat_normals[..., 2], 0.0, 1.0), 5.0), 0.0, 0.6
        )[..., None]
        if spec_tint > 0.0:
            t = np.float32(np.clip(spec_tint, 0.0, 1.0))
            spec_colour = (1.0 - t) + t * alb
            spec_rgb = spec[..., None] * (spec_colour + (1.0 - spec_colour) * fres)
        else:
            # Dielectric highlights are already white; the micro-rim only
            # gets half weight so steep slopes brighten without clipping.
            spec_rgb = spec[..., None] * (1.0 + 0.5 * fres)

    fibre_w = np.asarray(fibre, dtype=np.float32)
    if fibre_tangent is not None and float(fibre_w.max(initial=0.0)) > 0.0:
        if fibre_ior != 1.0:
            # Refract the light into a flat coat over the fibres. The view is
            # fixed at +z and refracts to itself, so only the light moves and
            # _fibre_lobe itself needs no change.
            ior = np.float32(fibre_ior)
            lx = light[0] / ior
            ly = light[1] / ior
            lz = np.sqrt(np.maximum(1.0 - lx * lx - ly * ly, np.float32(0.0)))
            light_fibre = np.asarray([lx, ly, lz], dtype=np.float32)
        else:
            light_fibre = light
        fib_lobe = _fibre_lobe(fibre_tangent, light_fibre, ndl, fibre_exponent)
        ray_w = np.asarray(ray_weight, dtype=np.float32)
        if ray_tangent is not None and float(ray_w.max(initial=0.0)) > 0.0:
            # Two populations mixed by weight, not two axes blended into one:
            # averaging fibre_tangent and ray_tangent first would produce a
            # direction that runs diagonally between grain and ray -- a fibre
            # angle that does not exist in the wood.
            w = np.broadcast_to(ray_w, h.shape).astype(np.float32)
            ray_lobe = _fibre_lobe(ray_tangent, light_fibre, ndl, fibre_exponent)
            fib = fibre_w * ((1.0 - w) * fib_lobe + w * np.float32(ray_gain) * ray_lobe)
        else:
            fib = fibre_w * fib_lobe
        colour = (
            np.ones(3, dtype=np.float32)
            if fibre_colour is None
            else as_rgb(np.asarray(fibre_colour, dtype=np.float32))
        )
        fib_rgb = fib[..., None] * colour
        spec_rgb = fib_rgb if spec_rgb is None else spec_rgb + fib_rgb

    if spec_rgb is not None:
        if conserve_energy:
            # Per channel, because the fibre lobe is *coloured*: a neutral
            # correction would hold the render's lightness and still move its
            # hue, and hue is exactly what the measurement pins down.
            d_mean = out.reshape(-1, 3).mean(axis=0)
            s_mean = spec_rgb.reshape(-1, 3).mean(axis=0)
            gain = np.clip(
                (d_mean - s_mean) / np.maximum(d_mean, np.float32(1e-6)), 0.2, 1.0
            ).astype(np.float32)
            out = out * gain[None, None, :]
        if return_parts:
            return out.astype(np.float32), spec_rgb.astype(np.float32)
        out = out + spec_rgb

    if return_parts:
        return out.astype(np.float32), np.zeros_like(out, dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fibre_lobe(
    tangent: np.ndarray,
    light: np.ndarray,
    ndl: np.ndarray,
    exponent: float,
) -> np.ndarray:
    """Longitudinal specular lobe about a fibre axis (Kajiya-Kay / Marschner).

    A fibre is a cylinder, so it does not have one mirror direction: it reflects
    into a **cone** about its own axis, and the lobe peaks where the light's and
    the view's inclinations to the axis are equal and opposite. Writing
    ``sin(theta_l) = t.l`` and ``sin(theta_v) = t.v``, that condition is
    ``theta_h = (theta_l + theta_v) / 2 == 0``, and ``cos(theta_h)**n`` is the
    lobe around it.

    This is *not* the same shape as the anisotropic Blinn-Phong lobe above, and
    the difference is the whole reason it exists here. Blinn-Phong's tangent
    frame only ever sees the tangent's in-plane part -- on a face-on flat surface
    the half-vector's projection has no z component at all, so a fibre's dip
    below the surface cancels out of it, and the dip is precisely what makes
    figured wood shimmer. Here a dip of +/-20 degrees swings the lobe over
    nearly its whole range.
    """
    t = np.asarray(tangent, dtype=np.float32)
    t = t / np.maximum(np.sqrt((t * t).sum(axis=-1, keepdims=True)), np.float32(1e-8))
    # View is straight down the +z axis, as everywhere else in this module.
    sin_l = np.clip((t * light).sum(axis=-1), -1.0, 1.0)
    sin_v = np.clip(t[..., 2], -1.0, 1.0)
    theta_h = np.float32(0.5) * (np.arcsin(sin_l) + np.arcsin(sin_v))
    lobe = np.power(np.clip(np.cos(theta_h), 0.0, 1.0), np.float32(exponent))
    return np.where(ndl > 0.0, lobe, 0.0).astype(np.float32)


def _cavity_shadow(
    h: np.ndarray,
    strength: np.float32,
    periodic: bool = False,
    depth_scale: float | None = None,
) -> np.ndarray:
    """Multiplicative darkening where the height field is locally concave.

    A 3x3 box mean stands in for the neighbourhood: pixels below their
    surroundings sit in a recess and lose up to ``strength`` of their light.

    ``depth_scale``, in ``h``'s own units, is the recess depth at which that
    loss is complete. ``None`` (the default) normalises by this field's own
    deepest recess instead, so darkening is always relative to whatever the
    single darkest feature happens to be -- fine for a field with one kind of
    recess, wrong once recesses of very different physical depths share a
    field (a fine vessel trough and a plank gap), where the deep one would
    otherwise wash out every shallower one's shadow.
    """
    # Edge-padded neighbourhood: most of these textures are not tileable, so
    # the opposite edge is not a neighbour. Paper's fields are periodic, so it
    # asks for wrap instead.
    hp = np.pad(h, 1, mode="wrap" if periodic else "edge")
    hh, ww = h.shape
    acc = np.zeros_like(h)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            acc = acc + hp[dy : dy + hh, dx : dx + ww]
    mean = acc * np.float32(1.0 / 9.0)
    depth = np.clip(mean - h, 0.0, None)
    if depth_scale is None:
        peak = float(depth.max())
        if peak > 1e-8:
            depth = depth / np.float32(peak)
    else:
        depth = np.clip(depth / np.float32(max(float(depth_scale), 1e-8)), 0.0, 1.0)
    return (1.0 - strength * depth).astype(np.float32)


def gaussian_blur(a: np.ndarray, sigma: float, mode: str = "wrap") -> np.ndarray:
    """Separable Gaussian blur, via FFT-free direct convolution.

    Small sigmas dominate here (0.3-6 px), so a truncated separable kernel is
    cheaper than a transform.

    Args:
        a: (H, W) field to blur.
        sigma: Gaussian standard deviation, in pixels.
        mode: ``"wrap"`` (the default) blurs periodically via ``np.roll``,
            bit-identical to every existing caller. ``"edge"`` instead pads
            the array by the kernel radius with edge replication
            (:func:`numpy.pad`'s ``mode="edge"``) and convolves separably
            without wraparound, then crops back to shape -- for a field that
            is not tileable (a board's own relief), so the blur is not mixed
            with its own opposite edge as a neighbour, which shows up as a
            bright or dark band down one side.

    Raises:
        ValueError: for any ``mode`` other than ``"wrap"`` or ``"edge"``.
    """
    if mode not in ("wrap", "edge"):
        raise ValueError(
            f"gaussian_blur mode must be 'wrap' or 'edge' (got mode={mode!r})"
        )
    a = np.asarray(a, dtype=np.float32)
    sigma = float(sigma)
    if sigma <= 1e-3:
        return a
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / np.float32(sigma)) ** 2)
    k /= k.sum()
    if mode == "wrap":
        out = np.zeros_like(a)
        for offset, weight in zip(range(-radius, radius + 1), k, strict=False):
            out += weight * np.roll(a, offset, axis=1)
        acc = np.zeros_like(out)
        for offset, weight in zip(range(-radius, radius + 1), k, strict=False):
            acc += weight * np.roll(out, offset, axis=0)
        return acc.astype(np.float32)
    # mode == "edge": pad by the kernel radius with edge replication, then
    # convolve separably by summing shifted slices of the padded array --
    # every slice start/end stays within the padded bounds by construction,
    # so this is the crop as well as the convolution.
    hh, ww = a.shape
    padded = np.pad(a, radius, mode="edge")
    across = np.zeros((hh + 2 * radius, ww), dtype=np.float32)
    for offset, weight in zip(range(-radius, radius + 1), k, strict=False):
        across += weight * padded[:, radius + offset : radius + offset + ww]
    out = np.zeros((hh, ww), dtype=np.float32)
    for offset, weight in zip(range(-radius, radius + 1), k, strict=False):
        out += weight * across[radius + offset : radius + offset + hh, :]
    return out.astype(np.float32)


def iso_highpass(field: np.ndarray, cutoff_px: float) -> np.ndarray:
    """S-filter a field to its roughness band, by **cut-off wavelength** in pixels.

    The ISO 16610-21 areal Gaussian filter is a low-pass whose transmission is
    50% at the cut-off wavelength ``lc``; the roughness band is what is left when
    that low-pass (the waviness) is subtracted. Its weighting function is a
    Gaussian of sd ``0.18739 * lc`` -- see :data:`ISO_SIGMA_PER_CUTOFF` -- so a
    "1 mm" filter written as ``sigma = 1 mm`` is really a **5.34 mm** cut-off and
    leaves cockle and millimetre-scale periodic structure inside the "roughness".

    Takes a cut-off rather than a sigma precisely so that the number a caller
    quotes is the number that defines the band.
    """
    return np.asarray(field, dtype=np.float32) - gaussian_blur(
        field, ISO_SIGMA_PER_CUTOFF * float(cutoff_px)
    )


def shade_translucent(
    albedo: np.ndarray,
    height: np.ndarray,
    *,
    light_dir: tuple[float, float, float] = (-0.5, -0.62, 0.6),
    diffuse_sigma: float = 2.0,
    diffuse_strength: float = 0.9,
    spec_strength: float = 2.2,
    wrap: float = 0.4,
    roughness_deg: float = 20.0,
    sheen: float | np.ndarray = 0.03,
    sheen_exponent: float = 14.0,
    ambient: float = 0.6,
    occlusion: float = 0.25,
    normalise: bool = True,
) -> np.ndarray:
    """Shade a translucent fibrous sheet, in linear light, using *two* normals.

    Light does not bounce off paper's surface; it enters the sheet, diffuses
    laterally over a point-spread radius -- roughly 90-250 um for uncoated
    stock, a few tens of um once a pigment coat scatters the light at the
    surface -- and leaves. Height variation finer than that radius therefore
    produces almost no *diffuse* shading -- it is low-pass filtered away --
    while the specular reflection at the air-fibre interface keeps every bit
    of it. Shading a paper height field with one normal has to choose between
    the two, and either choice looks wrong: sharp diffuse gives embossed
    plastic, soft everything gives a flat card.

    So the diffuse term uses a normal built from the *blurred* height and the
    sheen term uses one built from the full-detail height.

    Args:
        albedo: (H, W, 3) or (H, W) base colour, **linear light**.
        height: (H, W) height field.
        light_dir: direction towards the light.
        diffuse_sigma: subsurface point-spread radius in pixels.
        diffuse_strength: normal strength for the diffuse (blurred) normal.
        spec_strength: normal strength for the specular (sharp) normal.
        wrap: wrap-diffuse term, 0.35-0.5 for paper; softens the terminator
            the way forward scattering through a thin sheet does.
        roughness_deg: Oren-Nayar sigma. Paper is a rough Lambertian, ~15-25
            degrees, which flattens the terminator and adds retroreflection.
        sheen: weight of the broad dielectric sheen lobe. Uncoated paper is
            0.02-0.04; a scalar or an (H, W) array for calender freckling.
        sheen_exponent: Blinn-Phong exponent. Low (8-20) for uncoated: a
            tight lobe reads as plastic.
        ambient: ambient floor.
        occlusion: strength of directional occlusion from the blurred height.
        normalise: divide the lighting by its own mean, so the sheet's average
            reflectance is exactly the albedo it was given. Paper colour is
            specified as a measured L*a*b* value; without this the shading
            pass darkens it by an arbitrary amount and the measurement is lost.

    Returns:
        (H, W, 3) float32 in linear light (not clipped to display range).
    """
    alb = as_rgb(albedo)
    h = np.asarray(height, dtype=np.float32)

    n_spec = height_to_normal(h, spec_strength, periodic=True)
    n_diff = height_to_normal(
        gaussian_blur(h, diffuse_sigma), diffuse_strength, periodic=True
    )

    light = _unit(light_dir)
    ndl = (n_diff * light).sum(axis=-1)
    ndv = np.clip(n_diff[..., 2], 1e-4, 1.0)
    # Wrap diffuse: light entering the sheet re-emerges past the geometric
    # terminator, so the falloff is softer than Lambert.
    w = np.float32(max(wrap, 0.0))
    ndl_w = np.clip((ndl + w) / (1.0 + w), 0.0, 1.0)

    # Oren-Nayar (Fujii's cheap form), driven by the same soft normal.
    s2 = np.float32(np.deg2rad(roughness_deg) ** 2)
    a_on = 1.0 - 0.5 * s2 / (s2 + 0.33)
    b_on = 0.45 * s2 / (s2 + 0.09)
    ldv = np.float32(light[2])
    cos_delta = ldv - np.clip(ndl, 0.0, 1.0) * ndv
    denom = np.maximum(np.maximum(np.clip(ndl, 0.0, 1.0), ndv), 1e-4)
    on = np.float32(a_on) + np.float32(b_on) * np.where(
        cos_delta <= 0.0, 0.0, cos_delta / denom
    )

    lit = np.float32(ambient) + np.float32(1.0 - ambient) * ndl_w * on

    if occlusion > 0.0:
        lit = lit * _cavity_shadow(
            gaussian_blur(h, diffuse_sigma), np.float32(occlusion), periodic=True
        )
    if normalise:
        lit = lit / np.float32(max(float(lit.mean()), 1e-6))

    out = alb * lit[..., None]

    sheen_w = np.asarray(sheen, dtype=np.float32)
    if float(sheen_w.max(initial=0.0)) > 0.0:
        view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        half = _unit(tuple((light + view).tolist()))
        ndh = np.clip((n_spec * half).sum(axis=-1), 0.0, 1.0)
        spec = sheen_w * np.power(ndh, np.float32(sheen_exponent))
        spec = np.where((n_spec * light).sum(axis=-1) > 0.0, spec, 0.0)
        out = out + spec[..., None]

    return out.astype(np.float32)
