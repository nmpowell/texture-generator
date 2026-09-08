"""Thin-film interference colour: a spectrally integrated thickness -> RGB table.

A transparent film on a reflective substrate reflects from *both* of its
surfaces. The two paths differ by an optical path of ``2 * n1 * d * cos(theta1)``,
which is fixed while the wavelength varies, so the reflectance oscillates in
``1 / lambda`` -- and the period of that oscillation **shrinks as the thickness
grows**. That is why tempered steel runs through a colour ladder as its oxide
thickens, and why an oil slick is rainbow-coloured at its thin edges and pearly
grey in the middle: by a micron the fringes are finer than the eye's colour
matching functions and average away.

That shrinking period is also the trap. Evaluating the Airy summation at three
RGB primaries is confidently, catastrophically wrong above a few hundred
nanometres: the three samples alias against fringes narrower than the gaps
between them and return a saturated colour with no relation to what the eye
would see. So this module **integrates spectrally** -- 36 samples at 10 nm over
380-730 nm against the CIE 1931 2-degree observer -- and bakes the result into a
small lookup table at import time. The fringe spacing at the coarsest case here
(2 um of oil) is ``lambda^2 / (2*n*d)`` ~ 52 nm, comfortably above the 20 nm
Nyquist limit of a 10 nm grid, so the desaturation of a thick film is *earned*
by the integration rather than asserted.

Runtime cost is then one bilinear gather per pixel over a (256 thickness x
16 angle x 3) table -- 48 KB per optical system.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SYSTEMS",
    "TEMPER_LADDER",
    "FilmSystem",
    "film_table",
    "film_tint",
    "wavelength_rgb",
]

# CIE 1931 2-degree standard observer, 380-730 nm at 10 nm (36 samples).
_CMF_START_NM = 380.0
_CMF_STEP_NM = 10.0
_CMF = np.asarray(
    [
        [0.001368, 0.000039, 0.006450],
        [0.004243, 0.000120, 0.020050],
        [0.014310, 0.000396, 0.067850],
        [0.043510, 0.001210, 0.207400],
        [0.134380, 0.004000, 0.645600],
        [0.283900, 0.011600, 1.385600],
        [0.348280, 0.023000, 1.747060],
        [0.336200, 0.038000, 1.772110],
        [0.290800, 0.060000, 1.669200],
        [0.195360, 0.090980, 1.287640],
        [0.095640, 0.139020, 0.812950],
        [0.032010, 0.208020, 0.465180],
        [0.004900, 0.323000, 0.272000],
        [0.009300, 0.503000, 0.158200],
        [0.063270, 0.710000, 0.078250],
        [0.165500, 0.862000, 0.042160],
        [0.290400, 0.954000, 0.020300],
        [0.433450, 0.994950, 0.008750],
        [0.594500, 0.995000, 0.003900],
        [0.762100, 0.952000, 0.002100],
        [0.916300, 0.870000, 0.001650],
        [1.026300, 0.757000, 0.001100],
        [1.062200, 0.631000, 0.000800],
        [1.002600, 0.503000, 0.000340],
        [0.854450, 0.381000, 0.000190],
        [0.642400, 0.265000, 0.000050],
        [0.447900, 0.175000, 0.000020],
        [0.283500, 0.107000, 0.000000],
        [0.164900, 0.061000, 0.000000],
        [0.087400, 0.032000, 0.000000],
        [0.046770, 0.017000, 0.000000],
        [0.022700, 0.008210, 0.000000],
        [0.011359, 0.004102, 0.000000],
        [0.005790, 0.002091, 0.000000],
        [0.002899, 0.001047, 0.000000],
        [0.001440, 0.000520, 0.000000],
    ],
    dtype=np.float64,
)
_LAMBDA_NM = _CMF_START_NM + _CMF_STEP_NM * np.arange(_CMF.shape[0], dtype=np.float64)

# sRGB primaries, D65 (IEC 61966-2-1) -- the same matrix core.colour uses.
_XYZ_TO_RGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

N_THICKNESS = 256
N_ANGLE = 16
MAX_ANGLE_DEG = 80.0

_THETA = np.deg2rad(np.linspace(0.0, MAX_ANGLE_DEG, N_ANGLE))


class FilmSystem:
    """One film-on-substrate optical system: indices and a thickness range.

    Attributes:
        n_film: real refractive index of the transparent film.
        n_sub: complex refractive index of the substrate (a metal, so ``k`` is
            of the same order as ``n``).
        max_nm: top of the tabulated thickness range, in nanometres.
    """

    __slots__ = ("max_nm", "n_film", "n_sub", "name", "note")

    def __init__(
        self,
        name: str,
        n_film: float,
        n_sub: complex,
        max_nm: float,
        note: str = "",
    ) -> None:
        self.name = name
        self.n_film = float(n_film)
        self.n_sub = complex(n_sub)
        self.max_nm = float(max_nm)
        self.note = note


# UNVERIFIED domain knowledge: the film indices below are representative
# literature values, not measurements of any particular specimen.
#   - Steel tempering oxide is a mixed Fe3O4/Fe2O3 scale; the effective index of
#     the *interference-relevant* transparent layer is quoted anywhere from 1.8
#     to 2.6. 1.90 is chosen here because it puts the ladder's landmarks (purple
#     ~95 nm, royal blue ~130 nm) where the tempering table below wants them.
#   - Oil / diesel on steel: n ~ 1.45-1.50.
#   - Anodised titanium / niobium barrier oxide (TiO2 / Nb2O5): n ~ 2.3-2.5.
# NOTE on aluminium: coloured anodised aluminium is NOT interference. Its film
# is 5-25 um of *porous* oxide -- three orders of magnitude too thick to
# interfere -- and the colour is dye held in the pores. It is deliberately not
# modelled here.
SYSTEMS = {
    "oxide": FilmSystem("oxide", 1.90, 2.90 + 3.20j, 450.0, "steel tempering scale"),
    "oil": FilmSystem("oil", 1.47, 2.90 + 3.20j, 2200.0, "oil / fuel film on steel"),
    "titania": FilmSystem("titania", 2.40, 2.60 + 3.30j, 300.0, "anodised Ti / Nb"),
}

# Steel tempering ladder -- UNVERIFIED domain knowledge (workshop tables), kept
# here because the thickness field generators quote it. Its endpoints are
# consistent with the VERIFIED result that an interference film on steel runs
# brown -> blue -> gold -> purple -> green as thickness goes 80 -> 400 nm
# (Surface & Coatings Technology, S0257897206003495), and with the VERIFIED
# fact that steel tempering colours occur over 200-400 degrees C.
TEMPER_LADDER = (
    ("pale yellow", 200, 40),
    ("light straw", 220, 50),
    ("dark straw", 240, 60),
    ("brown", 255, 70),
    ("brown-purple", 265, 80),
    ("purple", 280, 95),
    ("dark blue", 290, 110),
    ("royal blue", 300, 130),
    ("pale grey-blue", 330, 150),
)


def _fresnel(
    n_i: np.ndarray, cos_i: np.ndarray, n_t: np.ndarray, cos_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Complex amplitude reflection coefficients ``(r_s, r_p)`` at one interface."""
    r_s = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)
    r_p = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)
    return r_s, r_p


def _spectral_reflectance(
    system: FilmSystem, thickness_nm: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Unpolarised reflectance ``R[thickness, angle, wavelength]`` via Airy.

    Three media -- air, the transparent film, the (absorbing) metal -- so the
    two-interface Airy summation is exact, with complex refraction angles from
    Snell's law and the s/p average that unpolarised illumination implies.
    """
    d = np.asarray(thickness_nm, dtype=np.float64)[:, None, None]
    th = np.asarray(theta, dtype=np.float64)[None, :, None]
    lam = _LAMBDA_NM[None, None, :]

    n0 = np.complex128(1.0)
    n1 = np.complex128(system.n_film)
    n2 = np.complex128(system.n_sub)

    sin0 = np.sin(th).astype(np.complex128)
    cos0 = np.cos(th).astype(np.complex128)
    cos1 = np.sqrt(1.0 - (n0 * sin0 / n1) ** 2)
    cos2 = np.sqrt(1.0 - (n0 * sin0 / n2) ** 2)

    r01_s, r01_p = _fresnel(n0, cos0, n1, cos1)
    r12_s, r12_p = _fresnel(n1, cos1, n2, cos2)

    # Round-trip phase through the film. This is the 1/lambda that everything
    # else follows from.
    beta = 2.0 * np.pi * n1 * d * cos1 / lam
    phase = np.exp(2.0j * beta)

    out = 0.0
    for r01, r12 in ((r01_s, r12_s), (r01_p, r12_p)):
        r = (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)
        out = out + np.abs(r) ** 2
    return (0.5 * out).astype(np.float64)


def wavelength_rgb(lam_nm: np.ndarray | float) -> np.ndarray:
    """Linear sRGB of a **monochromatic** wavelength, from the observer above.

    The spectral locus lies outside the sRGB gamut, so the negative lobes are
    clipped: what survives is the most saturated in-gamut colour with that hue,
    which is the honest answer for a display. Normalised by the peak of ``ybar``,
    so a 555 nm ray comes out at luminance ~1 and the caller's weight is the only
    scale factor.

    This is the *diffraction* counterpart to :func:`film_tint`. A film mixes the
    whole spectrum and needs the integral; a grating sends one wavelength per
    direction and needs a single point of the same table.

    Args:
        lam_nm: wavelength(s) in nanometres. Values outside 380-730 nm clamp to
            the ends of the table (both of which are nearly black anyway), so
            callers are expected to mask their own visible band.

    Returns:
        Float32 array of shape ``lam_nm.shape + (3,)``, linear light, >= 0.
    """
    lam = np.asarray(lam_nm, dtype=np.float64)
    xyz = np.stack(
        [np.interp(lam, _LAMBDA_NM, _CMF[:, i]) for i in range(3)], axis=-1
    ) / float(_CMF[:, 1].max())
    return np.clip(xyz @ _XYZ_TO_RGB.T, 0.0, None).astype(np.float32)


def film_table(system: FilmSystem) -> np.ndarray:
    """Build the ``(N_THICKNESS, N_ANGLE, 3)`` **linear** sRGB table for ``system``.

    Row 0 is the zero-thickness reference (bare substrate seen through a film of
    no depth), which is what :func:`film_tint` divides by so the result is a
    multiplicative tint that is exactly neutral where there is no film.
    """
    thickness = np.linspace(0.0, system.max_nm, N_THICKNESS)
    refl = _spectral_reflectance(system, thickness, _THETA)
    # Equal-energy illuminant: the substrate's own colour is divided out below,
    # so what survives is the interference, not a lamp.
    xyz = refl @ _CMF
    xyz = xyz / _CMF[:, 1].sum()
    rgb = xyz @ _XYZ_TO_RGB.T
    return np.clip(rgb, 0.0, None).astype(np.float32)


_TABLES = {name: film_table(sys) for name, sys in SYSTEMS.items()}


def table_for(system: str) -> np.ndarray:
    """The prebuilt table for a named system in :data:`SYSTEMS`."""
    try:
        return _TABLES[system]
    except KeyError:
        raise ValueError(
            f"unknown film system {system!r}; choose from {sorted(SYSTEMS)}"
        ) from None


def film_tint(
    system: str,
    thickness_nm: np.ndarray,
    cos_theta: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Look up the linear-RGB tint for a thickness field, relative to no film.

    Args:
        system: key into :data:`SYSTEMS`.
        thickness_nm: (H, W) film thickness in nanometres. Values above the
            system's tabulated range are clamped (a film that thick is neutral
            anyway, which is the whole point of integrating spectrally).
        cos_theta: (H, W) or scalar cosine of the incidence angle.

    Returns:
        (H, W, 3) float32 multiplier in linear light, ~1 where the film is
        absent and swinging above and below 1 per channel as it thickens.
    """
    table = table_for(system)
    sysdef = SYSTEMS[system]

    d = np.clip(np.asarray(thickness_nm, dtype=np.float32), 0.0, sysdef.max_nm)
    ti = d * np.float32((N_THICKNESS - 1) / sysdef.max_nm)

    theta = np.arccos(np.clip(np.asarray(cos_theta, dtype=np.float32), 0.0, 1.0))
    ai = np.clip(
        theta * np.float32((N_ANGLE - 1) / _THETA[-1]), 0.0, np.float32(N_ANGLE - 1)
    )
    ai = np.broadcast_to(ai, d.shape)

    # Bilinear gather, written out because the table is tiny and np.interp is
    # 1-D only. Two lerps, no per-pixel Airy summation anywhere.
    t0 = np.floor(ti).astype(np.int32)
    t1 = np.minimum(t0 + 1, N_THICKNESS - 1)
    tf = (ti - t0)[..., None]
    a0 = np.floor(ai).astype(np.int32)
    a1 = np.minimum(a0 + 1, N_ANGLE - 1)
    af = (ai - a0)[..., None]

    top = table[t0, a0] * (1.0 - af) + table[t0, a1] * af
    bot = table[t1, a0] * (1.0 - af) + table[t1, a1] * af
    value = top * (1.0 - tf) + bot * tf

    ref = table[0, a0] * (1.0 - af) + table[0, a1] * af
    return np.clip(value / np.maximum(ref, 1e-6), 0.0, 3.0).astype(np.float32)
