# Galvanised optics foundations

Status: implemented and numerically verified for exact conductor Fresnel,
the source-derived angular spectral LUT, and single-scattering GGX.

## Runtime API

The public foundations in `texture_generators.core` are:

```python
conductor_fresnel(cos_theta, n, k) -> numpy.ndarray[numpy.float64]
zinc_fresnel(cos_theta) -> numpy.ndarray[numpy.float64]
zinc_f0() -> numpy.ndarray[numpy.float64]
resource_hashes() -> collections.abc.Mapping[str, str]

openpbr_widths(roughness, anisotropy) -> tuple[numpy.ndarray, numpy.ndarray]
tangent_frame(normal, doubled_axis) -> tuple[numpy.ndarray, numpy.ndarray]
evaluate_ggx(normal, tangent, wi, wo, alpha_t, alpha_b, fresnel) -> numpy.ndarray
```

`conductor_fresnel` uses `m = n + ik`, complex transmitted-wave geometry and
selects the square root with positive imaginary part (positive real part for a
lossless tie). Inputs are float64 and broadcast using NumPy. `zinc_fresnel`
returns RGB on its final axis and linearly interpolates a cached, read-only
table in cosine. `zinc_f0()` is `(0.87517970, 0.86888876, 0.85507598)`.
`resource_hashes()` returns immutable entries named `Werner.yml`,
`CIE_std_illum_D65.csv`, `CIE_xyz_1931_2deg.csv`,
`CIE_std_illum_D65.csv_metadata_v2.json`,
`CIE_xyz_1931_2deg.csv_metadata.json`, `zinc_fresnel_lut.csv`, and the optical
`manifest.json`. Runtime checks every source/derived resource checksum and the
agreement between manifest F0 and the LUT endpoint before exposing identities.

`openpbr_widths` implements the OpenPBR 1.1 mapping, with both widths floored at
`1e-4`. `tangent_frame` decodes the clockwise raster doubled-angle axis and
projects `(cos(theta), -sin(theta), 0)` onto the signed normal. The bitangent is
`normal x tangent`. `evaluate_ggx` is the independently callable conductor
single-scattering BRDF: anisotropic GGX NDF, height-correlated Smith masking and
shadowing, positive-hemisphere gating, and no Lambert term.

## Sources and generated resources

The offline builder is `tools/galvanised/build_optics.py`. It downloads only
these named sources, rejects any source-byte mismatch, preserves the bytes and
metadata in `sources/`, interpolates n and k onto 360--830 nm at 1 nm, and uses
trapezoidal D65/CIE 1931 2 degree integration:

- Werner, Glantschnig and Ambrosch-Draxl, *J. Phys. Chem. Ref. Data* 38,
  1013--1092 (2009), DOI `10.1063/1.3243762`; database file `Werner.yml`, CC0,
  SHA-256 `c29023bed42520fbf3cec2272c9f31080429508073edf9124a0e76cbcaee6abf`.
- CIE standard illuminant D65, `CIE_std_illum_D65.csv`, CC BY-SA 4.0,
  SHA-256 `e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb`.
- CIE 1931 2 degree observer, `CIE_xyz_1931_2deg.csv`, CC BY-SA 4.0,
  SHA-256 `fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1`.

The source, CIE, and derived-table notices are separate files. The CIE-derived
table is labelled CC BY-SA 4.0 rather than folded into the repository's source
licence. `manifest.json` records URLs, hashes, integration policy, perfect-white
check, and derived-resource identity. Runtime performs no network access and
loads through `importlib.resources` from the installed package data root.
There is no source-tree fallback. Built wheel and rebuilt sdist smoke checks
exercise the same resources outside the source tree.

Direct DNS and the browser surface were unavailable, so primary-source checking
used the connected GitHub API. The Werner file was read from the upstream
refractiveindex.info database. Byte-identical CIE mirrors were accepted only
after their SHA-256 values matched the checksums published in the CIE metadata;
the metadata bytes and licence fields are preserved too. OpenPBR 1.1.1 was
checked at upstream commit `f8d6d947dfae4c9b599965a86c22826ea7a8dbfb`; its
specification gives the implemented width mapping verbatim.

## Numerical verification

`tests/test_galvanised_optics.py` contains substantive G8/G9 fixtures:

- analytic normal-incidence checks at seven independently transcribed Werner
  nodes, plus 100,001 dense grazing samples;
- target F0, independently recomputed perfect-reflector neutrality, immutable
  source identities, and angular LUT error against a separate 471-wavelength
  spectral evaluation;
- OpenPBR width mapping, signed-normal projection, stable fallback,
  orthogonality, and canonical clockwise axis conversion;
- reciprocity, isotropic rotation invariance, pi tangent symmetry, backface
  gating, finite grazing evaluation, and three Gauss-Legendre white-conductor
  energy integrations, each bounded by `1.005`.

## Limits and next work

The independently callable single-scattering evaluator loses energy at high
roughness by design. Production rendering adds the separate reciprocal
directional-albedo compensation model documented in
[the rendering record](galvanised-render.md), with packaged tables and white
furnace checks over a declared domain. Zinc F0 is never boosted, and no render
is normalised per image. The untested sharp-lobe/grazing extremes and appearance
calibration remain explicit limits.

The scalar optical model does not establish an orientation-resolved dielectric
tensor. GGX widths and anisotropy are rendering parameters, not assertions of
measured zinc morphology or anisotropy. D65-weighted RGB remains an illuminant-
specific approximation even after the final spectral LUT is regenerated.
