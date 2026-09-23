# Galvanised source and calibration register

Status: numerical optical sources are pinned; appearance calibration is incomplete.
The supplied research and design are implementation inputs, not substitutes for
their cited measurements. No photograph is distributed as a package resource.

## Numerical resources

| Resource | Provenance and use | Identity and terms |
| --- | --- | --- |
| Zinc complex IOR | Werner, Glantschnig and Ambrosch-Draxl (2009), [DOI 10.1063/1.3243762](https://doi.org/10.1063/1.3243762); numerical [Werner database file](https://github.com/polyanskiy/refractiveindex.info-database/blob/main/database/data/main/Zn/nk/Werner.yml) | Pinned local bytes and SHA-256 in the optics manifest; database file CC0; paper rights are separate |
| D65 illuminant | [Official CIE data](https://files.cie.co.at/Publications-datasets/CIE_std_illum_D65.csv) and its official metadata | Pinned source and metadata SHA-256; CIE attribution, CC BY-SA 4.0 |
| CIE 1931 2° observer | [Official CIE data](https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv) and its official metadata | Pinned source and metadata SHA-256; CIE attribution, CC BY-SA 4.0 |
| Angular zinc RGB table | `tools/galvanised/build_optics.py`, linear n/k interpolation before Fresnel, 360–830 nm at 1 nm, D65/observer integration | Derived CSV and algorithm manifest; CC BY-SA 4.0 with notices for the source datasets |
| GGX energy table | Numerical directional-albedo integration for the implemented masking model | Rebuild script, packaged table and associated method record; an approximation to the numerical model, not measured zinc |
| Morphology families | Authored templates informed by Kim et al. (2019) | Versioned JSON; no digitised specimen fit is claimed |

The optical manifest is
`src/texture_generators/data/galvanised/optics/manifest.json`.
Normal-incidence D65 linear-sRGB zinc reflectance is
`(0.8751797002, 0.8688887608, 0.8550759798)`.
Its scalar optical model does not supply orientation-resolved crystal optical
contrast. Grain contrast must arise from surface/lobe structure and lighting.

## Appearance evidence

| Source | Evidence actually available | Permitted calibration use and gap |
| --- | --- | --- |
| Kim et al. (2019), [Dendritic Morphologies of Hot-Dip Galvanized Zn–0.2 Wt Pct Al Coatings](https://link.springer.com/article/10.1007/s11661-019-05263-4) | Publisher abstract and figure images acquired into the external plan's `reference-assets` directory; Fig. 9 and Fig. 15 inspected during implementation | Fine attached branching and orientation-family structure; illustrations do not establish a universal family probability, RGB colour or roughness parameter |
| Strutzenberger and Faderl (1998), [Solidification and spangle formation](https://link.springer.com/article/10.1007/s11661-998-0144-8) | Named source and cited research account; full raw profilometry not acquired | Micron-scale relief is an initial envelope; no claim to have fitted original specimen height samples |
| [American Galvanizers Association: Wet Storage Stain](https://galvanizeit.org/education-and-resources/publications/wet-storage-stain) | Industry guidance and appearance descriptions | Qualitative moisture/confinement and deposit appearance; not a calibrated temporal corrosion model |
| [American Galvanizers Association: Hot-Dip Galvanized Coating Appearance](https://galvanizeit.org/uploads/publications/Galvanized_Coating_Appearance.pdf), pp. 2, 5 and 7 | Spangle, atmospheric weathering and wet-storage photographs and descriptions; inspected for the September 2026 release tuning | Qualitative grain contrast, dull grey patina and uneven pale deposits; uncontrolled photographs do not determine numerical roughness, reflectance or corrosion rates |
| Supplied research §10 | Explicitly labelled rendering defaults | Initial diameter, roughness, anisotropy and deposit-height authoring ranges; not universal metrology |

Publisher figures are for inspection, with their original publication rights.
They are not copied into this repository, wheel, notebook or example gallery.
The acquisition manifest in the external plan records source URLs and checksums.

## Calibration sequence and missing evidence

1. Fit the realised geometric grain median and CV using exact visible cell areas
   over at least 12 fixed development seeds. Keep raw weight variability distinct
   from cell-diameter CV. Archive the bounded candidate search.
2. Compare unlit branch morphology with scale-labelled micrographs, including
   different orientation families. Keep geometry separate from reflectance.
3. Measure combined relief and finite-band slopes on 12 different held-out seeds.
   Raw specimen profilometry is still needed for a metrological envelope.
4. Compare rich and single-lobe responses with shared-subpixel reference rendering
   across illumination, view and mip changes. Report numerical error separately
   from agreement with a physical specimen.
5. Compare deposit coverage and shape with independent weathered and wet-storage
   references. Batch defects require a separate specimen set and size statistics.

Controlled reflectance measurements, raw topography and an independent materials
review have not been obtained. Until then the implementation remains a physically
grounded authoring model with explicit evidence gaps. Numerical reproducibility
does not close those gaps.

The distribution's SPDX expression is
`Apache-2.0 AND CC-BY-SA-4.0 AND CC0-1.0`, reflecting the code and packaged
resources together. The code's Apache licence is unchanged. All four resource
notices are also declared as distribution `License-File` entries, alongside
`LICENSE`; they remain adjacent to the resources as well. This follows the
[Python packaging metadata scope](https://packaging.python.org/en/latest/specifications/core-metadata/#license-expression),
which describes the containing distribution archive rather than only its code.
