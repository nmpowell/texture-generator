# Galvanised metal contracts progress

## Status

Complete for the foundational-contract scope. No registry, CLI, project
configuration, optics, or grain-geometry files were changed.

## Scope

- `src/texture_generators/core/material.py`
- `src/texture_generators/core/physical.py`
- `src/texture_generators/core/random_fields.py`
- `src/texture_generators/materials/galvanised_config.py`
- `tests/test_galvanised_contracts.py`

## Verification

- `.venv/bin/python -m pytest tests/test_galvanised_contracts.py`
  - `16 passed in 0.13s`
- `.venv/bin/ruff check` on all owned Python files
  - `All checks passed!`
- `.venv/bin/ruff format` on all owned Python files
  - `5 files already formatted`
- `.venv/bin/mypy` on the four owned source modules
  - `Success: no issues found in 4 source files`

## Decisions

- Configuration resolution is schema defaults, named preset defaults, then
  explicit overrides. `from_resolved_mapping()` deliberately bypasses the
  current preset table for stable replay.
- `spangle_cv` means the target unweighted realised equivalent-diameter CV;
  it is not a raw power-weight parameter.
- The four initial presets and their provisional authoring values are exposed
  separately from the explicitly experimental `inconspicuous` and `batch`
  presets. All configurations carry a preset revision and calibration warning.
- Physical normal output uses signed float32 vectors in the right-handed
  `X=u`, `Y=Ly-v`, `+Z=out` frame. Small derivative axes are treated as flat,
  while sampled material maps reject dimensions below three.
- Rich lobe records use the exact non-object structured dtype from the plan.
  Validation checks frames, width ranges, four-lobes-per-material limits,
  per-pixel totals, and per-material totals against selected coverage maps.
- Random derivation uses versioned, length-delimited BLAKE2b semantic keys,
  PCG64 for bounded streams, and a stateless periodic integer lattice hash.
  Weather and topology namespaces are independent.

## Limitations

- This slice defines contracts and helpers only; it intentionally provides no
  public material generator or integration placeholder.
- Preset numbers remain physically grounded authoring defaults pending the
  calibration dossier and held-out specimen work described in the plan.
- Only the requested targeted test, lint, formatting, and type-check commands
  were run; no claim is made here about the concurrently changing full suite.
