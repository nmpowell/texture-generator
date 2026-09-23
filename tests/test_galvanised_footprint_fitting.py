"""Independent physical extraction, rejection and retained-record contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from texture_generators.core import footprint_fitting as fitting
from texture_generators.core import material_render
from texture_generators.core.material import LOBE_DTYPE
from texture_generators.core.material_render import ReferenceDerivativeConvergenceError
from texture_generators.materials.galvanised import build_state
from texture_generators.materials.galvanised_config import GalvanisedConfig


@pytest.fixture(scope="module")
def flat_state():
    return build_state(
        GalvanisedConfig(
            size_mm=(8, 6),
            spangle_diameter_mm=3,
            texture_strength=0,
            anisotropy=0,
            boundary_depth_um=0,
            micro_relief_um=0,
        ),
        seed=42,
    )


@pytest.fixture(scope="module")
def real_state():
    return build_state(GalvanisedConfig(size_mm=(8, 6), spangle_diameter_mm=3), seed=42)


def test_flat_physical_footprint_matches_constant_response(flat_state):
    result = fitting.fit_physical_footprint(
        flat_state, fitting.PhysicalFootprint((2, 2))
    )
    assert result.accepted
    assert result.records.dtype == LOBE_DTYPE
    assert 1 <= len(result.records) <= 4
    assert abs(float(result.records["weight"].astype(float).sum()) - 1) <= 1e-7
    np.testing.assert_array_equal(
        result.records["normal_ts"], np.tile([0, 0, 1], (len(result.records), 1))
    )
    assert result.report["accepted_spatial_rate"] == 16
    assert [v["rate"] for v in result.report["integration"]] == [8, 16, 17]
    for name in ("training", "held_out"):
        unit = result.arrays[f"candidates_0_{name}"][0]
        # The renderer's compensated-energy lookup has a small azimuth
        # interpolation error even at equal widths; retain its actual response.
        np.testing.assert_allclose(
            result.arrays[f"target_16_{name}"][0], unit, rtol=1e-6, atol=6e-6
        )
        assert result.report["storage_errors"][f"{name}_total"]["passed"]


def test_real_footprint_keeps_actual_candidates_and_is_batch_independent(real_state):
    footprint = fitting.PhysicalFootprint((2, 2))
    first = fitting.fit_physical_footprint(real_state, footprint)
    second = fitting.fit_physical_footprint(real_state, footprint, point_batch_size=7)
    assert first.accepted and second.accepted
    np.testing.assert_array_equal(first.records, second.records)
    for name, values in first.arrays.items():
        np.testing.assert_array_equal(values, second.arrays[name])
    candidates = first.arrays["candidates_0"]
    chosen = first.report["material_fits"]["0"]["candidate_indices"]
    for field in (
        "normal_ts",
        "tangent_ts",
        "alpha_t",
        "alpha_b",
        "optical_parameter_index",
    ):
        np.testing.assert_array_equal(
            first.records[field],
            candidates[chosen][field].astype(first.records[field].dtype),
        )
    assert len(first.records) <= 4
    assert np.all(first.records["weight"] > 0)
    assert abs(float(first.records["weight"].astype(float).sum()) - 1) <= 1e-7
    assert not first.records.flags.writeable


def test_angular_training_and_held_out_grids_are_disjoint():
    training = fitting.angular_directions()
    held_out = fitting.angular_directions(held_out=True)
    assert training.shape == (64, 3)
    assert held_out.shape == (76, 3)
    assert np.min(np.linalg.norm(training[:, None] - held_out[None, :], axis=-1)) > 0.01
    np.testing.assert_allclose(np.linalg.norm(training, axis=-1), 1, atol=1e-14)


def test_independent_spatial_check_rejects_false_dyadic_agreement(
    flat_state, monkeypatch
):
    original = fitting.sample_points

    def aliased(state, x, y):
        result = original(state, x, y)
        result["intrinsic_roughness"] = 0.3 + 0.1 * np.cos(
            2 * np.pi * 32 * (x - 2) / 0.125
        )
        return result

    monkeypatch.setattr(fitting, "sample_points", aliased)
    with pytest.raises(fitting.PhysicalFootprintFitError) as caught:
        fitting.fit_physical_footprint(flat_state, fitting.PhysicalFootprint((2, 2)))
    result = caught.value.result
    assert result.report["stage"] == "spatial"
    assert not result.accepted and len(result.records) == 0
    assert result.report["spatial_checks"][0]["successive"]["passed"]
    assert not result.report["spatial_checks"][0]["independent"]["passed"]
    assert result.report["integration"][-1]["rate"] == 33


def test_reference_checks_held_out_targets_independently(flat_state, monkeypatch):
    original = fitting._responses

    def oracle(records, directions, optical):
        response = original(records, directions, optical)
        if len(directions) == 76:
            x = records["position_mm"][:, 0]
            response *= (1 + 0.3 * np.cos(2 * np.pi * 32 * (x - 2) / 0.125))[:, None]
        return response

    monkeypatch.setattr(fitting, "_responses", oracle)
    with pytest.raises(fitting.PhysicalFootprintFitError) as caught:
        fitting.fit_physical_footprint(flat_state, fitting.PhysicalFootprint((2, 2)))
    result = caught.value.result
    assert result.report["stage"] == "spatial"
    errors = result.report["spatial_checks"][0]["independent"]["response_errors"]
    assert errors["training_total"]["passed"]
    assert not errors["held_out_total"]["passed"]


def test_positive_reference_material_without_candidate_is_rejected(
    flat_state, monkeypatch
):
    original = fitting.sample_points

    def thin_deposit(state, x, y):
        result = original(state, x, y)
        patina = np.where(np.abs(x - 2) < 0.006, 1e-4, 0)
        result["patina_coverage"] = patina
        result["metallic"] = 1 - patina
        return result

    monkeypatch.setattr(fitting, "sample_points", thin_deposit)
    with pytest.raises(
        fitting.PhysicalFootprintFitError, match="no 8x8 candidate"
    ) as caught:
        fitting.fit_physical_footprint(flat_state, fitting.PhysicalFootprint((2, 2)))
    result = caught.value.result
    assert result.report["stage"] == "candidates"
    assert result.report["coverage"][1] > 0
    assert len(result.arrays["candidates_1"]) == 0
    assert len(result.records) == 0


def test_derivative_nonconvergence_is_preserved(flat_state, monkeypatch):
    def unresolved(*args):
        raise ReferenceDerivativeConvergenceError(1, 0.1)

    monkeypatch.setattr(fitting, "_converged_reference_normals", unresolved)
    result = fitting.check_physical_footprints(
        flat_state, [fitting.PhysicalFootprint((2, 2))]
    )[0]
    assert not result.accepted
    assert result.report["stage"] == "derivative"
    assert "failed to converge" in result.report["reason"]
    assert len(result.records) == 0


def test_storage_is_rerendered_before_acceptance(flat_state, monkeypatch):
    original = fitting._stored_records

    def wrong_conversion(*args):
        result = original(*args)
        result["alpha_t"] *= 3
        return result

    monkeypatch.setattr(fitting, "_stored_records", wrong_conversion)
    with pytest.raises(fitting.PhysicalFootprintFitError) as caught:
        fitting.fit_physical_footprint(flat_state, fitting.PhysicalFootprint((2, 2)))
    result = caught.value.result
    assert result.report["stage"] == "storage"
    assert result.report["material_fits"]["0"]["validation_error"]["passed"]
    assert not result.accepted and len(result.records) == 0


def test_invalid_and_excess_work_requests_fail_before_sampling(flat_state):
    footprint = fitting.PhysicalFootprint((2, 2))
    for batch in (0, 65, True, 1.5):
        with pytest.raises(ValueError, match="point_batch_size"):
            fitting.fit_physical_footprint(
                flat_state, footprint, point_batch_size=batch
            )
    for items in ([], [footprint] * 17):
        with pytest.raises(ValueError, match=r"1\.\.16"):
            fitting.check_physical_footprints(flat_state, items)
    with pytest.raises(ValueError):
        fitting.PhysicalFootprint((2, 2), (0, 1))
    with pytest.raises(ValueError):
        fitting.PhysicalFootprint((float("nan"), 2))
    with pytest.raises(ValueError, match="one state tile"):
        fitting.fit_physical_footprint(
            flat_state, fitting.PhysicalFootprint((2, 2), (544 * 8, 544 * 6))
        )


def test_provenance_identifies_captured_arrays_and_versions(real_state):
    spectrum = real_state.micro_spectrum.copy()
    spectrum[:, 2] += np.pi
    spectrum.setflags(write=False)
    changed = replace(real_state, micro_spectrum=spectrum)
    footprint = fitting.PhysicalFootprint((2, 2))
    original = fitting._provenance(real_state, footprint)
    other = fitting._provenance(changed, footprint)
    assert original["material_key"] == other["material_key"]
    assert original["config"] == other["config"]
    assert original["source_sha256"] == other["source_sha256"]
    assert original["state_sha256"] != other["state_sha256"]
    assert (
        original["captured_state"]["record_metadata"]["dendrites.table_version"]
        == real_state.dendrites.table_version
    )
    strided = np.arange(300, dtype=np.float64).reshape(30, 10)[:, ::2]
    assert fitting._array_identity(strided) == fitting._array_identity(strided.copy())


def test_real_wet_transition_converges_with_fifth_derivative_refinement(monkeypatch):
    # Keep the original relief recipe so this physical clamp-crossing point
    # remains a regression fixture when release appearance defaults change.
    state = build_state(
        GalvanisedConfig(
            preset="wet_storage",
            size_mm=(8, 6),
            spangle_diameter_mm=3,
            dendrite_relief_um=2.5,
            trunk_relief_um=4.5,
        ),
        seed=42,
    )
    step = material_render._reference_derivative_step_mm(state, None)
    # This actual 33-grid point is just below the weather valley clamp at
    # substrate height zero. Its first four stencils straddle the slope change.
    x = np.array([0.0625])
    y = np.array([0.8693181818181819])
    with monkeypatch.context() as old_policy:
        old_policy.setattr(material_render, "_MAX_DERIVATIVE_HALVINGS", 4)
        with pytest.raises(ReferenceDerivativeConvergenceError):
            material_render._converged_reference_normals(state, x, y, step)
    normal, diagnostic = material_render._converged_reference_normals(state, x, y, step)
    assert diagnostic["max_halvings"] == 5
    assert diagnostic["max_accepted_component_difference"] <= 1e-5
    finer = material_render._physical_point_normal(
        state, x, y, (step[0] / 64, step[1] / 64)
    )
    np.testing.assert_allclose(normal, finer, atol=1e-8, rtol=0)
    result = fitting.fit_physical_footprint(
        state, fitting.PhysicalFootprint((0.0625, 0.8125))
    )
    assert result.accepted and len(result.records) == 12
    assert result.report["accepted_spatial_rate"] == 32
    # The footprint also contains the newer fine deposit modulation; other
    # points may need more refinement than this original fifth-step example.
    refinements = max(
        item["max_derivative_halvings"] for item in result.report["integration"]
    )
    assert 5 <= refinements <= material_render._MAX_DERIVATIVE_HALVINGS
    assert all(error["passed"] for error in result.report["storage_errors"].values())
