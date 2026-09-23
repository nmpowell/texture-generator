"""Response-space selection and rejection fixtures for the offline lobe fitter."""

import numpy as np
import pytest

from texture_generators.core.lobe_fitting import (
    MAX_REFINEMENT_SWEEPS,
    FitTolerances,
    LobeFitError,
    fit_material_lobes,
)
from texture_generators.core.material_render import _bsdf, _Optics

STRICT = FitTolerances(relative_rms=1e-9, dark_absolute_rms=1e-10)


def _fit(training, target, validation=None, valid_target=None, **kwargs):
    return fit_material_lobes(
        training,
        target,
        training if validation is None else validation,
        target if valid_target is None else valid_target,
        material_id=kwargs.pop("material_id", 0),
        coverage=kwargs.pop("coverage", 1.0),
        tolerances=kwargs.pop("tolerances", STRICT),
        **kwargs,
    )


def _predicted(result, responses):
    return np.asarray(result.weights) @ responses[list(result.candidate_indices)]


def _directions(*, held_out=False):
    count = 19 if held_out else 16
    angles = 2 * np.pi * (np.arange(count) + (0.37 if held_out else 0)) / count
    polar = [0.07, 0.18, 0.40, 0.80] if held_out else [0.04, 0.12, 0.25, 0.50]
    theta, phi = np.meshgrid(polar, angles, indexing="ij")
    return np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
    ).reshape(-1, 3)


def _physical_responses(material_id, axes, slopes, *, held_out=False):
    light = _directions(held_out=held_out)
    optical = _Optics(
        material_id,
        "conductor" if material_id == 0 else "dielectric",
        np.asarray([0.0, 0.0, 0.0] if material_id == 0 else [0.32, 0.35, 0.31]),
        0.3 if material_id == 0 else 0.64,
        0.5 if material_id == 0 else 0.0,
        1.5,
    )
    at, ab = (0.16, 0.045) if material_id == 0 else (0.4096, 0.4096)
    output = []
    for angle, slope in zip(axes, slopes, strict=True):
        normal = np.asarray([-slope, 0.0, 1.0])
        normal /= np.linalg.norm(normal)
        tangent = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        tangent -= np.dot(tangent, normal) * normal
        tangent /= np.linalg.norm(tangent)
        value = _bsdf(
            normal, tangent, light, np.asarray([0.0, 0.0, 1.0]), at, ab, optical
        )
        value *= np.maximum(light @ normal, 0.0)[:, None]
        output.append(value.ravel())
    return np.asarray(output)


def test_exact_candidate_is_retained_without_parameter_averaging():
    training = np.array([[1.0, 2.0, 0.1], [0.2, 0.3, 4.0], [0.7, 0.9, 1.2]])
    validation = np.array([[0.8, 2.0], [1.1, 0.3], [0.2, 1.3]])
    result = _fit(
        training, 0.37 * training[1], validation, 0.37 * validation[1], coverage=0.37
    )
    assert result.candidate_indices == (1,)
    assert result.weights == (0.37,)
    assert result.refinement_sweeps == 0
    assert result.search_termination == "numerical_zero"
    assert result.training_error.passed and result.validation_error.passed


@pytest.mark.parametrize("seed", range(6))
def test_exact_source_does_not_grow_extra_lobes_from_roundoff(seed):
    rng = np.random.default_rng(seed)
    train = rng.uniform(0.001, 7.0, (8, 31))
    valid = rng.uniform(0.001, 7.0, (8, 47))
    coverage = float(rng.uniform(0.01, 0.99))
    result = _fit(
        train, coverage * train[3], valid, coverage * valid[3], coverage=coverage
    )
    assert result.candidate_indices == (3,)
    assert result.weights == (coverage,)


def test_nonnegative_weights_close_exact_material_coverage():
    training = np.eye(4)
    validation = np.array([[1.0, 0.3], [0.2, 0.9], [0.1, 0.5], [0.6, 0.4]])
    weights = np.array([0.05, 0.10, 0.07, 0.15])
    result = _fit(
        training, weights @ training, validation, weights @ validation, coverage=0.37
    )
    assert result.candidate_indices == (0, 1, 2, 3)
    assert min(result.weights) > 0.0
    assert abs(sum(result.weights) - 0.37) < 1e-15
    np.testing.assert_allclose(result.weights, weights, atol=1e-14)


def test_identical_candidates_have_a_stable_original_ordinal_tie():
    training = np.ones((8, 7))
    validation = np.full((8, 11), 0.4)
    first = _fit(training, training[0], validation, validation[0])
    second = _fit(training, training[0], validation, validation[0])
    assert first == second
    assert first.candidate_indices == (0,)
    assert first.weights == (1.0,)


def test_held_out_failure_cannot_change_training_selection():
    training = np.ones((2, 4))
    validation = np.array([[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(LobeFitError, match="held-out") as caught:
        _fit(training, training[0], validation, validation[1])
    result = caught.value.result
    assert result.candidate_indices == (0,)
    assert result.training_error.passed
    assert not result.validation_error.passed


def test_five_separated_response_peaks_have_an_explicit_failure():
    training = np.eye(5)
    validation = np.eye(5)[:, ::-1]
    target = np.full(5, 0.2)
    with pytest.raises(LobeFitError, match="training") as caught:
        _fit(training, target, validation, target)
    result = caught.value.result
    assert len(result.weights) == 4
    assert abs(sum(result.weights) - 1.0) < 1e-15
    assert result.training_error.maximum_relative_error >= 0.99


def test_dark_absolute_error_and_sample_weights_are_enforced():
    training = np.array([[0.001, 0.002], [0.001, 0.002]])
    result = _fit(training, training[0], coverage=1.0)
    assert result.training_error.relative_rms == 0.0
    with pytest.raises(LobeFitError):
        _fit(training, np.array([0.001, 0.003]), max_lobes=1)
    weighted = _fit(
        training,
        np.array([0.001, 0.003]),
        training_sample_weights=[1, 0],
        validation_sample_weights=[1, 0],
    )
    assert weighted.training_error.passed


def test_orthogonal_actual_ggx_axes_survive_candidate_selection():
    axes = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    train = _physical_responses(0, axes, [0.0] * 4)
    valid = _physical_responses(0, axes, [0.0] * 4, held_out=True)
    weight = np.array([0.5, 0.0, 0.5, 0.0])
    result = _fit(train, weight @ train, valid, weight @ valid)
    assert result.candidate_indices == (0, 2)
    np.testing.assert_allclose(result.weights, (0.5, 0.5), atol=1e-12)
    np.testing.assert_allclose(_predicted(result, valid), weight @ valid, rtol=1e-10)


def test_exact_four_source_64_candidate_ggx_mixture_is_reconstructed():
    axes = np.tile(np.arange(8) * np.pi / 8, 8)
    slopes = np.repeat(np.linspace(-0.18, 0.18, 8), 8)
    train = _physical_responses(0, axes, slopes)
    valid = _physical_responses(0, axes, slopes, held_out=True)
    weight = np.zeros(64)
    weight[[0, 17, 42, 63]] = [0.15, 0.25, 0.35, 0.25]
    result = _fit(train, weight @ train, valid, weight @ valid)
    assert result.candidate_indices == (0, 17, 42, 63)
    assert result.refinement_sweeps == 2
    assert result.search_termination == "numerical_zero"
    assert result.trial_count == 730
    np.testing.assert_allclose(result.weights, weight[weight > 0], atol=1e-12)
    np.testing.assert_allclose(_predicted(result, valid), weight @ valid, rtol=1e-9)


def _local_minimum_responses():
    # A tetrahedron in the first three dimensions, offset in the fourth, is
    # closer than any single-replacement edge/face to the target. Replacing
    # two decoys at once would expose the exact last-two-candidate solution.
    training = 2.0 + np.array(
        [
            [1.0, 1.0, 1.0, 0.3],
            [1.0, -1.0, -1.0, 0.3],
            [-1.0, 1.0, -1.0, 0.3],
            [-1.0, -1.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, -2.0],
            [0.0, 0.0, 0.0, 2.0],
        ]
    )
    validation = training[:, [3, 1, 0, 2]]
    return training, validation, np.full(4, 2.0)


def test_single_swap_local_minimum_rejects_despite_exact_two_source_solution():
    training, validation, target = _local_minimum_responses()
    np.testing.assert_array_equal(0.5 * (training[4] + training[5]), target)
    with pytest.raises(LobeFitError, match="training") as caught:
        _fit(training, target, validation, target, tolerances=FitTolerances())
    result = caught.value.result
    assert result.candidate_indices == (0, 1, 2, 3)
    assert result.search_termination == "local_optimum"
    assert result.refinement_sweeps == 1
    assert result.training_error.relative_rms == pytest.approx(0.075)
    assert result.validation_error.relative_rms == pytest.approx(0.075)
    assert result.trial_count <= (1 + MAX_REFINEMENT_SWEEPS) * 4 * len(training)


def _physical_local_minimum_weights():
    weights = np.zeros(64)
    weights[[2, 9, 12, 30]] = [
        0.31507729041159366,
        0.3215818310722124,
        0.15036227833922555,
        0.21297860017696849,
    ]
    return weights


def test_actual_ggx_single_swap_local_minimum_is_explicitly_rejected():
    axes = np.tile(np.arange(8) * np.pi / 8, 8)
    slopes = np.repeat(np.linspace(-0.18, 0.18, 8), 8)
    train = _physical_responses(0, axes, slopes)
    valid = _physical_responses(0, axes, slopes, held_out=True)
    weight = _physical_local_minimum_weights()
    with pytest.raises(LobeFitError, match="training") as caught:
        _fit(train, weight @ train, valid, weight @ valid, tolerances=FitTolerances())
    result = caught.value.result
    assert result.candidate_indices == (1, 10, 12, 30)
    assert result.search_termination == "local_optimum"
    assert result.refinement_sweeps == 1
    assert result.training_error.relative_rms > 0.16
    assert result.validation_error.relative_rms > 0.22


def test_swap_refinement_does_not_use_held_out_data():
    axes = np.tile(np.arange(8) * np.pi / 8, 8)
    slopes = np.repeat(np.linspace(-0.18, 0.18, 8), 8)
    train = _physical_responses(0, axes, slopes)
    valid = _physical_responses(0, axes, slopes, held_out=True)
    weight = np.zeros(64)
    weight[[0, 17, 42, 63]] = [0.15, 0.25, 0.35, 0.25]
    accepted = _fit(train, weight @ train, valid, weight @ valid)
    with pytest.raises(LobeFitError, match="held-out") as caught:
        _fit(train, weight @ train, valid, valid[1])
    rejected = caught.value.result
    assert rejected.candidate_indices == accepted.candidate_indices
    assert rejected.weights == accepted.weights
    assert rejected.trial_count == accepted.trial_count
    assert rejected.refinement_sweeps == accepted.refinement_sweeps
    assert rejected.search_termination == accepted.search_termination


def test_refinement_budget_exhaustion_is_reported(monkeypatch):
    monkeypatch.setattr("texture_generators.core.lobe_fitting.MAX_REFINEMENT_SWEEPS", 1)
    axes = np.tile(np.arange(8) * np.pi / 8, 8)
    slopes = np.repeat(np.linspace(-0.18, 0.18, 8), 8)
    train = _physical_responses(0, axes, slopes)
    valid = _physical_responses(0, axes, slopes, held_out=True)
    weight = np.zeros(64)
    weight[[0, 17, 42, 63]] = [0.15, 0.25, 0.35, 0.25]
    with pytest.raises(LobeFitError, match="training") as caught:
        _fit(train, weight @ train, valid, weight @ valid)
    result = caught.value.result
    assert result.candidate_indices == (9, 17, 42, 63)
    assert result.search_termination == "sweep_limit"
    assert result.refinement_sweeps == 1
    assert result.trial_count == 490


def test_opposite_material_slope_associations_remain_distinct():
    training = [
        _physical_responses(material, [0.0, 0.0], [-0.15, 0.15]) for material in (0, 1)
    ]
    validation = [
        _physical_responses(material, [0.0, 0.0], [-0.15, 0.15], held_out=True)
        for material in (0, 1)
    ]
    totals = []
    for swap in (False, True):
        total = np.zeros(validation[0].shape[1])
        for material in (0, 1):
            source = material if not swap else 1 - material
            fit = _fit(
                training[material],
                0.5 * training[material][source],
                validation[material],
                0.5 * validation[material][source],
                material_id=material,
                coverage=0.5,
            )
            assert fit.material_id == material
            assert fit.candidate_indices == (source,)
            total += _predicted(fit, validation[material])
        totals.append(total)
    assert np.max(np.abs(totals[0] - totals[1])) > 0.1


def test_zero_coverage_and_work_limits():
    result = _fit(np.ones((2, 3)), np.zeros(3), coverage=0.0)
    assert result.candidate_indices == result.weights == ()
    assert result.search_termination == "zero_coverage"
    for shape in ((65, 3), (4, 4097)):
        with pytest.raises(ValueError, match="must have shape"):
            _fit(np.ones(shape), np.ones(shape[1]))
    with pytest.raises(ValueError, match="max_lobes"):
        _fit(np.ones((2, 3)), np.ones(3), max_lobes=5)
    with pytest.raises(ValueError, match="nonnegative"):
        _fit(np.array([[1.0, -1.0]]), np.ones(2))
    with pytest.raises(ValueError, match="positive total"):
        _fit(np.ones((2, 3)), np.ones(3), training_sample_weights=[0, 0, 0])
