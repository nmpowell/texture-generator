"""Bounded offline fitting of one material's sampled angular response.

The fitter selects actual candidate lobes; it never averages their frames,
widths or optical parameters. Response generation remains the caller's job.
This module is deliberately not connected to production material sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real
from typing import Any

import numpy as np

FIT_VERSION = "candidate-simplex-swap-2"
MAX_CANDIDATES = 64
MAX_RESPONSE_SAMPLES = 4096
MAX_REFINEMENT_SWEEPS = 8
_NUMERICAL_ZERO_LOSS = 32.0 * np.finfo(float).eps ** 2


@dataclass(frozen=True)
class FitTolerances:
    """Acceptance on each independent set of scalar radiance samples."""

    relative_rms: float = 0.05
    dark_absolute_rms: float = 0.001
    radiance_floor: float = 0.02

    def __post_init__(self) -> None:
        for name in ("relative_rms", "dark_absolute_rms", "radiance_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class ResponseError:
    """Weighted diagnostics; dark samples never use unstable relative error."""

    relative_rms: float
    dark_absolute_rms: float
    absolute_rms: float
    maximum_relative_error: float
    passed: bool

    def to_mapping(self) -> dict[str, float | bool]:
        return {
            "relative_rms": self.relative_rms,
            "dark_absolute_rms": self.dark_absolute_rms,
            "absolute_rms": self.absolute_rms,
            "maximum_relative_error": self.maximum_relative_error,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class LobeFit:
    """Selected source ordinals and absolute material-coverage weights."""

    material_id: int
    candidate_indices: tuple[int, ...]
    weights: tuple[float, ...]
    coverage: float
    training_error: ResponseError
    validation_error: ResponseError
    candidate_count: int
    trial_count: int
    tolerances: FitTolerances
    refinement_sweeps: int
    search_termination: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "fit_version": FIT_VERSION,
            "material_id": self.material_id,
            "candidate_indices": list(self.candidate_indices),
            "weights": list(self.weights),
            "coverage": self.coverage,
            "candidate_count": self.candidate_count,
            "trial_count": self.trial_count,
            "refinement_sweeps": self.refinement_sweeps,
            "max_refinement_sweeps": MAX_REFINEMENT_SWEEPS,
            "search_termination": self.search_termination,
            "tolerances": {
                "relative_rms": self.tolerances.relative_rms,
                "dark_absolute_rms": self.tolerances.dark_absolute_rms,
                "radiance_floor": self.tolerances.radiance_floor,
            },
            "training_error": self.training_error.to_mapping(),
            "validation_error": self.validation_error.to_mapping(),
        }


class LobeFitError(ValueError):
    """A bounded fit failed its contract; ``result`` retains its diagnostics."""

    def __init__(self, result: LobeFit) -> None:
        self.result = result
        failed = "training" if not result.training_error.passed else "held-out"
        super().__init__(
            f"material {result.material_id} exceeds {failed} response tolerance "
            f"with {len(result.weights)} candidate lobes; "
            "use the reference response or an explicitly approximate quality tier"
        )


@dataclass(frozen=True)
class _Solution:
    indices: tuple[int, ...]
    weights: np.ndarray
    loss: float


def _responses(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or not 1 <= array.shape[0] <= MAX_CANDIDATES
        or not 1 <= array.shape[1] <= MAX_RESPONSE_SAMPLES
    ):
        raise ValueError(
            f"{name} must have shape (1..{MAX_CANDIDATES}, 1..{MAX_RESPONSE_SAMPLES})"
        )
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative radiance")
    return result


def _vector(value: object, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real vector of length {length}")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _sample_weights(value: object | None, length: int, name: str) -> np.ndarray:
    weights = np.ones(length) if value is None else _vector(value, length, name)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have a finite positive total")
    return weights / total


def _better(trial: _Solution, incumbent: _Solution | None) -> bool:
    if incumbent is None:
        return True
    epsilon = np.finfo(float).eps
    # Normalized target norm is at most one. Do not introduce microscopic
    # extra lobes merely to improve the last rounding bits of an exact match.
    tolerance = _NUMERICAL_ZERO_LOSS + 8.0 * epsilon * max(trial.loss, incumbent.loss)
    if trial.loss < incumbent.loss - tolerance:
        return True
    if abs(trial.loss - incumbent.loss) <= tolerance:
        return (len(trial.indices), trial.indices) < (
            len(incumbent.indices),
            incumbent.indices,
        )
    return False


def _simplex_fit(
    responses: np.ndarray,
    target: np.ndarray,
    candidates: tuple[int, ...],
    coverage: float,
) -> _Solution:
    """Solve all nonempty active faces of a simplex of at most four weights."""
    best = None
    for count in range(1, len(candidates) + 1):
        for face in combinations(candidates, count):
            basis = responses[list(face)]
            if count == 1:
                weights = np.array([coverage])
            else:
                # Eliminate the final weight to impose sum(weights)=coverage.
                differences = (basis[:-1] - basis[-1]).T
                right = target - coverage * basis[-1]
                free = np.linalg.lstsq(differences, right, rcond=None)[0]
                weights = np.append(free, coverage - np.sum(free))
                if np.any(weights <= 0.0):
                    # Boundary solutions have their own lower-dimensional face.
                    continue
            residual = weights @ basis - target
            loss = float(residual @ residual)
            if not np.isfinite(loss):
                raise ValueError("response dynamic range exceeds float64 fitting")
            solution = _Solution(face, weights, loss)
            if _better(solution, best):
                best = solution
    assert best is not None
    return best


def _refine_support(
    responses: np.ndarray,
    target: np.ndarray,
    solution: _Solution,
    coverage: float,
    max_lobes: int,
) -> tuple[_Solution, int, int, str]:
    """Best-improving single replacements, with a fixed per-footprint bound.

    A face solve can drop candidates, so also consider one addition whenever
    the support has room. No held-out response or acceptance tolerance enters
    the search. The numerical-zero stop is the same roundoff floor as ties.
    """
    trials = 0
    for sweep in range(MAX_REFINEMENT_SWEEPS):
        if solution.loss <= _NUMERICAL_ZERO_LOSS:
            return solution, trials, sweep, "numerical_zero"
        supports: set[tuple[int, ...]] = set()
        for candidate in range(len(responses)):
            if candidate in solution.indices:
                continue
            if len(solution.indices) < max_lobes:
                supports.add(tuple(sorted((*solution.indices, candidate))))
            for removed in solution.indices:
                retained = (index for index in solution.indices if index != removed)
                supports.add(tuple(sorted((*retained, candidate))))
        best = solution
        for support in sorted(supports):
            trial = _simplex_fit(responses, target, support, coverage)
            trials += 1
            if _better(trial, best):
                best = trial
        if not _better(best, solution):
            return solution, trials, sweep + 1, "local_optimum"
        solution = best
    termination = (
        "numerical_zero" if solution.loss <= _NUMERICAL_ZERO_LOSS else "sweep_limit"
    )
    return solution, trials, MAX_REFINEMENT_SWEEPS, termination


def _error(
    actual: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    tolerances: FitTolerances,
) -> ResponseError:
    difference = actual - target
    lit = (target >= tolerances.radiance_floor) & (sample_weights > 0.0)
    dark = (target < tolerances.radiance_floor) & (sample_weights > 0.0)

    def rms(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        weights = sample_weights[mask]
        return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))

    relative = difference[lit] / target[lit]
    relative_rms = rms(relative, lit)
    dark_rms = rms(difference[dark], dark)
    absolute_rms = float(np.sqrt(np.sum(sample_weights * difference**2)))
    if not np.all(np.isfinite([relative_rms, dark_rms, absolute_rms])):
        raise ValueError("response dynamic range exceeds float64 error measurement")
    return ResponseError(
        relative_rms,
        dark_rms,
        absolute_rms,
        float(np.max(np.abs(relative), initial=0.0)),
        relative_rms <= tolerances.relative_rms
        and dark_rms <= tolerances.dark_absolute_rms,
    )


def fit_material_lobes(
    training_responses: object,
    training_target: object,
    validation_responses: object,
    validation_target: object,
    *,
    material_id: int,
    coverage: float,
    max_lobes: int = 4,
    tolerances: FitTolerances | None = None,
    training_sample_weights: object | None = None,
    validation_sample_weights: object | None = None,
) -> LobeFit:
    """Select ≤4 actual lobes of one material and validate unseen responses.

    Response matrices have shape ``(candidates, scalar angular samples)``.
    RGB may be flattened with matching sample weights. Candidate rows must
    identify the same physical lobes in both matrices. Targets contain the
    absolute footprint response of this material; candidate rows are responses
    at unit weight. The caller supplies disjoint training/validation angles.

    Training-only greedy selection is followed by at most eight best-improving
    single-swap sweeps. Each support uses exact active-face enumeration for its
    nonnegative, coverage-constrained weights. This is a bounded local search,
    not a global sparse optimum. Failed training or held-out acceptance raises
    LobeFitError; the result records whether refinement reached a local optimum,
    numerical zero, or its sweep limit.
    """
    if (
        isinstance(material_id, bool)
        or not isinstance(material_id, Integral)
        or not 0 <= material_id <= 255
    ):
        raise ValueError("material_id must be an unsigned byte integer")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, Real)
        or not np.isfinite(coverage)
        or not 0.0 <= coverage <= 1.0
    ):
        raise ValueError("coverage must be finite and lie in [0,1]")
    if (
        isinstance(max_lobes, bool)
        or not isinstance(max_lobes, Integral)
        or not 1 <= max_lobes <= 4
    ):
        raise ValueError("max_lobes must be an integer in [1,4]")
    limits = FitTolerances() if tolerances is None else tolerances
    if not isinstance(limits, FitTolerances):
        raise TypeError("tolerances must be FitTolerances")
    training = _responses(training_responses, "training_responses")
    validation = _responses(validation_responses, "validation_responses")
    if len(training) != len(validation):
        raise ValueError("training and validation must use the same candidate rows")
    train_target = _vector(training_target, training.shape[1], "training_target")
    valid_target = _vector(validation_target, validation.shape[1], "validation_target")
    train_weights = _sample_weights(
        training_sample_weights, training.shape[1], "training_sample_weights"
    )
    valid_weights = _sample_weights(
        validation_sample_weights, validation.shape[1], "validation_sample_weights"
    )
    scale = np.sqrt(train_weights) / np.maximum(train_target, limits.radiance_floor)
    scaled = training * scale
    target = train_target * scale
    if not np.all(np.isfinite(scaled)) or not np.all(np.isfinite(target)):
        raise ValueError("response dynamic range exceeds float64 fitting")
    solution = _Solution((), np.empty(0), float(target @ target))
    trials = 0
    refinement_sweeps = 0
    termination = "zero_coverage"
    if coverage > 0.0:
        for _ in range(min(int(max_lobes), len(training))):
            best = None
            for candidate in range(len(training)):
                if candidate in solution.indices:
                    continue
                chosen = tuple(sorted((*solution.indices, candidate)))
                trial = _simplex_fit(scaled, target, chosen, float(coverage))
                trials += 1
                if _better(trial, best):
                    best = trial
            if best is None or (solution.indices and not _better(best, solution)):
                break
            solution = best
        solution, refinement_trials, refinement_sweeps, termination = _refine_support(
            scaled, target, solution, float(coverage), int(max_lobes)
        )
        trials += refinement_trials
    train_actual = solution.weights @ training[list(solution.indices)]
    valid_actual = solution.weights @ validation[list(solution.indices)]
    result = LobeFit(
        int(material_id),
        solution.indices,
        tuple(float(weight) for weight in solution.weights),
        float(coverage),
        _error(train_actual, train_target, train_weights, limits),
        _error(valid_actual, valid_target, valid_weights, limits),
        len(training),
        trials,
        limits,
        refinement_sweeps,
        termination,
    )
    if not result.training_error.passed or not result.validation_error.passed:
        raise LobeFitError(result)
    return result
