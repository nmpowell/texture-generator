"""Resolution-independent periodic isotropic power diagrams, in millimetres.

IDs are original row indices, including hidden nuclei. Exact ties are ordered by
``(power metric, original ID, image offset X, image offset Y)``. Image offsets
refer to wrapped query coordinates and canonical seeds. The metric is evaluated
in float64, with rational comparison of ambiguous results. No raster or lighting
settings enter this module. NumPy and the standard library are sufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import numpy as np

from ._grain_geometry import (
    Cell,
    Hierarchy,
    build_cell,
    fragments,
    frozen_array,
    packed_segment_distances,
    segment_distances,
    update_winners,
)

__all__ = [
    "AreaStatistics",
    "GrainPartition",
    "OwnershipResult",
    "PlacementError",
    "QueryStats",
    "growth_weights",
    "initial_nucleus_count",
    "place_grains",
]


@dataclass(frozen=True)
class QueryStats:
    points: int
    candidate_evaluations: int
    max_candidates_per_point: int
    node_point_tests: int
    rational_comparisons: int
    peak_candidate_pairs: int

    @property
    def mean_candidates_per_point(self) -> float:
        return self.candidate_evaluations / self.points if self.points else 0.0


@dataclass(frozen=True)
class OwnershipResult:
    ids: np.ndarray
    displacement_mm: np.ndarray
    metric_mm2: np.ndarray
    image_offsets: np.ndarray
    stats: QueryStats

    @property
    def local_mm(self) -> np.ndarray:
        return self.displacement_mm


@dataclass(frozen=True)
class AreaStatistics:
    areas_mm2: np.ndarray
    area_error_mm2: np.ndarray
    active_ids: np.ndarray
    equivalent_diameters_mm: np.ndarray
    median_diameter_mm: float
    diameter_cv: float
    median_error_mm: float
    diameter_cv_error: float
    smallest_area_mm2: float
    small_cell_count: int
    hidden_count: int
    statistically_sufficient: bool
    total_area_error_mm2: float


def _positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _size(value: tuple[float, float]) -> np.ndarray:
    if len(value) != 2:
        raise ValueError("size_mm must contain two lengths")
    result = np.array([_positive(v, "size_mm") for v in value], dtype=np.float64)
    # Geometry is rational, but the public metric/bounds/areas must fit float64.
    if (
        not np.isfinite(result @ result)
        or np.any(result < np.sqrt(np.finfo(float).tiny))
        or float(result @ result) > np.finfo(float).max / 1024
    ):
        raise ValueError("size_mm is outside the supported float64 metric range")
    return result


def _chunk(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    return int(value)


@dataclass(frozen=True, init=False, eq=False)
class GrainPartition:
    """Immutable nuclei and a private lazy cache of exact periodic image cells.

    Working arrays are bounded by ``chunk_size * 16`` candidate pairs for the
    hierarchy and ``chunk_size * 64`` for the oracle. Returned arrays require
    O(number of query points) memory. Stream calls to bound output memory too.
    The default caches cells on demand; ``precompute_geometry`` is explicit.
    """

    points_mm: np.ndarray
    weights_mm2: np.ndarray
    size_mm: tuple[float, float]
    nucleus_ids: np.ndarray
    _tree: Hierarchy = field(repr=False)
    _cells: dict[int, Cell] = field(repr=False)

    def __init__(
        self, points_mm: object, weights_mm2: object, size_mm: tuple[float, float]
    ):
        size = _size(size_mm)
        points = np.asarray(points_mm, dtype=np.float64)
        weights = np.asarray(weights_mm2, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("points_mm must have nonempty shape (N, 2)")
        if len(points) > np.iinfo(np.uint32).max:
            raise ValueError("too many nuclei for uint32 IDs")
        if weights.shape != (len(points),):
            raise ValueError("weights_mm2 must have shape (N,)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
            raise ValueError("nuclei and weights must be finite")
        if np.max(np.abs(weights)) > np.finfo(float).max / 1024:
            raise ValueError("weights exceed the supported float64 metric range")
        points = frozen_array(np.remainder(points, size))
        weights = frozen_array(weights)
        object.__setattr__(self, "points_mm", points)
        object.__setattr__(self, "weights_mm2", weights)
        object.__setattr__(self, "size_mm", (float(size[0]), float(size[1])))
        object.__setattr__(
            self, "nucleus_ids", frozen_array(np.arange(len(points)), np.uint32)
        )
        object.__setattr__(
            self, "_tree", Hierarchy(points, weights, frozen_array(size))
        )
        object.__setattr__(self, "_cells", {})

    def _points(self, points: object) -> tuple[np.ndarray, tuple[int, ...]]:
        p = np.asarray(points, dtype=np.float64)
        if p.ndim < 1 or p.shape[-1] != 2:
            raise ValueError("query points must be finite with final dimension 2")
        flat = p.reshape(-1, 2)
        for start in range(0, len(flat), 65536):
            if not np.all(np.isfinite(flat[start : start + 65536])):
                raise ValueError("query points must be finite with final dimension 2")
        return flat, p.shape[:-1]

    def query(self, x: object, y: object, *, chunk_size: int = 2048) -> OwnershipResult:
        """Broadcast arbitrary X/Y coordinates and return periodic ownership."""
        xx, yy = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        return self.ownership(np.stack((xx, yy), axis=-1), chunk_size=chunk_size)

    def query_with_boundary(
        self, x: object, y: object, *, chunk_size: int = 2048
    ) -> tuple[OwnershipResult, np.ndarray]:
        """Return ownership and exact edge distance with one hierarchy query.

        The owner IDs only select complete cached cell geometry. Boundary
        constraints still use the independent whole-cell geometric search.
        """
        chunk_size = _chunk(chunk_size)
        xx, yy = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        points = np.stack((xx, yy), axis=-1)
        ownership = self.ownership(points, chunk_size=chunk_size)
        distance = self._boundary_distance(
            points, chunk_size=chunk_size, owner_ids=ownership.ids
        )
        return ownership, distance

    def ownership(self, points: object, *, chunk_size: int = 2048) -> OwnershipResult:
        return self._ownership(points, _chunk(chunk_size), oracle=False)

    def ownership_oracle(
        self, points: object, *, chunk_size: int = 1024
    ) -> OwnershipResult:
        """Exhaustive all-nucleus reference, bounded in points and seed blocks.

        For this isotropic metric only each seed's nearest periodic image can
        win. Antipodal image ties are resolved exactly before comparing seeds.
        """
        return self._ownership(points, _chunk(chunk_size), oracle=True)

    def _ownership(
        self, points: object, chunk_size: int, *, oracle: bool
    ) -> OwnershipResult:
        q, shape = self._points(points)
        n = len(q)
        ids = np.zeros(n, dtype=np.uint32)
        metric = np.full(n, np.inf)
        displacement = np.empty((n, 2), dtype=np.float64)
        images = np.zeros((n, 2), dtype=np.int8)
        total_candidates = max_candidates = 0
        node_tests = comparisons = peak = 0
        tree = self._tree
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            query = np.remainder(q[start:stop], tree.size)
            counts = np.zeros(len(query), dtype=np.int64)
            best, winners = metric[start:stop], ids[start:stop]
            offsets, local = images[start:stop], displacement[start:stop]
            if oracle:
                for first in range(0, len(self.points_mm), 64):
                    candidates = np.arange(first, min(first + 64, len(self.points_mm)))
                    comparisons += update_winners(
                        query,
                        candidates,
                        tree,
                        best,
                        winners,
                        offsets,
                        local,
                        raw_queries=q[start:stop],
                    )
                    counts += len(candidates)
                    peak = max(peak, len(query) * len(candidates))
                total_candidates += int(counts.sum())
                max_candidates = max(max_candidates, int(counts.max(initial=0)))
                continue
            # Each point has its own bounded DFS stack. Traverse them together
            # so sparse dense-population leaves do not cause a Python call per pixel.
            stack = np.empty((len(query), tree.depth), dtype=np.int32)
            tops = np.zeros(len(query), dtype=np.int32)
            current = np.zeros(len(query), dtype=np.int32)
            rows = np.arange(len(query))
            while len(rows):
                indices = current[rows]
                node_tests += len(rows)
                admitted = (
                    tree.lower_many(query[rows], indices) <= best[rows] + tree.guard
                )
                visit = rows[admitted]
                nodes = current[visit]
                inner = tree.left[nodes] >= 0
                branches, branch_nodes = visit[inner], nodes[inner]
                leaves, leaf_nodes = visit[~inner], nodes[~inner]
                current[rows] = -1
                if len(branches):
                    left, right = tree.left[branch_nodes], tree.right[branch_nodes]
                    lb = tree.lower_many(query[branches], left)
                    rb = tree.lower_many(query[branches], right)
                    left_first = lb <= rb
                    near = np.where(left_first, left, right)
                    far = np.where(left_first, right, left)
                    far_bound = np.where(left_first, rb, lb)
                    pending = far_bound <= best[branches] + tree.guard
                    push = branches[pending]
                    stack[push, tops[push]] = far[pending]
                    tops[push] += 1
                    current[branches] = near
                if len(leaves):
                    width = int(tree.leaf_count[leaf_nodes].max())
                    candidates = tree.leaf_ids[leaf_nodes, :width]
                    valid = (
                        np.arange(width)[None, :] < tree.leaf_count[leaf_nodes, None]
                    )
                    b, w, o, d = (
                        best[leaves].copy(),
                        winners[leaves].copy(),
                        offsets[leaves].copy(),
                        local[leaves].copy(),
                    )
                    comparisons += update_winners(
                        query[leaves],
                        candidates,
                        tree,
                        b,
                        w,
                        o,
                        d,
                        valid,
                        raw_queries=q[start:stop][leaves],
                    )
                    best[leaves], winners[leaves], offsets[leaves], local[leaves] = (
                        b,
                        w,
                        o,
                        d,
                    )
                    counts[leaves] += tree.leaf_count[leaf_nodes]
                    peak = max(peak, len(leaves) * width)
                pop = rows[(current[rows] < 0) & (tops[rows] > 0)]
                tops[pop] -= 1
                current[pop] = stack[pop, tops[pop]]
                rows = rows[current[rows] >= 0]
            total_candidates += int(counts.sum())
            max_candidates = max(max_candidates, int(counts.max(initial=0)))
        stats = QueryStats(
            n,
            total_candidates,
            max_candidates,
            node_tests,
            comparisons,
            peak,
        )
        return OwnershipResult(
            ids.reshape(shape),
            displacement.reshape((*shape, 2)),
            metric.reshape(shape),
            images.reshape((*shape, 2)),
            stats,
        )

    def _cell(self, grain_id: int) -> Cell:
        if (
            isinstance(grain_id, bool)
            or not isinstance(grain_id, (int, np.integer))
            or not 0 <= grain_id < len(self.points_mm)
        ):
            raise ValueError("grain_id must be an original nucleus index")
        if grain_id not in self._cells:
            self._cells[grain_id] = build_cell(grain_id, self._tree)
        return self._cells[grain_id]

    def precompute_geometry(self) -> None:
        """Explicitly build all exact cells; may be expensive for dense states."""
        for grain_id in range(len(self.points_mm)):
            self._cell(grain_id)

    def cell_fragments(self, grain_id: int) -> tuple[np.ndarray, ...]:
        """CCW polygons clipped into [0,Lx] x [0,Ly], with no area cutoff."""
        return fragments(self._cell(grain_id), grain_id, self._tree)

    def cell_segments(self, grain_id: int) -> np.ndarray:
        """Actual inter-ID segments of the canonical image cell, shape (E,2,2).

        Endpoints may lie outside the tile; translate periodically when sampling.
        Artificial self-image edges and tile edges are absent.
        """
        return self._cell(grain_id).segments

    def cell_area_exact(self, grain_id: int) -> Fraction:
        """Exact area of all fragments of one original nucleus, in mm²."""
        return self._cell(grain_id).area

    def cell_neighbours(self, grain_id: int) -> np.ndarray:
        """Original ID across each corresponding ``cell_segments`` edge."""
        return self._cell(grain_id).neighbours

    def geometry_stats(self) -> dict[str, int]:
        return {
            "cached_cells": len(self._cells),
            "candidate_images": sum(c.candidate_images for c in self._cells.values()),
            "nodes_visited": sum(c.nodes_visited for c in self._cells.values()),
            "segments": sum(len(c.segments) for c in self._cells.values()),
        }

    def boundary_distance(
        self, points: object, *, chunk_size: int = 2048
    ) -> np.ndarray:
        """Exact-segment periodic distance; infinity if no inter-ID edge exists.

        The closest boundary is on the owning grain: any path to another
        grain's boundary crosses the owner's boundary first. Its complete cell
        geometry uses whole-cell bounds, independently of ownership pruning.
        """
        return self._boundary_distance(points, chunk_size=_chunk(chunk_size))

    def _boundary_distance(
        self,
        points: object,
        *,
        chunk_size: int,
        owner_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        q, shape = self._points(points)
        known_ids = None if owner_ids is None else owner_ids.reshape(-1)
        result = np.full(len(q), np.inf)
        for start in range(0, len(q), chunk_size):
            query = np.remainder(q[start : start + chunk_size], self._tree.size)
            ids = (
                self.ownership(query, chunk_size=chunk_size).ids
                if known_ids is None
                else known_ids[start : start + len(query)]
            )
            unique, inverse = np.unique(ids, return_inverse=True)
            cells = [self._cell(int(grain_id)) for grain_id in unique]
            values = packed_segment_distances(query, cells, inverse, self._tree.size)
            for index, cell in enumerate(cells):
                if cell.area == 0:
                    # A lower-dimensional tie owner has no open cell of its own.
                    mask = inverse == index
                    values[mask] = self.boundary_distance_oracle(
                        query[mask], chunk_size=chunk_size
                    )
            result[start : start + len(query)] = values
        return result.reshape(shape)

    def boundary_distance_oracle(
        self, points: object, *, chunk_size: int = 1024
    ) -> np.ndarray:
        """All-cell/all-constraint segment reference, intentionally slow."""
        chunk_size = _chunk(chunk_size)
        q, shape = self._points(points)
        result = np.full(len(q), np.inf)
        for grain_id in range(len(self.points_mm)):
            cell = build_cell(grain_id, self._tree, exhaustive=True)
            for start in range(0, len(q), chunk_size):
                stop = min(start + chunk_size, len(q))
                result[start:stop] = np.minimum(
                    result[start:stop],
                    segment_distances(
                        np.remainder(q[start:stop], self._tree.size),
                        cell.segments,
                        self._tree.size,
                    ),
                )
        return result.reshape(shape)

    def adjacency(self) -> np.ndarray:
        """Unique sorted inter-ID pairs, O(edges) storage; never a dense matrix."""
        self.precompute_geometry()
        pairs = {
            tuple(sorted((i, int(j))))
            for i, c in self._cells.items()
            for j in c.neighbours
            if i != j
        }
        return frozen_array(sorted(pairs), np.uint32).reshape(-1, 2)

    def areas(self) -> np.ndarray:
        """Geometric areas by original ID, hidden nuclei retained as zeros."""
        values = []
        for i in range(len(self.points_mm)):
            area = self._cell(i).area
            value = float(area)
            if area > 0 and value == 0:
                raise FloatingPointError(
                    "positive cell area underflows float64; use cell_area_exact"
                )
            values.append(value)
        return frozen_array(values)

    def area_statistics(self, *, small_area_fraction: float = 1e-6) -> AreaStatistics:
        """Unweighted statistics over every geometrically positive cell.

        Errors enclose numerical rounding, not uncertainty in a fitted material
        model. Fewer than 12 cells is labelled statistically insufficient.
        ``small_area_fraction`` is diagnostic only and never excludes a cell.
        """
        if not np.isfinite(small_area_fraction) or small_area_fraction < 0:
            raise ValueError("small_area_fraction must be finite and nonnegative")
        area = self.areas()
        active = np.flatnonzero(area > 0)
        # Exact rational areas rounded once: one outward ULP is conservative.
        errors = np.where(area > 0, np.spacing(area), 0.0)
        a = area[active]
        # Take the root first so a positive subnormal area cannot disappear
        # during division by pi. Its exact rational value remains available.
        diameter = (2 / math.sqrt(math.pi)) * np.sqrt(a)
        derr = 8 * np.finfo(float).eps * diameter + 2 * errors[active] / (
            np.pi * diameter
        )
        median = float(np.median(diameter))
        mean = float(np.mean(diameter))
        cv = float(np.std(diameter / mean))
        # L2 Lipschitz bounds for population std and mean, then ratio propagation.
        mean_error = float(np.mean(derr))
        largest_error = float(np.max(derr))
        std_error = largest_error * float(np.sqrt(np.mean((derr / largest_error) ** 2)))
        cv_error = (std_error + cv * mean_error) / (mean - mean_error) + 16 * np.finfo(
            float
        ).eps
        total = math.prod(self.size_mm)
        return AreaStatistics(
            area,
            frozen_array(errors),
            frozen_array(active, np.uint32),
            frozen_array(diameter),
            median,
            cv,
            float(np.max(derr)),
            cv_error,
            float(a.min()),
            int(np.count_nonzero(a < total * small_area_fraction)),
            len(area) - len(a),
            len(a) >= 12,
            float(errors.sum()) + 8 * np.finfo(float).eps * total,
        )


class PlacementError(ValueError):
    """A bounded placement exhausted its proposal budget before the requested count."""

    def __init__(self, requested: int, achieved: int):
        self.requested = requested
        self.achieved = achieved
        super().__init__(
            f"Poisson-disc placement achieved {achieved} of {requested} nuclei; reduce minimum_distance_mm/count or increase candidates_per_point"
        )


def initial_nucleus_count(size_mm: tuple[float, float], diameter_mm: float) -> int:
    size = _size(size_mm)
    diameter = _positive(diameter_mm, "diameter_mm")
    count = (4 / math.pi) * (float(size[0]) / diameter) * (float(size[1]) / diameter)
    if not np.isfinite(count) or count > np.iinfo(np.uint32).max:
        raise ValueError("estimated population exceeds uint32 capacity")
    return max(1, round(count))


def _rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator) or not isinstance(
        rng.bit_generator, np.random.PCG64
    ):
        raise TypeError(
            "rng must be an explicit np.random.Generator(np.random.PCG64(seed))"
        )


def place_grains(
    size_mm: tuple[float, float],
    diameter_mm: float,
    rng: np.random.Generator,
    *,
    count: int | None = None,
    mode: Literal["poisson_disc", "uniform"] = "poisson_disc",
    minimum_distance_mm: float | None = None,
    candidates_per_point: int = 30,
) -> np.ndarray:
    """Bounded toroidal Poisson-disc placement by uniform sequential inhibition.

    Defaults use round(4A/piD²) seeds and separation 0.35D, both authoring
    choices awaiting geometric calibration. At most candidates_per_point*N
    proposals are tried, in draw order. There is no active-list growth front:
    proposals cover the whole tile even when stopping at a fixed population.
    Exhaustion raises PlacementError carrying the achieved population.
    """
    size = _size(size_mm)
    diameter = _positive(diameter_mm, "diameter_mm")
    _rng(rng)
    n = initial_nucleus_count(size_mm, diameter) if count is None else count
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or not 1 <= n <= np.iinfo(np.uint32).max
    ):
        raise ValueError("count must be a positive uint32-compatible integer")
    if mode not in ("poisson_disc", "uniform"):
        raise ValueError("mode must be 'poisson_disc' or 'uniform'")
    if mode == "uniform":
        return frozen_array(rng.random((n, 2)) * size)
    k = _chunk(candidates_per_point)
    radius = _positive(
        0.35 * diameter if minimum_distance_mm is None else minimum_distance_mm,
        "minimum_distance_mm",
    )
    # Sparse bins avoid allocating an empty grid. Capping its logical size
    # only makes bins wider; the 3x3 neighbour stencil remains conservative.
    with np.errstate(over="ignore"):
        ratio = np.minimum(size / radius, 2 * n)
    shape = np.maximum(1, np.floor(np.nextafter(ratio, -np.inf)).astype(np.int64))
    widths = size / shape
    bins: dict[tuple[int, int], list[int]] = {}
    points: list[np.ndarray] = []

    def key(p: np.ndarray) -> tuple[int, int]:
        ij = np.minimum(np.floor(p / widths).astype(np.int64), shape - 1)
        return int(ij[0]), int(ij[1])

    for _ in range(k * n):
        p = rng.random(2) * size
        ix, iy = key(p)
        neighbours = {
            (int((ix + dx) % shape[0]), int((iy + dy) % shape[1]))
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        }
        valid = True
        for neighbour in sorted(neighbours):
            for index in bins.get(neighbour, ()):
                delta = np.abs(p - points[index])
                delta = np.minimum(delta, size - delta)
                squared = float(delta @ delta)
                limit = radius * radius
                if abs(squared - limit) <= 32 * np.finfo(float).eps * (squared + limit):
                    exact = []
                    for axis in (0, 1):
                        d = abs(
                            Fraction(float(p[axis]))
                            - Fraction(float(points[index][axis]))
                        )
                        exact.append(min(d, Fraction(float(size[axis])) - d))
                    too_close = sum(d * d for d in exact) < Fraction(radius) ** 2
                else:
                    too_close = squared < limit
                if too_close:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            index = len(points)
            points.append(p)
            bins.setdefault((ix, iy), []).append(index)
            if len(points) == n:
                break
    if len(points) != n:
        raise PlacementError(n, len(points))
    return frozen_array(points)


def growth_weights(
    count: int,
    diameter_mm: float,
    rng: np.random.Generator,
    *,
    growth_sigma: float = 0.35,
    max_weight_fraction: float = 0.2,
) -> np.ndarray:
    """Bound positive lognormal growth controls into zero-mean mm² weights.

    ``growth_sigma`` is log-growth spread, **not** realised cell-diameter CV.
    The final absolute bound is max_weight_fraction * diameter_mm². Subtracting
    a common gauge changes no cell. Preset calibration belongs to the caller.
    """
    _rng(rng)
    diameter = _positive(diameter_mm, "diameter_mm")
    _chunk(count)
    if (
        isinstance(growth_sigma, (bool, np.bool_))
        or not np.isfinite(growth_sigma)
        or not 0 <= growth_sigma <= 10
    ):
        raise ValueError("growth_sigma must be finite and in [0, 10]")
    if (
        isinstance(max_weight_fraction, (bool, np.bool_))
        or not np.isfinite(max_weight_fraction)
        or max_weight_fraction < 0
    ):
        raise ValueError("max_weight_fraction must be finite and nonnegative")
    growth = rng.lognormal(0.0, growth_sigma, count)
    # (g-1)/(g+1) is a bounded transform of a strictly positive growth variable.
    raw = (growth - 1) / (growth + 1)
    raw -= np.mean(raw)
    bound = max_weight_fraction * diameter * diameter
    if not np.isfinite(bound):
        raise ValueError("weight bound overflows float64")
    raw *= bound / max(1.0, float(np.max(np.abs(raw))))
    return frozen_array(raw)
