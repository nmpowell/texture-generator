"""Independent analytic and adversarial fixtures for periodic power geometry."""

import math
from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from texture_generators.core.grains import (
    GrainPartition,
    PlacementError,
    growth_weights,
    initial_nucleus_count,
    place_grains,
)


def rng(seed=42):
    return np.random.Generator(np.random.PCG64(seed))


@pytest.mark.parametrize(
    "weights", ([0.0, 0.0, 0.0], [0.4, -0.2, 0.1], [0.4999, 0.0, -4.0])
)
def test_combined_ownership_boundary_preserves_independent_queries(weights):
    partition = GrainPartition(
        np.array([[0.0, 0.0], [0.5, 0.5], [0.1, 0.7]]), np.asarray(weights), (1.0, 1.0)
    )
    points = np.vstack(
        (rng(411).uniform(-2, 3, (500, 2)), [[0, 0], [0.25, 0.25], [1, 1]])
    )
    ownership, distance = partition.query_with_boundary(
        points[:, 0], points[:, 1], chunk_size=17
    )
    separate = partition.ownership(points, chunk_size=113)
    np.testing.assert_array_equal(ownership.ids, separate.ids)
    np.testing.assert_array_equal(ownership.metric_mm2, separate.metric_mm2)
    np.testing.assert_array_equal(
        distance, partition.boundary_distance(points, chunk_size=17)
    )
    np.testing.assert_allclose(
        distance, partition.boundary_distance_oracle(points), atol=1e-12
    )


def exact_nine_image_reference(partition, point):
    """Independent scalar rational reference: explicitly inspect all nine images."""
    query = np.remainder(point, partition.size_mm)
    options = []
    for i, (seed, weight) in enumerate(
        zip(partition.points_mm, partition.weights_mm2, strict=True)
    ):
        for offset in product((-1, 0, 1), repeat=2):
            d = [
                Fraction(float(query[k]))
                - Fraction(float(seed[k]))
                - offset[k] * Fraction(partition.size_mm[k])
                for k in (0, 1)
            ]
            metric = sum(v * v for v in d) - Fraction(float(weight))
            options.append((metric, i, offset))
    return min(options)


def polygon_area(vertices):
    # Independent fan area avoids large translated shoelace cancellation.
    v = vertices - vertices[0]
    return abs(np.sum(v[:-1, 0] * v[1:, 1] - v[1:, 0] * v[:-1, 1])) / 2


def test_immutable_original_ids_and_validation():
    points = np.array([[1.25, -0.5], [0.75, 0.5]])
    weights = np.array([0.25, 0.0])
    g = GrainPartition(points, weights, (1, 2))
    points[:] = 0
    weights[:] = 9
    np.testing.assert_array_equal(g.points_mm, [[0.25, 1.5], [0.75, 0.5]])
    np.testing.assert_array_equal(g.nucleus_ids, np.array([0, 1], dtype=np.uint32))
    assert g.points_mm.dtype == g.weights_mm2.dtype == np.float64
    for array in (g.points_mm, g.weights_mm2, g.nucleus_ids):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        g.size_mm = (2, 2)
    for p, w, size in [
        ([], [], (1, 1)),
        ([[0, 0]], [], (1, 1)),
        ([[0, np.nan]], [0], (1, 1)),
        ([[0, 0]], [np.inf], (1, 1)),
        ([[0, 0]], [0], (False, 1)),
    ]:
        with pytest.raises(ValueError):
            GrainPartition(p, w, size)
    with pytest.raises(ValueError):
        g.ownership([[np.nan, 0]])
    with pytest.raises(ValueError):
        g.ownership([[0, 0]], chunk_size=0)


def test_single_cell_full_torus_no_false_edges():
    g = GrainPartition([[0.2, 0.4]], [7], (2, 3))
    assert g.cell_area_exact(0) == 6
    assert g.areas()[0] == 6
    assert g.cell_segments(0).shape == (0, 2, 2)
    assert g.adjacency().shape == (0, 2)
    assert np.isinf(g.boundary_distance(rng().random((30, 2)) * [2, 3])).all()
    assert sum(polygon_area(p) for p in g.cell_fragments(0)) == pytest.approx(6)
    stats = g.area_statistics()
    assert stats.median_diameter_mm == pytest.approx(math.sqrt(24 / math.pi))
    assert stats.diameter_cv == 0
    assert not stats.statistically_sufficient


def test_equal_two_cell_strips_and_seams():
    g = GrainPartition([[0.25, 0.5], [0.75, 0.5]], [0, 0], (1, 1))
    points = np.array([[0, 0.2], [0.1, 0.9], [0.25, 0.5], [0.5, 0.8], [0.9, 0.1]])
    np.testing.assert_allclose(g.areas(), [0.5, 0.5])
    np.testing.assert_allclose(
        g.boundary_distance(points), [0, 0.1, 0.25, 0, 0.1], atol=1e-15
    )
    np.testing.assert_array_equal(g.ownership(points).ids, [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(g.adjacency(), [[0, 1]])
    np.testing.assert_array_equal(
        g.cell_neighbours(0), np.ones(len(g.cell_segments(0)), dtype=np.uint32)
    )


def test_three_strips_independent_distance():
    g = GrainPartition([[0, 0.5], [1, 0.5], [2, 0.5]], [0, 0, 0], (3, 1))
    q = np.array([[0, 0], [0.25, 0.2], [1.6, 0.7], [2.7, 0.5]])
    np.testing.assert_allclose(g.areas(), [1, 1, 1])
    np.testing.assert_allclose(g.boundary_distance(q), [0.5, 0.25, 0.1, 0.2])
    np.testing.assert_array_equal(g.adjacency(), [[0, 1], [0, 2], [1, 2]])


def test_hidden_cell_has_no_boundary():
    g = GrainPartition([[0.1, 0.35], [0.7, 0.35]], [0.4, 0], (1, 1))
    np.testing.assert_array_equal(g.areas(), [1, 0])
    assert np.isinf(g.boundary_distance([[0.2, 0.35], [0.9, 0.8]])).all()
    assert g.adjacency().size == 0
    assert g.area_statistics().hidden_count == 1
    assert g.cell_fragments(1) == ()


def test_boundary_is_segment_distance_not_infinite_bisector():
    g = GrainPartition([[0, 0], [0.5, 0.5]], [0.1, -0.1], (1, 1))
    q = [[0.49, 0]]
    expected = math.hypot(0.2, 0.01)
    assert g.boundary_distance(q)[0] == pytest.approx(expected, abs=1e-14)
    assert g.boundary_distance_oracle(q)[0] == pytest.approx(expected, abs=1e-14)
    np.testing.assert_allclose(g.areas(), [0.82, 0.18], atol=1e-15)


def test_near_hidden_positive_cell_is_never_dropped():
    g = GrainPartition([[0, 0], [0.5, 0.5]], [0.4999, 0], (1, 1))
    expected = 2 * (Fraction(1, 2) - Fraction(0.4999)) ** 2
    assert g.cell_area_exact(1) == expected
    assert g.areas()[1] == pytest.approx(2e-8, rel=1e-11)
    stats = g.area_statistics()
    np.testing.assert_array_equal(stats.active_ids, [0, 1])
    assert stats.small_cell_count == 1
    assert stats.smallest_area_mm2 > 0
    assert stats.hidden_count == 0
    assert stats.median_diameter_mm == pytest.approx(
        np.median(np.sqrt(4 * g.areas() / math.pi))
    )
    assert stats.diameter_cv == pytest.approx(
        np.std(stats.equivalent_diameters_mm) / np.mean(stats.equivalent_diameters_mm)
    )
    # A one-ULP reduction from the hiding threshold still gives a positive cell.
    tiny = GrainPartition([[0, 0], [0.5, 0.5]], [np.nextafter(0.5, 0), 0], (1, 1))
    assert tiny.areas()[1] > 0
    assert len(tiny.area_statistics().active_ids) == 2
    hidden = GrainPartition([[0, 0], [0.5, 0.5]], [0.5, 0], (1, 1))
    np.testing.assert_array_equal(hidden.areas(), [1, 0])
    assert np.isinf(hidden.boundary_distance([[0.5, 0.5]])[0])


def test_degenerate_zero_area_owner_and_real_adjacency():
    g = GrainPartition([[0.5, 0.5], [0.25, 0.5], [0.75, 0.5]], [-0.0625, 0, 0], (1, 1))
    np.testing.assert_array_equal(g.areas(), [0, 0.5, 0.5])
    assert g.ownership([[0.5, 0.5]]).ids[0] == 0
    np.testing.assert_array_equal(g.adjacency(), [[1, 2]])
    assert g.boundary_distance([[0.5, 0.5]])[0] == 0


def test_duplicate_nuclei_tie_original_ids():
    g = GrainPartition([[0.2, 0.3]] * 3, [0, 0, -1], (1, 1))
    np.testing.assert_array_equal(g.areas(), [1, 0, 0])
    np.testing.assert_array_equal(g.ownership(rng().random((50, 2))).ids, 0)
    assert np.isinf(g.boundary_distance([[0.6, 0.6]])[0])
    h = GrainPartition([[0.2, 0.3]] * 2, [0, 1], (1, 1))
    np.testing.assert_array_equal(h.areas(), [0, 1])
    assert h.ownership([[0, 0]]).ids[0] == 1


def test_lexicographic_nucleus_and_antipodal_image_ties():
    g = GrainPartition([[0.75, 0.75], [0.25, 0.25]], [0, 0], (1, 1))
    q = [[0, 0], [0.5, 0.5], [0.25, 0.25], [0.25, 0.75]]
    for result in (g.ownership(q), g.ownership_oracle(q)):
        for index, point in enumerate(q):
            _, expected_id, offset = exact_nine_image_reference(g, point)
            assert result.ids[index] == expected_id
            np.testing.assert_array_equal(result.image_offsets[index], offset)
    single = GrainPartition([[0.75, 0.75]], [0], (1, 1))
    result = single.ownership([[0.25, 0.25]])
    np.testing.assert_array_equal(result.image_offsets, [[-1, -1]])
    np.testing.assert_array_equal(result.local_mm, [[0.5, 0.5]])


def test_independent_rational_nine_image_oracle():
    random = rng(741)
    g = GrainPartition(random.random((9, 2)) * [2, 3], random.normal(0, 0.3, 9), (2, 3))
    points = random.uniform(-8, 8, (60, 2))
    result = g.ownership(points)
    for index, point in enumerate(points):
        metric, expected_id, offset = exact_nine_image_reference(g, point)
        assert result.ids[index] == expected_id
        np.testing.assert_array_equal(result.image_offsets[index], offset)
        assert result.metric_mm2[index] == pytest.approx(float(metric), abs=2e-14)


def test_100k_oracle_accelerator_adversarial_agreement():
    random = rng(9031)
    populations = []
    for n in (3, 17, 65, 129):
        points = random.random((n, 2)) * [2, 3]
        weights = random.normal(0, 0.15, n)
        populations.append(GrainPartition(points, weights, (2, 3)))
    # Distant high-weight nucleus and nonuniform cluster across a seam.
    points = np.vstack((random.uniform(0, 0.1, (63, 2)), [[1.0, 1.5]]))
    weights = np.zeros(64)
    weights[-1] = 4
    populations.append(GrainPartition(points, weights, (2, 3)))
    checked = 0
    for g in populations:
        points = random.uniform(-6, 9, (20000, 2))
        points[:20, 0] = np.tile([0, 2, np.nextafter(0, 1), np.nextafter(2, 0)], 5)
        fast = g.ownership(points, chunk_size=307)
        exact = g.ownership_oracle(points, chunk_size=521)
        np.testing.assert_array_equal(fast.ids, exact.ids)
        np.testing.assert_array_equal(fast.image_offsets, exact.image_offsets)
        np.testing.assert_array_equal(fast.displacement_mm, exact.displacement_mm)
        np.testing.assert_array_equal(fast.metric_mm2, exact.metric_mm2)
        assert fast.stats.peak_candidate_pairs <= 307 * 16
        checked += len(points)
    assert checked >= 100000


def test_arbitrary_periodic_queries_broadcast_and_chunks():
    random = rng(28)
    g = GrainPartition(
        random.random((37, 2)) * [2, 4], random.normal(0, 0.03, 37), (2, 4)
    )
    points = random.random((32, 2)) * [2, 4]
    a = g.ownership(points, chunk_size=1)
    b = g.ownership(points + np.array([8, -20]), chunk_size=7)
    np.testing.assert_array_equal(a.ids, b.ids)
    np.testing.assert_allclose(a.displacement_mm, b.displacement_mm, atol=3e-15)
    np.testing.assert_allclose(a.metric_mm2, b.metric_mm2, atol=3e-15)
    np.testing.assert_allclose(
        g.boundary_distance(points),
        g.boundary_distance(points + np.array([8, -20])),
        atol=3e-15,
    )
    grid = g.query(np.arange(3)[:, None], np.arange(4)[None, :])
    assert grid.ids.shape == (3, 4)
    assert grid.local_mm.shape == (3, 4, 2)
    assert g.query(0.2, 0.3).ids.shape == ()
    empty = g.ownership(np.empty((2, 0, 2)))
    assert empty.ids.shape == (2, 0)
    assert empty.stats.points == 0


def test_bounded_geometry_matches_all_constraint_reference():
    random = rng(681)
    g = GrainPartition(random.random((41, 2)), random.normal(0, 0.01, 41), (1, 1))
    q = random.random((40, 2))
    np.testing.assert_allclose(
        g.boundary_distance(q), g.boundary_distance_oracle(q), atol=1e-14
    )
    assert sum(g.cell_area_exact(i) for i in range(41)) == 1
    stats = g.area_statistics()
    assert abs(stats.areas_mm2.sum() - 1) <= stats.total_area_error_mm2
    for i in range(41):
        assert sum(polygon_area(p) for p in g.cell_fragments(i)) == pytest.approx(
            g.areas()[i], abs=5e-16
        )
    assert np.all(g.adjacency()[:, 0] < g.adjacency()[:, 1])


def test_placement_deterministic_toroidal_and_finite():
    a = place_grains((3, 2), 0.3, rng(41), count=60, minimum_distance_mm=0.18)
    b = place_grains((3, 2), 0.3, rng(41), count=60, minimum_distance_mm=0.18)
    np.testing.assert_array_equal(a, b)
    delta = np.abs(a[:, None] - a[None, :])
    delta = np.minimum(delta, [3, 2] - delta)
    distance = np.sqrt(np.sum(delta * delta, axis=-1))
    np.fill_diagonal(distance, np.inf)
    assert distance.min() >= 0.18
    with pytest.raises(PlacementError) as caught:
        place_grains(
            (1, 1), 0.5, rng(), count=3, minimum_distance_mm=1, candidates_per_point=5
        )
    assert caught.value.achieved == 1
    assert caught.value.requested == 3
    with pytest.raises(TypeError):
        place_grains((1, 1), 0.5, np.random.Generator(np.random.MT19937(1)))
    u = place_grains((3, 2), 0.3, rng(41), count=60, mode="uniform")
    np.testing.assert_array_equal(u, rng(41).random((60, 2)) * [3, 2])
    assert initial_nucleus_count((100, 100), 8) == 199
    assert initial_nucleus_count((100, 100), 0.4) == 79577


def test_growth_weights_bounded_gauge_and_replay():
    w = growth_weights(100, 0.8, rng(84), growth_sigma=0.8, max_weight_fraction=0.2)
    np.testing.assert_array_equal(
        w, growth_weights(100, 0.8, rng(84), growth_sigma=0.8, max_weight_fraction=0.2)
    )
    assert np.max(np.abs(w)) <= 0.2 * 0.8**2
    assert abs(w.mean()) < 1e-17
    np.testing.assert_array_equal(
        growth_weights(5, 1, rng(), growth_sigma=0), np.zeros(5)
    )
    np.testing.assert_array_equal(growth_weights(1, 1, rng()), [0])


def test_sub_ulp_wrapping_does_not_turn_near_ties_into_ties():
    # Remainder rounds these negative inputs to 1.0. Exact comparison must
    # retain which side of the seam the represented original point lies on.
    g = GrainPartition([[0.25, 0.5], [0.75, 0.5]], [0, 0], (1, 1))
    points = np.array(
        [[-np.nextafter(0.0, 1.0), 0.5], [-1e-17, 0.5], [0.0, 0.5], [1e-17, 0.5]]
    )
    for method in (g.ownership, g.ownership_oracle):
        np.testing.assert_array_equal(method(points).ids, [1, 1, 0, 0])
    # Similarly, image choice at an antipode must use the original coordinate.
    single = GrainPartition([[0.5, 0.5]], [0], (1, 1))
    for method in (single.ownership, single.ownership_oracle):
        np.testing.assert_array_equal(method(points).image_offsets[:, 0], [0, 0, -1, 0])


def test_finite_global_poisson_proposals_cover_tile():
    # A deterministic regression against truncating a single spreading front.
    points = place_grains((100, 100), 8, rng(42))
    quadrants = (points[:, 0] >= 50).astype(int) + 2 * (points[:, 1] >= 50)
    assert np.bincount(quadrants, minlength=4).min() > 30


def test_isolated_tie_owner_does_not_create_a_boundary():
    g = GrainPartition([[0.5, 0.5], [0, 0]], [0, 0.5], (1, 1))
    np.testing.assert_array_equal(g.areas(), [0, 1])
    assert g.ownership([[0.5, 0.5]]).ids[0] == 0
    assert np.isinf(g.boundary_distance([[0.5, 0.5]])[0])
    assert g.adjacency().shape == (0, 2)


@pytest.mark.parametrize("exponent", [-500, 500])
def test_statistics_do_not_underflow_or_overflow_when_rescaled(exponent):
    scale = math.ldexp(1.0, exponent)
    g = GrainPartition(
        np.array([[0, 0], [0.5, 0.5]]) * scale,
        np.array([0.1, -0.1]) * scale**2,
        (scale, scale),
    )
    stats = g.area_statistics()
    diameter = np.sqrt(4 * np.array([0.82, 0.18]) / math.pi)
    np.testing.assert_allclose(
        stats.equivalent_diameters_mm / scale, diameter, rtol=1e-14
    )
    assert stats.diameter_cv == pytest.approx(
        np.std(diameter) / np.mean(diameter), abs=1e-14
    )


def test_positive_subnormal_area_keeps_positive_equivalent_diameter():
    length = 1e-150
    g = GrainPartition(
        [[0, 0], [length / 2, length / 2]],
        [0.5 * length**2 - 1.5e-312, 0],
        (length, length),
    )
    stats = g.area_statistics()
    assert 0 < stats.smallest_area_mm2 < np.finfo(float).tiny
    assert len(stats.active_ids) == 2
    assert np.all(stats.equivalent_diameters_mm > 0)
    assert np.isfinite(stats.diameter_cv_error)


def test_nearest_boundary_is_not_the_metric_runner_up():
    g = GrainPartition([[2, 2], [2.1, 2], [2, 3]], [0, -0.01, 0.9], (10, 10))
    point = np.array([2.0, 2.0])
    metrics = np.sum((g.points_mm - point) ** 2, axis=1) - g.weights_mm2
    np.testing.assert_array_equal(np.argsort(metrics), [0, 1, 2])
    # Grain 1's edge is x=2.1, but grain 2's active edge is y=2.05.
    assert g.boundary_distance(point) == pytest.approx(0.05, abs=1e-14)
    assert g.boundary_distance_oracle(point) == pytest.approx(0.05, abs=1e-14)
