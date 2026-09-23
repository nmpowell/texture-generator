"""Independent support, attachment and sampling checks for branch morphology."""

from dataclasses import replace

import numpy as np

from texture_generators.core.dendrites import (
    SEGMENT_DTYPE,
    build_dendrites,
    evaluate_dendrite_fields,
)


def _fields(records, x, y, size=(24.0, 18.0), diameter=8.0):
    x, y = np.broadcast_arrays(x, y)
    return evaluate_dendrite_fields(
        records, np.zeros(x.shape, dtype=np.uint32), x, y, size, diameter
    )


def test_branches_have_geometric_parent_attachments_and_physical_widths():
    records = build_dendrites(np.array([[8.0, 8.0]]), 8.0, 4.0, 42)
    segments = records.segments
    assert set(segments["kind"]) == {0, 1, 2}
    for kind in (1, 2):
        parents = segments[segments["kind"] == kind - 1]
        children = segments[segments["kind"] == kind]
        delta = parents["end_mm"] - parents["start_mm"]
        lengths_squared = np.sum(delta * delta, axis=-1)
        for point in children["start_mm"]:
            offset = point - parents["start_mm"]
            fraction = np.sum(offset * delta, axis=-1) / lengths_squared
            projected = parents["start_mm"] + fraction[:, None] * delta
            attached = (fraction > 0.0) & (fraction < 1.0)
            attached &= np.linalg.norm(projected - point, axis=-1) < 1e-12
            assert np.any(attached)
    # No former grain-wide 0.26D Gabor strip remains in a regular 8 mm grain.
    assert np.max(segments["width_mm"]) < 0.10
    assert not segments.flags.writeable


def test_every_field_is_exactly_independent_of_chunks_and_traversal():
    records = build_dendrites(np.array([[0.2, 0.3]]), 8.0, 4.0, 91)
    rng = np.random.default_rng(37)
    x = rng.uniform(-24.0, 48.0, 911)
    y = rng.uniform(-18.0, 36.0, 911)
    expected = _fields(records, x, y)
    for stride in (1, 31, 128):
        chunks = [
            _fields(records, x[start : start + stride], y[start : start + stride])
            for start in range(0, len(x), stride)
        ]
        for field in (0, 1):
            actual = np.concatenate([chunk[field] for chunk in chunks])
            np.testing.assert_array_equal(actual, expected[field])
    order = rng.permutation(len(x))
    shuffled = _fields(records, x[order], y[order])
    for field in (0, 1):
        np.testing.assert_array_equal(shuffled[field], expected[field][order])


def test_spanning_one_grain_field_and_derivatives_are_periodic():
    records = build_dendrites(np.array([[0.12, 0.32]]), 1.1, 4.0, 91)
    x = np.array([0.003, 0.234, 0.5, 0.998])
    y = np.array([0.32, 0.49, 0.7, 0.15])
    epsilon = 1e-6
    for offset in (-epsilon, 0.0, epsilon):
        original = _fields(records, x + offset, y, (1.0, 1.0), 1.1)
        translated = _fields(records, x + offset + 3, y - 2, (1.0, 1.0), 1.1)
        for left, right in zip(original, translated, strict=True):
            np.testing.assert_allclose(left, right, atol=3e-12, rtol=0.0)
    left = _fields(records, np.array([-epsilon]), np.array([0.32]), (1.0, 1.0), 1.1)
    right = _fields(records, np.array([epsilon]), np.array([0.32]), (1.0, 1.0), 1.1)
    centre = _fields(records, np.array([0.0]), np.array([0.32]), (1.0, 1.0), 1.1)
    for a, b, c in zip(left, right, centre, strict=True):
        np.testing.assert_allclose((c - a) / epsilon, (b - c) / epsilon, atol=0.01)


def test_compact_kernel_value_and_derivative_vanish_at_its_support():
    original = build_dendrites(np.array([[4.0, 4.0]]), 2.0, 4.0, 1)
    segment = np.array(
        [(0, (4.0, 4.0), (6.0, 4.0), 0.1, 1.0, 0.31, 1, 0.4)], dtype=SEGMENT_DTYPE
    )
    records = replace(original, segments=segment)
    epsilon = 1e-6
    # At the midpoint, width is 0.1*(1-0.55/2); lateral support is 3 widths.
    edge_y = 4.0 + 3.0 * 0.1 * (1.0 - 0.55 / 2.0)
    for x, y, dx, dy in (
        (4.0, 4.0, epsilon, 0),
        (6.0, 4.0, -epsilon, 0),
        (5.0, edge_y, 0, -epsilon),
    ):
        outside = _fields(records, np.array([x - dx]), np.array([y - dy]))
        edge = _fields(records, np.array([x]), np.array([y]))
        inside = _fields(records, np.array([x + dx]), np.array([y + dy]))
        for a, b, c in zip(outside, edge, inside, strict=True):
            np.testing.assert_allclose(a, 0.0, atol=1e-25)
            np.testing.assert_allclose(b, 0.0, atol=1e-25)
            assert np.max(np.abs(c - b)) / epsilon < 1e-3


def test_state_is_replayable_and_tilt_is_safe_at_extremes():
    points = np.array([[2.0, 3.0], [9.0, 12.0]])
    first = build_dendrites(points, 8.0, 4.0, 29)
    second = build_dendrites(points, 8.0, 4.0, 29)
    np.testing.assert_array_equal(first.segments, second.segments)
    for concentration in (0.0, 1e-10, 1000.0):
        alternate = build_dendrites(points, 8.0, concentration, 29)
        assert np.all(np.isfinite(alternate.tilt_rad))
        np.testing.assert_array_equal(first.azimuth_rad, alternate.azimuth_rad)
        np.testing.assert_array_equal(first.segments, alternate.segments)


def test_direction_descriptor_preserves_cancellation_and_vanishing_support():
    original = build_dendrites(np.array([[4.0, 4.0]]), 2.0, 4.0, 1)
    segments = np.array(
        [
            (0, (4.0, 5.0), (6.0, 5.0), 0.1, 1.0, 0.31, 1, 0.4),
            (0, (5.0, 4.0), (5.0, 6.0), 0.1, 1.0, 0.31, 1, 0.4),
        ],
        dtype=SEGMENT_DTYPE,
    )
    crossed = replace(original, segments=segments)
    _, vector = _fields(crossed, np.array([5.0, 7.0]), np.array([5.0, 7.0]))
    np.testing.assert_allclose(vector, 0.0, atol=1e-15)
    single = replace(original, segments=segments[:1])
    _, vector = _fields(single, np.array([5.0]), np.array([5.0]))
    assert 0.0 < np.linalg.norm(vector) < 1.0
    grid = np.linspace(3, 7, 129)
    _, vector = _fields(crossed, grid[:, None], grid[None, :])
    assert np.max(np.linalg.norm(vector, axis=-1)) < 1.0


def test_projected_families_keep_coherent_sidearms_and_an_elongated_x():
    # Several orientation quantiles exercise every authoring family without
    # selecting a family from rendered brightness or changing the seed stream.
    points = np.column_stack((np.arange(64) * 4.0, np.zeros(64)))
    records = build_dendrites(points, 2.0, 4.0, 29)
    assert set(records.family_id) == set(range(len(records.family_names)))
    for grain_id, origin in enumerate(points):
        family = records.family_names[int(records.family_id[grain_id])]
        segments = records.segments[records.segments["grain_id"] == grain_id]
        primaries = segments[segments["kind"] == 0]
        secondaries = segments[segments["kind"] == 1]
        delta = primaries["end_mm"] - origin
        directions = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
        if family == "X":
            eigenvalues = np.linalg.eigvalsh(directions.T @ directions)
            # An equal-angle diagonal cross has equal eigenvalues. The
            # projected elongated-X family must actually be elongated.
            assert eigenvalues[1] > 3.0 * eigenvalues[0]
        if family != "two":
            branches = secondaries["end_mm"] - secondaries["start_mm"]
            branches /= np.linalg.norm(branches, axis=-1, keepdims=True)
            alignment = np.max(branches @ directions.T, axis=-1)
            assert np.all(alignment >= np.cos(np.deg2rad(2.0)) - 1e-12)

        # Each sidearm terminates in the angular sector of its actual parent;
        # it cannot cross the sector bisector into another primary fan.
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        starts = secondaries["start_mm"] - origin
        start_angles = np.arctan2(starts[:, 1], starts[:, 0])
        differences = (start_angles[:, None] - angles + np.pi) % (2 * np.pi) - np.pi
        parent = np.argmin(np.abs(differences), axis=-1)
        ends = secondaries["end_mm"] - origin
        end_angles = np.arctan2(ends[:, 1], ends[:, 0])
        deflection = (end_angles - angles[parent] + np.pi) % (2 * np.pi) - np.pi
        side = np.sign(deflection).astype(int)
        neighbour = (parent + side) % len(angles)
        gaps = (side * (angles[neighbour] - angles[parent])) % (2 * np.pi)
        assert np.all(np.abs(deflection) <= gaps / 2.0 + 1e-11)


def test_physical_support_bound_covers_all_branch_orders_and_spangle_scales():
    origin = np.array([[0.12, 0.32]])
    for diameter in (0.4, 1.5, 8.0, 16.0, 37.0):
        records = build_dendrites(origin, diameter, 4.0, 42)
        segments = records.segments
        start_radius = np.linalg.norm(segments["start_mm"] - origin, axis=-1)
        end_radius = np.linalg.norm(segments["end_mm"] - origin, axis=-1)
        assert np.max(np.maximum(start_radius, end_radius)) <= 1.15 * diameter
        # Evaluation's periodic image budget uses 1.25D, including all compact
        # lateral support, rather than just the centreline endpoints.
        support = np.maximum(start_radius, end_radius) + 3 * segments["width_mm"]
        assert np.max(support) < 1.25 * diameter
        lengths = np.linalg.norm(segments["end_mm"] - segments["start_mm"], axis=-1)
        assert np.all(lengths > 0.0)
        assert np.max(segments["width_mm"]) <= 0.045 * 1.1
        tertiary_lengths = lengths[segments["kind"] == 2]
        assert np.max(tertiary_lengths) <= min(0.135, diameter * 0.11) * 0.65 + 1e-12
