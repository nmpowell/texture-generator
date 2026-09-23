"""Private conservative search and exact polygon arithmetic for periodic grains."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from itertools import product

import numpy as np
from numpy.typing import DTypeLike

F = Fraction
Point = tuple[Fraction, Fraction]
# Normal, offset, original ID. The edge's interior satisfies a*x+b*y <= c.
Plane = tuple[Fraction, Fraction, Fraction, int]
Vertex = tuple[Point, Plane]
OFFSETS = tuple(product((-1, 0, 1), repeat=2))
EPS = np.finfo(np.float64).eps


def frozen_array(value: object, dtype: DTypeLike = np.float64) -> np.ndarray:
    """An immutable bytes owner prevents callers re-enabling WRITEABLE."""
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, slots=True)
class Node:
    lo: np.ndarray
    hi: np.ndarray
    max_weight: float
    ids: np.ndarray
    left: int = -1
    right: int = -1


class Hierarchy:
    def __init__(self, points: np.ndarray, weights: np.ndarray, size: np.ndarray):
        self.points = points
        self.weights = weights
        self.size = size
        self.nodes: list[Node] = []
        # This deliberately loose error envelope also covers cancellation in bounds.
        self.guard = max(
            256 * EPS * (float(size @ size) + float(np.max(np.abs(weights)))),
            1024 * float(np.nextafter(0.0, 1.0)),
        )
        self._build(np.arange(len(points)))
        self.lo = np.stack([node.lo for node in self.nodes])
        self.hi = np.stack([node.hi for node in self.nodes])
        self.max_weight = np.array([node.max_weight for node in self.nodes])
        self.left = np.array([node.left for node in self.nodes], dtype=np.int32)
        self.right = np.array([node.right for node in self.nodes], dtype=np.int32)
        self.leaf_ids = np.zeros((len(self.nodes), 16), dtype=np.int64)
        self.leaf_count = np.array([len(node.ids) for node in self.nodes])
        for index, node in enumerate(self.nodes):
            self.leaf_ids[index, : len(node.ids)] = node.ids
        self.depth = max(1, (len(points) - 1).bit_length() + 1)

    def _build(self, ids: np.ndarray) -> int:
        p = self.points[ids]
        lo, hi = p.min(axis=0), p.max(axis=0)
        index = len(self.nodes)
        self.nodes.append(Node(lo, hi, float(self.weights[ids].max()), ids))
        if len(ids) > 16:
            axis = int(np.argmax(hi - lo))
            ids = ids[np.argsort(p[:, axis], kind="stable")]
            mid = len(ids) // 2
            left, right = self._build(ids[:mid]), self._build(ids[mid:])
            self.nodes[index] = Node(
                lo,
                hi,
                self.nodes[index].max_weight,
                np.empty(0, dtype=int),
                left,
                right,
            )
        else:
            self.nodes[index] = Node(lo, hi, self.nodes[index].max_weight, np.sort(ids))
        return index

    def lower(self, q: np.ndarray, node: Node) -> np.ndarray:
        d = np.full_like(q, np.inf)
        for shift in (-1, 0, 1):
            d = np.minimum(
                d,
                np.maximum(
                    np.maximum(
                        node.lo + shift * self.size - q, q - node.hi - shift * self.size
                    ),
                    0,
                ),
            )
        return np.sum(d * d, axis=-1) - node.max_weight - self.guard

    def region_lower(self, lo: np.ndarray, hi: np.ndarray, node: Node) -> float:
        d = np.full(2, np.inf)
        for shift in (-1, 0, 1):
            d = np.minimum(
                d,
                np.maximum(
                    np.maximum(
                        node.lo + shift * self.size - hi,
                        lo - node.hi - shift * self.size,
                    ),
                    0,
                ),
            )
        return float(d @ d) - node.max_weight - self.guard

    def lower_many(self, q: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """One independently chosen node per point, vectorised over the batch."""
        lo, hi = self.lo[indices], self.hi[indices]
        bounds = []
        for axis in (0, 1):
            coordinate = q[:, axis]
            minimum = np.full(len(q), np.inf)
            for shift in (-1, 0, 1):
                translation = shift * self.size[axis]
                lower = lo[:, axis] + translation - coordinate
                upper = coordinate - hi[:, axis] - translation
                np.minimum(
                    minimum, np.maximum(np.maximum(lower, upper), 0), out=minimum
                )
            bounds.append(minimum)
        return (
            bounds[0] * bounds[0]
            + bounds[1] * bounds[1]
            - self.max_weight[indices]
            - self.guard
        )


def nearest_images(
    q: np.ndarray,
    seeds: np.ndarray,
    size: np.ndarray,
    raw_queries: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    seeds = np.broadcast_to(seeds, (len(q), *seeds.shape[-2:]))
    delta = q[:, None, :] - seeds
    half = size * 0.5
    offsets = np.where(delta <= -half, -1, np.where(delta > half, 1, 0)).astype(np.int8)
    # Rounded subtraction must not turn a near antipodal point into an image tie.
    ambiguous = np.argwhere(np.abs(np.abs(delta) - half) <= 8 * EPS * size)
    for row, col, axis in ambiguous:
        coordinate = (
            F(float(q[row, axis]))
            if raw_queries is None
            else F(float(raw_queries[row, axis])) % F(float(size[axis]))
        )
        exact = coordinate - F(float(seeds[row, col, axis]))
        h = F(float(size[axis])) / 2
        offsets[row, col, axis] = -1 if exact <= -h else (1 if exact > h else 0)
    return delta - offsets * size, offsets


def exact_metric(
    q: np.ndarray,
    seed: np.ndarray,
    weight: float,
    size: np.ndarray,
    offset: np.ndarray,
    raw_query: np.ndarray | None = None,
) -> Fraction:
    coordinate = [
        F(float(q[k]))
        if raw_query is None
        else F(float(raw_query[k])) % F(float(size[k]))
        for k in (0, 1)
    ]
    return sum(
        (
            (coordinate[k] - F(float(seed[k])) - int(offset[k]) * F(float(size[k])))
            ** 2
            for k in (0, 1)
        ),
        F(0),
    ) - F(float(weight))


def update_winners(
    q: np.ndarray,
    ids: np.ndarray,
    tree: Hierarchy,
    best: np.ndarray,
    winners: np.ndarray,
    offsets: np.ndarray,
    displacements: np.ndarray,
    valid: np.ndarray | None = None,
    raw_queries: np.ndarray | None = None,
) -> int:
    ids = np.broadcast_to(ids, (len(q), ids.shape[-1]))
    delta, images = nearest_images(q, tree.points[ids], tree.size, raw_queries)
    metrics = np.sum(delta * delta, axis=-1) - tree.weights[ids]
    if valid is not None:
        metrics = np.where(valid, metrics, np.inf)
    local = np.argmin(metrics, axis=1)
    rows = np.arange(len(q))
    values = metrics[rows, local]
    fallbacks = 0
    # Distinguish exact ties from metrics collapsed by floating-point arithmetic.
    ambiguous = np.sum(metrics <= values[:, None] + tree.guard, axis=1) > 1
    for row in np.flatnonzero(ambiguous):
        candidates = np.flatnonzero(metrics[row] <= values[row] + tree.guard)
        local[row] = min(
            candidates,
            key=lambda col: (
                exact_metric(
                    q[row],
                    tree.points[ids[row, col]],
                    tree.weights[ids[row, col]],
                    tree.size,
                    images[row, col],
                    None if raw_queries is None else raw_queries[row],
                ),
                int(ids[row, col]),
                tuple(images[row, col]),
            ),
        )
        fallbacks += len(candidates)
    values = metrics[rows, local]
    selected = ids[rows, local]
    replace = values < best
    close = np.isfinite(best) & (np.abs(values - best) <= tree.guard)
    for row in np.flatnonzero(close):
        new_key = (
            exact_metric(
                q[row],
                tree.points[selected[row]],
                tree.weights[selected[row]],
                tree.size,
                images[row, local[row]],
                None if raw_queries is None else raw_queries[row],
            ),
            int(selected[row]),
            tuple(images[row, local[row]]),
        )
        old_key = (
            exact_metric(
                q[row],
                tree.points[winners[row]],
                tree.weights[winners[row]],
                tree.size,
                offsets[row],
                None if raw_queries is None else raw_queries[row],
            ),
            int(winners[row]),
            tuple(offsets[row]),
        )
        replace[row] = new_key < old_key
        fallbacks += 2
    best[replace] = values[replace]
    winners[replace] = selected[replace]
    offsets[replace] = images[rows[replace], local[replace]]
    displacements[replace] = delta[rows[replace], local[replace]]
    return fallbacks


def clip(poly: list[Vertex], plane: Plane) -> list[Vertex]:
    """Exact closed half-plane clipping, retaining incoming edge provenance."""
    if not poly:
        return []
    a, b, c, _ = plane
    values = [a * p[0] + b * p[1] - c for p, _ in poly]
    if all(v <= 0 for v in values):
        # Coincident constraints: the strongest outward derivative determines
        # the cell across the edge, including self-images and zero-area rivals.
        result = list(poly)
        for k in range(len(poly)):
            if values[k - 1] == values[k] == 0:
                old = poly[k][1]
                if (a * a + b * b, -plane[3]) > (old[0] ** 2 + old[1] ** 2, -old[3]):
                    result[k] = (poly[k][0], plane)
        return result
    if all(v > 0 for v in values):
        return []
    out: list[Vertex] = []
    for k, (end, old_plane) in enumerate(poly):
        start = poly[k - 1][0]
        v0, v1 = values[k - 1], values[k]
        if (v0 <= 0) != (v1 <= 0):
            t = v0 / (v0 - v1)
            intersection = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
            )
            out.append((intersection, old_plane if v0 <= 0 else plane))
        if v1 <= 0:
            out.append((end, old_plane))
    # Zero-length edges do not carry a boundary or area.
    clean: list[Vertex] = []
    for vertex in out:
        if not clean or clean[-1][0] != vertex[0]:
            clean.append(vertex)
    if len(clean) > 1 and clean[0][0] == clean[-1][0]:
        clean[0] = (clean[0][0], clean[-1][1])
        clean.pop()
    return clean


def polygon_area(poly: list[Vertex]) -> Fraction:
    if len(poly) < 3:
        return F(0)
    return (
        abs(
            sum(
                (
                    poly[k - 1][0][0] * p[1] - p[0] * poly[k - 1][0][1]
                    for k, (p, _) in enumerate(poly)
                ),
                F(0),
            )
        )
        / 2
    )


@dataclass(frozen=True, slots=True)
class Cell:
    polygon: tuple[Point, ...]
    area: Fraction
    segments: np.ndarray
    neighbours: np.ndarray
    candidate_images: int
    nodes_visited: int


def build_cell(grain_id: int, tree: Hierarchy, *, exhaustive: bool = False) -> Cell:
    """Clip the central image against all non-excludable competitors.

    The starting self-image rectangle is retained throughout. Whole-polygon
    lower bounds, rather than the ownership stopping rule, exclude tree nodes.
    All constructions/predicates/areas use exact rationals of the input doubles.
    """
    size = tuple(F(float(v)) for v in tree.size)
    seed = tuple(F(float(v)) for v in tree.points[grain_id])
    weight = F(float(tree.weights[grain_id]))
    hx, hy = size[0] / 2, size[1] / 2
    planes = [
        (F(-2) * size[0], F(0), size[0] ** 2, grain_id),
        (F(0), F(-2) * size[1], size[1] ** 2, grain_id),
        (F(2) * size[0], F(0), size[0] ** 2, grain_id),
        (F(0), F(2) * size[1], size[1] ** 2, grain_id),
    ]
    poly: list[Vertex] = [
        ((-hx, -hy), planes[0]),
        ((hx, -hy), planes[1]),
        ((hx, hy), planes[2]),
        ((-hx, hy), planes[3]),
    ]
    queue = [
        (float(tree.lower(tree.points[grain_id : grain_id + 1], tree.nodes[0])[0]), 0)
    ]
    count = visited = 0
    while queue and poly:
        _, index = heappop(queue)
        node = tree.nodes[index]
        visited += 1
        if not exhaustive:
            vertices = np.asarray([[float(p[0]), float(p[1])] for p, _ in poly])
            world = vertices + tree.points[grain_id]
            lo = np.nextafter(world.min(axis=0), -np.inf)
            hi = np.nextafter(world.max(axis=0), np.inf)
            upper = (
                float(np.max(np.sum(vertices * vertices, axis=1)))
                - float(weight)
                + tree.guard
            )
            if tree.region_lower(lo, hi, node) > upper:
                continue
        if node.left >= 0:
            for child in (node.left, node.right):
                key = float(
                    tree.lower(tree.points[grain_id : grain_id + 1], tree.nodes[child])[
                        0
                    ]
                )
                heappush(queue, (key, child))
            continue
        # Near competitors shrink the polygon before more distant nodes arrive.
        delta, _ = nearest_images(
            tree.points[grain_id : grain_id + 1], tree.points[node.ids], tree.size
        )
        order = np.argsort(
            np.sum(delta[0] ** 2, axis=1) - tree.weights[node.ids], kind="stable"
        )
        for other in node.ids[order]:
            if other == grain_id:
                continue
            candidate_offsets = OFFSETS
            if not exhaustive:
                # A float filter may only discard constraints strictly inside
                # the roundoff envelope. All cuts and ambiguous signs still
                # use the rational construction below.
                vertices = np.asarray([[float(p[0]), float(p[1])] for p, _ in poly])
                delta_images = (
                    tree.points[other]
                    + np.asarray(OFFSETS) * tree.size
                    - tree.points[grain_id]
                )
                constants = (
                    np.sum(delta_images * delta_images, axis=1)
                    + tree.weights[grain_id]
                    - tree.weights[other]
                )
                values = 2 * delta_images @ vertices.T - constants[:, None]
                candidate_offsets = tuple(
                    OFFSETS[k]
                    for k in np.flatnonzero(np.max(values, axis=1) >= -tree.guard)
                )
            count += 9
            if not candidate_offsets:
                continue
            other_seed = tuple(F(float(v)) for v in tree.points[other])
            other_weight = F(float(tree.weights[other]))
            for ox, oy in candidate_offsets:
                dx = other_seed[0] + ox * size[0] - seed[0]
                dy = other_seed[1] + oy * size[1] - seed[1]
                c = dx * dx + dy * dy + weight - other_weight
                if dx == dy == 0:
                    if c < 0 or (c == 0 and other < grain_id):
                        poly = []
                        break
                    continue
                poly = clip(poly, (2 * dx, 2 * dy, c, int(other)))
                if not poly:
                    break
            if not poly:
                break
    area = polygon_area(poly)
    segments, neighbours = [], []
    if area > 0:
        for k, (end, plane) in enumerate(poly):
            start = poly[k - 1][0]
            if plane[3] != grain_id and start != end:
                segments.append(
                    [
                        [float(start[j] + seed[j]) for j in (0, 1)],
                        [float(end[j] + seed[j]) for j in (0, 1)],
                    ]
                )
                neighbours.append(plane[3])
    return Cell(
        tuple(p for p, _ in poly),
        area,
        frozen_array(segments).reshape(-1, 2, 2),
        frozen_array(neighbours, np.uint32),
        count,
        visited,
    )


def fragments(cell: Cell, grain_id: int, tree: Hierarchy) -> tuple[np.ndarray, ...]:
    """Return nonempty exact-clipped fragments inside the fundamental tile."""
    if cell.area == 0:
        return ()
    sx, sy = (F(float(v)) for v in tree.points[grain_id])
    lx, ly = (F(float(v)) for v in tree.size)
    tile_planes = [
        (F(-1), F(0), F(0), -1),
        (F(1), F(0), lx, -1),
        (F(0), F(-1), F(0), -1),
        (F(0), F(1), ly, -1),
    ]
    result = []
    # Actual edge provenance is already retained in segments/neighbours. Tile
    # clipping below only needs vertices; do not cache every rational plane.
    placeholder = (F(0), F(0), F(0), grain_id)
    for ox, oy in OFFSETS:
        tx, ty = sx + ox * lx, sy + oy * ly
        poly = [
            (
                (p[0] + tx, p[1] + ty),
                placeholder,
            )
            for p in cell.polygon
        ]
        for plane in tile_planes:
            poly = clip(poly, plane)
        if polygon_area(poly) > 0:
            result.append(frozen_array([[float(v) for v in p] for p, _ in poly]))
    return tuple(result)


def segment_distances(
    q: np.ndarray, segments: np.ndarray, size: np.ndarray
) -> np.ndarray:
    result = np.full(len(q), np.inf)
    # O(P) work arrays, even if an adversarial cell has O(N) edges.
    for segment in segments:
        edge = segment[1] - segment[0]
        denom = float(edge @ edge)
        for offset in OFFSETS:
            delta = q - segment[0] - np.asarray(offset) * size
            t = np.clip((delta @ edge) / denom, 0, 1) if denom else np.zeros(len(q))
            d = delta - t[:, None] * edge
            result = np.minimum(result, np.sqrt(np.sum(d * d, axis=1)))
    return result


def packed_segment_distances(
    q: np.ndarray, cells: list[Cell], owners: np.ndarray, size: np.ndarray
) -> np.ndarray:
    """Vectorise different owners' segment lists without a points x cells array."""
    counts = np.array([len(cell.segments) for cell in cells])
    squared = np.full(len(q), np.inf)
    if not counts.any():
        return squared
    segments = np.concatenate([cell.segments for cell in cells])
    # Segment geometry is invariant over every query in this chunk. Compute it
    # once, then gather only the segment used by each current point/edge pair.
    edges = segments[:, 1] - segments[:, 0]
    denominator = edges[:, 0] * edges[:, 0] + edges[:, 1] * edges[:, 1]
    starts = np.cumsum(np.r_[0, counts[:-1]])
    point_counts = counts[owners]
    translations = np.asarray(OFFSETS) * size
    for index in range(int(counts.max())):
        rows = np.flatnonzero(point_counts > index)
        segment_indices = starts[owners[rows]] + index
        start = segments[segment_indices, 0]
        edge = edges[segment_indices]
        denom = denominator[segment_indices]
        points = q[rows]
        minimum = squared[rows]
        for translation in translations:
            delta = points - start - translation
            t = np.divide(
                delta[:, 0] * edge[:, 0] + delta[:, 1] * edge[:, 1],
                denom,
                out=np.zeros(len(rows)),
                where=denom != 0,
            )
            d = delta - np.clip(t, 0, 1)[:, None] * edge
            minimum = np.minimum(minimum, d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
        squared[rows] = minimum
    return np.sqrt(squared)
