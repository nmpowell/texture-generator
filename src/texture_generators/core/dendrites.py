"""Resolution-independent, compact galvanised dendrite records and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

import numpy as np

from texture_generators.core.random_fields import topology_rng

SEGMENT_DTYPE = np.dtype(
    [
        ("grain_id", np.uint32),
        ("start_mm", np.float64, (2,)),
        ("end_mm", np.float64, (2,)),
        ("width_mm", np.float64),
        ("amplitude", np.float64),
        ("period_mm", np.float64),
        ("kind", np.uint8),
        ("phase", np.float64),
    ]
)
_SEGMENT_BLOCK_SIZE = 4096


def _readonly(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True)
class DendriteRecords:
    """Immutable physical skeleton and one coherent crystal orientation per grain."""

    azimuth_rad: np.ndarray
    tilt_rad: np.ndarray
    family_id: np.ndarray
    segments: np.ndarray
    family_names: tuple[str, ...]
    table_version: str
    direction_support_floor: float = 0.12


def _inverse_hemisphere_tilt(u: np.ndarray, concentration: float) -> np.ndarray:
    """Invert exp(k*cos(tilt)) on a hemisphere without overflow."""
    if concentration < 1e-7:
        cosine = u
    else:
        # log((1-u)+u*exp(k)) / k in a stable form.
        cosine = 1.0 + np.log(u + (1.0 - u) * np.exp(-concentration)) / concentration
    return np.arccos(np.clip(cosine, 0.0, 1.0))


def _radial_reach(offset: np.ndarray, direction: np.ndarray, radius: float) -> float:
    """Distance along a ray to a fixed physical support circle."""
    projection = float(np.dot(offset, direction))
    discriminant = radius**2 - float(np.dot(offset, offset)) + projection**2
    return max(0.0, -projection + np.sqrt(max(0.0, discriminant)))


def build_dendrites(
    points_mm: np.ndarray,
    diameter_mm: float,
    concentration: float,
    material_key: int,
) -> DendriteRecords:
    """Build impinging growth sectors with attached, irregularly spaced branches.

    Family directions remain coherent within a grain. Competition changes the
    extent and strength of those sectors; it does not rotate every fine branch
    independently or paint a radial brightness template onto the grain.
    """
    with (
        resources.files("texture_generators")
        .joinpath("data/galvanised/morphology.json")
        .open("r", encoding="utf-8") as stream
    ):
        table = json.load(stream)
    families = table["families"]
    construction = table["construction"]
    support_radius = diameter_mm * float(construction["support_radius_diameters"])
    quantiles = np.asarray(table["family_quantiles"], dtype=np.float64)
    if len(families) != len(quantiles) or quantiles[-1] != 1.0:
        raise ValueError("invalid morphology family table")

    count = len(points_mm)
    orient_rng = topology_rng(material_key, "crystal_orientations")
    azimuth = orient_rng.uniform(0.0, 2.0 * np.pi, count)
    tilt_quantile = orient_rng.random(count)
    tilt = _inverse_hemisphere_tilt(tilt_quantile, concentration)
    # Quantile bins are authoring family populations, not measured tilt angles.
    family_id = np.searchsorted(quantiles, tilt_quantile, side="right").astype(np.uint8)
    family_id = np.minimum(family_id, len(families) - 1)
    segments: list[tuple[object, ...]] = []
    blocks: list[np.ndarray] = []
    for grain_id in range(count):
        family = families[int(family_id[grain_id])]
        rng = topology_rng(material_key, "dendrite", grain_id)
        origin = np.asarray(points_mm[grain_id], dtype=np.float64)
        preferred = rng.uniform(0.0, 2.0 * np.pi)
        branch_spacing = float(construction["secondary_spacing_mm"])
        # Smaller spangles retain fewer branches, rather than scaling a fixed
        # decorative fern down to arbitrarily fine physical wavelengths.
        branch_spacing = min(branch_spacing, diameter_mm * 0.11)
        family_angles = np.deg2rad(np.asarray(family["angles_deg"], dtype=np.float64))
        for arm_index, (angle_deg, relative_length) in enumerate(
            zip(family["angles_deg"], family["lengths"], strict=True)
        ):
            angle = azimuth[grain_id] + np.deg2rad(angle_deg)
            direction = np.array([np.cos(angle), np.sin(angle)])
            normal = np.array([-direction[1], direction[0]])
            competition = 0.22 + 0.78 * np.exp(2.2 * (np.cos(angle - preferred) - 1))
            length = (
                diameter_mm
                * float(construction["primary_extent_diameters"])
                * relative_length
                * rng.uniform(0.80, 1.17)
                * (0.68 + 0.32 * competition)
            )
            sector_amplitude = rng.uniform(0.70, 1.12) * (0.55 + 0.45 * competition)
            end = origin + direction * length
            segments.append(
                (
                    grain_id,
                    origin,
                    end,
                    min(float(construction["primary_width_mm"]), diameter_mm * 0.006)
                    * rng.uniform(0.8, 1.1),
                    float(construction["primary_amplitude"]) * sector_amplitude,
                    rng.uniform(0.42, 0.83),
                    0,
                    rng.uniform(-np.pi, np.pi),
                )
            )
            # Each side has its own renewal sequence: paired equal attachments
            # make regular combs even when their amplitudes are randomized.
            for side in (-1.0, 1.0):
                attachment_distance = rng.uniform(0.08, 0.17) * length
                for _ in range(int(construction["max_secondary_per_side"])):
                    if attachment_distance >= 0.94 * length:
                        break
                    attachment = origin + direction * attachment_distance
                    angle_jitter = rng.uniform(-2.0, 2.0)
                    # Projected neighbours give one coherent family frame.
                    # A two-direction family has no lateral primary axis and
                    # keeps its explicitly authored secondary projection.
                    neighbour_index = (arm_index + int(side)) % len(family_angles)
                    gap = (
                        side
                        * (family_angles[neighbour_index] - family_angles[arm_index])
                    ) % (2.0 * np.pi)
                    branch_angle = (
                        gap
                        if gap < np.pi - 1e-6
                        else np.deg2rad(float(family["secondary_angle_deg"]))
                    )
                    branch_angle += np.deg2rad(angle_jitter)
                    direction2 = (
                        np.cos(branch_angle) * direction
                        + side * np.sin(branch_angle) * normal
                    )
                    arm_length = (
                        diameter_mm
                        * float(construction["secondary_extent_diameters"])
                        * rng.uniform(0.45, 1.10)
                    )
                    # Stop before the angular bisector shared with the next
                    # primary arm. This removes interpenetrating fans without
                    # imposing an identical leaf-shaped outer envelope.
                    if branch_angle > gap / 2.0:
                        sector_limit = (
                            attachment_distance
                            * np.sin(gap / 2.0)
                            / np.sin(branch_angle - gap / 2.0)
                        )
                        arm_length = min(arm_length, sector_limit * 0.98)
                    arm_length = min(
                        arm_length,
                        _radial_reach(attachment - origin, direction2, support_radius),
                    )
                    amplitude = (
                        float(construction["secondary_amplitude"])
                        * sector_amplitude
                        * rng.uniform(0.45, 1.25)
                    )
                    segments.append(
                        (
                            grain_id,
                            attachment,
                            attachment + direction2 * arm_length,
                            min(
                                float(construction["secondary_width_mm"]),
                                diameter_mm * 0.004,
                            )
                            * rng.uniform(0.7, 1.2),
                            amplitude,
                            rng.uniform(0.18, 0.49),
                            1,
                            rng.uniform(-np.pi, np.pi),
                        )
                    )
                    # Sparse tertiary growth starts on an existing secondary.
                    # It adds interruptions and fine branching, not a separate
                    # broad carrier with a grain-wide repeated phase.
                    if rng.random() < float(construction["tertiary_probability"]):
                        for tertiary_fraction in rng.uniform(0.20, 0.78, 2):
                            junction = (
                                attachment + direction2 * arm_length * tertiary_fraction
                            )
                            tertiary_length = (
                                arm_length
                                * (1.0 - tertiary_fraction)
                                * rng.uniform(0.24, 0.50)
                            )
                            # Fine tertiary branches stay below the sidearm
                            # spacing rather than forming long crossed grids.
                            tertiary_length = min(
                                tertiary_length,
                                branch_spacing
                                * float(construction["tertiary_reach_spacings"]),
                                _radial_reach(
                                    junction - origin, direction, support_radius
                                ),
                            )
                            segments.append(
                                (
                                    grain_id,
                                    junction,
                                    junction + direction * tertiary_length,
                                    min(
                                        float(construction["tertiary_width_mm"]),
                                        diameter_mm * 0.002,
                                    )
                                    * rng.uniform(0.75, 1.15),
                                    amplitude * rng.uniform(0.32, 0.55),
                                    rng.uniform(0.12, 0.31),
                                    2,
                                    rng.uniform(-np.pi, np.pi),
                                )
                            )
                    attachment_distance += branch_spacing * rng.uniform(0.55, 1.65)
        # Keep the short-lived Python tuples and endpoint arrays bounded to a
        # block plus one grain. Their insertion order is the segment order.
        if len(segments) >= _SEGMENT_BLOCK_SIZE:
            blocks.append(np.array(segments, dtype=SEGMENT_DTYPE))
            segments.clear()
    if segments:
        blocks.append(np.array(segments, dtype=SEGMENT_DTYPE))
    # The packed blocks are already in record order. Joining their byte views
    # makes the final array immutable without a concatenate-and-copy peak.
    packed = np.frombuffer(
        b"".join(memoryview(block).cast("B") for block in blocks),
        dtype=SEGMENT_DTYPE,
    )
    return DendriteRecords(
        _readonly(azimuth),
        _readonly(tilt),
        _readonly(family_id),
        packed,
        tuple(item["name"] for item in families),
        table["version"],
        float(construction["direction_support_floor"]),
    )


def evaluate_dendrites(
    records: DendriteRecords,
    grain_ids: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    size_mm: tuple[float, float],
    diameter_mm: float,
) -> np.ndarray:
    """Return periodised branch relief, retaining the simple diagnostic API."""
    return evaluate_dendrite_fields(
        records, grain_ids, x_mm, y_mm, size_mm, diameter_mm
    )[0]


def evaluate_dendrite_fields(
    records: DendriteRecords,
    grain_ids: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    size_mm: tuple[float, float],
    diameter_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate every compact support image that overlaps its owning grain.

    The morphology is explicitly periodised: a nearest nucleus image is never
    used as a discontinuous switch for the branch signal. The second result is
    a bounded doubled-angle coherence vector, not a unit direction: cancellation
    and vanishing support both reduce its magnitude smoothly to zero.
    """
    ids, x, y = np.broadcast_arrays(grain_ids, x_mm, y_mm)
    result = np.zeros(ids.shape, dtype=np.float64)
    direction_field = np.zeros((*ids.shape, 2), dtype=np.float64)
    lx, ly = size_mm
    max_offset_x = int(np.ceil(1.25 * diameter_mm / lx)) + 1
    max_offset_y = int(np.ceil(1.25 * diameter_mm / ly)) + 1
    if max_offset_x > 4 or max_offset_y > 4:
        raise ValueError("dendrite support exceeds the bounded periodic image budget")
    segment_ids = records.segments["grain_id"]
    for grain_id in np.unique(ids):
        selected = ids == grain_id
        px = np.remainder(x[selected], lx)
        py = np.remainder(y[selected], ly)
        field = np.zeros(px.shape, dtype=np.float64)
        local_direction = np.zeros((len(px), 2), dtype=np.float64)
        direction_support = np.zeros(px.shape, dtype=np.float64)
        xmin, xmax = float(np.min(px)), float(np.max(px))
        ymin, ymax = float(np.min(py)), float(np.max(py))
        start = int(np.searchsorted(segment_ids, grain_id, side="left"))
        stop = int(np.searchsorted(segment_ids, grain_id, side="right"))
        segments = records.segments[start:stop]
        a_all = segments["start_mm"]
        b_all = segments["end_mm"]
        support_all = 3.0 * segments["width_mm"]
        offsets_x = np.arange(-max_offset_x, max_offset_x + 1)
        offsets_y = np.arange(-max_offset_y, max_offset_y + 1)
        # Cull every exact support box at once, then visit the surviving
        # (segment, X image, Y image) triples in the original scalar order.
        # This bounds scratch by object count, never by pixels times objects.
        low_x = np.minimum(a_all[:, 0], b_all[:, 0])[:, None, None]
        high_x = np.maximum(a_all[:, 0], b_all[:, 0])[:, None, None]
        low_y = np.minimum(a_all[:, 1], b_all[:, 1])[:, None, None]
        high_y = np.maximum(a_all[:, 1], b_all[:, 1])[:, None, None]
        support_grid = support_all[:, None, None]
        shift_x = (offsets_x * lx)[None, :, None]
        shift_y = (offsets_y * ly)[None, None, :]
        overlaps = (
            (high_x + shift_x + support_grid >= xmin)
            & (low_x + shift_x - support_grid <= xmax)
            & (high_y + shift_y + support_grid >= ymin)
            & (low_y + shift_y - support_grid <= ymax)
        )
        for segment_index, ix, iy in np.argwhere(overlaps):
            segment = segments[segment_index]
            a = segment["start_mm"]
            b = segment["end_mm"]
            delta = b - a
            length2 = float(np.dot(delta, delta))
            width = float(segment["width_mm"])
            support = 3.0 * width
            ox, oy = int(offsets_x[ix]), int(offsets_y[iy])
            vx = px - (a[0] + ox * lx)
            vy = py - (a[1] + oy * ly)
            fraction = (vx * delta[0] + vy * delta[1]) / length2
            cross_all = (vx * delta[1] - vy * delta[0]) / np.sqrt(length2)
            near = (fraction > 0.0) & (fraction < 1.0) & (np.abs(cross_all) < support)
            if not np.any(near):
                continue
            fx = vx[near]
            fy = vy[near]
            t = np.clip(fraction[near], 0.0, 1.0)
            cross = (fx * delta[1] - fy * delta[0]) / np.sqrt(length2)
            along = t * np.sqrt(length2)
            root = np.clip(t / 0.12, 0.0, 1.0)
            tip = np.clip((1.0 - t) / 0.38, 0.0, 1.0)
            longitudinal = root**2 * (3.0 - 2.0 * root) * tip**2 * (3.0 - 2.0 * tip)
            # Both amplitude and width taper smoothly at termination.
            normalized_cross = cross / (width * (1.0 - 0.55 * t))
            compact = np.maximum(1.0 - (normalized_cross / 3.0) ** 2, 0.0) ** 2
            ridge = np.exp(-0.5 * normalized_cross**2) * compact * longitudinal
            phase = float(segment["phase"])
            cycle = 2.0 * np.pi * along / float(segment["period_mm"])
            # A weak, phase-modulated packet along each actual branch
            # avoids the previous wide parallel Gabor comb sectors.
            carrier = 0.84 + 0.16 * np.cos(
                cycle + phase + 0.7 * np.sin(cycle / 3.1 + phase)
            )
            field[near] += float(segment["amplitude"]) * ridge * carrier
            angle = np.arctan2(delta[1], delta[0])
            directional_weight = float(segment["amplitude"]) * ridge
            local_direction[near, 0] += directional_weight * np.cos(2.0 * angle)
            local_direction[near, 1] += directional_weight * np.sin(2.0 * angle)
            direction_support[near] += directional_weight
        result[selected] = field
        direction_field[selected] = local_direction / (
            direction_support[:, None] + records.direction_support_floor
        )
    return result, direction_field
