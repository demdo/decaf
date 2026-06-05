from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json

import numpy as np
from scipy.spatial import cKDTree

from tracking.hydramarker.model.state import CameraPose, SfMState


@dataclass(slots=True)
class MarkerGeometry:
    corner_rows: int
    corner_cols: int
    square_size_mm: float

    detectable_origin_row: int = 0
    detectable_origin_col: int = 0

    id_num_cols: int | None = None
    id_base: int = 0

    def __post_init__(self) -> None:
        if self.corner_rows < 2 or self.corner_cols < 2:
            raise ValueError("Need at least 2x2 full corner grid.")

        if self.square_size_mm <= 0.0:
            raise ValueError("square_size_mm must be positive.")

        if self.id_num_cols is None:
            self.id_num_cols = self.corner_cols

    def corner_id_absolute(self, row: int, col: int) -> int:
        return int(self.id_base + int(row) * int(self.id_num_cols) + int(col))

    def corner_id_local(self, local_row: int, local_col: int) -> int:
        row = self.detectable_origin_row + int(local_row)
        col = self.detectable_origin_col + int(local_col)
        return self.corner_id_absolute(row, col)

    def row_col_from_id(self, marker_id: int) -> tuple[int, int]:
        raw = int(marker_id) - int(self.id_base)

        if raw < 0:
            raise ValueError(f"Marker id {marker_id} is below id_base={self.id_base}.")

        row = raw // int(self.id_num_cols)
        col = raw % int(self.id_num_cols)

        return int(row), int(col)

    @property
    def origin_id(self) -> int:
        return self.corner_id_local(0, 0)

    @property
    def x_axis_id(self) -> int:
        return self.corner_id_local(0, 1)

    @property
    def y_axis_id(self) -> int:
        return self.corner_id_local(1, 0)


@dataclass(slots=True)
class AlignmentResult:
    success: bool
    message: str

    origin_id: int
    x_axis_id: int
    y_axis_id: int

    detectable_origin_row: int
    detectable_origin_col: int
    resolved_id_num_cols: int

    R_marker_sfm: np.ndarray
    origin_sfm: np.ndarray

    scale: float
    median_spacing_before_scale: float

    num_points: int
    num_poses: int

    alignment_mode: str = "topology"
    num_horizontal_edges: int = 0
    num_vertical_edges: int = 0


def load_marker_geometry(json_path: Path) -> MarkerGeometry:
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if "corner_rows" in meta and "corner_cols" in meta:
        corner_rows = int(meta["corner_rows"])
        corner_cols = int(meta["corner_cols"])
    else:
        rows = int(meta["rows"])
        cols = int(meta["cols"])
        corner_rows = rows + 1
        corner_cols = cols + 1

    if "square_size_mm" in meta:
        square_size_mm = float(meta["square_size_mm"])
    elif "cell_size_mm" in meta:
        square_size_mm = float(meta["cell_size_mm"])
    elif "square_size_cm" in meta:
        square_size_mm = 10.0 * float(meta["square_size_cm"])
    else:
        raise KeyError(
            "Marker JSON must contain square_size_mm, cell_size_mm, or square_size_cm."
        )

    id_encoding = meta.get("id_encoding", {})

    id_base = int(meta.get("id_base", id_encoding.get("id_base", 0)))
    id_num_cols = int(meta.get("id_num_cols", id_encoding.get("num_cols", corner_cols)))

    detectable_origin_row = int(
        meta.get("detectable_origin_row", id_encoding.get("origin_row", 0))
    )

    detectable_origin_col = int(
        meta.get("detectable_origin_col", id_encoding.get("origin_col", 0))
    )

    return MarkerGeometry(
        corner_rows=corner_rows,
        corner_cols=corner_cols,
        square_size_mm=square_size_mm,
        detectable_origin_row=detectable_origin_row,
        detectable_origin_col=detectable_origin_col,
        id_num_cols=id_num_cols,
        id_base=id_base,
    )


def _require_marker_point(
    state: SfMState,
    marker_id: int,
) -> np.ndarray:
    marker_id = int(marker_id)

    if marker_id not in state.marker_positions:
        raise KeyError(f"Required reference marker ID {marker_id} is missing in SfM state.")

    return np.asarray(
        state.marker_positions[marker_id],
        dtype=np.float64,
    ).reshape(3)


def _normalize(v: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))

    if n <= eps:
        raise ValueError("Cannot normalize near-zero vector.")

    return v / n


def _robust_average_direction(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("Need at least one vector.")

    unit_vectors = []

    lengths = np.asarray(
        [np.linalg.norm(v) for v in vectors],
        dtype=np.float64,
    )

    median_len = float(np.median(lengths))

    if median_len <= 1e-12:
        raise ValueError("Degenerate edge vectors.")

    for v in vectors:
        v = np.asarray(v, dtype=np.float64).reshape(3)
        length = float(np.linalg.norm(v))

        if length <= 1e-12:
            continue

        if length < 0.25 * median_len or length > 4.0 * median_len:
            continue

        unit_vectors.append(v / length)

    if not unit_vectors:
        raise ValueError("All edge vectors were rejected as outliers.")

    direction = np.mean(
        np.asarray(unit_vectors, dtype=np.float64),
        axis=0,
    )

    return _normalize(direction)


def _candidate_num_cols(geometry: MarkerGeometry) -> list[int]:
    values: list[int] = []

    def add(value: int | None) -> None:
        if value is None:
            return

        value = int(value)

        if value >= 2 and value not in values:
            values.append(value)

    add(geometry.id_num_cols)
    add(geometry.corner_cols)
    add(geometry.corner_cols - 1)
    add(geometry.corner_cols + 1)
    add(geometry.corner_cols + 2)

    return values


def _row_col_from_id_with_num_cols(
    marker_id: int,
    *,
    id_base: int,
    id_num_cols: int,
) -> tuple[int, int]:
    raw = int(marker_id) - int(id_base)

    if raw < 0:
        raise ValueError(f"Marker ID {marker_id} is below id_base={id_base}.")

    row = raw // int(id_num_cols)
    col = raw % int(id_num_cols)

    return int(row), int(col)


def _score_row_col_mapping(
    row_col_to_id: dict[tuple[int, int], int],
) -> tuple[int, int, int]:
    horizontal_edges = 0
    vertical_edges = 0

    for row, col in row_col_to_id.keys():
        if (row, col + 1) in row_col_to_id:
            horizontal_edges += 1

        if (row + 1, col) in row_col_to_id:
            vertical_edges += 1

    rows = [row for row, _ in row_col_to_id.keys()]
    cols = [col for _, col in row_col_to_id.keys()]

    row_span = max(rows) - min(rows) + 1
    col_span = max(cols) - min(cols) + 1
    area = row_span * col_span

    return horizontal_edges, vertical_edges, area


def resolve_row_col_mapping_from_ids(
    marker_ids: list[int],
    geometry: MarkerGeometry,
) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], int], int]:
    marker_ids = sorted(int(mid) for mid in marker_ids)

    if not marker_ids:
        raise ValueError("Need at least one marker ID.")

    best = None
    best_score = None

    for num_cols in _candidate_num_cols(geometry):
        id_to_row_col: dict[int, tuple[int, int]] = {}
        row_col_to_id: dict[tuple[int, int], int] = {}

        valid = True

        for marker_id in marker_ids:
            try:
                row, col = _row_col_from_id_with_num_cols(
                    marker_id,
                    id_base=geometry.id_base,
                    id_num_cols=num_cols,
                )
            except ValueError:
                valid = False
                break

            if (
                row < geometry.detectable_origin_row
                or row >= geometry.detectable_origin_row + geometry.corner_rows
                or col < geometry.detectable_origin_col
                or col >= geometry.detectable_origin_col + geometry.corner_cols
            ):
                valid = False
                break

            key = (row, col)

            if key in row_col_to_id:
                valid = False
                break

            id_to_row_col[marker_id] = key
            row_col_to_id[key] = marker_id

        if not valid:
            continue

        horizontal_edges, vertical_edges, area = _score_row_col_mapping(row_col_to_id)

        score = (
            horizontal_edges + vertical_edges,
            horizontal_edges,
            vertical_edges,
            -area,
            -num_cols,
        )

        if best_score is None or score > best_score:
            best_score = score
            best = (id_to_row_col, row_col_to_id, int(num_cols))

    if best is None:
        raise RuntimeError("Could not resolve row/col mapping from marker IDs.")

    return best


def _build_row_col_lookup(
    state: SfMState,
    geometry: MarkerGeometry,
) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], int], int]:
    return resolve_row_col_mapping_from_ids(
        marker_ids=sorted(int(mid) for mid in state.marker_positions.keys()),
        geometry=geometry,
    )


def collect_topology_neighbor_edges(
    state: SfMState,
    geometry: MarkerGeometry,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    _, row_col_to_id, resolved_num_cols = _build_row_col_lookup(
        state,
        geometry,
    )

    horizontal_edges: list[np.ndarray] = []
    vertical_edges: list[np.ndarray] = []

    for (row, col), marker_id in row_col_to_id.items():
        p = np.asarray(
            state.marker_positions[int(marker_id)],
            dtype=np.float64,
        ).reshape(3)

        right_id = row_col_to_id.get((row, col + 1))
        if right_id is not None:
            p_right = np.asarray(
                state.marker_positions[int(right_id)],
                dtype=np.float64,
            ).reshape(3)

            horizontal_edges.append(p_right - p)

        down_id = row_col_to_id.get((row + 1, col))
        if down_id is not None:
            p_down = np.asarray(
                state.marker_positions[int(down_id)],
                dtype=np.float64,
            ).reshape(3)

            vertical_edges.append(p_down - p)

    return horizontal_edges, vertical_edges, resolved_num_cols


def resolve_reference_ids_from_state(
    state: SfMState,
    geometry: MarkerGeometry,
) -> tuple[int, int, int, int, bool]:
    _, row_col_to_id, resolved_num_cols = _build_row_col_lookup(
        state,
        geometry,
    )

    entries = sorted(row_col_to_id.items(), key=lambda item: (item[0][0], item[0][1]))

    for (row, col), origin_id in entries:
        x_axis_id = row_col_to_id.get((row, col + 1))
        y_axis_id = row_col_to_id.get((row + 1, col))

        if x_axis_id is None or y_axis_id is None:
            continue

        return (
            int(origin_id),
            int(x_axis_id),
            int(y_axis_id),
            int(resolved_num_cols),
            True,
        )

    declared_origin_id = geometry.origin_id
    declared_x_axis_id = geometry.x_axis_id
    declared_y_axis_id = geometry.y_axis_id
    declared_num_cols = int(geometry.id_num_cols)

    if (
        declared_origin_id in state.marker_positions
        and declared_x_axis_id in state.marker_positions
        and declared_y_axis_id in state.marker_positions
    ):
        return (
            declared_origin_id,
            declared_x_axis_id,
            declared_y_axis_id,
            declared_num_cols,
            False,
        )

    raise KeyError(
        "Could not find a valid reference triplet in SfM state."
    )


def build_marker_frame_from_topology(
    state: SfMState,
    geometry: MarkerGeometry,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, bool, int, int]:
    (
        origin_id,
        x_axis_id,
        y_axis_id,
        resolved_num_cols,
        used_fallback,
    ) = resolve_reference_ids_from_state(
        state,
        geometry,
    )

    origin_sfm = _require_marker_point(
        state,
        origin_id,
    )

    horizontal_edges, vertical_edges, edge_resolved_num_cols = collect_topology_neighbor_edges(
        state,
        geometry,
    )

    resolved_num_cols = edge_resolved_num_cols

    if len(horizontal_edges) < 2 or len(vertical_edges) < 2:
        raise ValueError(
            "Not enough topology neighbor edges for robust alignment. "
            f"horizontal={len(horizontal_edges)}, vertical={len(vertical_edges)}"
        )

    x = _robust_average_direction(horizontal_edges)

    y_raw = _robust_average_direction(vertical_edges)
    y_raw = y_raw - float(np.dot(y_raw, x)) * x
    y = _normalize(y_raw)

    z = np.cross(x, y)
    z = _normalize(z)

    y = np.cross(z, x)
    y = _normalize(y)

    R_marker_sfm = np.vstack([x, y, z])

    return (
        R_marker_sfm,
        origin_sfm,
        origin_id,
        x_axis_id,
        y_axis_id,
        resolved_num_cols,
        used_fallback,
        len(horizontal_edges),
        len(vertical_edges),
    )


def build_marker_frame_from_three_points(
    state: SfMState,
    geometry: MarkerGeometry,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, bool, int, int]:
    (
        origin_id,
        x_axis_id,
        y_axis_id,
        resolved_num_cols,
        used_fallback,
    ) = resolve_reference_ids_from_state(
        state,
        geometry,
    )

    p0 = _require_marker_point(state, origin_id)
    px = _require_marker_point(state, x_axis_id)
    py = _require_marker_point(state, y_axis_id)

    x = _normalize(px - p0)

    y_raw = py - p0
    y_raw = y_raw - float(np.dot(y_raw, x)) * x
    y = _normalize(y_raw)

    z = np.cross(x, y)
    z = _normalize(z)

    y = np.cross(z, x)
    y = _normalize(y)

    R_marker_sfm = np.vstack([x, y, z])

    return (
        R_marker_sfm,
        p0,
        origin_id,
        x_axis_id,
        y_axis_id,
        resolved_num_cols,
        used_fallback,
        1,
        1,
    )


def build_marker_frame_from_state(
    state: SfMState,
    geometry: MarkerGeometry,
    *,
    alignment_mode: str = "topology",
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, bool, int, int]:
    if alignment_mode == "topology":
        return build_marker_frame_from_topology(
            state,
            geometry,
        )

    if alignment_mode in {"reference", "three_points", "3point"}:
        return build_marker_frame_from_three_points(
            state,
            geometry,
        )

    raise ValueError(
        f"Unknown alignment_mode {alignment_mode!r}. "
        "Use 'topology' or 'reference'."
    )


def transform_points_to_marker_frame(
    marker_positions: dict[int, np.ndarray],
    R_marker_sfm: np.ndarray,
    origin_sfm: np.ndarray,
) -> dict[int, np.ndarray]:
    transformed: dict[int, np.ndarray] = {}

    for marker_id, point in marker_positions.items():
        p = np.asarray(point, dtype=np.float64).reshape(3)
        transformed[int(marker_id)] = R_marker_sfm @ (p - origin_sfm)

    return transformed


def estimate_metric_scale_from_reference_edges(
    marker_positions_marker_frame: dict[int, np.ndarray],
    *,
    origin_id: int,
    x_axis_id: int,
    y_axis_id: int,
    expected_spacing_mm: float,
) -> tuple[float, float]:
    p0 = np.asarray(marker_positions_marker_frame[origin_id], dtype=np.float64).reshape(3)
    px = np.asarray(marker_positions_marker_frame[x_axis_id], dtype=np.float64).reshape(3)
    py = np.asarray(marker_positions_marker_frame[y_axis_id], dtype=np.float64).reshape(3)

    dx = float(np.linalg.norm(px - p0))
    dy = float(np.linalg.norm(py - p0))

    values = np.asarray([dx, dy], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 1e-12)]

    if values.size == 0:
        raise ValueError("Could not estimate scale from reference edges.")

    median_spacing = float(np.median(values))
    scale = float(expected_spacing_mm / median_spacing)

    return scale, median_spacing


def estimate_metric_scale_from_topology_edges(
    marker_positions_marker_frame: dict[int, np.ndarray],
    geometry: MarkerGeometry,
    *,
    expected_spacing_mm: float,
) -> tuple[float, float]:
    marker_ids = sorted(int(mid) for mid in marker_positions_marker_frame.keys())

    _, row_col_to_id, _ = resolve_row_col_mapping_from_ids(
        marker_ids=marker_ids,
        geometry=geometry,
    )

    id_to_point = {
        int(marker_id): np.asarray(point, dtype=np.float64).reshape(3)
        for marker_id, point in marker_positions_marker_frame.items()
    }

    distances: list[float] = []

    for (row, col), marker_id in row_col_to_id.items():
        p = id_to_point[int(marker_id)]

        right_id = row_col_to_id.get((row, col + 1))
        if right_id is not None:
            distances.append(
                float(np.linalg.norm(id_to_point[int(right_id)] - p))
            )

        down_id = row_col_to_id.get((row + 1, col))
        if down_id is not None:
            distances.append(
                float(np.linalg.norm(id_to_point[int(down_id)] - p))
            )

    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 1e-12)]

    if values.size == 0:
        raise ValueError("Could not estimate scale from topology edges.")

    median_spacing = float(np.median(values))
    scale = float(expected_spacing_mm / median_spacing)

    return scale, median_spacing


def estimate_metric_scale_from_neighbors(
    marker_positions_marker_frame: dict[int, np.ndarray],
    *,
    expected_spacing_mm: float,
) -> tuple[float, float]:
    if len(marker_positions_marker_frame) < 2:
        raise ValueError("Need at least two marker points for scale estimation.")

    points = np.asarray(
        [
            marker_positions_marker_frame[mid]
            for mid in sorted(marker_positions_marker_frame.keys())
        ],
        dtype=np.float64,
    ).reshape(-1, 3)

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)

    nearest = dists[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-12)]

    if nearest.size == 0:
        raise ValueError("Could not estimate nearest-neighbor spacing.")

    median_spacing = float(np.median(nearest))
    scale = float(expected_spacing_mm / median_spacing)

    return scale, median_spacing


def transform_pose_to_marker_frame(
    pose: CameraPose,
    *,
    R_marker_sfm: np.ndarray,
    origin_sfm: np.ndarray,
    scale: float,
) -> CameraPose:
    R_old = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
    t_old = np.asarray(pose.t, dtype=np.float64).reshape(3)

    R_new = R_old @ R_marker_sfm.T
    t_new = scale * (R_old @ origin_sfm + t_old)

    return CameraPose(
        R=R_new,
        t=t_new,
    )


def apply_alignment_to_state_inplace(
    state: SfMState,
    *,
    R_marker_sfm: np.ndarray,
    origin_sfm: np.ndarray,
    scale: float,
) -> None:
    new_marker_positions: dict[int, np.ndarray] = {}

    for marker_id, point in state.marker_positions.items():
        p = np.asarray(point, dtype=np.float64).reshape(3)
        p_marker = R_marker_sfm @ (p - origin_sfm)
        new_marker_positions[int(marker_id)] = scale * p_marker

    new_poses: dict[int, CameraPose] = {}

    for frame_id, pose in state.poses.items():
        new_poses[int(frame_id)] = transform_pose_to_marker_frame(
            pose,
            R_marker_sfm=R_marker_sfm,
            origin_sfm=origin_sfm,
            scale=scale,
        )

    state.marker_positions.clear()
    state.marker_positions.update(new_marker_positions)

    state.poses.clear()
    state.poses.update(new_poses)


def apply_metric_scale_to_state_inplace(
    state: SfMState,
    scale: float,
) -> None:
    scale = float(scale)

    for marker_id, point in list(state.marker_positions.items()):
        state.marker_positions[int(marker_id)] = (
            scale * np.asarray(point, dtype=np.float64).reshape(3)
        )

    for frame_id, pose in list(state.poses.items()):
        state.poses[int(frame_id)] = CameraPose(
            R=pose.R,
            t=scale * np.asarray(pose.t, dtype=np.float64).reshape(3),
        )


def regularize_marker_columns_xz_inplace(
    state: SfMState,
    marker_json_path: Path,
) -> dict[int, dict[str, float]]:
    geometry = load_marker_geometry(marker_json_path)
    id_to_row_col, _, _ = resolve_row_col_mapping_from_ids(
        marker_ids=sorted(int(mid) for mid in state.marker_positions.keys()),
        geometry=geometry,
    )

    ids_by_col: dict[int, list[int]] = {}

    for marker_id, (_, col) in id_to_row_col.items():
        ids_by_col.setdefault(int(col), []).append(int(marker_id))

    stats: dict[int, dict[str, float]] = {}

    for col, marker_ids in sorted(ids_by_col.items()):
        if len(marker_ids) < 2:
            continue

        xs = np.asarray(
            [
                np.asarray(state.marker_positions[mid], dtype=np.float64).reshape(3)[0]
                for mid in marker_ids
            ],
            dtype=np.float64,
        )
        zs = np.asarray(
            [
                np.asarray(state.marker_positions[mid], dtype=np.float64).reshape(3)[2]
                for mid in marker_ids
            ],
            dtype=np.float64,
        )

        x_target = float(np.median(xs))
        z_target = float(np.median(zs))

        x_range_before = float(np.max(xs) - np.min(xs))
        z_range_before = float(np.max(zs) - np.min(zs))
        x_std_before = float(np.std(xs))
        z_std_before = float(np.std(zs))

        for marker_id in marker_ids:
            p = np.asarray(
                state.marker_positions[int(marker_id)],
                dtype=np.float64,
            ).reshape(3).copy()
            p[0] = x_target
            p[2] = z_target
            state.marker_positions[int(marker_id)] = p

        stats[int(col)] = {
            "count": float(len(marker_ids)),
            "x_target": x_target,
            "z_target": z_target,
            "x_range_before": x_range_before,
            "z_range_before": z_range_before,
            "x_std_before": x_std_before,
            "z_std_before": z_std_before,
        }

    return stats


def regularize_marker_columns_z_inplace(
    state: SfMState,
    marker_json_path: Path,
) -> dict[int, dict[str, float]]:
    geometry = load_marker_geometry(marker_json_path)
    id_to_row_col, _, _ = resolve_row_col_mapping_from_ids(
        marker_ids=sorted(int(mid) for mid in state.marker_positions.keys()),
        geometry=geometry,
    )

    ids_by_col: dict[int, list[int]] = {}

    for marker_id, (_, col) in id_to_row_col.items():
        ids_by_col.setdefault(int(col), []).append(int(marker_id))

    stats: dict[int, dict[str, float]] = {}

    for col, marker_ids in sorted(ids_by_col.items()):
        if len(marker_ids) < 2:
            continue

        xs = np.asarray(
            [
                np.asarray(state.marker_positions[mid], dtype=np.float64).reshape(3)[0]
                for mid in marker_ids
            ],
            dtype=np.float64,
        )
        zs = np.asarray(
            [
                np.asarray(state.marker_positions[mid], dtype=np.float64).reshape(3)[2]
                for mid in marker_ids
            ],
            dtype=np.float64,
        )

        x_target = float(np.median(xs))
        z_target = float(np.median(zs))

        x_range_before = float(np.max(xs) - np.min(xs))
        z_range_before = float(np.max(zs) - np.min(zs))
        x_std_before = float(np.std(xs))
        z_std_before = float(np.std(zs))

        for marker_id in marker_ids:
            p = np.asarray(
                state.marker_positions[int(marker_id)],
                dtype=np.float64,
            ).reshape(3).copy()
            p[2] = z_target
            state.marker_positions[int(marker_id)] = p

        stats[int(col)] = {
            "mode": "z_only",
            "count": float(len(marker_ids)),
            "x_target": x_target,
            "z_target": z_target,
            "x_range_before": x_range_before,
            "z_range_before": z_range_before,
            "x_std_before": x_std_before,
            "z_std_before": z_std_before,
        }

    return stats


def align_state_to_marker_frame_inplace(
    state: SfMState,
    marker_json_path: Path,
    *,
    scale_metric: bool = True,
    scale_mode: str = "topology",
    alignment_mode: str = "topology",
) -> AlignmentResult:
    geometry = load_marker_geometry(marker_json_path)

    (
        R_marker_sfm,
        origin_sfm,
        origin_id,
        x_axis_id,
        y_axis_id,
        resolved_num_cols,
        used_fallback,
        num_horizontal_edges,
        num_vertical_edges,
    ) = build_marker_frame_from_state(
        state,
        geometry,
        alignment_mode=alignment_mode,
    )

    unscaled_points = transform_points_to_marker_frame(
        state.marker_positions,
        R_marker_sfm,
        origin_sfm,
    )

    if scale_metric:
        if scale_mode == "topology":
            scale, median_spacing = estimate_metric_scale_from_topology_edges(
                unscaled_points,
                geometry,
                expected_spacing_mm=geometry.square_size_mm,
            )
        elif scale_mode == "reference":
            scale, median_spacing = estimate_metric_scale_from_reference_edges(
                unscaled_points,
                origin_id=origin_id,
                x_axis_id=x_axis_id,
                y_axis_id=y_axis_id,
                expected_spacing_mm=geometry.square_size_mm,
            )
        elif scale_mode == "neighbors":
            scale, median_spacing = estimate_metric_scale_from_neighbors(
                unscaled_points,
                expected_spacing_mm=geometry.square_size_mm,
            )
        else:
            raise ValueError(
                f"Unknown scale_mode {scale_mode!r}. "
                "Use 'topology', 'neighbors' or 'reference'."
            )
    else:
        scale = 1.0
        median_spacing = float("nan")

    apply_alignment_to_state_inplace(
        state,
        R_marker_sfm=R_marker_sfm,
        origin_sfm=origin_sfm,
        scale=scale,
    )

    if scale_metric and scale_mode == "topology":
        correction, corrected_median_spacing = estimate_metric_scale_from_topology_edges(
            state.marker_positions,
            geometry,
            expected_spacing_mm=geometry.square_size_mm,
        )

        if np.isfinite(correction) and correction > 1e-12:
            apply_metric_scale_to_state_inplace(state, correction)
            scale *= correction
            median_spacing = corrected_median_spacing

    message = (
        "SfM state aligned to marker frame using automatically resolved "
        "topology-aware marker axes and topology-based metric scale."
    )

    return AlignmentResult(
        success=True,
        message=message,
        origin_id=origin_id,
        x_axis_id=x_axis_id,
        y_axis_id=y_axis_id,
        detectable_origin_row=geometry.detectable_origin_row,
        detectable_origin_col=geometry.detectable_origin_col,
        resolved_id_num_cols=resolved_num_cols,
        R_marker_sfm=R_marker_sfm,
        origin_sfm=origin_sfm,
        scale=scale,
        median_spacing_before_scale=median_spacing,
        num_points=len(state.marker_positions),
        num_poses=len(state.poses),
        alignment_mode=alignment_mode,
        num_horizontal_edges=num_horizontal_edges,
        num_vertical_edges=num_vertical_edges,
    )


def aligned_copy_of_state(
    state: SfMState,
    marker_json_path: Path,
    *,
    scale_metric: bool = True,
    scale_mode: str = "topology",
    alignment_mode: str = "topology",
) -> tuple[SfMState, AlignmentResult]:
    state_copy = copy.deepcopy(state)

    result = align_state_to_marker_frame_inplace(
        state_copy,
        marker_json_path,
        scale_metric=scale_metric,
        scale_mode=scale_mode,
        alignment_mode=alignment_mode,
    )

    return state_copy, result
