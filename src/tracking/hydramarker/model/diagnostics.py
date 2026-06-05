from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from collections import defaultdict
import json

import numpy as np


@dataclass(slots=True)
class ObservationErrorRecord:
    frame_id: int
    marker_id: int
    error_px: float
    is_outlier: bool = False


@dataclass(slots=True)
class MarkerErrorStats:
    marker_id: int
    observation_count: int
    outlier_count: int
    outlier_ratio: float
    mean_error_px: float
    median_error_px: float
    std_error_px: float
    max_error_px: float
    row: Optional[int] = None
    col: Optional[int] = None
    local_row: Optional[int] = None
    local_col: Optional[int] = None
    distance_to_center: Optional[float] = None
    border_distance: Optional[int] = None
    suggested_weight: float = 1.0


@dataclass(slots=True)
class FrameErrorStats:
    frame_id: int
    observation_count: int
    outlier_count: int
    outlier_ratio: float
    mean_error_px: float
    median_error_px: float
    std_error_px: float
    max_error_px: float
    suggested_weight: float = 1.0


@dataclass(slots=True)
class MarkerGeometryStats:
    marker_id: int
    x_mm: float
    y_mm: float
    z_mm: float
    abs_z_mm: float
    row: Optional[int] = None
    col: Optional[int] = None
    distance_to_center: Optional[float] = None
    border_distance: Optional[int] = None


def _safe_weight_from_error(
    value: float,
    *,
    good: float,
    bad: float,
    min_weight: float,
) -> float:
    if not np.isfinite(value):
        return float(min_weight)

    if value <= good:
        return 1.0

    if value >= bad:
        return float(min_weight)

    t = (float(value) - float(good)) / (float(bad) - float(good))
    return float((1.0 - t) + t * float(min_weight))


def _as_observation_records(
    errors: Sequence,
    *,
    ignored_observations: Optional[set[tuple[int, int]]] = None,
) -> list[ObservationErrorRecord]:
    ignored = ignored_observations or set()
    records: list[ObservationErrorRecord] = []

    for e in errors:
        frame_id = int(e.frame_id)
        marker_id = int(e.marker_id)

        records.append(
            ObservationErrorRecord(
                frame_id=frame_id,
                marker_id=marker_id,
                error_px=float(e.error_px),
                is_outlier=(frame_id, marker_id) in ignored,
            )
        )

    records.sort(key=lambda r: r.error_px, reverse=True)
    return records


def _marker_id_to_row_col_safe(
    marker_id: int,
    *,
    id_base: Optional[int],
    id_num_cols: Optional[int],
    origin_row: int = 0,
    origin_col: int = 0,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    if id_base is None or id_num_cols is None:
        return None, None, None, None

    if int(id_num_cols) <= 0:
        return None, None, None, None

    raw = int(marker_id) - int(id_base)

    if raw < 0:
        return None, None, None, None

    row = raw // int(id_num_cols)
    col = raw % int(id_num_cols)

    if col < int(origin_col):
        row -= 1
        col += int(id_num_cols)

    local_row = int(row) - int(origin_row)
    local_col = int(col) - int(origin_col)

    return int(row), int(col), int(local_row), int(local_col)


def _border_distance_safe(
    row: Optional[int],
    col: Optional[int],
    *,
    min_row: Optional[int],
    max_row: Optional[int],
    min_col: Optional[int],
    max_col: Optional[int],
) -> Optional[int]:
    if row is None or col is None:
        return None

    if min_row is None or max_row is None or min_col is None or max_col is None:
        return None

    return int(
        min(
            int(row) - int(min_row),
            int(max_row) - int(row),
            int(col) - int(min_col),
            int(max_col) - int(col),
        )
    )


def summarize_errors_by_marker_id(
    errors: Sequence,
    *,
    ignored_observations: Optional[set[tuple[int, int]]] = None,
    id_base: Optional[int] = None,
    id_num_cols: Optional[int] = None,
    origin_row: int = 0,
    origin_col: int = 0,
    marker_center_row_col: Optional[tuple[float, float]] = None,
    detectable_row_range: Optional[tuple[int, int]] = None,
    detectable_col_range: Optional[tuple[int, int]] = None,
) -> list[MarkerErrorStats]:
    records = _as_observation_records(
        errors,
        ignored_observations=ignored_observations,
    )

    grouped: dict[int, list[ObservationErrorRecord]] = {}

    for record in records:
        grouped.setdefault(int(record.marker_id), []).append(record)

    if detectable_row_range is None:
        min_row = max_row = None
    else:
        min_row, max_row = detectable_row_range

    if detectable_col_range is None:
        min_col = max_col = None
    else:
        min_col, max_col = detectable_col_range

    stats: list[MarkerErrorStats] = []

    for marker_id, marker_records in grouped.items():
        values = np.asarray(
            [r.error_px for r in marker_records if np.isfinite(r.error_px)],
            dtype=np.float64,
        )

        if values.size == 0:
            continue

        outlier_count = int(sum(1 for r in marker_records if r.is_outlier))
        observation_count = int(len(marker_records))

        row, col, local_row, local_col = _marker_id_to_row_col_safe(
            marker_id,
            id_base=id_base,
            id_num_cols=id_num_cols,
            origin_row=origin_row,
            origin_col=origin_col,
        )

        distance_to_center = None

        if row is not None and col is not None and marker_center_row_col is not None:
            cr, cc = marker_center_row_col
            distance_to_center = float(
                np.sqrt((float(row) - cr) ** 2 + (float(col) - cc) ** 2)
            )

        border_distance = _border_distance_safe(
            row,
            col,
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
        )

        mean_error = float(np.mean(values))

        suggested_weight = _safe_weight_from_error(
            mean_error,
            good=0.35,
            bad=0.8,
            min_weight=0.35,
        )

        stats.append(
            MarkerErrorStats(
                marker_id=int(marker_id),
                observation_count=observation_count,
                outlier_count=outlier_count,
                outlier_ratio=float(outlier_count / max(1, observation_count)),
                mean_error_px=mean_error,
                median_error_px=float(np.median(values)),
                std_error_px=float(np.std(values)),
                max_error_px=float(np.max(values)),
                row=row,
                col=col,
                local_row=local_row,
                local_col=local_col,
                distance_to_center=distance_to_center,
                border_distance=border_distance,
                suggested_weight=suggested_weight,
            )
        )

    stats.sort(
        key=lambda s: (
            s.mean_error_px,
            s.max_error_px,
            s.outlier_ratio,
        ),
        reverse=True,
    )

    return stats


def summarize_errors_by_frame_id(
    errors: Sequence,
    *,
    ignored_observations: Optional[set[tuple[int, int]]] = None,
) -> list[FrameErrorStats]:
    records = _as_observation_records(
        errors,
        ignored_observations=ignored_observations,
    )

    grouped: dict[int, list[ObservationErrorRecord]] = {}

    for record in records:
        grouped.setdefault(int(record.frame_id), []).append(record)

    stats: list[FrameErrorStats] = []

    for frame_id, frame_records in grouped.items():
        values = np.asarray(
            [r.error_px for r in frame_records if np.isfinite(r.error_px)],
            dtype=np.float64,
        )

        if values.size == 0:
            continue

        outlier_count = int(sum(1 for r in frame_records if r.is_outlier))
        observation_count = int(len(frame_records))
        mean_error = float(np.mean(values))

        suggested_weight = _safe_weight_from_error(
            mean_error,
            good=0.35,
            bad=0.9,
            min_weight=0.25,
        )

        stats.append(
            FrameErrorStats(
                frame_id=int(frame_id),
                observation_count=observation_count,
                outlier_count=outlier_count,
                outlier_ratio=float(outlier_count / max(1, observation_count)),
                mean_error_px=mean_error,
                median_error_px=float(np.median(values)),
                std_error_px=float(np.std(values)),
                max_error_px=float(np.max(values)),
                suggested_weight=suggested_weight,
            )
        )

    stats.sort(
        key=lambda s: (
            s.mean_error_px,
            s.max_error_px,
            s.outlier_ratio,
        ),
        reverse=True,
    )

    return stats


def summarize_marker_geometry(
    marker_geometry_json_path: Path,
) -> list[MarkerGeometryStats]:
    marker_geometry_json_path = Path(marker_geometry_json_path)

    with marker_geometry_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    corners = meta.get("corners", [])

    rows = [int(c["row"]) for c in corners if "row" in c]
    cols = [int(c["col"]) for c in corners if "col" in c]

    min_row = min(rows) if rows else None
    max_row = max(rows) if rows else None
    min_col = min(cols) if cols else None
    max_col = max(cols) if cols else None

    center_row = (
        0.5 * float(min_row + max_row)
        if min_row is not None and max_row is not None
        else None
    )

    center_col = (
        0.5 * float(min_col + max_col)
        if min_col is not None and max_col is not None
        else None
    )

    stats: list[MarkerGeometryStats] = []

    for c in corners:
        xyz = np.asarray(c["xyz_mm"], dtype=np.float64).reshape(3)

        row = int(c["row"]) if "row" in c else None
        col = int(c["col"]) if "col" in c else None

        distance_to_center = None

        if (
            row is not None
            and col is not None
            and center_row is not None
            and center_col is not None
        ):
            distance_to_center = float(
                np.sqrt(
                    (float(row) - center_row) ** 2
                    + (float(col) - center_col) ** 2
                )
            )

        border_distance = _border_distance_safe(
            row,
            col,
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
        )

        stats.append(
            MarkerGeometryStats(
                marker_id=int(c["id"]),
                x_mm=float(xyz[0]),
                y_mm=float(xyz[1]),
                z_mm=float(xyz[2]),
                abs_z_mm=float(abs(xyz[2])),
                row=row,
                col=col,
                distance_to_center=distance_to_center,
                border_distance=border_distance,
            )
        )

    stats.sort(key=lambda s: s.abs_z_mm, reverse=True)
    return stats


def _format_stats(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return "n=0"

    return (
        f"n={arr.size} "
        f"min={float(np.min(arr)):.4f} "
        f"median={float(np.median(arr)):.4f} "
        f"mean={float(np.mean(arr)):.4f} "
        f"max={float(np.max(arr)):.4f} "
        f"std={float(np.std(arr)):.4f}"
    )


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(b, dtype=np.float64).reshape(3)
            - np.asarray(a, dtype=np.float64).reshape(3)
        )
    )


def _marker_observation_counts(frames: Sequence) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)

    for frame in frames:
        for marker_id in frame.observations:
            counts[int(marker_id)] += 1

    return dict(counts)


def _error_stats_by_marker(errors: Sequence) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)

    for err in errors:
        value = float(getattr(err, "error_px", float("nan")))
        if np.isfinite(value):
            grouped[int(err.marker_id)].append(value)

    stats: dict[int, dict[str, float]] = {}

    for marker_id, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        stats[int(marker_id)] = {
            "count": float(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    return stats


def write_sfm_geometry_diagnostics(
    *,
    output_path: Path,
    state,
    marker_json_path: Path,
    observation_errors: Optional[Sequence] = None,
    triangulation_results: Optional[Sequence] = None,
    column_regularization_stats: Optional[dict[int, dict[str, float]]] = None,
    expected_corner_rows: Optional[int] = None,
    expected_corner_cols: Optional[int] = None,
    expected_spacing_mm: Optional[float] = None,
) -> Path:
    output_path = Path(output_path)
    marker_json_path = Path(marker_json_path)

    with marker_json_path.open("r", encoding="utf-8") as f:
        marker_meta = json.load(f)

    if expected_corner_rows is None:
        expected_corner_rows = int(
            marker_meta.get("corner_rows", int(marker_meta.get("rows", 0)) + 1)
        )

    if expected_corner_cols is None:
        expected_corner_cols = int(
            marker_meta.get("corner_cols", int(marker_meta.get("cols", 0)) + 1)
        )

    if expected_spacing_mm is None:
        expected_spacing_mm = float(
            marker_meta.get(
                "square_size_mm",
                10.0 * float(marker_meta.get("square_size_cm", 0.0)),
            )
        )

    id_num_cols = int(
        marker_meta.get(
            "id_num_cols",
            marker_meta.get("id_encoding", {}).get("num_cols", expected_corner_cols),
        )
    )
    id_base = int(
        marker_meta.get(
            "id_base",
            marker_meta.get("id_encoding", {}).get("id_base", 0),
        )
    )

    points = {
        int(marker_id): np.asarray(point, dtype=np.float64).reshape(3)
        for marker_id, point in state.marker_positions.items()
    }

    id_to_row_col: dict[int, tuple[int, int]] = {}
    row_col_to_id: dict[tuple[int, int], int] = {}

    for marker_id in sorted(points):
        raw = int(marker_id) - id_base
        if raw < 0:
            continue

        row = int(raw // id_num_cols)
        col = int(raw % id_num_cols)
        id_to_row_col[int(marker_id)] = (row, col)
        row_col_to_id[(row, col)] = int(marker_id)

    horizontal_edges: list[tuple[int, int, float]] = []
    vertical_edges: list[tuple[int, int, float]] = []

    for (row, col), marker_id in sorted(row_col_to_id.items()):
        right_id = row_col_to_id.get((row, col + 1))
        if right_id is not None:
            horizontal_edges.append(
                (
                    marker_id,
                    right_id,
                    _distance(points[marker_id], points[right_id]),
                )
            )

        down_id = row_col_to_id.get((row + 1, col))
        if down_id is not None:
            vertical_edges.append(
                (
                    marker_id,
                    down_id,
                    _distance(points[marker_id], points[down_id]),
                )
            )

    all_edge_distances = [d for _, _, d in horizontal_edges + vertical_edges]

    observation_counts = _marker_observation_counts(state.frames)
    error_stats = _error_stats_by_marker(observation_errors or [])

    triangulation_by_marker: dict[int, object] = {}
    for result in triangulation_results or []:
        triangulation_by_marker[int(result.marker_id)] = result

    column_regularization_stats = column_regularization_stats or {}

    expected_ids = {
        id_base + row * id_num_cols + col
        for row in range(int(expected_corner_rows))
        for col in range(int(expected_corner_cols))
    }
    missing_ids = sorted(expected_ids - set(points))
    extra_ids = sorted(set(points) - expected_ids)

    lines: list[str] = []
    lines.append("HydraMarker SfM Geometry Diagnostics")
    lines.append("=" * 44)
    lines.append(f"marker_json: {marker_json_path}")
    lines.append(f"corner_count: {len(points)}")
    lines.append(f"expected_corner_count: {expected_corner_rows * expected_corner_cols}")
    lines.append(f"expected_grid: rows={expected_corner_rows}, cols={expected_corner_cols}")
    lines.append(f"id_base: {id_base}")
    lines.append(f"id_num_cols: {id_num_cols}")
    lines.append(f"expected_spacing_mm: {float(expected_spacing_mm):.6f}")
    lines.append(f"missing_ids: {missing_ids}")
    lines.append(f"extra_ids: {extra_ids}")
    lines.append("")

    if points:
        arr = np.asarray(list(points.values()), dtype=np.float64)
        lines.append("Bounding Box [mm]")
        lines.append(f"  min_xyz: {np.min(arr, axis=0).round(6).tolist()}")
        lines.append(f"  max_xyz: {np.max(arr, axis=0).round(6).tolist()}")
        lines.append(f"  span_xyz: {(np.max(arr, axis=0) - np.min(arr, axis=0)).round(6).tolist()}")
        lines.append("")

    lines.append("Neighbor Distance Summary [mm]")
    lines.append(f"  horizontal row,col->row,col+1: {_format_stats([d for _, _, d in horizontal_edges])}")
    lines.append(f"  vertical row,col->row+1,col:   {_format_stats([d for _, _, d in vertical_edges])}")
    lines.append(f"  all topology edges:             {_format_stats(all_edge_distances)}")
    if all_edge_distances:
        deviations = [d - float(expected_spacing_mm) for d in all_edge_distances]
        lines.append(f"  all edge deviation from expected: {_format_stats(deviations)}")
    lines.append("")

    lines.append("Z Stability By Column [mm]")
    for col in range(int(expected_corner_cols)):
        ids = [
            row_col_to_id[(row, col)]
            for row in range(int(expected_corner_rows))
            if (row, col) in row_col_to_id
        ]
        zs = [float(points[mid][2]) for mid in ids]
        if not zs:
            lines.append(f"  col={col}: n=0")
            continue

        arr_z = np.asarray(zs, dtype=np.float64)
        lines.append(
            "  "
            f"col={col}: "
            f"ids={ids} "
            f"z_min={float(np.min(arr_z)):.4f} "
            f"z_median={float(np.median(arr_z)):.4f} "
            f"z_mean={float(np.mean(arr_z)):.4f} "
            f"z_max={float(np.max(arr_z)):.4f} "
            f"z_range={float(np.max(arr_z)-np.min(arr_z)):.4f} "
            f"z_std={float(np.std(arr_z)):.4f}"
        )
    lines.append("")

    if column_regularization_stats:
        first_stats = next(iter(column_regularization_stats.values()))
        if str(first_stats.get("mode", "")).lower() == "z_only":
            lines.append("Column Z Regularization Applied")
        else:
            lines.append("Column X/Z Regularization Applied")
        for col in sorted(column_regularization_stats):
            s = column_regularization_stats[col]
            lines.append(
                "  "
                f"col={col}: "
                f"n={int(s.get('count', 0))} "
                f"x_target={float(s.get('x_target', float('nan'))):.4f} "
                f"z_target={float(s.get('z_target', float('nan'))):.4f} "
                f"x_range_before={float(s.get('x_range_before', float('nan'))):.4f} "
                f"z_range_before={float(s.get('z_range_before', float('nan'))):.4f} "
                f"x_std_before={float(s.get('x_std_before', float('nan'))):.4f} "
                f"z_std_before={float(s.get('z_std_before', float('nan'))):.4f}"
            )
        lines.append("")

    lines.append("Z Stability By Row [mm]")
    for row in range(int(expected_corner_rows)):
        ids = [
            row_col_to_id[(row, col)]
            for col in range(int(expected_corner_cols))
            if (row, col) in row_col_to_id
        ]
        zs = [float(points[mid][2]) for mid in ids]
        if not zs:
            lines.append(f"  row={row}: n=0")
            continue

        arr_z = np.asarray(zs, dtype=np.float64)
        lines.append(
            "  "
            f"row={row}: "
            f"ids={ids} "
            f"z_min={float(np.min(arr_z)):.4f} "
            f"z_median={float(np.median(arr_z)):.4f} "
            f"z_mean={float(np.mean(arr_z)):.4f} "
            f"z_max={float(np.max(arr_z)):.4f} "
            f"z_range={float(np.max(arr_z)-np.min(arr_z)):.4f} "
            f"z_std={float(np.std(arr_z)):.4f}"
        )
    lines.append("")

    lines.append("Worst Topology Edge Deviations [mm]")
    edge_rows = []
    for label, edges in (("H", horizontal_edges), ("V", vertical_edges)):
        for id0, id1, distance in edges:
            edge_rows.append((abs(distance - float(expected_spacing_mm)), label, id0, id1, distance))
    edge_rows.sort(reverse=True)
    for deviation, label, id0, id1, distance in edge_rows[:25]:
        lines.append(
            f"  {label} {id0:3d}->{id1:3d}: "
            f"dist={distance:.4f} dev={distance - float(expected_spacing_mm):+.4f}"
        )
    lines.append("")

    lines.append("Per-Corner Diagnostics")
    lines.append(
        "  id row col x_mm y_mm z_mm obs reproj_mean reproj_median "
        "reproj_max tri_status tri_inliers tri_obs"
    )
    for marker_id in sorted(points):
        row, col = id_to_row_col.get(marker_id, (None, None))
        point = points[marker_id]
        err = error_stats.get(marker_id, {})
        tri = triangulation_by_marker.get(marker_id)
        tri_status = "bootstrap_or_existing"
        tri_inliers = ""
        tri_obs = ""
        if tri is not None:
            tri_status = "triangulated" if bool(tri.success) else "tri_failed"
            tri_inliers = str(int(getattr(tri, "num_inliers", 0)))
            tri_obs = str(int(getattr(tri, "num_observations", 0)))

        lines.append(
            "  "
            f"{marker_id:3d} "
            f"{'' if row is None else row:>3} "
            f"{'' if col is None else col:>3} "
            f"{float(point[0]): .6f} "
            f"{float(point[1]): .6f} "
            f"{float(point[2]): .6f} "
            f"{int(observation_counts.get(marker_id, 0)):4d} "
            f"{float(err.get('mean', float('nan'))): .4f} "
            f"{float(err.get('median', float('nan'))): .4f} "
            f"{float(err.get('max', float('nan'))): .4f} "
            f"{tri_status} "
            f"{tri_inliers:>4} "
            f"{tri_obs:>4}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return output_path
