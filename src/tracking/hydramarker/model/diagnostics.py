from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
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
