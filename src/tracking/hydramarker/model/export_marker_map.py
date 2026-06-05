from __future__ import annotations

from pathlib import Path
import json
from copy import deepcopy
from datetime import datetime

import numpy as np

from tracking.hydramarker.model.state import SfMState


def _load_json(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


def _get_meta(meta: dict):
    id_encoding = meta.get("id_encoding", {})

    return {
        "id_base": _safe_int(
            meta.get("id_base", id_encoding.get("id_base", 0)),
            0,
        ),
        "id_num_cols": _safe_int(
            meta.get("id_num_cols", id_encoding.get("num_cols", 0)),
            0,
        ),
        "origin_row": _safe_int(
            meta.get(
                "detectable_origin_row",
                id_encoding.get("origin_row", 0),
            ),
            0,
        ),
        "origin_col": _safe_int(
            meta.get(
                "detectable_origin_col",
                id_encoding.get("origin_col", 0),
            ),
            0,
        ),
        "detectable_rows": _safe_int(
            meta.get("detectable_corner_rows", 0),
            0,
        ),
        "detectable_cols": _safe_int(
            meta.get("detectable_corner_cols", 0),
            0,
        ),
        "corner_rows": _safe_int(
            meta.get("corner_rows", 0),
            0,
        ),
        "corner_cols": _safe_int(
            meta.get("corner_cols", 0),
            0,
        ),
        "rows": _safe_int(
            meta.get("rows", 0),
            0,
        ),
        "cols": _safe_int(
            meta.get("cols", 0),
            0,
        ),
    }


def _corner_id(
    row: int,
    col: int,
    *,
    id_base: int,
    id_num_cols: int,
) -> int:
    return int(id_base + row * id_num_cols + col)


def _invert_single_id(
    marker_id: int,
    *,
    id_base: int,
    id_num_cols: int,
    origin_row: int,
    origin_col: int,
    detectable_rows: int,
    detectable_cols: int,
):
    for row in range(origin_row, origin_row + detectable_rows):
        for col in range(origin_col, origin_col + detectable_cols):
            candidate = _corner_id(
                row,
                col,
                id_base=id_base,
                id_num_cols=id_num_cols,
            )

            if candidate == marker_id:
                return int(row), int(col)

    return None


def _candidate_num_cols(meta: dict) -> list[int]:
    values = []

    def add(v):
        v = int(v)
        if v > 0 and v not in values:
            values.append(v)

    add(meta["id_num_cols"])
    add(meta["detectable_cols"])
    add(meta["corner_cols"])
    add(meta["cols"])
    add(meta["corner_cols"] - 1)
    add(meta["cols"] + 1)

    return values


def _resolve_id_encoding(
    marker_ids: list[int],
    meta: dict,
):
    id_base = meta["id_base"]

    origin_row = meta["origin_row"]
    origin_col = meta["origin_col"]

    detectable_rows = meta["detectable_rows"]
    detectable_cols = meta["detectable_cols"]

    candidates = _candidate_num_cols(meta)

    if not candidates:
        raise RuntimeError("Could not generate id_num_cols candidates.")

    best = None
    best_score = -1

    for candidate_num_cols in candidates:
        mapping = {}
        used_positions = set()

        success = True

        for marker_id in marker_ids:
            rc = _invert_single_id(
                marker_id,
                id_base=id_base,
                id_num_cols=candidate_num_cols,
                origin_row=origin_row,
                origin_col=origin_col,
                detectable_rows=detectable_rows,
                detectable_cols=detectable_cols,
            )

            if rc is None:
                success = False
                break

            if rc in used_positions:
                success = False
                break

            mapping[int(marker_id)] = rc
            used_positions.add(rc)

        if not success:
            continue

        rows = [r for r, _ in mapping.values()]
        cols = [c for _, c in mapping.values()]

        row_span = max(rows) - min(rows) + 1
        col_span = max(cols) - min(cols) + 1

        compactness = row_span * col_span

        score = (
            len(mapping) * 1000
            - compactness
        )

        if score > best_score:
            best_score = score
            best = {
                "id_num_cols": int(candidate_num_cols),
                "mapping": mapping,
            }

    if best is None:
        raise RuntimeError(
            "Could not resolve a valid marker ID encoding automatically."
        )

    return best


def _round_xyz(
    xyz: np.ndarray,
    decimals: int | None,
) -> list[float]:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)

    if decimals is not None:
        xyz = np.round(xyz, int(decimals))

    return [
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
    ]


def _metric_normalization_from_topology(
    marker_positions: dict[int, np.ndarray],
    resolved_mapping: dict[int, tuple[int, int]],
    *,
    expected_spacing_mm: float,
) -> tuple[float, float, int]:
    id_by_row_col = {
        (int(row), int(col)): int(marker_id)
        for marker_id, (row, col) in resolved_mapping.items()
        if int(marker_id) in marker_positions
    }

    distances: list[float] = []

    for (row, col), marker_id in id_by_row_col.items():
        p = np.asarray(
            marker_positions[int(marker_id)],
            dtype=np.float64,
        ).reshape(3)

        for neighbor_key in ((row, col + 1), (row + 1, col)):
            neighbor_id = id_by_row_col.get(neighbor_key)
            if neighbor_id is None:
                continue

            q = np.asarray(
                marker_positions[int(neighbor_id)],
                dtype=np.float64,
            ).reshape(3)

            d = float(np.linalg.norm(q - p))
            if np.isfinite(d) and d > 1e-12:
                distances.append(d)

    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 1e-12)]

    if values.size == 0:
        return 1.0, float("nan"), 0

    median_spacing = float(np.median(values))
    scale = float(expected_spacing_mm / median_spacing)

    return scale, median_spacing, int(values.size)


def export_marker_geometry_json(
    state: SfMState,
    source_marker_json_path: str | Path,
    output_json_path: str | Path,
    *,
    xyz_decimals: int | None = 6,
    include_camera_poses: bool = False,
    overwrite_marker_type: str = "sfm_map",
) -> Path:
    """
    Export aligned SfM geometry as a universal MarkerGeometry JSON.

    Important:
        The row/col mapping is resolved automatically from the observed
        marker IDs. This makes the export robust against different marker
        placements/orientations as long as the IDs remain consistent.
    """

    source_marker_json_path = Path(source_marker_json_path)
    output_json_path = Path(output_json_path)

    meta_in = _load_json(source_marker_json_path)
    meta_out = deepcopy(meta_in)

    meta = _get_meta(meta_out)

    marker_ids = sorted(
        int(mid)
        for mid in state.marker_positions.keys()
    )

    if not marker_ids:
        raise ValueError(
            "Cannot export marker geometry: state has no marker positions."
        )

    resolved = _resolve_id_encoding(
        marker_ids=marker_ids,
        meta=meta,
    )

    resolved_num_cols = resolved["id_num_cols"]
    resolved_mapping = resolved["mapping"]

    marker_positions = {
        int(marker_id): np.asarray(point, dtype=np.float64).reshape(3)
        for marker_id, point in state.marker_positions.items()
    }

    metric_scale, median_spacing_before_metric_scale, metric_edge_count = (
        _metric_normalization_from_topology(
            marker_positions,
            resolved_mapping,
            expected_spacing_mm=float(meta_out["square_size_mm"]),
        )
    )

    corners = []

    for marker_id in marker_ids:
        row, col = resolved_mapping[int(marker_id)]

        xyz_mm = _round_xyz(
            metric_scale * marker_positions[int(marker_id)],
            xyz_decimals,
        )

        corners.append(
            {
                "id": int(marker_id),
                "row": int(row),
                "col": int(col),
                "xyz_mm": xyz_mm,
            }
        )

    meta_out["marker_type"] = overwrite_marker_type

    meta_out["geometry_source"] = {
        "type": "sfm_bundle_adjustment",
        "coordinate_frame": "marker",
        "units": "mm",
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_marker_json": str(source_marker_json_path),

        "num_exported_corners": int(len(corners)),

        "resolved_id_num_cols": int(resolved_num_cols),

        "metric_normalization_scale": float(metric_scale),
        "median_topology_spacing_before_metric_normalization": float(
            median_spacing_before_metric_scale
        ),
        "metric_normalization_edge_count": int(metric_edge_count),
        "expected_topology_spacing_mm": float(meta_out["square_size_mm"]),

        "id_base": int(meta["id_base"]),
        "detectable_origin_row": int(meta["origin_row"]),
        "detectable_origin_col": int(meta["origin_col"]),
        "detectable_corner_rows": int(meta["detectable_rows"]),
        "detectable_corner_cols": int(meta["detectable_cols"]),

        "note": (
            "Corner row/col mapping was automatically resolved from the "
            "observed marker IDs."
        ),
    }

    meta_out["corners"] = corners

    if include_camera_poses:
        camera_poses = []

        for frame_id in sorted(state.poses.keys()):
            pose = state.poses[int(frame_id)]

            camera_poses.append(
                {
                    "frame_id": int(frame_id),
                    "R": np.asarray(
                        pose.R,
                        dtype=np.float64,
                    ).round(
                        xyz_decimals if xyz_decimals is not None else 12
                    ).tolist(),
                    "t_mm": _round_xyz(
                        pose.t,
                        xyz_decimals,
                    ),
                }
            )

        meta_out["camera_poses"] = camera_poses

    output_json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            meta_out,
            f,
            indent=2,
        )

    return output_json_path
