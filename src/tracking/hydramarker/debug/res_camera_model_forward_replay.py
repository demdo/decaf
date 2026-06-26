"""Replay a logged HydraMarker run with alternative camera models.

This uses the stored 2D/3D correspondences from a JSONL tracker run, resolves
PnP with each camera model, and reports board-frame Z drift. Newer logs also
contain raw ChArUco table-calibration observations; when present, the table
frame is re-solved for each camera model as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


def _ensure_src_on_path() -> None:
    tracking_root = Path(__file__).resolve().parents[2]
    src_root = tracking_root.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_ensure_src_on_path()

from tracking.hydramarker.debug.debug_tracker_translation import _camera_from_run_start
from tracking.hydramarker import tracker as hydramarker_cpp
from tracking.hydramarker.config import TrackerConfig


RUN_PATH = Path("hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_forward.jsonl")
MODEL_PATHS = [
    ("old_logged", None),
    ("new_primary", Path("hydramarker/calib/hydramarker_camera_calibration_20260623_103300.npz")),
    ("new_standard5", Path("hydramarker/calib/hydramarker_camera_calibration_20260623_103300_standard5.npz")),
    ("new_no_k3", Path("hydramarker/calib/hydramarker_camera_calibration_20260623_103300_no_k3.npz")),
    ("new_rational8", Path("hydramarker/calib/hydramarker_camera_calibration_20260623_103300_rational8.npz")),
]


def _npz_scalar(npz: Any, key: str, default: Any = "") -> Any:
    if key not in npz:
        return default
    value = npz[key]
    try:
        return value.item()
    except Exception:
        try:
            return value.tolist()
        except Exception:
            return default


def load_model(label: str, path: Path | None, run_start: dict) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if path is None:
        K, dist = _camera_from_run_start(run_start)
        return K, dist, {"calibration_model": "run_start", "distortion_model": "logged"}

    npz = np.load(path, allow_pickle=True)
    K = np.asarray(npz["K"] if "K" in npz else npz["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(
        npz["dist"] if "dist" in npz else npz["opencv_dist_coeffs"],
        dtype=np.float64,
    ).reshape(-1, 1)
    info = {
        key: _npz_scalar(npz, key)
        for key in (
            "calibration_model",
            "distortion_model",
            "model_quality_score",
            "corner_reprojection_p95_px",
            "edge_corner_reprojection_mean_px",
            "radial_residual_mean_px",
            "radial_residual_abs_mean_px",
            "radial_turn_count",
            "radial_scale_min",
            "radial_scale_max",
            "selected_coverage_cells",
            "selected_coverage_total_cells",
        )
    }
    info["path"] = str(path)
    return K, dist, info


def make_pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def mean_unit_vectors(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    acc = np.zeros(3, dtype=np.float64)
    ref = normalize_vector(values[0])
    for value in values:
        unit = normalize_vector(value)
        if float(np.dot(unit, ref)) < 0.0:
            unit = -unit
        acc += unit
    return normalize_vector(acc)


def solve_pose(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray):
    if len(obj) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    try:
        rvec, tvec = cv2.solvePnPRefineVVS(obj, img, K, dist, rvec, tvec)
    except Exception:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
        except Exception:
            pass
    return np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1)


def solve_table_transform_from_observations(
    records: list[dict],
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for record in records:
        solved_frames: list[dict[str, Any]] = []
        for frame in record.get("frames") or []:
            if not bool(frame.get("used", False)):
                continue
            obj = np.asarray(frame.get("object_points_mm") or [], dtype=np.float64).reshape(-1, 3)
            img = np.asarray(frame.get("image_points_uv") or [], dtype=np.float64).reshape(-1, 2)
            solved = solve_pose(obj, img, K, dist)
            if solved is None:
                continue
            rvec, tvec = solved
            R, _ = cv2.Rodrigues(rvec)
            reproj = reprojection_stats(obj, img, rvec, tvec, K, dist)
            solved_frames.append(
                {
                    "rvec": rvec,
                    "tvec": tvec,
                    "R": np.asarray(R, dtype=np.float64).reshape(3, 3),
                    "normal": normalize_vector(R[:, 2]),
                    "x_axis": normalize_vector(R[:, 0]),
                    "origin": np.asarray(tvec, dtype=np.float64).reshape(3),
                    "rms": float(reproj["mean"]),
                }
            )
        if solved_frames:
            positions.append(
                {
                    "position_index": int(record.get("position_index", len(positions) + 1)),
                    "frames": solved_frames,
                    "median_rms": float(np.median([f["rms"] for f in solved_frames])),
                }
            )

    all_frames = [frame for position in positions for frame in position["frames"]]
    if not all_frames:
        return None, {"table_recomputed": False, "table_recompute_reason": "no usable table observations"}

    normal = mean_unit_vectors([frame["normal"] for frame in all_frames])
    source = min(positions, key=lambda position: position["median_rms"])
    source_frames = source["frames"]
    source_x = mean_unit_vectors([frame["x_axis"] for frame in source_frames])
    x_axis = source_x - float(np.dot(source_x, normal)) * normal
    if float(np.linalg.norm(x_axis)) <= 1e-9:
        source_y = mean_unit_vectors([frame["R"][:, 1] for frame in source_frames])
        x_axis = np.cross(source_y, normal)
    x_axis = normalize_vector(x_axis)
    y_axis = normalize_vector(np.cross(normal, x_axis))
    x_axis = normalize_vector(np.cross(y_axis, normal))
    origin = np.mean([frame["origin"] for frame in source_frames], axis=0)

    T_C_B = np.eye(4, dtype=np.float64)
    T_C_B[:3, :3] = np.column_stack([x_axis, y_axis, normal])
    T_C_B[:3, 3] = origin
    T_B_C = invert_transform(T_C_B)
    info = {
        "table_recomputed": True,
        "table_positions": int(len(positions)),
        "table_frames_used": int(len(all_frames)),
        "table_source_position": int(source["position_index"]),
        "table_frame_reproj_mean_px": float(np.mean([frame["rms"] for frame in all_frames])),
        "table_frame_reproj_median_px": float(np.median([frame["rms"] for frame in all_frames])),
    }
    return T_B_C, info


def reprojection_stats(obj: np.ndarray, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    residual = img - proj
    err = np.linalg.norm(residual, axis=1)
    rel = img - np.asarray([float(K[0, 2]), float(K[1, 2])], dtype=np.float64)
    radius = np.linalg.norm(rel, axis=1)
    valid = radius > 1e-9
    if np.any(valid):
        radial_unit = rel[valid] / radius[valid, None]
        tangential_unit = np.c_[-radial_unit[:, 1], radial_unit[:, 0]]
        res = residual[valid]
        radial = np.sum(res * radial_unit, axis=1)
        tangential = np.sum(res * tangential_unit, axis=1)
    else:
        radial = np.asarray([], dtype=np.float64)
        tangential = np.asarray([], dtype=np.float64)
    return {
        "mean": float(np.mean(err)),
        "p95": float(np.percentile(err, 95)),
        "radial_mean": float(np.mean(radial)) if radial.size else math.nan,
        "radial_abs_mean": float(np.mean(np.abs(radial))) if radial.size else math.nan,
        "tangential_abs_mean": float(np.mean(np.abs(tangential))) if tangential.size else math.nan,
    }


def frame_points(frame_detail: dict, common_ids: set[tuple[int, int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    obj: list[list[float]] = []
    img: list[list[float]] = []
    for corner in frame_detail.get("pose_corners") or []:
        if common_ids is not None:
            try:
                corner_id = (int(corner["global_row"]), int(corner["global_col"]))
            except Exception:
                continue
            if corner_id not in common_ids:
                continue
        xyz = corner.get("xyz_mm")
        uv = corner.get("uv_px")
        if not isinstance(xyz, list) or not isinstance(uv, list):
            continue
        xyz_f = [float(v) for v in xyz[:3]]
        uv_f = [float(v) for v in uv[:2]]
        if np.all(np.isfinite(xyz_f)) and np.all(np.isfinite(uv_f)):
            obj.append(xyz_f)
            img.append(uv_f)
    return np.asarray(obj, dtype=np.float64).reshape(-1, 3), np.asarray(img, dtype=np.float64).reshape(-1, 2)


def linear_slope_mm_per_100(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x < hi)
    if int(np.count_nonzero(mask)) < 3:
        return math.nan
    A = np.c_[x[mask], np.ones(int(np.count_nonzero(mask)))]
    slope, _ = np.linalg.lstsq(A, y[mask], rcond=None)[0]
    return float(100.0 * slope)


def replay_model(
    label: str,
    path: Path | None,
    *,
    run_start: dict,
    frames: list[dict],
    details: dict[int, dict],
    table_T_B_C: np.ndarray,
    table_observations: list[dict],
    common_ids: set[tuple[int, int]],
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    K, dist, info = load_model(label, path, run_start)
    recomputed_table, table_info = solve_table_transform_from_observations(
        table_observations,
        K,
        dist,
    )
    if recomputed_table is not None:
        table_T_B_C = recomputed_table

    depth_filter = hydramarker_cpp.create_pose_depth_filter(
        K,
        dist,
        TrackerConfig(
            pose_depth_filter_observation_std_mm=16.0,
            pose_depth_filter_process_std_mm=0.05,
            pose_depth_filter_initial_velocity_std_mm=0.1,
            pose_depth_filter_reprojection_guard_px=1.0,
        ),
    )
    dist_flat = np.asarray(dist, dtype=np.float64).reshape(-1)
    info.update(
        {
            "model": label,
            **table_info,
            "dist_n": int(dist_flat.size),
            "dist_abs_max": float(np.max(np.abs(dist_flat))) if dist_flat.size else 0.0,
            "dist": " ".join(f"{x:.12g}" for x in dist_flat),
        }
    )

    rows: list[dict[str, float]] = []
    for frame_data in frames:
        frame = int(frame_data["frame"])
        detail = details.get(frame, {})
        obj, img = frame_points(detail)
        solved = solve_pose(obj, img, K, dist)
        if solved is None:
            continue
        rvec, tvec = solved
        T_raw = table_T_B_C @ make_pose_matrix(rvec, tvec)
        raw_board = T_raw[:3, 3].astype(np.float64)

        filtered = depth_filter.update(
            rvec=rvec,
            tvec=tvec,
            object_points=obj,
            image_points=img,
        )
        T_filtered = table_T_B_C @ make_pose_matrix(filtered.rvec, filtered.tvec)
        filtered_board = T_filtered[:3, 3].astype(np.float64)

        obj_common, img_common = frame_points(detail, common_ids)
        common_z = math.nan
        solved_common = solve_pose(obj_common, img_common, K, dist)
        if solved_common is not None:
            common_z = float((table_T_B_C @ make_pose_matrix(*solved_common))[2, 3])

        reproj = reprojection_stats(obj, img, rvec, tvec, K, dist)
        rows.append(
            {
                "frame": frame,
                "points": float(len(obj)),
                "raw_x": float(raw_board[0]),
                "raw_y": float(raw_board[1]),
                "raw_z": float(raw_board[2]),
                "filtered_x": float(filtered_board[0]),
                "filtered_y": float(filtered_board[1]),
                "filtered_z": float(filtered_board[2]),
                "common_z": float(common_z),
                "camera_z_raw": float(tvec.reshape(3)[2]),
                "camera_z_filtered": float(filtered.tvec.reshape(3)[2]),
                "filter_delta_z": float(filtered.delta_z_mm),
                "filter_guard_alpha": float(filtered.guard_alpha),
                "reproj_mean": reproj["mean"],
                "reproj_p95": reproj["p95"],
                "radial_mean": reproj["radial_mean"],
                "radial_abs_mean": reproj["radial_abs_mean"],
                "tangential_abs_mean": reproj["tangential_abs_mean"],
            }
        )
    return info, rows


def summarize_model(info: dict[str, Any], rows: list[dict[str, float]], common_count: int) -> dict[str, Any]:
    raw_y = np.asarray([r["raw_y"] for r in rows], dtype=np.float64)
    raw_z = np.asarray([r["raw_z"] for r in rows], dtype=np.float64)
    filtered_z = np.asarray([r["filtered_z"] for r in rows], dtype=np.float64)
    common_z = np.asarray([r["common_z"] for r in rows], dtype=np.float64)
    y_travel = -(raw_y - raw_y[0])
    raw_rel = raw_z - raw_z[0]
    filtered_rel = filtered_z - filtered_z[0]
    common_valid = np.isfinite(common_z)
    common_rel = (
        common_z - common_z[common_valid][0]
        if np.any(common_valid)
        else np.full_like(common_z, np.nan, dtype=np.float64)
    )
    summary = {
        **info,
        "frames": len(rows),
        "common_corners": int(common_count),
        "y_range_mm": float(np.nanmax(y_travel) - np.nanmin(y_travel)),
        "raw_board_z_range_mm": float(np.nanmax(raw_rel) - np.nanmin(raw_rel)),
        "raw_board_z_last_mm": float(raw_rel[-1]),
        "filtered_board_z_range_mm": float(np.nanmax(filtered_rel) - np.nanmin(filtered_rel)),
        "filtered_board_z_last_mm": float(filtered_rel[-1]),
        "common_board_z_range_mm": float(np.nanmax(common_rel) - np.nanmin(common_rel)),
        "common_board_z_last_mm": float(common_rel[-1]),
        "raw_slope_0_55_mm_per_100": linear_slope_mm_per_100(y_travel, raw_rel, 0.0, 55.0),
        "raw_slope_55_75_mm_per_100": linear_slope_mm_per_100(y_travel, raw_rel, 55.0, 75.0),
        "raw_slope_75_91_mm_per_100": linear_slope_mm_per_100(y_travel, raw_rel, 75.0, 91.0),
        "filtered_slope_0_55_mm_per_100": linear_slope_mm_per_100(y_travel, filtered_rel, 0.0, 55.0),
        "filtered_slope_55_75_mm_per_100": linear_slope_mm_per_100(y_travel, filtered_rel, 55.0, 75.0),
        "filtered_slope_75_91_mm_per_100": linear_slope_mm_per_100(y_travel, filtered_rel, 75.0, 91.0),
        "reproj_mean_px": float(np.nanmean([r["reproj_mean"] for r in rows])),
        "reproj_p95_px": float(np.nanmean([r["reproj_p95"] for r in rows])),
        "radial_mean_px": float(np.nanmean([r["radial_mean"] for r in rows])),
        "radial_abs_mean_px": float(np.nanmean([r["radial_abs_mean"] for r in rows])),
        "tangential_abs_mean_px": float(np.nanmean([r["tangential_abs_mean"] for r in rows])),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a HydraMarker tracker JSONL with alternative camera models."
    )
    parser.add_argument(
        "run_path",
        nargs="?",
        type=Path,
        default=RUN_PATH,
        help="Tracker JSONL to replay.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for summary CSV and plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_path = Path(args.run_path)
    recs = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_start = next(r for r in recs if r.get("type") == "run_start")
    frames = [r["data"] for r in recs if r.get("type") == "frame"]
    details = {int(r["frame"]): r for r in recs if r.get("type") == "frame_detail"}
    table = next(r for r in recs if r.get("type") == "table_calibration")
    table_T_B_C = np.asarray(table["T_B_C"], dtype=np.float64).reshape(4, 4)
    table_observations = [
        r for r in recs if r.get("type") == "table_calibration_observations"
    ]

    common_ids: set[tuple[int, int]] | None = None
    for frame_data in frames:
        ids = {
            (int(c["global_row"]), int(c["global_col"]))
            for c in (details.get(int(frame_data["frame"]), {}).get("pose_corners") or [])
            if "global_row" in c and "global_col" in c
        }
        common_ids = ids if common_ids is None else common_ids & ids
    common_ids = common_ids or set()

    out_dir = Path(args.output_dir) if args.output_dir is not None else run_path.parent
    summary_path = out_dir / f"{run_path.stem}_camera_model_replay_summary.csv"
    plot_path = out_dir / f"{run_path.stem}_camera_model_replay.png"

    model_rows: dict[str, list[dict[str, float]]] = {}
    summaries: list[dict[str, Any]] = []
    for label, path in MODEL_PATHS:
        info, rows = replay_model(
            label,
            path,
            run_start=run_start,
            frames=frames,
            details=details,
            table_T_B_C=table_T_B_C,
            table_observations=table_observations,
            common_ids=common_ids,
        )
        model_rows[label] = rows
        summaries.append(summarize_model(info, rows, len(common_ids)))

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        keys: list[str] = []
        for row in summaries:
            for key in row:
                if key not in keys:
                    keys.append(key)
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    import matplotlib.pyplot as plt

    colors = {
        "old_logged": "#444444",
        "new_primary": "#d62728",
        "new_standard5": "#1f77b4",
        "new_no_k3": "#2ca02c",
        "new_rational8": "#9467bd",
    }
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.0), sharex=True)
    for summary in summaries:
        label = str(summary["model"])
        rows = model_rows[label]
        frame_arr = np.asarray([r["frame"] for r in rows], dtype=np.int64)
        raw_z = np.asarray([r["raw_z"] for r in rows], dtype=np.float64)
        filtered_z = np.asarray([r["filtered_z"] for r in rows], dtype=np.float64)
        common_z = np.asarray([r["common_z"] for r in rows], dtype=np.float64)
        raw_rel = raw_z - raw_z[0]
        filtered_rel = filtered_z - filtered_z[0]
        common_valid = np.isfinite(common_z)
        common_rel = (
            common_z - common_z[common_valid][0]
            if np.any(common_valid)
            else np.full_like(common_z, np.nan, dtype=np.float64)
        )
        axes[0].plot(frame_arr, raw_rel, label=label, color=colors.get(label), linewidth=1.5)
        axes[1].plot(frame_arr, filtered_rel, label=label, color=colors.get(label), linewidth=1.5)
        axes[2].plot(frame_arr, common_rel, label=label, color=colors.get(label), linewidth=1.5)

    titles = (
        "raw replay board-Z from all logged corners",
        "camera-Z Kalman replay board-Z",
        f"raw replay board-Z from common corners ({len(common_ids)} corners)",
    )
    for ax, title in zip(axes, titles):
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
        ax.grid(True)
        ax.set_ylabel("delta board Z [mm]")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left", ncols=2)
    axes[-1].set_xlabel("frame")
    fig.suptitle("Forward run offline replay with new camera calibration models", fontweight="bold")
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")

    print(f"summary {summary_path.resolve()}")
    print(f"plot {plot_path.resolve()}")
    for row in summaries:
        print(
            f"{row['model']}: raw_range={row['raw_board_z_range_mm']:.3f} "
            f"raw_last={row['raw_board_z_last_mm']:.3f} "
            f"filtered_range={row['filtered_board_z_range_mm']:.3f} "
            f"filtered_last={row['filtered_board_z_last_mm']:.3f} "
            f"reproj={row['reproj_mean_px']:.3f} radial={row['radial_mean_px']:.3f} "
            f"dist_abs_max={row['dist_abs_max']:.3f}"
        )


if __name__ == "__main__":
    main()
