"""Offline drift diagnostics for HydraMarker JSONL tracker runs.

The module reads structured tracker logs, reconstructs pose/corner series, and
plots drift, pose outages, feature coverage, and reprojection behavior for
post-run analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_STEP_FRAMES = (1783, 1784)
DEFAULT_PRE_SLOPE_MAX_FRAME = 1778
SCRIPT_VERSION = "debug_tracker_drift_movement_axis_2026-06-17"


def _ensure_qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def select_jsonl_with_qt() -> Path | None:
    try:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
    except ImportError as exc:
        raise RuntimeError(
            "No JSONL path was given and PySide6 is not available for the file dialog. "
            "Pass a run JSONL path on the command line or use the Qt-enabled environment."
        ) from exc

    _ensure_qt_app()
    default_dir = Path(__file__).resolve().parents[1] / "tests" / "hydramarker_tracker_runs"
    if not default_dir.exists():
        default_dir = Path.cwd()

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select HydraTracker drift run log",
        str(default_dir),
        "HydraTracker JSONL (*.jsonl);;All Files (*)",
    )
    if not path:
        QMessageBox.information(None, "HydraTracker drift debug", "No run log selected.")
        return None
    return Path(path)


def select_calibration_with_qt() -> Path | None:
    try:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
    except ImportError as exc:
        raise RuntimeError(
            "No calibration NPZ was given and PySide6 is not available for the file dialog. "
            "Pass --calib on the command line or use the Qt-enabled environment."
        ) from exc

    _ensure_qt_app()
    default_dir = Path(__file__).resolve().parents[1] / "data" / "realsense"
    if not default_dir.exists():
        default_dir = Path.cwd()

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select rational8 camera calibration NPZ",
        str(default_dir),
        "Camera calibration NPZ (*.npz);;All Files (*)",
    )
    if not path:
        QMessageBox.information(None, "HydraTracker drift debug", "No calibration file selected.")
        return None
    return Path(path)


def default_out_dir_for_jsonl(path: Path) -> Path:
    return path.with_name(f"{path.stem}_solver_debug")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class FrameBundle:
    frame: int
    frame_data: dict[str, Any] = field(default_factory=dict)
    correspondence_corners: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunData:
    path: Path
    run_id: str
    timestamp: str
    marker_json_path: str
    camera_intrinsics: dict[str, Any]
    board_T_B_C: np.ndarray | None
    board_record: dict[str, Any] | None
    frames: list[FrameBundle]


@dataclass(frozen=True)
class CameraModel:
    name: str
    K: np.ndarray
    dist: np.ndarray
    path: Path
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverSpec:
    name: str
    flag_name: str
    use_prev_guess: bool = False
    refine: bool = False
    ransac: bool = False


@dataclass
class PoseSolve:
    frame: int
    solver: str
    success: bool
    rvec: np.ndarray | None = None
    tvec_camera: np.ndarray | None = None
    tvec_board: np.ndarray | None = None
    point_count: int = 0
    inlier_count: int = 0
    centroid_u_px: float = np.nan
    centroid_v_px: float = np.nan
    span_u_px: float = np.nan
    span_v_px: float = np.nan
    reproj_mean_px: float = np.nan
    reproj_median_px: float = np.nan
    reproj_p95_px: float = np.nan
    reproj_max_px: float = np.nan
    residuals: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(shape)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _as_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(point, dtype=np.float64).reshape(3)
    return (np.asarray(T, dtype=np.float64).reshape(4, 4) @ p)[:3]


def load_run(path: Path) -> RunData:
    frames_by_id: dict[int, FrameBundle] = {}
    run_id = path.stem
    timestamp = ""
    marker_json_path = ""
    camera_intrinsics: dict[str, Any] = {}
    board_T_B_C: np.ndarray | None = None
    board_record: dict[str, Any] | None = None

    def bundle(frame_value: Any) -> FrameBundle:
        frame = _to_int(frame_value, default=len(frames_by_id))
        if frame not in frames_by_id:
            frames_by_id[frame] = FrameBundle(frame=frame)
        return frames_by_id[frame]

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            record_type = record.get("type")
            if record_type == "run_start":
                run_id = str(record.get("run_id") or run_id)
                timestamp = str(record.get("timestamp") or "")
                marker_json_path = str(record.get("marker_json_path") or "")
                camera_intrinsics = dict(record.get("camera_intrinsics") or {})
                continue

            if record_type == "board_pose":
                board_record = record
                board_T_B_C = _as_matrix(record.get("T_B_C"), (4, 4))
                continue

            if record_type == "frame":
                data = dict(record.get("data") or {})
                bundle(data.get("frame")).frame_data = data
                continue

            if record_type == "frame_detail":
                item = bundle(record.get("frame"))
                item.correspondence_corners = list(record.get("correspondence_corners") or [])
                continue

    frames = [frames_by_id[k] for k in sorted(frames_by_id)]
    if not frames:
        raise RuntimeError(f"No frame records found in {path}")

    return RunData(
        path=path,
        run_id=run_id,
        timestamp=timestamp,
        marker_json_path=marker_json_path,
        camera_intrinsics=camera_intrinsics,
        board_T_B_C=board_T_B_C,
        board_record=board_record,
        frames=frames,
    )


def first_npz_array(npz: Any, names: tuple[str, ...]) -> tuple[np.ndarray | None, str]:
    for name in names:
        if name in npz:
            return np.asarray(npz[name], dtype=np.float64), name
    return None, ""


def read_npz_image_size(npz: Any) -> list[int] | None:
    if "image_size" in npz:
        values = np.asarray(npz["image_size"]).reshape(-1)
        if values.size >= 2:
            return [int(values[0]), int(values[1])]

    width_keys = ("width", "image_width", "rgb_width")
    height_keys = ("height", "image_height", "rgb_height")
    width = next((int(np.asarray(npz[key]).reshape(-1)[0]) for key in width_keys if key in npz), None)
    height = next((int(np.asarray(npz[key]).reshape(-1)[0]) for key in height_keys if key in npz), None)
    if width is not None and height is not None:
        return [width, height]
    return None


def _npz_scalar(npz: Any, key: str) -> Any:
    if key not in npz:
        return ""
    arr = np.asarray(npz[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def load_camera_model_npz(path: Path) -> CameraModel:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {path}")

    with np.load(path, allow_pickle=True) as npz:
        K, K_key = first_npz_array(
            npz,
            ("K", "K_rgb", "camera_matrix", "camera_intrinsics", "intrinsics"),
        )
        if K is None:
            raise KeyError(
                f"{path} must contain one of: K, K_rgb, camera_matrix, camera_intrinsics, intrinsics"
            )

        dist, dist_key = first_npz_array(
            npz,
            (
                "dist",
                "dist_rgb",
                "dist_coeffs",
                "distortion_coeffs",
                "opencv_dist_coeffs",
                "effective_opencv_dist_coeffs",
            ),
        )
        if dist is None:
            raise KeyError(
                f"{path} must contain one of: dist, dist_rgb, dist_coeffs, distortion_coeffs, "
                "opencv_dist_coeffs, effective_opencv_dist_coeffs"
            )

        info = {
            "path": str(path),
            "K_key": K_key,
            "dist_key": dist_key,
            "image_size": read_npz_image_size(npz),
            "rms": _npz_scalar(npz, "rms"),
            "calibration_rms": _npz_scalar(npz, "calibration_rms"),
            "num_images_total": _npz_scalar(npz, "num_images_total"),
            "num_images_used": _npz_scalar(npz, "num_images_used"),
            "num_candidates_total": _npz_scalar(npz, "num_candidates_total"),
            "num_candidates_selected": _npz_scalar(npz, "num_candidates_selected"),
            "selected_reprojection_mean_px": _npz_scalar(npz, "selected_reprojection_mean_px"),
            "distortion_model": _npz_scalar(npz, "distortion_model"),
            "calibration_model": _npz_scalar(npz, "calibration_model"),
            "created_at": _npz_scalar(npz, "created_at"),
        }

    return CameraModel(
        name=path.stem,
        K=np.asarray(K, dtype=np.float64).reshape(3, 3),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        path=path,
        info=info,
    )


def write_camera_model_csv(path: Path, model: CameraModel) -> None:
    row = {
        "name": model.name,
        "path": str(model.path),
        "K_key": model.info.get("K_key", ""),
        "dist_key": model.info.get("dist_key", ""),
        "fx": model.K[0, 0],
        "fy": model.K[1, 1],
        "cx": model.K[0, 2],
        "cy": model.K[1, 2],
        "dist": " ".join(f"{x:.12g}" for x in model.dist.reshape(-1)),
        "dist_coeff_count": int(model.dist.reshape(-1).size),
        "image_size": model.info.get("image_size", ""),
        "rms": model.info.get("rms", ""),
        "num_images_used": model.info.get("num_images_used", ""),
        "num_candidates_total": model.info.get("num_candidates_total", ""),
        "selected_reprojection_mean_px": model.info.get("selected_reprojection_mean_px", ""),
        "distortion_model": model.info.get("distortion_model", ""),
        "calibration_model": model.info.get("calibration_model", ""),
        "created_at": model.info.get("created_at", ""),
    }
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def corner_identity(corner: dict[str, Any]) -> tuple[Any, ...] | None:
    row = corner.get("global_row")
    col = corner.get("global_col")
    if row is not None and col is not None:
        return (_to_int(row), _to_int(col))
    row = corner.get("local_row")
    col = corner.get("local_col")
    if row is not None and col is not None:
        return ("local", _to_int(row), _to_int(col))
    return None


def corner_id_to_text(ident: tuple[Any, ...] | None) -> str:
    if ident is None:
        return ""
    return ":".join(str(x) for x in ident)


def object_image_points(frame: FrameBundle) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    for corner in frame.correspondence_corners:
        xyz = corner.get("xyz_mm")
        uv = corner.get("uv_px")
        if not isinstance(xyz, list) or not isinstance(uv, list):
            continue
        if len(xyz) != 3 or len(uv) != 2:
            continue
        selected.append(corner)

    if not selected:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), []

    obj = np.asarray([corner["xyz_mm"] for corner in selected], dtype=np.float64).reshape(-1, 3)
    img = np.asarray([corner["uv_px"] for corner in selected], dtype=np.float64).reshape(-1, 2)
    return obj, img, selected


def _project_points(cv2, obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(obj, dtype=np.float64).reshape(-1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1),
    )
    return projected.reshape(-1, 2)


def refine_pose(cv2, obj: np.ndarray, img: np.ndarray, model: CameraModel, rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(cv2, "solvePnPRefineVVS"):
        rvec, tvec = cv2.solvePnPRefineVVS(obj, img, model.K, model.dist, rvec, tvec)
    elif hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, model.K, model.dist, rvec, tvec)
    return rvec, tvec


def solve_pose_for_frame(
    cv2,
    frame: FrameBundle,
    model: CameraModel,
    spec: SolverSpec,
    T_B_C: np.ndarray | None,
    min_points: int,
    previous_pose: tuple[np.ndarray, np.ndarray] | None,
    *,
    collect_residuals: bool = True,
) -> PoseSolve:
    obj, img, corners = object_image_points(frame)
    solve = PoseSolve(
        frame=frame.frame,
        solver=spec.name,
        success=False,
        point_count=int(len(corners)),
        inlier_count=0,
    )
    if len(corners) > 0:
        centroid = np.mean(img, axis=0)
        span = np.ptp(img, axis=0)
        solve.centroid_u_px = float(centroid[0])
        solve.centroid_v_px = float(centroid[1])
        solve.span_u_px = float(span[0])
        solve.span_v_px = float(span[1])
    if len(corners) < min_points:
        solve.error = f"too_few_points:{len(corners)}<{min_points}"
        return solve

    flag = getattr(cv2, spec.flag_name)
    try:
        if spec.ransac:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                img,
                model.K,
                model.dist,
                iterationsCount=150,
                reprojectionError=2.0,
                confidence=0.999,
                flags=flag,
            )
            if not ok:
                solve.error = "solvePnPRansac returned false"
                return solve
            if inliers is not None and len(inliers) >= min_points:
                inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
                solve.inlier_count = int(len(inlier_idx))
                if spec.refine:
                    rvec, tvec = refine_pose(cv2, obj[inlier_idx], img[inlier_idx], model, rvec, tvec)
            else:
                solve.inlier_count = 0 if inliers is None else int(len(inliers))
        elif spec.use_prev_guess and previous_pose is not None:
            rvec0, tvec0 = previous_pose
            ok, rvec, tvec = cv2.solvePnP(
                obj,
                img,
                model.K,
                model.dist,
                rvec=np.asarray(rvec0, dtype=np.float64).reshape(3, 1).copy(),
                tvec=np.asarray(tvec0, dtype=np.float64).reshape(3, 1).copy(),
                useExtrinsicGuess=True,
                flags=flag,
            )
            if not ok:
                solve.error = "solvePnP returned false"
                return solve
            solve.inlier_count = int(len(corners))
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, img, model.K, model.dist, flags=flag)
            if not ok:
                solve.error = "solvePnP returned false"
                return solve
            solve.inlier_count = int(len(corners))
            if spec.refine:
                rvec, tvec = refine_pose(cv2, obj, img, model, rvec, tvec)

        projected = _project_points(cv2, obj, rvec, tvec, model.K, model.dist)
        residual = projected - img
        errors = np.linalg.norm(residual, axis=1)

        tvec_camera = np.asarray(tvec, dtype=np.float64).reshape(3)
        tvec_board = _transform_point(T_B_C, tvec_camera) if T_B_C is not None else tvec_camera.copy()
        solve.success = True
        solve.rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        solve.tvec_camera = tvec_camera
        solve.tvec_board = tvec_board
        solve.reproj_mean_px = float(np.mean(errors))
        solve.reproj_median_px = float(np.median(errors))
        solve.reproj_p95_px = float(np.percentile(errors, 95))
        solve.reproj_max_px = float(np.max(errors))

        if collect_residuals:
            for idx, corner in enumerate(corners):
                ident = corner_identity(corner)
                uv = img[idx]
                solve.residuals.append(
                    {
                        "frame": frame.frame,
                        "solver": spec.name,
                        "corner_id": corner_id_to_text(ident),
                        "global_row": corner.get("global_row", ""),
                        "global_col": corner.get("global_col", ""),
                        "local_row": corner.get("local_row", ""),
                        "local_col": corner.get("local_col", ""),
                        "x_mm": obj[idx, 0],
                        "y_mm": obj[idx, 1],
                        "z_mm": obj[idx, 2],
                        "u_px": uv[0],
                        "v_px": uv[1],
                        "centroid_u_px": solve.centroid_u_px,
                        "centroid_v_px": solve.centroid_v_px,
                        "du_px": residual[idx, 0],
                        "dv_px": residual[idx, 1],
                        "error_px": errors[idx],
                        "radius_px": float(np.linalg.norm(uv - np.array([model.K[0, 2], model.K[1, 2]]))),
                    }
                )
        return solve
    except Exception as exc:
        solve.error = f"{type(exc).__name__}: {exc}"
        return solve


def solve_sequence(
    cv2,
    run: RunData,
    model: CameraModel,
    spec: SolverSpec,
    min_points: int,
    collect_residuals: bool,
) -> list[PoseSolve]:
    solves: list[PoseSolve] = []
    previous_pose: tuple[np.ndarray, np.ndarray] | None = None
    for frame in run.frames:
        solve = solve_pose_for_frame(
            cv2,
            frame,
            model,
            spec,
            run.board_T_B_C,
            min_points,
            previous_pose,
            collect_residuals=collect_residuals,
        )
        solves.append(solve)
        if spec.use_prev_guess and solve.success and solve.rvec is not None and solve.tvec_camera is not None:
            previous_pose = (solve.rvec.copy(), solve.tvec_camera.copy())
    return solves


def logged_series(run: RunData) -> dict[str, Any]:
    frames: list[int] = []
    tvecs: list[list[float]] = []
    extra_rows: list[dict[str, Any]] = []
    for frame in run.frames:
        data = frame.frame_data
        tvec = [
            _to_float(data.get("tvec_x_mm")),
            _to_float(data.get("tvec_y_mm")),
            _to_float(data.get("tvec_z_mm")),
        ]
        if run.board_T_B_C is not None and np.all(np.isfinite(tvec)):
            tvec = _transform_point(run.board_T_B_C, np.asarray(tvec, dtype=np.float64)).tolist()
        frames.append(frame.frame)
        tvecs.append(tvec)
        extra_rows.append(
            {
                "reproj_mean_px": _to_float(data.get("pose_reproj_mean_px")),
                "reproj_p95_px": _to_float(data.get("pose_reproj_p95_px")),
                "point_count": _to_int(data.get("num_points")),
                "inlier_count": "",
                "centroid_u_px": _to_float(data.get("pose_image_centroid_u_px")),
                "centroid_v_px": _to_float(data.get("pose_image_centroid_v_px")),
                "span_u_px": _to_float(data.get("pose_image_span_u_px")),
                "span_v_px": _to_float(data.get("pose_image_span_v_px")),
            }
        )
    return {
        "label": "logged",
        "frames": np.asarray(frames, dtype=np.int64),
        "tvec_abs": np.asarray(tvecs, dtype=np.float64).reshape(-1, 3),
        "extra": extra_rows,
    }


def series_from_solves(label: str, solves: list[PoseSolve], run_frames: list[FrameBundle]) -> dict[str, Any]:
    by_frame = {solve.frame: solve for solve in solves}
    frames: list[int] = []
    tvecs: list[list[float]] = []
    extra_rows: list[dict[str, Any]] = []
    for frame in run_frames:
        solve = by_frame.get(frame.frame)
        frames.append(frame.frame)
        if solve is not None and solve.success and solve.tvec_board is not None:
            tvecs.append(solve.tvec_board.tolist())
            extra_rows.append(
                {
                    "reproj_mean_px": solve.reproj_mean_px,
                    "reproj_p95_px": solve.reproj_p95_px,
                    "point_count": solve.point_count,
                    "inlier_count": solve.inlier_count,
                    "centroid_u_px": solve.centroid_u_px,
                    "centroid_v_px": solve.centroid_v_px,
                    "span_u_px": solve.span_u_px,
                    "span_v_px": solve.span_v_px,
                }
            )
        else:
            tvecs.append([np.nan, np.nan, np.nan])
            extra_rows.append(
                {
                    "reproj_mean_px": np.nan,
                    "reproj_p95_px": np.nan,
                    "point_count": 0 if solve is None else solve.point_count,
                    "inlier_count": 0 if solve is None else solve.inlier_count,
                    "centroid_u_px": np.nan if solve is None else solve.centroid_u_px,
                    "centroid_v_px": np.nan if solve is None else solve.centroid_v_px,
                    "span_u_px": np.nan if solve is None else solve.span_u_px,
                    "span_v_px": np.nan if solve is None else solve.span_v_px,
                }
            )
    return {
        "label": label,
        "frames": np.asarray(frames, dtype=np.int64),
        "tvec_abs": np.asarray(tvecs, dtype=np.float64).reshape(-1, 3),
        "extra": extra_rows,
    }


def compute_relative(tvec_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]:
    tvec_abs = np.asarray(tvec_abs, dtype=np.float64).reshape(-1, 3)
    valid = np.all(np.isfinite(tvec_abs), axis=1)
    if not np.any(valid):
        return np.full_like(tvec_abs, np.nan), valid, None
    origin_idx = int(np.where(valid)[0][0])
    return tvec_abs - tvec_abs[origin_idx], valid, origin_idx


def linear_slope_mm_per_100mm(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(valid)) < 3 or float(np.ptp(x[valid])) <= 1e-9:
        return np.nan
    A = np.c_[x[valid], np.ones(int(np.count_nonzero(valid)))]
    slope, _ = np.linalg.lstsq(A, y[valid], rcond=None)[0]
    return float(100.0 * slope)


def finite_range(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.nan
    return float(np.max(valid) - np.min(valid))


def value_at_frame(frames: np.ndarray, values: np.ndarray, frame: int) -> float:
    matches = np.where(frames == frame)[0]
    if len(matches) == 0:
        return np.nan
    return float(values[int(matches[0])])


def plane_corrected_stats(rel: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    xyz = rel[valid]
    if len(xyz) < 6:
        return {
            "plane_a": np.nan,
            "plane_b": np.nan,
            "plane_angle_deg": np.nan,
            "plane_z_range_mm": np.nan,
            "plane_final_z_mm": np.nan,
        }

    A = np.c_[xyz[:, 1], np.ones(len(xyz))]
    b, c = np.linalg.lstsq(A, xyz[:, 2], rcond=None)[0]
    a = 0.0
    corrected = rel[:, 2] - (b * rel[:, 1] + c)
    valid_indices = np.where(valid & np.isfinite(corrected))[0]
    if len(valid_indices):
        corrected = corrected - corrected[int(valid_indices[0])]
    valid_corr = corrected[valid & np.isfinite(corrected)]
    final = valid_corr[-1] if len(valid_corr) else np.nan
    return {
        "plane_a": float(a),
        "plane_b": float(b),
        "plane_angle_deg": float(math.degrees(math.atan(abs(b)))),
        "plane_z_range_mm": finite_range(corrected[valid]),
        "plane_final_z_mm": float(final),
    }


def summarize_series(
    series: dict[str, Any],
    step_frames: tuple[int, int],
    pre_slope_max_frame: int,
) -> dict[str, Any]:
    frames = np.asarray(series["frames"], dtype=np.int64)
    rel, valid, origin_idx = compute_relative(series["tvec_abs"])
    if origin_idx is None:
        return {
            "label": series["label"],
            "valid_frames": 0,
            "origin_frame": "",
        }

    y = rel[:, 1]
    z = rel[:, 2]
    extras = series.get("extra") or []
    centroid_v = np.asarray([_to_float(row.get("centroid_v_px")) for row in extras], dtype=np.float64)
    span_v = np.asarray([_to_float(row.get("span_v_px")) for row in extras], dtype=np.float64)
    reproj = np.asarray([_to_float(row.get("reproj_mean_px")) for row in extras], dtype=np.float64)

    step_a = value_at_frame(frames, z, step_frames[0])
    step_b = value_at_frame(frames, z, step_frames[1])
    valid_pre = valid & (frames <= pre_slope_max_frame)
    valid_zv = valid & np.isfinite(z) & np.isfinite(centroid_v)
    valid_zspan = valid & np.isfinite(z) & np.isfinite(span_v)
    valid_zerr = valid & np.isfinite(z) & np.isfinite(reproj)

    def corr_or_nan(mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        if int(np.count_nonzero(mask)) < 3:
            return np.nan
        if np.std(a[mask]) <= 1e-12 or np.std(b[mask]) <= 1e-12:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    tail_values = z[valid & np.isfinite(z)]
    summary = {
        "label": series["label"],
        "valid_frames": int(np.count_nonzero(valid)),
        "origin_frame": int(frames[origin_idx]),
        "x_range_mm": finite_range(rel[valid, 0]),
        "y_range_mm": finite_range(y[valid]),
        "z_range_mm": finite_range(z[valid]),
        "final_z_mm": float(tail_values[-1]) if len(tail_values) else np.nan,
        "z_vs_y_slope_mm_per_100mm": linear_slope_mm_per_100mm(y[valid], z[valid]),
        "pre_z_vs_y_slope_mm_per_100mm": linear_slope_mm_per_100mm(y[valid_pre], z[valid_pre]),
        "step_z_mm": step_b - step_a if np.isfinite(step_a) and np.isfinite(step_b) else np.nan,
        "z_corr_centroid_v": corr_or_nan(valid_zv, z, centroid_v),
        "z_corr_span_v": corr_or_nan(valid_zspan, z, span_v),
        "z_corr_reproj_mean": corr_or_nan(valid_zerr, z, reproj),
    }
    summary.update(plane_corrected_stats(rel, valid))
    return summary


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    keys: list[str] = []
    for summary in summaries:
        for key in summary:
            if key not in keys:
                keys.append(key)
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)


def write_timeseries_csv(path: Path, series_list: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for series in series_list:
        rel, valid, _ = compute_relative(series["tvec_abs"])
        frames = np.asarray(series["frames"], dtype=np.int64)
        extras = series.get("extra") or [{} for _ in range(len(frames))]
        for idx, frame in enumerate(frames):
            extra = extras[idx] if idx < len(extras) else {}
            rows.append(
                {
                    "label": series["label"],
                    "frame": int(frame),
                    "valid": int(bool(valid[idx])),
                    "rel_x_mm": rel[idx, 0],
                    "rel_y_mm": rel[idx, 1],
                    "rel_z_mm": rel[idx, 2],
                    "abs_x_mm": series["tvec_abs"][idx, 0],
                    "abs_y_mm": series["tvec_abs"][idx, 1],
                    "abs_z_mm": series["tvec_abs"][idx, 2],
                    "point_count": extra.get("point_count", ""),
                    "inlier_count": extra.get("inlier_count", ""),
                    "reproj_mean_px": extra.get("reproj_mean_px", ""),
                    "reproj_p95_px": extra.get("reproj_p95_px", ""),
                    "centroid_u_px": extra.get("centroid_u_px", ""),
                    "centroid_v_px": extra.get("centroid_v_px", ""),
                    "span_u_px": extra.get("span_u_px", ""),
                    "span_v_px": extra.get("span_v_px", ""),
                }
            )
    if not rows:
        return
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_solver_frame_csv(path: Path, solves: list[PoseSolve]) -> None:
    rows: list[dict[str, Any]] = []
    for solve in solves:
        rows.append(
            {
                "frame": solve.frame,
                "solver": solve.solver,
                "success": int(solve.success),
                "point_count": solve.point_count,
                "inlier_count": solve.inlier_count,
                "reproj_mean_px": solve.reproj_mean_px,
                "reproj_median_px": solve.reproj_median_px,
                "reproj_p95_px": solve.reproj_p95_px,
                "reproj_max_px": solve.reproj_max_px,
                "centroid_u_px": solve.centroid_u_px,
                "centroid_v_px": solve.centroid_v_px,
                "span_u_px": solve.span_u_px,
                "span_v_px": solve.span_v_px,
                "tvec_cam_x_mm": "" if solve.tvec_camera is None else solve.tvec_camera[0],
                "tvec_cam_y_mm": "" if solve.tvec_camera is None else solve.tvec_camera[1],
                "tvec_cam_z_mm": "" if solve.tvec_camera is None else solve.tvec_camera[2],
                "tvec_board_x_mm": "" if solve.tvec_board is None else solve.tvec_board[0],
                "tvec_board_y_mm": "" if solve.tvec_board is None else solve.tvec_board[1],
                "tvec_board_z_mm": "" if solve.tvec_board is None else solve.tvec_board[2],
                "error": solve.error,
            }
        )
    if not rows:
        return
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _movement_axis_fit_data(series: dict[str, Any]) -> dict[str, Any] | None:
    rel, valid, _ = compute_relative(series["tvec_abs"])
    frames = np.asarray(series["frames"], dtype=np.int64)
    mask = valid & np.all(np.isfinite(rel), axis=1)
    if int(np.count_nonzero(mask)) < 3:
        return None

    positions = rel[mask].astype(np.float64, copy=True)
    used_frames = frames[mask]
    center = np.mean(positions, axis=0)
    centered = positions - center
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = np.asarray(vh[0], dtype=np.float64).reshape(3)
    axis /= np.linalg.norm(axis)
    if axis[1] < 0.0:
        axis = -axis

    projection = centered @ axis
    line_points = center + projection[:, None] * axis
    orthogonal = positions - line_points
    orthogonal_error = np.linalg.norm(orthogonal, axis=1)

    board_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    board_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot_y = float(np.clip(np.dot(axis, board_y), -1.0, 1.0))
    angle_deg = float(math.degrees(math.acos(abs(dot_y))))

    z_axis = board_z - float(np.dot(board_z, axis)) * axis
    z_axis_norm = float(np.linalg.norm(z_axis))
    if z_axis_norm < 1e-12:
        z_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        z_axis = z_axis / z_axis_norm

    corrected_z = (positions - positions[0]) @ z_axis
    raw_z = positions[:, 2] - positions[0, 2]
    raw_z_range = finite_range(raw_z)
    corrected_z_range = finite_range(corrected_z)
    reduction = np.nan
    if np.isfinite(raw_z_range) and raw_z_range > 1e-12 and np.isfinite(corrected_z_range):
        reduction = float(100.0 * (1.0 - corrected_z_range / raw_z_range))

    axis_y = float(axis[1])
    row = {
        "label": series["label"],
        "valid_frames": int(len(positions)),
        "axis_x": float(axis[0]),
        "axis_y": axis_y,
        "axis_z": float(axis[2]),
        "angle_to_board_y_deg": angle_deg,
        "axis_x_per_100mm_board_y": np.nan if abs(axis_y) < 1e-12 else float(100.0 * axis[0] / axis_y),
        "axis_z_per_100mm_board_y": np.nan if abs(axis_y) < 1e-12 else float(100.0 * axis[2] / axis_y),
        "raw_z_range_mm": raw_z_range,
        "axis_corrected_z_range_mm": corrected_z_range,
        "z_range_reduction_percent": reduction,
        "raw_final_z_mm": float(raw_z[-1]),
        "axis_corrected_final_z_mm": float(corrected_z[-1]),
        "path_length_along_axis_mm": finite_range(projection),
        "orthogonal_rms_mm": float(np.sqrt(np.mean(orthogonal_error * orthogonal_error))),
        "orthogonal_p95_mm": float(np.percentile(orthogonal_error, 95)),
        "line_center_x_mm": float(center[0]),
        "line_center_y_mm": float(center[1]),
        "line_center_z_mm": float(center[2]),
    }
    return {
        "summary": row,
        "frames": used_frames,
        "positions": positions,
        "line_points": line_points,
        "projection": projection,
        "raw_z": raw_z,
        "corrected_z": corrected_z,
    }


def movement_axis_summary_rows(series_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series in series_list:
        data = _movement_axis_fit_data(series)
        if data is not None:
            rows.append(data["summary"])
    return rows


def write_movement_axis_summary_csv(path: Path, series_list: list[dict[str, Any]]) -> None:
    rows = movement_axis_summary_rows(series_list)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    columns = [
        ("label", 24),
        ("valid_frames", 6),
        ("z_range_mm", 10),
        ("final_z_mm", 10),
        ("z_vs_y_slope_mm_per_100mm", 12),
        ("step_z_mm", 10),
        ("plane_z_range_mm", 10),
        ("z_corr_centroid_v", 10),
    ]
    print("\n[drift] solver summary")
    print(" ".join(name[:width].ljust(width) for name, width in columns))
    for summary in summaries:
        print(" ".join(fmt(summary.get(name, ""), width) for name, width in columns))


def fmt(value: Any, width: int) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            text = "nan"
        else:
            text = f"{value:.4f}"
    else:
        text = str(value)
    return text[:width].ljust(width)


def plot_series(out_path: Path, series_list: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 7))
    for series in series_list:
        rel, valid, _ = compute_relative(series["tvec_abs"])
        if not np.any(valid):
            continue
        z = rel[:, 2].copy()
        z[~valid] = np.nan
        ax.plot(series["frames"], z, marker=".", linewidth=1.2, markersize=3, label=series["label"])

    ax.axhline(0.0, color="0.75", linestyle="--", linewidth=1.0)
    ax.set_title("Board-relative Z drift by frame")
    ax.set_xlabel("frame")
    ax.set_ylabel("relative T_B_T z [mm]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.count_nonzero(valid)) < 3:
        return np.nan
    residual = y_true[valid] - y_pred[valid]
    centered = y_true[valid] - float(np.mean(y_true[valid]))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum(centered * centered))
    if ss_tot <= 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def plot_z_vs_board_y(out_path: Path, series_list: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    for series in series_list:
        rel, valid, _ = compute_relative(series["tvec_abs"])
        y = rel[:, 1]
        z = rel[:, 2]
        mask = valid & np.isfinite(y) & np.isfinite(z)
        if int(np.count_nonzero(mask)) == 0:
            continue

        label = str(series["label"])
        ax.scatter(y[mask], z[mask], s=18, alpha=0.7, label=label)

        if int(np.count_nonzero(mask)) >= 6 and float(np.ptp(y[mask])) > 1e-9:
            order = np.argsort(y[mask])
            y_sorted = y[mask][order]
            z_sorted = z[mask][order]
            degree = 2 if int(np.count_nonzero(mask)) >= 12 else 1
            coeff = np.polyfit(y_sorted, z_sorted, degree)
            z_fit = np.polyval(coeff, y_sorted)
            r2 = _r2_score(z_sorted, z_fit)
            ax.plot(
                y_sorted,
                z_fit,
                linewidth=2.0,
                alpha=0.9,
                label=f"{label} fit R2={r2:.3f}",
            )

    ax.axhline(0.0, color="0.75", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="0.85", linestyle="--", linewidth=1.0)
    ax.set_title("Board-relative Z vs Board-relative Y")
    ax.set_xlabel("relative T_B_T y [mm]")
    ax.set_ylabel("relative T_B_T z [mm]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_movement_axis_fit(out_path: Path, series_list: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_items = [item for item in (_movement_axis_fit_data(series) for series in series_list) if item is not None]
    if not data_items:
        return

    fig, axes = plt.subplots(len(data_items), 2, figsize=(14, 5 * len(data_items)), squeeze=False)
    for row_idx, data in enumerate(data_items):
        summary = data["summary"]
        positions = data["positions"]
        line_points = data["line_points"]
        frames = data["frames"]
        raw_z = data["raw_z"]
        corrected_z = data["corrected_z"]

        order = np.argsort(line_points[:, 1])
        ax = axes[row_idx, 0]
        ax.scatter(positions[:, 1], positions[:, 2], s=18, alpha=0.65, label="pose")
        ax.plot(
            line_points[order, 1],
            line_points[order, 2],
            linewidth=2.0,
            label=(
                f"axis angle={summary['angle_to_board_y_deg']:.3f} deg, "
                f"z/100y={summary['axis_z_per_100mm_board_y']:.3f} mm"
            ),
        )
        ax.axhline(0.0, color="0.75", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="0.85", linestyle="--", linewidth=1.0)
        ax.set_title(f"{summary['label']}: best movement axis in Board y-z")
        ax.set_xlabel("relative T_B_T y [mm]")
        ax.set_ylabel("relative T_B_T z [mm]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[row_idx, 1]
        ax.plot(frames, raw_z, marker=".", linewidth=1.2, markersize=3, label=f"raw z range {summary['raw_z_range_mm']:.3f} mm")
        ax.plot(
            frames,
            corrected_z,
            marker=".",
            linewidth=1.2,
            markersize=3,
            label=f"axis-corrected z range {summary['axis_corrected_z_range_mm']:.3f} mm",
        )
        ax.axhline(0.0, color="0.75", linestyle="--", linewidth=1.0)
        ax.set_title(f"{summary['label']}: z before/after axis alignment")
        ax.set_xlabel("frame")
        ax.set_ylabel("relative z [mm]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_frame_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    if not value.strip():
        return ranges
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = [int(x.strip()) for x in part.split(":", 1)]
            start, end = min(a, b), max(a, b)
        else:
            start = end = int(part)
        ranges.append((start, end))
    return ranges


def frame_is_excluded(frame: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in ranges)


def main() -> None:
    print(f"[drift] script_version={SCRIPT_VERSION}")
    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] in ("--select", "select"):
        raw_args = raw_args[1:]

    parser = argparse.ArgumentParser(
        description=(
            "Offline drift diagnostics for HydraTracker drift JSONL runs. "
            "Loads one rational8 camera calibration and re-solves stored correspondence points "
            "with the OpenCV iterative PnP solver."
        )
    )
    parser.add_argument("jsonl", type=Path, nargs="?", help="HydraTracker run JSONL")
    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help="Rational8 camera calibration .npz. If omitted, a Qt file dialog opens.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to a drift_solver_debug folder next to the JSONL",
    )
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--exclude-frames", default="", help="Comma-separated frames/ranges to omit, e.g. 1782:1785")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--step-frames", default=f"{DEFAULT_STEP_FRAMES[0]},{DEFAULT_STEP_FRAMES[1]}")
    parser.add_argument("--pre-slope-max-frame", type=int, default=DEFAULT_PRE_SLOPE_MAX_FRAME)
    args = parser.parse_args(raw_args)

    if args.jsonl is None:
        args.jsonl = select_jsonl_with_qt()
        if args.jsonl is None:
            return

    if args.calib is None:
        args.calib = select_calibration_with_qt()
        if args.calib is None:
            return

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV Python module 'cv2' is required for PnP re-solving. "
            "Use the project Python environment that has opencv-python installed."
        ) from exc

    run = load_run(args.jsonl)
    excluded = parse_frame_ranges(args.exclude_frames)
    if excluded:
        kept_frames = [frame for frame in run.frames if not frame_is_excluded(frame.frame, excluded)]
        run = RunData(
            path=run.path,
            run_id=run.run_id,
            timestamp=run.timestamp,
            marker_json_path=run.marker_json_path,
            camera_intrinsics=run.camera_intrinsics,
            board_T_B_C=run.board_T_B_C,
            board_record=run.board_record,
            frames=kept_frames,
        )
        print(
            "[drift] excluded frames "
            + ", ".join(f"{start}:{end}" if start != end else str(start) for start, end in excluded)
        )
    if run.board_T_B_C is None:
        print("[drift] warning: no board_pose/T_B_C record found; camera-frame tvecs will be used")

    out_dir = args.out_dir or default_out_dir_for_jsonl(args.jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.jsonl.stem
    print(f"[drift] out_dir={out_dir}")

    step_parts = [int(x.strip()) for x in args.step_frames.split(",") if x.strip()]
    if len(step_parts) != 2:
        raise ValueError("--step-frames must contain exactly two comma-separated frame numbers")
    step_frames = (step_parts[0], step_parts[1])

    print(f"[drift] run={run.run_id} frames={len(run.frames)}")
    if run.board_record is not None:
        reproj = run.board_record.get("reprojection") or {}
        print(
            "[drift] board reprojection "
            f"mean={_to_float(reproj.get('mean_px')):.3f}px "
            f"p95={_to_float(reproj.get('p95_px')):.3f}px"
        )

    model = load_camera_model_npz(args.calib)
    print(
        "[drift] camera model "
        f"{model.name}: {model.path} "
        f"rms={model.info.get('rms', '')} "
        f"used={model.info.get('num_images_used', '')} "
        f"distortion={model.info.get('distortion_model', '')}"
    )
    camera_model_path = out_dir / "camera_model.csv"
    write_camera_model_csv(camera_model_path, model)
    print(f"[drift] wrote {camera_model_path}")

    spec = SolverSpec("iterative", "SOLVEPNP_ITERATIVE")
    print(f"[drift] solver: {spec.name}")

    logged = logged_series(run)
    series_list: list[dict[str, Any]] = [logged]
    all_solves = solve_sequence(
        cv2,
        run,
        model,
        spec,
        min_points=args.min_points,
        collect_residuals=False,
    )
    series_list.append(series_from_solves(spec.name, all_solves, run.frames))

    summaries = [
        summarize_series(series, step_frames, args.pre_slope_max_frame)
        for series in series_list
    ]
    print_summary(summaries)

    summary_path = out_dir / "summary.csv"
    timeseries_path = out_dir / "timeseries.csv"
    solver_frame_path = out_dir / "solver_frames.csv"
    movement_axis_path = out_dir / "movement_axis_summary.csv"
    write_summary_csv(summary_path, summaries)
    write_timeseries_csv(timeseries_path, series_list)
    write_solver_frame_csv(solver_frame_path, all_solves)
    write_movement_axis_summary_csv(movement_axis_path, series_list)
    print(f"[drift] wrote {summary_path}")
    print(f"[drift] wrote {timeseries_path}")
    print(f"[drift] wrote {solver_frame_path}")
    print(f"[drift] wrote {movement_axis_path}")

    if not args.no_plots:
        z_plot = out_dir / "z_by_frame.png"
        zy_plot = out_dir / "z_vs_board_y.png"
        axis_plot = out_dir / "movement_axis_fit.png"
        try:
            plot_series(z_plot, series_list)
            plot_z_vs_board_y(zy_plot, series_list)
            plot_movement_axis_fit(axis_plot, series_list)
            print(f"[drift] wrote {z_plot}")
            print(f"[drift] wrote {zy_plot}")
            print(f"[drift] wrote {axis_plot}")
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("[drift] plots skipped: matplotlib is not installed")


if __name__ == "__main__":
    main()
