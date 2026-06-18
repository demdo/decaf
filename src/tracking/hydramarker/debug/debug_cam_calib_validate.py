"""Validate saved camera calibration models on independent ChArUco frames.

Workflow:
1. Press SPACE to start recording validation frames from the RealSense color stream.
2. Move the ChArUco board through the full FOV.
3. Press SPACE again to stop; the OpenCV window closes.
4. Select one or more camera calibration NPZ files with the Qt file dialog.
5. The script measures reprojection residuals and writes CSVs/plots.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TRACKING_ROOT = SCRIPT_DIR.parents[1]
CALIB_CAMERA_PATH = TRACKING_ROOT / "hydramarker" / "calib" / "calib_camera.py"

K_KEYS = ("K", "K_rgb", "camera_matrix", "camera_intrinsics", "intrinsics")
DIST_KEYS = (
    "dist",
    "dist_rgb",
    "dist_coeffs",
    "distortion_coeffs",
    "opencv_dist_coeffs",
    "effective_opencv_dist_coeffs",
)

WINDOW_NAME = "HydraMarker Camera Calibration Validation"


def _load_calib_camera_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hydramarker_calib_camera_runtime",
        CALIB_CAMERA_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load calibration helpers from {CALIB_CAMERA_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calib_camera = _load_calib_camera_module()


@dataclass(frozen=True)
class CameraModel:
    name: str
    path: Path
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int] | None
    info: dict[str, Any]


@dataclass(frozen=True)
class ValidationFrame:
    index: int
    path: Path
    capture_time_s: float
    num_charuco: int
    num_aruco: int


@dataclass(frozen=True)
class DetectedValidationImage:
    index: int
    path: Path
    image: np.ndarray
    det: Any
    image_size: tuple[int, int]


def _ensure_qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is required for the file dialog. Pass --calib on the command line "
            "or run in the Qt-enabled project environment."
        ) from exc

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def select_calibrations_with_qt() -> list[Path] | None:
    try:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
    except ImportError as exc:
        raise RuntimeError(
            "No calibration NPZ files were given and PySide6 is not available for the file dialog. "
            "Pass --calib paths on the command line or use the Qt-enabled environment."
        ) from exc

    _ensure_qt_app()
    default_dir = TRACKING_ROOT / "hydramarker" / "data" / "realsense"
    if not default_dir.exists():
        default_dir = Path.cwd()

    paths, _ = QFileDialog.getOpenFileNames(
        None,
        "Select camera calibration NPZ files to validate",
        str(default_dir),
        "Camera calibration NPZ (*.npz);;All Files (*)",
    )
    if not paths:
        QMessageBox.information(
            None,
            "Camera calibration validation",
            "No calibration files selected.",
        )
        return None
    return [Path(path) for path in paths]


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "cam_calib_validation" / f"validation_{stamp}"


def first_npz_array(npz: Any, names: tuple[str, ...]) -> tuple[np.ndarray | None, str]:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name], dtype=np.float64), name
    return None, ""


def scalar_npz_value(npz: Any, key: str) -> Any:
    if key not in npz.files:
        return ""
    arr = np.asarray(npz[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def read_npz_image_size(npz: Any) -> tuple[int, int] | None:
    if "image_size" in npz.files:
        values = np.asarray(npz["image_size"]).reshape(-1)
        if values.size >= 2:
            return int(values[0]), int(values[1])

    width_keys = ("width", "image_width", "rgb_width")
    height_keys = ("height", "image_height", "rgb_height")
    width = next(
        (int(np.asarray(npz[key]).reshape(-1)[0]) for key in width_keys if key in npz.files),
        None,
    )
    height = next(
        (int(np.asarray(npz[key]).reshape(-1)[0]) for key in height_keys if key in npz.files),
        None,
    )
    if width is not None and height is not None:
        return width, height
    return None


def load_camera_model(path: Path) -> CameraModel:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as npz:
        K, K_key = first_npz_array(npz, K_KEYS)
        if K is None:
            raise KeyError(f"{path} must contain one of: {', '.join(K_KEYS)}")

        dist, dist_key = first_npz_array(npz, DIST_KEYS)
        if dist is None:
            raise KeyError(f"{path} must contain one of: {', '.join(DIST_KEYS)}")

        image_size = read_npz_image_size(npz)
        info = {
            "K_key": K_key,
            "dist_key": dist_key,
            "image_size": image_size,
            "created_at": scalar_npz_value(npz, "created_at"),
            "rms": scalar_npz_value(npz, "rms"),
            "calibration_rms": scalar_npz_value(npz, "calibration_rms"),
            "num_images_used": scalar_npz_value(npz, "num_images_used"),
            "selected_reprojection_mean_px": scalar_npz_value(
                npz,
                "selected_reprojection_mean_px",
            ),
            "distortion_model": scalar_npz_value(npz, "distortion_model"),
        }

    return CameraModel(
        name=path.stem,
        path=path,
        K=np.asarray(K, dtype=np.float64).reshape(3, 3),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        image_size=image_size,
        info=info,
    )


def draw_validation_overlay(
    frame_bgr: np.ndarray,
    det: Any | None,
    *,
    recording: bool,
    saved_count: int,
    min_corners: int,
) -> np.ndarray:
    vis = frame_bgr.copy()
    if det is not None:
        if det.aruco_ids is not None and len(det.aruco_ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, det.aruco_corners, det.aruco_ids)
        if det.charuco_corners is not None and det.charuco_ids is not None and det.num_charuco > 0:
            try:
                cv2.aruco.drawDetectedCornersCharuco(
                    vis,
                    det.charuco_corners,
                    det.charuco_ids,
                    (255, 255, 0),
                )
            except Exception:
                for u, v in det.charuco_corners.reshape(-1, 2):
                    cv2.circle(vis, (int(round(u)), int(round(v))), 4, (255, 255, 0), 2)

    charuco = "-" if det is None else str(int(det.num_charuco))
    aruco = "-" if det is None else str(int(det.num_aruco))
    good = det is not None and det.num_charuco >= min_corners
    if recording:
        status = "RECORDING VALIDATION FRAMES (SPACE stops)"
        color = (0, 210, 255)
    else:
        status = "READY (SPACE starts recording)"
        color = (0, 255, 0) if good else (0, 0, 255)

    lines = [
        status,
        f"Saved validation frames: {saved_count}",
        f"ArUco: {aruco}  ChArUco: {charuco}  min: {min_corners}",
        "Move board through center, edges, corners, near/far, tilted views.",
        "Keys: SPACE start/stop | Q/ESC quit",
    ]
    return calib_camera.draw_text_box(vis, lines, color=color)


def capture_validation_frames(
    *,
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    min_corners: int,
    capture_interval_s: float,
    max_frames: int,
) -> list[ValidationFrame]:
    board, aruco_dict, detector_params = calib_camera.make_charuco_board()
    frames_dir = output_dir / "validation_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pipeline = None
    saved: list[ValidationFrame] = []
    recording = False
    recording_started_s: float | None = None
    last_save_s = -float("inf")
    frame_index = 0
    last_frame: np.ndarray | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        pipeline, _profile = calib_camera.start_realsense(width=width, height=height, fps=fps)
        print("[cam_calib_validate] RealSense running.")
        print("[cam_calib_validate] SPACE: start/stop recording validation frames")
        print("[cam_calib_validate] Q/ESC: quit")

        while True:
            frame = calib_camera.get_color_frame_bgr(pipeline)
            got_new_frame = frame is not None
            if got_new_frame:
                frame_index += 1
            else:
                frame = last_frame
            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            last_frame = frame
            det = calib_camera.detect_charuco(
                frame,
                board=board,
                aruco_dict=aruco_dict,
                detector_params=detector_params,
            )
            now_s = time.monotonic()

            if (
                recording
                and got_new_frame
                and det.num_charuco >= min_corners
                and len(saved) < max_frames
                and now_s - last_save_s >= capture_interval_s
            ):
                if recording_started_s is None:
                    recording_started_s = now_s
                image_path = frames_dir / f"validation_{len(saved):04d}_frame_{frame_index:06d}.png"
                cv2.imwrite(str(image_path), frame)
                saved.append(
                    ValidationFrame(
                        index=frame_index,
                        path=image_path,
                        capture_time_s=now_s - recording_started_s,
                        num_charuco=int(det.num_charuco),
                        num_aruco=int(det.num_aruco),
                    )
                )
                last_save_s = now_s
                if len(saved) % 25 == 0:
                    print(
                        "[cam_calib_validate] Saved "
                        f"{len(saved)} validation frames "
                        f"(latest {det.num_charuco} ChArUco corners)."
                    )

            vis = draw_validation_overlay(
                frame,
                det,
                recording=recording,
                saved_count=len(saved),
                min_corners=min_corners,
            )
            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                saved = []
                break

            if key == 32:
                if not recording:
                    recording = True
                    recording_started_s = None
                    last_save_s = -float("inf")
                    print("[cam_calib_validate] Recording started.")
                else:
                    recording = False
                    if saved:
                        print(
                            "[cam_calib_validate] Recording stopped. "
                            f"{len(saved)} frames saved."
                        )
                        break
                    print("[cam_calib_validate] No valid frames saved yet; keep recording or try again.")

            if len(saved) >= max_frames:
                print(f"[cam_calib_validate] Reached max validation frames: {max_frames}")
                break

    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyWindow(WINDOW_NAME)
        cv2.waitKey(1)

    return saved


def collect_image_paths(images_dir: Path) -> list[Path]:
    images_dir = Path(images_dir).expanduser().resolve()
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in suffixes)
    if not paths:
        raise RuntimeError(f"No validation images found in {images_dir}")
    return paths


def load_detected_images(
    image_paths: Sequence[Path],
    *,
    min_corners: int,
) -> tuple[list[DetectedValidationImage], list[dict[str, Any]]]:
    board, aruco_dict, detector_params = calib_camera.make_charuco_board()
    detected: list[DetectedValidationImage] = []
    rows: list[dict[str, Any]] = []

    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append(
                {
                    "image_index": index,
                    "image_path": str(path),
                    "valid": 0,
                    "num_charuco": 0,
                    "num_aruco": 0,
                    "reason": "read_failed",
                }
            )
            continue

        det = calib_camera.detect_charuco(
            image,
            board=board,
            aruco_dict=aruco_dict,
            detector_params=detector_params,
        )
        valid = det.charuco_corners is not None and det.charuco_ids is not None and det.num_charuco >= min_corners
        rows.append(
            {
                "image_index": index,
                "image_path": str(path),
                "valid": int(valid),
                "num_charuco": int(det.num_charuco),
                "num_aruco": int(det.num_aruco),
                "reason": "" if valid else "not_enough_charuco",
            }
        )
        if not valid:
            continue

        detected.append(
            DetectedValidationImage(
                index=index,
                path=path,
                image=image,
                det=det,
                image_size=calib_camera._image_size(image),
            )
        )

    return detected, rows


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def image_radius_norm(points: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    x = (points[:, 0] - 0.5 * (width - 1)) / max(0.5 * (width - 1), 1.0)
    y = (points[:, 1] - 0.5 * (height - 1)) / max(0.5 * (height - 1), 1.0)
    return np.hypot(x, y) / np.sqrt(2.0)


def normalized_camera_radius(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = (points[:, 0] - float(K[0, 2])) / float(K[0, 0])
    y = (points[:, 1] - float(K[1, 2])) / float(K[1, 1])
    return np.hypot(x, y)


def validate_model_on_images(
    model: CameraModel,
    images: Sequence[DetectedValidationImage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    board, _aruco_dict, _detector_params = calib_camera.make_charuco_board()
    per_image_rows: list[dict[str, Any]] = []
    per_corner_rows: list[dict[str, Any]] = []

    for item in images:
        det = item.det
        obj_pts = calib_camera._charuco_object_points(board, det.charuco_ids)
        img_pts = det.charuco_corners.reshape(-1, 2).astype(np.float32)

        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                model.K,
                model.dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as exc:
            per_image_rows.append(
                {
                    "model": model.name,
                    "image_index": item.index,
                    "image_path": str(item.path),
                    "valid": 0,
                    "point_count": int(len(img_pts)),
                    "reason": f"solvePnP_error: {exc}",
                }
            )
            continue

        if not ok:
            per_image_rows.append(
                {
                    "model": model.name,
                    "image_index": item.index,
                    "image_path": str(item.path),
                    "valid": 0,
                    "point_count": int(len(img_pts)),
                    "reason": "solvePnP_failed",
                }
            )
            continue

        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, model.K, model.dist)
        projected = projected.reshape(-1, 2).astype(np.float64)
        measured = img_pts.astype(np.float64)
        residual = measured - projected
        err = np.linalg.norm(residual, axis=1)
        centroid = np.mean(measured, axis=0)
        span = np.ptp(measured, axis=0)
        img_radius = image_radius_norm(measured, item.image_size)
        cam_radius = normalized_camera_radius(measured, model.K)

        per_image_rows.append(
            {
                "model": model.name,
                "image_index": item.index,
                "image_path": str(item.path),
                "valid": 1,
                "point_count": int(len(err)),
                "mean_px": float(np.mean(err)),
                "median_px": float(np.median(err)),
                "rms_px": float(np.sqrt(np.mean(err * err))),
                "p95_px": percentile(err, 95),
                "p99_px": percentile(err, 99),
                "max_px": float(np.max(err)),
                "mean_du_px": float(np.mean(residual[:, 0])),
                "mean_dv_px": float(np.mean(residual[:, 1])),
                "mean_abs_du_px": float(np.mean(np.abs(residual[:, 0]))),
                "mean_abs_dv_px": float(np.mean(np.abs(residual[:, 1]))),
                "centroid_u_px": float(centroid[0]),
                "centroid_v_px": float(centroid[1]),
                "span_u_px": float(span[0]),
                "span_v_px": float(span[1]),
                "radius_img_mean": float(np.mean(img_radius)),
                "radius_cam_mean": float(np.mean(cam_radius)),
                "rvec_x": float(rvec.reshape(-1)[0]),
                "rvec_y": float(rvec.reshape(-1)[1]),
                "rvec_z": float(rvec.reshape(-1)[2]),
                "tvec_x_m": float(tvec.reshape(-1)[0]),
                "tvec_y_m": float(tvec.reshape(-1)[1]),
                "tvec_z_m": float(tvec.reshape(-1)[2]),
                "reason": "",
            }
        )

        ids = det.charuco_ids.reshape(-1).astype(int)
        width, height = item.image_size
        for idx, charuco_id in enumerate(ids):
            per_corner_rows.append(
                {
                    "model": model.name,
                    "image_index": item.index,
                    "image_path": str(item.path),
                    "charuco_id": int(charuco_id),
                    "u_px": float(measured[idx, 0]),
                    "v_px": float(measured[idx, 1]),
                    "projected_u_px": float(projected[idx, 0]),
                    "projected_v_px": float(projected[idx, 1]),
                    "du_px": float(residual[idx, 0]),
                    "dv_px": float(residual[idx, 1]),
                    "error_px": float(err[idx]),
                    "radius_img_norm": float(img_radius[idx]),
                    "radius_cam": float(cam_radius[idx]),
                    "grid_col": int(np.clip(np.floor(measured[idx, 0] / max(width, 1) * 8), 0, 7)),
                    "grid_row": int(np.clip(np.floor(measured[idx, 1] / max(height, 1) * 6), 0, 5)),
                }
            )

    return per_image_rows, per_corner_rows


def summarize_model(
    model: CameraModel,
    per_image_rows: Sequence[dict[str, Any]],
    per_corner_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    valid_images = [row for row in per_image_rows if int(row.get("valid", 0)) == 1]
    if not per_corner_rows:
        return {
            "model": model.name,
            "path": str(model.path),
            "valid_images": len(valid_images),
            "points_total": 0,
            "status": "no_points",
        }

    errors = np.asarray([float(row["error_px"]) for row in per_corner_rows], dtype=np.float64)
    du = np.asarray([float(row["du_px"]) for row in per_corner_rows], dtype=np.float64)
    dv = np.asarray([float(row["dv_px"]) for row in per_corner_rows], dtype=np.float64)
    r_img = np.asarray([float(row["radius_img_norm"]) for row in per_corner_rows], dtype=np.float64)

    slope = float("nan")
    corr = float("nan")
    if len(errors) >= 3 and np.std(r_img) > 1e-12 and np.std(errors) > 1e-12:
        slope = float(np.polyfit(r_img, errors, deg=1)[0])
        corr = float(np.corrcoef(r_img, errors)[0, 1])

    center = errors[r_img < 0.35]
    middle = errors[(r_img >= 0.35) & (r_img < 0.70)]
    edge = errors[r_img >= 0.70]

    return {
        "model": model.name,
        "path": str(model.path),
        "status": "ok",
        "valid_images": len(valid_images),
        "points_total": int(errors.size),
        "mean_px": float(np.mean(errors)),
        "median_px": float(np.median(errors)),
        "rms_px": float(np.sqrt(np.mean(errors * errors))),
        "p95_px": percentile(errors, 95),
        "p99_px": percentile(errors, 99),
        "max_px": float(np.max(errors)),
        "mean_du_px": float(np.mean(du)),
        "mean_dv_px": float(np.mean(dv)),
        "mean_abs_du_px": float(np.mean(np.abs(du))),
        "mean_abs_dv_px": float(np.mean(np.abs(dv))),
        "center_mean_px": float(np.mean(center)) if center.size else float("nan"),
        "middle_mean_px": float(np.mean(middle)) if middle.size else float("nan"),
        "edge_mean_px": float(np.mean(edge)) if edge.size else float("nan"),
        "error_vs_radius_slope_px": slope,
        "error_vs_radius_corr": corr,
        "calibration_rms": model.info.get("rms") or model.info.get("calibration_rms"),
        "calibration_selected_reprojection_mean_px": model.info.get(
            "selected_reprojection_mean_px",
        ),
        "distortion_model": model.info.get("distortion_model"),
        "dist_coeff_count": int(model.dist.reshape(-1).size),
        "fx": float(model.K[0, 0]),
        "fy": float(model.K[1, 1]),
        "cx": float(model.K[0, 2]),
        "cy": float(model.K[1, 2]),
        "dist": " ".join(f"{x:.12g}" for x in model.dist.reshape(-1)),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_capture_manifest(path: Path, frames: Sequence[ValidationFrame]) -> None:
    rows = [
        {
            "saved_index": i,
            "frame_index": frame.index,
            "image_path": str(frame.path),
            "capture_time_s": frame.capture_time_s,
            "num_charuco": frame.num_charuco,
            "num_aruco": frame.num_aruco,
        }
        for i, frame in enumerate(frames)
    ]
    write_csv(path, rows)


def _as_float_array(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def plot_outputs(
    *,
    output_dir: Path,
    models: Sequence[CameraModel],
    per_image_rows: Sequence[dict[str, Any]],
    per_corner_rows: Sequence[dict[str, Any]],
    image_size: tuple[int, int],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[cam_calib_validate] Matplotlib unavailable; skipping plots: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [model.name for model in models]

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in model_names:
        rows = [row for row in per_corner_rows if row["model"] == name]
        if not rows:
            continue
        radius = _as_float_array(rows, "radius_img_norm")
        error = _as_float_array(rows, "error_px")
        ax.scatter(radius, error, s=8, alpha=0.25, label=name)
        bins = np.linspace(0.0, 1.0, 11)
        xs: list[float] = []
        ys: list[float] = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (radius >= lo) & (radius < hi)
            if np.count_nonzero(mask) < 3:
                continue
            xs.append(float(0.5 * (lo + hi)))
            ys.append(float(np.mean(error[mask])))
        if xs:
            ax.plot(xs, ys, linewidth=2.0)
    ax.set_title("Validation reprojection error vs image radius")
    ax.set_xlabel("image radius norm")
    ax.set_ylabel("corner reprojection error [px]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "residual_vs_radius.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in model_names:
        rows = [row for row in per_image_rows if row["model"] == name and int(row.get("valid", 0)) == 1]
        if not rows:
            continue
        image_idx = _as_float_array(rows, "image_index")
        rms = _as_float_array(rows, "rms_px")
        ax.plot(image_idx, rms, marker="o", markersize=3, linewidth=1.2, label=name)
    ax.set_title("Validation reprojection RMS by frame")
    ax.set_xlabel("validation image index")
    ax.set_ylabel("RMS reprojection error [px]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "residual_by_frame.png", dpi=160)
    plt.close(fig)

    width, height = image_size
    cols = 8
    rows_n = 6
    fig, axes = plt.subplots(
        1,
        max(1, len(model_names)),
        figsize=(5.4 * max(1, len(model_names)), 4.6),
        squeeze=False,
    )
    for ax, name in zip(axes[0], model_names):
        rows = [row for row in per_corner_rows if row["model"] == name]
        heat = np.full((rows_n, cols), np.nan, dtype=np.float64)
        for grid_row in range(rows_n):
            for grid_col in range(cols):
                values = [
                    float(row["error_px"])
                    for row in rows
                    if int(row["grid_col"]) == grid_col and int(row["grid_row"]) == grid_row
                ]
                if values:
                    heat[grid_row, grid_col] = float(np.mean(values))
        im = ax.imshow(
            heat,
            origin="upper",
            extent=(0, width, height, 0),
            cmap="magma",
            aspect="auto",
        )
        ax.set_title(name)
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
        fig.colorbar(im, ax=ax, label="mean error [px]")
    fig.suptitle("Validation residual heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_heatmap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        max(1, len(model_names)),
        figsize=(5.4 * max(1, len(model_names)), 4.6),
        squeeze=False,
    )
    arrow_scale = 25.0
    for ax, name in zip(axes[0], model_names):
        rows = [row for row in per_corner_rows if row["model"] == name]
        xs: list[float] = []
        ys: list[float] = []
        us: list[float] = []
        vs: list[float] = []
        for grid_row in range(rows_n):
            for grid_col in range(cols):
                cell = [
                    row
                    for row in rows
                    if int(row["grid_col"]) == grid_col and int(row["grid_row"]) == grid_row
                ]
                if len(cell) < 3:
                    continue
                xs.append(float(np.mean([float(row["u_px"]) for row in cell])))
                ys.append(float(np.mean([float(row["v_px"]) for row in cell])))
                us.append(float(np.mean([float(row["du_px"]) for row in cell])) * arrow_scale)
                vs.append(float(np.mean([float(row["dv_px"]) for row in cell])) * arrow_scale)
        ax.imshow(np.ones((2, 2, 3), dtype=np.float64), extent=(0, width, height, 0), aspect="auto")
        if xs:
            ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1.0, color="tab:red")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title(f"{name}\nmean residual vectors x{arrow_scale:g}")
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Validation residual vector field")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_vector_field.png", dpi=160)
    plt.close(fig)


def run_validation(
    *,
    image_paths: Sequence[Path],
    calib_paths: Sequence[Path],
    output_dir: Path,
    min_corners: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [load_camera_model(path) for path in calib_paths]
    detected_images, detection_rows = load_detected_images(image_paths, min_corners=min_corners)

    write_csv(output_dir / "validation_image_detections.csv", detection_rows)
    if not detected_images:
        raise RuntimeError("No validation images with enough ChArUco corners were found.")

    image_size = detected_images[0].image_size
    all_per_image_rows: list[dict[str, Any]] = []
    all_per_corner_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model in models:
        if model.image_size is not None and tuple(model.image_size) != tuple(image_size):
            print(
                "[cam_calib_validate] Warning: image size mismatch for "
                f"{model.name}: model={model.image_size}, validation={image_size}"
            )
        per_image, per_corner = validate_model_on_images(model, detected_images)
        all_per_image_rows.extend(per_image)
        all_per_corner_rows.extend(per_corner)
        summary = summarize_model(model, per_image, per_corner)
        summary_rows.append(summary)
        print(
            "[cam_calib_validate] "
            f"{model.name}: valid_images={summary.get('valid_images')} "
            f"points={summary.get('points_total')} "
            f"rms={float(summary.get('rms_px', float('nan'))):.4f}px "
            f"p95={float(summary.get('p95_px', float('nan'))):.4f}px "
            f"edge_mean={float(summary.get('edge_mean_px', float('nan'))):.4f}px "
            f"slope={float(summary.get('error_vs_radius_slope_px', float('nan'))):.4f}"
        )

    write_csv(output_dir / "validation_summary.csv", summary_rows)
    write_csv(output_dir / "validation_residuals_by_image.csv", all_per_image_rows)
    write_csv(output_dir / "validation_residuals_by_corner.csv", all_per_corner_rows)
    plot_outputs(
        output_dir=output_dir,
        models=models,
        per_image_rows=all_per_image_rows,
        per_corner_rows=all_per_corner_rows,
        image_size=image_size,
    )

    print(f"[cam_calib_validate] Saved validation outputs -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture independent ChArUco validation frames and compare camera calibration NPZ models.",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        nargs="*",
        default=None,
        help="Camera calibration NPZ files. If omitted, a Qt file dialog opens after capture.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Validate existing images from this directory instead of recording RealSense frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for frames, CSVs and plots.",
    )
    parser.add_argument("--width", type=int, default=calib_camera.REALSENSE_WIDTH)
    parser.add_argument("--height", type=int, default=calib_camera.REALSENSE_HEIGHT)
    parser.add_argument("--fps", type=int, default=calib_camera.REALSENSE_FPS)
    parser.add_argument(
        "--min-corners",
        type=int,
        default=calib_camera.MIN_CHARUCO_CAPTURE,
        help="Minimum ChArUco corners for a validation frame.",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=calib_camera.AUTO_CAPTURE_INTERVAL_S,
        help="Minimum seconds between saved validation frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Maximum number of validation frames to save.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.images_dir is None:
        frames = capture_validation_frames(
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            min_corners=args.min_corners,
            capture_interval_s=args.capture_interval,
            max_frames=args.max_frames,
        )
        if not frames:
            raise RuntimeError("No validation frames were recorded.")
        write_capture_manifest(output_dir / "validation_capture_manifest.csv", frames)
        image_paths = [frame.path for frame in frames]
    else:
        image_paths = collect_image_paths(args.images_dir)

    calib_paths = list(args.calib) if args.calib else None
    if not calib_paths:
        calib_paths = select_calibrations_with_qt()
    if not calib_paths:
        raise RuntimeError("No camera calibration models selected.")

    run_validation(
        image_paths=image_paths,
        calib_paths=calib_paths,
        output_dir=output_dir,
        min_corners=args.min_corners,
    )


if __name__ == "__main__":
    main()
