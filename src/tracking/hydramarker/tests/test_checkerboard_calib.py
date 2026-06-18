from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


def _ensure_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[3]
    src = str(src_root)
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

from tracking.hydramarker.calib import calib_checkerboard


WINDOW_NAME = "HydraMarker Checkerboard Calib Repeat Test"
DEFAULT_RUNS = 10
DEFAULT_SECONDS_PER_RUN = 3.0
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "checkerboard_calib_runs"


@dataclass
class RunResult:
    run_index: int
    detections: int
    pose: calib_checkerboard.CheckerboardPose | None
    error: str | None = None


def _first_npz_array(npz: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in npz:
            return np.asarray(npz[name], dtype=np.float64)
    return None


def choose_file_qt(title: str, file_filter: str) -> Path | None:
    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        str(Path.cwd()),
        file_filter,
    )
    return Path(path).expanduser().resolve() if path else None


def load_camera_calibration(path: Path | None) -> tuple[np.ndarray, np.ndarray, Path]:
    if path is None:
        path = choose_file_qt(
            "Select camera calibration .npz",
            "NPZ files (*.npz);;All Files (*)",
        )
    if path is None:
        raise RuntimeError("No camera calibration file selected.")

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {path}")

    with np.load(path, allow_pickle=True) as npz:
        K = _first_npz_array(
            npz,
            ("K", "K_rgb", "camera_matrix", "camera_intrinsics", "intrinsics"),
        )
        if K is None:
            raise KeyError(
                "Camera calibration NPZ must contain one of: "
                "K, K_rgb, camera_matrix, camera_intrinsics, intrinsics."
            )

        dist = _first_npz_array(
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
                "Camera calibration NPZ must contain OpenCV distortion coefficients: "
                "dist, dist_rgb, dist_coeffs, distortion_coeffs, opencv_dist_coeffs, "
                "or effective_opencv_dist_coeffs."
            )

    return (
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        path,
    )


def _put_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.7,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_overlay(
    frame: np.ndarray,
    *,
    corners: np.ndarray | None,
    collecting: bool,
    started: bool,
    run_index: int,
    total_runs: int,
    detections: int,
    seconds_left: float,
) -> np.ndarray:
    vis = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(
            vis,
            calib_checkerboard.CHECKERBOARD_PATTERN,
            np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
            True,
        )

    if not started:
        status = f"SPACE starts {total_runs}x checkerboard calibration capture | q quits"
        color = (0, 255, 255) if corners is not None else (0, 0, 255)
    elif collecting:
        status = (
            f"RUN {run_index}/{total_runs} collecting "
            f"{max(0.0, seconds_left):.1f}s | detections={detections}"
        )
        color = (0, 210, 255)
    else:
        status = "Preparing next run..."
        color = (0, 255, 0)

    _put_text(vis, status, (28, 46), color=color)
    _put_text(
        vis,
        "Keep checkerboard and camera rigid during the full sequence.",
        (28, 78),
        color=(255, 255, 255),
        scale=0.62,
    )
    return vis


def _pose_record(result: RunResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "checkerboard_calib_run",
        "run_index": result.run_index,
        "detections": result.detections,
        "ok": result.pose is not None,
    }
    if result.error:
        record["error"] = result.error
    if result.pose is None:
        return record

    pose = result.pose
    quality = calib_checkerboard.pose_quality_dict(pose)
    record.update(
        {
            "frames_total": quality["frames_total"],
            "frames_used": quality["frames_used"],
            "reprojection_mean_px": pose.reproj_mean_px,
            "reprojection_median_px": pose.reproj_median_px,
            "reprojection_p95_px": pose.reproj_p95_px,
            "reprojection_max_px": pose.reproj_max_px,
            "corner_std_mean_px": pose.mean_corner_std_px,
            "corner_std_max_px": pose.max_corner_std_px,
            "pnp_flag": int(pose.pnp_flag),
            "solver_mode": pose.solver_mode,
            "candidate_count": pose.candidate_count,
            "selected_candidate_index": pose.selected_candidate_index,
            "alternative_rms_px": pose.alternative_rms_px,
            "alternative_error_gap_px": pose.alternative_error_gap_px,
            "alternative_error_ratio": pose.alternative_error_ratio,
            "alternative_likelihood_ratio": pose.alternative_likelihood_ratio,
            "alternative_translation_delta_mm": pose.alternative_translation_delta_mm,
            "alternative_rotation_delta_deg": pose.alternative_rotation_delta_deg,
            "pose_ambiguous": pose.pose_ambiguous,
            "rvec_cb": pose.rvec_cb.reshape(3).tolist(),
            "tvec_cb_mm": pose.tvec_cb_mm.reshape(3).tolist(),
            "camera_in_board_mm": pose.T_B_C[:3, 3].reshape(3).tolist(),
            "T_C_B": pose.T_C_B.tolist(),
            "T_B_C": pose.T_B_C.tolist(),
        }
    )
    return record


def _rotation_delta_deg(a: calib_checkerboard.CheckerboardPose, b: calib_checkerboard.CheckerboardPose) -> float:
    Ra, _ = cv2.Rodrigues(a.rvec_cb.reshape(3, 1))
    Rb, _ = cv2.Rodrigues(b.rvec_cb.reshape(3, 1))
    R = Rb @ Ra.T
    cos_angle = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _print_result(result: RunResult, first_pose: calib_checkerboard.CheckerboardPose | None) -> None:
    if result.pose is None:
        print(
            f"{result.run_index:>3} | FAIL | detections={result.detections:>3} | "
            f"{result.error or 'pose estimation failed'}"
        )
        return

    pose = result.pose
    t = pose.tvec_cb_mm.reshape(3)
    c = pose.T_B_C[:3, 3].reshape(3)
    rot_delta = 0.0 if first_pose is None else _rotation_delta_deg(first_pose, pose)
    print(
        f"{result.run_index:>3} | ok   | det={result.detections:>3} "
        f"used={pose.frames_used:>3}/{pose.frames_total:<3} "
        f"reproj mean/p95/max={pose.reproj_mean_px:6.3f}/"
        f"{pose.reproj_p95_px:6.3f}/{pose.reproj_max_px:6.3f}px "
        f"corner_std={pose.mean_corner_std_px:6.3f}px "
        f"board_z_cam={t[2]:8.3f}mm "
        f"cam_z_board={c[2]:8.3f}mm "
        f"rot_d0={rot_delta:7.4f}deg "
        f"alt_gap={pose.alternative_error_gap_px:7.4f}px "
        f"amb={int(pose.pose_ambiguous)}"
    )


def _stats_line(label: str, values: np.ndarray, unit: str) -> str:
    values = np.asarray(values, dtype=np.float64)
    return (
        f"{label}: mean={np.mean(values):.4f}{unit}, "
        f"std={np.std(values):.4f}{unit}, "
        f"range={np.ptp(values):.4f}{unit}, "
        f"min={np.min(values):.4f}{unit}, max={np.max(values):.4f}{unit}"
    )


def print_summary(results: list[RunResult]) -> None:
    ok = [result for result in results if result.pose is not None]
    print("\n[checkerboard_calib] summary")
    print(f"successful runs: {len(ok)}/{len(results)}")
    if not ok:
        return

    poses = [result.pose for result in ok if result.pose is not None]
    board_tvecs = np.asarray([pose.tvec_cb_mm.reshape(3) for pose in poses], dtype=np.float64)
    camera_in_board = np.asarray([pose.T_B_C[:3, 3].reshape(3) for pose in poses], dtype=np.float64)
    reproj_mean = np.asarray([pose.reproj_mean_px for pose in poses], dtype=np.float64)
    reproj_p95 = np.asarray([pose.reproj_p95_px for pose in poses], dtype=np.float64)
    corner_std = np.asarray([pose.mean_corner_std_px for pose in poses], dtype=np.float64)
    rot_delta = np.asarray([_rotation_delta_deg(poses[0], pose) for pose in poses], dtype=np.float64)
    alt_gap = np.asarray([pose.alternative_error_gap_px for pose in poses], dtype=np.float64)
    alt_lr = np.asarray(
        [pose.alternative_likelihood_ratio for pose in poses],
        dtype=np.float64,
    )

    print(_stats_line("reprojection mean", reproj_mean, "px"))
    print(_stats_line("reprojection p95", reproj_p95, "px"))
    print(_stats_line("corner std mean", corner_std, "px"))
    if np.any(np.isfinite(alt_gap)):
        print(_stats_line("alternative gap", alt_gap[np.isfinite(alt_gap)], "px"))
    if np.any(np.isfinite(alt_lr)):
        print(_stats_line("alternative likelihood", alt_lr[np.isfinite(alt_lr)], ""))
    print(f"ambiguous runs: {sum(1 for pose in poses if pose.pose_ambiguous)}")
    print(_stats_line("board origin z in camera", board_tvecs[:, 2], "mm"))
    print(_stats_line("camera origin z in board", camera_in_board[:, 2], "mm"))
    print(_stats_line("rotation delta vs run 1", rot_delta, "deg"))
    print(
        "tvec std xyz board-in-camera [mm]: "
        f"x={np.std(board_tvecs[:, 0]):.4f}, "
        f"y={np.std(board_tvecs[:, 1]):.4f}, "
        f"z={np.std(board_tvecs[:, 2]):.4f}"
    )
    print(
        "T_B_C translation std xyz camera-in-board [mm]: "
        f"x={np.std(camera_in_board[:, 0]):.4f}, "
        f"y={np.std(camera_in_board[:, 1]):.4f}, "
        f"z={np.std(camera_in_board[:, 2]):.4f}"
    )


def write_jsonl(
    results: list[RunResult],
    *,
    output_dir: Path,
    calibration_path: Path,
    runs: int,
    seconds_per_run: float,
    capture_interval_s: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"checkerboard_calib_{stamp}.jsonl"
    header = {
        "type": "checkerboard_calib_session",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "calibration_path": str(calibration_path),
        "runs": int(runs),
        "seconds_per_run": float(seconds_per_run),
        "capture_interval_s": float(capture_interval_s),
        "pattern_inner_corners": list(calib_checkerboard.CHECKERBOARD_PATTERN),
        "square_size_mm": float(calib_checkerboard.CHECKERBOARD_SQUARE_SIZE_MM),
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for result in results:
            f.write(json.dumps(_pose_record(result)) + "\n")
    return path


def collect_repeated_calibrations(
    pipe,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    runs: int,
    seconds_per_run: float,
    capture_interval_s: float,
    min_frames: int,
) -> list[RunResult]:
    results: list[RunResult] = []
    global_reference: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    started = False
    collecting = False
    current_run = 0
    run_started_s = 0.0
    last_capture_s = -float("inf")
    detections: list[np.ndarray] = []
    first_pose: calib_checkerboard.CheckerboardPose | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print(f"[checkerboard_calib] SPACE starts the {runs}-run capture, q/ESC quits.")

    try:
        while True:
            frame = calib_checkerboard.wait_color_frame_bgr(pipe)
            if frame is not None:
                last_frame = frame
            elif last_frame is not None:
                frame = last_frame
            else:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            corners = calib_checkerboard.detect_checkerboard_corners(frame)
            if corners is not None:
                if global_reference is None:
                    global_reference = corners
                corners = calib_checkerboard.align_detection_to_reference(corners, global_reference)

            now_s = time.monotonic()
            seconds_left = 0.0
            if collecting:
                elapsed_s = now_s - run_started_s
                seconds_left = seconds_per_run - elapsed_s
                if (
                    corners is not None
                    and now_s - last_capture_s >= capture_interval_s
                ):
                    detections.append(corners.copy())
                    last_capture_s = now_s

                if elapsed_s >= seconds_per_run:
                    result = _estimate_pose_for_run(
                        current_run,
                        detections,
                        K,
                        dist,
                        min_frames=min_frames,
                    )
                    if first_pose is None and result.pose is not None:
                        first_pose = result.pose
                    results.append(result)
                    _print_result(result, first_pose)

                    collecting = False
                    if current_run >= runs:
                        break

                    current_run += 1
                    detections = []
                    run_started_s = time.monotonic()
                    last_capture_s = -float("inf")
                    collecting = True

            vis = _draw_overlay(
                frame,
                corners=corners,
                collecting=collecting,
                started=started,
                run_index=max(1, current_run),
                total_runs=runs,
                detections=len(detections),
                seconds_left=seconds_left,
            )
            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32 and not started:
                started = True
                collecting = True
                current_run = 1
                detections = []
                run_started_s = time.monotonic()
                last_capture_s = -float("inf")
                print(
                    f"[checkerboard_calib] started: {runs} runs x "
                    f"{seconds_per_run:.1f}s"
                )
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass
        cv2.waitKey(1)

    return results


def _estimate_pose_for_run(
    run_index: int,
    detections: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    *,
    min_frames: int,
) -> RunResult:
    if len(detections) < min_frames:
        return RunResult(
            run_index=run_index,
            detections=len(detections),
            pose=None,
            error=f"too few detections ({len(detections)} < {min_frames})",
        )
    try:
        pose = calib_checkerboard.estimate_pose_from_detections(detections, K, dist)
    except Exception as exc:
        return RunResult(
            run_index=run_index,
            detections=len(detections),
            pose=None,
            error=str(exc),
        )
    return RunResult(run_index=run_index, detections=len(detections), pose=pose)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat live checkerboard pose calibration several times from one "
            "SPACE press and compare run-to-run stability."
        )
    )
    parser.add_argument("--calibration", type=Path, default=None, help="Camera calibration .npz")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS_PER_RUN)
    parser.add_argument(
        "--capture-interval-s",
        type=float,
        default=calib_checkerboard.DEFAULT_CAPTURE_INTERVAL_S,
    )
    parser.add_argument("--min-frames", type=int, default=calib_checkerboard.DEFAULT_MIN_FRAMES)
    parser.add_argument("--width", type=int, default=calib_checkerboard.REALSENSE_WIDTH)
    parser.add_argument("--height", type=int, default=calib_checkerboard.REALSENSE_HEIGHT)
    parser.add_argument("--fps", type=int, default=calib_checkerboard.REALSENSE_FPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-save", action="store_true", help="Do not write the JSONL summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    K, dist, calibration_path = load_camera_calibration(args.calibration)
    print(f"[checkerboard_calib] calibration: {calibration_path}")
    print(
        f"[checkerboard_calib] camera stream: "
        f"{args.width}x{args.height}@{args.fps}"
    )

    pipe = None
    try:
        pipe, _profile = calib_checkerboard.start_realsense_color_stream(
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
        )
        results = collect_repeated_calibrations(
            pipe,
            K,
            dist,
            runs=int(args.runs),
            seconds_per_run=float(args.seconds),
            capture_interval_s=float(args.capture_interval_s),
            min_frames=int(args.min_frames),
        )
    finally:
        if pipe is not None:
            pipe.stop()

    print_summary(results)
    if results and not args.no_save:
        out_path = write_jsonl(
            results,
            output_dir=Path(args.output_dir),
            calibration_path=calibration_path,
            runs=int(args.runs),
            seconds_per_run=float(args.seconds),
            capture_interval_s=float(args.capture_interval_s),
        )
        print(f"[checkerboard_calib] wrote {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
