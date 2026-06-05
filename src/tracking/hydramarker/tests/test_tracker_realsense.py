from __future__ import annotations

import csv
import math
import time
from collections import Counter
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.hydramarker.tracker import HydraTracker, TrackerConfig


# ============================================================
# Frame Logger / Diagnostics
# ============================================================

LOG_PATH = Path("hydramarker_frame_log.csv")
EVENT_LOG_PATH = Path("hydramarker_event_log.txt")
SUMMARY_PATH = Path("hydramarker_session_summary.txt")

FEW_GREEN_THRESHOLD = 5

COLUMNS = [
    # frame / timing
    "frame", "wall_ms",

    # tracker result
    "mode", "success", "message",
    "failure_stage", "failure_reason",

    # detection / pose counts
    "det_valid", "det_tracking", "det_stable", "det_corners",
    "pose_corners", "num_points", "num_inliers",
    "mean_err", "max_err", "confidence",
    "persistent_count",

    # useful ratios
    "green_ratio", "inlier_ratio",

    # per-frame motion diagnostics from detector corners
    "median_corner_motion_px",
    "mean_corner_motion_px",
    "p95_corner_motion_px",
    "max_corner_motion_px",
    "matched_corner_motion_count",
    "det_count_change",
    "det_count_change_abs",

    # pose jump diagnostics from accepted poses
    "pose_rotation_delta_deg",
    "pose_translation_delta_mm",

    # session counters directly in CSV
    "pose_available_pct",
    "current_outage", "longest_outage",
    "pose_frames", "no_pose_frames",
    "blue_only_total", "zero_green_total", "few_green_total",

    # cause counters directly in CSV
    "decode_fail_total",
    "correspondence_fail_total",
    "rotation_gate_fail_total",
    "translation_gate_fail_total",
    "motion_gate_fail_total",
    "pnp_fail_total",
    "reprojection_fail_total",
    "other_fail_total",
]

_log_file = None
_log_writer = None
_event_file = None

# Session statistics
_total_frames = 0
_pose_frames = 0
_no_pose_frames = 0

_current_outage = 0
_longest_outage = 0
_outage_lengths: list[int] = []

_blue_only_frames = 0
_zero_green_frames = 0
_few_green_frames = 0

_failure_counter: Counter[str] = Counter()

# Previous-frame state for motion diagnostics
_prev_detection_uv: Optional[np.ndarray] = None
_prev_det_count: Optional[int] = None
_prev_pose_rvec: Optional[np.ndarray] = None
_prev_pose_tvec: Optional[np.ndarray] = None


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{float(x):.{digits}f}"


def _extract_detection_uv(result) -> np.ndarray:
    pts = getattr(result, "detection_corners", [])
    uv = []
    for p in pts:
        try:
            uv.append([float(p.uv[0]), float(p.uv[1])])
        except Exception:
            pass
    if not uv:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(uv, dtype=np.float64).reshape(-1, 2)


def _nearest_neighbor_motion(prev_uv: Optional[np.ndarray], curr_uv: np.ndarray) -> dict[str, Optional[float] | int]:
    """
    Estimate image motion from blue detector corners.

    This does not require stable corner IDs. For every current detected corner,
    we find the nearest previous detected corner and use the nearest-neighbor
    distance as an approximate per-frame motion. This is intentionally simple
    and robust enough for debugging motion-related tracking failures.
    """
    if prev_uv is None or len(prev_uv) == 0 or len(curr_uv) == 0:
        return {
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
            "count": 0,
        }

    # Pairwise distances: curr x prev. Number of corners is small, so this is fine.
    diff = curr_uv[:, None, :] - prev_uv[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    nearest = np.min(dists, axis=1)

    # Reject implausibly large NN matches caused by detection topology changes.
    # Keep the threshold generous, because we want to see fast motion too.
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        return {
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
            "count": 0,
        }

    return {
        "median": float(np.median(nearest)),
        "mean": float(np.mean(nearest)),
        "p95": float(np.percentile(nearest, 95)),
        "max": float(np.max(nearest)),
        "count": int(len(nearest)),
    }


def _rotation_delta_deg(rvec_prev: Optional[np.ndarray], rvec_curr: Optional[np.ndarray]) -> Optional[float]:
    if rvec_prev is None or rvec_curr is None:
        return None
    try:
        R_prev, _ = cv2.Rodrigues(np.asarray(rvec_prev, dtype=np.float64).reshape(3, 1))
        R_curr, _ = cv2.Rodrigues(np.asarray(rvec_curr, dtype=np.float64).reshape(3, 1))
        R_delta = R_curr @ R_prev.T
        cos_angle = (np.trace(R_delta) - 1.0) / 2.0
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))
    except Exception:
        return None


def _translation_delta_mm(tvec_prev: Optional[np.ndarray], tvec_curr: Optional[np.ndarray]) -> Optional[float]:
    if tvec_prev is None or tvec_curr is None:
        return None
    try:
        a = np.asarray(tvec_prev, dtype=np.float64).reshape(3)
        b = np.asarray(tvec_curr, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(b - a))
    except Exception:
        return None


def classify_failure(result) -> tuple[str, str]:
    """
    Classify why a frame has no accepted pose.

    This is intentionally based on result.message so the test script works
    without requiring changes inside HydraTracker. Later, if HydraTracker exposes
    structured debug fields, this can be replaced by those fields.
    """
    if result.success:
        return "OK", "OK"

    msg = str(getattr(result, "message", "") or "")
    msg_l = msg.lower()

    if "no valid decoded patches" in msg_l:
        return "PATCH_DECODER", "NO_VALID_DECODED_PATCHES"

    if "correspondence build failed" in msg_l:
        return "CORRESPONDENCE", "CORRESPONDENCE_REJECTED"

    if "too few correspondences" in msg_l:
        return "CORRESPONDENCE", "CORRESPONDENCE_REJECTED"

    if "rotation jump too large" in msg_l:
        return "MOTION_GATE", "ROTATION_JUMP_TOO_LARGE"

    if "translation jump too large" in msg_l:
        return "MOTION_GATE", "TRANSLATION_JUMP_TOO_LARGE"

    if "motion gate rejected" in msg_l:
        return "MOTION_GATE", "MOTION_GATE_REJECTED"

    if "pnp" in msg_l and ("failed" in msg_l or "fail" in msg_l):
        return "PNP", "PNP_FAILED"

    if "too few inliers" in msg_l:
        return "PNP", "TOO_FEW_INLIERS"

    if "reprojection" in msg_l and ("too large" in msg_l or "rejected" in msg_l):
        return "REPROJECTION_GATE", "REPROJECTION_ERROR_TOO_LARGE"

    return "OTHER", "OTHER"


def log_open() -> None:
    global _log_file, _log_writer, _event_file

    _log_file = open(LOG_PATH, "w", newline="", encoding="utf-8")
    _log_writer = csv.DictWriter(_log_file, fieldnames=COLUMNS)
    _log_writer.writeheader()
    _log_file.flush()

    _event_file = open(EVENT_LOG_PATH, "w", encoding="utf-8")
    _event_file.write("HydraTracker event log\n")
    _event_file.write("======================\n\n")
    _event_file.flush()

    print(f"[frame_log] {LOG_PATH.resolve()}")
    print(f"[event_log] {EVENT_LOG_PATH.resolve()}")


def _write_event(line: str) -> None:
    if _event_file is None:
        return
    _event_file.write(line.rstrip() + "\n")
    _event_file.flush()


def log_frame(frame_idx: int, result, wall_ms: float, tracker: HydraTracker) -> None:
    global _total_frames, _pose_frames, _no_pose_frames
    global _current_outage, _longest_outage, _outage_lengths
    global _blue_only_frames, _zero_green_frames, _few_green_frames
    global _prev_detection_uv, _prev_det_count, _prev_pose_rvec, _prev_pose_tvec

    if _log_writer is None:
        return

    detection_uv = _extract_detection_uv(result)
    det_corners = int(len(detection_uv))
    pose_corners = int(_safe_len(getattr(result, "corners", [])))
    num_points = int(getattr(result, "num_points", 0) or 0)
    num_inliers = int(getattr(result, "num_inliers", 0) or 0)

    failure_stage, failure_reason = classify_failure(result)

    _total_frames += 1

    has_pose = bool(getattr(result, "success", False))
    was_in_outage = _current_outage > 0

    if has_pose:
        _pose_frames += 1
        if _current_outage > 0:
            _outage_lengths.append(_current_outage)
            _longest_outage = max(_longest_outage, _current_outage)
            pose_available_pct_now = 100.0 * _pose_frames / max(_total_frames, 1)
            _write_event(
                f"OUTAGE_END frame={frame_idx} "
                f"duration={_current_outage} "
                f"pose_available_pct={pose_available_pct_now:.2f}"
            )
            _current_outage = 0
    else:
        _no_pose_frames += 1
        _current_outage += 1
        _longest_outage = max(_longest_outage, _current_outage)
        _failure_counter[failure_reason] += 1

        event_kind = "OUTAGE_START" if not was_in_outage else "OUTAGE_CONT"
        extra = f" len={_current_outage}" if was_in_outage else ""
        _write_event(
            f"{event_kind} frame={frame_idx}{extra} "
            f"stage={failure_stage} reason={failure_reason} "
            f"det={det_corners} pose={pose_corners} "
            f"pts={num_points} inl={num_inliers} "
            f"msg={getattr(result, 'message', '')}"
        )

    if det_corners > 0 and pose_corners == 0:
        _blue_only_frames += 1

    if pose_corners == 0:
        _zero_green_frames += 1

    if pose_corners < FEW_GREEN_THRESHOLD:
        _few_green_frames += 1

    green_ratio = pose_corners / max(det_corners, 1)
    inlier_ratio = num_inliers / max(num_points, 1)
    pose_available_pct = 100.0 * _pose_frames / max(_total_frames, 1)

    det_count_change = 0 if _prev_det_count is None else det_corners - _prev_det_count
    det_count_change_abs = abs(det_count_change)

    motion = _nearest_neighbor_motion(_prev_detection_uv, detection_uv)

    # Pose delta is only meaningful between accepted poses.
    if has_pose and getattr(result, "rvec", None) is not None and getattr(result, "tvec", None) is not None:
        rot_delta = _rotation_delta_deg(_prev_pose_rvec, result.rvec)
        trans_delta = _translation_delta_mm(_prev_pose_tvec, result.tvec)
        _prev_pose_rvec = np.asarray(result.rvec, dtype=np.float64).reshape(3, 1).copy()
        _prev_pose_tvec = np.asarray(result.tvec, dtype=np.float64).reshape(3, 1).copy()
    else:
        rot_delta = None
        trans_delta = None

    row = {
        "frame": frame_idx,
        "wall_ms": f"{wall_ms:.1f}",

        "mode": result.mode.value,
        "success": int(has_pose),
        "message": getattr(result, "message", ""),
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,

        "det_valid": int(getattr(result, "detection_valid", False)),
        "det_tracking": int(getattr(result, "detection_tracking", False)),
        "det_stable": int(getattr(result, "detection_stable", False)),
        "det_corners": det_corners,
        "pose_corners": pose_corners,
        "num_points": num_points,
        "num_inliers": num_inliers,
        "mean_err": f"{float(getattr(result, 'mean_reprojection_error_px', 0.0)):.3f}",
        "max_err": f"{float(getattr(result, 'max_reprojection_error_px', 0.0)):.3f}",
        "confidence": f"{float(getattr(result, 'confidence', 0.0)):.3f}",
        "persistent_count": _safe_len(getattr(tracker, "_persistent_corners", [])),

        "green_ratio": f"{green_ratio:.3f}",
        "inlier_ratio": f"{inlier_ratio:.3f}",

        "median_corner_motion_px": _fmt_float(motion["median"]),
        "mean_corner_motion_px": _fmt_float(motion["mean"]),
        "p95_corner_motion_px": _fmt_float(motion["p95"]),
        "max_corner_motion_px": _fmt_float(motion["max"]),
        "matched_corner_motion_count": motion["count"],
        "det_count_change": det_count_change,
        "det_count_change_abs": det_count_change_abs,

        "pose_rotation_delta_deg": _fmt_float(rot_delta),
        "pose_translation_delta_mm": _fmt_float(trans_delta),

        "pose_available_pct": f"{pose_available_pct:.2f}",
        "current_outage": _current_outage,
        "longest_outage": _longest_outage,
        "pose_frames": _pose_frames,
        "no_pose_frames": _no_pose_frames,
        "blue_only_total": _blue_only_frames,
        "zero_green_total": _zero_green_frames,
        "few_green_total": _few_green_frames,

        "decode_fail_total": _failure_counter["NO_VALID_DECODED_PATCHES"],
        "correspondence_fail_total": _failure_counter["CORRESPONDENCE_REJECTED"],
        "rotation_gate_fail_total": _failure_counter["ROTATION_JUMP_TOO_LARGE"],
        "translation_gate_fail_total": _failure_counter["TRANSLATION_JUMP_TOO_LARGE"],
        "motion_gate_fail_total": (
            _failure_counter["MOTION_GATE_REJECTED"]
            + _failure_counter["ROTATION_JUMP_TOO_LARGE"]
            + _failure_counter["TRANSLATION_JUMP_TOO_LARGE"]
        ),
        "pnp_fail_total": _failure_counter["PNP_FAILED"] + _failure_counter["TOO_FEW_INLIERS"],
        "reprojection_fail_total": _failure_counter["REPROJECTION_ERROR_TOO_LARGE"],
        "other_fail_total": _failure_counter["OTHER"],
    }

    _log_writer.writerow(row)
    _log_file.flush()

    _prev_detection_uv = detection_uv.copy()
    _prev_det_count = det_corners


def log_close() -> None:
    global _current_outage

    if _current_outage > 0:
        _outage_lengths.append(_current_outage)
        _longest_outage = max(_longest_outage, _current_outage)
        _current_outage = 0

    pose_availability = 100.0 * _pose_frames / max(_total_frames, 1)
    blue_only_ratio = 100.0 * _blue_only_frames / max(_total_frames, 1)
    zero_green_ratio = 100.0 * _zero_green_frames / max(_total_frames, 1)
    few_green_ratio = 100.0 * _few_green_frames / max(_total_frames, 1)
    mean_outage = sum(_outage_lengths) / len(_outage_lengths) if _outage_lengths else 0.0

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("HydraTracker Session Summary\n")
        f.write("========================================\n\n")

        f.write(f"Total frames:                 {_total_frames}\n")
        f.write(f"Frames with pose:             {_pose_frames}\n")
        f.write(f"Frames without pose:          {_no_pose_frames}\n")
        f.write(f"Pose availability [%]:        {pose_availability:.2f}\n\n")

        f.write(f"Blue-only frames:             {_blue_only_frames}\n")
        f.write(f"Blue-only ratio [%]:          {blue_only_ratio:.2f}\n")
        f.write(f"Zero-green frames:            {_zero_green_frames}\n")
        f.write(f"Zero-green ratio [%]:         {zero_green_ratio:.2f}\n")
        f.write(f"Few-green frames (<{FEW_GREEN_THRESHOLD}):       {_few_green_frames}\n")
        f.write(f"Few-green ratio [%]:          {few_green_ratio:.2f}\n\n")

        f.write(f"Number of pose outages:       {len(_outage_lengths)}\n")
        f.write(f"Longest pose outage [frames]: {_longest_outage}\n")
        f.write(f"Mean pose outage [frames]:    {mean_outage:.2f}\n\n")

        f.write("Outage lengths [frames]:\n")
        if _outage_lengths:
            f.write(", ".join(str(x) for x in _outage_lengths) + "\n\n")
        else:
            f.write("none\n\n")

        f.write("Pose losses by cause:\n")
        if _no_pose_frames > 0:
            for reason, count in _failure_counter.most_common():
                pct = 100.0 * count / max(_no_pose_frames, 1)
                f.write(f"  {reason}: {count} ({pct:.1f} % of no-pose frames)\n")
        else:
            f.write("  none\n")

        f.write("\nMotion diagnostics columns in CSV:\n")
        f.write("  median_corner_motion_px, mean_corner_motion_px, p95_corner_motion_px, max_corner_motion_px\n")
        f.write("  det_count_change, det_count_change_abs\n")
        f.write("  pose_rotation_delta_deg, pose_translation_delta_mm\n")

    if _log_file and not _log_file.closed:
        _log_file.close()
        print(f"[frame_log] closed -> {LOG_PATH.resolve()}")

    if _event_file and not _event_file.closed:
        _event_file.close()
        print(f"[event_log] closed -> {EVENT_LOG_PATH.resolve()}")

    print(f"[summary] {SUMMARY_PATH.resolve()}")


# ============================================================
# Helpers
# ============================================================

def choose_file_qt(title: str, file_filter: str) -> Path:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def put_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def realsense_intrinsics_to_cv(profile) -> tuple[np.ndarray, np.ndarray]:
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array(
        [[intr.fx, 0.0, intr.ppx],
         [0.0, intr.fy, intr.ppy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.asarray(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
    return K, dist


def create_realsense_pipeline():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    return pipe, profile


def draw_detection_corners(vis: np.ndarray, result) -> None:
    for p in getattr(result, "detection_corners", []):
        u = int(round(p.uv[0]))
        v = int(round(p.uv[1]))
        cv2.circle(vis, (u, v), 3, (255, 180, 0), -1, cv2.LINE_AA)


def draw_pose_corners(vis: np.ndarray, result) -> None:
    if not result.success:
        return
    for p in result.corners:
        u = int(round(p.uv[0]))
        v = int(round(p.uv[1]))
        cv2.circle(vis, (u, v), 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(vis, f"{p.global_row},{p.global_col}",
                    (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 255, 0), 1, cv2.LINE_AA)


def draw_reprojection(vis: np.ndarray, result, K, dist) -> None:
    if not result.success or result.rvec is None or result.tvec is None:
        return
    if len(result.corners) == 0:
        return

    object_points = np.asarray([p.xyz_mm for p in result.corners], dtype=np.float64).reshape(-1, 3)
    measured = np.asarray([p.uv for p in result.corners], dtype=np.float64).reshape(-1, 2)

    projected, _ = cv2.projectPoints(
        object_points,
        result.rvec.reshape(3, 1),
        result.tvec.reshape(3, 1),
        K, dist,
    )
    projected = projected.reshape(-1, 2)

    for m, q in zip(measured, projected):
        cv2.circle(vis, (int(round(q[0])), int(round(q[1]))), 3, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.line(vis, (int(round(m[0])), int(round(m[1]))),
                      (int(round(q[0])), int(round(q[1]))), (255, 0, 255), 1, cv2.LINE_AA)


def draw_status(vis: np.ndarray, result, frame_idx: int, tracker: HydraTracker) -> None:
    detection_corners = getattr(result, "detection_corners", [])
    status_color = (0, 255, 0) if result.success else (0, 165, 255)
    failure_stage, failure_reason = classify_failure(result)

    line1 = (
        f"frame={frame_idx} | {result.mode.value} | ok={result.success} | "
        f"det={len(detection_corners)} | pose={'Y' if result.rvec is not None and result.tvec is not None else 'N'} | "
        f"vis={len(result.corners)} | "
        f"pts={result.num_points} | inl={result.num_inliers} | "
        f"pers={len(tracker._persistent_corners)}"
    )
    line2 = (
        f"mean={result.mean_reprojection_error_px:.3f}px | "
        f"max={result.max_reprojection_error_px:.3f}px | "
        f"conf={result.confidence:.2f} | "
        f"stage={failure_stage}"
    )

    put_text(vis, line1, (25, 35), color=status_color, scale=0.55)
    put_text(vis, line2, (25, 65), color=status_color, scale=0.50)
    put_text(vis, f"reason: {failure_reason}", (25, 95), color=(0, 255, 255), scale=0.46)
    put_text(vis,
             "blue=detector | green=global corr | magenta=reprojection | r=reset | q=quit",
             (25, 125), color=(255, 180, 0), scale=0.46)


def draw_debug(vis, result, K, dist, frame_idx, tracker) -> np.ndarray:
    draw_detection_corners(vis, result)
    draw_pose_corners(vis, result)
    draw_reprojection(vis, result, K, dist)
    draw_status(vis, result, frame_idx, tracker)
    return vis


def log_console(frame_idx: int, result, tracker, *, force: bool = False) -> None:
    if not force and result.success and frame_idx % 30 != 0:
        return

    failure_stage, failure_reason = classify_failure(result)
    print(
        "[test_tracker]",
        f"frame={frame_idx}",
        f"mode={result.mode.value}",
        f"success={result.success}",
        f"stage={failure_stage}",
        f"reason={failure_reason}",
        f"msg={result.message}",
        f"det={len(getattr(result, 'detection_corners', []))}",
        f"pose={'Y' if result.rvec is not None and result.tvec is not None else 'N'}",
        f"vis={len(result.corners)}",
        f"pts={result.num_points}",
        f"inl={result.num_inliers}",
        f"mean={result.mean_reprojection_error_px:.3f}",
        f"pers={len(tracker._persistent_corners)}",
    )


def make_tracker(field_path, marker_json_path, K, dist) -> HydraTracker:
    return HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_json_path),
        K=K,
        dist_coeffs=dist,
        config=TrackerConfig(
            min_points=6,
            min_inliers=5,
            max_mean_reprojection_error_px=4.0,
            max_max_reprojection_error_px=15.0,
            max_lost_frames=8,
            max_translation_jump_mm=120.0,
            # Drill kann sich beliebig schnell drehen — Rotations-Gate deaktiviert.
            # Nur Translation wird geprüft (max_translation_jump_mm).
            max_rotation_jump_deg=360.0,
            rotation_gate_scale_per_lost_frame=0.0,
            rotation_gate_max_deg=360.0,
            # On the drill/cylinder, low pts is often caused by visibility,
            # not LK drift. Do not reset dot state based on point count.
            dot_early_reset_pts_ratio=0.0,
            dot_early_reset_min_pts=6,
            pnp_ransac_iterations=500,
            pnp_ransac_reprojection_px=3.0,
            pnp_ransac_confidence=0.99,
            use_pose_prior=True,
            corr_min_votes=2,
            corr_discard_conflicts=True,
            corr_require_detection_stable=False,
            corr_enable_dominant_rotation_filter=True,
            corr_min_rotation_support=2,
            corr_min_rotation_support_ratio=0.55,
            # Use only real checkerboard detections for dot decoding.
            # Pose-projected cells are unsafe on a cylinder because projected
            # corners may be on the occluded side.
            enable_pose_propagation=False,

            # Safe decode-outage bridge: cached global IDs are matched by
            # last-pose reprojection, not stale UV proximity. This can bridge
            # short decoder dropouts without accepting newly ambiguous IDs.
            enable_temporal_correspondence_persistence=True,
            persistence_use_pose_projection=True,
            persistence_projection_max_reproj_px=12.0,
            persistence_projection_max_pose_error_px=2.5,

            # Current-frame dot decisions: no EMA warmup-lock.
            dot_use_temporal_smoothing=False,

            enable_debug_prints=True,
            log_path="hydramarker_tracker.log",
            log_to_console=False,
            dot_commit_frames=1,
            dot_revoke_frames=5,
            persistence_max_frames=8,
        ),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    field_path = choose_file_qt("Select HydraMarker .field file", "HydraMarker field (*.field)")
    marker_json_path = choose_file_qt("Select marker .json file", "Marker JSON (*.json)")

    pipe, profile = create_realsense_pipeline()
    K_rgb, dist_rgb = realsense_intrinsics_to_cv(profile)

    tracker = make_tracker(field_path, marker_json_path, K_rgb, dist_rgb)

    log_open()

    window_name = "HydraTracker RealSense Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    last_mode: Optional[str] = None
    last_success: Optional[bool] = None
    last_message: Optional[str] = None

    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_idx += 1
            frame = np.asanyarray(color_frame.get_data())

            t0 = time.perf_counter()
            result = tracker.process_frame(frame)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            # CSV log — every frame
            log_frame(frame_idx, result, wall_ms, tracker)

            # Console log — only on state changes or failures
            mode_changed = last_mode != result.mode.value
            success_changed = last_success != bool(result.success)
            message_changed = last_message != result.message
            force_log = mode_changed or success_changed or message_changed or not result.success
            log_console(frame_idx, result, tracker, force=force_log)

            last_mode = result.mode.value
            last_success = bool(result.success)
            last_message = result.message

            vis = draw_debug(frame.copy(), result, K_rgb, dist_rgb, frame_idx, tracker)
            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                tracker.reset()
                last_mode = last_success = last_message = None
                print("[test_tracker] reset")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        log_close()


if __name__ == "__main__":
    main()
