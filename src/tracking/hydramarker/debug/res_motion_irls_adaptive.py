from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from res_motion_irls_sweep import (  # noqa: E402
    COMPONENTS,
    DEFAULT_RUNS,
    _best_lag_metrics,
    _finite,
    _parse_list,
    _range,
    _rms,
    _score,
    _weighted_rms_residual,
)
from res_static_irls_replay import (  # noqa: E402
    IrlsConfig,
    _T_to_pose,
    _median,
    _percentile,
    _pose_to_T,
    _to_float,
    arrays_for_observation,
    arrays_for_window,
    build_age_maps,
    build_point_priors,
    build_static_uv_model,
    load_run,
    solve_irls_lie_pose,
)


# Choose which preset is used when you run the script without overrides.
# "fb_motion" is the constrained paper-style Lie-IRLS pass found for the
# moving fb run. "motion" keeps the older unconstrained 6-DOF pass, and
# "static_live" is the guarded static stabilizer.
DEFAULT_PRESET = "fb_motion"

ACTIVE_DOF_PRESETS: dict[str, tuple[int, ...] | None] = {
    "free6": None,
    "free": None,
    "all": None,
    "ty": (1,),
    "ty_rz": (1, 5),
    "ty_rot": (1, 3, 4, 5),
    "xy": (0, 1),
    "xy_rz": (0, 1, 5),
    "xy_rot": (0, 1, 3, 4, 5),
    "xyz_no_rot": (0, 1, 2),
}

DOF_LABELS = ("tx", "ty", "tz", "rx", "ry", "rz")

MOTION_IRLS_SETTINGS: dict[str, Any] = {
    "point_set": "correspondence",
    "active_dofs": "free6",
    "static_window_frames": "30",
    "static_window_decay": "0",
    "translation_gate_mm": "-1",
    "rotation_gate_deg": "0.5",
    "settle_frames": "1",
    "exit_frames": "3",
    "robust_c_px": "0.2",
    "uv_stability_scale_px": "0.05",
    "age_ramp_frames": "1",
    "max_step_translation_mm": "5",
    "max_step_rotation_deg": "3",
    "max_iterations": 6,
    "frame_stride": 1,
    "max_lag_frames": 12,
    "reproj_guard_px": "0.03",
    "raw_motion_baseline": 0,
    "plot": 1,
    "write_frame_csv": 1,
    "tag": "best_motion",
}

FB_CONSTRAINED_MOTION_SETTINGS: dict[str, Any] = {
    "point_set": "correspondence",
    "active_dofs": "xy_rot",
    "static_window_frames": "30",
    "static_window_decay": "0",
    "translation_gate_mm": "-1",
    "rotation_gate_deg": "0.5",
    "settle_frames": "1",
    "exit_frames": "3",
    "robust_c_px": "0.2",
    "uv_stability_scale_px": "0.05",
    "age_ramp_frames": "1",
    "max_step_translation_mm": "5",
    "max_step_rotation_deg": "5",
    "max_iterations": 6,
    "frame_stride": 1,
    "max_lag_frames": 12,
    "reproj_guard_px": "0.03",
    "raw_motion_baseline": 0,
    "plot": 1,
    "write_frame_csv": 1,
    "tag": "fb_best_xy_rot",
}

STATIC_LIVE_SETTINGS: dict[str, Any] = {
    "point_set": "correspondence",
    "active_dofs": "free6",
    "static_window_frames": "30",
    "static_window_decay": "0",
    "translation_gate_mm": "0.45",
    "rotation_gate_deg": "0.5",
    "settle_frames": "1",
    "exit_frames": "3",
    "robust_c_px": "0.2",
    "uv_stability_scale_px": "0.05",
    "age_ramp_frames": "1",
    "max_step_translation_mm": "40",
    "max_step_rotation_deg": "8",
    "max_iterations": 6,
    "frame_stride": 1,
    "max_lag_frames": 12,
    "reproj_guard_px": "0.03",
    "raw_motion_baseline": 1,
    "plot": 1,
    "write_frame_csv": 1,
    "tag": "best_live",
}

PRESETS: dict[str, dict[str, Any]] = {
    "fb": FB_CONSTRAINED_MOTION_SETTINGS,
    "fb_motion": FB_CONSTRAINED_MOTION_SETTINGS,
    "best_fb": FB_CONSTRAINED_MOTION_SETTINGS,
    "best_motion_constrained": FB_CONSTRAINED_MOTION_SETTINGS,
    "motion": MOTION_IRLS_SETTINGS,
    "best_motion": MOTION_IRLS_SETTINGS,
    "static": STATIC_LIVE_SETTINGS,
    "static_live": STATIC_LIVE_SETTINGS,
    "best_live": STATIC_LIVE_SETTINGS,
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_active_dofs(value: Any) -> tuple[int, ...] | None:
    text = str(value).strip().lower()
    if text in ACTIVE_DOF_PRESETS:
        return ACTIVE_DOF_PRESETS[text]
    if text in ("", "none", "null"):
        return None
    name_to_idx = {name: idx for idx, name in enumerate(DOF_LABELS)}
    active: list[int] = []
    for part in text.replace(";", ",").replace(" ", ",").split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in name_to_idx:
            active.append(name_to_idx[item])
            continue
        try:
            idx = int(item)
        except ValueError as exc:
            raise RuntimeError(
                f"Unknown active DOF {part!r}; use free6, xy_rot, xy_rz, or tx,ty,tz,rx,ry,rz"
            ) from exc
        if idx < 0 or idx >= len(DOF_LABELS):
            raise RuntimeError(f"Active DOF index out of range: {idx}")
        active.append(idx)
    deduped = tuple(dict.fromkeys(active))
    return deduped or None


def _active_dofs_label(active_dofs: tuple[int, ...] | None) -> str:
    if active_dofs is None:
        return "free6"
    return ",".join(DOF_LABELS[idx] for idx in active_dofs)


def _plot_adaptive_translation(run: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, Any]], tag: str) -> Path:
    from debug_tracker_translation import setup_plot_style
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    setup_plot_style(plt)
    path = Path(run["path"]).resolve()
    frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
    adaptive = np.asarray(
        [[_to_float(row.get("tvec_x_mm")), _to_float(row.get("tvec_y_mm")), _to_float(row.get("tvec_z_mm"))] for row in rows],
        dtype=np.float64,
    )
    logged = np.asarray(
        [
            [_to_float(row.get("logged_x_mm")), _to_float(row.get("logged_y_mm")), _to_float(row.get("logged_z_mm"))]
            for row in rows
        ],
        dtype=np.float64,
    )
    adaptive_rel = adaptive - adaptive[0].reshape(1, 3)
    logged_rel = logged - logged[0].reshape(1, 3)
    delta = adaptive - logged
    current_wrms = np.asarray([_to_float(row.get("current_reproj_weighted_rms_px")) for row in rows], dtype=np.float64)
    logged_wrms = np.asarray([_to_float(row.get("logged_current_reproj_weighted_rms_px")) for row in rows], dtype=np.float64)
    excess = np.maximum(0.0, current_wrms - logged_wrms)
    use_static = np.asarray([_to_float(row.get("use_static")) for row in rows], dtype=np.float64)
    guard_reject = np.asarray([_to_float(row.get("guard_reject")) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(13.5, 13.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.72, 0.85, 0.35]},
    )
    fig.subplots_adjust(top=0.88, hspace=0.28)
    raw_ranges = np.nanmax(logged_rel, axis=0) - np.nanmin(logged_rel, axis=0)
    adaptive_ranges = np.nanmax(adaptive_rel, axis=0) - np.nanmin(adaptive_rel, axis=0)
    detail_parts = []
    if summary.get("active_dofs") is not None:
        detail_parts.append(f"dof={summary.get('active_dofs')}")
    if summary.get("robust_c_px") is not None:
        detail_parts.append(f"c={_to_float(summary.get('robust_c_px')):.3g}px")
    if summary.get("max_step_translation_mm") is not None:
        detail_parts.append(f"stepT={_to_float(summary.get('max_step_translation_mm')):.3g}mm")
    if summary.get("max_step_rotation_deg") is not None:
        detail_parts.append(f"stepR={_to_float(summary.get('max_step_rotation_deg')):.3g}deg")
    if _to_float(summary.get("translation_gate_mm")) >= 0.0:
        detail_parts.append(
            f"static w={summary.get('static_window_frames')} tg={summary.get('translation_gate_mm')}mm "
            f"guard={summary.get('reproj_guard_px')}px"
        )
    detail = " ".join(str(part) for part in detail_parts if str(part).strip())
    fig.suptitle(
        "Adaptive IRLS translation replay\n"
        f"{path.name} | {detail}\n"
        f"raw range xyz=({raw_ranges[0]:.3f}, {raw_ranges[1]:.3f}, {raw_ranges[2]:.3f}) mm | "
        f"adaptive range xyz=({adaptive_ranges[0]:.3f}, {adaptive_ranges[1]:.3f}, {adaptive_ranges[2]:.3f}) mm",
        fontsize=13,
    )

    colors = {"x": "#4c78a8", "y": "#54a24b", "z": "#e45756"}
    for axis_idx, label in enumerate(("x", "y", "z")):
        ax = axes[axis_idx]
        ax.plot(frames, logged_rel[:, axis_idx], color="#8a8f98", linewidth=1.15, linestyle="--", label="logged/raw PnP")
        ax.plot(frames, adaptive_rel[:, axis_idx], color=colors[label], linewidth=1.55, label="adaptive IRLS output")
        ax.set_ylabel(f"{label} rel. [mm]")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=8)

    delta_ax = axes[3]
    for axis_idx, label in enumerate(("x", "y", "z")):
        delta_ax.plot(frames, delta[:, axis_idx], color=colors[label], linewidth=1.15, label=f"{label} adaptive - raw")
    delta_ax.axhline(0.0, color="#8a8f98", linewidth=0.8, linestyle="--")
    delta_ax.set_ylabel("delta [mm]")
    delta_ax.grid(True, alpha=0.28)
    delta_ax.legend(loc="upper right", fontsize=8, ncol=3)

    reproj_ax = axes[4]
    reproj_ax.plot(frames, logged_wrms, color="#8a8f98", linewidth=1.1, linestyle="--", label="raw current WRMS")
    reproj_ax.plot(frames, current_wrms, color="#f58518", linewidth=1.25, label="adaptive current WRMS")
    reproj_ax.plot(frames, excess, color="#b279a2", linewidth=1.15, label="excess")
    reproj_ax.set_ylabel("reproj. [px]")
    reproj_ax.grid(True, alpha=0.28)
    reproj_ax.legend(loc="upper right", fontsize=8)

    status_ax = axes[5]
    status = np.zeros_like(frames, dtype=np.float64)
    status = np.where(use_static >= 0.5, 1.0, status)
    status = np.where(guard_reject >= 0.5, 2.0, status)
    status_ax.imshow(
        status.reshape(1, -1),
        aspect="auto",
        extent=[float(frames[0]), float(frames[-1]), 0, 1],
        interpolation="nearest",
        cmap=ListedColormap(["#d8dde6", "#54a24b", "#b279a2"]),
        vmin=0,
        vmax=2,
    )
    status_ax.set_yticks([])
    status_ax.set_ylabel("state")
    status_ax.set_xlabel("frame")
    status_ax.set_title("gray = raw/motion   green = static IRLS active   purple = static candidate rejected by reprojection guard", fontsize=9)

    safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(tag).strip()) or "adaptive"
    out_path = path.with_name(f"{path.stem}_{safe_tag}_adaptive_irls_translation_plot.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[adaptive_irls] saved plot -> {out_path.resolve()}")
    return out_path


def _eval_indices(num_frames: int, stride: int) -> list[int]:
    stride = max(int(stride), 1)
    indices = list(range(0, int(num_frames), stride))
    if not indices or indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    return sorted(set(indices))


def _rvec_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    try:
        import cv2

        Ra, _ = cv2.Rodrigues(np.asarray(a, dtype=np.float64).reshape(3, 1))
        Rb, _ = cv2.Rodrigues(np.asarray(b, dtype=np.float64).reshape(3, 1))
        R = np.asarray(Ra, dtype=np.float64).reshape(3, 3) @ np.asarray(Rb, dtype=np.float64).reshape(3, 3).T
        trace = float(np.trace(R))
        angle = math.acos(float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        return math.degrees(angle)
    except Exception:
        return math.nan


def _low_motion_flags(
    observations,
    *,
    translation_gate_mm: float,
    rotation_gate_deg: float,
    settle_frames: int,
    exit_frames: int = 1,
) -> list[bool]:
    low_streak = 0
    high_streak = 0
    in_static = False
    flags: list[bool] = []
    for idx, obs in enumerate(observations):
        if idx == 0:
            trans_delta = 0.0
            rot_delta = 0.0
        else:
            prev = observations[idx - 1]
            trans_delta = float(np.linalg.norm(obs.original_tvec.reshape(3) - prev.original_tvec.reshape(3)))
            rot_delta = _rvec_delta_deg(obs.original_rvec, prev.original_rvec)
        is_low = (
            np.isfinite(trans_delta)
            and trans_delta <= float(translation_gate_mm)
            and (not np.isfinite(rot_delta) or rot_delta <= float(rotation_gate_deg))
        )
        if is_low:
            low_streak += 1
            high_streak = 0
        else:
            high_streak += 1
            low_streak = 0

        if not in_static and low_streak >= max(int(settle_frames), 1):
            in_static = True
        elif in_static and high_streak >= max(int(exit_frames), 1):
            in_static = False
        flags.append(in_static)
    return flags


def evaluate_run(
    run: dict[str, Any],
    static_model: dict[str, Any],
    *,
    static_window_frames: int,
    static_window_decay: float,
    translation_gate_mm: float,
    rotation_gate_deg: float,
    settle_frames: int,
    exit_frames: int,
    robust_c_px: float,
    uv_stability_scale_px: float,
    age_ramp_frames: int,
    active_dofs: tuple[int, ...] | None,
    max_step_translation_mm: float,
    max_step_rotation_deg: float,
    max_iterations: int,
    frame_stride: int,
    max_lag_frames: int,
    reproj_guard_px: float,
    raw_motion_baseline: bool,
    include_frame_rows: bool = False,
) -> dict[str, Any]:
    observations = list(run["observations"])
    reference = observations[0]
    T = _pose_to_T(reference.original_rvec.copy(), reference.original_tvec.copy())
    config = IrlsConfig(
        max_iterations=int(max_iterations),
        robust_c_px=float(robust_c_px),
        uv_stability_scale_px=float(uv_stability_scale_px),
        condition_boost=0.0,
        age_ramp_frames=int(age_ramp_frames),
        max_step_translation_mm=float(max_step_translation_mm),
        max_step_rotation_deg=float(max_step_rotation_deg),
        active_dofs=active_dofs,
    )
    priors = build_point_priors(observations, static_model, config)
    age_maps = build_age_maps(observations, age_ramp_frames=config.age_ramp_frames)
    static_flags = _low_motion_flags(
        observations,
        translation_gate_mm=translation_gate_mm,
        rotation_gate_deg=rotation_gate_deg,
        settle_frames=settle_frames,
        exit_frames=exit_frames,
    )

    rows: list[dict[str, Any]] = []
    for obs_idx in _eval_indices(len(observations), frame_stride):
        obs = observations[obs_idx]
        static_candidate = bool(static_flags[obs_idx])
        used_frame_count = 1
        guard_reject = False

        cur_obj, cur_uv, cur_weights, _cur_keys = arrays_for_observation(
            obs,
            priors=priors,
            age_weight_by_key=age_maps[obs_idx],
        )
        logged_T = _pose_to_T(obs.original_rvec.copy(), obs.original_tvec.copy())
        logged_current_wrms = _weighted_rms_residual(cur_obj, cur_uv, cur_weights, logged_T, run["K"], run["dist"])

        if raw_motion_baseline and not static_candidate:
            result_success = True
            point_count = len(cur_obj)
            T = logged_T.copy()
            cand_current_wrms = logged_current_wrms
        else:
            if static_candidate:
                win_obj, win_uv, win_weights, win_keys, used_frame_count = arrays_for_window(
                    observations,
                    obs_idx,
                    priors=priors,
                    age_maps=age_maps,
                    window_frames=int(static_window_frames),
                    window_decay=float(static_window_decay),
                )
            else:
                win_obj, win_uv, win_weights, win_keys = arrays_for_observation(
                    obs,
                    priors=priors,
                    age_weight_by_key=age_maps[obs_idx],
                )
                used_frame_count = 1

            result = solve_irls_lie_pose(
                win_obj,
                win_uv,
                win_weights,
                run["K"],
                run["dist"],
                T,
                config,
            )
            if result.success:
                T = result.T.copy()
            point_count = int(result.point_count)
            result_success = bool(result.success)
            cand_current_wrms = _weighted_rms_residual(cur_obj, cur_uv, cur_weights, T, run["K"], run["dist"])

            reproj_excess_now = max(0.0, float(cand_current_wrms) - float(logged_current_wrms))
            if (
                static_candidate
                and np.isfinite(float(reproj_guard_px))
                and reproj_excess_now > float(reproj_guard_px)
            ):
                guard_reject = True
                T = logged_T.copy()
                cand_current_wrms = logged_current_wrms
                used_frame_count = 1
                point_count = len(cur_obj)
                result_success = True

        use_static = static_candidate and not guard_reject
        _rvec, tvec = _T_to_pose(T)
        delta_logged = np.asarray(tvec, dtype=np.float64).reshape(3) - obs.original_tvec.reshape(3)

        rows.append(
            {
                "frame": int(obs.frame),
                "static_candidate": int(static_candidate),
                "use_static": int(use_static),
                "guard_reject": int(guard_reject),
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "logged_x_mm": float(obs.original_tvec[0]),
                "logged_y_mm": float(obs.original_tvec[1]),
                "logged_z_mm": float(obs.original_tvec[2]),
                "delta_logged_mm": float(np.linalg.norm(delta_logged)),
                "point_count": int(point_count),
                "used_frame_count": int(used_frame_count),
                "solved": int(result_success),
                "current_reproj_weighted_rms_px": float(cand_current_wrms),
                "logged_current_reproj_weighted_rms_px": float(logged_current_wrms),
            }
        )

    cand_t = np.asarray([[r["tvec_x_mm"], r["tvec_y_mm"], r["tvec_z_mm"]] for r in rows], dtype=np.float64)
    logged_t = np.asarray([[r["logged_x_mm"], r["logged_y_mm"], r["logged_z_mm"]] for r in rows], dtype=np.float64)
    cand_rel = cand_t - cand_t[0].reshape(1, 3)
    logged_rel = logged_t - logged_t[0].reshape(1, 3)
    logged_ranges = np.nanmax(logged_rel, axis=0) - np.nanmin(logged_rel, axis=0)
    cand_ranges = np.nanmax(cand_rel, axis=0) - np.nanmin(cand_rel, axis=0)
    movement_axis_idx = int(np.nanargmax(logged_ranges))
    movement_axis = COMPONENTS[movement_axis_idx]
    logged_axis_range = float(logged_ranges[movement_axis_idx])
    cand_axis_range = float(cand_ranges[movement_axis_idx])
    dynamic = bool(logged_axis_range >= 20.0)
    lag = _best_lag_metrics(
        logged_rel[:, movement_axis_idx],
        cand_rel[:, movement_axis_idx],
        max_lag=max(1, int(math.ceil(float(max_lag_frames) / max(float(frame_stride), 1.0)))),
    )

    current_wrms = [_to_float(r.get("current_reproj_weighted_rms_px")) for r in rows]
    logged_wrms = [_to_float(r.get("logged_current_reproj_weighted_rms_px")) for r in rows]
    reproj_excess = [
        max(0.0, _to_float(a) - _to_float(b))
        for a, b in zip(current_wrms, logged_wrms)
        if np.isfinite(_to_float(a)) and np.isfinite(_to_float(b))
    ]
    delta_logged = [_to_float(r.get("delta_logged_mm")) for r in rows]
    amplitude_ratio = cand_axis_range / logged_axis_range if logged_axis_range > 1e-9 else math.nan

    summary = {
        "run_id": str(run["run_id"]),
        "run_label": str(Path(run["path"]).stem),
        "is_dynamic": int(dynamic),
        "frames_evaluated": int(len(rows)),
        "static_window_frames": int(static_window_frames),
        "static_window_decay": float(static_window_decay),
        "translation_gate_mm": float(translation_gate_mm),
        "rotation_gate_deg": float(rotation_gate_deg),
        "settle_frames": int(settle_frames),
        "exit_frames": int(exit_frames),
        "robust_c_px": float(robust_c_px),
        "uv_stability_scale_px": float(uv_stability_scale_px),
        "age_ramp_frames": int(age_ramp_frames),
        "active_dofs": _active_dofs_label(active_dofs),
        "max_step_translation_mm": float(max_step_translation_mm),
        "max_step_rotation_deg": float(max_step_rotation_deg),
        "max_iterations": int(max_iterations),
        "frame_stride": int(frame_stride),
        "reproj_guard_px": float(reproj_guard_px),
        "raw_motion_baseline": int(bool(raw_motion_baseline)),
        "static_candidate_fraction": float(np.mean([_to_float(r.get("static_candidate")) for r in rows])),
        "static_frame_fraction": float(np.mean([_to_float(r.get("use_static")) for r in rows])),
        "guard_reject_fraction": float(np.mean([_to_float(r.get("guard_reject")) for r in rows])),
        "solve_failures": int(sum(1 for r in rows if int(r.get("solved", 0)) == 0)),
        "movement_axis": movement_axis,
        "logged_axis_range_mm": float(logged_axis_range),
        "candidate_axis_range_mm": float(cand_axis_range),
        "amplitude_ratio": float(amplitude_ratio),
        "amplitude_error": abs(float(amplitude_ratio) - 1.0) if np.isfinite(amplitude_ratio) else math.nan,
        "best_lag_eval_steps": _to_float(lag.get("best_lag_steps")),
        "best_lag_frames": _to_float(lag.get("best_lag_steps")) * float(frame_stride),
        "best_lag_corr": _to_float(lag.get("best_lag_corr")),
        "best_lag_rmse_mm": _to_float(lag.get("best_lag_rmse_mm")),
        "x_range_mm": _range(cand_rel[:, 0]),
        "y_range_mm": _range(cand_rel[:, 1]),
        "z_range_mm": _range(cand_rel[:, 2]),
        "raw_x_range_mm": float(logged_ranges[0]),
        "raw_y_range_mm": float(logged_ranges[1]),
        "raw_z_range_mm": float(logged_ranges[2]),
        "z_closure_mm": float(cand_rel[-1, 2] - cand_rel[0, 2]),
        "delta_logged_median_mm": _median(delta_logged),
        "delta_logged_p95_mm": _percentile(delta_logged, 95),
        "current_wrms_median_px": _median(current_wrms),
        "logged_wrms_median_px": _median(logged_wrms),
        "reproj_excess_median_px": _median(reproj_excess),
        "reproj_excess_p95_px": _percentile(reproj_excess, 95),
        "point_count_median": _median([_to_float(r.get("point_count")) for r in rows]),
        "used_frame_count_median": _median([_to_float(r.get("used_frame_count")) for r in rows]),
    }
    if include_frame_rows:
        summary["_frame_rows"] = rows
    return summary


def _score(run_rows: list[dict[str, Any]]) -> dict[str, float]:
    from res_motion_irls_sweep import _score as motion_score

    return motion_score(run_rows)


def _load_runs(paths: list[Path], point_set: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    loaded = []
    for path in paths:
        run = load_run(path.resolve(), point_set=point_set)
        loaded.append((run, build_static_uv_model(run["observations"])))
    return loaded


def _select_jsonl_with_qt() -> Path:
    from debug_tracker_translation import select_jsonl_with_qt

    selected = select_jsonl_with_qt()
    if selected is None:
        raise RuntimeError("No JSONL run selected")
    return Path(selected)


def _parse_args(argv: list[str]) -> dict[str, Any]:
    preset_name = DEFAULT_PRESET
    args_list = list(argv)
    idx = 0
    while idx < len(args_list):
        if args_list[idx] == "--preset":
            idx += 1
            if idx >= len(args_list):
                raise RuntimeError(f"--preset requires one of: {', '.join(sorted(PRESETS))}")
            preset_name = args_list[idx].strip().lower()
        idx += 1
    if preset_name not in PRESETS:
        raise RuntimeError(f"Unknown preset {preset_name!r}; use one of: {', '.join(sorted(PRESETS))}")

    args: dict[str, Any] = {
        "paths": [],
        "select": 0,
        "default_runs": 0,
        "preset": preset_name,
        **PRESETS[preset_name],
    }
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--preset":
            idx += 1
        elif arg == "--static-window-frames":
            idx += 1
            args["static_window_frames"] = argv[idx]
        elif arg == "--static-window-decay":
            idx += 1
            args["static_window_decay"] = argv[idx]
        elif arg == "--translation-gate-mm":
            idx += 1
            args["translation_gate_mm"] = argv[idx]
        elif arg == "--rotation-gate-deg":
            idx += 1
            args["rotation_gate_deg"] = argv[idx]
        elif arg == "--settle-frames":
            idx += 1
            args["settle_frames"] = argv[idx]
        elif arg == "--exit-frames":
            idx += 1
            args["exit_frames"] = argv[idx]
        elif arg == "--frame-stride":
            idx += 1
            args["frame_stride"] = int(argv[idx])
        elif arg == "--max-iterations":
            idx += 1
            args["max_iterations"] = int(argv[idx])
        elif arg == "--robust-c-px":
            idx += 1
            args["robust_c_px"] = argv[idx]
        elif arg == "--uv-stability-scale-px":
            idx += 1
            args["uv_stability_scale_px"] = argv[idx]
        elif arg == "--age-ramp-frames":
            idx += 1
            args["age_ramp_frames"] = argv[idx]
        elif arg == "--active-dofs":
            idx += 1
            args["active_dofs"] = argv[idx]
        elif arg == "--max-step-translation-mm":
            idx += 1
            args["max_step_translation_mm"] = argv[idx]
        elif arg == "--max-step-rotation-deg":
            idx += 1
            args["max_step_rotation_deg"] = argv[idx]
        elif arg == "--point-set":
            idx += 1
            args["point_set"] = argv[idx]
        elif arg == "--reproj-guard-px":
            idx += 1
            args["reproj_guard_px"] = argv[idx]
        elif arg == "--raw-motion-baseline":
            args["raw_motion_baseline"] = 1
        elif arg == "--no-raw-motion-baseline":
            args["raw_motion_baseline"] = 0
        elif arg == "--plot":
            args["plot"] = 1
        elif arg == "--no-plot":
            args["plot"] = 0
        elif arg == "--write-frame-csv":
            args["write_frame_csv"] = 1
        elif arg == "--no-frame-csv":
            args["write_frame_csv"] = 0
        elif arg in ("--select", "select"):
            args["select"] = 1
        elif arg == "--default-runs":
            args["default_runs"] = 1
        elif arg == "--tag":
            idx += 1
            args["tag"] = argv[idx]
        elif arg.endswith(".jsonl"):
            args["paths"].append(Path(arg))
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1
    if not args["paths"]:
        if args["default_runs"]:
            args["paths"] = [Path(p) for p in DEFAULT_RUNS]
        else:
            args["paths"] = [_select_jsonl_with_qt()]
    return args


def main() -> None:
    args = _parse_args(sys.argv[1:])
    paths = [Path(p) for p in args["paths"]]
    loaded_runs = _load_runs(paths, str(args["point_set"]).strip().lower())

    windows = _parse_list(args["static_window_frames"], int)
    decays = _parse_list(args["static_window_decay"], float)
    trans_gates = _parse_list(args["translation_gate_mm"], float)
    rot_gates = _parse_list(args["rotation_gate_deg"], float)
    settles = _parse_list(args["settle_frames"], int)
    exits = _parse_list(args["exit_frames"], int)
    guards = _parse_list(args["reproj_guard_px"], float)
    combos = list(itertools.product(windows, decays, trans_gates, rot_gates, settles, exits, guards))
    active_dofs = _parse_active_dofs(args["active_dofs"])
    print(
        f"[adaptive_irls] preset={args['preset']} runs={len(loaded_runs)} "
        f"combos={len(combos)} active_dofs={_active_dofs_label(active_dofs)} tag={args['tag']}"
    )

    run_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for combo_idx, (window, decay, trans_gate, rot_gate, settle, exit_frames, reproj_guard) in enumerate(combos, start=1):
        combo_rows = []
        for run, static_model in loaded_runs:
            row = evaluate_run(
                run,
                static_model,
                static_window_frames=int(window),
                static_window_decay=float(decay),
                translation_gate_mm=float(trans_gate),
                rotation_gate_deg=float(rot_gate),
                settle_frames=int(settle),
                exit_frames=int(exit_frames),
                robust_c_px=float(args["robust_c_px"]),
                uv_stability_scale_px=float(args["uv_stability_scale_px"]),
                age_ramp_frames=int(args["age_ramp_frames"]),
                active_dofs=active_dofs,
                max_step_translation_mm=float(args["max_step_translation_mm"]),
                max_step_rotation_deg=float(args["max_step_rotation_deg"]),
                max_iterations=int(args["max_iterations"]),
                frame_stride=int(args["frame_stride"]),
                max_lag_frames=int(args["max_lag_frames"]),
                reproj_guard_px=float(reproj_guard),
                raw_motion_baseline=bool(args["raw_motion_baseline"]),
                include_frame_rows=bool(args["plot"] or args["write_frame_csv"]),
            )
            frame_rows = row.pop("_frame_rows", None)
            if frame_rows:
                stem = Path(run["path"]).stem
                tag = str(args["tag"]).strip() or "adaptive"
                safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag)
                if args["write_frame_csv"]:
                    frame_csv = Path(run["path"]).resolve().with_name(f"{stem}_{safe_tag}_adaptive_irls_frames.csv")
                    _write_csv(frame_csv, frame_rows)
                    row["frame_csv"] = str(frame_csv.resolve())
                    print(f"[adaptive_irls] saved frames -> {frame_csv.resolve()}")
                if args["plot"]:
                    plot_path = _plot_adaptive_translation(run, row, frame_rows, tag)
                    row["plot_path"] = str(plot_path.resolve())
            combo_rows.append(row)
            run_rows.append(row)

        scored = _score(combo_rows)
        combined_rows.append(
            {
                "rank": 0,
                "static_window_frames": int(window),
                "static_window_decay": float(decay),
                "translation_gate_mm": float(trans_gate),
                "rotation_gate_deg": float(rot_gate),
                "settle_frames": int(settle),
                "exit_frames": int(exit_frames),
                "reproj_guard_px": float(reproj_guard),
                "raw_motion_baseline": int(bool(args["raw_motion_baseline"])),
                "active_dofs": _active_dofs_label(active_dofs),
                **scored,
            }
        )
        if combo_idx == 1 or combo_idx % 10 == 0 or combo_idx == len(combos):
            elapsed = time.perf_counter() - t0
            remaining = (elapsed / float(combo_idx)) * float(len(combos) - combo_idx)
            best = min(combined_rows, key=lambda r: _to_float(r.get("score")))
            print(
                "[adaptive_irls] "
                f"{combo_idx}/{len(combos)} elapsed={elapsed:.1f}s remaining={remaining:.1f}s "
                f"best_score={_to_float(best.get('score')):.4f} "
                f"w={best['static_window_frames']} tg={best['translation_gate_mm']} "
                f"rg={best['rotation_gate_deg']} settle={best['settle_frames']} "
                f"exit={best['exit_frames']} "
                f"guard={best['reproj_guard_px']}"
            )

    combined_rows.sort(key=lambda r: (_to_float(r.get("score")), _to_float(r.get("max_static_z_range_mm"))))
    for rank, row in enumerate(combined_rows, start=1):
        row["rank"] = int(rank)

    out_dir = paths[0].resolve().parent
    tag = str(args["tag"]).strip() or "adaptive"
    combined_csv = out_dir / f"hydramarker_motion_irls_adaptive_{tag}_combined.csv"
    runs_csv = out_dir / f"hydramarker_motion_irls_adaptive_{tag}_runs.csv"
    _write_csv(combined_csv, combined_rows)
    _write_csv(runs_csv, run_rows)
    print(f"[adaptive_irls] saved combined -> {combined_csv.resolve()}")
    print(f"[adaptive_irls] saved runs     -> {runs_csv.resolve()}")
    print("[adaptive_irls] top 10:")
    for row in combined_rows[:10]:
        print(
            "  "
            f"#{int(row['rank'])}: score={_to_float(row.get('score')):.4f}, "
            f"dyn_amp={_to_float(row.get('max_dynamic_amplitude_error')):.3f}, "
            f"dyn_lag={_to_float(row.get('max_dynamic_abs_lag_frames')):.1f}, "
            f"dyn_delta={_to_float(row.get('max_dynamic_delta_logged_p95_mm')):.3f}, "
            f"dyn_reproj={_to_float(row.get('max_dynamic_reproj_excess_p95_px')):.3f}, "
            f"static_z={_to_float(row.get('max_static_z_range_mm')):.3f}, "
            f"dof={row.get('active_dofs')}, "
            f"w={row['static_window_frames']}, tg={row['translation_gate_mm']}, "
            f"rg={row['rotation_gate_deg']}, settle={row['settle_frames']}, "
            f"exit={row['exit_frames']}, "
            f"guard={row['reproj_guard_px']}"
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[adaptive_irls] ERROR: {exc}")
        sys.exit(1)
