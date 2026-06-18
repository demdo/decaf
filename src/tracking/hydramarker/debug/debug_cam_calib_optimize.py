"""Optimize a rational8 camera calibration from existing ChArUco image sets.

This script uses two independent image folders:
1. calibration images used to fit new rational8 variants
2. validation images used only to score the variants

The default optimization is conservative: it fits rational8 on all valid
calibration images, ranks calibration images by their per-view reprojection
error, then refits rational8 after removing the worst frames while preserving
basic FOV coverage.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATE_PATH = SCRIPT_DIR / "debug_cam_calib_validate.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validate_mod = _load_module(
    "hydramarker_debug_cam_calib_validate_runtime",
    VALIDATE_PATH,
)
calib_camera = validate_mod.calib_camera


@dataclass
class CalibrationImage:
    index: int
    path: Path
    image: np.ndarray
    valid: bool
    num_charuco: int
    num_aruco: int
    centroid_u: float
    centroid_v: float
    bbox_area_norm: float
    grid_cell: tuple[int, int]
    initial_train_error_px: float | None = None


@dataclass(frozen=True)
class FitResult:
    variant: str
    path: Path
    kept_indices: list[int]
    removed_indices: list[int]
    requested_trim_fraction: float
    actual_trim_fraction: float
    train_rms: float
    train_reprojection_mean_px: float
    train_reprojection_p95_px: float
    train_reprojection_max_px: float
    K: np.ndarray
    dist: np.ndarray
    radial_stats: dict[str, Any]


def _ensure_qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is required for folder dialogs. Pass --calib-images-dir "
            "and --validation-images-dir on the command line, or run in the "
            "Qt-enabled project environment."
        ) from exc

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def select_directory_qt(title: str, default_dir: Path) -> Path | None:
    try:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is required for folder dialogs. Pass paths on the command line "
            "or run in the Qt-enabled project environment."
        ) from exc

    _ensure_qt_app()
    if not default_dir.exists():
        default_dir = Path.cwd()
    path = QFileDialog.getExistingDirectory(None, title, str(default_dir))
    if not path:
        QMessageBox.information(None, "Camera calibration optimization", f"No folder selected for: {title}")
        return None
    return Path(path)


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "cam_calib_optimization" / f"optimization_{stamp}"


def collect_image_paths(images_dir: Path) -> list[Path]:
    return validate_mod.collect_image_paths(images_dir)


def directory_has_images(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return any(child.is_file() and child.suffix.lower() in suffixes for child in path.iterdir())


def resolve_image_dir(path: Path, preferred_subdirs: Sequence[str]) -> Path:
    path = Path(path).expanduser().resolve()
    for subdir in preferred_subdirs:
        candidate = path / subdir
        if directory_has_images(candidate):
            print(f"[cam_calib_optimize] Using image subfolder: {candidate}")
            return candidate
    if directory_has_images(path):
        return path
    for subdir in preferred_subdirs:
        matches = [
            candidate
            for candidate in path.rglob(subdir)
            if candidate.is_dir() and directory_has_images(candidate)
        ]
        if matches:
            chosen = sorted(matches)[0]
            print(f"[cam_calib_optimize] Using image subfolder: {chosen}")
            return chosen
    return path


def load_calibration_images(
    image_paths: Sequence[Path],
    *,
    min_corners: int,
    grid_cols: int,
    grid_rows: int,
) -> list[CalibrationImage]:
    board, aruco_dict, detector_params = calib_camera.make_charuco_board()
    infos: list[CalibrationImage] = []

    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            infos.append(
                CalibrationImage(
                    index=index,
                    path=path,
                    image=np.empty((0, 0, 3), dtype=np.uint8),
                    valid=False,
                    num_charuco=0,
                    num_aruco=0,
                    centroid_u=float("nan"),
                    centroid_v=float("nan"),
                    bbox_area_norm=float("nan"),
                    grid_cell=(-1, -1),
                )
            )
            continue

        det = calib_camera.detect_charuco(
            image,
            board=board,
            aruco_dict=aruco_dict,
            detector_params=detector_params,
        )
        valid = det.charuco_corners is not None and det.charuco_ids is not None and det.num_charuco >= min_corners
        metrics = calib_camera._candidate_metrics(image, det)
        u_norm = float(np.nan_to_num(metrics.get("centroid_u_norm", 0.5), nan=0.5))
        v_norm = float(np.nan_to_num(metrics.get("centroid_v_norm", 0.5), nan=0.5))
        grid_cell = (
            int(np.clip(np.floor(u_norm * grid_cols), 0, grid_cols - 1)),
            int(np.clip(np.floor(v_norm * grid_rows), 0, grid_rows - 1)),
        )
        if not valid:
            grid_cell = (-1, -1)

        infos.append(
            CalibrationImage(
                index=index,
                path=path,
                image=image,
                valid=bool(valid),
                num_charuco=int(det.num_charuco),
                num_aruco=int(det.num_aruco),
                centroid_u=float(metrics.get("centroid_u", np.nan)),
                centroid_v=float(metrics.get("centroid_v", np.nan)),
                bbox_area_norm=float(metrics.get("bbox_area_norm", np.nan)),
                grid_cell=grid_cell,
            )
        )

    return infos


def _finite_values(values: Sequence[float | None]) -> np.ndarray:
    out = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return np.asarray(out, dtype=np.float64)


def safe_percentile(values: Sequence[float | None], q: float) -> float:
    arr = _finite_values(values)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def fit_rational8_variant(
    *,
    variant: str,
    all_infos: Sequence[CalibrationImage],
    kept_indices: Sequence[int],
    requested_trim_fraction: float,
    output_dir: Path,
    min_corners: int,
) -> FitResult:
    board, aruco_dict, detector_params = calib_camera.make_charuco_board()
    kept_indices = sorted(int(i) for i in kept_indices)
    images = [all_infos[i].image for i in kept_indices]
    if len(images) < 3:
        raise RuntimeError(f"{variant}: at least 3 valid calibration images are required.")

    image_size = calib_camera._image_size(images[0])
    flags = int(getattr(cv2, "CALIB_RATIONAL_MODEL", 0))
    K, dist, rms, stats = calib_camera.calibrate_charuco_intrinsics(
        calib_images=images,
        board=board,
        aruco_dict=aruco_dict,
        detector_params=detector_params,
        min_charuco_corners=min_corners,
        flags=flags,
        dist_coeff_count=8,
    )

    train_mean, train_per_view, _reproj_stats = calib_camera.reprojection_error_charuco(
        images,
        board=board,
        aruco_dict=aruco_dict,
        K=K,
        dist=dist,
        detector_params=detector_params,
        min_charuco_corners=min_corners,
    )
    train_values = _finite_values(train_per_view)
    radial_stats = calib_camera.radial_plausibility_stats(K, dist, image_size)
    valid_indices = [info.index for info in all_infos if info.valid]
    removed_indices = sorted(set(valid_indices) - set(kept_indices))
    actual_trim_fraction = (
        float(len(removed_indices) / len(valid_indices)) if valid_indices else 0.0
    )

    stats.update(
        {
            "calibration_model": f"rational8_optimized_{variant}",
            "calibration_model_description": (
                "OpenCV rational model optimized by FOV-protected per-view "
                "outlier removal from existing ChArUco calibration images"
            ),
            "distortion_model": "opencv_brown_conrady_rational",
            "selected_reprojection_mean_px": float(train_mean),
            "selected_reprojection_per_view_px": [
                float("nan") if value is None else float(value)
                for value in train_per_view
            ],
            "num_candidates_total": int(len(valid_indices)),
            "num_candidates_selected": int(len(kept_indices)),
            "selected_frame_indices": [int(i) for i in kept_indices],
            "selected_candidate_num_charuco": [
                int(all_infos[i].num_charuco) for i in kept_indices
            ],
            "selected_candidate_centroids_uv": [
                [float(all_infos[i].centroid_u), float(all_infos[i].centroid_v)]
                for i in kept_indices
            ],
            "selected_candidate_bbox_area_norm": [
                float(all_infos[i].bbox_area_norm) for i in kept_indices
            ],
            **radial_stats,
        }
    )

    path = output_dir / "models" / f"{variant}.npz"
    calib_camera.save_tracking_calibration_npz(
        path,
        K=K,
        dist=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        image_size=image_size,
        rms=float(rms),
        stats=stats,
    )

    return FitResult(
        variant=variant,
        path=path.resolve(),
        kept_indices=list(kept_indices),
        removed_indices=removed_indices,
        requested_trim_fraction=float(requested_trim_fraction),
        actual_trim_fraction=actual_trim_fraction,
        train_rms=float(rms),
        train_reprojection_mean_px=float(train_mean),
        train_reprojection_p95_px=safe_percentile(train_per_view, 95),
        train_reprojection_max_px=float(np.max(train_values)) if train_values.size else float("nan"),
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        radial_stats=radial_stats,
    )


def rank_images_by_initial_error(
    infos: Sequence[CalibrationImage],
    initial_per_view_errors: Sequence[float | None],
) -> None:
    valid_infos = [info for info in infos if info.valid]
    if len(initial_per_view_errors) != len(valid_infos):
        raise RuntimeError(
            "Initial per-view errors do not match the number of valid calibration images."
        )
    for info, error in zip(valid_infos, initial_per_view_errors):
        info.initial_train_error_px = None if error is None else float(error)


def select_fov_protected_subset(
    infos: Sequence[CalibrationImage],
    *,
    trim_fraction: float,
    min_views: int,
    min_per_cell: int,
) -> list[int]:
    valid_infos = [info for info in infos if info.valid]
    if not valid_infos:
        return []

    remove_target = int(np.ceil(len(valid_infos) * max(0.0, float(trim_fraction))))
    remove_target = max(0, min(remove_target, max(0, len(valid_infos) - min_views)))
    kept = {info.index for info in valid_infos}
    cell_counts: dict[tuple[int, int], int] = {}
    for info in valid_infos:
        cell_counts[info.grid_cell] = cell_counts.get(info.grid_cell, 0) + 1

    def sort_key(info: CalibrationImage) -> float:
        if info.initial_train_error_px is None:
            return -float("inf")
        return float(info.initial_train_error_px)

    removed = 0
    for info in sorted(valid_infos, key=sort_key, reverse=True):
        if removed >= remove_target:
            break
        if len(kept) <= min_views:
            break
        cell = info.grid_cell
        if cell_counts.get(cell, 0) <= min_per_cell:
            continue
        kept.remove(info.index)
        cell_counts[cell] -= 1
        removed += 1

    return sorted(kept)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    validate_mod.write_csv(path, rows)


def write_calibration_image_ranking(
    path: Path,
    infos: Sequence[CalibrationImage],
    fit_results: Sequence[FitResult],
) -> None:
    rows: list[dict[str, Any]] = []
    for info in infos:
        row: dict[str, Any] = {
            "image_index": info.index,
            "image_path": str(info.path),
            "valid": int(info.valid),
            "num_charuco": int(info.num_charuco),
            "num_aruco": int(info.num_aruco),
            "centroid_u": info.centroid_u,
            "centroid_v": info.centroid_v,
            "bbox_area_norm": info.bbox_area_norm,
            "grid_col": info.grid_cell[0],
            "grid_row": info.grid_cell[1],
            "initial_train_error_px": (
                ""
                if info.initial_train_error_px is None
                else float(info.initial_train_error_px)
            ),
        }
        for result in fit_results:
            row[f"kept_{result.variant}"] = int(info.index in result.kept_indices)
        rows.append(row)
    write_csv(path, rows)


def evaluate_saved_models(
    *,
    fit_results: Sequence[FitResult],
    validation_image_paths: Sequence[Path],
    output_dir: Path,
    min_corners: int,
) -> list[dict[str, Any]]:
    detected_images, detection_rows = validate_mod.load_detected_images(
        validation_image_paths,
        min_corners=min_corners,
    )
    write_csv(output_dir / "validation_image_detections.csv", detection_rows)
    if not detected_images:
        raise RuntimeError("No validation images with enough ChArUco corners were found.")

    model_paths = [result.path for result in fit_results]
    models = [validate_mod.load_camera_model(path) for path in model_paths]
    image_size = detected_images[0].image_size
    all_per_image_rows: list[dict[str, Any]] = []
    all_per_corner_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    result_by_name = {result.path.stem: result for result in fit_results}
    for model in models:
        per_image, per_corner = validate_mod.validate_model_on_images(model, detected_images)
        all_per_image_rows.extend(per_image)
        all_per_corner_rows.extend(per_corner)
        summary = validate_mod.summarize_model(model, per_image, per_corner)
        fit = result_by_name.get(model.path.stem)
        if fit is not None:
            summary.update(
                {
                    "variant": fit.variant,
                    "requested_trim_fraction": fit.requested_trim_fraction,
                    "actual_trim_fraction": fit.actual_trim_fraction,
                    "kept_image_count": len(fit.kept_indices),
                    "removed_image_count": len(fit.removed_indices),
                    "kept_indices": " ".join(str(i) for i in fit.kept_indices),
                    "removed_indices": " ".join(str(i) for i in fit.removed_indices),
                    "train_rms": fit.train_rms,
                    "train_reprojection_mean_px": fit.train_reprojection_mean_px,
                    "train_reprojection_p95_px": fit.train_reprojection_p95_px,
                    "train_reprojection_max_px": fit.train_reprojection_max_px,
                }
            )
        summary_rows.append(summary)
        print(
            "[cam_calib_optimize] "
            f"{model.name}: validation_rms={float(summary.get('rms_px', float('nan'))):.4f}px "
            f"p95={float(summary.get('p95_px', float('nan'))):.4f}px "
            f"edge={float(summary.get('edge_mean_px', float('nan'))):.4f}px"
        )

    write_csv(output_dir / "optimization_summary.csv", summary_rows)
    write_csv(output_dir / "validation_residuals_by_image.csv", all_per_image_rows)
    write_csv(output_dir / "validation_residuals_by_corner.csv", all_per_corner_rows)
    validate_mod.plot_outputs(
        output_dir=output_dir,
        models=models,
        per_image_rows=all_per_image_rows,
        per_corner_rows=all_per_corner_rows,
        image_size=image_size,
    )
    plot_optimization_scores(output_dir / "optimization_validation_score.png", summary_rows)
    return summary_rows


def plot_optimization_scores(path: Path, summary_rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[cam_calib_optimize] Matplotlib unavailable; skipping score plot: {exc}")
        return

    rows = [row for row in summary_rows if row.get("status") == "ok"]
    if not rows:
        return

    labels = [str(row.get("variant") or row.get("model")) for row in rows]
    rms = [float(row["rms_px"]) for row in rows]
    p95 = [float(row["p95_px"]) for row in rows]
    edge = [float(row["edge_mean_px"]) for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(rows)), 5))
    ax.plot(x, rms, marker="o", label="validation RMS")
    ax.plot(x, p95, marker="o", label="validation p95")
    ax.plot(x, edge, marker="o", label="validation edge mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("error [px]")
    ax.set_title("rational8 optimization validation score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def optimize_rational8(
    *,
    calibration_image_paths: Sequence[Path],
    validation_image_paths: Sequence[Path],
    output_dir: Path,
    min_corners: int,
    trim_fractions: Sequence[float],
    min_views: int,
    grid_cols: int,
    grid_rows: int,
    min_per_cell: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    infos = load_calibration_images(
        calibration_image_paths,
        min_corners=min_corners,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )
    valid_indices = [info.index for info in infos if info.valid]
    if len(valid_indices) < max(3, min_views):
        raise RuntimeError(
            f"Only {len(valid_indices)} valid calibration images found; need at least {max(3, min_views)}."
        )

    print(
        "[cam_calib_optimize] Loaded "
        f"{len(calibration_image_paths)} calibration images, {len(valid_indices)} valid."
    )
    print(
        "[cam_calib_optimize] Validation images: "
        f"{len(validation_image_paths)}"
    )

    fit_results: list[FitResult] = []
    all_result = fit_rational8_variant(
        variant="rational8_all",
        all_infos=infos,
        kept_indices=valid_indices,
        requested_trim_fraction=0.0,
        output_dir=output_dir,
        min_corners=min_corners,
    )
    fit_results.append(all_result)

    # Rank images only from the all-images rational8 fit. Validation data is
    # intentionally not used for selection.
    rank_images_by_initial_error(
        infos,
        validate_train_errors_by_indices(
            all_result,
            infos,
            min_corners=min_corners,
        ),
    )

    for trim_fraction in trim_fractions:
        trim_fraction = float(trim_fraction)
        if trim_fraction <= 0.0:
            continue
        kept = select_fov_protected_subset(
            infos,
            trim_fraction=trim_fraction,
            min_views=min_views,
            min_per_cell=min_per_cell,
        )
        if sorted(kept) == sorted(valid_indices):
            print(
                "[cam_calib_optimize] "
                f"trim {trim_fraction:.2f}: no removable image under FOV protection."
            )
            continue
        actual_removed = len(valid_indices) - len(kept)
        variant = f"rational8_trim{int(round(trim_fraction * 100)):02d}_fov"
        print(
            "[cam_calib_optimize] "
            f"{variant}: keeping {len(kept)}/{len(valid_indices)} "
            f"(removed {actual_removed})."
        )
        fit_results.append(
            fit_rational8_variant(
                variant=variant,
                all_infos=infos,
                kept_indices=kept,
                requested_trim_fraction=trim_fraction,
                output_dir=output_dir,
                min_corners=min_corners,
            )
        )

    write_calibration_image_ranking(
        output_dir / "calibration_image_ranking.csv",
        infos,
        fit_results,
    )
    summary_rows = evaluate_saved_models(
        fit_results=fit_results,
        validation_image_paths=validation_image_paths,
        output_dir=output_dir,
        min_corners=min_corners,
    )
    best = min(summary_rows, key=lambda row: float(row.get("rms_px", float("inf"))))
    print()
    print("[cam_calib_optimize] Best validation RMS:")
    print(f"  variant: {best.get('variant') or best.get('model')}")
    print(f"  path:    {best.get('path')}")
    print(f"  rms:     {float(best.get('rms_px', float('nan'))):.6f} px")
    print(f"  p95:     {float(best.get('p95_px', float('nan'))):.6f} px")
    print(f"  edge:    {float(best.get('edge_mean_px', float('nan'))):.6f} px")
    print(f"[cam_calib_optimize] Saved outputs -> {output_dir}")


def validate_train_errors_by_indices(
    fit_result: FitResult,
    infos: Sequence[CalibrationImage],
    *,
    min_corners: int,
) -> list[float | None]:
    board, aruco_dict, detector_params = calib_camera.make_charuco_board()
    valid_infos = [info for info in infos if info.valid]
    images = [info.image for info in valid_infos]
    _mean, per_view, _stats = calib_camera.reprojection_error_charuco(
        images,
        board=board,
        aruco_dict=aruco_dict,
        K=fit_result.K,
        dist=fit_result.dist,
        detector_params=detector_params,
        min_charuco_corners=min_corners,
    )
    return per_view


def parse_trim_fractions(values: Sequence[float]) -> list[float]:
    unique = sorted({max(0.0, min(0.9, float(v))) for v in values})
    if 0.0 not in unique:
        unique.insert(0, 0.0)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize rational8 camera calibration from existing ChArUco calibration "
            "images and score each variant on independent validation images."
        )
    )
    parser.add_argument(
        "--calib-images-dir",
        type=Path,
        default=None,
        help="Folder with calibration images. If omitted, a Qt folder dialog opens.",
    )
    parser.add_argument(
        "--validation-images-dir",
        type=Path,
        default=None,
        help="Folder with independent validation images. If omitted, a Qt folder dialog opens.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for optimized NPZs, CSVs and plots.",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=calib_camera.MIN_CHARUCO_CAPTURE,
        help="Minimum ChArUco corners for calibration/validation images.",
    )
    parser.add_argument(
        "--trim-fractions",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
        help="Fractions of worst calibration views to remove, FOV-protected.",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=18,
        help="Never keep fewer calibration views than this.",
    )
    parser.add_argument("--grid-cols", type=int, default=calib_camera.SELECTION_GRID_COLS)
    parser.add_argument("--grid-rows", type=int, default=calib_camera.SELECTION_GRID_ROWS)
    parser.add_argument(
        "--min-per-cell",
        type=int,
        default=1,
        help="Keep at least this many images in every occupied centroid grid cell.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    default_realsense_dir = (
        SCRIPT_DIR.parents[1]
        / "hydramarker"
        / "data"
        / "realsense"
    )
    default_validation_dir = SCRIPT_DIR / "cam_calib_validation"

    calib_dir = args.calib_images_dir
    if calib_dir is None:
        calib_dir = select_directory_qt(
            "Select calibration image folder, e.g. selected_views",
            default_realsense_dir,
        )
    if calib_dir is None:
        raise RuntimeError("No calibration image folder selected.")

    validation_dir = args.validation_images_dir
    if validation_dir is None:
        validation_dir = select_directory_qt(
            "Select independent validation image folder, e.g. validation_frames",
            default_validation_dir,
        )
    if validation_dir is None:
        raise RuntimeError("No validation image folder selected.")

    calib_dir = resolve_image_dir(calib_dir, ("selected_views",))
    validation_dir = resolve_image_dir(validation_dir, ("validation_frames",))

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir()
    )

    optimize_rational8(
        calibration_image_paths=collect_image_paths(calib_dir),
        validation_image_paths=collect_image_paths(validation_dir),
        output_dir=output_dir,
        min_corners=max(3, int(args.min_corners)),
        trim_fractions=parse_trim_fractions(args.trim_fractions),
        min_views=max(3, int(args.min_views)),
        grid_cols=max(1, int(args.grid_cols)),
        grid_rows=max(1, int(args.grid_rows)),
        min_per_cell=max(0, int(args.min_per_cell)),
    )


if __name__ == "__main__":
    main()
