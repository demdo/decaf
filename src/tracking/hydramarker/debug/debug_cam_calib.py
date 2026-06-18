"""Plot the radial distortion function of a saved OpenCV camera calibration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np


K_KEYS = ("K", "K_rgb", "camera_matrix", "camera_intrinsics", "intrinsics")
DIST_KEYS = (
    "dist",
    "dist_rgb",
    "dist_coeffs",
    "distortion_coeffs",
    "opencv_dist_coeffs",
    "effective_opencv_dist_coeffs",
)


@dataclass(frozen=True)
class CameraCalibration:
    path: Path
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int] | None
    info: dict[str, Any]


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


def choose_calibration_qt() -> Path | None:
    try:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is required for the file dialog. Pass --calib on the command line "
            "or run in the Qt-enabled project environment."
        ) from exc

    _ensure_qt_app()
    default_dir = Path(__file__).resolve().parents[1] / "data" / "realsense"
    if not default_dir.exists():
        default_dir = Path.cwd()

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select camera calibration NPZ",
        str(default_dir),
        "Camera calibration NPZ (*.npz);;All Files (*)",
    )
    if not path:
        QMessageBox.information(None, "Camera calibration debug", "No calibration selected.")
        return None
    return Path(path)


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
    width = next((int(np.asarray(npz[key]).reshape(-1)[0]) for key in width_keys if key in npz.files), None)
    height = next((int(np.asarray(npz[key]).reshape(-1)[0]) for key in height_keys if key in npz.files), None)
    if width is not None and height is not None:
        return width, height
    return None


def load_calibration(path: Path) -> CameraCalibration:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {path}")

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
            "num_candidates_total": scalar_npz_value(npz, "num_candidates_total"),
            "num_candidates_selected": scalar_npz_value(npz, "num_candidates_selected"),
            "selected_reprojection_mean_px": scalar_npz_value(npz, "selected_reprojection_mean_px"),
            "distortion_model": scalar_npz_value(npz, "distortion_model"),
        }

    return CameraCalibration(
        path=path,
        K=np.asarray(K, dtype=np.float64).reshape(3, 3),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        image_size=image_size,
        info=info,
    )


def distortion_coefficients(dist: np.ndarray) -> dict[str, float]:
    values = np.asarray(dist, dtype=np.float64).reshape(-1)
    names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tau_x", "tau_y")
    return {name: float(values[idx]) for idx, name in enumerate(names[: len(values)])}


def radial_scale(r: np.ndarray, dist: np.ndarray) -> np.ndarray:
    coeff = distortion_coefficients(dist)
    r2 = np.asarray(r, dtype=np.float64) ** 2
    r4 = r2 * r2
    r6 = r4 * r2

    numerator = (
        1.0
        + coeff.get("k1", 0.0) * r2
        + coeff.get("k2", 0.0) * r4
        + coeff.get("k3", 0.0) * r6
    )

    if len(dist) >= 8:
        denominator = (
            1.0
            + coeff.get("k4", 0.0) * r2
            + coeff.get("k5", 0.0) * r4
            + coeff.get("k6", 0.0) * r6
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return numerator / denominator

    return numerator


def image_radius_stats(calib: CameraCalibration) -> dict[str, float]:
    if calib.image_size is None:
        return {}

    width, height = calib.image_size
    fx = float(calib.K[0, 0])
    fy = float(calib.K[1, 1])
    cx = float(calib.K[0, 2])
    cy = float(calib.K[1, 2])

    points = {
        "left_edge": (0.0, cy),
        "right_edge": (float(width - 1), cy),
        "top_edge": (cx, 0.0),
        "bottom_edge": (cx, float(height - 1)),
        "top_left": (0.0, 0.0),
        "top_right": (float(width - 1), 0.0),
        "bottom_left": (0.0, float(height - 1)),
        "bottom_right": (float(width - 1), float(height - 1)),
    }

    stats: dict[str, float] = {}
    for name, (u, v) in points.items():
        x = (u - cx) / fx
        y = (v - cy) / fy
        stats[name] = float(np.hypot(x, y))

    stats["max_corner"] = max(stats[name] for name in ("top_left", "top_right", "bottom_left", "bottom_right"))
    stats["max_axis_edge"] = max(stats[name] for name in ("left_edge", "right_edge", "top_edge", "bottom_edge"))
    return stats


def default_output_path(calib_path: Path) -> Path:
    return calib_path.with_name(f"{calib_path.stem}_radial_distortion.png")


def print_calibration_summary(calib: CameraCalibration, radius_stats: dict[str, float]) -> None:
    coeff = distortion_coefficients(calib.dist)
    print()
    print("Camera calibration radial distortion")
    print("====================================")
    print(f"File: {calib.path}")
    print(f"Image size: {calib.image_size}")
    print(f"K key: {calib.info['K_key']}  dist key: {calib.info['dist_key']}")
    print(f"Distortion model: {calib.info.get('distortion_model')}")
    print(f"Created at: {calib.info.get('created_at')}")
    print(f"RMS: {calib.info.get('rms')}")
    print(f"Images used: {calib.info.get('num_images_used')}")
    if calib.info.get("num_candidates_total") != "":
        print(
            "Candidates: "
            f"{calib.info.get('num_candidates_selected')}/"
            f"{calib.info.get('num_candidates_total')}"
        )
    print()
    print("K:")
    print(calib.K)
    print()
    print(f"dist length: {len(calib.dist)}")
    print("dist:")
    print(calib.dist.tolist())
    print()
    print("Named coefficients:")
    for name, value in coeff.items():
        print(f"  {name:5s} = {value: .12g}")
    print()

    if radius_stats:
        print("Normalized image radii:")
        for key in (
            "max_axis_edge",
            "max_corner",
            "left_edge",
            "right_edge",
            "top_edge",
            "bottom_edge",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ):
            print(f"  {key:14s} = {radius_stats[key]:.6f}")
        print()


def plot_radial_function(
    calib: CameraCalibration,
    *,
    output_path: Path,
    samples: int,
    show: bool,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to plot the radial distortion function. "
            "Run this script in the same Python environment that you use for the drift plots."
        ) from exc

    radius_stats = image_radius_stats(calib)
    r_max = float(radius_stats.get("max_corner", 1.0))
    r_plot = max(1.0, r_max * 1.08)
    r = np.linspace(0.0, r_plot, int(samples))
    scale = radial_scale(r, calib.dist)

    fx = float(calib.K[0, 0])
    fy = float(calib.K[1, 1])
    f_mean = 0.5 * (fx + fy)
    radial_shift_px = r * (scale - 1.0) * f_mean

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    fig.suptitle(f"Radial distortion: {calib.path.name}")

    axes[0].plot(r, scale, linewidth=2.0, label="radial scale")
    axes[0].axhline(1.0, color="0.65", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("scale")
    axes[0].grid(True, alpha=0.28)

    axes[1].plot(r, radial_shift_px, linewidth=2.0, color="tab:orange", label="radial shift")
    axes[1].axhline(0.0, color="0.65", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("normalized radius r")
    axes[1].set_ylabel("approx. radial shift [px]")
    axes[1].grid(True, alpha=0.28)

    markers = []
    if radius_stats:
        markers = [
            ("axis edge", radius_stats["max_axis_edge"]),
            ("corner", radius_stats["max_corner"]),
        ]

    for label, value in markers:
        for ax in axes:
            ax.axvline(value, color="tab:red", linestyle=":", linewidth=1.3)
            ax.text(
                value,
                0.97,
                label,
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=9,
                color="tab:red",
                rotation=90,
            )

    coeff = distortion_coefficients(calib.dist)
    coeff_text = "\n".join(
        f"{name}={value:.4g}"
        for name, value in coeff.items()
        if name.startswith("k")
    )
    axes[0].text(
        0.015,
        0.04,
        coeff_text,
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a camera calibration NPZ and plot its OpenCV radial distortion function."
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help="Calibration .npz path. If omitted, a Qt file dialog opens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <calib>_radial_distortion.png next to the NPZ.",
    )
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a Matplotlib window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_path = Path(args.calib) if args.calib is not None else choose_calibration_qt()
    if calib_path is None:
        return

    calib = load_calibration(calib_path)
    radius_stats = image_radius_stats(calib)
    print_calibration_summary(calib, radius_stats)

    output_path = Path(args.output) if args.output is not None else default_output_path(calib.path)
    plot_radial_function(
        calib,
        output_path=output_path,
        samples=max(50, int(args.samples)),
        show=not bool(args.no_show),
    )


if __name__ == "__main__":
    main()
