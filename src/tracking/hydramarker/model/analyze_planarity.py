from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtWidgets import QApplication, QFileDialog


def load_points_from_marker_geometry(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points_raw = data["corners"]

    ids = []
    pts = []

    for p in points_raw:
        marker_id = int(p["id"])
        xyz = p["xyz_mm"]

        ids.append(marker_id)
        pts.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    return np.asarray(ids, dtype=int), np.asarray(pts, dtype=float)


def fit_best_plane(points: np.ndarray):
    centroid = points.mean(axis=0)
    centered = points - centroid

    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    distances = centered @ normal

    # orient normal roughly toward +z for readability
    if normal[2] < 0:
        normal = -normal
        distances = -distances

    return centroid, normal, distances


def analyze_planarity(json_path: Path, show_plot: bool = True):
    ids, pts = load_points_from_marker_geometry(json_path)

    if len(pts) < 3:
        raise ValueError("At least 3 points are required to fit a plane.")

    centroid, normal, distances = fit_best_plane(pts)

    rms = float(np.sqrt(np.mean(distances**2)))
    max_abs = float(np.max(np.abs(distances)))

    min_dist = float(np.min(distances))
    max_dist = float(np.max(distances))

    print("\n=== HydraMarker Planarity Analysis ===")
    print(f"File: {json_path}")
    print(f"Number of points: {len(pts)}")

    print("\nBest-fit plane:")
    print(f"  centroid [mm] = {centroid}")
    print(f"  normal        = {normal}")

    print("\nPoint-to-plane distances:")
    print(f"  RMS distance [mm]     = {rms:.6f}")
    print(f"  max abs distance [mm] = {max_abs:.6f}")
    print(f"  min distance [mm]     = {min_dist:.6f}")
    print(f"  max distance [mm]     = {max_dist:.6f}")

    print("\nPer-point residuals:")
    order = np.argsort(ids)

    for idx in order:
        print(f"  id {ids[idx]:4d}: {distances[idx]: .6f} mm")

    if show_plot:
        plot_planarity(
            ids=ids,
            pts=pts,
            centroid=centroid,
            normal=normal,
            distances=distances,
        )

    return {
        "ids": ids,
        "points_mm": pts,
        "centroid_mm": centroid,
        "normal": normal,
        "distances_mm": distances,
        "rms_mm": rms,
        "max_abs_mm": max_abs,
        "min_mm": min_dist,
        "max_mm": max_dist,
    }


def plot_planarity(
    ids: np.ndarray,
    pts: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
    distances: np.ndarray,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=distances,
        s=50,
    )

    for marker_id, p in zip(ids, pts):
        ax.text(
            p[0],
            p[1],
            p[2],
            str(marker_id),
            fontsize=8,
        )

    # visualize best-fit plane
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20),
    )

    if abs(normal[2]) > 1e-8:
        zz = (
            centroid[2]
            - normal[0] * (xx - centroid[0])
            - normal[1] * (yy - centroid[1])
        ) / normal[2]

        ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.25,
            linewidth=0,
        )

    # visualize normal vector
    scale = max(
        np.ptp(pts[:, 0]),
        np.ptp(pts[:, 1]),
        max(np.ptp(pts[:, 2]), 1.0),
    ) * 0.25

    ax.quiver(
        centroid[0],
        centroid[1],
        centroid[2],
        normal[0],
        normal[1],
        normal[2],
        length=scale,
        normalize=True,
    )

    ax.set_title("HydraMarker Planarity Analysis")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    cb = fig.colorbar(scatter, ax=ax, shrink=0.7)
    cb.set_label("signed distance to best-fit plane [mm]")

    ax.set_box_aspect(
        [
            max(np.ptp(pts[:, 0]), 1.0),
            max(np.ptp(pts[:, 1]), 1.0),
            max(np.ptp(pts[:, 2]), 1.0),
        ]
    )

    plt.tight_layout()
    plt.show()


def select_json_with_qt() -> Path | None:
    app = QApplication.instance()

    if app is None:
        app = QApplication([])

    json_path_str, _ = QFileDialog.getOpenFileName(
        None,
        "Select marker_geometry_sfm.json",
        "",
        "JSON Files (*.json)",
    )

    if not json_path_str:
        return None

    return Path(json_path_str)


def main():
    json_path = select_json_with_qt()

    if json_path is None:
        print("No file selected.")
        return

    analyze_planarity(
        json_path=json_path,
        show_plot=True,
    )


if __name__ == "__main__":
    main()