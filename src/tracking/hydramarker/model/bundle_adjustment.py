"""Bundle-adjustment utilities for HydraMarker model refinement.

The functions optimize camera poses and marker-point geometry from recorded
observations, producing a more consistent model for tracking and export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import json
import os

import cv2
import numpy as np
import pyceres

from tracking.hydramarker.model.state import CameraPose, SfMState


@dataclass(slots=True)
class BundleAdjustmentResult:
    success: bool
    message: str

    num_cameras: int = 0
    num_fixed_cameras: int = 0
    num_points: int = 0
    num_observations: int = 0

    num_regularization_edges: int = 0
    regularization_target_spacing: float = float("nan")
    topology_regularization_weight: float = 0.0

    num_cell_regularization_terms: int = 0
    cell_shape_regularization_weight: float = 0.0

    num_cylinder_regularization_terms: int = 0
    cylinder_regularization_weight: float = 0.0
    cylinder_radius: float = float("nan")
    cylinder_center_x: float = float("nan")
    cylinder_center_z: float = float("nan")

    num_cylinder_grid_regularization_terms: int = 0
    cylinder_grid_regularization_weight: float = 0.0
    cylinder_grid_radius: float = float("nan")
    cylinder_grid_center_x: float = float("nan")
    cylinder_grid_center_z: float = float("nan")
    cylinder_grid_theta_step: float = float("nan")
    cylinder_grid_y_step: float = float("nan")

    initial_median_error_px: float = float("nan")
    initial_mean_error_px: float = float("nan")

    final_median_error_px: float = float("nan")
    final_mean_error_px: float = float("nan")


@dataclass(slots=True)
class FrameReprojectionStats:
    frame_id: int

    num_observations: int

    median_error_px: float
    mean_error_px: float
    max_error_px: float


@dataclass(slots=True)
class ObservationReprojectionError:
    frame_id: int
    marker_id: int
    error_px: float


@dataclass(slots=True)
class PyCeresOptions:
    linear_solver: Optional["pyceres.LinearSolverType"] = None
    loss: Optional[str] = "huber"
    loss_scale: float = 1.0
    # Iteration CAP, not a target: convergence still terminates early via
    # the Ceres tolerances. 100 truncated the large 2026-07-22 session
    # (2282 recorded frames -> ~1000 BA cameras) mid-descent.
    max_iterations: int = 300
    progress_to_stdout: bool = False
    report_full: bool = False


def _pose_to_rvec_tvec(
    pose: CameraPose,
) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(pose.R)
    tvec = pose.t.reshape(3, 1)

    return (
        rvec.astype(np.float64),
        tvec.astype(np.float64),
    )


def _pose_from_block(
    block: np.ndarray,
) -> CameraPose:
    rvec = block[:3].reshape(3, 1)
    t = block[3:].reshape(3)

    R, _ = cv2.Rodrigues(rvec)

    return CameraPose(R=R, t=t)


def _make_loss_function(
    loss: Optional[str],
    scale: float,
) -> "pyceres.LossFunction":
    if loss is None:
        return pyceres.TrivialLoss()

    normalized = loss.lower()

    if normalized in {"none", "trivial"}:
        return pyceres.TrivialLoss()

    if normalized == "huber":
        return pyceres.HuberLoss(scale)

    if normalized == "cauchy":
        return pyceres.CauchyLoss(scale)

    if normalized in {"soft_l1", "softl1"}:
        return pyceres.SoftLOneLoss(scale)

    raise ValueError(f"Unsupported loss function: {loss}")


class _ReprojectionCost(pyceres.CostFunction):

    __slots__ = (
        "_observed",
        "_K",
        "_dist",
        "_sqrt_weight",
        "_eps",
    )

    def __init__(
        self,
        observed_uv: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        sqrt_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])

        self._observed = np.asarray(
            observed_uv,
            dtype=np.float64,
        ).reshape(2)

        self._K = np.asarray(
            K,
            dtype=np.float64,
        ).reshape(3, 3)

        self._dist = np.asarray(
            dist_coeffs,
            dtype=np.float64,
        ).reshape(-1, 1)

        self._sqrt_weight = float(sqrt_weight)
        self._eps = float(eps)

    def _project(
        self,
        camera_block: np.ndarray,
        point_block: np.ndarray,
    ) -> np.ndarray:
        rvec = np.asarray(
            camera_block[:3],
            dtype=np.float64,
        ).reshape(3, 1)

        tvec = np.asarray(
            camera_block[3:],
            dtype=np.float64,
        ).reshape(3, 1)

        point = np.asarray(
            point_block,
            dtype=np.float64,
        ).reshape(1, 3)

        projected, _ = cv2.projectPoints(
            point,
            rvec,
            tvec,
            self._K,
            self._dist,
        )

        return projected.reshape(2)

    def _residual(
        self,
        camera_block: np.ndarray,
        point_block: np.ndarray,
    ) -> np.ndarray:
        return self._sqrt_weight * (
            self._project(camera_block, point_block)
            - self._observed
        )

    def _finite_difference(
        self,
        camera_block: np.ndarray,
        point_block: np.ndarray,
        *,
        wrt_camera: bool,
    ) -> np.ndarray:
        base = camera_block if wrt_camera else point_block
        dim = base.shape[0]

        jac = np.zeros(
            (2, dim),
            dtype=np.float64,
        )

        for col in range(dim):
            delta = np.zeros_like(base)
            delta[col] = self._eps

            if wrt_camera:
                res_plus = self._residual(
                    camera_block + delta,
                    point_block,
                )
                res_minus = self._residual(
                    camera_block - delta,
                    point_block,
                )
            else:
                res_plus = self._residual(
                    camera_block,
                    point_block + delta,
                )
                res_minus = self._residual(
                    camera_block,
                    point_block - delta,
                )

            jac[:, col] = (
                res_plus - res_minus
            ) / (2.0 * self._eps)

        return jac

    def Evaluate(
        self,
        parameters,
        residuals,
        jacobians,
    ) -> bool:
        camera_block = np.asarray(
            parameters[0],
            dtype=np.float64,
        )

        point_block = np.asarray(
            parameters[1],
            dtype=np.float64,
        )

        residual = self._residual(
            camera_block,
            point_block,
        )

        residuals[0] = residual[0]
        residuals[1] = residual[1]

        if jacobians is not None:

            if jacobians[0] is not None:
                jac_cam = self._finite_difference(
                    camera_block,
                    point_block,
                    wrt_camera=True,
                ).reshape(-1)

                for idx, value in enumerate(jac_cam):
                    jacobians[0][idx] = value

            if jacobians[1] is not None:
                jac_point = self._finite_difference(
                    camera_block,
                    point_block,
                    wrt_camera=False,
                ).reshape(-1)

                for idx, value in enumerate(jac_point):
                    jacobians[1][idx] = value

        return True


class _NeighborDistanceCost(pyceres.CostFunction):

    __slots__ = (
        "_target",
        "_weight",
        "_eps",
    )

    def __init__(
        self,
        target_distance: float,
        weight: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3, 3])

        self._target = float(target_distance)
        self._weight = float(weight)
        self._eps = float(eps)

    def _residual(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> float:
        d = float(np.linalg.norm(p1 - p0))

        return self._weight * (
            d - self._target
        )

    def _finite_difference(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        wrt_first: bool,
    ) -> np.ndarray:
        base = p0 if wrt_first else p1

        jac = np.zeros(
            (1, 3),
            dtype=np.float64,
        )

        for col in range(3):
            delta = np.zeros_like(base)
            delta[col] = self._eps

            if wrt_first:
                r_plus = self._residual(
                    p0 + delta,
                    p1,
                )

                r_minus = self._residual(
                    p0 - delta,
                    p1,
                )

            else:
                r_plus = self._residual(
                    p0,
                    p1 + delta,
                )

                r_minus = self._residual(
                    p0,
                    p1 - delta,
                )

            jac[0, col] = (
                r_plus - r_minus
            ) / (2.0 * self._eps)

        return jac

    def Evaluate(
        self,
        parameters,
        residuals,
        jacobians,
    ) -> bool:
        p0 = np.asarray(
            parameters[0],
            dtype=np.float64,
        ).reshape(3)

        p1 = np.asarray(
            parameters[1],
            dtype=np.float64,
        ).reshape(3)

        residuals[0] = self._residual(p0, p1)

        if jacobians is not None:

            if jacobians[0] is not None:
                jac0 = self._finite_difference(
                    p0,
                    p1,
                    wrt_first=True,
                ).reshape(-1)

                for idx, value in enumerate(jac0):
                    jacobians[0][idx] = value

            if jacobians[1] is not None:
                jac1 = self._finite_difference(
                    p0,
                    p1,
                    wrt_first=False,
                ).reshape(-1)

                for idx, value in enumerate(jac1):
                    jacobians[1][idx] = value

        return True


class _CellShapeConsistencyCost(pyceres.CostFunction):

    __slots__ = (
        "_weight",
        "_eps",
    )

    def __init__(
        self,
        weight: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.set_num_residuals(3)
        self.set_parameter_block_sizes(
            [3, 3, 3, 3]
        )

        self._weight = float(weight)
        self._eps = float(eps)

    def _residual(
        self,
        p00: np.ndarray,
        p10: np.ndarray,
        p01: np.ndarray,
        p11: np.ndarray,
    ) -> np.ndarray:
        top = float(np.linalg.norm(p10 - p00))
        bottom = float(np.linalg.norm(p11 - p01))

        left = float(np.linalg.norm(p01 - p00))
        right = float(np.linalg.norm(p11 - p10))

        diag_a = float(np.linalg.norm(p11 - p00))
        diag_b = float(np.linalg.norm(p10 - p01))

        return self._weight * np.asarray(
            [
                top - bottom,
                left - right,
                diag_a - diag_b,
            ],
            dtype=np.float64,
        )

    def _finite_difference(
        self,
        blocks: list[np.ndarray],
        *,
        block_index: int,
    ) -> np.ndarray:
        jac = np.zeros(
            (3, 3),
            dtype=np.float64,
        )

        for col in range(3):

            plus = [b.copy() for b in blocks]
            minus = [b.copy() for b in blocks]

            plus[block_index][col] += self._eps
            minus[block_index][col] -= self._eps

            r_plus = self._residual(
                plus[0],
                plus[1],
                plus[2],
                plus[3],
            )

            r_minus = self._residual(
                minus[0],
                minus[1],
                minus[2],
                minus[3],
            )

            jac[:, col] = (
                r_plus - r_minus
            ) / (2.0 * self._eps)

        return jac

    def Evaluate(
        self,
        parameters,
        residuals,
        jacobians,
    ) -> bool:

        blocks = [
            np.asarray(
                parameters[i],
                dtype=np.float64,
            ).reshape(3)
            for i in range(4)
        ]

        residual = self._residual(
            blocks[0],
            blocks[1],
            blocks[2],
            blocks[3],
        )

        for i in range(3):
            residuals[i] = residual[i]

        if jacobians is not None:

            for block_idx in range(4):

                if jacobians[block_idx] is None:
                    continue

                jac = self._finite_difference(
                    blocks,
                    block_index=block_idx,
                ).reshape(-1)

                for idx, value in enumerate(jac):
                    jacobians[block_idx][idx] = value

        return True


class _CylinderSurfaceCost(pyceres.CostFunction):

    __slots__ = (
        "_center_x",
        "_center_z",
        "_radius",
        "_weight",
        "_eps",
    )

    def __init__(
        self,
        *,
        center_x: float,
        center_z: float,
        radius: float,
        weight: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3])

        self._center_x = float(center_x)
        self._center_z = float(center_z)
        self._radius = float(radius)
        self._weight = float(weight)
        self._eps = float(eps)

    def _residual(
        self,
        point: np.ndarray,
    ) -> float:
        point = np.asarray(
            point,
            dtype=np.float64,
        ).reshape(3)

        dx = float(point[0] - self._center_x)
        dz = float(point[2] - self._center_z)
        radial_distance = float(np.hypot(dx, dz))

        return self._weight * (
            radial_distance - self._radius
        )

    def _finite_difference(
        self,
        point: np.ndarray,
    ) -> np.ndarray:
        jac = np.zeros(
            (1, 3),
            dtype=np.float64,
        )

        for col in range(3):
            delta = np.zeros_like(point)
            delta[col] = self._eps

            r_plus = self._residual(point + delta)
            r_minus = self._residual(point - delta)

            jac[0, col] = (
                r_plus - r_minus
            ) / (2.0 * self._eps)

        return jac

    def Evaluate(
        self,
        parameters,
        residuals,
        jacobians,
    ) -> bool:
        point = np.asarray(
            parameters[0],
            dtype=np.float64,
        ).reshape(3)

        residuals[0] = self._residual(point)

        if jacobians is not None and jacobians[0] is not None:
            jac = self._finite_difference(point).reshape(-1)

            for idx, value in enumerate(jac):
                jacobians[0][idx] = value

        return True


class _PointPositionPriorCost(pyceres.CostFunction):

    __slots__ = (
        "_target",
        "_weight",
    )

    def __init__(
        self,
        *,
        target: np.ndarray,
        weight: float,
    ) -> None:
        super().__init__()

        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3])

        self._target = np.asarray(
            target,
            dtype=np.float64,
        ).reshape(3)

        self._weight = float(weight)

    def Evaluate(
        self,
        parameters,
        residuals,
        jacobians,
    ) -> bool:
        point = np.asarray(
            parameters[0],
            dtype=np.float64,
        ).reshape(3)

        residual = self._weight * (
            point - self._target
        )

        for idx in range(3):
            residuals[idx] = residual[idx]

        if jacobians is not None and jacobians[0] is not None:
            for row in range(3):
                for col in range(3):
                    jacobians[0][row * 3 + col] = (
                        self._weight if row == col else 0.0
                    )

        return True


def _marker_id_to_row_col(
    marker_id: int,
    *,
    id_base: int,
    id_num_cols: int,
    origin_col: int = 0,
) -> tuple[int, int]:

    raw = int(marker_id) - int(id_base)

    if raw < 0:
        raise ValueError(
            f"marker_id {marker_id} is below id_base={id_base}."
        )

    row = raw // int(id_num_cols)
    col = raw % int(id_num_cols)

    if col < int(origin_col):
        row -= 1
        col += int(id_num_cols)

    return int(row), int(col)


def _load_id_encoding_for_topology(
    marker_json_path: Path,
) -> tuple[int, int, int, int]:

    with Path(marker_json_path).open("r", encoding="utf-8") as f:
        meta = json.load(f)

    id_encoding = meta.get("id_encoding", {})

    id_base = int(meta.get("id_base", id_encoding.get("id_base", 0)))
    id_num_cols = int(meta.get("id_num_cols", id_encoding.get("num_cols", 0)))

    origin_row = int(id_encoding.get("origin_row", 0))
    origin_col = int(id_encoding.get("origin_col", 0))

    if id_num_cols <= 0:
        raise KeyError("Could not determine id_num_cols from marker JSON.")

    return id_base, id_num_cols, origin_row, origin_col


def _load_grid_spacing_mm(
    marker_json_path: Path,
) -> float:
    with Path(marker_json_path).open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if "square_size_mm" in meta:
        spacing = float(meta["square_size_mm"])
    elif "cell_size_mm" in meta:
        spacing = float(meta["cell_size_mm"])
    elif "square_size_cm" in meta:
        spacing = 10.0 * float(meta["square_size_cm"])
    else:
        raise KeyError(
            "Could not determine marker grid spacing from marker JSON."
        )

    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Marker grid spacing must be positive.")

    return float(spacing)


def _collect_topology_edges_from_marker_blocks(
    marker_blocks: dict[int, np.ndarray],
    *,
    id_base: int,
    id_num_cols: int,
    origin_col: int,
) -> list[tuple[int, int]]:

    row_col_to_id: dict[tuple[int, int], int] = {}

    for marker_id in marker_blocks.keys():
        row, col = _marker_id_to_row_col(
            int(marker_id),
            id_base=id_base,
            id_num_cols=id_num_cols,
            origin_col=origin_col,
        )

        row_col_to_id[(row, col)] = int(marker_id)

    edges: list[tuple[int, int]] = []

    for (row, col), marker_id in row_col_to_id.items():
        right_id = row_col_to_id.get((row, col + 1))

        if right_id is not None:
            edges.append((int(marker_id), int(right_id)))

        down_id = row_col_to_id.get((row + 1, col))

        if down_id is not None:
            edges.append((int(marker_id), int(down_id)))

    return edges


def _collect_topology_cells_from_marker_blocks(
    marker_blocks: dict[int, np.ndarray],
    *,
    id_base: int,
    id_num_cols: int,
    origin_col: int,
) -> list[tuple[int, int, int, int]]:

    row_col_to_id: dict[tuple[int, int], int] = {}

    for marker_id in marker_blocks.keys():
        row, col = _marker_id_to_row_col(
            int(marker_id),
            id_base=id_base,
            id_num_cols=id_num_cols,
            origin_col=origin_col,
        )

        row_col_to_id[(row, col)] = int(marker_id)

    cells: list[tuple[int, int, int, int]] = []

    for (row, col), p00_id in row_col_to_id.items():
        p10_id = row_col_to_id.get((row, col + 1))
        p01_id = row_col_to_id.get((row + 1, col))
        p11_id = row_col_to_id.get((row + 1, col + 1))

        if p10_id is None or p01_id is None or p11_id is None:
            continue

        cells.append(
            (
                int(p00_id),
                int(p10_id),
                int(p01_id),
                int(p11_id),
            )
        )

    return cells


def _estimate_target_spacing_from_edges(
    marker_blocks: dict[int, np.ndarray],
    edges: list[tuple[int, int]],
) -> float:

    distances = []

    for id0, id1 in edges:

        p0 = np.asarray(
            marker_blocks[int(id0)],
            dtype=np.float64,
        ).reshape(3)

        p1 = np.asarray(
            marker_blocks[int(id1)],
            dtype=np.float64,
        ).reshape(3)

        d = float(np.linalg.norm(p1 - p0))

        if np.isfinite(d) and d > 1e-12:
            distances.append(d)

    if not distances:
        raise ValueError(
            "Could not estimate topology target spacing."
        )

    return float(
        np.median(
            np.asarray(
                distances,
                dtype=np.float64,
            )
        )
    )


def _fit_cylinder_xz_from_marker_blocks(
    marker_blocks: dict[int, np.ndarray],
) -> tuple[float, float, float]:
    points = np.asarray(
        [
            np.asarray(block, dtype=np.float64).reshape(3)[[0, 2]]
            for block in marker_blocks.values()
        ],
        dtype=np.float64,
    ).reshape(-1, 2)

    points = points[np.isfinite(points).all(axis=1)]

    if points.shape[0] < 3:
        raise ValueError(
            "Need at least three finite points to fit cylinder cross-section."
        )

    x = points[:, 0]
    z = points[:, 1]

    A = np.column_stack(
        [
            x,
            z,
            np.ones_like(x),
        ]
    )
    b = -(
        x * x
        + z * z
    )

    coeffs, *_ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    a, b_coeff, c = (
        float(coeffs[0]),
        float(coeffs[1]),
        float(coeffs[2]),
    )

    center_x = -0.5 * a
    center_z = -0.5 * b_coeff
    radius_sq = center_x * center_x + center_z * center_z - c

    if not np.isfinite(radius_sq) or radius_sq <= 1e-12:
        raise ValueError("Could not fit valid cylinder radius.")

    radius = float(np.sqrt(radius_sq))

    return center_x, center_z, radius


def _normalize_cylinder_center_xz(
    center_xz: Optional[Sequence[float]],
) -> tuple[float, float] | None:
    if center_xz is None:
        return None

    arr = np.asarray(
        center_xz,
        dtype=np.float64,
    ).reshape(-1)

    if arr.size < 2:
        raise ValueError(
            "cylinder_center_xz must contain at least two values."
        )

    center_x = float(arr[0])
    center_z = float(arr[1])

    if not np.isfinite(center_x) or not np.isfinite(center_z):
        raise ValueError("cylinder_center_xz contains non-finite values.")

    return center_x, center_z


@dataclass(slots=True)
class _CylinderGridTargets:
    positions: dict[int, np.ndarray]
    center_x: float
    center_z: float
    radius: float
    theta_step: float
    y_step: float


def _robust_line_intercept(
    values: np.ndarray,
    coords: np.ndarray,
    slope: float,
) -> float:
    return float(
        np.median(
            values
            - float(slope) * coords
        )
    )


def _build_cylinder_grid_targets_from_marker_blocks(
    marker_blocks: dict[int, np.ndarray],
    *,
    id_base: int,
    id_num_cols: int,
    origin_col: int,
    target_spacing: float,
    cylinder_radius: Optional[float] = None,
    cylinder_center_xz: Optional[Sequence[float]] = None,
    use_chord_spacing: bool = True,
) -> _CylinderGridTargets:
    if len(marker_blocks) < 3:
        raise ValueError(
            "Need at least three marker points for cylinder grid targets."
        )

    center_xz = _normalize_cylinder_center_xz(
        cylinder_center_xz
    )

    fitted_center_x = fitted_center_z = fitted_radius = None

    if (
        center_xz is None
        or cylinder_radius is None
        or not np.isfinite(float(cylinder_radius))
        or float(cylinder_radius) <= 1e-12
    ):
        (
            fitted_center_x,
            fitted_center_z,
            fitted_radius,
        ) = _fit_cylinder_xz_from_marker_blocks(marker_blocks)

    if center_xz is None:
        center_x = float(fitted_center_x)
        center_z = float(fitted_center_z)
    else:
        center_x = float(center_xz[0])
        center_z = float(center_xz[1])

    if (
        cylinder_radius is None
        or not np.isfinite(float(cylinder_radius))
        or float(cylinder_radius) <= 1e-12
    ):
        radius = float(fitted_radius)
    else:
        radius = float(cylinder_radius)

    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("Cylinder grid radius must be positive.")

    rows = []
    cols = []
    angles = []
    y_values = []
    by_col: dict[int, list[float]] = {}

    for marker_id, block in marker_blocks.items():
        point = np.asarray(
            block,
            dtype=np.float64,
        ).reshape(3)

        if not np.isfinite(point).all():
            continue

        row, col = _marker_id_to_row_col(
            int(marker_id),
            id_base=id_base,
            id_num_cols=id_num_cols,
            origin_col=origin_col,
        )

        theta = float(
            np.arctan2(
                point[2] - center_z,
                point[0] - center_x,
            )
        )

        rows.append(float(row))
        cols.append(float(col))
        angles.append(theta)
        y_values.append(float(point[1]))
        by_col.setdefault(int(col), []).append(theta)

    if len(rows) < 3 or len(by_col) < 2:
        raise ValueError(
            "Need at least two populated columns for cylinder grid targets."
        )

    unique_cols = np.asarray(
        sorted(by_col),
        dtype=np.float64,
    )

    col_angles = np.asarray(
        [
            float(
                np.median(
                    np.asarray(
                        by_col[int(col)],
                        dtype=np.float64,
                    )
                )
            )
            for col in unique_cols
        ],
        dtype=np.float64,
    )

    unwrapped_col_angles = np.unwrap(col_angles)

    if unique_cols.size >= 2:
        fitted_theta_step, _ = np.polyfit(
            unique_cols,
            unwrapped_col_angles,
            deg=1,
        )
        fitted_theta_step = float(fitted_theta_step)
    else:
        fitted_theta_step = 0.0

    if use_chord_spacing and 0.0 < float(target_spacing) < 2.0 * radius:
        theta_step_abs = float(
            2.0
            * np.arcsin(
                min(
                    1.0,
                    float(target_spacing) / (2.0 * radius),
                )
            )
        )
    elif (not use_chord_spacing) and float(target_spacing) > 0.0:
        theta_step_abs = float(target_spacing) / radius
    else:
        theta_step_abs = abs(float(fitted_theta_step))

    if theta_step_abs <= 1e-12:
        raise ValueError(
            "Could not estimate a valid cylinder grid column step."
        )

    sign = 1.0 if fitted_theta_step >= 0.0 else -1.0
    theta_step = sign * theta_step_abs

    theta0 = _robust_line_intercept(
        unwrapped_col_angles,
        unique_cols,
        theta_step,
    )

    rows_arr = np.asarray(
        rows,
        dtype=np.float64,
    )

    y_arr = np.asarray(
        y_values,
        dtype=np.float64,
    )

    y_step = float(target_spacing)
    y0 = _robust_line_intercept(
        y_arr,
        rows_arr,
        y_step,
    )

    positions: dict[int, np.ndarray] = {}

    for marker_id, block in marker_blocks.items():
        point = np.asarray(
            block,
            dtype=np.float64,
        ).reshape(3)

        if not np.isfinite(point).all():
            continue

        row, col = _marker_id_to_row_col(
            int(marker_id),
            id_base=id_base,
            id_num_cols=id_num_cols,
            origin_col=origin_col,
        )

        theta = theta0 + theta_step * float(col)

        positions[int(marker_id)] = np.asarray(
            [
                center_x + radius * np.cos(theta),
                y0 + y_step * float(row),
                center_z + radius * np.sin(theta),
            ],
            dtype=np.float64,
        )

    if not positions:
        raise ValueError("Cylinder grid target map is empty.")

    return _CylinderGridTargets(
        positions=positions,
        center_x=float(center_x),
        center_z=float(center_z),
        radius=float(radius),
        theta_step=float(theta_step),
        y_step=float(y_step),
    )


def _build_frame_list(
    state: SfMState,
    frame_ids: Optional[Sequence[int]],
) -> list[int]:

    posed_ids = state.posed_frame_ids()

    if not posed_ids:
        raise ValueError(
            "No posed frames available for bundle adjustment."
        )

    anchor_id = posed_ids[0]

    if frame_ids is None:
        selected = list(posed_ids)

    else:
        selected = sorted(
            {
                int(frame_id)
                for frame_id in frame_ids
                if state.has_pose(int(frame_id))
            }
        )

        if anchor_id not in selected:
            selected.insert(0, anchor_id)

    selected = [
        frame_id
        for frame_id in selected
        if state.has_pose(frame_id)
    ]

    if len(selected) < 2:
        raise ValueError(
            "Need at least two posed frames for bundle adjustment."
        )

    return selected


def _collect_marker_ids(
    state: SfMState,
    frame_ids: Sequence[int],
) -> list[int]:

    marker_ids: set[int] = set()

    for frame_id in frame_ids:
        frame = state.get_frame(int(frame_id))

        for marker_id in frame.observations:
            if marker_id in state.marker_positions:
                marker_ids.add(int(marker_id))

    if not marker_ids:
        raise ValueError(
            "No observed reconstructed markers for bundle adjustment."
        )

    return sorted(marker_ids)


def _initialize_camera_blocks(
    state: SfMState,
    frame_ids: Sequence[int],
) -> dict[int, np.ndarray]:

    blocks: dict[int, np.ndarray] = {}

    for frame_id in frame_ids:
        pose = state.poses[int(frame_id)]
        rvec, tvec = _pose_to_rvec_tvec(pose)

        blocks[int(frame_id)] = np.ascontiguousarray(
            np.concatenate(
                [
                    rvec.reshape(3),
                    tvec.reshape(3),
                ],
                axis=0,
            ),
            dtype=np.float64,
        )

    return blocks


def _initialize_marker_blocks(
    state: SfMState,
    marker_ids: Sequence[int],
) -> dict[int, np.ndarray]:

    blocks: dict[int, np.ndarray] = {}

    for marker_id in marker_ids:
        blocks[int(marker_id)] = np.ascontiguousarray(
            state.marker_positions[int(marker_id)].reshape(3),
            dtype=np.float64,
        )

    return blocks


def _normalize_ignored_observations(
    ignored_observations: Optional[set[tuple[int, int]]],
) -> set[tuple[int, int]]:

    if ignored_observations is None:
        return set()

    return {
        (
            int(frame_id),
            int(marker_id),
        )
        for frame_id, marker_id in ignored_observations
    }


def compute_observation_reprojection_errors(
    state: SfMState,
    frame_ids: Optional[Sequence[int]] = None,
    *,
    ignored_observations: Optional[set[tuple[int, int]]] = None,
) -> list[ObservationReprojectionError]:

    if frame_ids is None:
        frame_ids = state.posed_frame_ids()

    ignored = _normalize_ignored_observations(
        ignored_observations
    )

    result: list[ObservationReprojectionError] = []

    for frame_id in frame_ids:
        frame_id = int(frame_id)

        if not state.has_pose(frame_id):
            continue

        pose = state.poses[frame_id]
        frame = state.get_frame(frame_id)

        rvec, tvec = _pose_to_rvec_tvec(pose)

        for marker_id, obs in frame.observations.items():
            marker_id = int(marker_id)

            if (frame_id, marker_id) in ignored:
                continue

            point = state.marker_positions.get(marker_id)
            if point is None:
                continue

            projected, _ = cv2.projectPoints(
                point.reshape(1, 3),
                rvec,
                tvec,
                state.calibration.K,
                state.calibration.dist_coeffs,
            )

            uv_hat = projected.reshape(2)
            uv_obs = np.asarray(
                obs.uv,
                dtype=np.float64,
            ).reshape(2)

            error_px = float(
                np.linalg.norm(uv_hat - uv_obs)
            )

            result.append(
                ObservationReprojectionError(
                    frame_id=frame_id,
                    marker_id=marker_id,
                    error_px=error_px,
                )
            )

    result.sort(
        key=lambda e: e.error_px,
        reverse=True,
    )

    return result


def select_observation_outliers(
    errors: Sequence[ObservationReprojectionError],
    *,
    absolute_threshold_px: float = 2.0,
    mad_sigma: float = 3.5,
    max_fraction: float = 0.03,
    min_error_px: float = 1.0,
    max_per_marker_fraction: Optional[float] = None,
    max_per_marker_count: Optional[int] = None,
) -> set[tuple[int, int]]:

    if not errors:
        return set()

    values = np.asarray(
        [e.error_px for e in errors],
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if values.size == 0:
        return set()

    median = float(np.median(values))
    mad = float(
        np.median(
            np.abs(values - median)
        )
    )

    robust_sigma = 1.4826 * mad

    if robust_sigma <= 1e-12:
        adaptive_threshold = absolute_threshold_px
    else:
        adaptive_threshold = median + mad_sigma * robust_sigma

    threshold = max(
        float(min_error_px),
        min(
            float(absolute_threshold_px),
            float(adaptive_threshold),
        ),
    )

    candidates = [
        e
        for e in errors
        if np.isfinite(e.error_px)
        and e.error_px > threshold
    ]

    if max_fraction is not None and max_fraction > 0.0:
        max_count = int(
            np.ceil(
                float(max_fraction) * len(errors)
            )
        )

        max_count = max(1, max_count)
        candidates = candidates[:max_count]

    if (
        max_per_marker_fraction is not None
        and max_per_marker_fraction > 0.0
    ) or (
        max_per_marker_count is not None
        and max_per_marker_count > 0
    ):
        total_by_marker: dict[int, int] = {}
        kept_by_marker: dict[int, int] = {}
        balanced_candidates = []

        for e in errors:
            total_by_marker[int(e.marker_id)] = (
                total_by_marker.get(int(e.marker_id), 0)
                + 1
            )

        for e in candidates:
            marker_id = int(e.marker_id)
            marker_total = max(1, total_by_marker.get(marker_id, 1))
            marker_limit = marker_total

            if (
                max_per_marker_fraction is not None
                and max_per_marker_fraction > 0.0
            ):
                marker_limit = min(
                    marker_limit,
                    max(
                        1,
                        int(
                            np.ceil(
                                float(max_per_marker_fraction)
                                * marker_total
                            )
                        ),
                    ),
                )

            if (
                max_per_marker_count is not None
                and max_per_marker_count > 0
            ):
                marker_limit = min(
                    marker_limit,
                    int(max_per_marker_count),
                )

            already_kept = kept_by_marker.get(marker_id, 0)
            if already_kept >= marker_limit:
                continue

            balanced_candidates.append(e)
            kept_by_marker[marker_id] = already_kept + 1

        candidates = balanced_candidates

    ignored = {
        (
            int(e.frame_id),
            int(e.marker_id),
        )
        for e in candidates
    }


    return ignored


def _soft_weight_from_error(
    error_px: float,
    *,
    good_px: float,
    bad_px: float,
    min_weight: float,
) -> float:

    error_px = float(error_px)

    if not np.isfinite(error_px):
        return float(min_weight)

    if error_px <= float(good_px):
        return 1.0

    if error_px >= float(bad_px):
        return float(min_weight)

    t = (
        float(error_px) - float(good_px)
    ) / (
        float(bad_px) - float(good_px)
    )

    return float(
        (1.0 - t) * 1.0
        + t * float(min_weight)
    )


def compute_adaptive_observation_weights(
    errors: Sequence[ObservationReprojectionError],
    *,
    marker_good_px: float = 0.35,
    marker_bad_px: float = 0.80,
    frame_good_px: float = 0.35,
    frame_bad_px: float = 0.90,
    min_marker_weight: float = 0.35,
    min_frame_weight: float = 0.25,
    min_observation_weight: float = 0.10,
    print_summary: bool = False,
) -> dict[tuple[int, int], float]:

    if not errors:
        return {}

    marker_values: dict[int, list[float]] = {}
    frame_values: dict[int, list[float]] = {}

    for e in errors:
        if not np.isfinite(e.error_px):
            continue

        marker_values.setdefault(
            int(e.marker_id),
            [],
        ).append(float(e.error_px))

        frame_values.setdefault(
            int(e.frame_id),
            [],
        ).append(float(e.error_px))

    marker_weights: dict[int, float] = {}

    for marker_id, values in marker_values.items():
        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        mean_error = float(np.mean(arr))

        marker_weights[int(marker_id)] = _soft_weight_from_error(
            mean_error,
            good_px=marker_good_px,
            bad_px=marker_bad_px,
            min_weight=min_marker_weight,
        )

    frame_weights: dict[int, float] = {}

    for frame_id, values in frame_values.items():
        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        mean_error = float(np.mean(arr))

        frame_weights[int(frame_id)] = _soft_weight_from_error(
            mean_error,
            good_px=frame_good_px,
            bad_px=frame_bad_px,
            min_weight=min_frame_weight,
        )

    observation_weights: dict[tuple[int, int], float] = {}

    for e in errors:
        frame_id = int(e.frame_id)
        marker_id = int(e.marker_id)

        marker_weight = marker_weights.get(
            marker_id,
            1.0,
        )

        frame_weight = frame_weights.get(
            frame_id,
            1.0,
        )

        weight = float(marker_weight * frame_weight)

        weight = max(
            float(min_observation_weight),
            min(
                1.0,
                weight,
            ),
        )

        observation_weights[
            (
                frame_id,
                marker_id,
            )
        ] = weight


    return observation_weights


def _normalize_observation_weights(
    observation_weights: Optional[dict[tuple[int, int], float]],
) -> dict[tuple[int, int], float]:

    if observation_weights is None:
        return {}

    normalized: dict[tuple[int, int], float] = {}

    for key, value in observation_weights.items():
        frame_id, marker_id = key

        weight = float(value)

        if not np.isfinite(weight):
            continue

        normalized[
            (
                int(frame_id),
                int(marker_id),
            )
        ] = max(
            1e-6,
            weight,
        )

    return normalized


def compute_reprojection_errors(
    state: SfMState,
    frame_ids: Optional[Sequence[int]] = None,
    *,
    ignored_observations: Optional[set[tuple[int, int]]] = None,
) -> dict[int, np.ndarray]:

    if frame_ids is None:
        frame_ids = state.posed_frame_ids()

    ignored = _normalize_ignored_observations(
        ignored_observations
    )

    errors: dict[int, np.ndarray] = {}

    for frame_id in frame_ids:
        frame_id = int(frame_id)

        if not state.has_pose(frame_id):
            continue

        pose = state.poses[frame_id]
        frame = state.get_frame(frame_id)

        rvec, tvec = _pose_to_rvec_tvec(pose)

        frame_errors = []

        for marker_id, obs in frame.observations.items():
            marker_id = int(marker_id)

            if (frame_id, marker_id) in ignored:
                continue

            point = state.marker_positions.get(marker_id)
            if point is None:
                continue

            projected, _ = cv2.projectPoints(
                point.reshape(1, 3),
                rvec,
                tvec,
                state.calibration.K,
                state.calibration.dist_coeffs,
            )

            uv_hat = projected.reshape(2)
            uv_obs = np.asarray(
                obs.uv,
                dtype=np.float64,
            ).reshape(2)

            frame_errors.append(
                float(np.linalg.norm(uv_hat - uv_obs))
            )

        errors[frame_id] = np.asarray(
            frame_errors,
            dtype=np.float64,
        )

    return errors


def compute_median_mean_reprojection_error(
    state: SfMState,
    frame_ids: Optional[Sequence[int]] = None,
    *,
    ignored_observations: Optional[set[tuple[int, int]]] = None,
) -> tuple[float, float]:

    per_frame = compute_reprojection_errors(
        state,
        frame_ids=frame_ids,
        ignored_observations=ignored_observations,
    )

    all_errors = (
        np.concatenate(
            [
                errors
                for errors in per_frame.values()
                if errors.size > 0
            ],
            axis=0,
        )
        if per_frame
        else np.empty(0, dtype=np.float64)
    )

    if all_errors.size == 0:
        return (
            float("nan"),
            float("nan"),
        )

    return (
        float(np.median(all_errors)),
        float(np.mean(all_errors)),
    )


def compute_frame_reprojection_statistics(
    state: SfMState,
    frame_ids: Optional[Sequence[int]] = None,
    *,
    ignored_observations: Optional[set[tuple[int, int]]] = None,
) -> list[FrameReprojectionStats]:

    per_frame = compute_reprojection_errors(
        state,
        frame_ids=frame_ids,
        ignored_observations=ignored_observations,
    )

    stats: list[FrameReprojectionStats] = []

    for frame_id, errors in per_frame.items():
        errors = np.asarray(
            errors,
            dtype=np.float64,
        ).reshape(-1)

        if errors.size == 0:
            stats.append(
                FrameReprojectionStats(
                    frame_id=int(frame_id),
                    num_observations=0,
                    median_error_px=float("nan"),
                    mean_error_px=float("nan"),
                    max_error_px=float("nan"),
                )
            )

            continue

        stats.append(
            FrameReprojectionStats(
                frame_id=int(frame_id),
                num_observations=int(errors.size),
                median_error_px=float(np.median(errors)),
                mean_error_px=float(np.mean(errors)),
                max_error_px=float(np.max(errors)),
            )
        )

    stats.sort(
        key=lambda s: (
            np.nan_to_num(
                s.median_error_px,
                nan=np.inf,
            ),
            -s.num_observations,
        )
    )

    return stats


def print_frame_reprojection_statistics(
    stats: Sequence[FrameReprojectionStats],
    *,
    max_rows: Optional[int] = None,
) -> None:
    return


def select_good_frames_for_bundle_adjustment(
    state: SfMState,
    *,
    frame_ids: Optional[Sequence[int]] = None,
    min_observations: int = 15,
    max_median_error_px: float = 1.5,
    keep_anchor_frame: bool = True,
) -> list[int]:

    stats = compute_frame_reprojection_statistics(
        state,
        frame_ids=frame_ids,
    )

    posed_ids = state.posed_frame_ids()

    if not posed_ids:
        return []

    anchor_id = int(posed_ids[0])

    selected: list[int] = []

    for s in stats:
        keep = True

        if s.num_observations < int(min_observations):
            keep = False

        if (
            np.isfinite(s.median_error_px)
            and s.median_error_px
            > float(max_median_error_px)
        ):
            keep = False

        if keep:
            selected.append(int(s.frame_id))

    if keep_anchor_frame and anchor_id not in selected:
        selected.insert(0, anchor_id)

    selected = sorted(set(selected))


    return selected


def _update_state_from_blocks(
    state: SfMState,
    camera_blocks: dict[int, np.ndarray],
    marker_blocks: dict[int, np.ndarray],
    *,
    optimized_frame_ids: set[int],
) -> None:

    for frame_id in optimized_frame_ids:
        state.add_pose(
            int(frame_id),
            _pose_from_block(
                camera_blocks[int(frame_id)]
            ),
        )

    for marker_id, block in marker_blocks.items():
        state.add_marker_position(
            int(marker_id),
            np.asarray(
                block,
                dtype=np.float64,
            ).reshape(3),
        )


def run_bundle_adjustment(
    state: SfMState,
    *,
    frame_ids: Optional[Sequence[int]] = None,
    options: Optional[PyCeresOptions] = None,
    update_state: bool = True,
    marker_json_path: Optional[Path] = None,
    topology_regularization_weight: float = 0.0,
    cell_shape_regularization_weight: float = 0.0,
    cylinder_regularization_weight: float = 0.0,
    cylinder_radius: Optional[float] = None,
    cylinder_center_xz: Optional[Sequence[float]] = None,
    cylinder_grid_regularization_weight: float = 0.0,
    cylinder_grid_radius: Optional[float] = None,
    cylinder_grid_center_xz: Optional[Sequence[float]] = None,
    cylinder_grid_use_chord_spacing: bool = True,
    ignored_observations: Optional[set[tuple[int, int]]] = None,
    observation_weights: Optional[dict[tuple[int, int], float]] = None,
) -> BundleAdjustmentResult:

    num_regularization_edges = 0
    regularization_target_spacing = float("nan")
    num_cell_regularization_terms = 0
    num_cylinder_regularization_terms = 0
    cylinder_radius_used = float("nan")
    cylinder_center_x_used = float("nan")
    cylinder_center_z_used = float("nan")
    num_cylinder_grid_regularization_terms = 0
    cylinder_grid_radius_used = float("nan")
    cylinder_grid_center_x_used = float("nan")
    cylinder_grid_center_z_used = float("nan")
    cylinder_grid_theta_step_used = float("nan")
    cylinder_grid_y_step_used = float("nan")

    ignored = _normalize_ignored_observations(
        ignored_observations
    )
    
    weights = _normalize_observation_weights(
        observation_weights
    )

    try:
        opts = options or PyCeresOptions()

        selected_frame_ids = _build_frame_list(
            state,
            frame_ids,
        )

        anchor_id = int(selected_frame_ids[0])

        optimized_frame_ids = {
            int(frame_id)
            for frame_id in selected_frame_ids
            if int(frame_id) != anchor_id
        }

        marker_ids = _collect_marker_ids(
            state,
            selected_frame_ids,
        )

        median_before, mean_before = compute_median_mean_reprojection_error(
            state,
            frame_ids=selected_frame_ids,
            ignored_observations=ignored,
        )

        camera_blocks = _initialize_camera_blocks(
            state,
            selected_frame_ids,
        )

        marker_blocks = _initialize_marker_blocks(
            state,
            marker_ids,
        )

        problem = pyceres.Problem()

        for frame_id, block in camera_blocks.items():
            problem.add_parameter_block(
                block,
                block.size,
            )

            if int(frame_id) == anchor_id:
                problem.set_parameter_block_constant(block)

        for block in marker_blocks.values():
            problem.add_parameter_block(
                block,
                block.size,
            )

        num_observations = 0

        loss_function = _make_loss_function(
            opts.loss,
            opts.loss_scale,
        )

        for frame_id in selected_frame_ids:
            frame_id = int(frame_id)
            frame = state.get_frame(frame_id)
            cam_block = camera_blocks[frame_id]

            for marker_id, obs in frame.observations.items():
                marker_id = int(marker_id)

                if (frame_id, marker_id) in ignored:
                    continue

                if marker_id not in marker_blocks:
                    continue

                point_block = marker_blocks[marker_id]

                weight = weights.get(
                    (
                        int(frame_id),
                        int(marker_id),
                    ),
                    1.0,
                )

                sqrt_weight = float(
                    np.sqrt(
                        max(
                            1e-6,
                            float(weight),
                        )
                    )
                )

                cost_function = _ReprojectionCost(
                    np.asarray(
                        obs.uv,
                        dtype=np.float64,
                    ),
                    state.calibration.K,
                    state.calibration.dist_coeffs,
                    sqrt_weight=sqrt_weight,
                )

                problem.add_residual_block(
                    cost_function,
                    loss_function,
                    [
                        cam_block,
                        point_block,
                    ],
                )

                num_observations += 1

        if marker_json_path is not None:

            (
                id_base,
                id_num_cols,
                origin_row,
                origin_col,
            ) = _load_id_encoding_for_topology(Path(marker_json_path))

            if float(topology_regularization_weight) > 0.0:

                topology_edges = _collect_topology_edges_from_marker_blocks(
                    marker_blocks,
                    id_base=id_base,
                    id_num_cols=id_num_cols,
                    origin_col=origin_col,
                )

                if topology_edges:
                    regularization_target_spacing = (
                        _estimate_target_spacing_from_edges(
                            marker_blocks,
                            topology_edges,
                        )
                    )

                    for id0, id1 in topology_edges:
                        if (
                            id0 not in marker_blocks
                            or id1 not in marker_blocks
                        ):
                            continue

                        cost_function = _NeighborDistanceCost(
                            target_distance=regularization_target_spacing,
                            weight=float(
                                topology_regularization_weight
                            ),
                        )

                        problem.add_residual_block(
                            cost_function,
                            pyceres.TrivialLoss(),
                            [
                                marker_blocks[int(id0)],
                                marker_blocks[int(id1)],
                            ],
                        )

                        num_regularization_edges += 1

            if float(cell_shape_regularization_weight) > 0.0:

                topology_cells = _collect_topology_cells_from_marker_blocks(
                    marker_blocks,
                    id_base=id_base,
                    id_num_cols=id_num_cols,
                    origin_col=origin_col,
                )

                for (
                    p00_id,
                    p10_id,
                    p01_id,
                    p11_id,
                ) in topology_cells:

                    if (
                        p00_id not in marker_blocks
                        or p10_id not in marker_blocks
                        or p01_id not in marker_blocks
                        or p11_id not in marker_blocks
                    ):
                        continue

                    cost_function = _CellShapeConsistencyCost(
                        weight=float(
                            cell_shape_regularization_weight
                        ),
                    )

                    problem.add_residual_block(
                        cost_function,
                        pyceres.TrivialLoss(),
                        [
                            marker_blocks[int(p00_id)],
                            marker_blocks[int(p10_id)],
                            marker_blocks[int(p01_id)],
                            marker_blocks[int(p11_id)],
                        ],
                    )

                    num_cell_regularization_terms += 1

            if float(cylinder_grid_regularization_weight) > 0.0:
                target_spacing = _load_grid_spacing_mm(
                    Path(marker_json_path)
                )

                grid_radius = (
                    cylinder_grid_radius
                    if cylinder_grid_radius is not None
                    else cylinder_radius
                )

                grid_center_xz = (
                    cylinder_grid_center_xz
                    if cylinder_grid_center_xz is not None
                    else cylinder_center_xz
                )

                grid_targets = _build_cylinder_grid_targets_from_marker_blocks(
                    marker_blocks,
                    id_base=id_base,
                    id_num_cols=id_num_cols,
                    origin_col=origin_col,
                    target_spacing=target_spacing,
                    cylinder_radius=grid_radius,
                    cylinder_center_xz=grid_center_xz,
                    use_chord_spacing=bool(cylinder_grid_use_chord_spacing),
                )

                cylinder_grid_radius_used = grid_targets.radius
                cylinder_grid_center_x_used = grid_targets.center_x
                cylinder_grid_center_z_used = grid_targets.center_z
                cylinder_grid_theta_step_used = grid_targets.theta_step
                cylinder_grid_y_step_used = grid_targets.y_step

                for marker_id, target in grid_targets.positions.items():
                    if int(marker_id) not in marker_blocks:
                        continue

                    cost_function = _PointPositionPriorCost(
                        target=target,
                        weight=float(cylinder_grid_regularization_weight),
                    )

                    problem.add_residual_block(
                        cost_function,
                        pyceres.TrivialLoss(),
                        [
                            marker_blocks[int(marker_id)],
                        ],
                    )

                    num_cylinder_grid_regularization_terms += 1

        if float(cylinder_regularization_weight) > 0.0:
            center_xz = _normalize_cylinder_center_xz(
                cylinder_center_xz
            )

            if (
                center_xz is None
                or cylinder_radius is None
                or not np.isfinite(float(cylinder_radius))
                or float(cylinder_radius) <= 1e-12
            ):
                (
                    fitted_center_x,
                    fitted_center_z,
                    fitted_radius,
                ) = _fit_cylinder_xz_from_marker_blocks(marker_blocks)

                if center_xz is None:
                    center_xz = (
                        fitted_center_x,
                        fitted_center_z,
                    )

                if (
                    cylinder_radius is None
                    or not np.isfinite(float(cylinder_radius))
                    or float(cylinder_radius) <= 1e-12
                ):
                    cylinder_radius = fitted_radius

            cylinder_center_x_used = float(center_xz[0])
            cylinder_center_z_used = float(center_xz[1])
            cylinder_radius_used = float(cylinder_radius)

            for block in marker_blocks.values():
                cost_function = _CylinderSurfaceCost(
                    center_x=cylinder_center_x_used,
                    center_z=cylinder_center_z_used,
                    radius=cylinder_radius_used,
                    weight=float(cylinder_regularization_weight),
                )

                problem.add_residual_block(
                    cost_function,
                    pyceres.TrivialLoss(),
                    [
                        block,
                    ],
                )

                num_cylinder_regularization_terms += 1

        if num_observations == 0:
            raise ValueError(
                "No valid observations added to Ceres problem."
            )

        solver_options = pyceres.SolverOptions()
        solver_options.max_num_iterations = opts.max_iterations
        # (SPARSE_SCHUR was tried for the ~1000-camera session and was
        # SLOWER on this pyceres build — no fast sparse backend; identical
        # numbers, so behaviour-neutral either way. Dense stays.)
        solver_options.linear_solver_type = (
            opts.linear_solver
            or pyceres.LinearSolverType.DENSE_SCHUR
        )
        solver_options.num_threads = max(1, (os.cpu_count() or 1) - 1)
        solver_options.minimizer_progress_to_stdout = (
            bool(opts.progress_to_stdout)
            or bool(opts.report_full)
        )

        summary = pyceres.SolverSummary()

        pyceres.solve(
            solver_options,
            problem,
            summary,
        )

        if opts.report_full:
            print(summary.FullReport())

        if not summary.IsSolutionUsable():
            return BundleAdjustmentResult(
                success=False,
                message=f"Ceres failed: {summary.message}",
                num_cameras=len(selected_frame_ids),
                num_fixed_cameras=1,
                num_points=len(marker_ids),
                num_observations=num_observations,
                num_regularization_edges=num_regularization_edges,
                regularization_target_spacing=regularization_target_spacing,
                topology_regularization_weight=float(
                    topology_regularization_weight
                ),
                num_cell_regularization_terms=num_cell_regularization_terms,
                cell_shape_regularization_weight=float(
                    cell_shape_regularization_weight
                ),
                num_cylinder_regularization_terms=(
                    num_cylinder_regularization_terms
                ),
                cylinder_regularization_weight=float(
                    cylinder_regularization_weight
                ),
                cylinder_radius=cylinder_radius_used,
                cylinder_center_x=cylinder_center_x_used,
                cylinder_center_z=cylinder_center_z_used,
                num_cylinder_grid_regularization_terms=(
                    num_cylinder_grid_regularization_terms
                ),
                cylinder_grid_regularization_weight=float(
                    cylinder_grid_regularization_weight
                ),
                cylinder_grid_radius=cylinder_grid_radius_used,
                cylinder_grid_center_x=cylinder_grid_center_x_used,
                cylinder_grid_center_z=cylinder_grid_center_z_used,
                cylinder_grid_theta_step=cylinder_grid_theta_step_used,
                cylinder_grid_y_step=cylinder_grid_y_step_used,
                initial_median_error_px=median_before,
                initial_mean_error_px=mean_before,
            )

        if update_state:
            _update_state_from_blocks(
                state,
                camera_blocks,
                marker_blocks,
                optimized_frame_ids=optimized_frame_ids,
            )

        median_after, mean_after = compute_median_mean_reprojection_error(
            state,
            frame_ids=selected_frame_ids,
            ignored_observations=ignored,
        )

        return BundleAdjustmentResult(
            success=True,
            message=summary.message,
            num_cameras=len(selected_frame_ids),
            num_fixed_cameras=1,
            num_points=len(marker_ids),
            num_observations=num_observations,
            num_regularization_edges=num_regularization_edges,
            regularization_target_spacing=regularization_target_spacing,
            topology_regularization_weight=float(
                topology_regularization_weight
            ),
            num_cell_regularization_terms=num_cell_regularization_terms,
            cell_shape_regularization_weight=float(
                cell_shape_regularization_weight
            ),
            num_cylinder_regularization_terms=(
                num_cylinder_regularization_terms
            ),
            cylinder_regularization_weight=float(
                cylinder_regularization_weight
            ),
            cylinder_radius=cylinder_radius_used,
            cylinder_center_x=cylinder_center_x_used,
            cylinder_center_z=cylinder_center_z_used,
            num_cylinder_grid_regularization_terms=(
                num_cylinder_grid_regularization_terms
            ),
            cylinder_grid_regularization_weight=float(
                cylinder_grid_regularization_weight
            ),
            cylinder_grid_radius=cylinder_grid_radius_used,
            cylinder_grid_center_x=cylinder_grid_center_x_used,
            cylinder_grid_center_z=cylinder_grid_center_z_used,
            cylinder_grid_theta_step=cylinder_grid_theta_step_used,
            cylinder_grid_y_step=cylinder_grid_y_step_used,
            initial_median_error_px=median_before,
            initial_mean_error_px=mean_before,
            final_median_error_px=median_after,
            final_mean_error_px=mean_after,
        )

    except Exception as exc:
        return BundleAdjustmentResult(
            success=False,
            message=f"Bundle adjustment failed: {exc}",
            num_regularization_edges=num_regularization_edges,
            regularization_target_spacing=regularization_target_spacing,
            topology_regularization_weight=float(
                topology_regularization_weight
            ),
            num_cell_regularization_terms=num_cell_regularization_terms,
            cell_shape_regularization_weight=float(
                cell_shape_regularization_weight
            ),
            num_cylinder_regularization_terms=(
                num_cylinder_regularization_terms
            ),
            cylinder_regularization_weight=float(
                cylinder_regularization_weight
            ),
            cylinder_radius=cylinder_radius_used,
            cylinder_center_x=cylinder_center_x_used,
            cylinder_center_z=cylinder_center_z_used,
            num_cylinder_grid_regularization_terms=(
                num_cylinder_grid_regularization_terms
            ),
            cylinder_grid_regularization_weight=float(
                cylinder_grid_regularization_weight
            ),
            cylinder_grid_radius=cylinder_grid_radius_used,
            cylinder_grid_center_x=cylinder_grid_center_x_used,
            cylinder_grid_center_z=cylinder_grid_center_z_used,
            cylinder_grid_theta_step=cylinder_grid_theta_step_used,
            cylinder_grid_y_step=cylinder_grid_y_step_used,
        )


def print_bundle_adjustment_summary(
    result: BundleAdjustmentResult,
) -> None:

    print()
    print("=" * 70)
    print("HYDRAMARKER BUNDLE ADJUSTMENT")
    print("=" * 70)
    print(f"success                : {result.success}")
    print(f"message                : {result.message}")
    print(f"cameras                : {result.num_cameras}")
    print(f"fixed cameras          : {result.num_fixed_cameras}")
    print(f"points                 : {result.num_points}")
    print(f"observations           : {result.num_observations}")

    print()
    print("topology regularization")
    print(f"  neighbor edges       : {result.num_regularization_edges}")
    print(f"  target spacing       : {result.regularization_target_spacing:.6f}")
    print(f"  weight               : {result.topology_regularization_weight:.6f}")

    print()
    print("cell shape regularization")
    print(f"  cells                : {result.num_cell_regularization_terms}")
    print(f"  weight               : {result.cell_shape_regularization_weight:.6f}")

    print()
    print("cylinder regularization")
    print(f"  points               : {result.num_cylinder_regularization_terms}")
    print(f"  weight               : {result.cylinder_regularization_weight:.6f}")
    print(f"  radius               : {result.cylinder_radius:.6f}")
    print(
        "  center x/z           : "
        f"{result.cylinder_center_x:.6f}, {result.cylinder_center_z:.6f}"
    )

    print()
    print("cylinder grid regularization")
    print(f"  points               : {result.num_cylinder_grid_regularization_terms}")
    print(f"  weight               : {result.cylinder_grid_regularization_weight:.6f}")
    print(f"  radius               : {result.cylinder_grid_radius:.6f}")
    print(
        "  center x/z           : "
        f"{result.cylinder_grid_center_x:.6f}, "
        f"{result.cylinder_grid_center_z:.6f}"
    )
    print(f"  theta step           : {result.cylinder_grid_theta_step:.6f}")
    print(f"  y step               : {result.cylinder_grid_y_step:.6f}")

    print()
    print("reprojection error [px]")
    print(f"  initial median       : {result.initial_median_error_px:.4f}")
    print(f"  initial mean         : {result.initial_mean_error_px:.4f}")
    print(f"  final median         : {result.final_median_error_px:.4f}")
    print(f"  final mean           : {result.final_mean_error_px:.4f}")

    print("=" * 70)
    print()
