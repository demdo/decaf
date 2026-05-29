from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import json

import cv2
import numpy as np
import pyceres

from overlay.tracking.hydramarker.model.state import CameraPose, SfMState


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
    max_iterations: int = 100
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
    ignored_observations: Optional[set[tuple[int, int]]] = None,
    observation_weights: Optional[dict[tuple[int, int], float]] = None,
) -> BundleAdjustmentResult:

    num_regularization_edges = 0
    regularization_target_spacing = float("nan")
    num_cell_regularization_terms = 0

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

        if num_observations == 0:
            raise ValueError(
                "No valid observations added to Ceres problem."
            )

        solver_options = pyceres.SolverOptions()
        solver_options.max_num_iterations = opts.max_iterations
        solver_options.linear_solver_type = (
            opts.linear_solver
            or pyceres.LinearSolverType.DENSE_SCHUR
        )
        solver_options.minimizer_progress_to_stdout = opts.report_full

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
    print("reprojection error [px]")
    print(f"  initial median       : {result.initial_median_error_px:.4f}")
    print(f"  initial mean         : {result.initial_mean_error_px:.4f}")
    print(f"  final median         : {result.final_median_error_px:.4f}")
    print(f"  final mean           : {result.final_mean_error_px:.4f}")

    print("=" * 70)
    print()