from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from overlay.tools.homography import build_board_xyz_canonical
from tracking.transforms import invert_transform, transform_to_rvec_tvec


# ============================================================
# Helpers
# ============================================================

def normalize_dist_coeffs(dist_coeffs: np.ndarray | None) -> np.ndarray:
    """
    Normalize distortion coefficients to a float64 column vector.

    If None is given, a zero-distortion model is assumed.
    """
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)

    dist = np.asarray(dist_coeffs, dtype=np.float64)
    return dist.reshape(-1, 1)


def compute_reprojection_error_px(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    """
    Project 3D object points into the image and compute reprojection errors.

    Returns
    -------
    uv_proj : (N,2)
    mean_px : float
    median_px : float
    max_px : float
    per_point_px : (N,)
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    uv_proj, _ = cv2.projectPoints(object_points_xyz, rvec, tvec, K, dist)
    uv_proj = uv_proj.reshape(-1, 2)

    per_point_px = np.linalg.norm(image_points_uv - uv_proj, axis=1)
    mean_px = float(np.mean(per_point_px))
    median_px = float(np.median(per_point_px))
    max_px = float(np.max(per_point_px))

    return uv_proj, mean_px, median_px, max_px, per_point_px


def _compute_inlier_idx(
    inliers: np.ndarray | None,
    num_points: int,
) -> np.ndarray:
    """
    Convert OpenCV-style inlier output to a flat int64 index array.

    If no inliers are provided, all points are treated as inliers.
    """
    if inliers is None or len(inliers) == 0:
        return np.arange(num_points, dtype=np.int64)

    return np.asarray(inliers, dtype=np.int64).reshape(-1)


def _compute_reprojection_stats_from_subset(
    reproj_errors_px: np.ndarray,
    inlier_idx: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute mean / median / max reprojection error on the selected subset.
    """
    reproj_errors_px = np.asarray(reproj_errors_px, dtype=np.float64).reshape(-1)
    inlier_idx = np.asarray(inlier_idx, dtype=np.int64).reshape(-1)

    if len(reproj_errors_px) == 0:
        raise ValueError("At least one reprojection error is required.")

    if len(inlier_idx) == 0:
        inlier_idx = np.arange(len(reproj_errors_px), dtype=np.int64)

    inlier_errs = reproj_errors_px[inlier_idx]

    return (
        float(np.mean(inlier_errs)),
        float(np.median(inlier_errs)),
        float(np.max(inlier_errs)),
    )


def _refine_pose_iterative(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine an existing pose estimate using SOLVEPNP_ITERATIVE
    with useExtrinsicGuess=True.
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    rvec_init = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
    tvec_init = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1)

    success, rvec_ref, tvec_ref = cv2.solvePnP(
        object_points_xyz,
        image_points_uv,
        K,
        dist,
        rvec=rvec_init,
        tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise RuntimeError("Iterative pose refinement failed.")

    return (
        np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1),
    )


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def make_transform_from_rvec_tvec(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from OpenCV rvec/tvec.

    This is intentionally public and generic so tracker modules can reuse the
    same transform construction without duplicating OpenCV Rodrigues logic.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    R, _ = cv2.Rodrigues(rvec)
    return _make_transform(R, tvec)


def _rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _rotation_deviation_deg(R_ref: np.ndarray, R: np.ndarray) -> float:
    R_rel = (
        np.asarray(R_ref, dtype=np.float64).reshape(3, 3).T
        @ np.asarray(R, dtype=np.float64).reshape(3, 3)
    )
    return _rotation_angle_deg(R_rel)


def _rigid_fit_kabsch(
    board_xyz_mm: np.ndarray,
    cam_xyz_mm: np.ndarray,
) -> dict:
    """
    Estimate T_bc^depth from 3D-3D correspondences using Kabsch / SVD.

    Parameters
    ----------
    board_xyz_mm : (N, 3)
        Canonical board points in board frame, mm.
    cam_xyz_mm : (N, 3)
        Reconstructed board points in camera frame, mm.

    Returns
    -------
    dict with keys:
        R, t, T, mean_mm, median_mm, rms_mm, max_mm, det_R
    """
    A = np.asarray(board_xyz_mm, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(cam_xyz_mm, dtype=np.float64).reshape(-1, 3)

    if A.shape != B.shape:
        raise ValueError("board_xyz_mm and cam_xyz_mm must have identical shape.")

    ca = np.mean(A, axis=0)
    cb = np.mean(B, axis=0)

    A0 = A - ca
    B0 = B - cb

    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = cb - R @ ca

    B_hat = (R @ A.T).T + t
    errs = np.linalg.norm(B_hat - B, axis=1)

    return {
        "R": R,
        "t": t,
        "T": _make_transform(R, t),
        "mean_mm": float(np.mean(errs)),
        "median_mm": float(np.median(errs)),
        "rms_mm": float(np.sqrt(np.mean(errs ** 2))),
        "max_mm": float(np.max(errs)),
        "det_R": float(np.linalg.det(R)),
    }


def _trust_region_from_depth_rms(
    rms_mm: float,
    half_board_extent_mm: float = 12.7,
    trans_scale: float = 100.0,
    rot_scale: float = 50.0,
    trans_min_mm: float = 1.0,
    trans_max_mm: float = 10.0,
    rot_min_deg: float = 1.0,
    rot_max_deg: float = 15.0,
) -> dict:
    gamma_t_mm_raw = trans_scale * float(rms_mm)
    gamma_t_mm = min(max(gamma_t_mm_raw, trans_min_mm), trans_max_mm)

    gamma_r_rad_raw = rot_scale * float(rms_mm) / float(half_board_extent_mm)
    gamma_r_deg_raw = float(np.degrees(gamma_r_rad_raw))
    gamma_r_deg = min(max(gamma_r_deg_raw, rot_min_deg), rot_max_deg)

    return {
        "gamma_t_mm_raw": gamma_t_mm_raw,
        "gamma_t_mm": gamma_t_mm,
        "gamma_r_deg_raw": gamma_r_deg_raw,
        "gamma_r_deg": gamma_r_deg,
        "trans_min_mm": trans_min_mm,
        "trans_max_mm": trans_max_mm,
        "rot_min_deg": rot_min_deg,
        "rot_max_deg": rot_max_deg,
    }


def _score_candidate_against_depth(
    T_depth: np.ndarray,
    T_ippe: np.ndarray,
    gamma_t_mm: float,
    gamma_r_deg: float,
) -> dict:
    R_d = np.asarray(T_depth[:3, :3], dtype=np.float64)
    t_d = np.asarray(T_depth[:3, 3], dtype=np.float64).reshape(3)
    R_i = np.asarray(T_ippe[:3, :3], dtype=np.float64)
    t_i = np.asarray(T_ippe[:3, 3], dtype=np.float64).reshape(3)

    delta_t_mm = float(np.linalg.norm(t_i - t_d))
    delta_r_deg = float(_rotation_deviation_deg(R_d, R_i))
    feasible = bool((delta_r_deg <= gamma_r_deg) and (delta_t_mm <= gamma_t_mm))

    score = (delta_r_deg / gamma_r_deg) ** 2 + (delta_t_mm / gamma_t_mm) ** 2

    return {
        "delta_t_mm": delta_t_mm,
        "delta_r_deg": delta_r_deg,
        "score": float(score),
        "feasible": feasible,
    }


# ============================================================
# Result types
# ============================================================

@dataclass(frozen=True)
class IppeCandidateResult:
    candidate_index: int
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    uv_proj: np.ndarray
    reproj_errors_px: np.ndarray
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float


@dataclass(frozen=True)
class PoseSolveResult:
    rvec: np.ndarray
    tvec: np.ndarray
    method: str
    raw_pnp_flag: int
    used_extrinsic_guess: bool
    candidate_index: int | None
    candidate_index_rgb: int | None
    candidate_index_xray: int | None
    refined_with_iterative: bool
    refinement_pnp_flag: int | None

    inliers: np.ndarray | None
    inlier_idx: np.ndarray

    uv_proj: np.ndarray
    reproj_errors_px: np.ndarray
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float

    all_candidates: list[IppeCandidateResult] | None = None
    all_candidates_rgb: list[IppeCandidateResult] | None = None


# ============================================================
# IPPE internals
# ============================================================

@dataclass(frozen=True)
class _IppeCandidate:
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    reproj_mean_px: float


def _build_ippe_candidates(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> list[_IppeCandidate]:
    """
    Run cv2.solvePnPGeneric with SOLVEPNP_IPPE and collect all returned
    solutions as ``_IppeCandidate`` objects.
    """
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        objectPoints=object_points_xyz,
        imagePoints=image_points_uv,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success or rvecs is None or tvecs is None or len(rvecs) == 0:
        raise RuntimeError("SOLVEPNP_IPPE failed.")

    if len(rvecs) != len(tvecs):
        raise RuntimeError("IPPE returned inconsistent candidate counts.")

    candidates: list[_IppeCandidate] = []

    for rv_raw, tv_raw in zip(rvecs, tvecs):
        rv = np.asarray(rv_raw, dtype=np.float64).reshape(3, 1)
        tv = np.asarray(tv_raw, dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rv)

        _, mean_px, _, _, _ = compute_reprojection_error_px(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            rvec=rv,
            tvec=tv,
            K=K,
            dist=dist,
        )

        candidates.append(
            _IppeCandidate(
                rvec=rv,
                tvec=tv,
                R=R,
                reproj_mean_px=mean_px,
            )
        )

    return candidates


def _export_ippe_candidate_result(
    candidate: _IppeCandidate,
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    candidate_index: int,
) -> IppeCandidateResult:
    uv_proj, mean_px, median_px, max_px, per_point_px = compute_reprojection_error_px(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        rvec=candidate.rvec,
        tvec=candidate.tvec,
        K=K,
        dist=dist,
    )

    return IppeCandidateResult(
        candidate_index=int(candidate_index),
        rvec=np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(candidate.tvec, dtype=np.float64).reshape(3, 1),
        R=np.asarray(candidate.R, dtype=np.float64).reshape(3, 3),
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=float(mean_px),
        reproj_median_px=float(median_px),
        reproj_max_px=float(max_px),
    )


# ============================================================
# IPPE candidate selection
# ============================================================

def _select_ippe_candidate(
    candidates: list[_IppeCandidate],
    *,
    use_xray_ippe_selection_rule: bool = False,
) -> int:
    if len(candidates) != 2:
        raise ValueError(
            f"Expected exactly 2 IPPE candidates, got {len(candidates)}."
        )

    if not use_xray_ippe_selection_rule:
        s0, s1 = candidates[0], candidates[1]
        return 0 if s0.reproj_mean_px <= s1.reproj_mean_px else 1

    infos = []
    z_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for i, s in enumerate(candidates):
        R_bx = np.asarray(s.R, dtype=np.float64).reshape(3, 3)

        # X-ray z-axis expressed in board frame.
        # Columns of R_bx are board axes in xray frame.
        # Rows / columns of R_xb = R_bx.T give xray axes in board frame.
        R_xb = R_bx.T
        z_x_in_b = R_xb[:, 2]

        # dot close to -1 <=> best antiparallel
        dot_z = float(np.dot(z_b, z_x_in_b))

        infos.append(
            {
                "idx": i,
                "z_x_in_b": z_x_in_b,
                "dot_z": dot_z,
                "reproj_mean_px": float(s.reproj_mean_px),
            }
        )

    print("\n=== DEBUG XRAY IPPE SELECTION ===")
    for info in infos:
        print(f"\nCandidate {info['idx']}:")
        print(f"  z_x_in_b       = {info['z_x_in_b']}")
        print(f"  dot(z_b, z_x)  = {info['dot_z']:.6f}")
        print(f"  reproj mean px = {info['reproj_mean_px']:.6f}")

    # smaller dot => more antiparallel
    if infos[0]["dot_z"] != infos[1]["dot_z"]:
        return 0 if infos[0]["dot_z"] < infos[1]["dot_z"] else 1

    # fallback: smaller reprojection error
    e0 = infos[0]["reproj_mean_px"]
    e1 = infos[1]["reproj_mean_px"]
    return 0 if e0 <= e1 else 1


def _select_ippe_candidate_rgb(
    candidates: list[_IppeCandidate],
    T_bc_depth: np.ndarray,
    *,
    gamma_t_mm: float,
    gamma_r_deg: float,
) -> int:
    """
    Select the RGB-side IPPE candidate using a depth-based trust region.
    """
    if len(candidates) != 2:
        raise ValueError(f"Expected exactly 2 IPPE candidates, got {len(candidates)}.")

    scores = []

    for cand in candidates:
        T_ippe = _make_transform(cand.R, cand.tvec.ravel())
        scores.append(
            _score_candidate_against_depth(
                T_depth=T_bc_depth,
                T_ippe=T_ippe,
                gamma_t_mm=gamma_t_mm,
                gamma_r_deg=gamma_r_deg,
            )
        )

    print("\n=== DEBUG RGB IPPE SELECTION ===")

    for i, s in enumerate(scores):
        print(f"\nCandidate {i}:")
        print(f"  delta_t_mm   = {s['delta_t_mm']:.4f}")
        print(f"  delta_r_deg  = {s['delta_r_deg']:.4f}")
        print(f"  score        = {s['score']:.4f}")
        print(f"  feasible     = {s['feasible']}")

    print("\nTrust region:")
    print(f"  gamma_t_mm   = {gamma_t_mm:.4f}")
    print(f"  gamma_r_deg  = {gamma_r_deg:.4f}")

    print("\nDepth reference T_bc:")
    print(T_bc_depth)

    f0 = bool(scores[0]["feasible"])
    f1 = bool(scores[1]["feasible"])

    if f0 != f1:
        return 0 if f0 else 1

    if f0 and f1:
        return 0 if scores[0]["score"] <= scores[1]["score"] else 1

    raise RuntimeError(
        "RGB IPPE candidate selection failed: neither candidate lies within "
        "the depth-based trust region."
    )


# ============================================================
# Core solvers
# ============================================================

def _solve_pose_iterative(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    rvec_init: np.ndarray | None = None,
    tvec_init: np.ndarray | None = None,
    use_extrinsic_guess: bool = False,
    refine_with_iterative: bool = True,
) -> PoseSolveResult:
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(dist_coeffs)

    if object_points_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    used_guess = bool(
        use_extrinsic_guess
        and rvec_init is not None
        and tvec_init is not None
    )

    if used_guess:
        rvec_init = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
        tvec_init = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1)

        success, rvec, tvec = cv2.solvePnP(
            object_points_xyz,
            image_points_uv,
            K,
            dist,
            rvec=rvec_init,
            tvec=tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        success, rvec, tvec = cv2.solvePnP(
            object_points_xyz,
            image_points_uv,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success:
        raise RuntimeError("SOLVEPNP_ITERATIVE failed.")

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    if refine_with_iterative:
        rvec, tvec = _refine_pose_iterative(
            object_points_xyz,
            image_points_uv,
            K,
            dist,
            rvec,
            tvec,
        )

    uv_proj, _, _, _, per_point_px = compute_reprojection_error_px(
        object_points_xyz,
        image_points_uv,
        rvec,
        tvec,
        K,
        dist,
    )

    inlier_idx = _compute_inlier_idx(None, num_points=len(image_points_uv))
    mean_px, median_px, max_px = _compute_reprojection_stats_from_subset(
        per_point_px,
        inlier_idx,
    )

    return PoseSolveResult(
        rvec=rvec,
        tvec=tvec,
        method="iterative",
        raw_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE),
        used_extrinsic_guess=used_guess,
        candidate_index=None,
        candidate_index_rgb=None,
        candidate_index_xray=None,
        refined_with_iterative=bool(refine_with_iterative),
        refinement_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE) if refine_with_iterative else None,
        inliers=None,
        inlier_idx=inlier_idx,
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=mean_px,
        reproj_median_px=median_px,
        reproj_max_px=max_px,
        all_candidates=None,
        all_candidates_rgb=None,
    )


def _solve_pose_iterative_ransac(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    refine_with_iterative: bool = True,
    ransac_reprojection_error_px: float = 8.0,
    ransac_confidence: float = 0.99,
    ransac_iterations_count: int = 100,
) -> PoseSolveResult:
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(dist_coeffs)

    if object_points_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points_xyz,
        imagePoints=image_points_uv,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=float(ransac_reprojection_error_px),
        confidence=float(ransac_confidence),
        iterationsCount=int(ransac_iterations_count),
    )

    if not success:
        raise RuntimeError("SOLVEPNP_ITERATIVE_RANSAC failed.")

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    inliers = None if inliers is None else np.asarray(inliers, dtype=np.int64)

    if refine_with_iterative:
        inlier_idx_for_refine = _compute_inlier_idx(
            inliers,
            num_points=len(image_points_uv),
        )

        object_refine = object_points_xyz[inlier_idx_for_refine]
        image_refine = image_points_uv[inlier_idx_for_refine]

        rvec, tvec = _refine_pose_iterative(
            object_refine,
            image_refine,
            K,
            dist,
            rvec,
            tvec,
        )

    uv_proj, _, _, _, per_point_px = compute_reprojection_error_px(
        object_points_xyz,
        image_points_uv,
        rvec,
        tvec,
        K,
        dist,
    )

    inlier_idx = _compute_inlier_idx(inliers, num_points=len(image_points_uv))
    mean_px, median_px, max_px = _compute_reprojection_stats_from_subset(
        per_point_px,
        inlier_idx,
    )

    return PoseSolveResult(
        rvec=rvec,
        tvec=tvec,
        method="iterative_ransac",
        raw_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE),
        used_extrinsic_guess=False,
        candidate_index=None,
        candidate_index_rgb=None,
        candidate_index_xray=None,
        refined_with_iterative=bool(refine_with_iterative),
        refinement_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE) if refine_with_iterative else None,
        inliers=inliers,
        inlier_idx=inlier_idx,
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=mean_px,
        reproj_median_px=median_px,
        reproj_max_px=max_px,
        all_candidates=None,
        all_candidates_rgb=None,
    )


def _solve_pose_ippe(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    refine_with_iterative: bool = True,
    use_xray_ippe_selection_rule: bool = False,
) -> PoseSolveResult:
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(dist_coeffs)

    if object_points_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    candidates = _build_ippe_candidates(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        K=K,
        dist=dist,
    )

    all_candidates = [
        _export_ippe_candidate_result(
            candidate=c,
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist=dist,
            candidate_index=i,
        )
        for i, c in enumerate(candidates)
    ]

    best_idx = _select_ippe_candidate(
        candidates,
        use_xray_ippe_selection_rule=use_xray_ippe_selection_rule,
    )

    best = candidates[best_idx]

    rvec = best.rvec
    tvec = best.tvec

    if refine_with_iterative:
        rvec, tvec = _refine_pose_iterative(
            object_points_xyz,
            image_points_uv,
            K,
            dist,
            rvec,
            tvec,
        )

    uv_proj, _, _, _, per_point_px = compute_reprojection_error_px(
        object_points_xyz,
        image_points_uv,
        rvec,
        tvec,
        K,
        dist,
    )

    inlier_idx = _compute_inlier_idx(None, num_points=len(image_points_uv))
    mean_px, median_px, max_px = _compute_reprojection_stats_from_subset(
        per_point_px,
        inlier_idx,
    )

    return PoseSolveResult(
        rvec=rvec,
        tvec=tvec,
        method="ippe",
        raw_pnp_flag=int(cv2.SOLVEPNP_IPPE),
        used_extrinsic_guess=False,
        candidate_index=int(best_idx),
        candidate_index_rgb=None,
        candidate_index_xray=None,
        refined_with_iterative=bool(refine_with_iterative),
        refinement_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE) if refine_with_iterative else None,
        inliers=None,
        inlier_idx=inlier_idx,
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=mean_px,
        reproj_median_px=median_px,
        reproj_max_px=max_px,
        all_candidates=all_candidates,
        all_candidates_rgb=None,
    )


def _solve_pose_ippe_handeye(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K_xray: np.ndarray,
    checkerboard_corners_uv: np.ndarray,
    K_rgb: np.ndarray,
    *,
    dist_coeffs_rgb: np.ndarray | None = None,
    pitch_mm: float = 2.54,
    steps_per_edge: int = 10,
    refine_with_iterative: bool = False,
    refine_rgb_iterative: bool = False,
    refine_xray_iterative: bool = False,
) -> PoseSolveResult:
    cam_points_xyz_m = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K_xray = np.asarray(K_xray, dtype=np.float64).reshape(3, 3)
    checkerboard_corners_uv = np.asarray(
        checkerboard_corners_uv,
        dtype=np.float64,
    ).reshape(3, 2)
    K_rgb = np.asarray(K_rgb, dtype=np.float64).reshape(3, 3)

    dist_xray = normalize_dist_coeffs(None)
    dist_rgb = normalize_dist_coeffs(dist_coeffs_rgb)

    if refine_with_iterative:
        refine_rgb_iterative = True
        refine_xray_iterative = True

    n_pts = (steps_per_edge + 1) ** 2

    if image_points_uv.shape[0] != n_pts:
        raise ValueError(
            f"Expected {n_pts} X-ray image points for steps_per_edge={steps_per_edge}, "
            f"got {image_points_uv.shape[0]}."
        )

    if cam_points_xyz_m.shape != (n_pts, 3):
        raise ValueError(
            f"Expected object_points_xyz with shape ({n_pts}, 3) "
            f"for pose_method='ippe_handeye', got {cam_points_xyz_m.shape}."
        )

    pts3d_board_mm = build_board_xyz_canonical(
        nu=int(steps_per_edge),
        nv=int(steps_per_edge),
        pitch_mm=float(pitch_mm),
    )

    p_tl, p_tr, p_bl = checkerboard_corners_uv
    step_x = (p_tr - p_tl) / float(steps_per_edge)
    step_y = (p_bl - p_tl) / float(steps_per_edge)

    pts2d_rgb = np.array(
        [
            p_tl + alpha * step_x + beta * step_y
            for beta in range(steps_per_edge + 1)
            for alpha in range(steps_per_edge + 1)
        ],
        dtype=np.float64,
    )

    cam_points_xyz_mm = cam_points_xyz_m * 1000.0
    depth_fit = _rigid_fit_kabsch(pts3d_board_mm, cam_points_xyz_mm)
    T_bc_depth = depth_fit["T"]

    tr = _trust_region_from_depth_rms(depth_fit["rms_mm"])

    candidates_rgb = _build_ippe_candidates(
        pts3d_board_mm,
        pts2d_rgb,
        K_rgb,
        dist_rgb,
    )

    best_idx_rgb = _select_ippe_candidate_rgb(
        candidates_rgb,
        T_bc_depth=T_bc_depth,
        gamma_t_mm=tr["gamma_t_mm"],
        gamma_r_deg=tr["gamma_r_deg"],
    )

    best_rgb = candidates_rgb[best_idx_rgb]

    rvec_bc = np.asarray(best_rgb.rvec, dtype=np.float64).reshape(3, 1)
    tvec_bc = np.asarray(best_rgb.tvec, dtype=np.float64).reshape(3, 1)

    if refine_rgb_iterative:
        rvec_bc, tvec_bc = _refine_pose_iterative(
            pts3d_board_mm,
            pts2d_rgb,
            K_rgb,
            dist_rgb,
            rvec_bc,
            tvec_bc,
        )

    R_bc, _ = cv2.Rodrigues(rvec_bc)
    T_bc = _make_transform(R_bc, tvec_bc.ravel())

    candidates_xray = _build_ippe_candidates(
        pts3d_board_mm,
        image_points_uv,
        K_xray,
        dist_xray,
    )

    best_idx_xray = _select_ippe_candidate(
        candidates_xray,
        use_xray_ippe_selection_rule=True,
    )

    best_xray = candidates_xray[best_idx_xray]

    rvec_bx = np.asarray(best_xray.rvec, dtype=np.float64).reshape(3, 1)
    tvec_bx = np.asarray(best_xray.tvec, dtype=np.float64).reshape(3, 1)

    if refine_xray_iterative:
        rvec_bx, tvec_bx = _refine_pose_iterative(
            pts3d_board_mm,
            image_points_uv,
            K_xray,
            dist_xray,
            rvec_bx,
            tvec_bx,
        )

    R_bx, _ = cv2.Rodrigues(rvec_bx)
    T_bx = _make_transform(R_bx, tvec_bx.ravel())

    T_cx = T_bx @ invert_transform(T_bc)
    T_cx[:3, 3] *= 1e-3  # mm -> m

    rvec, tvec = transform_to_rvec_tvec(T_cx)

    uv_proj, _, _, _, per_point_px = compute_reprojection_error_px(
        pts3d_board_mm,
        image_points_uv,
        rvec_bx,
        tvec_bx,
        K_xray,
        dist_xray,
    )

    inlier_idx = _compute_inlier_idx(None, num_points=len(image_points_uv))
    mean_px, median_px, max_px = _compute_reprojection_stats_from_subset(
        per_point_px,
        inlier_idx,
    )

    all_candidates_rgb = [
        _export_ippe_candidate_result(
            c,
            pts3d_board_mm,
            pts2d_rgb,
            K_rgb,
            dist_rgb,
            i,
        )
        for i, c in enumerate(candidates_rgb)
    ]

    all_candidates_xray = [
        _export_ippe_candidate_result(
            c,
            pts3d_board_mm,
            image_points_uv,
            K_xray,
            dist_xray,
            i,
        )
        for i, c in enumerate(candidates_xray)
    ]

    return PoseSolveResult(
        rvec=rvec,
        tvec=tvec,
        method="ippe_handeye",
        raw_pnp_flag=int(cv2.SOLVEPNP_IPPE),
        used_extrinsic_guess=False,
        candidate_index=None,
        candidate_index_rgb=int(best_idx_rgb),
        candidate_index_xray=int(best_idx_xray),
        refined_with_iterative=bool(refine_rgb_iterative or refine_xray_iterative),
        refinement_pnp_flag=(
            int(cv2.SOLVEPNP_ITERATIVE)
            if (refine_rgb_iterative or refine_xray_iterative)
            else None
        ),
        inliers=None,
        inlier_idx=inlier_idx,
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=mean_px,
        reproj_median_px=median_px,
        reproj_max_px=max_px,
        all_candidates=all_candidates_xray,
        all_candidates_rgb=all_candidates_rgb,
    )


# ============================================================
# Public API
# ============================================================

def solve_pose(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    dist_coeffs_rgb: np.ndarray | None = None,
    pose_method: str = "iterative",
    rvec_init: np.ndarray | None = None,
    tvec_init: np.ndarray | None = None,
    use_extrinsic_guess: bool = False,
    refine_with_iterative: bool = True,
    refine_rgb_iterative: bool = False,
    refine_xray_iterative: bool = False,
    use_xray_ippe_selection_rule: bool = False,
    ransac_reprojection_error_px: float = 8.0,
    ransac_confidence: float = 0.99,
    ransac_iterations_count: int = 100,
    pitch_mm: float = 2.54,
    checkerboard_corners_uv: np.ndarray | None = None,
    K_rgb: np.ndarray | None = None,
    steps_per_edge: int = 10,
) -> PoseSolveResult:
    """
    Dispatch pose estimation by method name.

    Parameters
    ----------
    object_points_xyz : (N, 3)
        3D object points.

        For method='ippe_handeye', these must be the reconstructed board
        points in the camera frame (meters). They are used to compute the
        depth-based reference pose T_bc^depth for RGB-side IPPE
        disambiguation.
    image_points_uv : (N, 2)
        2D image points. For method='ippe_handeye', these are the X-ray
        marker positions.
    K : (3, 3)
        Camera / X-ray intrinsic matrix.
    dist_coeffs : array-like or None
        Distortion coefficients for the main imaging system represented by K.
        For iterative / iterative_ransac / ippe this is the distortion used
        directly in the solve.
    dist_coeffs_rgb : array-like or None
        RGB camera distortion coefficients. Only used for method='ippe_handeye'.
    pose_method : str
        One of 'iterative', 'iterative_ransac', 'ippe', 'ippe_handeye'.
    use_xray_ippe_selection_rule : bool
        Only used for method='ippe'. If True, use the setup-specific
        source-location rule in the object frame for candidate selection.
    checkerboard_corners_uv : (3, 2) or None
        The three extreme checkerboard corners [TL, TR, BL] in the RGB
        image. Required for method='ippe_handeye'.
    K_rgb : (3, 3) or None
        RGB camera intrinsic matrix. Required for method='ippe_handeye'.
    steps_per_edge : int
        Grid steps per edge for method='ippe_handeye'. Default 10 -> 121 points.

    Refinement options for ippe_handeye
    -----------------------------------
    refine_with_iterative : bool
        Backward-compatible switch. If True, refine both T_bc and T_bx.

    refine_rgb_iterative : bool
        If True, refine only the RGB-side pose T_bc.

    refine_xray_iterative : bool
        If True, refine only the X-ray-side pose T_bx.
    """
    method = str(pose_method).lower().strip()

    if method == "iterative":
        return _solve_pose_iterative(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist_coeffs=dist_coeffs,
            rvec_init=rvec_init,
            tvec_init=tvec_init,
            use_extrinsic_guess=use_extrinsic_guess,
            refine_with_iterative=refine_with_iterative,
        )

    if method == "iterative_ransac":
        return _solve_pose_iterative_ransac(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist_coeffs=dist_coeffs,
            refine_with_iterative=refine_with_iterative,
            ransac_reprojection_error_px=ransac_reprojection_error_px,
            ransac_confidence=ransac_confidence,
            ransac_iterations_count=ransac_iterations_count,
        )

    if method == "ippe":
        return _solve_pose_ippe(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist_coeffs=dist_coeffs,
            refine_with_iterative=refine_with_iterative,
            use_xray_ippe_selection_rule=use_xray_ippe_selection_rule,
        )

    if method == "ippe_handeye":
        if checkerboard_corners_uv is None:
            raise ValueError("pose_method='ippe_handeye' requires checkerboard_corners_uv.")

        if K_rgb is None:
            raise ValueError("pose_method='ippe_handeye' requires K_rgb.")

        return _solve_pose_ippe_handeye(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K_xray=K,
            checkerboard_corners_uv=checkerboard_corners_uv,
            K_rgb=K_rgb,
            dist_coeffs_rgb=dist_coeffs_rgb,
            pitch_mm=pitch_mm,
            steps_per_edge=steps_per_edge,
            refine_with_iterative=refine_with_iterative,
            refine_rgb_iterative=refine_rgb_iterative,
            refine_xray_iterative=refine_xray_iterative,
        )

    raise ValueError(
        f"Unknown pose_method '{pose_method}'. "
        f"Expected 'iterative', 'iterative_ransac', 'ippe', or 'ippe_handeye'."
    )