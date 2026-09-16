# tracking/hydramarker/tool_assets.py
#
# Canonical loader for a trackable tool: the file triple
#
#   .field                  HydraMarker state matrix (detector input)
#   marker .json            metric marker geometry (SfM map preferred) +
#                           surface_model.fitted cylinder = drill axis
#   pivot_calibration .npz  tool tip in the MARKER frame (calib_tool_tip.py)
#
# The tip position lives in the pivot npz (`p_tip`), the drill axis lives in
# the marker json (`surface_model.fitted.axis_point_mm` / `axis_dir`) — the
# same split every existing consumer uses (see tests/test_tracker_tool_tip.py).
# This module is the single place that knows that split; GUI code and scripts
# should load tools through here instead of re-parsing the files.
#
# No Qt / OpenCV / camera dependencies: plain numpy + json, safe to import
# from anywhere.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Fallback drill axis in the marker frame when the marker json carries no
# fitted cylinder (matches tests/test_tracker_tool_tip.py).
DEFAULT_AXIS_DIR = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class ToolAssets:
    """Everything needed to track one tool and draw its tip + axis.

    All geometry is in the MARKER frame, millimetres.
    """

    field_path: Path
    marker_json_path: Path
    pivot_npz_path: Path

    name: str                      # marker json "name"
    marker_type: str               # "sfm_map" (live tracking) or "planar"

    p_tip_mm: np.ndarray           # (3,) tool tip
    sigma_norm_mm: float           # pivot residual (NaN if absent)
    pivot_method: str              # "pivot_sphere_axis" | "pivot_sphere" | ""

    axis_point_mm: np.ndarray      # (3,) point on the drill axis
    axis_dir: np.ndarray           # (3,) unit direction of the drill axis
    axis_from_surface_model: bool  # False -> DEFAULT_AXIS_DIR fallback

    # (N, 3) marker corner positions from the json corner list (SfM maps);
    # None for planar jsons without an explicit corner list.
    corners_xyz_mm: Optional[np.ndarray] = None

    warnings: Tuple[str, ...] = field(default=())


def _load_pivot_npz(path: Path) -> tuple[np.ndarray, float, str, Optional[str]]:
    with np.load(path, allow_pickle=False) as data:
        if "p_tip" not in data.files:
            raise ValueError(
                f"'{path.name}' enthaelt kein 'p_tip' — keine gueltige "
                f"Pivot-Kalibrierung (Keys: {', '.join(sorted(data.files))})"
            )
        p_tip = np.asarray(data["p_tip"], dtype=np.float64).reshape(3)
        sigma = float(data["sigma_norm_mm"]) if "sigma_norm_mm" in data.files else float("nan")
        method = str(data["method"]) if "method" in data.files else ""
        ref_json = str(data["marker_json_path"]) if "marker_json_path" in data.files else None
    return p_tip, sigma, method, ref_json


def load_tool_assets(
    field_path: str | Path,
    marker_json_path: str | Path,
    pivot_npz_path: str | Path,
) -> ToolAssets:
    """Load and cross-validate the tool file triple.

    Raises ValueError with a user-presentable message on hard errors;
    soft inconsistencies are collected in ToolAssets.warnings.
    """
    field_path = Path(field_path)
    marker_json_path = Path(marker_json_path)
    pivot_npz_path = Path(pivot_npz_path)

    for p, what in (
        (field_path, ".field"),
        (marker_json_path, "Marker-JSON"),
        (pivot_npz_path, "Pivot-Kalibrierung (.npz)"),
    ):
        if not p.is_file():
            raise ValueError(f"{what} nicht gefunden: {p}")

    warnings: list[str] = []

    geo = json.loads(marker_json_path.read_text(encoding="utf-8"))
    if not isinstance(geo, dict) or "name" not in geo:
        raise ValueError(f"'{marker_json_path.name}' ist kein Marker-JSON (kein 'name').")

    name = str(geo.get("name", marker_json_path.stem))
    marker_type = str(geo.get("marker_type", ""))
    if marker_type != "sfm_map":
        warnings.append(
            f"Marker-JSON ist '{marker_type or 'unbekannt'}' statt 'sfm_map' — "
            f"fuer Live-Tracking ist die metrische SfM-Geometrie gedacht."
        )

    fitted = (geo.get("surface_model") or {}).get("fitted") or {}
    axis_dir_raw = fitted.get("axis_dir")
    axis_point_raw = fitted.get("axis_point_mm")
    if axis_dir_raw is not None:
        axis_dir = np.asarray(axis_dir_raw, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(axis_dir))
        if norm <= 0.0:
            raise ValueError(f"'{marker_json_path.name}': axis_dir hat Laenge 0.")
        axis_dir = axis_dir / norm
        axis_point = (
            np.asarray(axis_point_raw, dtype=np.float64).reshape(3)
            if axis_point_raw is not None
            else np.zeros(3)
        )
        axis_from_surface_model = True
    else:
        axis_dir = DEFAULT_AXIS_DIR.copy()
        axis_point = np.zeros(3)
        axis_from_surface_model = False
        warnings.append(
            "Marker-JSON hat kein surface_model.fitted — Achse faellt auf "
            "(0, 1, 0) im Marker-Frame zurueck."
        )

    corners_xyz_mm = None
    corner_list = geo.get("corners")
    if isinstance(corner_list, list) and corner_list:
        try:
            corners_xyz_mm = np.asarray(
                [c["xyz_mm"] for c in corner_list], dtype=np.float64
            ).reshape(-1, 3)
        except (KeyError, TypeError, ValueError):
            corners_xyz_mm = None

    p_tip, sigma, method, ref_json = _load_pivot_npz(pivot_npz_path)

    # Soft check: was the pivot calibrated against this marker geometry?
    # Two-level rule, because the tool identity lives in the FOLDER name
    # (data/drill/PRO6202M/...) while the json filenames repeat across
    # tools: a differing tool folder is a real mix-up; within the same
    # folder, refinement variants of the same model (e.g. '..._ir3d' vs.
    # '..._ir3d_subpix', sub-mm corner differences far below sigma_norm)
    # count as matching.
    if ref_json:
        ref_path = Path(ref_json)
        ref_dir = ref_path.parent.name
        chosen_dir = marker_json_path.resolve().parent.name
        ref_stem = ref_path.stem
        chosen_stem = marker_json_path.stem
        same_family = (
            not ref_stem
            or ref_stem == chosen_stem
            or chosen_stem.startswith(ref_stem + "_")
            or ref_stem.startswith(chosen_stem + "_")
        )
        if ref_dir and chosen_dir and ref_dir != chosen_dir:
            warnings.append(
                f"Pivot-Kalibrierung stammt aus Tool-Ordner '{ref_dir}', "
                f"das Marker-JSON aus '{chosen_dir}' — vermutlich vertauscht."
            )
        elif not same_family:
            warnings.append(
                f"Pivot-Kalibrierung wurde gegen '{ref_stem}' erstellt, "
                f"gewaehlt ist '{chosen_stem}'."
            )

    return ToolAssets(
        field_path=field_path,
        marker_json_path=marker_json_path,
        pivot_npz_path=pivot_npz_path,
        name=name,
        marker_type=marker_type,
        p_tip_mm=p_tip,
        sigma_norm_mm=sigma,
        pivot_method=method,
        axis_point_mm=axis_point,
        axis_dir=axis_dir,
        axis_from_surface_model=axis_from_surface_model,
        corners_xyz_mm=corners_xyz_mm,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Directory auto-fill (used by the GUI page: pick one file, suggest the rest)
# ---------------------------------------------------------------------------

def suggest_tool_files(directory: str | Path) -> dict[str, Optional[Path]]:
    """Scan a tool directory and suggest the file triple.

    Returns {"field": Path|None, "marker_json": Path|None, "pivot_npz": Path|None}.
    A slot stays None when the choice would be ambiguous or nothing matches.
    """
    directory = Path(directory)
    out: dict[str, Optional[Path]] = {"field": None, "marker_json": None, "pivot_npz": None}
    if not directory.is_dir():
        return out

    fields = sorted(directory.glob("*.field"))
    if len(fields) == 1:
        out["field"] = fields[0]

    jsons = sorted(directory.glob("*.json"))
    # Prefer the metric SfM geometry over the nominal planar json.
    sfm = [p for p in jsons if "sfm" in p.name.lower()]
    # Among several SfM variants, the IR-refined model is the tracking one.
    ir3d = [p for p in sfm if "ir3d" in p.name.lower()]
    if len(sfm) == 1:
        out["marker_json"] = sfm[0]
    elif len(ir3d) == 1:
        out["marker_json"] = ir3d[0]
    elif len(sfm) == 0 and len(jsons) == 1:
        out["marker_json"] = jsons[0]

    pivots = [p for p in sorted(directory.glob("*.npz")) if "pivot_calibration" in p.name.lower()]
    if len(pivots) == 1:
        out["pivot_npz"] = pivots[0]

    return out
