"""Offline re-extraction of SfM observations from a saved session's frames.

The live recorder (record_observations_realsense.py) needs a camera. This script
replays the frames it already saved through the SAME tracker configuration and
the SAME acceptance gates, so a session can be re-observed with a different
corner-measurement operator -- without recording anything again.

Why: the tracked-corner operator (checker_tracked_refine_method) changes the
recorded pixel observations (measured 2026-07-29: 93% of observations differ,
median 0.38 px), so a model built from them inherits that operator. To judge an
operator without that circularity, the model has to be rebuilt from observations
extracted with the SAME operator the runtime will use.

    python reextract_observations_offline.py FIELD MODEL.json FRAMES_DIR \
           CAM_CALIB.npz OUT.npz [--refine-method subpix|quadratic_form]

TWO LIMITATIONS, by construction -- state them when reporting results:

  1. TEMPORAL CONTEXT. The recorder saved only the frames it ACCEPTED (pose from
     decode, enough correspondences, >= 1 px motion since the last save), so
     consecutive saved frames are not consecutive camera frames. Replaying them
     back to back gives the optical-flow path larger inter-frame jumps than the
     live run had. The extraction is therefore NOT a bit-reproduction of the
     original observation file.

  2. FRAME COVERAGE. Only what the writer kept on disk can be re-observed (a
     session may hold more observation entries than saved images).

Both apply IDENTICALLY to every operator, so a COMPARISON between two
re-extracted sets is fair even though neither reproduces the original file. For
that reason: always re-extract the baseline operator too, and compare the two
re-extracted models against each other -- never a re-extracted model against the
original one.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tracking.hydramarker.tracker import HydraTracker  # noqa: E402
from tracking.hydramarker.model.observations import (  # noqa: E402
    FrameObservation,
    save_observations_npz,
)
from tracking.hydramarker.model.record_observations_realsense import (  # noqa: E402
    MIN_SAVE_MOTION_PX,
    frame_from_correspondence_corners,
    make_sfm_tracker,
    median_shared_displacement_px,
    read_num_corner_cols,
    should_save_frame,
)


def load_cam_calib(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    K = dist = None
    for k in ("K", "camera_matrix", "matrix", "K_rgb"):
        if k in z.files:
            K = np.asarray(z[k], float).reshape(3, 3)
            break
    for k in ("dist", "dist_coeffs", "distortion", "dist_rgb"):
        if k in z.files:
            dist = np.asarray(z[k], float).reshape(-1, 1)
            break
    if K is None:
        raise SystemExit(f"no camera matrix in {path}")
    if dist is None:
        dist = np.zeros((5, 1))
    return K, dist


def color_frame_indices(frames_dir: Path) -> list[int]:
    """Saved colour frames (frame_XXXXXX.png), excluding the IR companions."""
    out = []
    for p in sorted(frames_dir.glob("frame_*.png")):
        stem = p.stem
        if "_ir" in stem:
            continue
        try:
            out.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def build_tracker(field: Path, model_json: Path, K, dist, method: str):
    """The recorder's own tracker config, with the operator applied BEFORE
    construction.

    The config is copied into the C++ engine when the tracker is built, so
    assigning tracker.config.<field> afterwards is silently ignored (verified
    2026-07-29: the override left qf_count identical in both arms).
    """
    proto = make_sfm_tracker(field, model_json, K, dist)
    cfg = dataclasses.replace(proto.config,
                              checker_tracked_refine_method=method)
    del proto
    tracker = HydraTracker(field_path=str(field),
                           marker_json_path=str(model_json),
                           K=K, dist_coeffs=dist, config=cfg)
    if tracker.config.checker_tracked_refine_method != method:
        raise SystemExit("refine method did not reach the tracker config")
    return tracker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("field")
    ap.add_argument("model_json")
    ap.add_argument("frames_dir")
    ap.add_argument("cam_calib_npz")
    ap.add_argument("out_npz")
    ap.add_argument("--refine-method",
                    choices=("subpix", "quadratic_form"), default="subpix")
    # The recorder required a decode-sourced pose for every stored observation.
    # Kept by default; relaxing it trades identity strictness for coverage.
    ap.add_argument("--allow-nondecode", action="store_true")
    ap.add_argument("--require-ir-pair", action="store_true",
                    help="keep only frames that also have irL AND irR on disk "
                         "(the subset the IR post-pass can use)")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    field = Path(args.field)
    model_json = Path(args.model_json)
    out_npz = Path(args.out_npz)

    K, dist = load_cam_calib(Path(args.cam_calib_npz))
    num_cols = read_num_corner_cols(model_json)
    idxs = color_frame_indices(frames_dir)
    if args.require_ir_pair:
        idxs = [i for i in idxs
                if (frames_dir / f"frame_{i:06d}_irL.png").exists()
                and (frames_dir / f"frame_{i:06d}_irR.png").exists()]
    print(f"[reextract] method={args.refine_method}  num_corner_cols={num_cols}")
    print(f"[reextract] colour frames to replay: {len(idxs)}")

    tracker = build_tracker(field, model_json, K, dist, args.refine_method)

    observations: list[FrameObservation] = []
    reasons: dict[str, int] = {}
    qf_corners = 0
    for i in idxs:
        img = cv2.imread(str(frames_dir / f"frame_{i:06d}.png"))
        if img is None:
            continue
        result = tracker.process_frame(img)
        tm = dict(getattr(result, "timings_ms", {}) or {})
        qf_corners += int(float(tm.get("checkerboard_tracking_qf_count", 0) or 0))

        obs = frame_from_correspondence_corners(
            frame_id=i, result=result, num_cols=num_cols,
            require_decode_source=not args.allow_nondecode)

        keep, reason = should_save_frame(result, obs)
        if keep and args.allow_nondecode and reason == "not decode":
            keep = True
        if keep and observations:
            motion = median_shared_displacement_px(observations[-1], obs)
            if motion is not None and motion < MIN_SAVE_MOTION_PX:
                keep, reason = False, f"motion < {MIN_SAVE_MOTION_PX:.1f}px"
        reasons[reason] = reasons.get(reason, 0) + 1
        if keep:
            observations.append(obs)

    print(f"[reextract] kept {len(observations)} / {len(idxs)} frames")
    print(f"[reextract] qf-measured corners during replay: {qf_corners} "
          f"(expected 0 for subpix)")
    for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {c:5d}  {r}")
    if not observations:
        raise SystemExit("[reextract] nothing kept - check the gates")
    if args.refine_method == "subpix" and qf_corners:
        raise SystemExit("[reextract] subpix requested but QF ran - the config "
                         "did not reach the engine")

    save_observations_npz(out_npz, observations)
    n_obs = sum(len(f.observations) for f in observations)
    print(f"[reextract] wrote {out_npz}  ({len(observations)} frames, "
          f"{n_obs} corner observations, "
          f"{n_obs / max(1, len(observations)):.1f} per frame)")


if __name__ == "__main__":
    main()
