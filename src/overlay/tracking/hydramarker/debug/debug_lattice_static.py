from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QFileDialog

import hydramarker_cpp


# ============================================================
# QT
# ============================================================

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


# ============================================================
# LOAD IMAGE
# ============================================================

img_path, _ = QFileDialog.getOpenFileName(
    None,
    "Select HydraMarker Image",
    "",
    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
)

if not img_path:
    raise RuntimeError("No image selected")

img_path = Path(img_path)

img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
if img_bgr is None:
    raise RuntimeError(f"Could not load image:\n{img_path}")

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

out_dir = img_path.parent / f"{img_path.stem}_cpp_debug"
out_dir.mkdir(exist_ok=True)


# ============================================================
# C++ DEBUG CALL
# ============================================================

detector = hydramarker_cpp.CheckerboardDetector()

dbg = detector.debug_recovery_stages(gray)


# ============================================================
# HELPERS
# ============================================================

def xy(p):
    return int(round(p.x)), int(round(p.y))


def save(name: str, vis: np.ndarray):
    out_path = out_dir / name
    cv2.imwrite(str(out_path), vis)
    print("saved:", out_path)


def draw_label(
    vis: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color=(255, 255, 255),
    scale: float = 0.5,
    thickness: int = 1,
):
    cv2.putText(
        vis,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_header(vis: np.ndarray, text: str, color=(255, 255, 255)):
    draw_label(vis, text, (20, 35), color=color, scale=0.8, thickness=2)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("============================================================")
print("C++ CHECKERBOARD DEBUG")
print("============================================================")
print(f"image:                {img_path.name}")
print(f"raw candidates:       {len(dbg.raw_candidates)}")
print(f"refined corners:      {len(dbg.refined_corners)}")
print(f"valid refined points: {len(dbg.valid_refined_points)}")
print(f"has lattice:          {dbg.has_lattice}")
print(f"has detection:        {dbg.has_detection}")

if dbg.has_lattice:
    print(f"lattice points:       {len(dbg.lattice.points)}")
    print(f"spacing_u:            {dbg.lattice.spacing_u:.2f}")
    print(f"spacing_v:            {dbg.lattice.spacing_v:.2f}")
    print(f"origin:               ({dbg.lattice.origin.x:.1f}, {dbg.lattice.origin.y:.1f})")
    print(f"axis_u:               ({dbg.lattice.axis_u.x:.3f}, {dbg.lattice.axis_u.y:.3f})")
    print(f"axis_v:               ({dbg.lattice.axis_v.x:.3f}, {dbg.lattice.axis_v.y:.3f})")

if dbg.has_detection:
    print(f"final corners:        {len(dbg.detection.corners)}")
    print(f"final cells:          {len(dbg.detection.cells)}")
    print(f"cols x rows:          {dbg.detection.cols} x {dbg.detection.rows}")

print("============================================================")


# ============================================================
# 01 RAW CANDIDATES FROM C++
# ============================================================

vis = img_bgr.copy()

for p in dbg.raw_candidates:
    cv2.circle(vis, xy(p), 4, (0, 255, 255), -1, cv2.LINE_AA)

draw_header(vis, f"C++ raw candidates: {len(dbg.raw_candidates)}", (0, 255, 255))
save("01_cpp_raw_candidates.png", vis)


# ============================================================
# 02 REFINED CORNERS FROM C++
# green = valid, red = rejected
# ============================================================

vis = img_bgr.copy()

valid_count = 0
rejected_count = 0

for c in dbg.refined_corners:
    p = xy(c.uv)

    if c.valid:
        color = (0, 255, 0)
        valid_count += 1
    else:
        color = (0, 0, 255)
        rejected_count += 1

    cv2.circle(vis, p, 5, color, -1, cv2.LINE_AA)

    draw_label(
        vis,
        f"{c.correlation:.2f}",
        (p[0] + 5, p[1] - 5),
        color=color,
        scale=0.35,
    )

draw_header(
    vis,
    f"C++ refined: valid={valid_count}, rejected={rejected_count}",
)
save("02_cpp_refined_valid_rejected.png", vis)


# ============================================================
# 03 EXACT INPUT TO C++ LatticeModel.fit()
# ============================================================

vis = img_bgr.copy()

for p in dbg.valid_refined_points:
    cv2.circle(vis, xy(p), 5, (0, 255, 0), -1, cv2.LINE_AA)

draw_header(
    vis,
    f"Exact C++ LatticeModel input: {len(dbg.valid_refined_points)} valid refined points",
    (0, 255, 0),
)
save("03_cpp_lattice_input.png", vis)


# ============================================================
# 04 C++ LATTICE RESULT
# green = valid lattice point, red = rejected/invalid
# labels = continuous ij and residual from C++
# ============================================================

vis = img_bgr.copy()

if dbg.has_lattice:
    origin = dbg.lattice.origin
    axis_u = dbg.lattice.axis_u
    axis_v = dbg.lattice.axis_v

    o = xy(origin)

    u_end = (
        int(round(origin.x + axis_u.x * dbg.lattice.spacing_u * 5.0)),
        int(round(origin.y + axis_u.y * dbg.lattice.spacing_u * 5.0)),
    )

    v_end = (
        int(round(origin.x + axis_v.x * dbg.lattice.spacing_v * 5.0)),
        int(round(origin.y + axis_v.y * dbg.lattice.spacing_v * 5.0)),
    )

    cv2.arrowedLine(vis, o, u_end, (255, 0, 0), 3, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(vis, o, v_end, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.18)

    for lp in dbg.lattice.points:
        p = xy(lp.uv)
        color = (0, 255, 0) if lp.valid else (0, 0, 255)

        cv2.circle(vis, p, 5, color, -1, cv2.LINE_AA)

        draw_label(
            vis,
            f"{lp.ij.x:.1f},{lp.ij.y:.1f}",
            (p[0] + 6, p[1] - 6),
            color=color,
            scale=0.35,
        )

        draw_label(
            vis,
            f"r={lp.residual:.1f}",
            (p[0] + 6, p[1] + 9),
            color=(255, 255, 255),
            scale=0.30,
        )

    draw_header(
        vis,
        f"C++ lattice: points={len(dbg.lattice.points)}, spacing=({dbg.lattice.spacing_u:.1f},{dbg.lattice.spacing_v:.1f})",
    )
else:
    draw_header(vis, "C++ lattice FAILED", (0, 0, 255))

save("04_cpp_lattice_result.png", vis)


# ============================================================
# 05 C++ FINAL DETECTION / GRIDBUILDER RESULT
# ============================================================

vis = img_bgr.copy()

if dbg.has_detection:
    for cell in dbg.detection.cells:
        q = cell.corner_uv

        for k in range(4):
            p0 = xy(q[k])
            p1 = xy(q[(k + 1) % 4])
            cv2.line(vis, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(vis, xy(cell.center_uv), 2, (255, 0, 255), -1, cv2.LINE_AA)

    for c in dbg.detection.corners:
        p = xy(c.uv)

        cv2.circle(vis, p, 4, (0, 255, 0), -1, cv2.LINE_AA)

        draw_label(
            vis,
            f"{c.i},{c.j}",
            (p[0] + 5, p[1] - 5),
            color=(0, 255, 0),
            scale=0.35,
        )

    draw_header(
        vis,
        f"C++ final detection: corners={len(dbg.detection.corners)}, cells={len(dbg.detection.cells)}",
    )
else:
    draw_header(vis, "C++ GridBuilder / final detection FAILED", (0, 0, 255))

save("05_cpp_final_detection.png", vis)


# ============================================================
# 06 COMBINED DIAGNOSIS IMAGE
# ============================================================

vis = img_bgr.copy()

# raw yellow
for p in dbg.raw_candidates:
    cv2.circle(vis, xy(p), 3, (0, 255, 255), -1, cv2.LINE_AA)

# valid refined green
for p in dbg.valid_refined_points:
    cv2.circle(vis, xy(p), 4, (0, 255, 0), -1, cv2.LINE_AA)

# lattice invalid red
if dbg.has_lattice:
    for lp in dbg.lattice.points:
        if not lp.valid:
            cv2.circle(vis, xy(lp.uv), 7, (0, 0, 255), 2, cv2.LINE_AA)

# final grid cyan
if dbg.has_detection:
    for cell in dbg.detection.cells:
        q = cell.corner_uv
        for k in range(4):
            cv2.line(vis, xy(q[k]), xy(q[(k + 1) % 4]), (255, 255, 0), 1, cv2.LINE_AA)

draw_header(
    vis,
    "yellow=raw, green=valid refined, red=lattice rejected, cyan=final grid",
)
save("06_cpp_combined_diagnosis.png", vis)


# ============================================================
# SHOW FINAL
# ============================================================

cv2.namedWindow("cpp_debug_final_detection", cv2.WINDOW_NORMAL)
cv2.imshow("cpp_debug_final_detection", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

print()
print("DONE")
print("debug output folder:")
print(out_dir)