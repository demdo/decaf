from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtWidgets import QApplication, QFileDialog

import hydramarker_cpp


def choose_file_qt(title, file_filter):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def round_int(x):
    return int(round(float(x)))


def pxy(p):
    return round_int(p.x), round_int(p.y)


def put_text(img, text, pos, color=(0, 255, 255), scale=0.5, thickness=1):
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def patch_color(index):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (180, 80, 255),
        (80, 180, 255),
    ]
    return colors[index % len(colors)]


def build_checker_cell_lookup(checker_detection):
    if checker_detection is None:
        return {}

    return {(c.i, c.j): c for c in checker_detection.cells}


def patch_cell(checker_cell_lookup, patch_row, patch_col):
    return checker_cell_lookup.get((patch_col, patch_row))


def patch_boundary_points(patch, checker_cell_lookup):
    k = patch.k
    r0 = patch.row
    c0 = patch.col

    tl = patch_cell(checker_cell_lookup, r0, c0)
    tr = patch_cell(checker_cell_lookup, r0, c0 + k - 1)
    br = patch_cell(checker_cell_lookup, r0 + k - 1, c0 + k - 1)
    bl = patch_cell(checker_cell_lookup, r0 + k - 1, c0)

    if tl is None or tr is None or br is None or bl is None:
        return None

    return np.array(
        [
            [tl.corner_uv[0].x, tl.corner_uv[0].y],
            [tr.corner_uv[1].x, tr.corner_uv[1].y],
            [br.corner_uv[2].x, br.corner_uv[2].y],
            [bl.corner_uv[3].x, bl.corner_uv[3].y],
        ],
        dtype=np.int32,
    )


def draw_camera_corners(vis, checker_detection):
    if checker_detection is None:
        return

    for corner in checker_detection.corners:
        x, y = pxy(corner.uv)

        cv2.circle(
            vis,
            (x, y),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            vis,
            (x, y),
            6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def draw_camera_patches(vis, checker_detection, decoded_patches):
    if checker_detection is None:
        return

    lookup = build_checker_cell_lookup(checker_detection)

    for idx, decoded in enumerate(decoded_patches):
        if not decoded.valid:
            continue

        pts = patch_boundary_points(decoded.local, lookup)
        if pts is None:
            continue

        color = patch_color(idx)

        cv2.polylines(
            vis,
            [pts],
            True,
            color,
            2,
            cv2.LINE_AA,
        )

        center = pts.mean(axis=0)

        put_text(
            vis,
            f"{idx}",
            (int(center[0]), int(center[1])),
            color,
            0.6,
            2,
        )


def draw_correspondence_corners(vis, correspondences):
    for corr in correspondences:
        x, y = pxy(corr.uv)

        cv2.circle(
            vis,
            (x, y),
            7,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            vis,
            (x, y),
            10,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        put_text(
            vis,
            f"{corr.global_row},{corr.global_col}",
            (x + 8, y - 8),
            (0, 255, 0),
            0.45,
            1,
        )


def draw_marker_patches(marker_vis, marker_field, decoded_patches):
    vis = marker_vis.copy()

    img_h, img_w = vis.shape[:2]

    field_w = marker_field.width()
    field_h = marker_field.height()

    cell_w = img_w / float(field_w)
    cell_h = img_h / float(field_h)

    for idx, decoded in enumerate(decoded_patches):
        if not decoded.valid:
            continue

        color = patch_color(idx)
        k = decoded.local.k

        x1 = int(round(decoded.global_col * cell_w))
        y1 = int(round(decoded.global_row * cell_h))

        x2 = int(round((decoded.global_col + k) * cell_w))
        y2 = int(round((decoded.global_row + k) * cell_h))

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            4,
            cv2.LINE_AA,
        )

        put_text(
            vis,
            f"{idx}",
            (x1 + 8, y1 + 24),
            color,
            0.7,
            2,
        )

    return vis


def draw_marker_correspondence_corners(marker_vis, marker_field, correspondences):
    img_h, img_w = marker_vis.shape[:2]

    field_w = marker_field.width()
    field_h = marker_field.height()

    cell_w = img_w / float(field_w)
    cell_h = img_h / float(field_h)

    for corr in correspondences:
        x = int(round(corr.global_col * cell_w))
        y = int(round(corr.global_row * cell_h))

        cv2.circle(
            marker_vis,
            (x, y),
            7,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            marker_vis,
            (x, y),
            10,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def save_snapshot_and_correspondences(vis, build_result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    img_path = output_dir / f"corr_builder_{ts}.png"
    txt_path = output_dir / f"corr_builder_{ts}.txt"

    ok = cv2.imwrite(str(img_path), vis)
    if not ok:
        raise RuntimeError(f"Could not save snapshot: {img_path}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("HydraMarker CorrespondenceBuilder result\n")
        f.write("=" * 130 + "\n\n")

        f.write(f"decoded_patches_used:       {build_result.decoded_patches_used}\n")
        f.write(f"assignments_total:         {build_result.assignments_total}\n")
        f.write(f"assignments_accepted:      {build_result.assignments_accepted}\n")
        f.write(f"assignments_conflicted:    {build_result.assignments_conflicted}\n")
        f.write(f"corners_without_geometry:  {build_result.corners_without_geometry}\n")
        f.write(f"correspondences:           {len(build_result.correspondences)}\n\n")

        f.write(
            f"{'idx':>4} | "
            f"{'local_row':>9} {'local_col':>9} | "
            f"{'global_row':>10} {'global_col':>10} | "
            f"{'u_px':>10} {'v_px':>10} | "
            f"{'X_mm':>12} {'Y_mm':>12} {'Z_mm':>12} | "
            f"{'votes':>5}\n"
        )
        f.write("-" * 130 + "\n")

        for idx, corr in enumerate(build_result.correspondences):
            f.write(
                f"{idx:4d} | "
                f"{corr.local_row:9d} {corr.local_col:9d} | "
                f"{corr.global_row:10d} {corr.global_col:10d} | "
                f"{corr.uv.x:10.3f} {corr.uv.y:10.3f} | "
                f"{corr.xyz_mm.x:12.3f} "
                f"{corr.xyz_mm.y:12.3f} "
                f"{corr.xyz_mm.z:12.3f} | "
                f"{corr.votes:5d}\n"
            )

    print(f"Saved snapshot:        {img_path}")
    print(f"Saved correspondences: {txt_path}")


def create_dot_detector():
    dot_cfg = hydramarker_cpp.DotDetectorConfig()

    dot_cfg.canonical_size = 80
    dot_cfg.canonical_margin_px = 4.0
    dot_cfg.min_dot_contrast = 8.0
    dot_cfg.strong_dot_contrast = 35.0

    dot_cfg.commit_threshold = 0.45
    dot_cfg.revoke_threshold = 0.20

    dot_cfg.uncertainty_low = 0.40
    dot_cfg.uncertainty_high = 0.55

    dot_cfg.warmup_frames = 1

    return hydramarker_cpp.DotDetector(dot_cfg)


def main():
    field_path = choose_file_qt(
        "Select HydraMarker .field file",
        "HydraMarker field (*.field);;All files (*.*)",
    )

    marker_json_path = choose_file_qt(
        "Select marker geometry .json file",
        "HydraMarker JSON (*.json);;All files (*.*)",
    )

    marker_img_path = choose_file_qt(
        "Select marker image",
        "Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)",
    )

    marker_field = hydramarker_cpp.MarkerField.loadFromFile(
        str(field_path)
    )

    marker_geometry = hydramarker_cpp.MarkerGeometry.load_from_json(
        str(marker_json_path)
    )

    marker_img = cv2.imread(
        str(marker_img_path),
        cv2.IMREAD_COLOR,
    )

    if marker_img is None:
        raise RuntimeError(
            f"Could not load marker image: {marker_img_path}"
        )

    patch_size = marker_field.patchSize()

    checker_detector = hydramarker_cpp.CheckerboardDetector()
    dot_detector = create_dot_detector()
    patch_extractor = hydramarker_cpp.PatchExtractor()

    decoder_cfg = hydramarker_cpp.PatchDecoderConfig()
    decoder_cfg.require_geometry_valid = True
    decoder_cfg.accept_ambiguous = False

    patch_decoder = hydramarker_cpp.PatchDecoder(decoder_cfg)

    corr_cfg = hydramarker_cpp.CorrespondenceBuilderConfig()
    corr_cfg.min_votes = 1
    corr_cfg.discard_conflicts = True
    corr_cfg.require_detection_stable = False

    corr_builder = hydramarker_cpp.CorrespondenceBuilder(corr_cfg)

    output_dir = (
        Path(__file__).resolve().parent
        / "corr_builder_snapshots"
    )

    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(
        rs.stream.color,
        1920,
        1080,
        rs.format.bgr8,
        30,
    )

    pipe.start(cfg)

    window_name = "HydraMarker CorrespondenceBuilder"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    last_vis = None
    last_build_result = None

    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(
                color_frame.get_data()
            )

            checker_detection = checker_detector.detect(img)

            if checker_detection is not None:
                dot_detection = dot_detector.detect(
                    img,
                    checker_detection,
                )

                patches = patch_extractor.extract(
                    dot_detection,
                    patch_size,
                )

                decoded_patches = patch_decoder.decode(
                    patches,
                    marker_field,
                )

                build_result = corr_builder.build(
                    checker_detection,
                    decoded_patches,
                    marker_geometry,
                )

            else:
                decoded_patches = []
                build_result = hydramarker_cpp.CorrespondenceBuildResult()

            left = img.copy()

            draw_camera_corners(
                left,
                checker_detection,
            )

            draw_camera_patches(
                left,
                checker_detection,
                decoded_patches,
            )

            draw_correspondence_corners(
                left,
                build_result.correspondences,
            )

            right = draw_marker_patches(
                marker_img,
                marker_field,
                decoded_patches,
            )

            draw_marker_correspondence_corners(
                right,
                marker_field,
                build_result.correspondences,
            )

            target_h = left.shape[0]
            scale = target_h / float(right.shape[0])

            right = cv2.resize(
                right,
                (
                    int(round(right.shape[1] * scale)),
                    target_h,
                ),
                interpolation=cv2.INTER_AREA,
            )

            gap = np.zeros(
                (target_h, 8, 3),
                dtype=np.uint8,
            )

            vis = np.hstack([
                left,
                gap,
                right,
            ])

            put_text(
                vis,
                f"decoded patches: {len(decoded_patches)}",
                (25, 40),
                (0, 255, 255),
                0.8,
                2,
            )

            put_text(
                vis,
                f"correspondences: {len(build_result.correspondences)}",
                (25, 75),
                (0, 255, 0),
                0.8,
                2,
            )

            put_text(
                vis,
                f"conflicts: {build_result.assignments_conflicted}",
                (25, 110),
                (0, 180, 255),
                0.7,
                2,
            )

            put_text(
                vis,
                "SPACE save PNG+TXT | ESC quit",
                (25, 145),
                (0, 255, 255),
                0.6,
                1,
            )

            last_vis = vis
            last_build_result = build_result

            cv2.imshow(
                window_name,
                last_vis,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key == 32 and last_vis is not None and last_build_result is not None:
                save_snapshot_and_correspondences(
                    last_vis,
                    last_build_result,
                    output_dir,
                )

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()