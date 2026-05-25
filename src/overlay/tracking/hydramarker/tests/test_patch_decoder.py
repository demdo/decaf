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


def save_png(vis, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    path = output_dir / f"patch_decoder_{ts}.png"

    cv2.imwrite(str(path), vis)

    print(f"Saved screenshot: {path}")


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

    marker_img_path = choose_file_qt(
        "Select marker image",
        "Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)",
    )

    marker_field = hydramarker_cpp.MarkerField.loadFromFile(
        str(field_path)
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

    patch_decoder = hydramarker_cpp.PatchDecoder(
        decoder_cfg
    )

    output_dir = (
        Path(__file__).resolve().parent
        / "patch_decoder_snapshots"
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

    window_name = "HydraMarker PatchDecoder"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    paused = False

    last_vis = None

    try:
        while True:
            if not paused:
                frames = pipe.wait_for_frames()

                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                img = np.asanyarray(
                    color_frame.get_data()
                )

                checker_detection = checker_detector.detect(
                    img
                )

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

                else:
                    dot_detection = None
                    patches = []
                    decoded_patches = []

                left = img.copy()

                draw_camera_patches(
                    left,
                    checker_detection,
                    decoded_patches,
                )

                right = draw_marker_patches(
                    marker_img,
                    marker_field,
                    decoded_patches,
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
                    "SPACE save screenshot | p pause | ESC quit",
                    (25, 75),
                    (0, 255, 255),
                    0.6,
                    1,
                )

                last_vis = vis

            if last_vis is not None:
                cv2.imshow(
                    window_name,
                    last_vis,
                )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key == ord("p"):
                paused = not paused

            elif key == 32 and last_vis is not None:
                save_png(
                    last_vis,
                    output_dir,
                )

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()