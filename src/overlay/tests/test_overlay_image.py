import sys
import numpy as np
import cv2

from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


def show_image_bgr(img_bgr, title="Overlay"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape

    qimg = QImage(
        img_rgb.data,
        w,
        h,
        ch * w,
        QImage.Format_RGB888
    ).copy()

    label = QLabel()
    label.setWindowTitle(title)
    label.setPixmap(QPixmap.fromImage(qimg))
    label.resize(w, h)
    label.show()

    return label


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select overlay NPZ file",
        "",
        "NPZ files (*.npz)"
    )

    if not file_path:
        print("No file selected.")
        return

    data = np.load(file_path, allow_pickle=True)

    print("Available keys:")
    for k in data.keys():
        arr = data[k]
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        print(f"  {k}: shape={shape}, dtype={dtype}")

    if "overlay_bgr" not in data:
        raise KeyError("Key 'overlay_bgr' not found in selected NPZ file.")

    overlay_bgr = data["overlay_bgr"]

    viewer = show_image_bgr(overlay_bgr, title="Saved Overlay Image")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()