from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from tracking.hydramarker.backend import cpp_impl as hm


class PosePropagationMixin:
    def _build_pose_propagated_detection(
        self,
        image_shape: Tuple[int, int],
    ):
        """Build a synthetic detection by projecting geometry corners."""
        if self.config.decode_only_mode or not self.config.enable_pose_propagation:
            return None

        if self.frame_index <= self._pose_propagation_block_until_frame:
            return None

        rvec = self.pose_tracker.rvec
        tvec = self.pose_tracker.tvec

        if rvec is None or tvec is None:
            return None

        if (
            self._last_good_reproj_px < 0.0
            or self._last_good_reproj_px > self.config.pose_propagation_max_reproj_px
        ):
            return None

        rows = self.geometry.corner_rows()
        cols = self.geometry.corner_cols()
        border = self.config.pose_propagation_border_px
        h, w = image_shape[0], image_shape[1]

        # Collect all valid 3D geometry corners.
        obj_pts = []
        row_col_list = []

        for gr in range(rows):
            for gc in range(cols):
                if not self.geometry.has_corner(gr, gc):
                    continue
                pt = self.geometry.corner_point(gr, gc)
                obj_pts.append([pt.x, pt.y, pt.z])
                row_col_list.append((gr, gc))

        if len(obj_pts) < self.config.min_points:
            return None

        obj_pts_np = np.array(obj_pts, dtype=np.float64).reshape(-1, 3)

        projected, _ = cv2.projectPoints(
            obj_pts_np,
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            self.K,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)

        # Build synthetic GridCorners for visible projections only.
        # global_row -> corner.j, global_col -> corner.i
        detection = hm.CheckerboardDetection()
        ij_to_uv: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for idx, (gr, gc) in enumerate(row_col_list):
            u, v = float(projected[idx, 0]), float(projected[idx, 1])

            if u < border or v < border or u >= w - border or v >= h - border:
                continue

            corner = hm.GridCorner()
            corner.j = gr   # row = vertikal = j
            corner.i = gc   # col = horizontal = i
            corner.uv = hm.Point2f()
            corner.uv.x = u
            corner.uv.y = v
            corner.visibility_score = 1.0

            detection.corners.append(corner)
            ij_to_uv[(gc, gr)] = (u, v)  # key: (i,j)

        if len(detection.corners) < self.config.min_points:
            return None

        # Build synthetic cells from projected corners.
        # Cell (i,j) hat Corners: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
        for ci, cj in list(ij_to_uv.keys()):
            if (ci+1, cj) not in ij_to_uv:
                continue
            if (ci+1, cj+1) not in ij_to_uv:
                continue
            if (ci, cj+1) not in ij_to_uv:
                continue

            cell = hm.GridCell()
            cell.i = ci
            cell.j = cj

            p00 = ij_to_uv[(ci,   cj)]
            p10 = ij_to_uv[(ci+1, cj)]
            p11 = ij_to_uv[(ci+1, cj+1)]
            p01 = ij_to_uv[(ci,   cj+1)]

            def make_pt(xy):
                p = hm.Point2f()
                p.x = xy[0]
                p.y = xy[1]
                return p

            cell.corner_uv = [make_pt(p00), make_pt(p10), make_pt(p11), make_pt(p01)]
            cell.center_uv = make_pt((
                (p00[0]+p10[0]+p11[0]+p01[0]) * 0.25,
                (p00[1]+p10[1]+p11[1]+p01[1]) * 0.25,
            ))

            detection.cells.append(cell)

        if len(detection.cells) == 0:
            return None

        if not self._detection_has_decodeable_cell_span(detection):
            return None

        detection.tracking = True
        detection.stable = True

        return detection

