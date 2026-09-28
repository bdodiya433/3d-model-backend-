"""
camera_alignment.py - Camera calibration and 3D projection module.
Estimates camera intrinsics, extrinsics (rotation and translation), and establishes
a common metric 3D world coordinate system from 6 multi-view photographs.
Completely generic and product-independent.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger("backend.camera_alignment")


class CameraCalibration:
    """
    Represents a single calibrated camera with intrinsics K, extrinsics (R, T),
    distortion, and projection functions.
    """

    def __init__(
        self,
        view_name: str,
        image_shape: Tuple[int, int],  # (H, W)
        K: np.ndarray,                 # 3x3
        R: np.ndarray,                 # 3x3 (world to camera rotation)
        T: np.ndarray,                 # 3x1 (world to camera translation)
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        self.view_name = view_name
        self.image_shape = image_shape
        self.H, self.W = image_shape
        self.K = K.astype(np.float64)
        self.R = R.astype(np.float64)
        self.T = T.reshape(3, 1).astype(np.float64)
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

        # Camera optical center in world coordinates: C = -R^T * T
        self.center_world = (-self.R.T @ self.T).ravel()

        # Camera viewing direction in world space (optical axis: Z_cam = (0, 0, 1))
        self.view_dir_world = (self.R.T @ np.array([[0.0], [0.0], [1.0]])).ravel()
        self.view_dir_world /= np.linalg.norm(self.view_dir_world)

        # Projection matrix P = K * [R | T] (3x4)
        self.P = self.K @ np.hstack([self.R, self.T])

    @property
    def shift_x(self) -> float:
        """Blender normalized horizontal sensor shift."""
        return -float(self.K[0, 2] - self.W / 2.0) / float(self.W)

    @property
    def shift_y(self) -> float:
        """Blender normalized vertical sensor shift."""
        return float(self.K[1, 2] - self.H / 2.0) / float(self.W)

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects an array of 3D world points (N, 3) into camera image pixel coordinates.
        Returns:
            uv: (N, 2) image coordinates (x, y) in pixels [0, W) x [0, H)
            depth: (N,) depth along camera optical axis (positive = in front of camera)
            in_bounds: (N,) boolean mask indicating points falling inside the image frame
        """
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        N = len(pts)
        pts_cam = (self.R @ pts.T + self.T).T  # (N, 3)
        depth = pts_cam[:, 2]

        # Points behind camera
        valid_depth = depth > 1e-4

        safe_depth = np.where(valid_depth, depth, 1.0)
        u_hom = (pts_cam[:, 0] * self.K[0, 0] / safe_depth) + self.K[0, 2]
        v_hom = (pts_cam[:, 1] * self.K[1, 1] / safe_depth) + self.K[1, 2]

        in_bounds = (
            valid_depth
            & (u_hom >= 0)
            & (u_hom < self.W)
            & (v_hom >= 0)
            & (v_hom < self.H)
        )

        uv = np.stack([u_hom, v_hom], axis=-1)
        return uv, depth, in_bounds


class CameraAlignment:
    """
    Estimates and optimizes camera poses (extrinsics) and intrinsics for all 6 views
    using cross-view silhouette consistency in a unified 3D metric coordinate system.
    """

    def __init__(self):
        self.calibrations: Dict[str, CameraCalibration] = {}
        self.object_bounds: Dict[str, float] = {}

    def calibrate_views(
        self,
        segmented_views: Dict[str, Dict[str, Any]],
        obj_diameter: float = 1.0,
        obj_height: float = 1.5,
    ) -> Dict[str, CameraCalibration]:
        """
        Calibrates cameras for all provided segmented views based on silhouette bounding
        extents and multi-view geometric consistency.
        """
        self.calibrations.clear()

        # 1. Establish unified 3D bounding box centered at (0, 0, 0)
        front_data = segmented_views.get("front", next(iter(segmented_views.values())))
        img_h, img_w = front_data["mask"].shape[:2]

        # Height span across lateral views (front, back, left, right)
        lateral_keys = [k for k in ["front", "back", "left", "right"] if k in segmented_views]
        height_px_list = [segmented_views[k]["height"] for k in lateral_keys]
        max_h_px = max(height_px_list) if height_px_list else img_h * 0.75

        # Widths
        w_px_front = segmented_views.get("front", {}).get("width", img_w * 0.45)
        w_px_left = segmented_views.get("left", {}).get("width", img_w * 0.45)

        # Unified aspect ratios in 3D metric units (Z is vertical)
        half_z = 0.5 * obj_height
        half_x = 0.5 * obj_diameter * (w_px_front / max(max_h_px, 1.0)) * (obj_height / obj_diameter)
        half_y = 0.5 * obj_diameter * (w_px_left / max(max_h_px, 1.0)) * (obj_height / obj_diameter)

        self.object_bounds = {
            "half_x": float(half_x),
            "half_y": float(half_y),
            "half_z": float(half_z),
        }

        # 2. Camera Intrinsics
        # Standard product focal length (~45 deg vertical FOV)
        focal_len = float(img_h * 1.35)

        # Canonical rotation matrices for 6 cardinal orientations:
        # World convention: +Z = Up, +Y = Back, -Y = Front, +X = Right, -X = Left.
        canonical_rotations = {
            "front": np.array([
                [1.0,  0.0,  0.0],
                [0.0,  0.0, -1.0],
                [0.0,  1.0,  0.0],
            ]),
            "back": np.array([
                [-1.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0],
                [ 0.0, -1.0,  0.0],
            ]),
            "left": np.array([
                [ 0.0, -1.0,  0.0],
                [ 0.0,  0.0, -1.0],
                [ 1.0,  0.0,  0.0],
            ]),
            "right": np.array([
                [ 0.0,  1.0,  0.0],
                [ 0.0,  0.0, -1.0],
                [-1.0,  0.0,  0.0],
            ]),
            "top": np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0],
            ]),
            "bottom": np.array([
                [-1.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0],
                [ 0.0,  0.0,  1.0],
            ]),
        }

        # 3. Optimize distance & center for each view based on mask bounding box
        for view_name, vdata in segmented_views.items():
            mask = vdata["mask"]
            H, W = mask.shape[:2]
            bbox = vdata["bbox"]  # (xmin, ymin, xmax, ymax)
            bw = vdata["width"]
            bh = vdata["height"]

            cx_img = (bbox[0] + bbox[2]) / 2.0
            cy_img = (bbox[1] + bbox[3]) / 2.0

            K = np.array([
                [focal_len, 0.0, cx_img],
                [0.0, focal_len, cy_img],
                [0.0, 0.0, 1.0],
            ])

            R = canonical_rotations.get(view_name, canonical_rotations["front"]).copy()

            # Distance derived from silhouette pixel coverage
            if view_name in ("top", "bottom"):
                obj_extent = max(half_x, half_y) * 2.0
                dist = (focal_len * obj_extent) / max(bw, bh, 1.0)
            else:
                obj_extent = half_z * 2.0
                dist = (focal_len * obj_extent) / max(bh, 1.0)

            # Translation vector T: optical center at distance along viewing axis
            T = np.array([[0.0], [0.0], [dist]])

            calib = CameraCalibration(
                view_name=view_name,
                image_shape=(H, W),
                K=K,
                R=R,
                T=T,
            )
            self.calibrations[view_name] = calib
            logger.info(
                f"Calibrated view '{view_name}': optical_center={calib.center_world.round(2)}, "
                f"distance={dist:.2f}, focal_len={focal_len:.1f}px"
            )

        # 4. Multi-view Silhouette Pose Refinement:
        # Refine non-lateral views (e.g. top, bottom) against the provisional 3D visual hull
        lateral_views = [k for k in ["front", "back", "left", "right"] if k in self.calibrations]
        non_lateral_views = [k for k in ["top", "bottom"] if k in segmented_views]

        if len(lateral_views) >= 2 and non_lateral_views:
            N_fast = 60
            margin = 1.15
            x_vals = np.linspace(-half_x * margin, half_x * margin, N_fast, dtype=np.float32)
            y_vals = np.linspace(-half_y * margin, half_y * margin, N_fast, dtype=np.float32)
            z_vals = np.linspace(-half_z * margin, half_z * margin, N_fast, dtype=np.float32)
            X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
            pts_fast = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

            inside_lat = np.ones(len(pts_fast), dtype=bool)
            for lv in lateral_views:
                cal = self.calibrations[lv]
                uv, depth, in_b = cal.project(pts_fast)
                m = (segmented_views[lv]["mask"] > 128)
                H_lv, W_lv = m.shape[:2]
                u_c = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, W_lv - 1)
                v_c = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, H_lv - 1)
                inside_lat &= (in_b & m[v_c, u_c])

            ref_pts = pts_fast[inside_lat]

            if len(ref_pts) > 100:
                for vname in non_lateral_views:
                    gt_mask = (segmented_views[vname]["mask"] > 128).astype(np.uint8)
                    H_v, W_v = gt_mask.shape[:2]
                    base_calib = self.calibrations[vname]
                    base_R = canonical_rotations[vname]
                    base_dist = float(base_calib.T[2, 0])
                    K_v = base_calib.K

                    best_iou = -1.0
                    best_R = base_R
                    best_dist = base_dist

                    angle_range = range(-75, 80, 5)
                    dist_multipliers = [0.85, 1.0, 1.15, 1.3, 1.5]

                    for theta_deg in angle_range:
                        theta = np.radians(theta_deg)
                        R_pitch = np.array([
                            [1.0, 0.0, 0.0],
                            [0.0, np.cos(theta), -np.sin(theta)],
                            [0.0, np.sin(theta),  np.cos(theta)]
                        ])
                        R_cand = R_pitch @ base_R

                        for dm in dist_multipliers:
                            cand_dist = base_dist * dm
                            T_cand = np.array([[0.0], [0.0], [cand_dist]])
                            test_c = CameraCalibration(vname, (H_v, W_v), K_v, R_cand, T_cand)
                            uv, depth, in_b = test_c.project(ref_pts)
                            valid = in_b & (depth > 0)
                            if not np.any(valid):
                                continue
                            u_p = np.clip(np.round(uv[valid, 0]).astype(np.int32), 0, W_v - 1)
                            v_p = np.clip(np.round(uv[valid, 1]).astype(np.int32), 0, H_v - 1)
                            proj_m = np.zeros((H_v, W_v), dtype=np.uint8)
                            proj_m[v_p, u_p] = 1
                            kernel_sz = max(5, int(W_v / N_fast * 1.8))
                            if kernel_sz % 2 == 0:
                                kernel_sz += 1
                            proj_m = cv2.dilate(proj_m, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sz, kernel_sz)))

                            inter = np.logical_and(proj_m, gt_mask).sum()
                            union = np.logical_or(proj_m, gt_mask).sum()
                            iou = inter / max(union, 1)

                            if iou > best_iou:
                                best_iou = iou
                                best_R = R_cand
                                best_dist = cand_dist

                    self.calibrations[vname] = CameraCalibration(
                        view_name=vname,
                        image_shape=(H_v, W_v),
                        K=K_v,
                        R=best_R,
                        T=np.array([[0.0], [0.0], [best_dist]]),
                    )
                    logger.info(
                        f"Pose-refined view '{vname}': IoU={best_iou:.3f}, dist={best_dist:.2f}, "
                        f"optical_center={self.calibrations[vname].center_world.round(2)}"
                    )

        return self.calibrations


# Backward-compatibility alias
CameraCalibrator = CameraAlignment
