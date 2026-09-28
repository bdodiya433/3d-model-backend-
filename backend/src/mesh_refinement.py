"""
mesh_refinement.py - Multi-View 3D Geometry Refinement Module.

Refines an initial Visual Hull / Marching Cubes mesh using continuous multi-view
silhouette ray constraints, view-dependent grazing weights, adaptive tangential
Laplacian regularization, geometric feature preservation, and multi-view consensus.

Completely product-agnostic and template-free. Works for any arbitrary physical object.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt
import trimesh

logger = logging.getLogger("backend.mesh_refinement")


class MeshRefiner:
    """
    Refines an initial 3D mesh using multi-view silhouette consensus from 6 calibrated views.
    """

    def __init__(
        self,
        num_iterations: int = 8,
        step_size: float = 0.22,
        lambda_smooth: float = 0.18,
        lambda_shape: float = 0.06,
        lambda_feature: float = 0.25,
        convergence_tol: float = 1e-5,
        max_displacement_per_iter: float = 0.02,
    ):
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.lambda_smooth = lambda_smooth
        self.lambda_shape = lambda_shape
        self.lambda_feature = lambda_feature
        self.convergence_tol = convergence_tol
        self.max_displacement_per_iter = max_displacement_per_iter

    @staticmethod
    def sample_sdf_bilinear(sdf_2d: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Samples a 2D Signed Distance Field at continuous floating-point coordinates (u, v)
        using bilinear interpolation.
        Points outside the image frame evaluate to -100.0 (strictly outside).
        """
        H, W = sdf_2d.shape[:2]
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)

        in_bounds = (u >= 0.0) & (u <= float(W - 1)) & (v >= 0.0) & (v <= float(H - 1))

        # Continuous bilinear sampling via map_coordinates (order=1 is bilinear)
        sampled = map_coordinates(sdf_2d, [v, u], order=1, mode="constant", cval=-100.0)
        sampled[~in_bounds] = -100.0
        return sampled

    @staticmethod
    def compute_vertex_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Estimates local surface curvature / feature sharpness per vertex using normal
        variation across 1-ring neighbors. Returns values in [0, 1].
        """
        n_verts = len(mesh.vertices)
        curvature = np.zeros(n_verts, dtype=np.float32)
        normals = mesh.vertex_normals
        neighbors = mesh.vertex_neighbors

        for i in range(n_verts):
            adj = neighbors[i]
            if len(adj) > 0:
                diffs = np.linalg.norm(normals[adj] - normals[i], axis=1)
                curvature[i] = float(np.mean(diffs))

        max_c = curvature.max()
        if max_c > 1e-6:
            curvature /= max_c
        return np.clip(curvature, 0.0, 1.0)

    @staticmethod
    def compute_tangential_laplacian(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Computes the tangential component of the uniform Laplacian operator:
            L_tangent(V_i) = L(V_i) - (L(V_i) . n_i) * n_i
        Tangential smoothing distributes vertices evenly across the surface
        WITHOUT deflating, shrinking, or pinching the underlying 3D volume.
        """
        verts = mesh.vertices
        normals = mesh.vertex_normals
        neighbors = mesh.vertex_neighbors
        n_verts = len(verts)

        lap = np.zeros_like(verts)
        for i in range(n_verts):
            adj = neighbors[i]
            if len(adj) > 0:
                mean_adj = verts[adj].mean(axis=0)
                delta = mean_adj - verts[i]
                n_i = normals[i]
                norm_comp = np.dot(delta, n_i) * n_i
                lap[i] = delta - norm_comp

        return lap

    def refine_mesh(
        self,
        initial_mesh: trimesh.Trimesh,
        calibrations: Dict[str, Any],
        segmented_views: Dict[str, Dict[str, Any]],
        sdfs: Dict[str, np.ndarray],
        output_debug_dir: Optional[Path] = None,
        blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender",
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Executes iterative multi-view silhouette contour refinement under surface regularization.

        Returns:
            (refined_mesh, refinement_report)
        """
        mesh = copy.deepcopy(initial_mesh)
        initial_verts = mesh.vertices.copy()
        initial_extents = mesh.extents.copy()

        # Compute stable geometric features to preserve sharp corners & thin structures
        feature_mask = self.compute_vertex_curvature(mesh)
        high_curv_count = int(np.sum(feature_mask > 0.40))
        logger.info(
            f"Initialized MeshRefiner: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces, "
            f"high-curvature feature vertices={high_curv_count:,}"
        )

        history: List[Dict[str, Any]] = []
        best_mesh = copy.deepcopy(mesh)
        best_objective = float("inf")
        best_iteration = 0

        # Pre-extract image sizes and camera positions
        cam_data = {}
        for vname in ["front", "back", "left", "right", "top", "bottom"]:
            if vname in calibrations and vname in sdfs:
                cal = calibrations[vname]
                cam_data[vname] = {
                    "calib": cal,
                    "sdf": sdfs[vname],
                    "center": cal.center_world,
                    "view_dir": cal.view_dir_world,
                    "W": cal.W,
                    "H": cal.H,
                }

        # Conversion scale: 3D metric units per pixel for displacement
        ref_h_px = max(segmented_views["front"]["height"], 100.0)
        units_per_px = float(initial_extents[2] / ref_h_px) if initial_extents[2] > 0 else 0.002

        if output_debug_dir:
            output_debug_dir = Path(output_debug_dir).resolve()
            output_debug_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(1, self.num_iterations + 1):
            verts = mesh.vertices
            normals = mesh.vertex_normals
            n_verts = len(verts)

            disp_accumulator = np.zeros_like(verts)
            weight_accumulator = np.zeros(n_verts, dtype=np.float32)

            total_sil_error = 0.0
            sil_sample_count = 0

            # 1. Project vertices into each calibrated camera
            per_camera_disp: Dict[str, np.ndarray] = {}
            for vname, cdata in cam_data.items():
                cal = cdata["calib"]
                sdf_2d = cdata["sdf"]
                cam_center = cdata["center"]

                uv, depth, in_bounds = cal.project(verts)
                valid_proj = in_bounds & (depth > 0.01)

                # Vector from camera optical center to each vertex
                view_vecs = verts - cam_center
                view_dists = np.linalg.norm(view_vecs, axis=-1, keepdims=True)
                view_dirs = view_vecs / np.maximum(view_dists, 1e-6)

                # Viewing angle: cos(theta) = normal . (-view_dir)
                cos_theta = np.sum(normals * (-view_dirs), axis=-1)

                # Grazing/contour weighting:
                # Vertices near silhouette horizon (cos(theta) ~ 0) strongly constrain contour!
                grazing_weight = np.exp(-((cos_theta) ** 2) / (2.0 * (0.28 ** 2)))
                visible_mask = valid_proj & (cos_theta > -0.15)

                if not np.any(visible_mask):
                    continue

                u_vals = uv[visible_mask, 0].astype(np.float32)
                v_vals = uv[visible_mask, 1].astype(np.float32)

                # Continuous bilinear sub-pixel SDF sampling
                dist_px = self.sample_sdf_bilinear(sdf_2d, u_vals, v_vals)

                # Metrics tracking
                valid_samples = dist_px > -90.0
                total_sil_error += float(np.sum(np.abs(dist_px[valid_samples])))
                sil_sample_count += int(np.sum(valid_samples))

                # Near silhouette contour:
                # - If outside silhouette (dist_px < 0): push inward along surface normal.
                # - If inside near contour (dist_px > 0 and grazing): pull outward along surface normal.
                # - Interior points (> 15px inside) have zero silhouette displacement.
                near_contour = (dist_px >= -25.0) & (dist_px <= 15.0)
                active_idx = np.where(visible_mask)[0][near_contour]
                active_dist_px = dist_px[near_contour]
                active_weights = grazing_weight[active_idx]

                if len(active_idx) == 0:
                    continue

                # 3D displacement along surface normal
                disp_3d = (active_dist_px[:, None] * units_per_px) * normals[active_idx]

                disp_accumulator[active_idx] += active_weights[:, None] * disp_3d
                weight_accumulator[active_idx] += active_weights
                per_camera_disp[vname] = (active_idx, disp_3d)

            # 2. Multi-View Consensus Displacement
            consensus_disp = np.zeros_like(verts)
            has_weight = weight_accumulator > 1e-5
            consensus_disp[has_weight] = disp_accumulator[has_weight] / weight_accumulator[has_weight, None]

            # 3. Consensus Conflict Mitigation:
            # If moving a vertex along consensus_disp worsens its projection into another view
            # (where it is already outside the silhouette), damp the displacement.
            for vname, cdata in cam_data.items():
                cal = cdata["calib"]
                sdf_2d = cdata["sdf"]
                test_pos = verts + consensus_disp * self.step_size
                uv_test, _, in_b = cal.project(test_pos)
                d_test = self.sample_sdf_bilinear(sdf_2d, uv_test[:, 0], uv_test[:, 1])
                # If test position is outside silhouette and got worse
                worse = (d_test < -1.0)
                consensus_disp[worse] *= 0.5

            # 4. Shape preservation constraint: prevent drifting excessively from initial visual hull
            shape_disp = (initial_verts - verts) * self.lambda_shape

            # 5. Adaptive Tangential Laplacian Regularization (smooths surface without shrinking)
            tangent_lap = self.compute_tangential_laplacian(mesh)
            smooth_weight = self.lambda_smooth * (1.0 - 0.70 * feature_mask[:, None])
            reg_disp = smooth_weight * tangent_lap

            # 6. Combined Total Displacement Update
            total_disp = self.step_size * (consensus_disp + shape_disp + reg_disp)

            # Clamp maximum displacement per iteration to maintain physical stability
            disp_norm = np.linalg.norm(total_disp, axis=-1, keepdims=True)
            scale = np.minimum(1.0, self.max_displacement_per_iter / np.maximum(disp_norm, 1e-6))
            total_disp *= scale

            # Update vertex positions
            new_verts = verts + total_disp

            # Stability check: no NaNs or Infs
            if np.any(np.isnan(new_verts)) or np.any(np.isinf(new_verts)):
                logger.warning(f"Refinement iteration {iteration}: Encountered NaN/Inf, stopping iteration.")
                break

            mesh.vertices = new_verts
            mesh.fix_normals()
            non_degen = mesh.nondegenerate_faces()
            if len(non_degen) < len(mesh.faces):
                mesh.update_faces(non_degen)

            # Global objective evaluation
            mean_sil_err = total_sil_error / max(sil_sample_count, 1)
            mean_step = float(np.mean(disp_norm))
            max_step = float(np.max(disp_norm))
            modified_count = int(np.sum(disp_norm > 1e-4))
            smoothness_cost = float(np.mean(np.linalg.norm(tangent_lap, axis=-1)))
            obj_val = mean_sil_err + self.lambda_smooth * smoothness_cost * 100.0

            iter_info = {
                "iteration": iteration,
                "mean_silhouette_error_px": round(mean_sil_err, 3),
                "mean_displacement_3d": round(mean_step, 5),
                "max_displacement_3d": round(max_step, 5),
                "modified_vertices": modified_count,
                "smoothness_cost": round(smoothness_cost, 5),
                "objective_value": round(obj_val, 4),
                "num_vertices": len(mesh.vertices),
            }
            history.append(iter_info)
            logger.info(
                f"Refinement iter [{iteration:02d}/{self.num_iterations}]: SilErr={mean_sil_err:.2f}px, "
                f"AvgDisp={mean_step:.5f}, MaxDisp={max_step:.4f}, ModVerts={modified_count:,}, Obj={obj_val:.3f}"
            )

            # Save debug iteration folder if requested
            if output_debug_dir:
                iter_dir = output_debug_dir / f"refinement_iteration_{iteration:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                iter_obj = iter_dir / f"mesh_iter_{iteration:02d}.obj"
                mesh.export(str(iter_obj))

                # Render intermediate debug silhouettes and overlays
                self._save_iteration_debug_artifacts(
                    iter_mesh_path=iter_obj,
                    iter_dir=iter_dir,
                    calibrations=calibrations,
                    segmented_views=segmented_views,
                    blender_bin=blender_bin,
                )

                (iter_dir / "metrics.json").write_text(json.dumps(iter_info, indent=2))

            if obj_val < best_objective:
                best_objective = obj_val
                best_mesh = copy.deepcopy(mesh)
                best_iteration = iteration

            # Convergence check on active vertices
            active_mask = disp_norm[:, 0] > 1e-6
            mean_active_step = float(np.mean(disp_norm[active_mask])) if np.any(active_mask) else 0.0
            if (mean_active_step < self.convergence_tol and iteration >= 3) or modified_count == 0:
                logger.info(f"Refinement converged at iteration {iteration} (active mean step={mean_active_step:.6f} < {self.convergence_tol})")
                break

        final_mesh = best_mesh
        logger.info(
            f"Mesh refinement complete: Best iteration #{best_iteration} (Objective={best_objective:.3f})."
        )

        report = {
            "initial_vertices": len(initial_verts),
            "final_vertices": len(final_mesh.vertices),
            "best_iteration": best_iteration,
            "best_objective": round(best_objective, 4),
            "history": history,
        }

        if output_debug_dir:
            metrics_file = output_debug_dir / "refinement_metrics.json"
            metrics_file.write_text(json.dumps(report, indent=2))
            logger.info(f"Saved refinement metrics to {metrics_file}")

        return final_mesh, report

    def _save_iteration_debug_artifacts(
        self,
        iter_mesh_path: Path,
        iter_dir: Path,
        calibrations: Dict[str, Any],
        segmented_views: Dict[str, Dict[str, Any]],
        blender_bin: str,
    ):
        """
        Renders debug silhouettes and color-coded overlays for an intermediate iteration.
        """
        from backend.src.geometry_validation import GeometryValidator
        validator = GeometryValidator(blender_bin=blender_bin)
        try:
            validator.validate_geometry(
                mesh_path=iter_mesh_path,
                segmented_views=segmented_views,
                calibrations=calibrations,
                output_dir=iter_dir,
            )
        except Exception as e:
            logger.warning(f"Could not render debug iteration validation: {e}")
