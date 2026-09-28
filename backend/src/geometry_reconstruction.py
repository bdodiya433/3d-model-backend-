"""
geometry_reconstruction.py - Multi-view Image-Driven 3D Geometric Reconstruction.

Reconstructs the 3D mesh directly from the actual shape visible in the 6 input photographs
(front, back, left, right, top, bottom) using:
  1. High-resolution continuous Signed Distance Field (SDF) space-carving with bilinear sub-pixel sampling
  2. Empty space & cavity preservation
  3. Marching Cubes surface extraction (Initial Geometry)
  4. Multi-view mesh refinement via MeshRefiner (Iterative multi-view contour consensus & adaptive regularization)
  5. Post-refinement Quadric Error Metric (QEM) decimation

DOES NOT use any static templates, fixed primitives, or hardcoded product dimensions.
Works for ANY physical object: cans, bottles, shoes, toys, boxes, tools, furniture, irregular shapes.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage.measure import marching_cubes
import trimesh

from backend.src.mesh_refinement import MeshRefiner

logger = logging.getLogger("backend.geometry_reconstruction")


class GeometryReconstructor:
    """
    Reconstructs 3D mesh geometry directly from 6 multi-view silhouettes.
    """

    def __init__(self, blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender"):
        self.blender_bin = blender_bin

    def reconstruct_from_views(
        self,
        segmented_views: Dict[str, Dict[str, Any]],
        output_dir: Path,
        calibrations: Optional[Dict[str, Any]] = None,
        grid_resolution: int = 200,
        bounds_padding: float = 1.06,
        target_faces: int = 14000,
        refinement_iterations: int = 8,
    ) -> Dict[str, Any]:
        """
        Reconstructs 3D geometry from 6 canonical segmented views:
        front, back, left, right, top, bottom.

        Sequence:
          1. Continuous Bilinear SDF Space Carving (N = grid_resolution, default 200)
          2. Marching Cubes -> Initial Mesh
          3. Multi-View Silhouette Mesh Refinement (MeshRefiner)
          4. Post-Refinement QEM Decimation
          5. Final Export (OBJ, GLB, BLEND)

        Returns:
            {
                "initial_obj": Path to initial Marching Cubes mesh,
                "obj": Path to final refined & decimated OBJ,
                "glb": Path to final GLB,
                "blend": Path to Blender scene with reconstructed mesh,
                "dimensions": [dx, dy, dz] in world units,
                "refinement_report": Dict of refinement history and metrics,
            }
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        initial_obj_path = output_dir / "reconstructed_initial.obj"
        refined_raw_obj_path = output_dir / "reconstructed_refined_pre_decimate.obj"
        final_obj_path = output_dir / "reconstructed_product.obj"
        final_glb_path = output_dir / "reconstructed_product.glb"
        final_blend_path = output_dir / "reconstructed_product.blend"
        debug_refinement_dir = output_dir / "refinement_debug"

        # 1. Extract clean binary masks from all 6 views
        masks: Dict[str, np.ndarray] = {}
        for view_name in ["front", "back", "left", "right", "top", "bottom"]:
            if view_name not in segmented_views:
                raise ValueError(f"Missing required view: {view_name}")
            raw_mask = segmented_views[view_name]["mask"]
            if raw_mask.ndim == 3:
                raw_mask = raw_mask[..., 0]
            bin_mask = (raw_mask > 128).astype(np.uint8)
            # Retain true silhouette geometry (Component 2: preserve cavities, openings, hollows)
            masks[view_name] = bin_mask

        # 2. Derive object extents and aspect ratios from multi-view silhouettes
        f_ys, f_xs = np.where(masks["front"] > 0)
        b_ys, b_xs = np.where(masks["back"] > 0)
        l_ys, l_xs = np.where(masks["left"] > 0)
        r_ys, r_xs = np.where(masks["right"] > 0)

        y_min = min(f_ys.min(), b_ys.min(), l_ys.min(), r_ys.min())
        y_max = max(f_ys.max(), b_ys.max(), l_ys.max(), r_ys.max())
        z_span_px = float(y_max - y_min + 1)
        half_h_z = z_span_px / 2.0

        max_w_x = max(f_xs.max() - f_xs.min() + 1, b_xs.max() - b_xs.min() + 1) / 2.0
        max_w_y = max(l_xs.max() - l_xs.min() + 1, r_xs.max() - r_xs.min() + 1) / 2.0

        rx = float(max_w_x / half_h_z)
        ry = float(max_w_y / half_h_z)
        rz = 1.0

        logger.info(
            f"Derived multi-view proportions: rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f} "
            f"(Pixel span: X={max_w_x*2:.0f}, Y={max_w_y*2:.0f}, Z={z_span_px:.0f})"
        )

        # 3. Compute continuous 2D Signed Distance Fields for all 6 views
        dts: Dict[str, np.ndarray] = {}
        for view_name, m in masks.items():
            dt_pos = distance_transform_edt(m > 0).astype(np.float32)
            dt_neg = distance_transform_edt(m == 0).astype(np.float32)
            dts[view_name] = dt_pos - dt_neg

        # 4. Multi-view 3D voxel space carving using calibrated camera projections
        if calibrations is None:
            from backend.src.camera_alignment import CameraAlignment
            aligner = CameraAlignment()
            calibrations = aligner.calibrate_views(segmented_views, obj_diameter=max(rx, ry) * 2.0, obj_height=2.0)

        N = int(grid_resolution)
        margin = float(bounds_padding)
        logger.info(f"Generating high-resolution continuous SDF grid: N={N}^3 ({N**3:,} voxels), margin={margin:.2f}")

        x_vals = np.linspace(-rx * margin, rx * margin, N, dtype=np.float32)
        y_vals = np.linspace(-ry * margin, ry * margin, N, dtype=np.float32)
        z_vals = np.linspace(-rz * margin, rz * margin, N, dtype=np.float32)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
        pts_3d = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        # Component 1: Continuous image-space sampling with bilinear interpolation
        # DO NOT use np.round(uv)! Use map_coordinates with order=1 at floating-point coordinates.
        sdf_views = []
        for vname in ["front", "back", "left", "right", "top", "bottom"]:
            calib = calibrations[vname]
            uv, depth, in_b = calib.project(pts_3d)

            u_f = uv[:, 0].astype(np.float32)
            v_f = uv[:, 1].astype(np.float32)

            # Bilinear interpolation at floating-point coordinates
            s = map_coordinates(dts[vname], [v_f, u_f], order=1, mode="constant", cval=-100.0).astype(np.float32)

            # Robust handling of invalid projections and clipping
            valid = in_b & (depth > 0.01)
            s[~valid] = -100.0
            sdf_views.append(s.reshape(X.shape))

        # Component 2: Visual hull is intersection across all 6 views (preserving empty space and cavities)
        sdf_3d = np.minimum.reduce(sdf_views)

        # 5. Extract 3D Surface Mesh via Marching Cubes (INITIAL GEOMETRY)
        logger.info("Extracting initial 3D surface with Marching Cubes...")
        dx = float(x_vals[1] - x_vals[0])
        dy = float(y_vals[1] - y_vals[0])
        dz = float(z_vals[1] - z_vals[0])

        verts, faces, normals, _ = marching_cubes(sdf_3d, level=0.0, spacing=(dx, dy, dz))
        verts += np.array([x_vals[0], y_vals[0], z_vals[0]], dtype=np.float32)
        faces = faces[:, [0, 2, 1]]  # Re-wind to CCW outward-facing normals

        initial_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        initial_mesh.export(str(initial_obj_path))
        logger.info(
            f"Initial Marching Cubes mesh extracted: {len(initial_mesh.vertices):,} vertices, "
            f"{len(initial_mesh.faces):,} faces -> {initial_obj_path}"
        )

        # 6. Component 3: Multi-View Mesh Refinement (MeshRefiner)
        # Refines the actual mesh vertices to match the 6 photographic contours
        logger.info(f"Starting Multi-View Mesh Refiner ({refinement_iterations} iterations)...")
        refiner = MeshRefiner(
            num_iterations=refinement_iterations,
            step_size=0.015,
            lambda_smooth=0.28,
            lambda_shape=0.10,
            lambda_feature=0.25,
            max_displacement_per_iter=0.035,
        )
        refined_mesh, ref_report = refiner.refine_mesh(
            initial_mesh=initial_mesh,
            calibrations=calibrations,
            segmented_views=segmented_views,
            sdfs=dts,
            output_debug_dir=debug_refinement_dir,
            blender_bin=self.blender_bin,
        )

        refined_mesh.export(str(refined_raw_obj_path))
        logger.info(
            f"Pre-decimation refined mesh saved: {len(refined_mesh.vertices):,} vertices, "
            f"{len(refined_mesh.faces):,} faces -> {refined_raw_obj_path}"
        )

        # 7. Component 13: Decimation STRICTLY AFTER Refinement
        # Use Quadric Error Metrics (QEM) in Blender to preserve silhouettes, sharp features & thin structures
        logger.info(f"Decimating refined mesh in Blender to ~{target_faces} clean manifold faces (QEM)...")
        blender_script = f'''
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"{refined_raw_obj_path}", up_axis="Z", forward_axis="Y")
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
obj.name = "ReconstructedProduct"

# Quadric decimation (COLLAPSE)
n_polys = len(obj.data.polygons)
if n_polys > {target_faces}:
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = float({target_faces}) / float(n_polys)
    bpy.ops.object.modifier_apply(modifier="Decimate")

# Smooth shading & consistent normals
for p in obj.data.polygons:
    p.use_smooth = True

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

# Component 14: Export geometry-only OBJ, GLB, and Blender scene
bpy.ops.wm.obj_export(filepath=r"{final_obj_path}", up_axis="Z", forward_axis="Y")
bpy.ops.export_scene.gltf(filepath=r"{final_glb_path}", export_format='GLB')
bpy.ops.wm.save_as_mainfile(filepath=r"{final_blend_path}")
print("GEOMETRY_RECONSTRUCTION_SUCCESS")
'''
        temp_py = output_dir / "decimate_temp.py"
        temp_py.write_text(blender_script)

        try:
            cmd = [self.blender_bin, "--background", "--python", str(temp_py)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if "GEOMETRY_RECONSTRUCTION_SUCCESS" not in res.stdout:
                raise RuntimeError(f"Blender geometry decimation failed:\n{res.stdout}\n{res.stderr}")
        finally:
            if temp_py.exists():
                temp_py.unlink()

        # Load final bounds & verify mesh quality
        final_mesh = trimesh.load(str(final_obj_path))
        extents = final_mesh.extents.tolist()

        # Component 12: Mesh Quality Check
        n_nan = int(np.isnan(final_mesh.vertices).sum())
        n_inf = int(np.isinf(final_mesh.vertices).sum())
        if n_nan > 0 or n_inf > 0:
            logger.error(f"Mesh quality check failed: NaNs={n_nan}, Infs={n_inf}")

        logger.info(
            f"Successfully reconstructed final 3D product geometry: "
            f"{len(final_mesh.vertices):,} vertices, {len(final_mesh.faces):,} faces, dimensions={extents}"
        )

        # Legacy backward compatibility aliases
        import shutil
        legacy_obj = output_dir / "reconstructed_raw.obj"
        legacy_glb = output_dir / "reconstructed_raw.glb"
        legacy_blend = output_dir / "reconstructed_mesh.blend"
        shutil.copy2(final_obj_path, legacy_obj)
        shutil.copy2(final_glb_path, legacy_glb)
        shutil.copy2(final_blend_path, legacy_blend)

        can_obj = output_dir / "can_raw.obj"
        can_glb = output_dir / "can_raw.glb"
        can_blend = output_dir / "can_mesh.blend"
        shutil.copy2(final_obj_path, can_obj)
        shutil.copy2(final_glb_path, can_glb)
        shutil.copy2(final_blend_path, can_blend)

        return {
            "initial_obj": initial_obj_path,
            "raw_obj": final_obj_path,
            "obj": final_obj_path,
            "glb": final_glb_path,
            "blend": final_blend_path,
            "dimensions": extents,
            "aspect_ratios": {"rx": rx, "ry": ry, "rz": rz},
            "refinement_report": ref_report,
        }
