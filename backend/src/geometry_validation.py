"""
geometry_validation.py - Silhouette-based 3D Geometry Validation Module.

Evaluates the reconstructed 3D mesh against the original multi-view silhouettes
and renders intermediate turntable angles for visual geometric verification.

Outputs:
  - 6 cardinal input masks: input_<view>_mask.png
  - 6 cardinal rendered silhouettes: render_<view>_silhouette.png
  - 6 cardinal color-coded overlays: <view>_overlay.png (Green=Match, Red=Excess, Blue=Missing)
  - 6 cardinal clay renders: clay_<view>.png
  - 5 intermediate turntable clay renders: intermediate_45.png, intermediate_135.png, ...
  - Full per-view metrics: IoU, Precision, Recall, BBox Width/Height Error, Contour Distance (px)
  - Global acceptance summary: mean IoU, min IoU, mean precision, min precision, mean recall, max contour error
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger("backend.geometry_validation")


class GeometryValidator:
    """
    Validates reconstructed 3D geometry against input segmentation silhouettes.
    """

    def __init__(self, blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender"):
        self.blender_bin = blender_bin

    def render_mesh_silhouettes(
        self,
        mesh_path: Path,
        output_dir: Path,
        calibrations: Dict[str, Any],
        resolution: int = 512,
    ) -> Dict[str, np.ndarray]:
        """
        Renders binary silhouettes of the reconstructed 3D mesh from all 6 camera viewpoints
        using the exact camera calibration parameters (focal length, sensor shifts, extrinsics).
        """
        mesh_path = Path(mesh_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        silhouettes: Dict[str, np.ndarray] = {}

        # Serialize calibrated camera extrinsics and parameters for Blender
        cam_configs = {}
        cv2blender = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
        for vname in ["front", "back", "left", "right", "top", "bottom"]:
            if vname in calibrations:
                c = calibrations[vname]
                R_b = (c.R.T @ cv2blender).tolist()
                loc_b = c.center_world.tolist()
                cam_configs[vname] = {
                    "matrix_rot": R_b,
                    "location": loc_b,
                    "focal_px": float(c.K[0, 0]),
                    "width": int(c.W),
                    "height": int(c.H),
                    "shift_x": float(c.shift_x),
                    "shift_y": float(c.shift_y),
                }

        config_json_path = output_dir / "calib_render_config.json"
        config_json_path.write_text(json.dumps(cam_configs))

        render_script = f'''
import bpy
import json
import math
from mathutils import Matrix, Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"{mesh_path}", up_axis="Z", forward_axis="Y")
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

with open(r"{config_json_path}", "r") as f:
    configs = json.load(f)

cam_data = bpy.data.cameras.new(name="Cam")
cam_obj = bpy.data.objects.new(name="Cam", object_data=cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'FLAT'
bpy.context.scene.display.shading.color_type = 'SINGLE'
bpy.context.scene.display.shading.single_color = (1.0, 1.0, 1.0)
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.resolution_x = {resolution}
bpy.context.scene.render.resolution_y = {resolution}

cam_data.sensor_fit = 'HORIZONTAL'
cam_data.sensor_width = 36.0

for view_name, cfg in configs.items():
    R_3x3 = Matrix(cfg["matrix_rot"])
    loc = Vector(cfg["location"])
    mat_world = Matrix.Translation(loc) @ R_3x3.to_4x4()
    cam_obj.matrix_world = mat_world
    
    # Set lens focal length & optical center shifts
    f_px = cfg["focal_px"]
    w_px = cfg["width"]
    cam_data.lens = (f_px / w_px) * 36.0
    cam_data.shift_x = cfg["shift_x"]
    cam_data.shift_y = cfg["shift_y"]
    
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = r"{output_dir}" + f"/sil_{{view_name}}.png"
    bpy.ops.render.render(write_still=True)

print("SILHOUETTE_RENDERS_SUCCESS")
'''
        temp_py = output_dir / "render_sil.py"
        temp_py.write_text(render_script)

        try:
            cmd = [self.blender_bin, "--background", "--python", str(temp_py)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if "SILHOUETTE_RENDERS_SUCCESS" not in res.stdout:
                logger.warning(f"Silhouette rendering output:\n{res.stdout}\n{res.stderr}")
        finally:
            if temp_py.exists():
                temp_py.unlink()

        for v in ["front", "back", "left", "right", "top", "bottom"]:
            p = output_dir / f"sil_{v}.png"
            if p.exists():
                img = Image.open(p).convert("RGBA")
                alpha = np.array(img)[..., 3]
                silhouettes[v] = (alpha > 10).astype(np.uint8)
                p.unlink()

        return silhouettes

    def render_intermediate_views(
        self,
        mesh_path: Path,
        output_dir: Path,
        resolution: int = 512,
    ) -> Dict[str, Path]:
        """
        Component 11: Render intermediate turntable views to detect geometric artifacts
        (pinching, bulges, flat faceting, twisted surfaces, discontinuities):
          1. FRONT -> RIGHT = 45°
          2. RIGHT -> BACK = 135°
          3. BACK -> LEFT = 225°
          4. LEFT -> FRONT = 315°
          5. Elevated 3/4 view = 45° azimuth, 30° elevation
        Also renders 6 cardinal clay views (front, back, left, right, top, bottom).
        Uses a clean neutral matte clay material.
        """
        mesh_path = Path(mesh_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        render_script = f'''
import bpy
import math
from mathutils import Vector, Matrix

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"{mesh_path}", up_axis="Z", forward_axis="Y")
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Smooth shading
for p in obj.data.polygons:
    p.use_smooth = True

# Neutral clay material
mat = bpy.data.materials.new(name="MatteClay")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.88, 0.88, 0.88, 1.0)
bsdf.inputs["Roughness"].default_value = 0.50
bsdf.inputs["Metallic"].default_value = 0.0
obj.data.materials.clear()
obj.data.materials.append(mat)

# Bounding box & center
bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
center = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
max_dim = max(obj.dimensions)
dist = max(max_dim * 1.6, 2.2)

# Camera & studio lighting
cam_data = bpy.data.cameras.new(name="Cam")
cam_obj = bpy.data.objects.new(name="Cam", object_data=cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'STUDIO'
bpy.context.scene.display.shading.studio_light = 'Default'
bpy.context.scene.display.shading.color_type = 'MATERIAL'
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.resolution_x = {resolution}
bpy.context.scene.render.resolution_y = {resolution}

cx, cy, cz = center.x, center.y, center.z

# Helper: look at object center
def set_cam_transform(cam, pos, target):
    cam.location = pos
    direction = (Vector(target) - Vector(pos)).normalized()
    # Rotation matrix from forward direction (-Z in Blender camera) and up vector (+Z)
    up = Vector((0.0, 0.0, 1.0))
    if abs(direction.z) > 0.999:
        up = Vector((0.0, 1.0, 0.0))
    right = direction.cross(up).normalized()
    cam_up = right.cross(direction).normalized()
    R = Matrix((right, cam_up, -direction)).transposed()
    cam.rotation_euler = R.to_euler()

# 1. Five intermediate turntable angles
intermediate_views = {{
    "intermediate_45": (45.0, 0.0),
    "intermediate_135": (135.0, 0.0),
    "intermediate_225": (225.0, 0.0),
    "intermediate_315": (315.0, 0.0),
    "intermediate_elevated_30": (45.0, 30.0),
}}

for vname, (az, el) in intermediate_views.items():
    az_rad = math.radians(az)
    el_rad = math.radians(el)
    r_horiz = dist * math.cos(el_rad)
    px = cx + r_horiz * math.sin(az_rad)
    py = cy - r_horiz * math.cos(az_rad)
    pz = cz + dist * math.sin(el_rad)
    set_cam_transform(cam_obj, (px, py, pz), (cx, cy, cz))
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = r"{output_dir}" + f"/{{vname}}.png"
    bpy.ops.render.render(write_still=True)

# 2. Six cardinal clay views
cardinal_views = {{
    "clay_front":  (cx, cy - dist, cz),
    "clay_back":   (cx, cy + dist, cz),
    "clay_left":   (cx - dist, cy, cz),
    "clay_right":  (cx + dist, cy, cz),
    "clay_top":    (cx, cy, cz + dist),
    "clay_bottom": (cx, cy, cz - dist),
}}

for vname, pos in cardinal_views.items():
    set_cam_transform(cam_obj, pos, (cx, cy, cz))
    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = r"{output_dir}" + f"/{{vname}}.png"
    bpy.ops.render.render(write_still=True)

print("INTERMEDIATE_RENDERS_SUCCESS")
'''
        temp_py = output_dir / "render_inter.py"
        temp_py.write_text(render_script)

        rendered_files = {}
        try:
            cmd = [self.blender_bin, "--background", "--python", str(temp_py)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if "INTERMEDIATE_RENDERS_SUCCESS" not in res.stdout:
                logger.warning(f"Intermediate rendering notes:\n{res.stdout}\n{res.stderr}")
        finally:
            if temp_py.exists():
                temp_py.unlink()

        for name in [
            "intermediate_45", "intermediate_135", "intermediate_225",
            "intermediate_315", "intermediate_elevated_30",
            "clay_front", "clay_back", "clay_left", "clay_right", "clay_top", "clay_bottom",
        ]:
            p = output_dir / f"{name}.png"
            if p.exists():
                rendered_files[name] = p

        return rendered_files

    def validate_geometry(
        self,
        mesh_path: Path,
        segmented_views: Dict[str, Dict[str, Any]],
        calibrations: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Validates the reconstructed 3D mesh against the original input silhouettes.
        Computes IoU, Precision, Recall, BBox Width/Height Errors, and Contour Distance per view.
        Saves intermediate input masks, rendered silhouettes, and color-coded overlay images.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        rendered_sils = self.render_mesh_silhouettes(
            mesh_path=mesh_path,
            output_dir=output_dir,
            calibrations=calibrations,
            resolution=512,
        )

        metrics: Dict[str, Any] = {
            "per_view": {},
            "mean_iou": 0.0,
            "min_iou": 0.0,
            "mean_precision": 0.0,
            "min_precision": 0.0,
            "mean_recall": 0.0,
            "max_contour_error": 0.0,
            "mean_bbox_width_error": 0.0,
            "mean_bbox_height_error": 0.0,
            "mean_contour_difference_px": 0.0,
            "overall_consistency": 0.0,
            "passed": True,
            "saved_artifacts": {},
        }

        ious = []
        precisions = []
        recalls = []
        bbox_w_errors = []
        bbox_h_errors = []
        contour_diffs = []

        for vname in ["front", "back", "left", "right", "top", "bottom"]:
            if vname not in segmented_views or vname not in rendered_sils:
                continue

            vdata = segmented_views[vname]

            # 1. Original input mask resized to 512x512
            orig_mask = vdata["mask"]
            if orig_mask.ndim == 3:
                orig_mask = orig_mask[..., 0]
            orig_bin = (orig_mask > 0).astype(np.uint8)
            orig_resized = cv2.resize(orig_bin, (512, 512), interpolation=cv2.INTER_NEAREST)

            rend_bin = rendered_sils[vname]

            # Component 9: Save required debug silhouettes: input_<view>_mask.png & render_<view>_silhouette.png
            input_mask_path = output_dir / f"input_{vname}_mask.png"
            render_sil_path = output_dir / f"render_{vname}_silhouette.png"
            cv2.imwrite(str(input_mask_path), orig_resized * 255)
            cv2.imwrite(str(render_sil_path), rend_bin * 255)

            # 2. Compute intersection and union
            inter = np.logical_and(orig_resized, rend_bin).sum()
            union = np.logical_or(orig_resized, rend_bin).sum()
            orig_area = float(orig_resized.sum())
            rend_area = float(rend_bin.sum())

            iou = float(inter / max(union, 1))
            precision = float(inter / max(rend_area, 1.0))
            recall = float(inter / max(orig_area, 1.0))

            # 3. Bounding-box width and height errors
            orig_ys, orig_xs = np.where(orig_resized > 0)
            rend_ys, rend_xs = np.where(rend_bin > 0)

            if len(orig_xs) > 0:
                orig_w = float(orig_xs.max() - orig_xs.min() + 1)
                orig_h = float(orig_ys.max() - orig_ys.min() + 1)
            else:
                orig_w, orig_h = 1.0, 1.0

            if len(rend_xs) > 0:
                rend_w = float(rend_xs.max() - rend_xs.min() + 1)
                rend_h = float(rend_ys.max() - rend_ys.min() + 1)
            else:
                rend_w, rend_h = 0.0, 0.0

            bbox_w_err = float(abs(rend_w - orig_w) / max(orig_w, 1.0))
            bbox_h_err = float(abs(rend_h - orig_h) / max(orig_h, 1.0))

            # 4. Contour difference: bidirectional boundary distance
            orig_dist = distance_transform_edt(1 - orig_resized)
            rend_dist = distance_transform_edt(1 - rend_bin)

            rend_edge = cv2.Canny(rend_bin * 255, 100, 200) > 0
            orig_edge = cv2.Canny(orig_resized * 255, 100, 200) > 0

            d1 = float(orig_dist[rend_edge].mean()) if np.any(rend_edge) else 50.0
            d2 = float(rend_dist[orig_edge].mean()) if np.any(orig_edge) else 50.0
            contour_diff = float(0.5 * (d1 + d2))

            # 5. Component 9: Color-coded overlay image
            # Overlay convention:
            # GREEN = matching geometry (True Positive)
            # RED = excess geometry (False Positive)
            # BLUE = missing geometry (False Negative)
            overlay = np.zeros((512, 512, 3), dtype=np.uint8)
            overlay[:] = [20, 20, 20]  # dark canvas

            tp = (orig_resized > 0) & (rend_bin > 0)
            fp = (orig_resized == 0) & (rend_bin > 0)
            fn = (orig_resized > 0) & (rend_bin == 0)

            # In BGR:
            overlay[tp] = [0, 230, 0]     # Vibrant Green (Match)
            overlay[fp] = [40, 40, 240]   # Vibrant Red (Excess geometry)
            overlay[fn] = [240, 120, 0]   # Vibrant Blue (Missing geometry)

            overlay_path = output_dir / f"{vname}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            bbox_w_errors.append(bbox_w_err)
            bbox_h_errors.append(bbox_h_err)
            contour_diffs.append(contour_diff)

            metrics["per_view"][vname] = {
                "iou": round(iou, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "bbox_width_error": round(bbox_w_err, 4),
                "bbox_height_error": round(bbox_h_err, 4),
                "contour_distance_px": round(contour_diff, 2),
                "rendered_area_px": int(rend_area),
                "input_area_px": int(orig_area),
                "artifacts": {
                    "input_mask": str(input_mask_path),
                    "render_silhouette": str(render_sil_path),
                    "overlay": str(overlay_path),
                },
            }
            logger.info(
                f"Geometry validation [{vname.upper()}]: IoU={iou:.4f}, Prec={precision:.4f}, "
                f"Rec={recall:.4f}, BBoxW_Err={bbox_w_err:.3f}, BBoxH_Err={bbox_h_err:.3f}, "
                f"ContourDist={contour_diff:.2f}px"
            )

        if ious:
            metrics["mean_iou"] = round(float(np.mean(ious)), 4)
            metrics["min_iou"] = round(float(np.min(ious)), 4)
            metrics["mean_precision"] = round(float(np.mean(precisions)), 4)
            metrics["min_precision"] = round(float(np.min(precisions)), 4)
            metrics["mean_recall"] = round(float(np.mean(recalls)), 4)
            metrics["max_contour_error"] = round(float(np.max(contour_diffs)), 2)
            metrics["mean_bbox_width_error"] = round(float(np.mean(bbox_w_errors)), 4)
            metrics["mean_bbox_height_error"] = round(float(np.mean(bbox_h_errors)), 4)
            metrics["mean_contour_difference_px"] = round(float(np.mean(contour_diffs)), 2)
            metrics["overall_consistency"] = round(
                float(0.4 * metrics["mean_iou"] + 0.3 * metrics["mean_precision"] + 0.3 * metrics["mean_recall"]),
                4,
            )

        # Acceptance threshold
        if metrics["min_iou"] < 0.65 or metrics["mean_iou"] < 0.80:
            metrics["passed"] = False
            logger.warning(
                f"Geometry validation NOT passed: Mean IoU={metrics['mean_iou']}, Min IoU={metrics['min_iou']}"
            )
        else:
            logger.info(
                f"Geometry validation PASSED: Mean IoU={metrics['mean_iou']}, Min IoU={metrics['min_iou']}, "
                f"Consistency={metrics['overall_consistency']}"
            )

        return metrics
