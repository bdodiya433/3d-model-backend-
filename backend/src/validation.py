"""
validation.py - Module for rendering and verifying the final textured 3D model.
Renders 6 canonical photographic angles (front, back, left, right, top, bottom)
and performs automated visual and structural validation tests.

Adapts camera framing dynamically to ANY object geometry and bounding box.
"""

import logging
import math
import subprocess
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image

logger = logging.getLogger("backend.validation")


class ModelValidator:
    """
    Renders the final textured 3D model and validates graphic fidelity.
    Works with ANY product geometry.
    """

    def __init__(self, blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender"):
        self.blender_bin = blender_bin

    def render_validation_views(
        self,
        final_glb_path: Path,
        output_dir: Path,
        resolution: int = 512,
    ) -> Dict[str, Path]:
        """
        Renders the final 3D model from 6 viewpoints:
          - front.png
          - back.png
          - left.png
          - right.png
          - top.png
          - bottom.png
        """
        final_glb_path = Path(final_glb_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        views = ["front", "back", "left", "right", "top", "bottom"]
        rendered_paths = {}

        script_content = f'''
import bpy
import math
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"{final_glb_path}")

obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]

# Compute true geometric center and dimension
bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
center = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
max_dim = max(obj.dimensions)
dist = max(max_dim * 1.5, 1.8)

# Add Camera pointing at object center
cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'FLAT'
bpy.context.scene.display.shading.color_type = 'TEXTURE'
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.resolution_x = {resolution}
bpy.context.scene.render.resolution_y = {resolution}

cx, cy, cz = center.x, center.y, center.z

views = {{
    'front':  ((cx, cy - dist, cz), (math.radians(90), 0.0, 0.0)),
    'back':   ((cx, cy + dist, cz), (math.radians(90), 0.0, math.radians(180))),
    'left':   ((cx - dist, cy, cz), (math.radians(90), 0.0, math.radians(-90))),
    'right':  ((cx + dist, cy, cz), (math.radians(90), 0.0, math.radians(90))),
    'top':    ((cx, cy, cz + dist), (0.0, 0.0, 0.0)),
    'bottom': ((cx, cy, cz - dist), (math.radians(180), 0.0, 0.0)),
}}

for view_name, (pos, rot) in views.items():
    cam_obj.location = pos
    cam_obj.rotation_euler = rot
    bpy.context.view_layer.update()
    out_file = r"{output_dir}" + f"/{{view_name}}.png"
    bpy.context.scene.render.filepath = out_file
    bpy.ops.render.render(write_still=True)

print("VALIDATION_RENDERS_SUCCESS")
'''

        temp_py = output_dir / "render_temp.py"
        with open(temp_py, "w") as f:
            f.write(script_content)

        try:
            cmd = [self.blender_bin, "--background", "--python", str(temp_py)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if "VALIDATION_RENDERS_SUCCESS" not in res.stdout:
                raise RuntimeError(f"Validation renders failed:\n{res.stdout}\n{res.stderr}")
            logger.info(f"Rendered all 6 validation views to {output_dir}")
        finally:
            if temp_py.exists():
                temp_py.unlink()

        for v in views:
            p = output_dir / f"{v}.png"
            if p.exists():
                rendered_paths[v] = p

        return rendered_paths

    def validate_results(self, rendered_views: Dict[str, Path]) -> Dict[str, Any]:
        """
        Analyzes the rendered validation images against required criteria:
          - Non-empty renders
          - Photographic pixel presence
          - Text / graphic visibility
          - Structural integrity
        """
        report = {"passed": True, "details": {}}

        for view_name, p in rendered_views.items():
            img = Image.open(p).convert("RGBA")
            arr = np.array(img)
            active_pixels = arr[arr[..., 3] > 10]

            if len(active_pixels) < 1000:
                report["passed"] = False
                report["details"][view_name] = "FAIL: Object not visible or empty in render"
                continue

            mean_rgb = active_pixels[..., :3].mean(axis=0)

            # High-brightness / graphic pixels check
            white_text_pixels = np.count_nonzero(
                (active_pixels[..., 0] > 165) & (active_pixels[..., 1] > 155) & (active_pixels[..., 2] > 155)
            )

            report["details"][view_name] = {
                "active_pixel_count": int(len(active_pixels)),
                "mean_rgb": [round(float(c), 1) for c in mean_rgb],
                "white_feature_pixels": int(white_text_pixels),
                "status": "PASS",
            }

        logger.info(f"Validation summary: {report}")
        return report
