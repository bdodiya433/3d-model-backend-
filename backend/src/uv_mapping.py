"""
uv_mapping.py - Universal UV Unwrapping Module for Arbitrary 3D Meshes.

Applies topology-agnostic, low-fragmentation UV unwrapping directly on the
reconstructed 3D mesh (can, bottle, shoe, toy, box, irregular object).
Minimizes seams and maximizes texel density for sharp printed details.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("backend.uv_mapping")


class UniversalUVMapper:
    """
    Generic UV mapper that operates directly on any arbitrary reconstructed 3D mesh.
    Completely product-independent: no cylindrical, rectangular, or primitive assumptions.
    """

    def __init__(self, blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender"):
        self.blender_bin = blender_bin

    def create_uv_layout(
        self,
        mesh_blend_path: Path,
        output_dir: Path,
        resolution: int = 4096,
        angle_limit: float = 66.0,
        island_margin: float = 0.003,
    ) -> Dict[str, Path]:
        """
        Unwraps the reconstructed mesh into clean, non-overlapping, low-fragmentation UV islands.

        Returns:
            {
                "blend": Path to Blender scene with unwrapped mesh,
                "obj": Path to unwrapped OBJ,
                "uv_layout": Path to visual preview of UV layout,
            }
        """
        mesh_blend_path = Path(mesh_blend_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        uv_blend = output_dir / "reconstructed_uv.blend"
        uv_obj = output_dir / "reconstructed_uv.obj"
        uv_layout_png = output_dir / "uv_layout.png"

        script_content = f'''
import bpy
import math

bpy.ops.wm.open_mainfile(filepath=r"{mesh_blend_path}")

mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objs:
    raise RuntimeError("No mesh object found in blend file")

obj = mesh_objs[0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# 1. Clear any legacy UV layers
while obj.data.uv_layers:
    obj.data.uv_layers.remove(obj.data.uv_layers[0])

# 2. Topology-Agnostic Low-Fragmentation UV Unwrapping
# Angle limit of 66 degrees keeps large coherent product surfaces together (e.g. shoe upper, can body, box sides)
# island_margin prevents mipmap bleeding while maximizing texel density
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

bpy.ops.uv.smart_project(
    angle_limit=math.radians({angle_limit}),
    island_margin={island_margin},
    correct_aspect=True,
    scale_to_bounds=True
)

bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.wm.save_as_mainfile(filepath=r"{uv_blend}")
bpy.ops.wm.obj_export(filepath=r"{uv_obj}", up_axis="Z", forward_axis="Y")
print("UNIVERSAL_UV_MAPPING_SUCCESS")
'''
        temp_py = output_dir / "uv_temp.py"
        with open(temp_py, "w") as f:
            f.write(script_content)

        try:
            cmd = [self.blender_bin, "--background", "--python", str(temp_py)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if "UNIVERSAL_UV_MAPPING_SUCCESS" not in res.stdout:
                raise RuntimeError(f"Universal UV mapping failed:\n{res.stdout}\n{res.stderr}")
            logger.info(f"Successfully generated universal UV layout in {uv_blend}")
        finally:
            if temp_py.exists():
                temp_py.unlink()

        self._generate_uv_diagram(uv_layout_png, uv_blend, resolution=min(resolution, 2048))

        return {
            "uv_layout": uv_layout_png,
            "blend": uv_blend,
            "obj": uv_obj,
        }

    def _generate_uv_diagram(self, output_png: Path, blend_path: Path, resolution: int = 2048) -> None:
        """
        Renders an informational preview diagram of the UV islands.
        """
        script = f'''
import bpy
import numpy as np

bpy.ops.wm.open_mainfile(filepath=r"{blend_path}")
obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
mesh = obj.data

uv_layer = mesh.uv_layers.active
if uv_layer:
    coords = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", coords)
    np.save(r"{output_png}.npy", coords)
print("UV_DUMP_SUCCESS")
'''
        temp_py = output_png.parent / "dump_uv.py"
        npy_path = Path(f"{output_png}.npy")
        try:
            with open(temp_py, "w") as f:
                f.write(script)
            subprocess.run([self.blender_bin, "--background", "--python", str(temp_py)], capture_output=True)
            if npy_path.exists():
                coords = np.load(str(npy_path)).reshape(-1, 2)
                npy_path.unlink()

                img = Image.new("RGBA", (resolution, resolution), (20, 20, 25, 255))
                draw = ImageDraw.Draw(img)
                # Draw UV island points / wire segments
                u_px = np.clip((coords[:, 0] * (resolution - 1)).astype(int), 0, resolution - 1)
                v_px = np.clip(((1.0 - coords[:, 1]) * (resolution - 1)).astype(int), 0, resolution - 1)
                for i in range(0, len(u_px) - 2, 3):
                    pts = [(u_px[i], v_px[i]), (u_px[i+1], v_px[i+1]), (u_px[i+2], v_px[i+2])]
                    draw.polygon(pts, outline=(80, 180, 240, 180), fill=(40, 70, 100, 80))
                img.save(output_png, "PNG")
                return
        except Exception as e:
            logger.warning(f"Could not render detailed UV preview: {e}")
        finally:
            if temp_py.exists():
                temp_py.unlink()

        # Fallback diagram
        img = Image.new("RGBA", (resolution, resolution), (25, 25, 30, 255))
        img.save(output_png, "PNG")


# Backward-compatibility alias
CanUVMapper = UniversalUVMapper
