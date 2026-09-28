"""
blender_pipeline.py - Module for automating headless Blender operations.
Loads the UV-mapped model, attaches the PBR material, packs the high-res texture,
and exports standalone self-contained GLB, glTF, and OBJ files.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("backend.blender_pipeline")


class BlenderPipeline:
    """
    Coordinates Blender execution to assemble and export the final 3D assets.
    """

    def __init__(self, blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender"):
        self.blender_bin = blender_bin

    def export_final_assets(
        self,
        can_uv_blend_path: Path,
        texture_basecolor_path: Path,
        output_dir: Path,
        texture_normal_path: Optional[Path] = None,
        texture_roughness_path: Optional[Path] = None,
        texture_metallic_path: Optional[Path] = None,
        product_name: str = "reconstructed_product",
    ) -> Dict[str, Path]:
        """
        Creates and saves standalone self-contained 3D assets:
          - reconstructed_product.glb
          - reconstructed_product.gltf
          - reconstructed_product.obj
          - texture_basecolor.png
          - texture_metallic.png (if available)
          - texture_normal.png (if available)
          - texture_roughness.png (if available)
        """
        can_uv_blend_path = Path(can_uv_blend_path).resolve()
        texture_basecolor_path = Path(texture_basecolor_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        out_glb = output_dir / f"{product_name}.glb"
        out_gltf = output_dir / f"{product_name}.gltf"
        out_obj = output_dir / f"{product_name}.obj"
        out_tex = output_dir / "texture_basecolor.png"

        # Copy texture to final output directory
        shutil.copy2(texture_basecolor_path, out_tex)

        metal_setup = ""
        if texture_metallic_path and Path(texture_metallic_path).exists():
            out_metal = output_dir / "texture_metallic.png"
            shutil.copy2(texture_metallic_path, out_metal)
            metal_setup = f'''
metal_img = bpy.data.images.load(r"{out_metal}")
try:
    metal_img.colorspace_settings.name = "Non-Color"
except Exception:
    pass
metal_img.pack()
metal_node = nodes.new("ShaderNodeTexImage")
metal_node.image = metal_img
links.new(metal_node.outputs["Color"], bsdf.inputs["Metallic"])
'''
        else:
            metal_setup = 'bsdf.inputs["Metallic"].default_value = 0.0\n'

        norm_setup = ""
        if texture_normal_path and Path(texture_normal_path).exists():
            out_norm = output_dir / "texture_normal.png"
            shutil.copy2(texture_normal_path, out_norm)
            norm_setup = f'''
norm_img = bpy.data.images.load(r"{out_norm}")
try:
    norm_img.colorspace_settings.name = "Non-Color"
except Exception:
    pass
norm_img.pack()
norm_node = nodes.new("ShaderNodeTexImage")
norm_node.image = norm_img
nmap = nodes.new("ShaderNodeNormalMap")
nmap.inputs["Strength"].default_value = 0.8
links.new(norm_node.outputs["Color"], nmap.inputs["Color"])
links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
'''

        rough_setup = ""
        if texture_roughness_path and Path(texture_roughness_path).exists():
            out_rough = output_dir / "texture_roughness.png"
            shutil.copy2(texture_roughness_path, out_rough)
            rough_setup = f'''
rough_img = bpy.data.images.load(r"{out_rough}")
try:
    rough_img.colorspace_settings.name = "Non-Color"
except Exception:
    pass
rough_img.pack()
rough_node = nodes.new("ShaderNodeTexImage")
rough_node.image = rough_img
links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])
'''

        script_content = f'''
import bpy
import os

bpy.ops.wm.open_mainfile(filepath=r"{can_uv_blend_path}")

obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# 1. Setup Material with Packed BaseColor Texture
mat = bpy.data.materials.new(name=f"{product_name}_PBR")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

bsdf = nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Roughness"].default_value = 0.40
if "Specular IOR Level" in bsdf.inputs:
    bsdf.inputs["Specular IOR Level"].default_value = 0.50
elif "Specular" in bsdf.inputs:
    bsdf.inputs["Specular"].default_value = 0.50

# Load and pack BaseColor texture (sRGB — photographic colour data)
tex_img = bpy.data.images.load(r"{out_tex}")
try:
    tex_img.colorspace_settings.name = "sRGB"
except Exception:
    pass
tex_img.pack()

tex_node = nodes.new("ShaderNodeTexImage")
tex_node.image = tex_img
links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
{metal_setup}
{rough_setup}
{norm_setup}
out_node = nodes.new("ShaderNodeOutputMaterial")
links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

obj.data.materials.clear()
obj.data.materials.append(mat)

for p in obj.data.polygons:
    p.use_smooth = True

# 2. Export Standalone Self-Contained GLB
bpy.ops.export_scene.gltf(
    filepath=r"{out_glb}",
    export_format='GLB',
    export_materials='EXPORT',
    export_image_format='AUTO',
    export_apply=True
)

# 3. Export glTF (.gltf + .bin)
bpy.ops.export_scene.gltf(
    filepath=r"{out_gltf}",
    export_format='GLTF_SEPARATE',
    export_materials='EXPORT',
    export_image_format='AUTO',
    export_apply=True
)

# 4. Export OBJ + MTL
try:
    bpy.ops.wm.obj_export(filepath=r"{out_obj}", export_materials=True)
except Exception:
    bpy.ops.export_scene.obj(filepath=r"{out_obj}", use_materials=True)

print("FINAL_ASSET_EXPORT_SUCCESS")
'''

        temp_py = output_dir / "export_temp.py"
        with open(temp_py, "w") as f:
            f.write(script_content)

        try:
            cmd = [self.blender_bin, "--background", "--python", str(temp_py)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if "FINAL_ASSET_EXPORT_SUCCESS" not in res.stdout:
                raise RuntimeError(f"Final Blender asset export failed:\n{res.stdout}\n{res.stderr}")
            logger.info(f"Successfully exported final 3D models to {output_dir}")
        finally:
            if temp_py.exists():
                temp_py.unlink()

        # Create backward-compatibility copy if needed
        legacy_glb = output_dir / "coca_cola_can.glb"
        if out_glb != legacy_glb and out_glb.exists():
            shutil.copy2(out_glb, legacy_glb)

        return {
            "glb": out_glb,
            "legacy_glb": legacy_glb,
            "gltf": out_gltf,
            "obj": out_obj,
            "texture": out_tex,
        }
