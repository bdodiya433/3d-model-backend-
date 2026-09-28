"""
material.py - Module for creating realistic aluminum can PBR materials in Blender.
Applies metallic properties, roughness variation, and embeds the BaseColor texture.
"""

import logging
from pathlib import Path

logger = logging.getLogger("backend.material")


class CanMaterialBuilder:
    """
    Constructs Blender Principled BSDF node network for realistic aluminum beverage cans.
    """

    def __init__(self, metallic: float = 0.85, roughness: float = 0.35):
        self.metallic = metallic
        self.roughness = roughness

    def get_blender_setup_script(
        self,
        texture_path: Path,
        normal_path: Optional[Path] = None,
        roughness_path: Optional[Path] = None
    ) -> str:
        """
        Returns the Blender Python snippet to setup the PBR material on the active mesh object.
        """
        texture_path = Path(texture_path).resolve()
        normal_script = ""
        if normal_path and Path(normal_path).exists():
            norm_res = Path(normal_path).resolve()
            normal_script = f'''
# Normal Map for droplet and embossed relief
norm_img = bpy.data.images.load(r"{norm_res}")
try:
    norm_img.colorspace_settings.name = "Non-Color"
except Exception:
    pass
norm_img.pack()

norm_tex_node = nodes.new("ShaderNodeTexImage")
norm_tex_node.image = norm_img

norm_map_node = nodes.new("ShaderNodeNormalMap")
norm_map_node.inputs["Strength"].default_value = 1.6
links.new(norm_tex_node.outputs["Color"], norm_map_node.inputs["Color"])
links.new(norm_map_node.outputs["Normal"], bsdf.inputs["Normal"])
'''

        rough_script = ""
        if roughness_path and Path(roughness_path).exists():
            rough_res = Path(roughness_path).resolve()
            rough_script = f'''
# Roughness Map for wet water droplet specular highlights
rough_img = bpy.data.images.load(r"{rough_res}")
try:
    rough_img.colorspace_settings.name = "Non-Color"
except Exception:
    pass
rough_img.pack()

rough_tex_node = nodes.new("ShaderNodeTexImage")
rough_tex_node.image = rough_img
links.new(rough_tex_node.outputs["Color"], bsdf.inputs["Roughness"])
'''

        return f'''
# Setup PBR Aluminum Material
obj = bpy.context.active_object
mat = bpy.data.materials.new(name="CocaCola_PBR_Material")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

# Principled BSDF
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Metallic"].default_value = {self.metallic}
bsdf.inputs["Roughness"].default_value = {self.roughness}

# BaseColor Texture Image
tex_img = bpy.data.images.load(r"{texture_path}")
tex_img.pack()

tex_node = nodes.new("ShaderNodeTexImage")
tex_node.image = tex_img
links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
{normal_script}
{rough_script}
# Material Output
out_node = nodes.new("ShaderNodeOutputMaterial")
links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

obj.data.materials.clear()
obj.data.materials.append(mat)
'''
