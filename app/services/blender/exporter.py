import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("app.services.blender.exporter")


class BlenderExporter:
    """
    Service responsible for converting 3D GLB models into native Blender (.blend) files
    by executing Blender in headless background mode.
    """

    def __init__(self, blender_path: Optional[str] = None):
        # Locate Blender executable dynamically on system if not provided
        self.blender_path = blender_path or self._find_blender_executable()

    def _find_blender_executable(self) -> str:
        # Standard macOS installation path
        macos_path = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        if macos_path.exists():
            return str(macos_path)

        # Environment or system PATH
        env_path = os.getenv("BLENDER_PATH")
        if env_path and Path(env_path).exists():
            return env_path

        return "blender"  # Fallback to system command

    def export_to_blend(self, glb_path: Path, output_blend_path: Path) -> Path:
        """
        Converts a .glb file to a native .blend file using headless Blender execution.
        """
        if not glb_path.exists():
            raise FileNotFoundError(f"Input GLB file not found: {glb_path}")

        logger.info(f"Converting GLB asset '{glb_path.name}' to Blender project file '.blend'...")

        # Inline Python script executed inside Blender engine
        blender_python_script = """
import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after --

glb_input = argv[0]
blend_output = argv[1]

# Clear default scene objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import GLB/GLTF scene
print(f"Blender: Importing GLB with materials & textures from {glb_input}...")
bpy.ops.import_scene.gltf(filepath=glb_input)

bpy.context.scene.display_settings.display_device = 'sRGB'
bpy.context.scene.view_settings.view_transform = 'Standard'
bpy.context.scene.view_settings.look = 'None'

# Configure all imported mesh objects for editing & wireframe visibility
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.name = "Editable_3D_Mesh"
        
        # 1. Clean viewport display setup: Textured view by default
        obj.show_wire = False
        obj.show_all_edges = False
        obj.display_type = 'TEXTURED'
        
        # 2. Select object and switch to Edit Mode for topology cleanup
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        # 3. Convert triangles to clean quads for easy part editing and recoloring
        try:
            bpy.ops.mesh.tris_convert_to_quads(limit=0.698132, uvs=True)
        except Exception as e:
            print(f"Tris to quads conversion note: {e}")
            
        # 4. Recalculate outer normals for clean shading
        try:
            bpy.ops.mesh.normals_make_consistent(inside=False)
        except Exception as e:
            print(f"Normals recalculation note: {e}")
            
        # Switch back to Object Mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # 5. Enable Smooth Shading
        for poly in obj.data.polygons:
            poly.use_smooth = True

# Setup Shader Nodes for Color Attributes & PBR Material in Blender
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        color_layer_name = None
        if hasattr(obj.data, "color_attributes") and obj.data.color_attributes:
            color_layer_name = obj.data.color_attributes[0].name
            print(f"Blender: Found color attribute: {color_layer_name}")

        if not obj.data.materials:
            mat = bpy.data.materials.new(name="PBR_Material")
            mat.use_nodes = True
            obj.data.materials.append(mat)

        for mat in obj.data.materials:
            if mat and mat.use_nodes and mat.node_tree:
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf:
                    base_color_input = bsdf.inputs.get('Base Color')
                    if base_color_input:
                        if not base_color_input.is_linked and color_layer_name:
                            attr_node = nodes.new(type='ShaderNodeColorAttribute')
                            attr_node.name = "Vertex_Color_Attribute"
                            attr_node.attribute_name = color_layer_name
                            attr_node.location = (bsdf.location.x - 280, bsdf.location.y)
                            links.new(attr_node.outputs['Color'], base_color_input)

                    if 'Roughness' in bsdf.inputs and not bsdf.inputs['Roughness'].is_linked:
                        bsdf.inputs['Roughness'].default_value = 0.4
                    if 'Metallic' in bsdf.inputs and not bsdf.inputs['Metallic'].is_linked:
                        bsdf.inputs['Metallic'].default_value = 0.05

# Add a standard Sun Light and Camera if missing for instant viewport rendering
if "Sun" not in bpy.data.objects:
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 3.0
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    light_object.location = (5.0, 5.0, 10.0)
    bpy.context.collection.objects.link(light_object)

if "Camera" not in bpy.data.objects:
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_object = bpy.data.objects.new(name="Camera", object_data=cam_data)
    cam_object.location = (0.0, -4.0, 2.0)
    cam_object.rotation_euler = (1.2, 0.0, 0.0)
    bpy.context.collection.objects.link(cam_object)

# Enable Material Viewport Shading for 3D View areas so textures & colors load instantly on launch
for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

# Save as optimized editable .blend file
print(f"Blender: Saving scene to {blend_output}...")
bpy.ops.wm.save_as_mainfile(filepath=blend_output)
print("Blender: Conversion completed with full image colors, viewport material shading, and color edit node setup!")
"""

        script_file = glb_path.parent / f"convert_{glb_path.stem}.py"
        try:
            script_file.write_text(blender_python_script)

            cmd = [
                self.blender_path,
                "--background",
                "--python",
                str(script_file),
                "--",
                str(glb_path),
                str(output_blend_path),
            ]

            logger.info(f"Executing Blender CLI: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                logger.error(f"Blender conversion error (exit code {result.returncode}):\n{result.stderr}")
                raise RuntimeError(f"Blender failed to export .blend file: {result.stderr or result.stdout}")

            if not output_blend_path.exists():
                raise FileNotFoundError(f"Output .blend file was not created by Blender at: {output_blend_path}")

            logger.info(f"Successfully generated Blender project file: {output_blend_path}")
            return output_blend_path

        except Exception as e:
            logger.error(f"Blender exporter exception: {e}", exc_info=True)
            raise e
        finally:
            # Clean up temporary python script
            if script_file.exists():
                try:
                    script_file.unlink()
                except Exception:
                    pass
