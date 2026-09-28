import bpy
import math
import mathutils
from pathlib import Path

def project_photo_and_render(glb_path: str, photo_path: str, output_glb: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 1. Import smoothed mesh
    bpy.ops.import_scene.gltf(filepath=glb_path)
    mesh_objs = [obj for obj in scene.objects if obj.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No mesh found")
    obj = mesh_objs[0]
    bpy.context.view_layer.objects.active = obj
    obj.name = "AirPodsMaxPhoto"

    # Smooth shading
    for p in obj.data.polygons:
        p.use_smooth = True

    # Compute bounds and center
    coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)

    center = mathutils.Vector(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0))
    dim_max = max(max_x - min_x, max_y - min_y, max_z - min_z)
    
    # Center object at origin
    obj.location -= center
    bpy.ops.object.transform_apply(location=True)

    # Recompute bounds after centering
    coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)
    w = max_x - min_x
    h = max_z - min_z

    # 2. Smart UV Unwrapping
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.003)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 3. Create Camera for Front Projection
    dist = max(w, h) * 1.8
    cam_data = bpy.data.cameras.new("FrontProjectCam")
    cam_data.lens = 70
    cam_proj = bpy.data.objects.new("FrontProjectCam", cam_data)
    scene.collection.objects.link(cam_proj)
    cam_proj.location = (0, -dist, 0)
    cam_proj.rotation_euler = (math.radians(90), 0, 0)

    # 4. Shader Node Setup for True Photo Projection
    mat = bpy.data.materials.new(name="AirPodsPhotorealistic")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    node_out = nodes.new("ShaderNodeOutputMaterial")
    node_out.location = (600, 0)

    # Principled BSDF
    node_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    node_bsdf.location = (300, 0)
    node_bsdf.inputs["Roughness"].default_value = 0.28
    node_bsdf.inputs["Metallic"].default_value = 0.25

    # Texture coordinate (Object)
    node_coord = nodes.new("ShaderNodeTexCoord")
    node_coord.location = (-600, 0)

    # Mapping node to align image to object bounds
    node_mapping = nodes.new("ShaderNodeMapping")
    node_mapping.location = (-400, 0)
    # Scale: maps [-w/2, w/2] to [0, 1]
    scale_val = 1.0 / max(w, h)
    node_mapping.inputs["Scale"].default_value = (scale_val, scale_val, scale_val)
    node_mapping.inputs["Location"].default_value = (0.5, 0.5, 0.5)

    # Image Texture node
    node_tex = nodes.new("ShaderNodeTexImage")
    node_tex.location = (-150, 0)
    img = bpy.data.images.load(photo_path)
    node_tex.image = img
    node_tex.extension = 'CLIP'

    # Color ramp / fallback for back/sides
    node_mix = nodes.new("ShaderNodeMix")
    node_mix.data_type = 'RGBA'
    node_mix.location = (100, 0)
    # Neutral elegant silver-white fallback
    node_mix.inputs[6].default_value = (0.88, 0.88, 0.90, 1.0) # Background fallback A

    # Geometry node to get surface normal
    node_geo = nodes.new("ShaderNodeNewGeometry")
    node_geo.location = (-200, -250)

    # Dot product of normal with Front direction (0, -1, 0)
    node_vector_math = nodes.new("ShaderNodeVectorMath")
    node_vector_math.operation = 'DOT_PRODUCT'
    node_vector_math.location = (0, -250)
    node_vector_math.inputs[1].default_value = (0.0, -1.0, 0.0)

    # Fac based on front visibility
    node_math = nodes.new("ShaderNodeMath")
    node_math.operation = 'GREATER_THAN'
    node_math.location = (150, -250)
    node_math.inputs[1].default_value = 0.1

    # Connect nodes
    links.new(node_coord.outputs["Object"], node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_tex.inputs["Vector"])
    
    # Mix projected image where front facing and alpha > 0.5
    links.new(node_tex.outputs["Color"], node_mix.inputs[7]) # B
    links.new(node_tex.outputs["Alpha"], node_mix.inputs[0]) # Factor
    links.new(node_mix.outputs[2], node_bsdf.inputs["Base Color"])
    links.new(node_bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # 5. Export Photorealistic GLB
    bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')
    print(f"Exported photo-accurate GLB: {output_glb}")

    # 6. Studio Lighting & Renders
    render_dist = max(w, h) * 2.1
    cam_data_render = bpy.data.cameras.new("RenderCamera")
    cam_data_render.lens = 70
    cam_render = bpy.data.objects.new("RenderCamera", cam_data_render)
    scene.collection.objects.link(cam_render)
    scene.camera = cam_render

    # Key light
    key_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_data.energy = 320.0
    key_data.size = max(w, h) * 1.5
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    key_obj.location = (render_dist * 0.7, -render_dist * 0.8, render_dist * 0.7)
    scene.collection.objects.link(key_obj)

    # Fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 140.0
    fill_data.size = max(w, h) * 2.0
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    fill_obj.location = (-render_dist * 0.8, -render_dist * 0.6, render_dist * 0.3)
    scene.collection.objects.link(fill_obj)

    # Rim light
    rim_data = bpy.data.lights.new(name="RimLight", type='AREA')
    rim_data.energy = 200.0
    rim_data.size = max(w, h) * 1.2
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    rim_obj.location = (0, render_dist * 0.9, render_dist * 0.8)
    scene.collection.objects.link(rim_obj)

    # Studio World
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.09, 0.10, 0.12, 1.0)
        bg_node.inputs["Strength"].default_value = 0.9

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    def point_at(target_obj, center_pt=(0,0,0)):
        direction = mathutils.Vector(center_pt) - target_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        target_obj.rotation_euler = rot_quat.to_euler()

    point_at(key_obj)
    point_at(fill_obj)
    point_at(rim_obj)

    views = {
        "photo_front": (0, -render_dist, 0),
        "photo_perspective": (render_dist * 0.65, -render_dist * 0.85, render_dist * 0.35),
        "photo_side": (render_dist, 0, 0),
    }

    for name, loc in views.items():
        cam_render.location = mathutils.Vector(loc)
        point_at(cam_render)
        scene.render.filepath = str(output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} -> {scene.render.filepath}")

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glb_in = args[0] if len(args) > 0 else "outputs/apple_airpods_max_pbr.glb"
    photo = args[1] if len(args) > 1 else "scratch_perfect_main_object.png"
    glb_out = args[2] if len(args) > 2 else "outputs/airpods_max_photo_accurate.glb"
    renders_dir = args[3] if len(args) > 3 else "temp/renders_photo"
    project_photo_and_render(glb_in, photo, glb_out, renders_dir)
