import bpy
import math
import mathutils
from pathlib import Path

def project_realistic_apple_airpods(glb_path: str, photo_path: str, output_glb: str, output_dir: str):
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
    obj.name = "AirPodsMax"

    for p in obj.data.polygons:
        p.use_smooth = True

    # 2. Smooth & Decimate for crisp CAD finish
    smooth = obj.modifiers.new("Smooth", "SMOOTH")
    smooth.factor = 0.85
    smooth.iterations = 15
    bpy.ops.object.modifier_apply(modifier="Smooth")

    # Center object
    coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)

    center = mathutils.Vector(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0))
    obj.location -= center
    bpy.ops.object.transform_apply(location=True)

    # Recompute bounds
    coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)
    w = max_x - min_x
    h = max_z - min_z
    dim_max = max(w, h, max_y - min_y)

    # 3. Smart UV Unwrapping
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.003)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 4. Shader Setup: Realistic Apple Materials + Photographic Front Detail with Normal Falloff
    mat = bpy.data.materials.new(name="AirPodsRealisticPBR")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (1000, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (700, 0)
    # Satin Apple Anodized Aluminum / Premium finish
    bsdf.inputs["Roughness"].default_value = 0.28
    bsdf.inputs["Metallic"].default_value = 0.45

    # Texture Coordinate (Object)
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-700, 200)

    # Mapping
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-500, 200)
    s = 1.0 / max(w, h)
    mapping.inputs["Scale"].default_value = (s, s, s)
    # Map [-w/2, w/2] and [-h/2, h/2] to [0, 1]
    mapping.inputs["Location"].default_value = (0.5, 0.5, 0.5)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    # Image Texture
    tex_img = nodes.new("ShaderNodeTexImage")
    tex_img.location = (-250, 200)
    tex_img.image = bpy.data.images.load(photo_path)
    tex_img.extension = 'CLIP'
    links.new(mapping.outputs["Vector"], tex_img.inputs["Vector"])

    # Geometry Normal for visibility falloff
    geo = nodes.new("ShaderNodeNewGeometry")
    geo.location = (-600, -200)

    # Dot product of normal with Front (-Y axis: [0, -1, 0])
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = 'DOT_PRODUCT'
    dot.location = (-350, -200)
    dot.inputs[1].default_value = (0.0, -1.0, 0.0)
    links.new(geo.outputs["Normal"], dot.inputs[0])

    # MapRange: smooth falloff between normal dot product 0.1 and 0.6
    falloff = nodes.new("ShaderNodeMapRange")
    falloff.location = (-100, -200)
    falloff.inputs["From Min"].default_value = 0.15
    falloff.inputs["From Max"].default_value = 0.70
    falloff.inputs["To Min"].default_value = 0.0
    falloff.inputs["To Max"].default_value = 1.0
    links.new(dot.outputs["Value"], falloff.inputs["Value"])

    # Multiply falloff by Photo Alpha
    mix_fac = nodes.new("ShaderNodeMath")
    mix_fac.operation = 'MULTIPLY'
    mix_fac.location = (150, 0)
    links.new(falloff.outputs["Result"], mix_fac.inputs[0])
    links.new(tex_img.outputs["Alpha"], mix_fac.inputs[1])

    # Mix Color: Photo vs Apple Silver-White Satin
    mix_col = nodes.new("ShaderNodeMix")
    mix_col.data_type = 'RGBA'
    mix_col.location = (400, 0)
    # Pure Apple Silver-White Studio color
    mix_col.inputs[6].default_value = (0.91, 0.91, 0.92, 1.0) # Base A
    links.new(mix_fac.outputs["Value"], mix_col.inputs[0])      # Factor
    links.new(tex_img.outputs["Color"], mix_col.inputs[7])      # Photo B

    links.new(mix_col.outputs[2], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # 5. Export GLB
    bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')
    print(f"Exported refined photorealistic GLB: {output_glb}")

    # 6. Studio Lighting & High-End Product Photography Renders
    dist = dim_max * 2.15
    cam_data = bpy.data.cameras.new("StudioCam")
    cam_data.lens = 75
    cam_obj = bpy.data.objects.new("StudioCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Key light: Soft wide area light
    key_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_data.energy = 420.0
    key_data.size = dim_max * 2.0
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    key_obj.location = (dist * 0.75, -dist * 0.85, dist * 0.7)
    scene.collection.objects.link(key_obj)

    # Fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 180.0
    fill_data.size = dim_max * 2.5
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    fill_obj.location = (-dist * 0.85, -dist * 0.6, dist * 0.3)
    scene.collection.objects.link(fill_obj)

    # Rim light
    rim_data = bpy.data.lights.new(name="RimLight", type='AREA')
    rim_data.energy = 280.0
    rim_data.size = dim_max * 1.5
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    rim_obj.location = (0, dist * 0.9, dist * 0.8)
    scene.collection.objects.link(rim_obj)

    # Studio World
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.10, 0.11, 0.13, 1.0)
        bg_node.inputs["Strength"].default_value = 0.8

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    def point_at(target_obj):
        direction = mathutils.Vector((0, 0, 0)) - target_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        target_obj.rotation_euler = rot_quat.to_euler()

    point_at(key_obj)
    point_at(fill_obj)
    point_at(rim_obj)

    views = {
        "apple_v2_front": (0, -dist, 0),
        "apple_v2_perspective": (dist * 0.65, -dist * 0.85, dist * 0.35),
        "apple_v2_side": (dist, 0, 0),
    }

    for name, loc in views.items():
        cam_obj.location = mathutils.Vector(loc)
        point_at(cam_obj)
        scene.render.filepath = str(output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} -> {scene.render.filepath}")

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    in_glb = args[0] if len(args) > 0 else "scratch_headphone_white.glb"
    photo = args[1] if len(args) > 1 else "scratch_perfect_main_object.png"
    glb_out = args[2] if len(args) > 2 else "outputs/airpods_max_v2.glb"
    r_dir = args[3] if len(args) > 3 else "temp/renders_apple_v2"
    project_realistic_apple_airpods(in_glb, photo, glb_out, r_dir)
