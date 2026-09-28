import bpy
import math
import mathutils
from pathlib import Path

def setup_multiview_and_render(glb_in: str, views_dir: str, glb_out: str, renders_dir: str):
    renders_dir = Path(renders_dir)
    renders_dir.mkdir(parents=True, exist_ok=True)
    views_dir = Path(views_dir)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 1. Import smoothed mesh
    bpy.ops.import_scene.gltf(filepath=glb_in)
    mesh_objs = [obj for obj in scene.objects if obj.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No mesh found")
    obj = mesh_objs[0]
    bpy.context.view_layer.objects.active = obj
    obj.name = "AirPodsMax"

    # Smooth shading
    for p in obj.data.polygons:
        p.use_smooth = True

    # Compute bounding box & center
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
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)

    w_x = max_x - min_x
    w_y = max_y - min_y
    w_z = max_z - min_z
    dim_max = max(w_x, w_y, w_z)

    # 2. UV Unwrapping
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.003)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 3. Create Multi-View Texture Projection Shader
    mat = bpy.data.materials.new(name="AirPodsMultiView")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (1200, 0)

    bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf_node.location = (900, 0)
    bsdf_node.inputs["Roughness"].default_value = 0.32
    bsdf_node.inputs["Metallic"].default_value = 0.35

    # Coordinates
    coord_node = nodes.new("ShaderNodeTexCoord")
    coord_node.location = (-1000, 0)

    geo_node = nodes.new("ShaderNodeNewGeometry")
    geo_node.location = (-1000, -400)

    # Separate Normal into X, Y, Z
    sep_norm = nodes.new("ShaderNodeSeparateXYZ")
    sep_norm.location = (-750, -400)
    links.new(geo_node.outputs["Normal"], sep_norm.inputs["Vector"])

    # Load 4 primary photos: Front, Back, Left, Right
    front_img = bpy.data.images.load(str(views_dir / "front.png"))
    back_img = bpy.data.images.load(str(views_dir / "back.png"))
    left_img = bpy.data.images.load(str(views_dir / "left.png"))
    right_img = bpy.data.images.load(str(views_dir / "right.png"))

    # Mapping scale
    s_val = 1.0 / max(w_x, w_z)
    s_val_side = 1.0 / max(w_y, w_z)

    # Helper to create projection branch
    def make_proj(name, img, loc_y, rot_euler, scale_xyz):
        map_n = nodes.new("ShaderNodeMapping")
        map_n.location = (-500, loc_y)
        map_n.inputs["Scale"].default_value = scale_xyz
        map_n.inputs["Location"].default_value = (0.5, 0.5, 0.5)
        map_n.inputs["Rotation"].default_value = rot_euler
        links.new(coord_node.outputs["Object"], map_n.inputs["Vector"])

        tex_n = nodes.new("ShaderNodeTexImage")
        tex_n.location = (-250, loc_y)
        tex_n.image = img
        tex_n.extension = 'CLIP'
        links.new(map_n.outputs["Vector"], tex_n.inputs["Vector"])
        return tex_n

    # Front (Looking at -Y): mapped on X and Z
    tex_front = make_proj("Front", front_img, 300, (math.radians(90), 0, 0), (s_val, s_val, s_val))
    # Back (Looking at +Y)
    tex_back = make_proj("Back", back_img, 0, (math.radians(90), 0, math.radians(180)), (s_val, s_val, s_val))
    # Left (Looking at -X): mapped on Y and Z
    tex_left = make_proj("Left", left_img, -300, (math.radians(90), 0, math.radians(-90)), (s_val_side, s_val_side, s_val_side))
    # Right (Looking at +X)
    tex_right = make_proj("Right", right_img, -600, (math.radians(90), 0, math.radians(90)), (s_val_side, s_val_side, s_val_side))

    # Weighting: Front vs Back based on Normal Y
    # Normal.y < 0 -> Front, Normal.y > 0 -> Back
    norm_y_clamp = nodes.new("ShaderNodeMapRange")
    norm_y_clamp.location = (-400, -150)
    norm_y_clamp.inputs["From Min"].default_value = -0.3
    norm_y_clamp.inputs["From Max"].default_value = 0.3
    norm_y_clamp.inputs["To Min"].default_value = 0.0
    norm_y_clamp.inputs["To Max"].default_value = 1.0
    links.new(sep_norm.outputs["Y"], norm_y_clamp.inputs["Value"])

    mix_fb = nodes.new("ShaderNodeMix")
    mix_fb.data_type = 'RGBA'
    mix_fb.location = (50, 150)
    links.new(norm_y_clamp.outputs["Result"], mix_fb.inputs[0]) # Fac
    links.new(tex_front.outputs["Color"], mix_fb.inputs[6])      # A (Front)
    links.new(tex_back.outputs["Color"], mix_fb.inputs[7])       # B (Back)

    # Weighting: Left vs Right based on Normal X
    norm_x_clamp = nodes.new("ShaderNodeMapRange")
    norm_x_clamp.location = (-400, -450)
    norm_x_clamp.inputs["From Min"].default_value = -0.3
    norm_x_clamp.inputs["From Max"].default_value = 0.3
    norm_x_clamp.inputs["To Min"].default_value = 0.0
    norm_x_clamp.inputs["To Max"].default_value = 1.0
    links.new(sep_norm.outputs["X"], norm_x_clamp.inputs["Value"])

    mix_lr = nodes.new("ShaderNodeMix")
    mix_lr.data_type = 'RGBA'
    mix_lr.location = (50, -350)
    links.new(norm_x_clamp.outputs["Result"], mix_lr.inputs[0]) # Fac
    links.new(tex_left.outputs["Color"], mix_lr.inputs[6])       # A (Left)
    links.new(tex_right.outputs["Color"], mix_lr.inputs[7])      # B (Right)

    # Mix Front/Back with Left/Right based on abs(Normal.X) vs abs(Normal.Y)
    math_abs_x = nodes.new("ShaderNodeMath")
    math_abs_x.operation = 'ABSOLUTE'
    math_abs_x.location = (-200, -250)
    links.new(sep_norm.outputs["X"], math_abs_x.inputs[0])

    math_abs_y = nodes.new("ShaderNodeMath")
    math_abs_y.operation = 'ABSOLUTE'
    math_abs_y.location = (-200, -100)
    links.new(sep_norm.outputs["Y"], math_abs_y.inputs[0])

    side_fac = nodes.new("ShaderNodeMath")
    side_fac.operation = 'GREATER_THAN'
    side_fac.location = (50, -100)
    links.new(math_abs_x.outputs["Value"], side_fac.inputs[0])
    links.new(math_abs_y.outputs["Value"], side_fac.inputs[1])

    mix_final = nodes.new("ShaderNodeMix")
    mix_final.data_type = 'RGBA'
    mix_final.location = (450, 0)
    links.new(side_fac.outputs["Value"], mix_final.inputs[0])
    links.new(mix_fb.outputs[2], mix_final.inputs[6]) # A: Front/Back
    links.new(mix_lr.outputs[2], mix_final.inputs[7]) # B: Left/Right

    # Output to BSDF
    links.new(mix_final.outputs[2], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], out_node.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # 4. Export GLB
    bpy.ops.export_scene.gltf(filepath=glb_out, export_format='GLB')
    print(f"Exported photorealistic GLB: {glb_out}")

    # 5. Studio Lighting & High-End Product Photography Renders
    dist = dim_max * 2.2
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 75
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
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
    rim_data.energy = 260.0
    rim_data.size = dim_max * 1.5
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    rim_obj.location = (0, dist * 0.9, dist * 0.8)
    scene.collection.objects.link(rim_obj)

    # Studio World
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.08, 0.09, 0.11, 1.0)
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
        "final_front": (0, -dist, 0),
        "final_perspective": (dist * 0.65, -dist * 0.85, dist * 0.35),
        "final_side": (dist, 0, 0),
        "final_back": (0, dist, 0),
    }

    for name, loc in views.items():
        cam_obj.location = mathutils.Vector(loc)
        point_at(cam_obj)
        scene.render.filepath = str(renders_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} -> {scene.render.filepath}")

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    in_glb = args[0] if len(args) > 0 else "scratch_headphone_white.glb"
    v_dir = args[1] if len(args) > 1 else "temp/clean_views"
    out_glb = args[2] if len(args) > 2 else "outputs/airpods_max_photorealistic.glb"
    r_dir = args[3] if len(args) > 3 else "temp/renders_multiview"
    setup_multiview_and_render(in_glb, v_dir, out_glb, r_dir)
