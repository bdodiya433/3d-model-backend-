import bpy
import sys
import math
from pathlib import Path

def render_white_clay(glb_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 2. Import GLB
    bpy.ops.import_scene.gltf(filepath=glb_path)
    
    # Find imported mesh objects
    mesh_objs = [obj for obj in scene.objects if obj.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No mesh objects found in imported GLB")

    # Create clay material
    mat = bpy.data.materials.new(name="WhiteClay")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        # Beautiful clean studio white
        bsdf.inputs["Base Color"].default_value = (0.96, 0.96, 0.97, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.25
        bsdf.inputs["Metallic"].default_value = 0.05

    import mathutils
    coords = [obj.matrix_world @ mathutils.Vector(c) for obj in mesh_objs for c in obj.bound_box]
    for obj in mesh_objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for p in obj.data.polygons:
            p.use_smooth = True
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)

    center = mathutils.Vector(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0))
    dim_max = max(max_x - min_x, max_y - min_y, max_z - min_z)
    dist = dim_max * 2.1

    # Camera setup
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 65  # 65mm portrait lens (no perspective distortion)
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Studio Lighting (bright, crisp studio)
    # Key light
    key_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_data.energy = 220.0
    key_data.size = dim_max * 2.0
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    key_obj.location = (center.x + dist * 0.7, center.y - dist * 0.8, center.z + dist * 0.9)
    scene.collection.objects.link(key_obj)

    # Fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 110.0
    fill_data.size = dim_max * 2.5
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    fill_obj.location = (center.x - dist * 0.8, center.y - dist * 0.6, center.z + dist * 0.4)
    scene.collection.objects.link(fill_obj)

    # Rim / Back light
    rim_data = bpy.data.lights.new(name="RimLight", type='AREA')
    rim_data.energy = 140.0
    rim_data.size = dim_max * 1.5
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    rim_obj.location = (center.x, center.y + dist * 0.9, center.z + dist * 0.9)
    scene.collection.objects.link(rim_obj)

    # World background: clean studio soft gradient/light
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.12, 0.13, 0.15, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0

    # Render settings
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.film_transparent = False

    def point_at(obj, target):
        direction = target - obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        obj.rotation_euler = rot_quat.to_euler()

    point_at(key_obj, center)
    point_at(fill_obj, center)
    point_at(rim_obj, center)

    # Views to render
    views = {
        "perspective": (center.x + dist * 0.7, center.y - dist * 0.9, center.z + dist * 0.4),
        "front": (center.x, center.y - dist, center.z),
        "back": (center.x, center.y + dist, center.z),
        "side": (center.x + dist, center.y, center.z),
        "top": (center.x, center.y - 0.001, center.z + dist),
    }

    for name, loc in views.items():
        cam_obj.location = mathutils.Vector(loc)
        point_at(cam_obj, center)
        scene.render.filepath = str(output_dir / f"clay_{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} view -> {scene.render.filepath}")

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glb = args[0] if len(args) > 0 else "scratch_headphone_white.glb"
    out = args[1] if len(args) > 1 else "temp/renders_clay"
    render_white_clay(glb, out)
