import bpy
import math
import mathutils
from pathlib import Path

def refine_and_render_apple_airpods(glb_input: str, glb_output: str, render_output_dir: str):
    render_output_dir = Path(render_output_dir)
    render_output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 1. Import raw GLB
    bpy.ops.import_scene.gltf(filepath=glb_input)
    mesh_objs = [obj for obj in scene.objects if obj.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No mesh found")
    obj = mesh_objs[0]
    bpy.context.view_layer.objects.active = obj
    obj.name = "AirPodsMax"

    print(f"Initial raw polygons: {len(obj.data.polygons)}")

    # 2. Smooth voxel noise (Laplacian / Smooth modifier)
    smooth_mod = obj.modifiers.new("Smooth", "SMOOTH")
    smooth_mod.factor = 0.8
    smooth_mod.iterations = 12
    bpy.ops.object.modifier_apply(modifier="Smooth")

    # 3. Decimate to clean CAD density (~50,000 polys)
    n_poly = len(obj.data.polygons)
    target_poly = 55000
    if n_poly > target_poly:
        dec_mod = obj.modifiers.new("Decimate", "DECIMATE")
        dec_mod.ratio = float(target_poly) / float(n_poly)
        bpy.ops.object.modifier_apply(modifier="Decimate")

    # Second pass subtle smooth to relax decimated vertices
    smooth_mod2 = obj.modifiers.new("Smooth", "SMOOTH")
    smooth_mod2.factor = 0.5
    smooth_mod2.iterations = 4
    bpy.ops.object.modifier_apply(modifier="Smooth")

    # Auto smooth normals for crisp industrial look
    for p in obj.data.polygons:
        p.use_smooth = True

    # 4. Apple Realistic PBR Materials & Shaders
    # A. Anodized Silver Aluminum (Body / Earcups)
    mat_al = bpy.data.materials.new(name="AnodizedAluminum")
    mat_al.use_nodes = True
    bsdf_al = mat_al.node_tree.nodes.get("Principled BSDF")
    if bsdf_al:
        bsdf_al.inputs["Base Color"].default_value = (0.86, 0.87, 0.89, 1.0)
        bsdf_al.inputs["Metallic"].default_value = 0.94
        bsdf_al.inputs["Roughness"].default_value = 0.22

    # B. Acoustic Fabric Cushion (Soft Memory Foam Fabric)
    mat_cushion = bpy.data.materials.new(name="AcousticFabric")
    mat_cushion.use_nodes = True
    bsdf_cushion = mat_cushion.node_tree.nodes.get("Principled BSDF")
    if bsdf_cushion:
        bsdf_cushion.inputs["Base Color"].default_value = (0.78, 0.79, 0.81, 1.0)
        bsdf_cushion.inputs["Metallic"].default_value = 0.0
        bsdf_cushion.inputs["Roughness"].default_value = 0.75

    # C. Breathable Mesh Canopy (Headband Knit)
    mat_mesh = bpy.data.materials.new(name="KnitCanopy")
    mat_mesh.use_nodes = True
    bsdf_mesh = mat_mesh.node_tree.nodes.get("Principled BSDF")
    if bsdf_mesh:
        bsdf_mesh.inputs["Base Color"].default_value = (0.82, 0.83, 0.85, 1.0)
        bsdf_mesh.inputs["Metallic"].default_value = 0.0
        bsdf_mesh.inputs["Roughness"].default_value = 0.80

    # D. Polished Stainless Steel (Telescoping Arms & Digital Crown)
    mat_steel = bpy.data.materials.new(name="PolishedSteel")
    mat_steel.use_nodes = True
    bsdf_steel = mat_steel.node_tree.nodes.get("Principled BSDF")
    if bsdf_steel:
        bsdf_steel.inputs["Base Color"].default_value = (0.95, 0.96, 0.97, 1.0)
        bsdf_steel.inputs["Metallic"].default_value = 1.0
        bsdf_steel.inputs["Roughness"].default_value = 0.08

    # E. Silicone Matte Frame (Headband Arch)
    mat_silicone = bpy.data.materials.new(name="SiliconeWhite")
    mat_silicone.use_nodes = True
    bsdf_sil = mat_silicone.node_tree.nodes.get("Principled BSDF")
    if bsdf_sil:
        bsdf_sil.inputs["Base Color"].default_value = (0.92, 0.93, 0.94, 1.0)
        bsdf_sil.inputs["Metallic"].default_value = 0.0
        bsdf_sil.inputs["Roughness"].default_value = 0.38

    # Assign materials to slots
    obj.data.materials.clear()
    obj.data.materials.append(mat_al)       # Slot 0: Aluminum body
    obj.data.materials.append(mat_cushion)  # Slot 1: Cushions
    obj.data.materials.append(mat_mesh)     # Slot 2: Knit Canopy
    obj.data.materials.append(mat_steel)    # Slot 3: Steel stems
    obj.data.materials.append(mat_silicone) # Slot 4: Silicone headband

    # Intelligently assign polygons by spatial zones & surface normals
    coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)
    span_z = max_z - min_z
    span_x = max_x - min_x
    mid_x = (min_x + max_x) / 2.0
    mid_z = (min_z + max_z) / 2.0

    print("Assigning Apple product material zones...")
    for poly in obj.data.polygons:
        # Compute face center in local/world coords
        f_center = obj.matrix_world @ poly.center
        f_norm = obj.matrix_world.to_3x3() @ poly.normal
        f_norm.normalize()

        # Headband zone: top 35% of model (Z > min_z + 0.65 * span_z)
        if f_center.z > min_z + 0.62 * span_z:
            # Topmost canopy vs outer silicone rim
            if f_center.z > min_z + 0.88 * span_z and abs(f_center.x - mid_x) < 0.35 * span_x:
                poly.material_index = 2  # Knit canopy
            else:
                poly.material_index = 4  # Silicone headband frame
        # Telescoping stem zone: transition between earcups and headband
        elif f_center.z > min_z + 0.48 * span_z and f_center.z <= min_z + 0.62 * span_z:
            poly.material_index = 3  # Polished steel
        # Earcups zone: bottom 50%
        else:
            # Cushion vs Aluminum shell
            # Cushions face inward (normal points toward center X)
            points_inward = (f_center.x > mid_x and f_norm.x < -0.2) or (f_center.x < mid_x and f_norm.x > 0.2)
            if points_inward:
                poly.material_index = 1  # Acoustic Fabric cushion
            else:
                poly.material_index = 0  # Anodized Aluminum shell

    # Export refined GLB
    bpy.ops.export_scene.gltf(filepath=glb_output, export_format='GLB')
    print(f"Refined Apple GLB saved to: {glb_output}")

    # 5. Studio Lighting & Photorealistic Product Rendering
    center = mathutils.Vector((mid_x, (min_y + max_y) / 2.0, mid_z))
    dim_max = max(max_x - min_x, max_y - min_y, max_z - min_z)
    dist = dim_max * 2.2

    cam_data = bpy.data.cameras.new("StudioCam")
    cam_data.lens = 75  # 75mm portrait lens (crisp telephoto product shot)
    cam_obj = bpy.data.objects.new("StudioCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Studio Lighting Setup (Apple High-Key Studio Lighting)
    # Key light: Soft Area Light from top-front-right
    key_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_data.energy = 380.0
    key_data.size = dim_max * 1.5
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    key_obj.location = (center.x + dist * 0.8, center.y - dist * 0.9, center.z + dist * 0.7)
    scene.collection.objects.link(key_obj)

    # Fill light: Gentle diffuse fill from front-left
    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 160.0
    fill_data.size = dim_max * 2.0
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    fill_obj.location = (center.x - dist * 0.9, center.y - dist * 0.6, center.z + dist * 0.3)
    scene.collection.objects.link(fill_obj)

    # Rim light / Edge highlight from behind
    rim_data = bpy.data.lights.new(name="RimLight", type='AREA')
    rim_data.energy = 240.0
    rim_data.size = dim_max * 1.2
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    rim_obj.location = (center.x, center.y + dist * 0.9, center.z + dist * 0.8)
    scene.collection.objects.link(rim_obj)

    # Studio World: Apple gradient dark studio backdrop
    scene.world = bpy.data.worlds.new("AppleStudioWorld")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.08, 0.09, 0.11, 1.0)
        bg_node.inputs["Strength"].default_value = 0.8

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    def point_at(light_or_cam, target):
        direction = target - light_or_cam.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        light_or_cam.rotation_euler = rot_quat.to_euler()

    point_at(key_obj, center)
    point_at(fill_obj, center)
    point_at(rim_obj, center)

    render_views = {
        "apple_front": (center.x, center.y - dist, center.z),
        "apple_perspective": (center.x + dist * 0.68, center.y - dist * 0.88, center.z + dist * 0.38),
        "apple_side": (center.x + dist, center.y, center.z),
        "apple_cushion_detail": (center.x - dist * 0.55, center.y - dist * 0.75, center.z - dist * 0.15),
    }

    for name, loc in render_views.items():
        cam_obj.location = mathutils.Vector(loc)
        point_at(cam_obj, center)
        scene.render.filepath = str(render_output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} -> {scene.render.filepath}")

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    in_glb = args[0] if len(args) > 0 else "scratch_headphone_white.glb"
    out_glb = args[1] if len(args) > 1 else "outputs/apple_airpods_max_pbr.glb"
    renders = args[2] if len(args) > 2 else "temp/renders_apple"
    refine_and_render_apple_airpods(in_glb, out_glb, renders)
