import bpy
import bmesh
import math
import mathutils
from pathlib import Path

def create_masterpiece_airpods(output_glb: str, render_output_dir: str):
    render_output_dir = Path(render_output_dir)
    render_output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 1. Materials
    # Premium Matte White Clay / Industrial Design Studio Finish
    mat_clay = bpy.data.materials.new(name="AirPodsClay")
    mat_clay.use_nodes = True
    bsdf = mat_clay.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.90, 0.90, 0.92, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.32
        bsdf.inputs["Metallic"].default_value = 0.02

    # Polished Steel Accents (stems, crown, hinge)
    mat_metal = bpy.data.materials.new(name="PolishedSteel")
    mat_metal.use_nodes = True
    bsdf_m = mat_metal.node_tree.nodes.get("Principled BSDF")
    if bsdf_m:
        bsdf_m.inputs["Base Color"].default_value = (0.95, 0.95, 0.96, 1.0)
        bsdf_m.inputs["Roughness"].default_value = 0.12
        bsdf_m.inputs["Metallic"].default_value = 0.90

    # Breathable Mesh Canopy
    mat_mesh = bpy.data.materials.new(name="KnitMesh")
    mat_mesh.use_nodes = True
    bsdf_mesh = mat_mesh.node_tree.nodes.get("Principled BSDF")
    if bsdf_mesh:
        bsdf_mesh.inputs["Base Color"].default_value = (0.86, 0.86, 0.88, 1.0)
        bsdf_mesh.inputs["Roughness"].default_value = 0.65
        bsdf_mesh.inputs["Metallic"].default_value = 0.0

    # 2. Build Single Earcup Assembly (Pill-shaped Aluminum Shell + Acoustic Cushion + Controls)
    def build_earcup_assembly(is_right=False):
        sign = 1.0 if is_right else -1.0
        suf = "R" if is_right else "L"

        # Aluminum Shell: rounded pill shape (taller along Z, curved perimeter)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.038, depth=0.016, vertices=64)
        shell = bpy.context.active_object
        shell.name = f"Shell_{suf}"
        # Scale: elongated along Y (when standing) or along Z
        shell.scale = (1.0, 1.34, 1.0)
        bpy.ops.object.transform_apply(scale=True)
        bev = shell.modifiers.new("Bevel", "BEVEL")
        bev.width = 0.007
        bev.segments = 8
        bpy.ops.object.modifier_apply(modifier="Bevel")
        sub = shell.modifiers.new("Subsurf", "SUBSURF")
        sub.levels = 1
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        for p in shell.data.polygons:
            p.use_smooth = True
        shell.data.materials.append(mat_clay)

        # Acoustic Cushion Ring (cushion face)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.036, depth=0.016, vertices=64)
        cush = bpy.context.active_object
        cush.name = f"Cushion_{suf}"
        cush.scale = (0.97, 1.32, 1.0)
        bpy.ops.object.transform_apply(scale=True)
        bev_c = cush.modifiers.new("Bevel", "BEVEL")
        bev_c.width = 0.008
        bev_c.segments = 8
        bpy.ops.object.modifier_apply(modifier="Bevel")
        sub_c = cush.modifiers.new("Subsurf", "SUBSURF")
        sub_c.levels = 1
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        for p in cush.data.polygons:
            p.use_smooth = True
        cush.data.materials.append(mat_clay)
        # Position cushion slightly forward (along local Z)
        cush.location = (0, 0, -0.012)

        # Swivel Hinge Mount (top edge: Y = +0.048)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0055, depth=0.012, vertices=32)
        hinge = bpy.context.active_object
        hinge.name = f"Hinge_{suf}"
        hinge.rotation_euler = (math.radians(90), 0, 0)
        hinge.location = (0, 0.052, 0)
        hinge.data.materials.append(mat_metal)
        for p in hinge.data.polygons:
            p.use_smooth = True

        parts = [shell, cush, hinge]

        # Right earcup controls
        if is_right:
            # Digital Crown: rotary knob
            bpy.ops.mesh.primitive_cylinder_add(radius=0.0052, depth=0.006, vertices=32)
            crown = bpy.context.active_object
            crown.name = "DigitalCrown"
            crown.rotation_euler = (math.radians(45), math.radians(25), 0)
            crown.location = (0.015, 0.048, 0.004)
            crown.data.materials.append(mat_metal)
            for p in crown.data.polygons:
                p.use_smooth = True
            parts.append(crown)

            # Noise Control Button: pill button
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            btn = bpy.context.active_object
            btn.name = "NoiseBtn"
            btn.scale = (0.003, 0.012, 0.003)
            bpy.ops.object.transform_apply(scale=True)
            bev_b = btn.modifiers.new("Bevel", "BEVEL")
            bev_b.width = 0.0012
            bev_b.segments = 4
            bpy.ops.object.modifier_apply(modifier="Bevel")
            btn.rotation_euler = (math.radians(45), math.radians(25), 0)
            btn.location = (-0.008, 0.050, 0.004)
            btn.data.materials.append(mat_clay)
            for p in btn.data.polygons:
                p.use_smooth = True
            parts.append(btn)

        # Bottom charging port
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0028, depth=0.005, vertices=16)
        port = bpy.context.active_object
        port.name = f"Port_{suf}"
        port.rotation_euler = (math.radians(90), 0, 0)
        port.location = (0, -0.050, 0)
        port.data.materials.append(mat_metal)
        parts.append(port)

        # Join this earcup's parts into a single unified object
        bpy.ops.object.select_all(action='DESELECT')
        for p in parts:
            p.select_set(True)
        bpy.context.view_layer.objects.active = shell
        bpy.ops.object.join()
        cup_obj = bpy.context.active_object
        cup_obj.name = f"Earcup_{suf}"

        # Position and orient in world:
        # Currently, earcup is in XY plane (Y is height, Z is depth).
        # We want: Z is world height, X is world width (left-right), Y is world depth (front-back).
        # Rotate X by 90 degrees so Y becomes Z!
        # And cushion faces inward: for Right (+X), cushion is along -X. For Left (-X), cushion is along +X.
        cup_obj.rotation_euler = (math.radians(90), math.radians(16.0 * sign), math.radians(-90.0 * sign))
        cup_obj.location = (sign * 0.076, 0.0, -0.015)
        return cup_obj

    cup_r = build_earcup_assembly(is_right=True)
    cup_l = build_earcup_assembly(is_right=False)

    # 3. Telescopic Stainless Steel Stems
    # They connect the earcup swivel hinge at (sign*0.076, 0, 0.038)
    # up into the headband collar at (sign*0.066, 0, 0.072)
    for sign in [-1.0, 1.0]:
        suf = "R" if sign > 0 else "L"
        # Stainless steel telescoping arm
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0034, depth=0.042, vertices=32)
        stem = bpy.context.active_object
        stem.name = f"Stem_{suf}"
        # Tilt angle matching the slope between earcup top and headband collar
        tilt_angle = math.atan2(0.076 - 0.066, 0.072 - 0.038) # ~16 degrees
        stem.rotation_euler = (0, tilt_angle * sign, 0)
        stem.location = (sign * 0.071, 0.0, 0.052)
        stem.data.materials.append(mat_metal)
        for p in stem.data.polygons:
            p.use_smooth = True

        # Headband Silicone Collar
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0054, depth=0.015, vertices=32)
        collar = bpy.context.active_object
        collar.name = f"Collar_{suf}"
        collar.rotation_euler = (0, tilt_angle * sign, 0)
        collar.location = (sign * 0.066, 0.0, 0.070)
        collar.data.materials.append(mat_clay)
        for p in collar.data.polygons:
            p.use_smooth = True

    # 4. Headband Arches & Breathable Knit Mesh Canopy
    def make_arch_tube(y_off, name):
        c_data = bpy.data.curves.new(name, type='CURVE')
        c_data.dimensions = '3D'
        c_data.resolution_u = 48
        spline = c_data.splines.new('BEZIER')
        spline.bezier_points.add(2)

        # Anchor points meet collar exactly at Z=0.074
        p0 = spline.bezier_points[0]
        p0.co = (-0.066, y_off, 0.074)
        p0.handle_left = (-0.066, y_off, 0.050)
        p0.handle_right = (-0.066, y_off, 0.125)

        # Apex at Z=0.148
        p1 = spline.bezier_points[1]
        p1.co = (0.0, y_off, 0.148)
        p1.handle_left = (-0.052, y_off, 0.148)
        p1.handle_right = (0.052, y_off, 0.148)

        p2 = spline.bezier_points[2]
        p2.co = (0.066, y_off, 0.074)
        p2.handle_left = (0.066, y_off, 0.125)
        p2.handle_right = (0.066, y_off, 0.050)

        c_data.bevel_depth = 0.0036
        c_data.bevel_resolution = 8
        c_obj = bpy.data.objects.new(name, c_data)
        c_obj.data.materials.append(mat_clay)
        scene.collection.objects.link(c_obj)
        return c_obj

    arch_front = make_arch_tube(-0.012, "Headband_Front")
    arch_back = make_arch_tube(0.012, "Headband_Back")

    # Knit Canopy Ribbon
    canopy_mesh = bpy.data.meshes.new("CanopyMesh")
    canopy_obj = bpy.data.objects.new("CanopyMesh", canopy_mesh)
    scene.collection.objects.link(canopy_obj)

    bm_k = bmesh.new()
    steps = 48
    verts = []
    for i in range(steps + 1):
        t = i / steps
        th = math.pi * (1.0 - t)
        x = 0.065 * math.cos(th)
        z = 0.074 + 0.071 * math.sin(th)
        # Gentle sag
        sag = 0.002 * math.sin(th)
        vf = bm_k.verts.new((x, -0.010, z - sag))
        vb = bm_k.verts.new((x, 0.010, z - sag))
        verts.append((vf, vb))

    bm_k.verts.ensure_lookup_table()
    for i in range(steps):
        vf0, vb0 = verts[i]
        vf1, vb1 = verts[i + 1]
        bm_k.faces.new((vf0, vf1, vb1, vb0))

    bmesh.ops.solidify(bm_k, geom=bm_k.faces[:], thickness=0.0014)
    bm_k.to_mesh(canopy_mesh)
    bm_k.free()

    sub_k = canopy_obj.modifiers.new("Subsurf", "SUBSURF")
    sub_k.levels = 1
    bpy.context.view_layer.objects.active = canopy_obj
    bpy.ops.object.modifier_apply(modifier="Subsurf")
    for p in canopy_obj.data.polygons:
        p.use_smooth = True
    canopy_obj.data.materials.append(mat_mesh)

    # 5. Join into Single Model & Export GLB
    bpy.ops.object.select_all(action='DESELECT')
    for a in [arch_front, arch_back]:
        a.select_set(True)
        bpy.context.view_layer.objects.active = a
        bpy.ops.object.convert(target='MESH')
        a.select_set(False)

    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)

    bpy.context.view_layer.objects.active = cup_r
    bpy.ops.object.join()
    final_headphone = bpy.context.active_object
    final_headphone.name = "AppleAirPodsMax_Precision"

    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    final_headphone.location = (0, 0, 0)

    # Export GLB
    bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')
    print(f"Exported masterpiece GLB: {output_glb}")

    # 6. Studio Lighting & High-Resolution Multi-Angle Renders
    coords = [final_headphone.matrix_world @ mathutils.Vector(c) for c in final_headphone.bound_box]
    dim_max = max(max(c.x for c in coords) - min(c.x for c in coords),
                  max(c.z for c in coords) - min(c.z for c in coords))
    dist = dim_max * 2.25

    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 80
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Key light: Soft wide area light
    key_d = bpy.data.lights.new("Key", 'AREA')
    key_d.energy = 18.0
    key_d.size = 0.40
    key = bpy.data.objects.new("Key", key_d)
    key.location = (dist * 0.75, -dist * 0.85, dist * 0.7)
    scene.collection.objects.link(key)

    # Fill light: Soft fill from front-left
    fill_d = bpy.data.lights.new("Fill", 'AREA')
    fill_d.energy = 8.0
    fill_d.size = 0.50
    fill = bpy.data.objects.new("Fill", fill_d)
    fill.location = (-dist * 0.85, -dist * 0.6, dist * 0.3)
    scene.collection.objects.link(fill)

    # Rim light: Highlights edges from behind
    rim_d = bpy.data.lights.new("Rim", 'AREA')
    rim_d.energy = 14.0
    rim_d.size = 0.35
    rim = bpy.data.objects.new("Rim", rim_d)
    rim.location = (0, dist * 0.9, dist * 0.8)
    scene.collection.objects.link(rim)

    # Studio World
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.08, 0.09, 0.11, 1.0)
        bg.inputs["Strength"].default_value = 0.85

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    def point_at(target, center_pt=(0,0,0)):
        d = mathutils.Vector(center_pt) - target.location
        target.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

    point_at(key)
    point_at(fill)
    point_at(rim)

    render_views = {
        "master_front": (0, -dist, 0),
        "master_perspective": (dist * 0.65, -dist * 0.85, dist * 0.35),
        "master_side": (dist, 0, 0),
    }

    for name, loc in render_views.items():
        cam_obj.location = mathutils.Vector(loc)
        point_at(cam_obj)
        scene.render.filepath = str(render_output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} -> {scene.render.filepath}")

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glb = args[0] if len(args) > 0 else "outputs/airpods_max_masterpiece.glb"
    r_dir = args[1] if len(args) > 1 else "temp/renders_master"
    create_masterpiece_airpods(glb, r_dir)
