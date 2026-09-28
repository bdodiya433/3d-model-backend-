import bpy
import bmesh
import math
import mathutils
from pathlib import Path

def generate_perfect_airpods_max(glb_path: str, render_path: str):
    render_path = Path(render_path)
    render_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 1. Studio Clay Shader (Matte Off-White with soft cavity contrast)
    mat_body = bpy.data.materials.new(name="AirPodsClay")
    mat_body.use_nodes = True
    bsdf = mat_body.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.88, 0.88, 0.90, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.35
        bsdf.inputs["Metallic"].default_value = 0.05

    # Metal accent shader (for stems, crown, hinges)
    mat_steel = bpy.data.materials.new(name="AirPodsSteel")
    mat_steel.use_nodes = True
    bsdf_s = mat_steel.node_tree.nodes.get("Principled BSDF")
    if bsdf_s:
        bsdf_s.inputs["Base Color"].default_value = (0.95, 0.95, 0.96, 1.0)
        bsdf_s.inputs["Roughness"].default_value = 0.10
        bsdf_s.inputs["Metallic"].default_value = 0.85

    # Knit mesh canopy shader
    mat_mesh = bpy.data.materials.new(name="AirPodsMesh")
    mat_mesh.use_nodes = True
    bsdf_m = mat_mesh.node_tree.nodes.get("Principled BSDF")
    if bsdf_m:
        bsdf_m.inputs["Base Color"].default_value = (0.84, 0.84, 0.86, 1.0)
        bsdf_m.inputs["Roughness"].default_value = 0.65
        bsdf_m.inputs["Metallic"].default_value = 0.0

    # 2. Geometry: Build Earcups
    # Proportions based directly on reference photo:
    # Earcup Height: 0.098m, Width: 0.076m, Depth: 0.034m
    def make_cup(is_right=False):
        sign = 1.0 if is_right else -1.0
        suf = "R" if is_right else "L"

        # Base pill cylinder for aluminum shell
        # Start with a rounded cylinder
        bpy.ops.mesh.primitive_cylinder_add(radius=0.038, depth=0.016, vertices=64)
        shell = bpy.context.active_object
        shell.name = f"Shell_{suf}"
        # Stretch along Z to make squircle/pill shape
        shell.scale = (1.0, 1.0, 1.30)
        bpy.ops.object.transform_apply(scale=True)
        # Flatten back face and round front edges
        bev = shell.modifiers.new("Bevel", "BEVEL")
        bev.width = 0.008
        bev.segments = 8
        bpy.ops.object.modifier_apply(modifier="Bevel")

        # Soft Apple contour
        sub = shell.modifiers.new("Subsurf", "SUBSURF")
        sub.levels = 1
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        shell.data.materials.append(mat_body)
        for p in shell.data.polygons:
            p.use_smooth = True

        # Acoustic Cushion Ring
        bpy.ops.mesh.primitive_cylinder_add(radius=0.036, depth=0.018, vertices=64)
        cush = bpy.context.active_object
        cush.name = f"Cushion_{suf}"
        cush.scale = (1.0, 1.0, 1.28)
        bpy.ops.object.transform_apply(scale=True)
        bev_c = cush.modifiers.new("Bevel", "BEVEL")
        bev_c.width = 0.008
        bev_c.segments = 8
        bpy.ops.object.modifier_apply(modifier="Bevel")
        sub_c = cush.modifiers.new("Subsurf", "SUBSURF")
        sub_c.levels = 1
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        cush.data.materials.append(mat_body)
        for p in cush.data.polygons:
            p.use_smooth = True

        # Cushion sits on inner face (facing toward center X)
        cush.location = (-sign * 0.016, 0, 0)
        cush.parent = shell

        # Swivel hinge connector (top of earcup)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0055, depth=0.012, vertices=32)
        swivel = bpy.context.active_object
        swivel.name = f"Swivel_{suf}"
        swivel.location = (0, 0, 0.048)
        swivel.data.materials.append(mat_steel)
        for p in swivel.data.polygons:
            p.use_smooth = True
        swivel.parent = shell

        # Controls on Right earcup
        if is_right:
            # Digital Crown: rotary dial at top-back
            bpy.ops.mesh.primitive_cylinder_add(radius=0.0050, depth=0.005, vertices=32)
            crown = bpy.context.active_object
            crown.name = "DigitalCrown"
            crown.rotation_euler = (math.radians(-25), 0, math.radians(-30))
            crown.location = (0.012, -0.015, 0.046)
            crown.data.materials.append(mat_steel)
            for p in crown.data.polygons:
                p.use_smooth = True
            crown.parent = shell

            # Noise Control Button (pill toggle)
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            btn = bpy.context.active_object
            btn.name = "NoiseButton"
            btn.scale = (0.003, 0.011, 0.003)
            bpy.ops.object.transform_apply(scale=True)
            bev_b = btn.modifiers.new("Bevel", "BEVEL")
            bev_b.width = 0.0012
            bev_b.segments = 4
            bpy.ops.object.modifier_apply(modifier="Bevel")
            btn.rotation_euler = (math.radians(-25), 0, math.radians(-30))
            btn.location = (0.010, 0.008, 0.048)
            btn.data.materials.append(mat_body)
            btn.parent = shell

        # Upright orientation:
        # Earcups stand upright, cushion points inward toward center X
        # Earcup cylinder has depth along Z by default. We rotate it so depth is along X:
        # Rotation: Y=90 places cylinder axis along X (cushion pointing toward -sign*X)
        shell.rotation_euler = (math.radians(8.0), math.radians(90.0 + 14.0 * sign), 0)
        shell.location = (sign * 0.070, 0.0, -0.015)
        return shell

    cup_r = make_cup(is_right=True)
    cup_l = make_cup(is_right=False)

    # 3. Telescopic Stainless Steel Stems & Headband Collars
    for sign in [-1.0, 1.0]:
        suf = "R" if sign > 0 else "L"
        # Telescoping steel rod
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0034, depth=0.045, vertices=32)
        stem = bpy.context.active_object
        stem.name = f"Stem_{suf}"
        stem.rotation_euler = (0, math.radians(9.0 * sign), 0)
        stem.location = (sign * 0.073, 0.0, 0.044)
        stem.data.materials.append(mat_steel)
        for p in stem.data.polygons:
            p.use_smooth = True

        # Headband silicone collar
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0055, depth=0.016, vertices=32)
        collar = bpy.context.active_object
        collar.name = f"Collar_{suf}"
        collar.rotation_euler = (0, math.radians(9.0 * sign), 0)
        collar.location = (sign * 0.068, 0.0, 0.066)
        collar.data.materials.append(mat_body)
        for p in collar.data.polygons:
            p.use_smooth = True

    # 4. Headband Frame & Breathable Mesh Canopy
    # Twin curved tubular frame
    def make_arch(y_off, name):
        c_data = bpy.data.curves.new(name, type='CURVE')
        c_data.dimensions = '3D'
        c_data.resolution_u = 48
        spline = c_data.splines.new('BEZIER')
        spline.bezier_points.add(2)

        p0 = spline.bezier_points[0]
        p0.co = (-0.067, y_off, 0.072)
        p0.handle_left = (-0.067, y_off, 0.050)
        p0.handle_right = (-0.067, y_off, 0.125)

        p1 = spline.bezier_points[1]
        p1.co = (0.0, y_off, 0.146)
        p1.handle_left = (-0.054, y_off, 0.146)
        p1.handle_right = (0.054, y_off, 0.146)

        p2 = spline.bezier_points[2]
        p2.co = (0.067, y_off, 0.072)
        p2.handle_left = (0.067, y_off, 0.125)
        p2.handle_right = (0.067, y_off, 0.050)

        c_data.bevel_depth = 0.0036
        c_data.bevel_resolution = 8
        c_obj = bpy.data.objects.new(name, c_data)
        c_obj.data.materials.append(mat_body)
        scene.collection.objects.link(c_obj)
        return c_obj

    arch_f = make_arch(-0.012, "ArchFront")
    arch_b = make_arch(0.012, "ArchBack")

    # Breathable Knit Mesh Canopy ribbon between arches
    canopy_mesh = bpy.data.meshes.new("CanopyMesh")
    canopy_obj = bpy.data.objects.new("CanopyMesh", canopy_mesh)
    scene.collection.objects.link(canopy_obj)

    bm = bmesh.new()
    steps = 48
    verts = []
    for i in range(steps + 1):
        t = i / steps
        th = math.pi * (1.0 - t)
        x = 0.066 * math.cos(th)
        z = 0.073 + 0.071 * math.sin(th)
        sag = 0.002 * math.sin(th)
        vf = bm.verts.new((x, -0.010, z - sag))
        vb = bm.verts.new((x, 0.010, z - sag))
        verts.append((vf, vb))

    bm.verts.ensure_lookup_table()
    for i in range(steps):
        vf0, vb0 = verts[i]
        vf1, vb1 = verts[i + 1]
        bm.faces.new((vf0, vf1, vb1, vb0))

    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.0012)
    bm.to_mesh(canopy_mesh)
    bm.free()

    sub_k = canopy_obj.modifiers.new("Subsurf", "SUBSURF")
    sub_k.levels = 1
    bpy.context.view_layer.objects.active = canopy_obj
    bpy.ops.object.modifier_apply(modifier="Subsurf")
    for p in canopy_obj.data.polygons:
        p.use_smooth = True
    canopy_obj.data.materials.append(mat_mesh)

    # 5. Join all objects
    bpy.ops.object.select_all(action='DESELECT')
    for a in [arch_f, arch_b]:
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
    final_headphone.name = "AppleAirPodsMax"
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    final_headphone.location = (0, 0, 0)

    # Export GLB
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB')
    print(f"Exported: {glb_path}")

    # 6. Realistic Studio Lighting (Balanced for 20cm object: 12 Watts)
    coords = [final_headphone.matrix_world @ mathutils.Vector(c) for c in final_headphone.bound_box]
    dim_max = max(max(c.x for c in coords) - min(c.x for c in coords),
                  max(c.z for c in coords) - min(c.z for c in coords))
    dist = dim_max * 2.3

    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 75
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Soft studio lights
    key_d = bpy.data.lights.new("Key", 'AREA')
    key_d.energy = 16.0
    key_d.size = 0.40
    key = bpy.data.objects.new("Key", key_d)
    key.location = (dist * 0.75, -dist * 0.85, dist * 0.7)
    scene.collection.objects.link(key)

    fill_d = bpy.data.lights.new("Fill", 'AREA')
    fill_d.energy = 7.0
    fill_d.size = 0.50
    fill = bpy.data.objects.new("Fill", fill_d)
    fill.location = (-dist * 0.85, -dist * 0.6, dist * 0.3)
    scene.collection.objects.link(fill)

    rim_d = bpy.data.lights.new("Rim", 'AREA')
    rim_d.energy = 11.0
    rim_d.size = 0.35
    rim = bpy.data.objects.new("Rim", rim_d)
    rim.location = (0, dist * 0.9, dist * 0.8)
    scene.collection.objects.link(rim)

    # World
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.07, 0.08, 0.10, 1.0)
        bg.inputs["Strength"].default_value = 0.8

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    def point_at(target, center_pt=(0,0,0)):
        d = mathutils.Vector(center_pt) - target.location
        target.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

    point_at(key)
    point_at(fill)
    point_at(rim)

    # Render Front and Perspective
    out_dir = render_path.parent
    for name, loc in [("cad_front", (0, -dist, 0)), ("cad_perspective", (dist * 0.65, -dist * 0.85, dist * 0.35)), ("cad_side", (dist, 0, 0))]:
        cam_obj.location = mathutils.Vector(loc)
        point_at(cam_obj)
        scene.render.filepath = str(out_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name}")

if __name__ == "__main__":
    generate_perfect_airpods_max("outputs/airpods_max_cad_perfect.glb", "temp/renders_cad/cad_front.png")
