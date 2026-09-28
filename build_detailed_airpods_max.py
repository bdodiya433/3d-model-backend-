import bpy
import bmesh
import math
import mathutils
from pathlib import Path

def build_detailed_airpods_max(output_glb: str, output_renders_dir: str):
    output_renders_dir = Path(output_renders_dir)
    output_renders_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 2. Materials Setup
    # Matte White Clay Material (Clean Studio Product Design)
    mat_clay = bpy.data.materials.new(name="StudioClayWhite")
    mat_clay.use_nodes = True
    bsdf_clay = mat_clay.node_tree.nodes.get("Principled BSDF")
    if bsdf_clay:
        bsdf_clay.inputs["Base Color"].default_value = (0.94, 0.94, 0.95, 1.0)
        bsdf_clay.inputs["Roughness"].default_value = 0.32
        bsdf_clay.inputs["Metallic"].default_value = 0.0

    # Polished Metal / Accent (for stems and crown)
    mat_metal = bpy.data.materials.new(name="StudioMetalWhite")
    mat_metal.use_nodes = True
    bsdf_metal = mat_metal.node_tree.nodes.get("Principled BSDF")
    if bsdf_metal:
        bsdf_metal.inputs["Base Color"].default_value = (0.96, 0.96, 0.97, 1.0)
        bsdf_metal.inputs["Roughness"].default_value = 0.12
        bsdf_metal.inputs["Metallic"].default_value = 0.80

    # Mesh Canopy Fabric (slightly softer matte)
    mat_canopy = bpy.data.materials.new(name="StudioCanopyMesh")
    mat_canopy.use_nodes = True
    bsdf_canopy = mat_canopy.node_tree.nodes.get("Principled BSDF")
    if bsdf_canopy:
        bsdf_canopy.inputs["Base Color"].default_value = (0.91, 0.91, 0.92, 1.0)
        bsdf_canopy.inputs["Roughness"].default_value = 0.70
        bsdf_canopy.inputs["Metallic"].default_value = 0.0

    # Collection for organized parts
    airpods_col = bpy.data.collections.new("AirPodsMax")
    scene.collection.children.link(airpods_col)

    # --- HELPER: Squircle Profile (AirPods Max Earcup Form) ---
    # In Apple design, the earcup profile is a superellipse / rounded rectangle
    # Width ~ 74mm, Height ~ 95mm, Corner radius ~ 34mm
    def create_squircle_curve(w=0.076, h=0.096, r=0.032, n_pts=64):
        pts = []
        hx = w / 2.0 - r
        hz = h / 2.0 - r
        # 4 corners
        corners = [
            (hx, hz, 0, math.pi / 2),
            (-hx, hz, math.pi / 2, math.pi),
            (-hx, -hz, math.pi, 3 * math.pi / 2),
            (hx, -hz, 3 * math.pi / 2, 2 * math.pi)
        ]
        pts_per_corner = n_pts // 4
        for cx, cz, a_start, a_end in corners:
            for i in range(pts_per_corner):
                t = a_start + (a_end - a_start) * (i / pts_per_corner)
                x = cx + r * math.cos(t)
                z = cz + r * math.sin(t)
                pts.append((x, z))
        return pts

    # --- 3. BUILD EARCUPS ---
    def build_earcup(is_right=False):
        sign = 1.0 if is_right else -1.0
        suffix = "R" if is_right else "L"

        # A. Aluminum Shell
        # We model the outer shell by lofting concentric squircle rings
        mesh_shell = bpy.data.meshes.new(f"Shell_{suffix}")
        obj_shell = bpy.data.objects.new(mesh_shell.name, mesh_shell)
        airpods_col.objects.link(obj_shell)

        bm = bmesh.new()
        squircle_pts = create_squircle_curve(w=0.076, h=0.096, r=0.032, n_pts=48)
        n = len(squircle_pts)

        # Cross-section profiles along depth (Y axis):
        # Y = 0 (outer back face cap) to Y = 0.024 (front rim where cushion attaches)
        y_steps = [
            (0.000, 0.88, 0.0),    # Flat back center
            (0.004, 0.96, 0.0),    # Chamfer curve
            (0.010, 1.00, 0.0),    # Full outer body
            (0.022, 1.00, 0.0),    # Main cylinder
            (0.025, 0.98, -0.004)  # Inner lip rim
        ]

        ring_verts = []
        for y, scale, offset in y_steps:
            r_verts = []
            for x, z in squircle_pts:
                v = bm.verts.new((x * scale, y, z * scale + offset))
                r_verts.append(v)
            ring_verts.append(r_verts)

        bm.verts.ensure_lookup_table()

        # Fill back face (first ring)
        bm.faces.new(ring_verts[0])

        # Bridge rings
        for r_idx in range(len(y_steps) - 1):
            r1 = ring_verts[r_idx]
            r2 = ring_verts[r_idx + 1]
            for i in range(n):
                i_next = (i + 1) % n
                bm.faces.new((r1[i], r1[i_next], r2[i_next], r2[i]))

        bm.to_mesh(mesh_shell)
        bm.free()

        # Add Subdivision Surface for silky Apple curve
        sub = obj_shell.modifiers.new("Subsurf", "SUBSURF")
        sub.levels = 1
        bpy.context.view_layer.objects.active = obj_shell
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        for p in obj_shell.data.polygons:
            p.use_smooth = True
        obj_shell.data.materials.append(mat_clay)

        # B. Acoustic Ear Cushion (Separate Ergonomic Ring)
        mesh_cush = bpy.data.meshes.new(f"Cushion_{suffix}")
        obj_cush = bpy.data.objects.new(mesh_cush.name, mesh_cush)
        airpods_col.objects.link(obj_cush)

        bm_c = bmesh.new()
        # Cushions have a thicker profile with an acoustic cavity in center
        cush_rings = [
            (0.025, 0.98, 0.0),   # Base at rim
            (0.034, 1.02, 0.0),   # Cushion bulge
            (0.044, 0.98, 0.0),   # Soft inner face
            (0.046, 0.80, 0.0),   # Inner cavity drop
            (0.032, 0.65, 0.0),   # Deep cavity bottom (speaker grille window)
        ]

        c_verts = []
        for y, scale, offset in cush_rings:
            cv = []
            for x, z in squircle_pts:
                v = bm_c.verts.new((x * scale, y, z * scale + offset))
                cv.append(v)
            c_verts.append(cv)

        bm_c.verts.ensure_lookup_table()
        for r_idx in range(len(cush_rings) - 1):
            r1 = c_verts[r_idx]
            r2 = c_verts[r_idx + 1]
            for i in range(n):
                i_next = (i + 1) % n
                bm_c.faces.new((r1[i], r1[i_next], r2[i_next], r2[i]))

        # Bottom floor of inner cavity
        bm_c.faces.new(c_verts[-1])

        bm_c.to_mesh(mesh_cush)
        bm_c.free()

        sub_c = obj_cush.modifiers.new("Subsurf", "SUBSURF")
        sub_c.levels = 1
        bpy.context.view_layer.objects.active = obj_cush
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        for p in obj_cush.data.polygons:
            p.use_smooth = True
        obj_cush.data.materials.append(mat_clay)
        obj_cush.parent = obj_shell

        # C. Swivel Stem Connector (Top Pivot Hinge)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0058, depth=0.012, vertices=32)
        swivel = bpy.context.active_object
        swivel.name = f"Swivel_{suffix}"
        swivel.location = (0.0, 0.010, 0.052)
        swivel.data.materials.append(mat_metal)
        for p in swivel.data.polygons:
            p.use_smooth = True
        swivel.parent = obj_shell
        airpods_col.objects.link(swivel)
        scene.collection.objects.unlink(swivel)

        # D. Controls on Right Earcup (Digital Crown & Button)
        if is_right:
            # Digital Crown: knurled rotary dial at top-back
            bpy.ops.mesh.primitive_cylinder_add(radius=0.0052, depth=0.0055, vertices=32)
            crown = bpy.context.active_object
            crown.name = "DigitalCrown"
            crown.rotation_euler = (math.radians(-25), 0, math.radians(-30))
            crown.location = (0.018, 0.002, 0.048)
            crown.data.materials.append(mat_metal)
            for p in crown.data.polygons:
                p.use_smooth = True
            crown.parent = obj_shell
            airpods_col.objects.link(crown)
            scene.collection.objects.unlink(crown)

            # Noise Control Button (pill toggle)
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            btn = bpy.context.active_object
            btn.name = "NoiseButton"
            btn.scale = (0.0035, 0.012, 0.0035)
            bpy.ops.object.transform_apply(scale=True)
            bev_b = btn.modifiers.new("Bevel", "BEVEL")
            bev_b.width = 0.0014
            bev_b.segments = 4
            bpy.ops.object.modifier_apply(modifier="Bevel")
            btn.rotation_euler = (math.radians(-25), 0, math.radians(-30))
            btn.location = (0.008, 0.002, 0.050)
            btn.data.materials.append(mat_clay)
            btn.parent = obj_shell
            airpods_col.objects.link(btn)
            scene.collection.objects.unlink(btn)

        # E. Bottom Ports (USB-C port slot & Mic Holes)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0028, depth=0.004, vertices=16)
        port = bpy.context.active_object
        port.name = f"Port_{suffix}"
        port.location = (0.0, 0.010, -0.050)
        port.rotation_euler = (math.radians(90), 0, 0)
        port.data.materials.append(mat_metal)
        port.parent = obj_shell
        airpods_col.objects.link(port)
        scene.collection.objects.unlink(port)

        # Positioning & Tilt
        # Authentic ergonomic 14-degree inward tilt
        obj_shell.rotation_euler = (0, math.radians(14.0 * sign), 0)
        obj_shell.location = (sign * 0.076, 0.0, -0.028)

        return obj_shell

    cup_r = build_earcup(is_right=True)
    cup_l = build_earcup(is_right=False)

    # --- 4. TELESCOPING STAINLESS STEEL STEMS ---
    for sign in [-1.0, 1.0]:
        suf = "R" if sign > 0 else "L"
        # Dual-cylinder sleek telescopic arm
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0034, depth=0.048, vertices=28)
        stem = bpy.context.active_object
        stem.name = f"TelescopingStem_{suf}"
        stem.rotation_euler = (0, math.radians(11.0 * sign), 0)
        stem.location = (sign * 0.072, 0.010, 0.040)
        stem.data.materials.append(mat_metal)
        for p in stem.data.polygons:
            p.use_smooth = True
        airpods_col.objects.link(stem)
        scene.collection.objects.unlink(stem)

        # Silicone Sleeve Collar where stem enters headband
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0052, depth=0.014, vertices=28)
        collar = bpy.context.active_object
        collar.name = f"Collar_{suf}"
        collar.rotation_euler = (0, math.radians(10.0 * sign), 0)
        collar.location = (sign * 0.067, 0.010, 0.066)
        collar.data.materials.append(mat_clay)
        for p in collar.data.polygons:
            p.use_smooth = True
        airpods_col.objects.link(collar)
        scene.collection.objects.unlink(collar)

    # --- 5. HEADBAND FRAME (Twin Stainless Steel Arches) ---
    def make_headband_tube(y_offset, name):
        curve_data = bpy.data.curves.new(name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 32
        spline = curve_data.splines.new('BEZIER')
        spline.bezier_points.add(2)

        # Left anchor
        p0 = spline.bezier_points[0]
        p0.co = (-0.066, y_offset, 0.068)
        p0.handle_left = (-0.066, y_offset, 0.045)
        p0.handle_right = (-0.066, y_offset, 0.120)

        # Apex (Top center)
        p1 = spline.bezier_points[1]
        p1.co = (0.0, y_offset, 0.148)
        p1.handle_left = (-0.055, y_offset, 0.148)
        p1.handle_right = (0.055, y_offset, 0.148)

        # Right anchor
        p2 = spline.bezier_points[2]
        p2.co = (0.066, y_offset, 0.068)
        p2.handle_left = (0.066, y_offset, 0.120)
        p2.handle_right = (0.066, y_offset, 0.045)

        curve_data.bevel_depth = 0.0036
        curve_data.bevel_resolution = 6

        tube_obj = bpy.data.objects.new(name, curve_data)
        tube_obj.data.materials.append(mat_clay)
        airpods_col.objects.link(tube_obj)
        return tube_obj

    tube_front = make_headband_tube(-0.008, "HeadbandTube_Front")
    tube_back = make_headband_tube(0.028, "HeadbandTube_Back")

    # --- 6. BREATHABLE KNIT MESH CANOPY ---
    # Suspended tightly between the front and back headband tubes
    mesh_canopy_data = bpy.data.meshes.new("MeshCanopy")
    canopy_obj = bpy.data.objects.new("MeshCanopy", mesh_canopy_data)
    airpods_col.objects.link(canopy_obj)

    bm_k = bmesh.new()
    n_canopy_steps = 48
    width_span = 0.034  # Between -0.007 and +0.027

    canopy_verts = []
    for i in range(n_canopy_steps + 1):
        t = i / n_canopy_steps
        theta = math.pi * (1.0 - t)
        x = 0.064 * math.cos(theta)
        # Parabolic tension curve: canopy hangs slightly below the top tubes (ergonomic cushion)
        z = 0.070 + 0.075 * math.sin(theta)
        # Sag in center for comfortable head fit
        sag = 0.003 * math.sin(theta)

        v_f = bm_k.verts.new((x, -0.006, z - sag))
        v_b = bm_k.verts.new((x, 0.026, z - sag))
        canopy_verts.append((v_f, v_b))

    bm_k.verts.ensure_lookup_table()
    for i in range(n_canopy_steps):
        vf0, vb0 = canopy_verts[i]
        vf1, vb1 = canopy_verts[i + 1]
        bm_k.faces.new((vf0, vf1, vb1, vb0))

    # Give canopy mesh realistic subtle thickness
    bmesh.ops.solidify(bm_k, geom=bm_k.faces[:], thickness=0.0016)
    bm_k.to_mesh(mesh_canopy_data)
    bm_k.free()

    sub_k = canopy_obj.modifiers.new("Subsurf", "SUBSURF")
    sub_k.levels = 1
    bpy.context.view_layer.objects.active = canopy_obj
    bpy.ops.object.modifier_apply(modifier="Subsurf")
    for p in canopy_obj.data.polygons:
        p.use_smooth = True
    canopy_obj.data.materials.append(mat_canopy)

    # --- 7. UNIFY SCENE AND EXPORT HIGH-DETAIL GLB ---
    bpy.ops.object.select_all(action='DESELECT')
    # Convert curve tubes to mesh
    for t_obj in [tube_front, tube_back]:
        t_obj.select_set(True)
        bpy.context.view_layer.objects.active = t_obj
        bpy.ops.object.convert(target='MESH')
        t_obj.select_set(False)

    # Select all objects in collection
    for obj in airpods_col.objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = cup_r
    bpy.ops.object.join()
    final_airpods = bpy.context.active_object
    final_airpods.name = "AppleAirPodsMax_Detailed"

    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    final_airpods.location = (0, 0, 0)

    # Export GLB
    bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')
    print(f"Exported detailed CAD-quality GLB: {output_glb}")

    # --- 8. PROFESSIONAL STUDIO LIGHTING & RENDERS ---
    coords = [final_airpods.matrix_world @ mathutils.Vector(c) for c in final_airpods.bound_box]
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)
    dim_max = max(max_x - min_x, max_y - min_y, max_z - min_z)
    dist = dim_max * 2.15

    cam_data = bpy.data.cameras.new("StudioCam")
    cam_data.lens = 85  # 85mm portrait telephoto (zero perspective distortion, crisp Apple catalog view)
    cam_obj = bpy.data.objects.new("StudioCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Studio Lighting: Softbox Key Light
    key_data = bpy.data.lights.new(name="KeySoftbox", type='AREA')
    key_data.energy = 360.0
    key_data.size = dim_max * 2.2
    key_obj = bpy.data.objects.new("KeySoftbox", key_data)
    key_obj.location = (dist * 0.7, -dist * 0.85, dist * 0.75)
    scene.collection.objects.link(key_obj)

    # Fill Light: Soft bounce
    fill_data = bpy.data.lights.new(name="FillBounce", type='AREA')
    fill_data.energy = 160.0
    fill_data.size = dim_max * 2.8
    fill_obj = bpy.data.objects.new("FillBounce", fill_data)
    fill_obj.location = (-dist * 0.85, -dist * 0.6, dist * 0.25)
    scene.collection.objects.link(fill_obj)

    # Rim / Hair Light: Highlights the mesh canopy arch and earcup silhouette
    rim_data = bpy.data.lights.new(name="RimSilhouette", type='AREA')
    rim_data.energy = 260.0
    rim_data.size = dim_max * 1.5
    rim_obj = bpy.data.objects.new("RimSilhouette", rim_data)
    rim_obj.location = (0.0, dist * 0.9, dist * 0.9)
    scene.collection.objects.link(rim_obj)

    # Bottom Up-Light: Soft subtle bounce underneath earcups
    under_data = bpy.data.lights.new(name="UnderLight", type='AREA')
    under_data.energy = 80.0
    under_data.size = dim_max * 1.5
    under_obj = bpy.data.objects.new("UnderLight", under_data)
    under_obj.location = (0.0, -dist * 0.5, -dist * 0.7)
    scene.collection.objects.link(under_obj)

    # World: Apple dark neutral studio backdrop
    scene.world = bpy.data.worlds.new("StudioBackdrop")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.07, 0.08, 0.09, 1.0)
        bg_node.inputs["Strength"].default_value = 0.85

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    def point_at(target_obj, center_pt=(0, 0, 0)):
        direction = mathutils.Vector(center_pt) - target_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        target_obj.rotation_euler = rot_quat.to_euler()

    point_at(key_obj)
    point_at(fill_obj)
    point_at(rim_obj)
    point_at(under_obj)

    render_views = {
        "detail_front": (0, -dist, 0),
        "detail_perspective": (dist * 0.62, -dist * 0.88, dist * 0.38),
        "detail_side": (dist, 0.010, 0),
        "detail_crown_closeup": (dist * 0.35, -dist * 0.55, dist * 0.35),
    }

    for name, loc in render_views.items():
        cam_obj.location = mathutils.Vector(loc)
        point_at(cam_obj, (0, 0, 0 if "crown" not in name else 0.03))
        scene.render.filepath = str(output_renders_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} -> {scene.render.filepath}")

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glb_path = args[0] if len(args) > 0 else "outputs/airpods_max_precision_detailed.glb"
    out_dir = args[1] if len(args) > 1 else "temp/renders_detailed"
    build_detailed_airpods_max(glb_path, out_dir)
