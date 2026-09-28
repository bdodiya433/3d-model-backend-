import bpy
import bmesh
import math
import mathutils
from pathlib import Path

def sgn(v):
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)

def squircle_point(t, a, b, n=3.8):
    ct = math.cos(t)
    st = math.sin(t)
    x = a * sgn(ct) * (abs(ct) ** (2.0 / n))
    y = b * sgn(st) * (abs(st) ** (2.0 / n))
    return x, y

def create_airpods_masterpiece(output_glb: str, render_dir: str):
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # -------------------------------------------------------------
    # 1. MATERIALS (Studio Precision Satin Aluminum & Chrome)
    # -------------------------------------------------------------
    # Silver Satin Anodized Aluminum (Shell & Accents)
    mat_aluminum = bpy.data.materials.new(name="Apple_Aluminum")
    mat_aluminum.use_nodes = True
    b_alum = mat_aluminum.node_tree.nodes.get("Principled BSDF")
    if b_alum:
        b_alum.inputs["Base Color"].default_value = (0.78, 0.79, 0.81, 1.0) # Realistic reflectance
        b_alum.inputs["Roughness"].default_value = 0.30
        b_alum.inputs["Metallic"].default_value = 0.25

    # Mirror Stainless Steel (Stems, Hinge Pivot, Ferrule, Digital Crown)
    mat_steel = bpy.data.materials.new(name="Apple_Stainless_Steel")
    mat_steel.use_nodes = True
    b_steel = mat_steel.node_tree.nodes.get("Principled BSDF")
    if b_steel:
        b_steel.inputs["Base Color"].default_value = (0.92, 0.93, 0.95, 1.0)
        b_steel.inputs["Roughness"].default_value = 0.08
        b_steel.inputs["Metallic"].default_value = 0.95

    # Acoustic Fabric Cushion (Soft tactile memory foam)
    mat_cushion = bpy.data.materials.new(name="Apple_Acoustic_Cushion")
    mat_cushion.use_nodes = True
    b_cush = mat_cushion.node_tree.nodes.get("Principled BSDF")
    if b_cush:
        b_cush.inputs["Base Color"].default_value = (0.72, 0.73, 0.75, 1.0)
        b_cush.inputs["Roughness"].default_value = 0.80
        b_cush.inputs["Metallic"].default_value = 0.0

    # Inner Speaker Grill Fabric (Dark acoustically transparent mesh)
    mat_grill = bpy.data.materials.new(name="Apple_Speaker_Grill")
    mat_grill.use_nodes = True
    b_grill = mat_grill.node_tree.nodes.get("Principled BSDF")
    if b_grill:
        b_grill.inputs["Base Color"].default_value = (0.22, 0.23, 0.25, 1.0)
        b_grill.inputs["Roughness"].default_value = 0.90
        b_grill.inputs["Metallic"].default_value = 0.0

    # Breathable Mesh Canopy
    mat_mesh = bpy.data.materials.new(name="Apple_Canopy_Mesh")
    mat_mesh.use_nodes = True
    b_mesh = mat_mesh.node_tree.nodes.get("Principled BSDF")
    if b_mesh:
        b_mesh.inputs["Base Color"].default_value = (0.74, 0.75, 0.77, 1.0)
        b_mesh.inputs["Roughness"].default_value = 0.65
        b_mesh.inputs["Metallic"].default_value = 0.0

    # White Silicone Headband Frame
    mat_silicone = bpy.data.materials.new(name="Apple_Silicone_Band")
    mat_silicone.use_nodes = True
    b_sil = mat_silicone.node_tree.nodes.get("Principled BSDF")
    if b_sil:
        b_sil.inputs["Base Color"].default_value = (0.80, 0.81, 0.83, 1.0)
        b_sil.inputs["Roughness"].default_value = 0.35
        b_sil.inputs["Metallic"].default_value = 0.02

    # -------------------------------------------------------------
    # 2. EARCUP MODELING WITH PIVOT ORIGIN
    # -------------------------------------------------------------
    # Local coordinate system:
    # Origin (0,0,0) is at the TOP PIVOT POINT!
    # X: front (+) to back (-)
    # Y: inner cushion (+) to outer aluminum dome (-)
    # Z: top (0) to bottom (-0.102)
    def build_precision_earcup(is_right=False):
        parts = []
        suf = "R" if is_right else "L"
        steps = 64

        # Earcup center is at (0, -0.005, -0.051)
        z_center = -0.051
        y_center = -0.005

        # --- A. ALUMINUM SHELL ---
        shell_mesh = bpy.data.meshes.new(f"ShellMesh_{suf}")
        shell_obj = bpy.data.objects.new(f"Shell_{suf}", shell_mesh)
        scene.collection.objects.link(shell_obj)
        parts.append(shell_obj)

        bm_s = bmesh.new()
        shell_profiles = [
            (-0.0220, 0.024, 0.033, 3.2), # Apex dome center
            (-0.0210, 0.032, 0.041, 3.4), # Dome shoulder
            (-0.0175, 0.037, 0.047, 3.7), # Curve bevel
            (-0.0110, 0.040, 0.050, 3.8), # Main body
            ( 0.0000, 0.040, 0.050, 3.8), # Mid body
            ( 0.0040, 0.039, 0.049, 3.8), # Front rim seam
        ]

        shell_rings = []
        for y_rel, a, b, n in shell_profiles:
            v_ring = []
            y_pos = y_center + y_rel
            for i in range(steps):
                t = (2.0 * math.pi * i) / steps
                x, z_rel = squircle_point(t, a, b, n)
                # Dome curvature on back
                dome_y = y_pos
                if y_rel < -0.015:
                    r_norm = math.sqrt((x / a)**2 + (z_rel / b)**2)
                    dome_y -= 0.0028 * (1.0 - min(1.0, r_norm**2))
                z = z_center + z_rel
                v = bm_s.verts.new((x, dome_y, z))
                v_ring.append(v)
            shell_rings.append(v_ring)

        # Cap back dome
        dome_apex = bm_s.verts.new((0, y_center - 0.0255, z_center))
        for i in range(steps):
            i_next = (i + 1) % steps
            bm_s.faces.new((dome_apex, shell_rings[0][i], shell_rings[0][i_next]))

        # Bridge shell loft rings
        for r in range(len(shell_profiles) - 1):
            r1 = shell_rings[r]
            r2 = shell_rings[r + 1]
            for i in range(steps):
                i_next = (i + 1) % steps
                bm_s.faces.new((r1[i], r1[i_next], r2[i_next], r2[i]))

        bm_s.to_mesh(shell_mesh)
        bm_s.free()

        sub_s = shell_obj.modifiers.new("Subsurf", "SUBSURF")
        sub_s.levels = 1
        bpy.context.view_layer.objects.active = shell_obj
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        for p in shell_obj.data.polygons:
            p.use_smooth = True
        shell_obj.data.materials.append(mat_aluminum)

        # --- B. ACOUSTIC MEMORY FOAM CUSHION WITH EAR CAVITY ---
        cush_mesh = bpy.data.meshes.new(f"CushionMesh_{suf}")
        cush_obj = bpy.data.objects.new(f"Cushion_{suf}", cush_mesh)
        scene.collection.objects.link(cush_obj)
        parts.append(cush_obj)

        bm_c = bmesh.new()
        cush_profiles = [
            (0.0040, 0.0385, 0.0485, 3.8), # Contact rim
            (0.0120, 0.0390, 0.0490, 3.8), # Pillow bulge
            (0.0205, 0.0360, 0.0460, 3.6), # Crest outer
            (0.0220, 0.0310, 0.0410, 3.4), # Front face summit (ear contact)
            (0.0200, 0.0240, 0.0340, 3.2), # Inner rim crest
            (0.0120, 0.0200, 0.0300, 3.0), # Ear cavity wall
            (0.0050, 0.0180, 0.0280, 2.8), # Cavity base
        ]
        cush_rings = []
        for y_rel, a, b, n in cush_profiles:
            v_ring = []
            y_pos = y_center + y_rel
            for i in range(steps):
                t = (2.0 * math.pi * i) / steps
                x, z_rel = squircle_point(t, a, b, n)
                z = z_center + z_rel
                v = bm_c.verts.new((x, y_pos, z))
                v_ring.append(v)
            cush_rings.append(v_ring)

        for r in range(len(cush_profiles) - 1):
            r1 = cush_rings[r]
            r2 = cush_rings[r + 1]
            for i in range(steps):
                i_next = (i + 1) % steps
                bm_c.faces.new((r1[i], r1[i_next], r2[i_next], r2[i]))

        # Inner speaker grill plate
        grill_center = bm_c.verts.new((0, y_center + 0.0050, z_center))
        for i in range(steps):
            i_next = (i + 1) % steps
            bm_c.faces.new((grill_center, cush_rings[-1][i_next], cush_rings[-1][i]))

        bm_c.to_mesh(cush_mesh)
        bm_c.free()

        sub_c = cush_obj.modifiers.new("Subsurf", "SUBSURF")
        sub_c.levels = 1
        bpy.context.view_layer.objects.active = cush_obj
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        for p in cush_obj.data.polygons:
            p.use_smooth = True
        cush_obj.data.materials.append(mat_cushion)
        cush_obj.data.materials.append(mat_grill)
        # Inner speaker cloth assignment
        for p in cush_obj.data.polygons:
            dist_to_center = math.sqrt(p.center.x**2 + (p.center.z - z_center)**2)
            if p.center.y < (y_center + 0.007) and dist_to_center < 0.029:
                p.material_index = 1

        # --- C. TOP HINGE PIVOT (Exactly at Origin 0,0,0) ---
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0054, depth=0.012, vertices=32)
        pivot = bpy.context.active_object
        pivot.name = f"Pivot_{suf}"
        pivot.location = (0.0, 0.0, 0.002)
        pivot.data.materials.append(mat_steel)
        for p in pivot.data.polygons:
            p.use_smooth = True
        parts.append(pivot)

        # Chrome bezel socket ring
        bpy.ops.mesh.primitive_torus_add(major_radius=0.0062, minor_radius=0.0012, major_segments=32, minor_segments=16)
        bezel = bpy.context.active_object
        bezel.location = (0.0, 0.0, -0.002)
        bezel.data.materials.append(mat_steel)
        for p in bezel.data.polygons:
            p.use_smooth = True
        parts.append(bezel)

        # --- D. CONTROLS (Right Earcup Only) ---
        if is_right:
            # 1. Digital Crown (Rotary Dial on top-rear shoulder)
            bpy.ops.mesh.primitive_cylinder_add(radius=0.0052, depth=0.0046, vertices=48)
            crown = bpy.context.active_object
            crown.name = "Apple_Digital_Crown"
            crown.location = (-0.019, -0.004, -0.005)
            crown.rotation_euler = (math.radians(-25), math.radians(22), math.radians(-32))
            crown.data.materials.append(mat_steel)
            for p in crown.data.polygons:
                p.use_smooth = True
            parts.append(crown)

            # 2. Noise Button (Pill button adjacent to crown)
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            btn = bpy.context.active_object
            btn.name = "Apple_Noise_Button"
            btn.scale = (0.0085, 0.0026, 0.0032)
            bpy.ops.object.transform_apply(scale=True)
            bev_btn = btn.modifiers.new("Bevel", "BEVEL")
            bev_btn.width = 0.0012
            bev_btn.segments = 4
            bpy.ops.object.modifier_apply(modifier="Bevel")
            btn.location = (-0.008, -0.006, -0.002)
            btn.rotation_euler = (math.radians(-25), math.radians(16), math.radians(-20))
            btn.data.materials.append(mat_aluminum)
            for p in btn.data.polygons:
                p.use_smooth = True
            parts.append(btn)

        # Bottom Lightning / USB-C Port & Mic holes (Flush/recessed inside shell, NOT protruding)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0022, depth=0.003, vertices=16)
        port = bpy.context.active_object
        port.name = f"Port_{suf}"
        port.location = (0.0, y_center, -0.1015)
        port.data.materials.append(mat_steel)
        parts.append(port)

        for m_sign in [-1.0, 1.0]:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.0009, depth=0.002, vertices=12)
            mic = bpy.context.active_object
            mic.location = (m_sign * 0.014, y_center, -0.1008)
            mic.data.materials.append(mat_steel)
            parts.append(mic)

        # Join into single earcup object with origin at pivot (0,0,0)
        bpy.ops.object.select_all(action='DESELECT')
        for p in parts:
            p.select_set(True)
        bpy.context.view_layer.objects.active = shell_obj
        bpy.ops.object.join()
        earcup = bpy.context.active_object
        earcup.name = f"AirPods_Earcup_{suf}"
        earcup.location = (0, 0, 0)
        return earcup

    # -------------------------------------------------------------
    # 3. POSITION & ORIENT EARCUPS
    # -------------------------------------------------------------
    # World Coordinates:
    # Stem is vertical in front view: Collar at X = +/- 0.062, Pivot at X = +/- 0.062
    # Earcup tilts inward at 23.5 degrees from the pivot down towards neck center.
    earcup_r = build_precision_earcup(is_right=True)
    earcup_l = build_precision_earcup(is_right=False)

    earcup_r.rotation_euler = (math.radians(-6), math.radians(-23.5), math.radians(90))
    earcup_r.location = (0.062, 0.0, 0.046)

    earcup_l.rotation_euler = (math.radians(6), math.radians(23.5), math.radians(-90))
    earcup_l.location = (-0.062, 0.0, 0.046)

    # -------------------------------------------------------------
    # 4. TELESCOPIC STEMS & COLLARS (Collinear & Vertical)
    # -------------------------------------------------------------
    for sign in [-1.0, 1.0]:
        suf = "R" if sign > 0 else "L"
        p_pt = mathutils.Vector((sign * 0.062, 0.0, 0.046))
        c_pt = mathutils.Vector((sign * 0.062, 0.0, 0.086))
        stem_vec = c_pt - p_pt
        stem_len = stem_vec.length
        stem_dir = stem_vec.normalized()
        mid_pt = (p_pt + c_pt) * 0.5
        rot_stem = stem_dir.to_track_quat('Z', 'Y').to_euler()

        # Stainless Steel Telescoping Arm
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0034, depth=stem_len + 0.006, vertices=32)
        stem = bpy.context.active_object
        stem.name = f"Stem_{suf}"
        stem.location = mid_pt
        stem.rotation_euler = rot_stem
        stem.data.materials.append(mat_steel)
        for p in stem.data.polygons:
            p.use_smooth = True

        # Chrome Ferrule Ring
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0048, depth=0.0045, vertices=32)
        ferrule = bpy.context.active_object
        ferrule.name = f"Ferrule_{suf}"
        ferrule.location = c_pt - stem_dir * 0.002
        ferrule.rotation_euler = rot_stem
        ferrule.data.materials.append(mat_steel)
        for p in ferrule.data.polygons:
            p.use_smooth = True

        # Headband Silicone Collar
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0058, depth=0.016, vertices=32)
        collar = bpy.context.active_object
        collar.name = f"Collar_{suf}"
        collar.location = c_pt + stem_dir * 0.008
        collar.rotation_euler = rot_stem
        bev_c = collar.modifiers.new("Bevel", "BEVEL")
        bev_c.width = 0.0014
        bev_c.segments = 4
        bpy.ops.object.modifier_apply(modifier="Bevel")
        collar.data.materials.append(mat_silicone)
        for p in collar.data.polygons:
            p.use_smooth = True

    # -------------------------------------------------------------
    # 5. HEADBAND DUAL-ARCH FRAME & BREATHABLE CANOPY
    # -------------------------------------------------------------
    def build_headband_tube(y_off, name):
        c_data = bpy.data.curves.new(name, type='CURVE')
        c_data.dimensions = '3D'
        c_data.resolution_u = 64
        spline = c_data.splines.new('BEZIER')
        spline.bezier_points.add(2)

        # Collars meet arch at X = +/- 0.062, Z = 0.094
        p0 = spline.bezier_points[0]
        p0.co = (-0.062, y_off, 0.094)
        p0.handle_left = (-0.062, y_off, 0.076)
        p0.handle_right = (-0.062, y_off, 0.138)

        # Smooth wide elliptical apex at X = 0, Z = 0.158
        p1 = spline.bezier_points[1]
        p1.co = (0.0, y_off, 0.158)
        p1.handle_left = (-0.044, y_off, 0.158)
        p1.handle_right = (0.044, y_off, 0.158)

        p2 = spline.bezier_points[2]
        p2.co = (0.062, y_off, 0.094)
        p2.handle_left = (0.062, y_off, 0.138)
        p2.handle_right = (0.062, y_off, 0.076)

        c_data.bevel_depth = 0.0034
        c_data.bevel_resolution = 8
        c_obj = bpy.data.objects.new(name, c_data)
        c_obj.data.materials.append(mat_silicone)
        scene.collection.objects.link(c_obj)
        return c_obj

    arch_f = build_headband_tube(-0.011, "ArchFront")
    arch_b = build_headband_tube( 0.011, "ArchBack")

    # Knit Canopy Ribbon
    canopy_mesh = bpy.data.meshes.new("CanopyMesh")
    canopy_obj = bpy.data.objects.new("Apple_Canopy_Mesh", canopy_mesh)
    scene.collection.objects.link(canopy_obj)

    bm_k = bmesh.new()
    steps_k = 60
    canopy_verts = []
    for i in range(steps_k + 1):
        t = i / steps_k
        th = math.pi * (1.0 - t)
        x = 0.061 * math.cos(th)
        z = 0.094 + 0.064 * math.sin(th)
        sag = 0.0032 * math.sin(th)
        vf = bm_k.verts.new((x, -0.0095, z - sag))
        vb = bm_k.verts.new((x,  0.0095, z - sag))
        canopy_verts.append((vf, vb))

    bm_k.verts.ensure_lookup_table()
    for i in range(steps_k):
        vf0, vb0 = canopy_verts[i]
        vf1, vb1 = canopy_verts[i + 1]
        bm_k.faces.new((vf0, vf1, vb1, vb0))

    bmesh.ops.solidify(bm_k, geom=bm_k.faces[:], thickness=0.0015)
    bm_k.to_mesh(canopy_mesh)
    bm_k.free()

    sub_k = canopy_obj.modifiers.new("Subsurf", "SUBSURF")
    sub_k.levels = 1
    bpy.context.view_layer.objects.active = canopy_obj
    bpy.ops.object.modifier_apply(modifier="Subsurf")
    for p in canopy_obj.data.polygons:
        p.use_smooth = True
    canopy_obj.data.materials.append(mat_mesh)

    # -------------------------------------------------------------
    # 6. EXPORT MASTERPIECE GLB
    # -------------------------------------------------------------
    bpy.ops.object.select_all(action='DESELECT')
    for a in [arch_f, arch_b]:
        a.select_set(True)
        bpy.context.view_layer.objects.active = a
        bpy.ops.object.convert(target='MESH')
        a.select_set(False)

    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)

    bpy.context.view_layer.objects.active = earcup_r
    bpy.ops.object.join()
    master_model = bpy.context.active_object
    master_model.name = "Apple_AirPods_Max_Masterpiece"

    # Export GLB
    bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')
    print(f"[SUCCESS] Exported Masterpiece GLB: {output_glb}")

    # -------------------------------------------------------------
    # 7. HIGH-FIDELITY STUDIO RENDERING (Calibrated Pro Shading)
    # -------------------------------------------------------------
    coords = [master_model.matrix_world @ mathutils.Vector(c) for c in master_model.bound_box]
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)

    center = mathutils.Vector(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5))
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    dim_z = max_z - min_z
    max_dim = max(dim_x, dim_y, dim_z)

    cam_data = bpy.data.cameras.new("AppleStudioCam")
    cam_data.lens = 70 # 70mm focal length
    cam_obj = bpy.data.objects.new("AppleStudioCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Distance that captures entire headphone with generous aesthetic margins
    dist = max_dim * 2.5

    # Calibrated Softbox Studio Lighting (Watts adjusted for ~0.5m scale)
    # Key Light (Front-Right softbox)
    key_d = bpy.data.lights.new("KeyLight", 'AREA')
    key_d.energy = 24.0
    key_d.size = 0.50
    key_obj = bpy.data.objects.new("KeyLight", key_d)
    key_obj.location = (center.x + dist * 0.65, center.y - dist * 0.75, center.z + dist * 0.55)
    scene.collection.objects.link(key_obj)

    # Fill Light (Front-Left soft cool fill)
    fill_d = bpy.data.lights.new("FillLight", 'AREA')
    fill_d.energy = 12.0
    fill_d.size = 0.60
    fill_obj = bpy.data.objects.new("FillLight", fill_d)
    fill_obj.location = (center.x - dist * 0.75, center.y - dist * 0.60, center.z + dist * 0.30)
    scene.collection.objects.link(fill_obj)

    # Rim Light (Top-Back accent light for aluminum highlights)
    rim_d = bpy.data.lights.new("RimLight", 'AREA')
    rim_d.energy = 18.0
    rim_d.size = 0.35
    rim_obj = bpy.data.objects.new("RimLight", rim_d)
    rim_obj.location = (center.x, center.y + dist * 0.85, center.z + dist * 0.75)
    scene.collection.objects.link(rim_obj)

    # Bottom Fill (Subtle kick light from below)
    bot_d = bpy.data.lights.new("BottomLight", 'AREA')
    bot_d.energy = 8.0
    bot_d.size = 0.40
    bot_obj = bpy.data.objects.new("BottomLight", bot_d)
    bot_obj.location = (center.x, center.y - dist * 0.50, center.z - dist * 0.60)
    scene.collection.objects.link(bot_obj)

    def point_to_center(obj):
        d = center - obj.location
        obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

    point_to_center(key_obj)
    point_to_center(fill_obj)
    point_to_center(rim_obj)
    point_to_center(bot_obj)

    # Studio World Background: Pure deep Apple studio black (#000000)
    scene.world = bpy.data.worlds.new("AppleWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.005, 0.005, 0.006, 1.0)
        bg.inputs["Strength"].default_value = 1.0

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.view_settings.view_transform = 'AgX'

    def capture(name, offset_vec):
        cam_obj.location = center + mathutils.Vector(offset_vec)
        point_to_center(cam_obj)
        scene.render.filepath = str(render_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[RENDERED] {name} -> {scene.render.filepath}")

    capture("master_front", (0, -dist, 0))
    capture("master_perspective", (dist * 0.62, -dist * 0.78, dist * 0.40))
    capture("master_side", (dist, 0, 0))
    capture("master_top", (0, 0, dist))

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glb_out = args[0] if len(args) > 0 else "outputs/airpods_max_masterpiece.glb"
    r_dir = args[1] if len(args) > 1 else "temp/renders_masterpiece"
    create_airpods_masterpiece(glb_out, r_dir)
