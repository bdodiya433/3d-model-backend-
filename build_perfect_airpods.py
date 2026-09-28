import bpy
import bmesh
import math
import mathutils
from pathlib import Path

def sgn(v):
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)

def squircle_point(t, a, b, n=3.6):
    ct = math.cos(t)
    st = math.sin(t)
    x = a * sgn(ct) * (abs(ct) ** (2.0 / n))
    y = b * sgn(st) * (abs(st) ** (2.0 / n))
    return x, y

def create_squircle_curve(a, b, n=3.6, steps=64):
    pts = []
    for i in range(steps):
        t = (2.0 * math.pi * i) / steps
        x, y = squircle_point(t, a, b, n)
        pts.append((x, y))
    return pts

def create_airpods_max_cad(output_glb: str, render_dir: str):
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # -------------------------------------------------------------
    # 1. MATERIALS (Studio Clay / Clean White & Polished Metal)
    # -------------------------------------------------------------
    # Premium Satin Matte Aluminum (Earcups & Collar)
    mat_shell = bpy.data.materials.new(name="Apple_Anodized_Aluminum")
    mat_shell.use_nodes = True
    bsdf_shell = mat_shell.node_tree.nodes.get("Principled BSDF")
    if bsdf_shell:
        bsdf_shell.inputs["Base Color"].default_value = (0.92, 0.92, 0.94, 1.0)
        bsdf_shell.inputs["Roughness"].default_value = 0.28
        bsdf_shell.inputs["Metallic"].default_value = 0.15

    # Polished Chrome / Stainless Steel (Stems, Hinge Pivot, Crown)
    mat_steel = bpy.data.materials.new(name="Apple_Stainless_Steel")
    mat_steel.use_nodes = True
    bsdf_steel = mat_steel.node_tree.nodes.get("Principled BSDF")
    if bsdf_steel:
        bsdf_steel.inputs["Base Color"].default_value = (0.96, 0.96, 0.97, 1.0)
        bsdf_steel.inputs["Roughness"].default_value = 0.08
        bsdf_steel.inputs["Metallic"].default_value = 0.95

    # Acoustic Fabric Cushion (Slightly softer matte white texture)
    mat_cushion = bpy.data.materials.new(name="Apple_Acoustic_Fabric")
    mat_cushion.use_nodes = True
    bsdf_cush = mat_cushion.node_tree.nodes.get("Principled BSDF")
    if bsdf_cush:
        bsdf_cush.inputs["Base Color"].default_value = (0.88, 0.88, 0.90, 1.0)
        bsdf_cush.inputs["Roughness"].default_value = 0.70
        bsdf_cush.inputs["Metallic"].default_value = 0.0

    # Inner Speaker Mesh (Dark acoustically transparent fabric)
    mat_speaker = bpy.data.materials.new(name="Apple_Speaker_Grill")
    mat_speaker.use_nodes = True
    bsdf_spk = mat_speaker.node_tree.nodes.get("Principled BSDF")
    if bsdf_spk:
        bsdf_spk.inputs["Base Color"].default_value = (0.45, 0.46, 0.48, 1.0)
        bsdf_spk.inputs["Roughness"].default_value = 0.85
        bsdf_spk.inputs["Metallic"].default_value = 0.0

    # Breathable Mesh Canopy (Headband)
    mat_mesh = bpy.data.materials.new(name="Apple_Canopy_Mesh")
    mat_mesh.use_nodes = True
    bsdf_mesh = mat_mesh.node_tree.nodes.get("Principled BSDF")
    if bsdf_mesh:
        bsdf_mesh.inputs["Base Color"].default_value = (0.87, 0.88, 0.90, 1.0)
        bsdf_mesh.inputs["Roughness"].default_value = 0.65
        bsdf_mesh.inputs["Metallic"].default_value = 0.0

    # -------------------------------------------------------------
    # 2. DETAILED EARCUP BUILDER
    # -------------------------------------------------------------
    # Local axes of earcup:
    # X: front (+) to back (-)
    # Z: bottom (-) to top (+)
    # Y: inside (+, cushion) to outside (-, aluminum back)
    def build_earcup(is_right=False):
        parts = []
        suf = "R" if is_right else "L"

        # --- A. ALUMINUM SHELL ---
        # Profile lofting: from outer dome (Y = -0.016) to cushion rim (Y = +0.006)
        shell_mesh = bpy.data.meshes.new(f"ShellMesh_{suf}")
        shell_obj = bpy.data.objects.new(f"Shell_{suf}", shell_mesh)
        scene.collection.objects.link(shell_obj)
        parts.append(shell_obj)

        bm_s = bmesh.new()
        rings = [
            # (Y_pos, scale_a, scale_b, n)
            (-0.018, 0.032, 0.043, 3.4), # back dome center
            (-0.016, 0.036, 0.047, 3.5), # back curve
            (-0.012, 0.038, 0.049, 3.6), # main perimeter
            (-0.002, 0.038, 0.049, 3.6), # mid body
            ( 0.005, 0.0375, 0.0485, 3.6), # rim edge
        ]
        ring_verts = []
        steps = 64
        for y_pos, a, b, n in rings:
            v_ring = []
            for i in range(steps):
                t = (2.0 * math.pi * i) / steps
                x, z = squircle_point(t, a, b, n)
                # gentle spherical curvature on back dome
                dome_y = y_pos
                if y_pos < -0.012:
                    dist_norm = math.sqrt((x/a)**2 + (z/b)**2)
                    dome_y -= 0.003 * (1.0 - min(1.0, dist_norm**2))
                v = bm_s.verts.new((x, dome_y, z))
                v_ring.append(v)
            ring_verts.append(v_ring)

        # Cap the back dome
        center_back = bm_s.verts.new((0, -0.021, 0))
        for i in range(steps):
            i_next = (i + 1) % steps
            bm_s.faces.new((center_back, ring_verts[0][i], ring_verts[0][i_next]))

        # Bridge shell rings
        for r in range(len(rings) - 1):
            r1 = ring_verts[r]
            r2 = ring_verts[r + 1]
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
        shell_obj.data.materials.append(mat_shell)

        # --- B. ACOUSTIC CUSHION WITH EAR CAVITY ---
        cush_mesh = bpy.data.meshes.new(f"CushionMesh_{suf}")
        cush_obj = bpy.data.objects.new(f"Cushion_{suf}", cush_mesh)
        scene.collection.objects.link(cush_obj)
        parts.append(cush_obj)

        bm_c = bmesh.new()
        # Outer squircle at rim: a=0.037, b=0.048, Y=0.005
        # Cushion peak: a=0.035, b=0.046, Y=0.022
        # Cushion inner drop: a=0.019, b=0.029, Y=0.020
        # Cavity floor: a=0.017, b=0.027, Y=0.008
        cush_profiles = [
            (0.005, 0.037, 0.048, 3.6), # contact rim
            (0.016, 0.037, 0.048, 3.6), # outer bulge
            (0.023, 0.034, 0.045, 3.4), # front contact crest
            (0.023, 0.027, 0.037, 3.2), # inner contact crest
            (0.016, 0.020, 0.030, 3.0), # cavity wall
            (0.008, 0.018, 0.028, 2.8), # cavity floor
        ]
        cush_rings = []
        for y_pos, a, b, n in cush_profiles:
            v_ring = []
            for i in range(steps):
                t = (2.0 * math.pi * i) / steps
                x, z = squircle_point(t, a, b, n)
                v = bm_c.verts.new((x, y_pos, z))
                v_ring.append(v)
            cush_rings.append(v_ring)

        # Bridge cushion rings
        for r in range(len(cush_profiles) - 1):
            r1 = cush_rings[r]
            r2 = cush_rings[r + 1]
            for i in range(steps):
                i_next = (i + 1) % steps
                bm_c.faces.new((r1[i], r1[i_next], r2[i_next], r2[i]))

        # Inner speaker grill plate
        center_floor = bm_c.verts.new((0, 0.008, 0))
        for i in range(steps):
            i_next = (i + 1) % steps
            bm_c.faces.new((center_floor, cush_rings[-1][i_next], cush_rings[-1][i]))

        bm_c.to_mesh(cush_mesh)
        bm_c.free()

        sub_c = cush_obj.modifiers.new("Subsurf", "SUBSURF")
        sub_c.levels = 1
        bpy.context.view_layer.objects.active = cush_obj
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        for p in cush_obj.data.polygons:
            p.use_smooth = True
        cush_obj.data.materials.append(mat_cushion)

        # --- C. TOP HINGE PIVOT (where stem docks into earcup) ---
        # Top apex is at X=0, Z=0.049, Y=-0.004
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0055, depth=0.012, vertices=32)
        pivot = bpy.context.active_object
        pivot.name = f"Pivot_{suf}"
        pivot.location = (0.0, -0.004, 0.051)
        pivot.data.materials.append(mat_steel)
        for p in pivot.data.polygons:
            p.use_smooth = True
        parts.append(pivot)

        # Pivot collar bevel ring
        bpy.ops.mesh.primitive_torus_add(major_radius=0.0062, minor_radius=0.0014, major_segments=32, minor_segments=16)
        p_ring = bpy.context.active_object
        p_ring.location = (0.0, -0.004, 0.049)
        p_ring.data.materials.append(mat_steel)
        for p in p_ring.data.polygons:
            p.use_smooth = True
        parts.append(p_ring)

        # --- D. CONTROLS (Right earcup only: Digital Crown & Noise Button) ---
        if is_right:
            # Digital Crown: fluted rotary dial on top-rear shoulder
            # Position: rear side (negative X in our local frame), near top (Z ~ 0.044)
            bpy.ops.mesh.primitive_cylinder_add(radius=0.0050, depth=0.0048, vertices=32)
            crown = bpy.context.active_object
            crown.name = "DigitalCrown"
            # Angle pointing outwards from shoulder
            crown.rotation_euler = (math.radians(-15), math.radians(28), math.radians(-30))
            crown.location = (-0.018, -0.002, 0.046)
            crown.data.materials.append(mat_steel)
            for p in crown.data.polygons:
                p.use_smooth = True
            parts.append(crown)

            # Noise Control Button: elongated pill-shaped button right in front of crown
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            btn = bpy.context.active_object
            btn.name = "NoiseButton"
            btn.scale = (0.008, 0.0025, 0.0032)
            bpy.ops.object.transform_apply(scale=True)
            bev_b = btn.modifiers.new("Bevel", "BEVEL")
            bev_b.width = 0.0012
            bev_b.segments = 4
            bpy.ops.object.modifier_apply(modifier="Bevel")
            btn.rotation_euler = (math.radians(-15), math.radians(20), math.radians(-20))
            btn.location = (-0.008, -0.004, 0.0485)
            btn.data.materials.append(mat_shell)
            for p in btn.data.polygons:
                p.use_smooth = True
            parts.append(btn)

        # Bottom charging port slot & microphone holes
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0022, depth=0.006, vertices=16)
        port = bpy.context.active_object
        port.name = f"Port_{suf}"
        port.location = (0.0, -0.004, -0.049)
        port.data.materials.append(mat_steel)
        parts.append(port)

        # Join all earcup parts into a single object
        bpy.ops.object.select_all(action='DESELECT')
        for p in parts:
            p.select_set(True)
        bpy.context.view_layer.objects.active = shell_obj
        bpy.ops.object.join()
        earcup = bpy.context.active_object
        earcup.name = f"AirPodsMax_Earcup_{suf}"
        return earcup

    # -------------------------------------------------------------
    # 3. POSITION EARCUPS IN WORLD SPACE
    # -------------------------------------------------------------
    # World Coordinates:
    # X: Left (-) to Right (+)
    # Y: Front (-) to Back (+)
    # Z: Bottom (-) to Top (+)
    # In world space:
    # Earcups are tilted inward: bottom is closer together, top is wider.
    # Angle inward around Y axis: ~18 degrees.
    # Cushion face (local +Y) points inward:
    # - Right earcup (+X): cushion points toward -X.
    # - Left earcup (-X): cushion points toward +X.
    # Toe-in angle around Z axis: ~6 degrees (front of cups angled slightly inward).

    earcup_r = build_earcup(is_right=True)
    earcup_l = build_earcup(is_right=False)

    # Orientation:
    # Local frame: X = front-to-back, Y = inner cushion to outer shell, Z = bottom to top
    # For Right earcup:
    # We want local X -> World +Y (or -Y depending on front/back)
    # We want local Y -> World -X (cushion pointing inward toward -X)
    # We want local Z -> World +Z with inward tilt ~18 deg
    # In Euler XYZ:
    # Rotate Z by 90 deg: local X -> Y, local Y -> -X!
    # Inward tilt around Y by -18 deg for Right, +18 deg for Left!
    earcup_r.rotation_euler = (math.radians(-6), math.radians(-19), math.radians(90))
    earcup_r.location = (0.068, 0.0, 0.00)

    earcup_l.rotation_euler = (math.radians(6), math.radians(19), math.radians(-90))
    earcup_l.location = (-0.068, 0.0, 0.00)

    # -------------------------------------------------------------
    # 4. TELESCOPIC STAINLESS STEEL STEMS & SILICONE COLLARS
    # -------------------------------------------------------------
    # The stem connects the top of each earcup pivot directly to the headband collar.
    # Let's inspect the world coordinate of the earcup pivot:
    # Local pivot was at (0, -0.004, 0.051).
    # In world space, right pivot is around X = 0.053, Y = 0.004, Z = 0.048.
    # The headband collar is at X = 0.060, Y = 0.0, Z = 0.082.
    for sign in [-1.0, 1.0]:
        suf = "R" if sign > 0 else "L"
        # Pivot world position:
        p_world_x = sign * 0.053
        p_world_y = 0.002
        p_world_z = 0.048

        c_world_x = sign * 0.062
        c_world_y = 0.000
        c_world_z = 0.084

        # Stem cylinder between pivot and collar
        dx = c_world_x - p_world_x
        dy = c_world_y - p_world_y
        dz = c_world_z - p_world_z
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        mid_x = (p_world_x + c_world_x) * 0.5
        mid_y = (p_world_y + c_world_y) * 0.5
        mid_z = (p_world_z + c_world_z) * 0.5

        bpy.ops.mesh.primitive_cylinder_add(radius=0.0032, depth=length + 0.008, vertices=32)
        stem = bpy.context.active_object
        stem.name = f"Stem_{suf}"
        stem.location = (mid_x, mid_y, mid_z)
        # Point cylinder along vector
        v = mathutils.Vector((dx, dy, dz)).normalized()
        stem.rotation_euler = v.to_track_quat('Z', 'Y').to_euler()
        stem.data.materials.append(mat_steel)
        for p in stem.data.polygons:
            p.use_smooth = True

        # Headband Stainless Steel Ring / Ferrule
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0048, depth=0.004, vertices=32)
        ferrule = bpy.context.active_object
        ferrule.name = f"Ferrule_{suf}"
        ferrule.location = (c_world_x - dx*0.1, c_world_y - dy*0.1, c_world_z - dz*0.1)
        ferrule.rotation_euler = v.to_track_quat('Z', 'Y').to_euler()
        ferrule.data.materials.append(mat_steel)
        for p in ferrule.data.polygons:
            p.use_smooth = True

        # Headband Silicone Collar
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0058, depth=0.016, vertices=32)
        collar = bpy.context.active_object
        collar.name = f"Collar_{suf}"
        collar.location = (c_world_x + dx*0.15, c_world_y + dy*0.15, c_world_z + dz*0.15)
        collar.rotation_euler = v.to_track_quat('Z', 'Y').to_euler()
        bev = collar.modifiers.new("Bevel", "BEVEL")
        bev.width = 0.0014
        bev.segments = 4
        bpy.ops.object.modifier_apply(modifier="Bevel")
        collar.data.materials.append(mat_shell)
        for p in collar.data.polygons:
            p.use_smooth = True

    # -------------------------------------------------------------
    # 5. HEADBAND DUAL-ARCH FRAME & BREATHABLE MESH CANOPY
    # -------------------------------------------------------------
    # Apple AirPods Max headband has two smooth stainless steel tube arches:
    # Front arch at Y = -0.012, Rear arch at Y = +0.012.
    # Rising from the collars (X = +/- 0.065, Z = 0.090) to apex (X = 0, Z = 0.162).
    def build_arch_spline(y_off, name):
        c_data = bpy.data.curves.new(name, type='CURVE')
        c_data.dimensions = '3D'
        c_data.resolution_u = 48
        spline = c_data.splines.new('BEZIER')
        spline.bezier_points.add(2)

        p0 = spline.bezier_points[0]
        p0.co = (-0.065, y_off, 0.090)
        p0.handle_left = (-0.065, y_off, 0.070)
        p0.handle_right = (-0.065, y_off, 0.140)

        p1 = spline.bezier_points[1]
        p1.co = (0.0, y_off, 0.162)
        p1.handle_left = (-0.048, y_off, 0.162)
        p1.handle_right = (0.048, y_off, 0.162)

        p2 = spline.bezier_points[2]
        p2.co = (0.065, y_off, 0.090)
        p2.handle_left = (0.065, y_off, 0.140)
        p2.handle_right = (0.065, y_off, 0.070)

        c_data.bevel_depth = 0.0034
        c_data.bevel_resolution = 8
        c_obj = bpy.data.objects.new(name, c_data)
        c_obj.data.materials.append(mat_shell)
        scene.collection.objects.link(c_obj)
        return c_obj

    arch_front = build_arch_spline(-0.011, "ArchFront")
    arch_back = build_arch_spline(0.011, "ArchBack")

    # Knit Canopy Mesh (suspended hammock ribbon spanning between front & back arches)
    canopy_mesh = bpy.data.meshes.new("CanopyMesh")
    canopy_obj = bpy.data.objects.new("AirPodsMax_Canopy", canopy_mesh)
    scene.collection.objects.link(canopy_obj)

    bm_k = bmesh.new()
    steps = 56
    ribbon_verts = []
    for i in range(steps + 1):
        t = i / steps
        th = math.pi * (1.0 - t)
        # Arch profile
        x = 0.064 * math.cos(th)
        z = 0.090 + 0.070 * math.sin(th)
        # Natural downward hammock sag
        sag = 0.0035 * math.sin(th)
        vf = bm_k.verts.new((x, -0.009, z - sag))
        vb = bm_k.verts.new((x,  0.009, z - sag))
        ribbon_verts.append((vf, vb))

    bm_k.verts.ensure_lookup_table()
    for i in range(steps):
        vf0, vb0 = ribbon_verts[i]
        vf1, vb1 = ribbon_verts[i + 1]
        bm_k.faces.new((vf0, vf1, vb1, vb0))

    bmesh.ops.solidify(bm_k, geom=bm_k.faces[:], thickness=0.0016)
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
    # 6. COMBINE MESHES & EXPORT GLB
    # -------------------------------------------------------------
    bpy.ops.object.select_all(action='DESELECT')
    for a in [arch_front, arch_back]:
        a.select_set(True)
        bpy.context.view_layer.objects.active = a
        bpy.ops.object.convert(target='MESH')
        a.select_set(False)

    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)

    bpy.context.view_layer.objects.active = earcup_r
    bpy.ops.object.join()
    final_model = bpy.context.active_object
    final_model.name = "Apple_AirPods_Max"

    # Center origin
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    final_model.location = (0, 0, 0)

    # Save GLB
    bpy.ops.export_scene.gltf(filepath=output_glb, export_format='GLB')
    print(f"[SUCCESS] Exported Precision GLB: {output_glb}")

    # -------------------------------------------------------------
    # 7. CALIBRATED STUDIO LIGHTING & RENDERING (Apple Style)
    # -------------------------------------------------------------
    # Bound box to compute camera distance perfectly framed
    coords = [final_model.matrix_world @ mathutils.Vector(c) for c in final_model.bound_box]
    dim_x = max(c.x for c in coords) - min(c.x for c in coords)
    dim_y = max(c.y for c in coords) - min(c.y for c in coords)
    dim_z = max(c.z for c in coords) - min(c.z for c in coords)
    max_dim = max(dim_x, dim_y, dim_z)

    # Camera with ~85mm portrait focal length
    cam_data = bpy.data.cameras.new("StudioCamera")
    cam_data.lens = 85
    cam_obj = bpy.data.objects.new("StudioCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Studio Lights
    def make_light(name, ltype, energy, size, pos):
        ld = bpy.data.lights.new(name, ltype)
        ld.energy = energy
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = pos
        scene.collection.objects.link(lo)
        d = -mathutils.Vector(pos).normalized()
        lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return lo

    dist_cam = max_dim * 3.3
    # Key light: upper-right softbox
    make_light("KeyLight", 'AREA', 22.0, 0.5, (dist_cam*0.7, -dist_cam*0.8, dist_cam*0.7))
    # Fill light: soft cool fill from left
    make_light("FillLight", 'AREA', 9.0, 0.6, (-dist_cam*0.8, -dist_cam*0.6, dist_cam*0.3))
    # Rim light: top rim highlight for aluminum contours
    make_light("RimLight", 'AREA', 16.0, 0.4, (0, dist_cam*0.7, dist_cam*0.8))
    # Bottom kick: subtle reflection on bottom earcups
    make_light("BottomKick", 'AREA', 6.0, 0.4, (0, -dist_cam*0.5, -dist_cam*0.5))

    # Dark background exactly matching Apple marketing photos
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.015, 0.015, 0.018, 1.0)
        bg.inputs["Strength"].default_value = 0.5

    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    def render_shot(name, cam_loc, target_pt=(0, 0, 0)):
        cam_obj.location = mathutils.Vector(cam_loc)
        direction = mathutils.Vector(target_pt) - cam_obj.location
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = str(render_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[RENDERED] {name} -> {scene.render.filepath}")

    render_shot("apple_cad_front", (0, -dist_cam, 0))
    render_shot("apple_cad_perspective", (dist_cam * 0.7, -dist_cam * 0.8, dist_cam * 0.4))
    render_shot("apple_cad_side", (dist_cam, 0, 0))
    render_shot("apple_cad_top", (0, 0, dist_cam))
    render_shot("apple_cad_bottom", (0, 0, -dist_cam))

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glb = args[0] if len(args) > 0 else "outputs/airpods_max_precision_cad.glb"
    rdir = args[1] if len(args) > 1 else "temp/renders_perfect"
    create_airpods_max_cad(glb, rdir)
