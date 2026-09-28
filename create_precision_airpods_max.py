import bpy
import bmesh
import math
import mathutils
from pathlib import Path

def create_precision_airpods_max(output_glb_path: str, output_renders_dir: str):
    output_renders_dir = Path(output_renders_dir)
    output_renders_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # 2. Materials
    # White Matte / Clay Material
    mat_white = bpy.data.materials.new(name="AirPodsWhiteClay")
    mat_white.use_nodes = True
    bsdf = mat_white.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.94, 0.94, 0.95, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.28
        bsdf.inputs["Metallic"].default_value = 0.0

    # Polished Steel Material (for stems)
    mat_steel = bpy.data.materials.new(name="PolishedSteel")
    mat_steel.use_nodes = True
    bsdf_steel = mat_steel.node_tree.nodes.get("Principled BSDF")
    if bsdf_steel:
        bsdf_steel.inputs["Base Color"].default_value = (0.95, 0.95, 0.96, 1.0)
        bsdf_steel.inputs["Roughness"].default_value = 0.15
        bsdf_steel.inputs["Metallic"].default_value = 0.85

    # Mesh Canopy Material (knit fabric look)
    mat_mesh = bpy.data.materials.new(name="CanopyMesh")
    mat_mesh.use_nodes = True
    bsdf_mesh = mat_mesh.node_tree.nodes.get("Principled BSDF")
    if bsdf_mesh:
        bsdf_mesh.inputs["Base Color"].default_value = (0.90, 0.90, 0.92, 1.0)
        bsdf_mesh.inputs["Roughness"].default_value = 0.65
        bsdf_mesh.inputs["Metallic"].default_value = 0.0

    # 3. Precision Geometry Construction
    # Coordinate system: Z = Up, Y = Depth (Front to Back), X = Width (Left to Right)
    # Origin (0,0,0) is center between earcups

    # --- EAR CUPS ---
    def make_earcup(is_right=False):
        sign = 1.0 if is_right else -1.0
        
        # Outer aluminum shell: rounded pill shape
        # Base dimensions (in meters): height=0.095, width=0.075, depth=0.035
        bm = bmesh.new()
        
        # Create rounded oblong cylinder
        n_pts = 32
        h_half = 0.026
        r = 0.034
        depth_half = 0.016
        
        # Earcup profile
        verts_outer = []
        for i in range(n_pts):
            angle = 2 * math.pi * i / n_pts
            # Pill shape: two half circles connected by vertical segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            x = r * cos_a
            z = r * sin_a + (h_half if sin_a > 0 else -h_half)
            verts_outer.append((x, z))
            
        mesh_cup = bpy.data.meshes.new(f"Earcup_{'R' if is_right else 'L'}")
        obj_cup = bpy.data.objects.new(mesh_cup.name, mesh_cup)
        scene.collection.objects.link(obj_cup)
        
        # Build using Blender cube + subdivision & bevel for perfect CAD curvature
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        temp_cube = bpy.context.active_object
        temp_cube.name = f"TempCup_{'R' if is_right else 'L'}"
        temp_cube.scale = (0.072, 0.028, 0.094)
        bpy.ops.object.transform_apply(scale=True)
        
        # Bevel modifier for classic Apple rounded chamfer
        bev = temp_cube.modifiers.new("Bevel", "BEVEL")
        bev.width = 0.022
        bev.segments = 8
        bev.profile = 0.7
        
        # Subdivision surface for organic smoothness
        sub = temp_cube.modifiers.new("Subsurf", "SUBSURF")
        sub.levels = 2
        sub.render_levels = 2
        bpy.ops.object.modifier_apply(modifier="Bevel")
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        
        # Copy mesh data to our cup
        obj_cup.data = temp_cube.data
        scene.collection.objects.unlink(temp_cube)
        
        # Add smooth shading
        for p in obj_cup.data.polygons:
            p.use_smooth = True
        obj_cup.data.materials.append(mat_white)
        
        # EAR CUSHION (inner acoustic fabric cushion)
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        cushion = bpy.context.active_object
        cushion.name = f"Cushion_{'R' if is_right else 'L'}"
        cushion.scale = (0.070, 0.020, 0.092)
        bpy.ops.object.transform_apply(scale=True)
        
        bev_c = cushion.modifiers.new("Bevel", "BEVEL")
        bev_c.width = 0.018
        bev_c.segments = 8
        sub_c = cushion.modifiers.new("Subsurf", "SUBSURF")
        sub_c.levels = 2
        bpy.ops.object.modifier_apply(modifier="Bevel")
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        
        for p in cushion.data.polygons:
            p.use_smooth = True
        cushion.data.materials.append(mat_white)
        
        # Position cushion slightly inward from aluminum shell
        cushion.location = (-sign * 0.012, 0, 0)
        cushion.parent = obj_cup
        
        # Digital Crown and Button (on right earcup only)
        if is_right:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.0055, depth=0.006, vertices=24)
            crown = bpy.context.active_object
            crown.name = "DigitalCrown"
            crown.rotation_euler = (0, math.radians(-35), math.radians(20))
            crown.location = (0.015, 0.015, 0.046)
            crown.data.materials.append(mat_steel)
            crown.parent = obj_cup
            
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            btn = bpy.context.active_object
            btn.name = "NoiseControlBtn"
            btn.scale = (0.003, 0.012, 0.003)
            bpy.ops.object.transform_apply(scale=True)
            btn.rotation_euler = (0, math.radians(-35), math.radians(20))
            btn.location = (0.012, -0.015, 0.044)
            btn.data.materials.append(mat_white)
            btn.parent = obj_cup

        # Swivel Hinge mount atop earcup
        bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=0.010, vertices=24)
        hinge = bpy.context.active_object
        hinge.name = f"Hinge_{'R' if is_right else 'L'}"
        hinge.location = (0, 0, 0.052)
        hinge.data.materials.append(mat_steel)
        hinge.parent = obj_cup
        
        # Position & tilt earcup
        # Tilt angle ~14 degrees inward
        tilt = math.radians(14.0 * sign)
        obj_cup.rotation_euler = (0, tilt, 0)
        obj_cup.location = (sign * 0.075, 0, -0.02)
        return obj_cup

    cup_r = make_earcup(is_right=True)
    cup_l = make_earcup(is_right=False)

    # --- TELESCOPIC STEMS ---
    for sign in [-1.0, 1.0]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0032, depth=0.042, vertices=24)
        stem = bpy.context.active_object
        stem.name = f"Stem_{'R' if sign > 0 else 'L'}"
        # Match angle of earcup
        stem.rotation_euler = (0, math.radians(10.0 * sign), 0)
        stem.location = (sign * 0.071, 0, 0.045)
        for p in stem.data.polygons:
            p.use_smooth = True
        stem.data.materials.append(mat_steel)

    # --- HEADBAND FRAME & CANOPY ---
    # We construct a curved Bezier arch for the headband
    curve_data = bpy.data.curves.new("HeadbandCurve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 32
    
    spline = curve_data.splines.new('BEZIER')
    # 3 points: Left, Apex (Top), Right
    spline.bezier_points.add(2)
    
    # Left anchor
    p0 = spline.bezier_points[0]
    p0.co = (-0.068, 0, 0.065)
    p0.handle_left = (-0.068, 0, 0.045)
    p0.handle_right = (-0.068, 0, 0.115)
    
    # Apex (Top center)
    p1 = spline.bezier_points[1]
    p1.co = (0.0, 0, 0.142)
    p1.handle_left = (-0.052, 0, 0.142)
    p1.handle_right = (0.052, 0, 0.142)
    
    # Right anchor
    p2 = spline.bezier_points[2]
    p2.co = (0.068, 0, 0.065)
    p2.handle_left = (0.068, 0, 0.115)
    p2.handle_right = (0.068, 0, 0.045)

    # Bevel depth for outer silicone frame
    curve_data.bevel_depth = 0.0055
    curve_data.bevel_resolution = 6
    
    curve_obj = bpy.data.objects.new("HeadbandFrame", curve_data)
    scene.collection.objects.link(curve_obj)
    curve_obj.data.materials.append(mat_white)

    # MESH CANOPY (The breathable knit mesh suspended inside the headband)
    # Generated as a lofted ribbon
    canopy_mesh = bpy.data.meshes.new("MeshCanopy")
    canopy_obj = bpy.data.objects.new("MeshCanopy", canopy_mesh)
    scene.collection.objects.link(canopy_obj)
    
    bm = bmesh.new()
    n_seg = 40
    width_canopy = 0.032
    
    # Parametric arch points
    arch_pts = []
    for i in range(n_seg + 1):
        t = i / n_seg
        theta = math.pi * (1.0 - t)  # from pi down to 0
        x = 0.065 * math.cos(theta)
        # Parabolic height curve
        z = 0.068 + 0.071 * math.sin(theta)
        # Slight subtle taper towards ends
        w_curr = width_canopy * (0.65 + 0.35 * math.sin(theta))
        arch_pts.append((x, z, w_curr))
        
    for x, z, w_c in arch_pts:
        bm.verts.new((x, -w_c / 2.0, z - 0.004))
        bm.verts.new((x, w_c / 2.0, z - 0.004))
        
    bm.verts.ensure_lookup_table()
    for i in range(n_seg):
        v0 = bm.verts[2 * i]
        v1 = bm.verts[2 * i + 1]
        v2 = bm.verts[2 * (i + 1) + 1]
        v3 = bm.verts[2 * (i + 1)]
        bm.faces.new((v0, v1, v2, v3))
        
    # Extrude slightly for thickness (solidify)
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.002)
    bm.to_mesh(canopy_mesh)
    bm.free()
    
    sub_canopy = canopy_obj.modifiers.new("Subsurf", "SUBSURF")
    sub_canopy.levels = 1
    bpy.context.view_layer.objects.active = canopy_obj
    bpy.ops.object.modifier_apply(modifier="Subsurf")
    for p in canopy_obj.data.polygons:
        p.use_smooth = True
    canopy_obj.data.materials.append(mat_mesh)

    # 4. Join all components into unified clean model
    bpy.ops.object.select_all(action='DESELECT')
    # Convert curve to mesh first
    curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.convert(target='MESH')
    
    # Select all mesh objects
    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.join()
    final_model = bpy.context.active_object
    final_model.name = "AppleAirPodsMax_White"
    
    # Scale to standard normalized bounding box (height ~ 1.8 units)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
    final_model.location = (0, 0, 0)
    
    # Export GLB
    bpy.ops.export_scene.gltf(filepath=output_glb_path, export_format='GLB')
    print(f"Exported precision GLB to {output_glb_path}")

    # 5. Studio Lighting & Multi-Angle Renders
    all_coords = [final_model.matrix_world @ mathutils.Vector(c) for c in final_model.bound_box]
    min_x = min(c.x for c in all_coords)
    max_x = max(c.x for c in all_coords)
    min_y = min(c.y for c in all_coords)
    max_y = max(c.y for c in all_coords)
    min_z = min(c.z for c in all_coords)
    max_z = max(c.z for c in all_coords)
    center = mathutils.Vector(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0))
    dim_max = max(max_x - min_x, max_y - min_y, max_z - min_z)
    dist = dim_max * 2.1

    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 70
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Studio Lighting
    key_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_data.energy = 450.0
    key_data.size = dim_max * 1.8
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    key_obj.location = (center.x + dist * 0.7, center.y - dist * 0.8, center.z + dist * 0.8)
    scene.collection.objects.link(key_obj)

    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 220.0
    fill_data.size = dim_max * 2.2
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    fill_obj.location = (center.x - dist * 0.8, center.y - dist * 0.5, center.z + dist * 0.3)
    scene.collection.objects.link(fill_obj)

    rim_data = bpy.data.lights.new(name="RimLight", type='AREA')
    rim_data.energy = 280.0
    rim_data.size = dim_max * 1.5
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    rim_obj.location = (center.x, center.y + dist * 0.9, center.z + dist * 0.9)
    scene.collection.objects.link(rim_obj)

    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.10, 0.11, 0.13, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0

    scene.render.resolution_x = 768
    scene.render.resolution_y = 768

    def point_at(obj, target):
        direction = target - obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        obj.rotation_euler = rot_quat.to_euler()

    point_at(key_obj, center)
    point_at(fill_obj, center)
    point_at(rim_obj, center)

    render_angles = {
        "precision_front": (center.x, center.y - dist, center.z),
        "precision_perspective": (center.x + dist * 0.65, center.y - dist * 0.85, center.z + dist * 0.35),
        "precision_side": (center.x + dist, center.y, center.z),
        "precision_top": (center.x, center.y - 0.001, center.z + dist),
    }

    for name, loc in render_angles.items():
        cam_obj.location = mathutils.Vector(loc)
        point_at(cam_obj, center)
        scene.render.filepath = str(output_renders_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name} -> {scene.render.filepath}")

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glb_out = args[0] if len(args) > 0 else "outputs/precision_airpods_max.glb"
    renders_out = args[1] if len(args) > 1 else "temp/renders_precision"
    create_precision_airpods_max(glb_out, renders_out)
