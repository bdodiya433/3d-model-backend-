import bpy
import bmesh
import math
import mathutils
import numpy as np
from pathlib import Path

def run_authoritative_texturing(
    glb_input: str,
    views_dir: str,
    output_glb: str,
    renders_dir: str,
    tex_res: int = 2048
):
    views_dir = Path(views_dir).resolve()
    renders_dir = Path(renders_dir).resolve()
    renders_dir.mkdir(parents=True, exist_ok=True)
    output_glb = Path(output_glb).resolve()
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("STARTING AUTHORITATIVE SINGLE-SOURCE TEXTURE PROJECTION")
    print("=======================================================")

    # 1. Reset Scene & Import GLB
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.import_scene.gltf(filepath=glb_input)

    mesh_objs = [o for o in scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No mesh object found in GLB")

    obj = mesh_objs[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.name = "AuthoritativeReconstruction"

    # Decimate if excessively dense (keeps geometry crisp & lightweight)
    n_polys = len(obj.data.polygons)
    print(f"Loaded mesh with {n_polys} polygons.")
    if n_polys > 150000:
        dec = obj.modifiers.new("Decimate", "DECIMATE")
        dec.ratio = 0.04
        bpy.ops.object.modifier_apply(modifier="Decimate")
        print(f"Decimated to {len(obj.data.polygons)} polygons.")

    # Smooth shading
    for p in obj.data.polygons:
        p.use_smooth = True

    # Center object at origin
    coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in coords); max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords); max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords); max_z = max(c.z for c in coords)
    center = mathutils.Vector(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0))
    obj.location -= center
    bpy.ops.object.transform_apply(location=True)

    # Recompute bounds
    coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in coords); max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords); max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords); max_z = max(c.z for c in coords)
    w_x = max_x - min_x
    w_y = max_y - min_y
    w_z = max_z - min_z
    dim_max = max(w_x, w_y, w_z)
    print(f"Object bounds: X={w_x:.3f}, Y={w_y:.3f}, Z={w_z:.3f}, max={dim_max:.3f}")

    # 2. Smart UV Unwrapping (non-overlapping UV islands)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.003)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 3. Load the 6 clean reference images using Blender's native image loader
    view_names = ["front", "back", "left", "right", "top", "bottom"]
    photos = {}
    bboxes = {}
    for vn in view_names:
        p_path = views_dir / f"{vn}.png"
        if p_path.exists():
            b_img = bpy.data.images.load(str(p_path))
            h_im, w_im = b_img.size[1], b_img.size[0]
            arr = np.array(b_img.pixels[:], dtype=np.float32).reshape((h_im, w_im, 4))
            photos[vn] = arr
            # Find tight foreground bbox in Blender array coordinates (row 0 is bottom)
            op_rows = np.where(np.any(arr[..., 3] > 0.1, axis=1))[0]
            op_cols = np.where(np.any(arr[..., 3] > 0.1, axis=0))[0]
            if len(op_rows) > 0 and len(op_cols) > 0:
                bboxes[vn] = (int(op_cols.min()), int(op_rows.min()), int(op_cols.max()), int(op_rows.max()))
            else:
                bboxes[vn] = (0, 0, w_im - 1, h_im - 1)
            print(f"Loaded view '{vn}': {arr.shape}, bbox={bboxes[vn]}")
        else:
            print(f"WARNING: View '{vn}' not found at {p_path}")

    # 4. Strict Winner-Takes-All Per-Vertex Color Projection
    # Vertices in world space:
    # Front is -Y, Back is +Y, Left is -X, Right is +X, Top is +Z, Bottom is -Z
    mesh = obj.data
    n_verts = len(mesh.vertices)
    v_pos = np.zeros((n_verts, 3), dtype=np.float32)
    v_norm = np.zeros((n_verts, 3), dtype=np.float32)
    for i, v in enumerate(mesh.vertices):
        v_pos[i] = v.co
        v_norm[i] = v.normal

    # Normalise object coordinates to [0, 1] per axis
    u_x = np.clip((v_pos[:, 0] - min_x) / max(w_x, 1e-6), 0.0, 1.0)
    u_y = np.clip((v_pos[:, 1] - min_y) / max(w_y, 1e-6), 0.0, 1.0)
    u_z = np.clip((v_pos[:, 2] - min_z) / max(w_z, 1e-6), 0.0, 1.0)

    nx = v_norm[:, 0]
    ny = v_norm[:, 1]
    nz = v_norm[:, 2]

    # Scores matrix: (n_verts, 6)
    # Order: 0: front, 1: back, 2: left, 3: right, 4: top, 5: bottom
    scores = np.zeros((n_verts, 6), dtype=np.float32)
    colors = np.zeros((n_verts, 6, 4), dtype=np.float32)

    def sample_cam(vn: str, u_coord: np.ndarray, v_coord: np.ndarray) -> np.ndarray:
        img = photos[vn]
        xmin, ymin, xmax, ymax = bboxes[vn]
        h, w = img.shape[:2]
        # Map [0, 1] to object bbox in image
        px_x = np.clip(np.round(xmin + u_coord * (xmax - xmin)), 0, w - 1).astype(int)
        px_y = np.clip(np.round(ymin + v_coord * (ymax - ymin)), 0, h - 1).astype(int)
        return img[px_y, px_x]

    # 1. Top (+Z)
    if "top" in photos:
        is_top = nz > 0.40
        col = sample_cam("top", u_x, 1.0 - u_y)
        is_valid = is_top & (col[..., 3] > 0.15)
        scores[is_valid, 4] = (nz[is_valid] ** 1.5) * 1.2
        colors[:, 4] = col

    # 2. Bottom (-Z)
    if "bottom" in photos:
        is_bot = -nz > 0.40
        col = sample_cam("bottom", u_x, u_y)
        is_valid = is_bot & (col[..., 3] > 0.15)
        scores[is_valid, 5] = ((-nz[is_valid]) ** 1.5) * 1.2
        colors[:, 5] = col

    is_pole = (scores[:, 4] > 0.55) | (scores[:, 5] > 0.55)

    # 3. Front (-Y): Camera at (0, -dist, 0) looking at Front (+Y direction)
    if "front" in photos:
        is_front = (~is_pole) & (-ny > 0.10)
        col = sample_cam("front", u_x, u_z)
        is_valid = is_front & (col[..., 3] > 0.15)
        scores[is_valid, 0] = (-ny[is_valid]) ** 2.0
        colors[:, 0] = col

    # 4. Back (+Y): Camera at (0, +dist, 0) looking at Back (-Y direction)
    if "back" in photos:
        is_back = (~is_pole) & (ny > 0.10)
        col = sample_cam("back", 1.0 - u_x, u_z)
        is_valid = is_back & (col[..., 3] > 0.15)
        scores[is_valid, 1] = (ny[is_valid]) ** 2.0
        colors[:, 1] = col

    # 5. Left (-X): Camera at (-dist, 0, 0) looking at Left (+X direction)
    if "left" in photos:
        is_left = (~is_pole) & (-nx > 0.10)
        col = sample_cam("left", 1.0 - u_y, u_z)
        is_valid = is_left & (col[..., 3] > 0.15)
        scores[is_valid, 2] = (-nx[is_valid]) ** 2.0
        colors[:, 2] = col

    # 6. Right (+X): Camera at (+dist, 0, 0) looking at Right (-X direction)
    if "right" in photos:
        is_right = (~is_pole) & (nx > 0.10)
        col = sample_cam("right", u_y, u_z)
        is_valid = is_right & (col[..., 3] > 0.15)
        scores[is_valid, 3] = (nx[is_valid]) ** 2.0
        colors[:, 3] = col

    # Winner-Takes-All Authoritative Assignment:
    best_cam = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)

    final_vcols = np.zeros((n_verts, 4), dtype=np.float32)
    for i in range(6):
        active = (best_cam == i) & (max_scores > 1e-4)
        final_vcols[active] = colors[active, i]

    # Clean Apple satin white/silver for non-photographic margins (e.g. inside cavities)
    unassigned = (max_scores <= 1e-4) | (final_vcols[:, 3] < 0.15)
    final_vcols[unassigned] = [0.93, 0.93, 0.95, 1.0]
    final_vcols[:, 3] = 1.0

    # Apply vertex colors to mesh attribute
    col_attr = mesh.color_attributes.new(name="AuthoritativeCol", type='FLOAT_COLOR', domain='POINT')
    flat_colors = final_vcols.ravel()
    col_attr.data.foreach_set("color", flat_colors)

    # 5. Cycles Bake Vertex Colors to UV Texture
    mat = bpy.data.materials.new(name="AuthoritativeMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    emit_node = nodes.new("ShaderNodeEmission")
    attr_node = None
    for ntype in ['ShaderNodeAttribute', 'ShaderNodeVertexColor', 'ShaderNodeColorAttribute']:
        try:
            attr_node = nodes.new(ntype)
            if hasattr(attr_node, 'attribute_name'):
                attr_node.attribute_name = "AuthoritativeCol"
            elif hasattr(attr_node, 'layer_name'):
                attr_node.layer_name = "AuthoritativeCol"
            break
        except Exception:
            continue
    if attr_node is None:
        raise RuntimeError("Could not create attribute node in Blender")

    links.new(attr_node.outputs["Color"], emit_node.inputs["Color"])
    links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

    bake_img = bpy.data.images.new("Authoritative_BaseColor", width=tex_res, height=tex_res)
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bake_img
    nodes.active = tex_node
    tex_node.select = True

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Remove non-mesh objects and ensure obj is active
    for o in list(bpy.data.objects):
        if o.type != 'MESH':
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 2
    scene.cycles.samples = 1
    scene.cycles.bake_type = 'EMIT'
    print(f"Baking {tex_res}x{tex_res} BaseColor texture...")
    bpy.ops.object.bake(type='EMIT')

    bake_tex_path = output_glb.parent / "texture_basecolor.png"
    bake_img.filepath_raw = str(bake_tex_path)
    bake_img.file_format = 'PNG'
    bake_img.save()
    print(f"Saved Authoritative BaseColor -> {bake_tex_path}")

    # 6. Setup PBR Material with Baked Texture
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out_m = nodes.new("ShaderNodeOutputMaterial")
    tex_m = nodes.new("ShaderNodeTexImage")
    tex_m.image = bake_img
    links.new(tex_m.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out_m.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = 0.32
    bsdf.inputs["Metallic"].default_value = 0.40

    # 7. Export Final GLB
    bpy.ops.export_scene.gltf(filepath=str(output_glb), export_format='GLB')
    print(f"SUCCESS: Exported Authoritative GLB -> {output_glb} ({output_glb.stat().st_size:,} bytes)")

    # 8. Render All 6 Validation Views + Perspective
    cam_data = bpy.data.cameras.new("ValCam")
    cam_data.lens = 70
    cam_obj = bpy.data.objects.new("ValCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Studio Lights
    def add_light(name, energy, size, pos):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = energy
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = pos
        scene.collection.objects.link(lo)
        d = mathutils.Vector((0, 0, 0)) - lo.location
        lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

    dist = dim_max * 2.2
    add_light("Key", 220.0, dim_max * 2.0, (dist * 0.7, -dist * 0.8, dist * 0.8))
    add_light("Fill", 110.0, dim_max * 2.5, (-dist * 0.8, -dist * 0.6, dist * 0.4))
    add_light("Rim", 140.0, dim_max * 1.5, (0, dist * 0.9, dist * 0.9))

    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.10, 0.10, 0.12, 1.0)
        bg.inputs["Strength"].default_value = 1.0

    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32

    views = {
        "front": (0, -dist, 0),
        "back": (0, dist, 0),
        "left": (-dist, 0, 0),
        "right": (dist, 0, 0),
        "top": (0, -0.001, dist),
        "bottom": (0, 0.001, -dist),
        "perspective": (dist * 0.65, -dist * 0.85, dist * 0.45)
    }

    rendered_files = {}
    for vn, pos in views.items():
        cam_obj.location = mathutils.Vector(pos)
        d = mathutils.Vector((0, 0, 0)) - cam_obj.location
        cam_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        r_path = renders_dir / f"{vn}.png"
        scene.render.filepath = str(r_path)
        bpy.ops.render.render(write_still=True)
        rendered_files[vn] = str(r_path)
        print(f"Rendered {vn} -> {r_path}")

    print("ALL VALIDATION RENDERS COMPLETE.")
    return rendered_files

if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glb_in = args[0] if len(args) > 0 else "scratch_headphone_white.glb"
    v_dir = args[1] if len(args) > 1 else "temp/clean_views"
    glb_out = args[2] if len(args) > 2 else "temp/authoritative_model/model.glb"
    r_dir = args[3] if len(args) > 3 else "temp/authoritative_model/renders"

    run_authoritative_texturing(glb_in, v_dir, glb_out, r_dir)
