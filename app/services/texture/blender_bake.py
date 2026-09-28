import bpy
import sys
import os
import json
import time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def load_image_pixels(filepath):
    """Loads a PNG/JPEG image using Blender's native image engine into an (H, W, 4) float32 array in [0, 1]."""
    img = bpy.data.images.load(filepath)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    # In Blender, row 0 is bottom-left. Flip vertically so row 0 is top-left (standard image space)
    px = np.flipud(px)
    bpy.data.images.remove(img)
    return px


def save_texture_image(name, arr_float, filepath):
    """Saves an (H, W, 4) float32 array in [0, 1] as an sRGB PNG file using Blender."""
    h, w = arr_float.shape[:2]
    old_img = bpy.data.images.get(name)
    if old_img is not None:
        bpy.data.images.remove(old_img)

    img = bpy.data.images.new(name, width=w, height=h)
    img.colorspace_settings.name = 'sRGB'
    img.pixels.foreach_set(arr_float.ravel())
    img.filepath_raw = filepath
    img.file_format = 'PNG'
    img.save()
    return img


def dilate_texture(rgba_float, valid_mask, steps=6):
    """Dilates active UV texels into empty background in pure NumPy to eliminate island seam fringes."""
    arr = rgba_float.copy()
    cur_mask = valid_mask.copy()
    for _ in range(steps):
        up = np.roll(cur_mask, -1, axis=0)
        down = np.roll(cur_mask, 1, axis=0)
        left = np.roll(cur_mask, -1, axis=1)
        right = np.roll(cur_mask, 1, axis=1)
        border = (~cur_mask) & (up | down | left | right)

        neighbor_sum = np.zeros_like(arr)
        count = np.zeros(cur_mask.shape, dtype=np.float32)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            sm = np.roll(cur_mask, (dy, dx), axis=(0, 1))
            sc = np.roll(arr, (dy, dx), axis=(0, 1))
            vs = sm & border
            neighbor_sum[vs] += sc[vs]
            count[vs] += 1

        has_n = border & (count > 0)
        arr[has_n] = neighbor_sum[has_n] / count[has_n, None]
        cur_mask[has_n] = True
    return arr


def run_bake():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]

    glb_input = argv[0]
    glb_output = argv[1]
    blend_output = argv[2] if len(argv) > 2 and argv[2] != "None" else None
    tex_res = int(argv[3]) if len(argv) > 3 else 2048
    views_json_path = argv[4] if len(argv) > 4 and argv[4] != "None" else None
    debug_dir = argv[5] if len(argv) > 5 and argv[5] != "None" else None

    print(f"Blender Baker: Loading {glb_input}")
    print(f"Blender Baker: Target Resolution: {tex_res}x{tex_res}, Debug Dir: {debug_dir}")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # 1. Reset scene & Enforce Strictly Neutral Photographic Color Management (Part 8 & 15)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.view_transform = 'Standard'  # NEVER AgX or Filmic!
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.exposure = 0.0
    bpy.context.scene.view_settings.gamma = 1.0

    # 2. Import GLB
    bpy.ops.import_scene.gltf(filepath=glb_input)

    # Remove non-mesh objects
    for o in list(bpy.data.objects):
        if o.type != 'MESH':
            bpy.data.objects.remove(o, do_unlink=True)

    mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not mesh_objs:
        print("ERROR: No mesh found in input GLB.")
        sys.exit(1)

    obj = mesh_objs[0]
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # 3. Smart UV Unwrap with Adequate Island Padding (Part 13)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')

    for poly in obj.data.polygons:
        poly.use_smooth = True

    # 4. Bake High-Resolution World Position & Normal Maps
    print("Blender Baker: Baking Position and Normal maps for texel-level projection...")
    pos_img = bpy.data.images.new("Temp_PosMap", width=tex_res, height=tex_res, float_buffer=True)
    norm_img = bpy.data.images.new("Temp_NormMap", width=tex_res, height=tex_res, float_buffer=True)

    temp_mat = bpy.data.materials.new("TempBakeMat")
    temp_mat.use_nodes = True
    nodes = temp_mat.node_tree.nodes
    nodes.clear()
    tex_node = nodes.new('ShaderNodeTexImage')
    nodes.active = tex_node
    tex_node.select = True
    obj.data.materials.clear()
    obj.data.materials.append(temp_mat)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.render.threads_mode = 'FIXED'
    bpy.context.scene.render.threads = 2
    bpy.context.scene.cycles.samples = 1

    # Bake POSITION
    tex_node.image = pos_img
    bpy.ops.object.bake(type='POSITION')
    pos_bl = np.array(pos_img.pixels[:], dtype=np.float32).reshape((tex_res, tex_res, 4))[..., :3]

    # Bake NORMAL
    tex_node.image = norm_img
    bpy.ops.object.bake(type='NORMAL', normal_space='OBJECT')
    norm_raw = np.array(norm_img.pixels[:], dtype=np.float32).reshape((tex_res, tex_res, 4))
    valid_uv_mask = norm_raw[..., 3] > 0.5
    # In Blender, NORMAL bake maps [-1, 1] to [0, 1] range. Decode back to unit vectors:
    norm_bl = norm_raw[..., :3] * 2.0 - 1.0

    # Free temporary bake images from memory
    bpy.data.images.remove(pos_img)
    bpy.data.images.remove(norm_img)

    # 5. Convert Coordinates to Canonical glTF Space (Part 4)
    # glTF: +Y is UP, +Z is FRONT, +X is RIGHT
    # Blender: +Z is UP, -Y is FRONT, +X is RIGHT
    # X_gltf = X_bl, Y_gltf = Z_bl, Z_gltf = -Y_bl
    pos_gltf = np.zeros_like(pos_bl)
    pos_gltf[..., 0] = pos_bl[..., 0]
    pos_gltf[..., 1] = pos_bl[..., 2]
    pos_gltf[..., 2] = -pos_bl[..., 1]

    norm_gltf = np.zeros_like(norm_bl)
    norm_gltf[..., 0] = norm_bl[..., 0]
    norm_gltf[..., 1] = norm_bl[..., 2]
    norm_gltf[..., 2] = -norm_bl[..., 1]

    norm_len = np.linalg.norm(norm_gltf, axis=-1)
    norm_gltf[valid_uv_mask] /= np.maximum(norm_len[valid_uv_mask, None], 1e-6)

    # Export UV Debug layout if requested (Part 13)
    if debug_dir:
        uv_debug_path = os.path.join(debug_dir, "11_uv_debug.png")
        try:
            uv_debug = np.zeros((tex_res, tex_res, 4), dtype=np.float32)
            uv_debug[valid_uv_mask] = [0.15, 0.55, 0.85, 1.0]
            island_borders = valid_uv_mask ^ np.roll(valid_uv_mask, 1, axis=0) ^ np.roll(valid_uv_mask, 1, axis=1)
            uv_debug[island_borders] = [1.0, 1.0, 1.0, 1.0]
            save_texture_image("UV_Debug_Layout", uv_debug, uv_debug_path)
            print(f"Saved UV debug layout to {uv_debug_path}")
        except Exception as e:
            print(f"UV layout export note: {e}")

    print(f"Blender Baker: Valid active UV texels: {np.count_nonzero(valid_uv_mask)} / {tex_res*tex_res}")

    # Compute Canonical Mesh Bounding Box
    all_verts_bl = np.array([v.co for v in obj.data.vertices], dtype=np.float32)
    all_verts_gltf = np.zeros_like(all_verts_bl)
    all_verts_gltf[:, 0] = all_verts_bl[:, 0]
    all_verts_gltf[:, 1] = all_verts_bl[:, 2]
    all_verts_gltf[:, 2] = -all_verts_bl[:, 1]

    min_b = all_verts_gltf.min(axis=0)
    max_b = all_verts_gltf.max(axis=0)
    span = np.maximum(max_b - min_b, 1e-5)
    max_extent = float(np.max(span))
    eps = 0.035 * max_extent

    print(f"Canonical Bounds: Min={min_b}, Max={max_b}, Extent={max_extent:.3f}")

    # 6. Load View Images and Metadata (Part 4)
    views_data = {}
    if views_json_path and os.path.exists(views_json_path):
        with open(views_json_path, "r") as f:
            views_data = json.load(f)

    if not views_data:
        print("WARNING: No views_json provided.")
        sys.exit(1)

    loaded_images = {}
    for vname, vinfo in views_data.items():
        ipath = vinfo.get("image_path")
        if ipath and os.path.exists(ipath):
            try:
                px = load_image_pixels(ipath)
                loaded_images[vname] = {
                    "pixels": px,
                    "meta": vinfo.get("metadata", {}),
                }
                print(f"Loaded view '{vname}' ({px.shape[1]}x{px.shape[0]})")
            except Exception as e:
                print(f"Could not load image for view '{vname}': {e}")

    if not loaded_images:
        print("ERROR: No view images could be loaded for texture baking.")
        sys.exit(1)

    # 7. C++ Accelerated Occlusion Depth Maps via BVHTree (Part 5 & 6)
    print("Blender Baker: Building BVHTree and computing camera depth maps...")
    t0 = time.time()
    bvh = BVHTree.FromPolygons([v.co for v in obj.data.vertices], [p.vertices for p in obj.data.polygons])

    depth_res = 256
    cam_dist = max_extent * 3.0
    depth_buffers = {}

    for vname in loaded_images.keys():
        dmap = np.full((depth_res, depth_res), 9999.0, dtype=np.float32)
        if vname == "front":
            xs = np.linspace(min_b[0], max_b[0], depth_res)
            ys = np.linspace(min_b[1], max_b[1], depth_res)
            X_g, Y_g = np.meshgrid(xs, ys)
            for r in range(depth_res):
                for c in range(depth_res):
                    orig = Vector((X_g[r, c], -cam_dist, Y_g[r, c]))
                    hit_loc, _, _, _ = bvh.ray_cast(orig, Vector((0, 1, 0)))
                    if hit_loc:
                        dmap[r, c] = max_b[2] - (-hit_loc.y)
        elif vname == "back":
            xs = np.linspace(max_b[0], min_b[0], depth_res)
            ys = np.linspace(min_b[1], max_b[1], depth_res)
            X_g, Y_g = np.meshgrid(xs, ys)
            for r in range(depth_res):
                for c in range(depth_res):
                    orig = Vector((X_g[r, c], cam_dist, Y_g[r, c]))
                    hit_loc, _, _, _ = bvh.ray_cast(orig, Vector((0, -1, 0)))
                    if hit_loc:
                        dmap[r, c] = (-hit_loc.y) - min_b[2]
        elif vname == "right":
            zs_gltf = np.linspace(max_b[2], min_b[2], depth_res)
            ys_gltf = np.linspace(min_b[1], max_b[1], depth_res)
            Z_g, Y_g = np.meshgrid(zs_gltf, ys_gltf)
            for r in range(depth_res):
                for c in range(depth_res):
                    orig = Vector((cam_dist, -Z_g[r, c], Y_g[r, c]))
                    hit_loc, _, _, _ = bvh.ray_cast(orig, Vector((-1, 0, 0)))
                    if hit_loc:
                        dmap[r, c] = max_b[0] - hit_loc.x
        elif vname == "left":
            zs_gltf = np.linspace(min_b[2], max_b[2], depth_res)
            ys_gltf = np.linspace(min_b[1], max_b[1], depth_res)
            Z_g, Y_g = np.meshgrid(zs_gltf, ys_gltf)
            for r in range(depth_res):
                for c in range(depth_res):
                    orig = Vector((-cam_dist, -Z_g[r, c], Y_g[r, c]))
                    hit_loc, _, _, _ = bvh.ray_cast(orig, Vector((1, 0, 0)))
                    if hit_loc:
                        dmap[r, c] = hit_loc.x - min_b[0]
        elif vname == "top":
            xs_gltf = np.linspace(min_b[0], max_b[0], depth_res)
            zs_gltf = np.linspace(max_b[2], min_b[2], depth_res)
            X_g, Z_g = np.meshgrid(xs_gltf, zs_gltf)
            for r in range(depth_res):
                for c in range(depth_res):
                    orig = Vector((X_g[r, c], -Z_g[r, c], cam_dist))
                    hit_loc, _, _, _ = bvh.ray_cast(orig, Vector((0, 0, -1)))
                    if hit_loc:
                        dmap[r, c] = max_b[1] - hit_loc.z
        elif vname == "bottom":
            xs_gltf = np.linspace(min_b[0], max_b[0], depth_res)
            zs_gltf = np.linspace(min_b[2], max_b[2], depth_res)
            X_g, Z_g = np.meshgrid(xs_gltf, zs_gltf)
            for r in range(depth_res):
                for c in range(depth_res):
                    orig = Vector((X_g[r, c], -Z_g[r, c], -cam_dist))
                    hit_loc, _, _, _ = bvh.ray_cast(orig, Vector((0, 0, 1)))
                    if hit_loc:
                        dmap[r, c] = hit_loc.z - min_b[1]

        depth_buffers[vname] = dmap
    print(f"Blender Baker: Computed {len(depth_buffers)} depth maps in {time.time() - t0:.3f}s.")

    # 8. Texel-Level Candidate Selection & Projection (Part 5, 6, 7, 8)
    print("Blender Baker: Projecting photographic texels with occlusion filtering and view scoring...")

    # Canonical camera forward vectors (+Z is front, +X is right, +Y is top)
    gltf_cam_vectors = {
        "front":  np.array([0.0,  0.0,  1.0], dtype=np.float32),
        "back":   np.array([0.0,  0.0, -1.0], dtype=np.float32),
        "right":  np.array([1.0,  0.0,  0.0], dtype=np.float32),
        "left":   np.array([-1.0, 0.0,  0.0], dtype=np.float32),
        "top":    np.array([0.0,  1.0,  0.0], dtype=np.float32),
        "bottom": np.array([0.0, -1.0,  0.0], dtype=np.float32),
    }

    # Extract coordinates of valid texels
    valid_y, valid_x = np.where(valid_uv_mask)
    n_active = len(valid_y)
    p_active = pos_gltf[valid_y, valid_x]      # (N, 3)
    n_active_vec = norm_gltf[valid_y, valid_x] # (N, 3)

    view_names = list(loaded_images.keys())
    n_views = len(view_names)

    cand_scores = np.zeros((n_active, n_views), dtype=np.float32)
    cand_colors = np.zeros((n_active, n_views, 4), dtype=np.float32)

    # Configurable scoring weights (Part 7)
    W_NORMAL = 0.40
    W_VIS = 0.25
    W_QUALITY = 0.15
    W_DIST = 0.10
    W_CENTER = 0.10

    for v_idx, vname in enumerate(view_names):
        if vname not in gltf_cam_vectors:
            continue
        c_vec = gltf_cam_vectors[vname]
        im_px = loaded_images[vname]["pixels"]
        im_h, im_w = im_px.shape[:2]
        meta = loaded_images[vname]["meta"]

        # a) Back-face test (Part 5): normal dot camera vector > 0.02
        dot = np.sum(n_active_vec * c_vec, axis=-1)
        facing_mask = dot > 0.02

        if not np.any(facing_mask):
            continue

        # b) Projection into normalized image space [0, 1] x [0, 1]
        X = p_active[:, 0]
        Y = p_active[:, 1]
        Z = p_active[:, 2]

        if vname == "front":
            u_norm = (X - min_b[0]) / span[0]
            v_norm = (Y - min_b[1]) / span[1]
            exp_depth = max_b[2] - Z
        elif vname == "back":
            u_norm = (max_b[0] - X) / span[0]
            v_norm = (Y - min_b[1]) / span[1]
            exp_depth = Z - min_b[2]
        elif vname == "right":
            u_norm = (max_b[2] - Z) / span[2]
            v_norm = (Y - min_b[1]) / span[1]
            exp_depth = max_b[0] - X
        elif vname == "left":
            u_norm = (Z - min_b[2]) / span[2]
            v_norm = (Y - min_b[1]) / span[1]
            exp_depth = X - min_b[0]
        elif vname == "top":
            u_norm = (X - min_b[0]) / span[0]
            v_norm = (max_b[2] - Z) / span[2]
            exp_depth = max_b[1] - Y
        elif vname == "bottom":
            u_norm = (X - min_b[0]) / span[0]
            v_norm = (Z - min_b[2]) / span[2]
            exp_depth = Y - min_b[1]

        # Apply letterbox offset & scale if present
        if meta and "offset_x" in meta and "scale" in meta:
            scale = meta["scale"]
            ox = meta["offset_x"]
            oy = meta["offset_y"]
            cw = meta.get("crop_width", im_w)
            ch = meta.get("crop_height", im_h)
            px_x = ox + u_norm * (cw * scale)
            px_y = oy + (1.0 - v_norm) * (ch * scale)
            u_px = np.clip(np.round(px_x), 0, im_w - 1).astype(int)
            v_px = np.clip(np.round(px_y), 0, im_h - 1).astype(int)
        else:
            u_px = np.clip(np.round(u_norm * (im_w - 1)), 0, im_w - 1).astype(int)
            v_px = np.clip(np.round((1.0 - v_norm) * (im_h - 1)), 0, im_h - 1).astype(int)

        # c) Image bounds & Foreground alpha test
        in_bounds = (u_norm >= -0.05) & (u_norm <= 1.05) & (v_norm >= -0.05) & (v_norm <= 1.05)
        sampled = im_px[v_px, u_px]
        has_alpha = sampled[..., 3] > 0.12

        valid_candidate = facing_mask & in_bounds & has_alpha

        # d) Occlusion Depth Test (Part 6)
        if vname in depth_buffers and np.any(valid_candidate):
            dmap = depth_buffers[vname]

            u_grid = np.clip(np.round(u_norm * (depth_res - 1)), 0, depth_res - 1).astype(int)
            v_grid = np.clip(np.round(v_norm * (depth_res - 1)), 0, depth_res - 1).astype(int)

            hit_depth = dmap[v_grid, u_grid]
            is_occluded = exp_depth > (hit_depth + eps)
            valid_candidate = valid_candidate & (~is_occluded)

        # e) View Quality Score (Part 7)
        if np.any(valid_candidate):
            norm_score = np.clip(dot[valid_candidate], 0.0, 1.0) ** 1.4
            vis_score = 1.0
            qual_score = 1.0 if vname == "front" else 0.85
            dist_score = 1.0 - np.clip(exp_depth[valid_candidate] / max_extent, 0.0, 1.0)
            center_dist = np.sqrt((u_norm[valid_candidate] - 0.5)**2 + (v_norm[valid_candidate] - 0.5)**2)
            center_score = 1.0 - np.clip(center_dist * 1.2, 0.0, 1.0)

            total_score = (
                W_NORMAL * norm_score +
                W_VIS * vis_score +
                W_QUALITY * qual_score +
                W_DIST * dist_score +
                W_CENTER * center_score
            )
            cand_scores[valid_candidate, v_idx] = total_score
            cand_colors[valid_candidate, v_idx] = sampled[valid_candidate]

    # 9. Multi-View Confidence Blending (Part 11)
    print("Blender Baker: Performing seamless confidence-weighted multi-view blending...")
    sum_scores = np.sum(cand_scores, axis=1)
    seen_mask = sum_scores > 1e-4
    best_cam = np.argmax(cand_scores, axis=1)

    final_active_colors = np.zeros((n_active, 4), dtype=np.float32)

    weights = (cand_scores ** 2.0)
    w_sum = np.sum(weights, axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    norm_weights = weights / w_sum

    for v_idx in range(n_views):
        final_active_colors += norm_weights[:, v_idx:v_idx+1] * cand_colors[:, v_idx]

    # Handle unseen/shadowed texels: fallback to median primary color
    if not np.all(seen_mask):
        unseen_count = np.count_nonzero(~seen_mask)
        primary_key = "front" if "front" in loaded_images else list(loaded_images.keys())[0]
        p_fg = loaded_images[primary_key]["pixels"][loaded_images[primary_key]["pixels"][..., 3] > 0.15]
        fallback_rgb = np.median(p_fg, axis=0) if len(p_fg) > 0 else np.array([0.84, 0.84, 0.84, 1.0], dtype=np.float32)
        final_active_colors[~seen_mask] = fallback_rgb
        print(f"Blender Baker: Diffused {unseen_count} occluded/unseen texels using reference median.")

    # 10. Construct Final BaseColor Texture Map & Dilate UV Margins (Part 14)
    texture_rgba = np.zeros((tex_res, tex_res, 4), dtype=np.float32)
    final_active_colors[:, 3] = 1.0
    texture_rgba[valid_y, valid_x] = np.clip(final_active_colors, 0.0, 1.0)

    # Dilate active texels into empty UV background in pure NumPy
    texture_rgba = dilate_texture(texture_rgba, valid_uv_mask, steps=6)
    texture_rgba[..., 3] = 1.0

    # 11. Save Debug Outputs (Part 18)
    if debug_dir:
        # Save final texture PNG
        final_tex_path = os.path.join(debug_dir, "19_final_texture.png")
        save_texture_image("Final_Texture_Debug", texture_rgba, final_tex_path)

        # Save visibility debug map (color-coded by winning camera view)
        vis_debug = np.zeros((tex_res, tex_res, 4), dtype=np.float32)
        vis_debug[..., 3] = 1.0
        cam_colors_palette = [
            [0.90, 0.15, 0.15, 1.0], # Front: Red
            [0.15, 0.30, 0.90, 1.0], # Back: Blue
            [0.15, 0.85, 0.25, 1.0], # Right: Green
            [0.90, 0.85, 0.15, 1.0], # Left: Yellow
            [0.15, 0.90, 0.90, 1.0], # Top: Cyan
            [0.90, 0.15, 0.90, 1.0], # Bottom: Magenta
        ]
        for i in range(min(n_views, len(cam_colors_palette))):
            active_c = (best_cam == i) & seen_mask
            vis_debug[valid_y[active_c], valid_x[active_c]] = cam_colors_palette[i]
        save_texture_image("Visibility_Debug", vis_debug, os.path.join(debug_dir, "18_visibility_debug.png"))
        print(f"Saved debug artifacts to {debug_dir}")

    # 12. Connect Baked Texture to Principled BSDF & Pack (Part 8 & 15)
    bake_png_path = os.path.splitext(glb_output)[0] + "_bake_tex.png"
    bake_img = save_texture_image("BaseColor_Texture", texture_rgba, bake_png_path)
    bake_img.colorspace_settings.name = 'sRGB'
    bake_img.pack()

    # Create Material
    obj.data.materials.clear()
    mat = bpy.data.materials.new(name="PBR_Photographic_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.40
    bsdf.inputs['Metallic'].default_value = 0.05

    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = bake_img
    img_node.image.colorspace_settings.name = 'sRGB'
    links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])

    final_out = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], final_out.inputs['Surface'])
    obj.data.materials.append(mat)

    # 13. Export Production GLB
    print(f"Blender Baker: Exporting Production GLB to {glb_output}...")
    try:
        bpy.ops.export_scene.gltf(
            filepath=glb_output,
            export_format='GLB',
            export_materials='EXPORT',
            export_image_format='AUTO',
            export_apply=True
        )
        print("Blender Baker: GLB export completed successfully.")
    except Exception as e:
        print(f"ERROR exporting GLB: {e}")
        sys.exit(1)

    # 14. Save native .blend project file if requested
    if blend_output:
        try:
            print(f"Blender Baker: Saving native .blend to {blend_output}...")
            sun_data = bpy.data.lights.new(name="Sun", type='SUN')
            sun_data.energy = 3.5
            sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)
            sun_obj.location = (4.0, -4.0, 8.0)
            bpy.context.collection.objects.link(sun_obj)

            cam_data = bpy.data.cameras.new(name="Camera")
            cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
            cam_obj.location = (0.0, -3.5, 1.0)
            cam_obj.rotation_euler = (1.57, 0.0, 0.0)
            bpy.context.collection.objects.link(cam_obj)

            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.shading.type = 'MATERIAL'

            bpy.ops.wm.save_as_mainfile(filepath=blend_output)
            print("Blender Baker: .blend project file saved.")
        except Exception as e:
            print(f"Warning saving .blend: {e}")

    # 15. Automated Color Validation Report (Part 19 & 21)
    report = {
        "tex_resolution": tex_res,
        "active_texels": int(n_active),
        "seen_texels": int(np.count_nonzero(seen_mask)),
        "unseen_texels": int(np.count_nonzero(~seen_mask)),
        "coverage_percent": round(float(np.count_nonzero(seen_mask) / max(1, n_active) * 100), 2),
    }

    if "front" in loaded_images and np.any(seen_mask):
        front_idx = view_names.index("front")
        front_cand = (best_cam == front_idx) & seen_mask
        if np.any(front_cand):
            src_col = cand_colors[front_cand, front_idx][..., :3] * 255.0
            baked_col = final_active_colors[front_cand][..., :3] * 255.0
            diff = np.abs(src_col - baked_col)
            report["front_mean_rgb_error"] = round(float(np.mean(diff)), 2)
            report["front_median_rgb_error"] = round(float(np.median(diff)), 2)
            report["front_high_error_percent"] = round(float(np.count_nonzero(np.max(diff, axis=-1) > 15) / max(1, len(diff)) * 100), 2)

    val_json_path = os.path.splitext(glb_output)[0] + "_color_validation.json"
    with open(val_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Blender Baker Validation Report: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    try:
        run_bake()
    except Exception as exc:
        print(f"FATAL: Blender Baker failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
