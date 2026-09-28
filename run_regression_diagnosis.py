"""
run_regression_diagnosis.py - Strict diagnostic comparison between CURRENT and PREVIOUS pipelines.
Executes both on the EXACT SAME 6 input images and computes all required comparison metrics.
"""

import sys
import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.measure import marching_cubes
from PIL import Image

ROOT = Path(__file__).parent.resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.segmentation import ObjectSegmenter
from src.camera_alignment import CameraAlignment
from src.geometry_validation import GeometryValidator


def run_previous_pipeline(input_dir: Path, output_dir: Path, blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender"):
    """
    Executes the PREVIOUS working generic pipeline:
      - Visual hull space carving with Z-bounded top/bottom profiles
      - Marching cubes + Taubin volume-preserving smoothing (8 iters)
      - Continuous cylindrical body + top/bottom disc UV unwrapping
      - Panoramic multi-view photographic texture blending
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    texture_dir = output_dir / "texture"
    final_dir = output_dir / "final"
    renders_dir = output_dir / "renders"
    for d in [model_dir, texture_dir, final_dir, renders_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Segment
    segmenter = ObjectSegmenter()
    segmented_views = segmenter.segment_all(input_dir)

    masks = {}
    for view_name in ["front", "back", "left", "right", "top", "bottom"]:
        raw_m = segmented_views[view_name]["mask"]
        if raw_m.ndim == 3:
            raw_m = raw_m[..., 0]
        bin_m = binary_fill_holes(raw_m > 128).astype(np.uint8)
        masks[view_name] = bin_m

    # 2. Extents
    f_ys, f_xs = np.where(masks["front"] > 0)
    b_ys, b_xs = np.where(masks["back"] > 0)
    l_ys, l_xs = np.where(masks["left"] > 0)
    r_ys, r_xs = np.where(masks["right"] > 0)
    t_ys, t_xs = np.where(masks["top"] > 0)
    bot_ys, bot_xs = np.where(masks["bottom"] > 0)

    y_min = min(f_ys.min(), b_ys.min(), l_ys.min(), r_ys.min())
    y_max = max(f_ys.max(), b_ys.max(), l_ys.max(), r_ys.max())
    z_span_px = float(y_max - y_min + 1)
    half_h_z = z_span_px / 2.0

    max_w_x = max(f_xs.max() - f_xs.min() + 1, b_xs.max() - b_xs.min() + 1) / 2.0
    max_w_y = max(l_xs.max() - l_xs.min() + 1, r_xs.max() - r_xs.min() + 1) / 2.0

    cx_f = (f_xs.min() + f_xs.max()) / 2.0
    cy_f = (y_min + y_max) / 2.0
    cx_b = (b_xs.min() + b_xs.max()) / 2.0
    cy_b = (y_min + y_max) / 2.0
    cx_l = (l_xs.min() + l_xs.max()) / 2.0
    cy_l = (y_min + y_max) / 2.0
    cx_r = (r_xs.min() + r_xs.max()) / 2.0
    cy_r = (y_min + y_max) / 2.0

    cx_t = (t_xs.min() + t_xs.max()) / 2.0
    cy_t = (t_ys.min() + t_ys.max()) / 2.0
    hw_t_x = (t_xs.max() - t_xs.min() + 1) / 2.0
    hw_t_y = (t_ys.max() - t_ys.min() + 1) / 2.0

    cx_bot = (bot_xs.min() + bot_xs.max()) / 2.0
    cy_bot = (bot_ys.min() + bot_ys.max()) / 2.0
    hw_bot_x = (bot_xs.max() - bot_xs.min() + 1) / 2.0
    hw_bot_y = (bot_ys.max() - bot_ys.min() + 1) / 2.0

    rx = float(max_w_x / half_h_z)
    ry = float(max_w_y / half_h_z)
    rz = 1.0

    # 3. 2D Distance Fields
    dts = {}
    for vname, m in masks.items():
        dt_pos = distance_transform_edt(m)
        dt_neg = distance_transform_edt(1 - m)
        dts[vname] = dt_pos - dt_neg

    H_img, W_img = masks["front"].shape[:2]
    N = 180
    margin = 1.12
    x_vals = np.linspace(-rx * margin, rx * margin, N, dtype=np.float32)
    y_vals = np.linspace(-ry * margin, ry * margin, N, dtype=np.float32)
    z_vals = np.linspace(-rz * margin, rz * margin, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

    def sample_dt(name, u_grid, v_grid):
        u_c = np.clip(np.round(u_grid).astype(np.int32), 0, W_img - 1)
        v_c = np.clip(np.round(v_grid).astype(np.int32), 0, H_img - 1)
        res = dts[name][v_c, u_c].astype(np.float32)
        out_b = (u_grid < 0) | (u_grid >= W_img) | (v_grid < 0) | (v_grid >= H_img)
        res[out_b] = -100.0
        return res

    sdf_f = sample_dt("front", cx_f + X * half_h_z, cy_f - Z * half_h_z)
    sdf_b = sample_dt("back", cx_b - X * half_h_z, cy_b - Z * half_h_z)
    sdf_l = sample_dt("left", cx_l + Y * half_h_z, cy_l - Z * half_h_z)
    sdf_r = sample_dt("right", cx_r - Y * half_h_z, cy_r - Z * half_h_z)
    sdf_t = sample_dt("top", cx_t + (X / rx) * hw_t_x, cy_t - (Y / ry) * hw_t_y)
    sdf_bot = sample_dt("bottom", cx_bot + (X / rx) * hw_bot_x, cy_bot + (Y / ry) * hw_bot_y)

    sdf_sides = np.minimum(np.minimum(sdf_f, sdf_b), np.minimum(sdf_l, sdf_r))
    top_blend = np.clip((Z - 0.75) / 0.20, 0.0, 1.0)
    sdf_t_weighted = (1.0 - top_blend) * 100.0 + top_blend * sdf_t
    sdf = np.minimum(sdf_sides, sdf_t_weighted)

    bot_blend = np.clip((-Z - 0.75) / 0.20, 0.0, 1.0)
    sdf_bot_weighted = (1.0 - bot_blend) * 100.0 + bot_blend * sdf_bot
    sdf = np.minimum(sdf, sdf_bot_weighted)

    # 4. Marching cubes
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    dz = z_vals[1] - z_vals[0]
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(dx, dy, dz))
    verts += np.array([x_vals[0], y_vals[0], z_vals[0]], dtype=np.float32)
    faces = faces[:, [0, 2, 1]]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Taubin smoothing only (clean volume preservation, NO vertex distortion)
    trimesh.smoothing.filter_taubin(mesh, iterations=8)

    raw_obj = model_dir / "reconstructed_raw.obj"
    glb_path = model_dir / "reconstructed_raw.glb"
    blend_path = model_dir / "reconstructed_mesh.blend"
    mesh.export(str(raw_obj))

    # Blender decimation to 14,000 faces
    import subprocess
    script = f'''
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"{raw_obj}", up_axis="Z", forward_axis="Y")
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
n_polys = len(obj.data.polygons)
if n_polys > 14000:
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.ratio = 14000.0 / float(n_polys)
    bpy.ops.object.modifier_apply(modifier="Decimate")
for p in obj.data.polygons:
    p.use_smooth = True
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.obj_export(filepath=r"{raw_obj}", up_axis="Z", forward_axis="Y")
bpy.ops.export_scene.gltf(filepath=r"{glb_path}", export_format='GLB')
bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")
'''
    py_tmp = model_dir / "decimate.py"
    py_tmp.write_text(script)
    subprocess.run([blender_bin, "--background", "--python", str(py_tmp)], capture_output=True)
    py_tmp.unlink(missing_ok=True)

    # 5. Continuous Cylindrical + Cap UV Unwrapping (Generic for any upright product)
    uv_blend = model_dir / "reconstructed_uv.blend"
    uv_obj = model_dir / "reconstructed_uv.obj"
    uv_png = model_dir / "uv_layout.png"
    uv_script = f'''
import bpy
import numpy as np
import math
from PIL import Image, ImageDraw

bpy.ops.wm.open_mainfile(filepath=r"{blend_path}")
obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
mesh = obj.data
uv_layer = mesh.uv_layers.new(name="UVMap") if not mesh.uv_layers else mesh.uv_layers[0]

n_verts = len(mesh.vertices)
n_loops = len(mesh.loops)
coords = np.empty((n_verts, 3), dtype=np.float32)
normals = np.empty((n_verts, 3), dtype=np.float32)
mesh.vertices.foreach_get("co", coords.ravel())
mesh.vertices.foreach_get("normal", normals.ravel())

min_b = coords.min(axis=0)
max_b = coords.max(axis=0)
dim = np.maximum(max_b - min_b, 1e-6)

vz = (coords[:, 2] - min_b[2]) / dim[2]
is_top = (vz > 0.88) & (normals[:, 2] > 0.35)
is_bottom = (vz < 0.12) & (normals[:, 2] < -0.35)

# Continuous 360-degree azimuth
thetas = np.arctan2(coords[:, 0], -coords[:, 1])
u_body = (thetas / (2.0 * math.pi)) + 0.5
v_body = np.clip(vz * 0.74, 0.0, 0.74)

loop_verts = np.empty(n_loops, dtype=np.int32)
mesh.loops.foreach_get("vertex_index", loop_verts)
uv_data = np.zeros((n_loops, 2), dtype=np.float32)

r_xy = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
r_top_ref = max(float(np.max(r_xy[is_top])), 1e-4) if np.any(is_top) else 0.5
r_bot_ref = max(float(np.max(r_xy[is_bottom])), 1e-4) if np.any(is_bottom) else 0.5

# Top cap: centered at (0.25, 0.875) with radius 0.10
# Bottom cap: centered at (0.75, 0.875) with radius 0.10
for i in range(n_loops):
    vi = loop_verts[i]
    if is_top[vi]:
        ux = coords[vi, 0] / r_top_ref * 0.10 + 0.25
        vy = coords[vi, 1] / r_top_ref * 0.10 + 0.875
        uv_data[i] = [np.clip(ux, 0.13, 0.37), np.clip(vy, 0.755, 0.995)]
    elif is_bottom[vi]:
        ux = coords[vi, 0] / r_bot_ref * 0.10 + 0.75
        vy = coords[vi, 1] / r_bot_ref * 0.10 + 0.875
        uv_data[i] = [np.clip(ux, 0.63, 0.87), np.clip(vy, 0.755, 0.995)]
    else:
        uv_data[i] = [u_body[vi], v_body[vi]]

# Fix seam triangles that cross u=0 / u=1 boundary
for poly in mesh.polygons:
    p_loops = poly.loop_indices
    u_vals = [uv_data[li][0] for li in p_loops]
    if max(u_vals) - min(u_vals) > 0.5:
        for li in p_loops:
            if uv_data[li][0] < 0.5:
                uv_data[li][0] += 1.0

uv_layer.data.foreach_set("uv", uv_data.ravel())
bpy.ops.wm.save_as_mainfile(filepath=r"{uv_blend}")
bpy.ops.wm.obj_export(filepath=r"{uv_obj}", up_axis="Z", forward_axis="Y")
'''
    py_uv = model_dir / "uv_script.py"
    py_uv.write_text(uv_script)
    res_uv = subprocess.run([blender_bin, "--background", "--python", str(py_uv)], capture_output=True, text=True)
    if not uv_blend.exists():
        raise RuntimeError(f"UV layout script failed:\n{res_uv.stdout}\n{res_uv.stderr}")
    py_uv.unlink(missing_ok=True)

    # 6. Panoramic Photographic Texture Blending
    # pyrefly: ignore [missing-import]
    from src.texture_projection import TextureProjector
    projector = TextureProjector()
    cropped_views = projector.extract_cropped_views(segmented_views)

    W = H = 4096
    front_rgb = cv2.resize(cropped_views["front"][..., :3], (1024, 1024))
    fill_color = np.median(front_rgb.reshape(-1, 3), axis=0).astype(np.uint8)
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    canvas[:, :, :3] = fill_color
    canvas[:, :, 3] = 255

    body_h = int(H * 0.74)
    # Blend 4 lateral views horizontally: back_left -> left -> front -> right -> back_right
    bl_crop = cropped_views["back"][:, :cropped_views["back"].shape[1]//2, :3]
    br_crop = cropped_views["back"][:, cropped_views["back"].shape[1]//2:, :3]
    l_crop = cropped_views["left"][..., :3]
    f_crop = cropped_views["front"][..., :3]
    r_crop = cropped_views["right"][..., :3]

    w_seg = W // 4
    # Seamless horizontal placement
    strip_bl = cv2.resize(bl_crop, (w_seg // 2, body_h))
    strip_l = cv2.resize(l_crop, (w_seg, body_h))
    strip_f = cv2.resize(f_crop, (w_seg, body_h))
    strip_r = cv2.resize(r_crop, (w_seg, body_h))
    strip_br = cv2.resize(br_crop, (w_seg // 2, body_h))

    body_pano = np.hstack([strip_bl, strip_l, strip_f, strip_r, strip_br])
    body_pano = cv2.resize(body_pano, (W, body_h))
    canvas[:body_h, :W, :3] = body_pano

    # Place top and bottom circular caps into top region
    top_img = cv2.resize(cropped_views["top"][..., :3], (int(W * 0.20), int(H * 0.20)))
    bot_img = cv2.resize(cropped_views["bottom"][..., :3], (int(W * 0.20), int(H * 0.20)))

    # Circle masks
    ch, cw = top_img.shape[:2]
    c_mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.circle(c_mask, (cw // 2, ch // 2), min(cw, ch) // 2, 255, -1)

    y_top = int(H * 0.775)
    x_top = int(W * 0.15)
    canvas_slice_t = canvas[y_top : y_top + ch, x_top : x_top + cw, :3]
    canvas_slice_t[c_mask > 0] = top_img[c_mask > 0]

    x_bot = int(W * 0.65)
    canvas_slice_b = canvas[y_top : y_top + ch, x_bot : x_bot + cw, :3]
    canvas_slice_b[c_mask > 0] = bot_img[c_mask > 0]

    tex_basecolor = texture_dir / "texture_basecolor.png"
    Image.fromarray(canvas).save(tex_basecolor)

    # 7. Final GLB Export
    final_glb = final_dir / "reconstructed_product.glb"
    export_script = f'''
import bpy
bpy.ops.wm.open_mainfile(filepath=r"{uv_blend}")
obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]
mat = bpy.data.materials.new(name="ProductPBR")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
tex_node.image = bpy.data.images.load(r"{tex_basecolor}")
mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
obj.data.materials.clear()
obj.data.materials.append(mat)
bpy.ops.export_scene.gltf(filepath=r"{final_glb}", export_format='GLB')
'''
    uv_blend = uv_blend.resolve()
    uv_obj = uv_obj.resolve()
    final_glb = final_glb.resolve()
    tex_basecolor = tex_basecolor.resolve()

    py_exp = final_dir / "export.py"
    py_exp.write_text(export_script)
    res = subprocess.run([blender_bin, "--background", "--python", str(py_exp)], capture_output=True, text=True)
    if not final_glb.exists() or final_glb.stat().st_size < 1000:
        raise RuntimeError(f"GLB export failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    py_exp.unlink(missing_ok=True)

    # 8. Render validation views
    from src.validation import ModelValidator
    validator = ModelValidator(blender_bin=blender_bin)
    validator.render_validation_views(final_glb, renders_dir, resolution=512)

    final_mesh = trimesh.load(str(raw_obj))
    return {
        "raw_obj": raw_obj,
        "glb": final_glb,
        "dimensions": [round(float(d), 4) for d in final_mesh.extents],
        "vertices": len(final_mesh.vertices),
        "faces": len(final_mesh.faces),
    }


def main():
    input_dir = Path("backend/input").resolve()
    curr_dir = Path("temp/current_test").resolve()
    prev_dir = Path("temp/previous_test").resolve()

    print("======================================================================")
    print("STEP 1: RUNNING PREVIOUS PIPELINE (TAUBIN SMOOTHING + CONTINUOUS UV)")
    print("======================================================================")
    t0 = time.time()
    prev_stats = run_previous_pipeline(input_dir, prev_dir)
    prev_elapsed = round(time.time() - t0, 2)
    print(f"PREVIOUS pipeline completed in {prev_elapsed}s: {prev_stats}")

    print("\n======================================================================")
    print("STEP 2: RUNNING CURRENT PIPELINE (GEOMETRY REFINEMENT + SMART UV)")
    print("======================================================================")
    from src.main import run_reconstruction
    t1 = time.time()
    curr_res = run_reconstruction(input_dir, curr_dir, texture_resolution=4096)
    curr_elapsed = round(time.time() - t1, 2)
    curr_mesh = trimesh.load(str(curr_res["models"]["obj"]))
    curr_stats = {
        "raw_obj": Path(curr_res["models"]["obj"]),
        "glb": Path(curr_res["final"]["glb"]),
        "dimensions": [round(float(d), 4) for d in curr_mesh.extents],
        "vertices": len(curr_mesh.vertices),
        "faces": len(curr_mesh.faces),
    }
    print(f"CURRENT pipeline completed in {curr_elapsed}s: {curr_stats}")

    print("\n======================================================================")
    print("STEP 3: EVALUATING SILHOUETTE METRICS FOR BOTH ON EXACT SAME 6 MASKS")
    print("======================================================================")
    from src.segmentation import ObjectSegmenter
    from src.camera_alignment import CameraAlignment
    segmenter = ObjectSegmenter()
    segmented_views = segmenter.segment_all(input_dir)
    aligner = CameraAlignment()
    calibs = aligner.calibrate_views(segmented_views)

    validator = GeometryValidator()
    print("Evaluating PREVIOUS mesh silhouette metrics...")
    val_prev = validator.validate_geometry(
        mesh_path=prev_stats["raw_obj"],
        segmented_views=segmented_views,
        calibrations=calibs,
        output_dir=prev_dir / "val_tmp",
    )

    print("Evaluating CURRENT mesh silhouette metrics...")
    val_curr = validator.validate_geometry(
        mesh_path=curr_stats["raw_obj"],
        segmented_views=segmented_views,
        calibrations=calibs,
        output_dir=curr_dir / "val_tmp",
    )

    comparison = {
        "previous": {
            "elapsed_s": prev_elapsed,
            "dimensions": prev_stats["dimensions"],
            "vertices": prev_stats["vertices"],
            "faces": prev_stats["faces"],
            "mean_iou": val_prev["mean_iou"],
            "mean_precision": val_prev["mean_precision"],
            "mean_recall": val_prev["mean_recall"],
            "mean_reproj_error_px": val_prev.get("mean_reprojection_error_px", 0.0),
            "per_view": val_prev["per_view"],
        },
        "current": {
            "elapsed_s": curr_elapsed,
            "dimensions": curr_stats["dimensions"],
            "vertices": curr_stats["vertices"],
            "faces": curr_stats["faces"],
            "mean_iou": val_curr["mean_iou"],
            "mean_precision": val_curr["mean_precision"],
            "mean_recall": val_curr["mean_recall"],
            "mean_reproj_error_px": val_curr.get("mean_reprojection_error_px", 0.0),
            "per_view": val_curr["per_view"],
        }
    }

    report_path = Path("temp/regression_comparison.json")
    report_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved regression comparison report to {report_path}")
    print("\nSUMMARY COMPARISON TABLE:")
    print("Metric                 | PREVIOUS (Clean Visual Hull) | CURRENT (Degraded Refinement)")
    print("-----------------------+------------------------------+------------------------------")
    print(f"Elapsed Time           | {prev_elapsed}s                       | {curr_elapsed}s")
    print(f"Dimensions (X, Y, Z)   | {prev_stats['dimensions']}     | {curr_stats['dimensions']}")
    print(f"Vertex Count           | {prev_stats['vertices']}                        | {curr_stats['vertices']}")
    print(f"Face Count             | {prev_stats['faces']}                        | {curr_stats['faces']}")
    print(f"Mean Silhouette IoU    | {val_prev['mean_iou']:.4f}                       | {val_curr['mean_iou']:.4f}")
    print(f"Mean Precision         | {val_prev['mean_precision']:.4f}                       | {val_curr['mean_precision']:.4f}")
    print(f"Mean Recall            | {val_prev['mean_recall']:.4f}                       | {val_curr['mean_recall']:.4f}")
    print(f"Mean Reproj Error (px) | {val_prev.get('mean_reprojection_error_px', 0.0):.2f}px                     | {val_curr.get('mean_reprojection_error_px', 0.0):.2f}px")


if __name__ == "__main__":
    main()

