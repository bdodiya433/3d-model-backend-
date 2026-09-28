import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

def isolate_main_object(image_path: str, output_path: str, padding_ratio: float = 0.15) -> Image.Image:
    """
    Intelligently finds and isolates the single main object from the image:
    1. Removes background (including dark/black backgrounds and internal cavities/holes).
    2. Strips out extraneous text labels (like 'Front View' or watermarks).
    3. Centers the isolated object on a square transparent canvas with padding.
    """
    print(f"[1/4] Isolating main object from: {image_path}")
    img_pil = Image.open(image_path).convert("RGBA")
    arr = np.array(img_pil)

    # Detect if background is predominantly dark (like black studio photos)
    # Check 4 corners
    h, w, _ = arr.shape
    corners = [
        arr[0:20, 0:20, :3],
        arr[0:20, w-20:w, :3],
        arr[h-20:h, 0:20, :3],
        arr[h-20:h, w-20:w, :3]
    ]
    corner_brightness = np.mean([np.mean(c) for c in corners])
    is_dark_bg = corner_brightness < 40

    if is_dark_bg:
        # Distance from black background
        # Background is black (R, G, B < 28)
        rgb_norm = np.linalg.norm(arr[..., :3].astype(np.float32), axis=-1)
        raw_mask = (rgb_norm > 35).astype(np.uint8) * 255
    else:
        # Use rembg session
        from rembg import remove, new_session
        session = new_session("u2netp")
        rem_img = remove(img_pil, session=session)
        raw_mask = np.array(rem_img.getchannel("A"), dtype=np.uint8)
        raw_mask = np.where(raw_mask > 20, 255, 0).astype(np.uint8)

    # Strip bottom text banner: bottom 12% often contains text like 'Front View'
    # If there's isolated text at the bottom, find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    
    # Filter out background label 0
    if num_labels > 1:
        # Find the largest connected component (the main physical object)
        main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_area = stats[main_label, cv2.CC_STAT_AREA]
        
        # Keep components that are either the main component or significantly connected parts
        clean_mask = np.zeros_like(raw_mask)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            top = stats[lbl, cv2.CC_STAT_TOP]
            left = stats[lbl, cv2.CC_STAT_LEFT]
            height = stats[lbl, cv2.CC_STAT_HEIGHT]
            width = stats[lbl, cv2.CC_STAT_WIDTH]
            
            # If component is tiny and located in the bottom 12% (text region), discard it
            if top > (h * 0.88) and area < (main_area * 0.05):
                continue
            # Keep if area is at least 0.5% of main object (e.g. separate earcups or arches)
            if area >= (main_area * 0.005) or lbl == main_label:
                clean_mask[labels == lbl] = 255
    else:
        clean_mask = raw_mask

    # Find tight bounding box of the clean mask
    ys, xs = np.where(clean_mask > 0)
    if len(ys) == 0:
        raise RuntimeError("No main object could be detected in image!")

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    # Crop both image and mask
    obj_h = ymax - ymin + 1
    obj_w = xmax - xmin + 1
    
    # Create RGBA with clean mask
    arr_clean = arr.copy()
    arr_clean[clean_mask == 0] = [0, 0, 0, 0]
    cropped_arr = arr_clean[ymin:ymax+1, xmin:xmax+1]

    # Pad to square transparent canvas
    max_dim = max(obj_w, obj_h)
    pad = int(max_dim * padding_ratio)
    canvas_size = max_dim + 2 * pad
    
    square_canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    offset_x = pad + (max_dim - obj_w) // 2
    offset_y = pad + (max_dim - obj_h) // 2
    square_canvas[offset_y:offset_y+obj_h, offset_x:offset_x+obj_w] = cropped_arr

    final_pil = Image.fromarray(square_canvas, mode="RGBA").resize((512, 512), Image.Resampling.LANCZOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_pil.save(output_path, "PNG")
    print(f"[1/4] Main object isolated & saved -> {output_path}")
    return final_pil

def run_hunyuan_ai_generation(input_image: Image.Image, output_glb: str):
    """
    Executes Tencent Hunyuan3D-2 AI model on the isolated main object image.
    """
    print(f"[2/4] Running Hunyuan3D-2 AI inference on isolated main object...")
    # Add third-party Hunyuan3D-2 to path
    backend_dir = Path(".").resolve()
    third_party = backend_dir / "third_party" / "Hunyuan3D-2"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))

    from app.ai.providers.hunyuan.utils import get_torch_device, get_torch_dtype
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    device = get_torch_device()
    dtype = get_torch_dtype(device)
    print(f"Device: {device}, Dtype: {dtype}")

    ckpt_path = Path("weights/hunyuan3d/model.fp16.safetensors")
    config_path = Path("weights/hunyuan3d/config.yaml")

    print("Loading Hunyuan3D pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=str(ckpt_path),
        config_path=str(config_path),
        device=device,
        dtype=dtype,
        use_safetensors=True
    )

    print("Starting Diffusion inference (20 steps)...")
    with torch.inference_mode():
        mesh = pipeline(
            image=input_image,
            num_inference_steps=20,
            octree_resolution=256,
            guidance_scale=5.0
        )

    target_mesh = mesh[0] if isinstance(mesh, list) else mesh
    output_glb = Path(output_glb)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    
    # Export raw GLB
    raw_glb_path = output_glb.parent / f"{output_glb.stem}_raw.glb"
    target_mesh.export(str(raw_glb_path))
    print(f"[2/4] Raw AI Mesh exported -> {raw_glb_path}")
    return raw_glb_path

def refine_mesh_in_blender(raw_glb: Path, final_glb: Path, renders_dir: Path):
    """
    Cleans, smooths, and refines the AI mesh in Blender with clean studio white clay finish:
    1. Voxel remesh / smooth surface artifacts.
    2. Decimate to clean polygon budget.
    3. Recompute smooth normals.
    4. Apply studio clay material.
    5. Render multi-angle verification images.
    """
    print(f"[3/4] Refining AI mesh in Blender: {raw_glb} -> {final_glb}")
    blender_script = f"""
import bpy
import bmesh
import math
import mathutils
from pathlib import Path

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Import raw AI mesh
bpy.ops.import_scene.gltf(filepath=r'{raw_glb}')
mesh_objs = [o for o in scene.objects if o.type == 'MESH']
if not mesh_objs:
    raise RuntimeError('No mesh found')

for obj in mesh_objs:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

bpy.ops.object.join()
ai_mesh = bpy.context.active_object
ai_mesh.name = 'AI_Generated_Model'

# Center & Normalize
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
ai_mesh.location = (0, 0, 0)

# Clean & Smooth: Smooth modifier
sm = ai_mesh.modifiers.new('Smooth', 'SMOOTH')
sm.factor = 0.5
sm.iterations = 4
bpy.ops.object.modifier_apply(modifier='Smooth')

# Decimate for clean game-ready / web-ready geometry
dec = ai_mesh.modifiers.new('Decimate', 'DECIMATE')
dec.ratio = 0.15 # 85% reduction of noisy marching cubes triangles
bpy.ops.object.modifier_apply(modifier='Decimate')

for p in ai_mesh.data.polygons:
    p.use_smooth = True

# Studio Clay Material
mat = bpy.data.materials.new(name='StudioClay')
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.90, 0.90, 0.92, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.35
    bsdf.inputs['Metallic'].default_value = 0.05

ai_mesh.data.materials.clear()
ai_mesh.data.materials.append(mat)

# Export clean GLB
bpy.ops.export_scene.gltf(filepath=r'{final_glb}', export_format='GLB')
print('Refined GLB successfully exported.')

# 4. Render 4 Studio Views
r_dir = Path(r'{renders_dir}')
r_dir.mkdir(parents=True, exist_ok=True)

coords = [ai_mesh.matrix_world @ mathutils.Vector(c) for c in ai_mesh.bound_box]
min_c = mathutils.Vector((min(c.x for c in coords), min(c.y for c in coords), min(c.z for c in coords)))
max_c = mathutils.Vector((max(c.x for c in coords), max(c.y for c in coords), max(c.z for c in coords)))
center = (min_c + max_c) * 0.5
max_dim = max(max_c.x - min_c.x, max_c.y - min_c.y, max_c.z - min_c.z)
dist = max_dim * 2.4

cam_data = bpy.data.cameras.new('Cam')
cam_data.lens = 70
cam_obj = bpy.data.objects.new('Cam', cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Lights
def add_light(name, energy, size, pos):
    ld = bpy.data.lights.new(name, 'AREA')
    ld.energy = energy
    ld.size = size
    lo = bpy.data.objects.new(name, ld)
    lo.location = pos
    scene.collection.objects.link(lo)
    d = center - lo.location
    lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return lo

add_light('Key', 25.0, 0.5, (center.x + dist*0.65, center.y - dist*0.75, center.z + dist*0.55))
add_light('Fill', 12.0, 0.6, (center.x - dist*0.75, center.y - dist*0.60, center.z + dist*0.30))
add_light('Rim', 18.0, 0.35, (center.x, center.y + dist*0.85, center.z + dist*0.75))

scene.world = bpy.data.worlds.new('World')
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get('Background')
if bg:
    bg.inputs['Color'].default_value = (0.01, 0.01, 0.012, 1.0)
    bg.inputs['Strength'].default_value = 1.0

scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.view_settings.view_transform = 'AgX'

def render_shot(name, offset):
    cam_obj.location = center + mathutils.Vector(offset)
    d = center - cam_obj.location
    cam_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(r_dir / f'{{name}}.png')
    bpy.ops.render.render(write_still=True)
    print(f'Rendered {{name}} -> {{scene.render.filepath}}')

render_shot('ai_model_front', (0, -dist, 0))
render_shot('ai_model_perspective', (dist*0.62, -dist*0.78, dist*0.40))
render_shot('ai_model_side', (dist, 0, 0))
render_shot('ai_model_top', (0, 0, dist))
"""
    script_path = Path("temp/blender_refine_ai.py")
    script_path.write_text(blender_script)

    import subprocess
    subprocess.run([
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "--background",
        "--python",
        str(script_path)
    ], check=True)
    print(f"[3/4] Blender mesh refinement & rendering completed!")

if __name__ == "__main__":
    input_img = "uploads/0ab155de-665b-46bd-afde-705c0a2c1070.png"
    isolated_img_path = "temp/ai_pipeline/main_object_isolated.png"
    final_glb = Path("outputs/ai_generated_airpods.glb")
    renders_dir = Path("temp/ai_pipeline/renders")

    # Step 1: Find and isolate main object
    isolated_pil = isolate_main_object(input_img, isolated_img_path)

    # Step 2: Run Hunyuan3D-2 AI
    raw_glb = run_hunyuan_ai_generation(isolated_pil, str(final_glb))

    # Step 3: Refine in Blender & Render
    refine_mesh_in_blender(raw_glb, final_glb, renders_dir)
    print(f"[4/4] COMPLETE AI PIPELINE FINISHED SUCCESSFULLY!")
