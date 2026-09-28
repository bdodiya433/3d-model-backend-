import sys
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
import cv2

# Load Hunyuan mesh
mesh = trimesh.load('scratch/hunyuan_turtle_test.glb')
if isinstance(mesh, trimesh.Scene):
    mesh = list(mesh.geometry.values())[0]

print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Clean view images mapping from uploads:
# ad51a8b8-cf94-48cb-ac1b-da4ff65ee565.png has "front.png"
# 7d31392a-a854-4985-85af-73d084c8a911.png is bottom (belly)
# d592711d-e0e5-4fef-8765-5af6ace0f90f.png is top (shell)
# 36649da7-54a5-4b90-9510-baff63230dbb.png has "left.png"
# 2b18c6c9-8f61-48b5-a9dc-bb28da8f08dd.png is right
# e254e523-2672-4d36-84c0-bc4f15c20079.png has "back.png"

views = {
    'top': 'uploads/d592711d-e0e5-4fef-8765-5af6ace0f90f_preprocessed.png',
    'bottom': 'uploads/7d31392a-a854-4985-85af-73d084c8a911_preprocessed.png',
    'front': 'uploads/ad51a8b8-cf94-48cb-ac1b-da4ff65ee565_preprocessed.png',
    'back': 'uploads/e254e523-2672-4d36-84c0-bc4f15c20079_preprocessed.png',
    'left': 'uploads/36649da7-54a5-4b90-9510-baff63230dbb_preprocessed.png',
    'right': 'uploads/2b18c6c9-8f61-48b5-a9dc-bb28da8f08dd_preprocessed.png'
}

loaded_images = {}
for name, p in views.items():
    img = Image.open(p).convert("RGBA")
    arr = np.array(img)
    # Strip any text at the bottom 15% (e.g. "front.png", "left.png", "back.png")
    h, w = arr.shape[:2]
    # Check bottom 15%
    alpha = arr[..., 3]
    # Crop to active foreground
    ys, xs = np.where(alpha > 30)
    if len(ys) > 0:
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        # If there's text at bottom, isolate main component
        bin_mask = (alpha > 30).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask)
        if num_labels > 2:
            main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            # Keep components except tiny bottom text
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_TOP] > h * 0.85 and stats[lbl, cv2.CC_STAT_AREA] < stats[main_label, cv2.CC_STAT_AREA] * 0.05:
                    arr[labels == lbl] = [0, 0, 0, 0]
        
        # Recompute bounding box of clean object
        ys_c, xs_c = np.where(arr[..., 3] > 30)
        if len(ys_c) > 0:
            arr_cropped = arr[ys_c.min():ys_c.max()+1, xs_c.min():xs_c.max()+1]
            loaded_images[name] = arr_cropped
            print(f"Loaded {name}: {arr_cropped.shape}")
        else:
            loaded_images[name] = arr
    else:
        loaded_images[name] = arr

# Vertices & Normals
coords = mesh.vertices.copy()
normals = mesh.vertex_normals.copy()
n_verts = len(coords)

# Normalize coords
b_min, b_max = coords.min(axis=0), coords.max(axis=0)
b_dim = np.maximum(b_max - b_min, 1e-6)

# Normalised object coordinates in [0, 1]
# X: Left (-X, u=0) to Right (+X, u=1)
# Y: Back (-Y, v=0) to Front (+Y, v=1)
# Z: Bottom (-Z, w=0) to Top (+Z, w=1)
nx = normals[:, 0]
ny = normals[:, 1]
nz = normals[:, 2]

vx = (coords[:, 0] - b_min[0]) / b_dim[0]
vy = (coords[:, 1] - b_min[1]) / b_dim[1]
vz = (coords[:, 2] - b_min[2]) / b_dim[2]

def sample_arr(img_arr, u_arr, v_arr):
    # u in [0, 1] -> x in [0, w-1]
    # v in [0, 1] -> y in [0, h-1] (note v=0 is bottom in standard, so image y = (1 - v) * (h-1))
    h, w = img_arr.shape[:2]
    px = np.clip(np.round(u_arr * (w - 1)), 0, w - 1).astype(int)
    py = np.clip(np.round((1.0 - v_arr) * (h - 1)), 0, h - 1).astype(int)
    return img_arr[py, px]

# Scores: 0: top, 1: bottom, 2: front, 3: back, 4: right, 5: left
scores = np.zeros((n_verts, 6), dtype=np.float32)
colors = np.zeros((n_verts, 6, 4), dtype=np.float32)

# 1. TOP (+Z, nz > 0): looking from +Z down onto turtle
# Horizontal axis is X (left to right), Vertical axis is Y (tail to head)
if 'top' in loaded_images:
    col_t = sample_arr(loaded_images['top'], vx, vy)
    mask_t = (nz > 0.15) & (col_t[..., 3] > 30)
    scores[mask_t, 0] = (nz[mask_t] ** 1.5) * 1.5
    colors[:, 0] = col_t

# 2. BOTTOM (-Z, -nz > 0): looking from -Z up onto turtle belly
# Horizontal axis is X (left to right), Vertical axis is Y (tail to head)
if 'bottom' in loaded_images:
    col_b = sample_arr(loaded_images['bottom'], vx, vy)
    mask_b = (-nz > 0.15) & (col_b[..., 3] > 30)
    scores[mask_b, 1] = ((-nz[mask_b]) ** 1.5) * 1.5
    colors[:, 1] = col_b

# 3. FRONT (+Y, ny > 0): looking from front towards head
# Horizontal axis is X, Vertical axis is Z (bottom to top)
if 'front' in loaded_images:
    col_f = sample_arr(loaded_images['front'], vx, vz)
    mask_f = (ny > 0.20) & (col_f[..., 3] > 30)
    scores[mask_f, 2] = (ny[mask_f] ** 2.0)
    colors[:, 2] = col_f

# 4. BACK (-Y, -ny > 0): looking from back towards tail
if 'back' in loaded_images:
    col_bk = sample_arr(loaded_images['back'], 1.0 - vx, vz)
    mask_bk = (-ny > 0.20) & (col_bk[..., 3] > 30)
    scores[mask_bk, 3] = ((-ny[mask_bk]) ** 2.0)
    colors[:, 3] = col_bk

# 5. RIGHT (+X, nx > 0): looking from right flipper side
if 'right' in loaded_images:
    col_r = sample_arr(loaded_images['right'], 1.0 - vy, vz)
    mask_r = (nx > 0.20) & (col_r[..., 3] > 30)
    scores[mask_r, 4] = (nx[mask_r] ** 2.0)
    colors[:, 4] = col_r

# 6. LEFT (-X, -nx > 0): looking from left flipper side
if 'left' in loaded_images:
    col_l = sample_arr(loaded_images['left'], vy, vz)
    mask_l = (-nx > 0.20) & (col_l[..., 3] > 30)
    scores[mask_l, 5] = ((-nx[mask_l]) ** 2.0)
    colors[:, 5] = col_l

best_idx = np.argmax(scores, axis=1)
max_s = np.max(scores, axis=1)

final_colors = np.zeros((n_verts, 4), dtype=np.float32)
for i in range(6):
    m_i = (best_idx == i) & (max_s > 1e-4)
    final_colors[m_i] = colors[m_i, i]

# Fallback for unassigned vertices (median of top shell)
fallback = np.median(loaded_images['top'][loaded_images['top'][..., 3] > 30], axis=0)
final_colors[max_s <= 1e-4] = fallback
final_colors[:, 3] = 255.0

vcols = np.clip(final_colors, 0, 255).astype(np.uint8)
mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, vertex_colors=vcols)
mesh.export('scratch/hunyuan_turtle_colored.glb')
print("Successfully saved scratch/hunyuan_turtle_colored.glb with vertex colors!")
