# 3D Product Reconstruction Backend (Coca-Cola Can)

Complete offline Python backend for reconstructing a realistic 3D aluminum Coca-Cola can from 6 reference photographs.

## Architecture Overview

```text
backend/
├── input/                  # 6 reference photographs
│   ├── front.jpg
│   ├── back.jpg
│   ├── left.jpg
│   ├── right.jpg
│   ├── top.jpg
│   └── bottom.jpg
│
├── checkpoints/            # Segmentation checkpoints (SAM 2.1 / U2-Net)
│
├── src/                    # Modular Python pipeline components
│   ├── segmentation.py     # Can silhouette & foreground segmentation
│   ├── can_geometry.py     # Model-First realistic 3D aluminum can geometry
│   ├── uv_mapping.py       # Continuous cylindrical & planar UV layouts
│   ├── camera_alignment.py # 6 canonical viewpoint alignment & projection
│   ├── texture_projection.py # Original photograph pixel extraction
│   ├── texture_blending.py # Multi-view blending with zero ghosting
│   ├── material.py         # PBR aluminum metallic shader setup
│   ├── blender_pipeline.py # Blender headless automation & GLB export
│   ├── validation.py       # 6-view renders & automated feature verification
│   └── main.py             # 14-step sequential orchestrator
│
└── output/
    ├── model/              # Raw model assets (can_raw.glb, can_raw.obj, can_mesh.blend, uv_layout.png, can_uv.blend)
    ├── texture/            # Master textures (texture_basecolor.png, texture_preview.png)
    ├── renders/            # 6 validation renders (front.png, back.png, left.png, right.png, top.png, bottom.png)
    └── final/              # Final self-contained models (coca_cola_can.glb, coca_cola_can.gltf, coca_cola_can.obj)
```

## Key Pipeline Principles

1. **Model-First Architecture**: 3D mesh geometry (body curvature, upper shoulder, lower transition, raised rim, recessed lid, pull tab, concave base) is constructed and UV unwrapped *before* texture projection.
2. **Text from Original Photos**: All branding ("Coca-Cola", "ORIGINAL TASTE", "330 ml", "Coke", Nutrition Information table, ingredients, barcodes) comes directly from original photograph pixels. No AI text generation or rewriting.
3. **Ghost-Free Multi-View Blending**: Front and Back cameras own their frontal angles completely to prevent perspective ghosting and double letters. Side cameras blend smoothly in lateral transition zones.
4. **Standalone Self-Contained GLB**: Textures are packed directly into the GLB binary buffer, instantly viewable in Blender or any standard glTF 2.0 / WebGL viewer.
5. **100% Offline on Apple Silicon Mac M4**: Uses MPS/CPU with fixed 2-thread Cycles/Eevee/Workbench headless rendering.

## Quick Start

### 1. Requirements

- Python 3.10+
- Blender 3.6+ / 4.x / 5.x installed at `/Applications/Blender.app`
- Dependencies installed via `pip install -r backend/requirements.txt`

### 2. Running Reconstruction Pipeline

```bash
python backend/src/main.py --input backend/input --output backend/output --res 2048
```
