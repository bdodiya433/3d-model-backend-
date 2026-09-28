"""
main.py - Master pipeline runner for the Generic 6-Photo 3D Product Reconstruction Backend.
Executes the strict Image-Driven, Model-First pipeline for ANY physical product/object:
  1. Load photographs
  2. Extract foreground object & alpha mask (ObjectSegmenter)
  3. Calibrate cameras & common 3D world coordinate system (CameraAlignment)
  4. Reconstruct image-driven 3D geometry (GeometryReconstructor: Visual Hull SDF + Marching Cubes)
  5. Validate 3D geometry against multi-view silhouettes (GeometryValidator)
  6. Generate universal, topology-agnostic UV layout (UniversalUVMapper)
  7. Extract cropped photographic views (TextureProjector)
  8. Visibility-aware 3D texture projection onto mesh (TextureBlender)
  9. Assemble PBR materials & export standalone GLB (BlenderPipeline)
  10. Render validation views & compute fidelity metrics (ModelValidator)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from src.segmentation import ObjectSegmenter
from src.camera_alignment import CameraAlignment
from src.geometry_reconstruction import GeometryReconstructor
from src.geometry_validation import GeometryValidator
from src.uv_mapping import UniversalUVMapper
from src.texture_projection import TextureProjector
from src.texture_blending import TextureBlender
from src.blender_pipeline import BlenderPipeline
from src.validation import ModelValidator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend.main")


def run_reconstruction(
    input_dir: Path,
    output_dir: Path,
    blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender",
    texture_resolution: int = 4096,
    product_name: str = "reconstructed_product",
    skip_texture: bool = True,
):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    model_dir = output_dir / "model"
    texture_dir = output_dir / "texture"
    renders_dir = output_dir / "renders"
    final_dir = output_dir / "final"

    for d in [model_dir, texture_dir, renders_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    logger.info("=================================================================")
    logger.info("STARTING GENERIC 6-PHOTO IMAGE-TO-3D RECONSTRUCTION PIPELINE")
    logger.info(f"Input Directory:  {input_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Blender Path:     {blender_bin}")
    logger.info(f"Resolution:       {texture_resolution}x{texture_resolution}")
    logger.info("=================================================================")

    # STEP 1: Load reference photographs
    logger.info("[STEP 1/10] Loading reference photographs...")
    input_files = list(input_dir.glob("*.*"))
    logger.info(f"Found {len(input_files)} source files in {input_dir}")

    # STEP 2: Extract foreground object & alpha mask
    logger.info("[STEP 2/10] Extracting foreground object from each photograph (ObjectSegmenter)...")
    segmenter = ObjectSegmenter()
    segmented_views = segmenter.segment_all(input_dir)
    if not segmented_views:
        raise RuntimeError("No input views could be segmented. Check input directory.")

    # STEP 3: Camera Calibration & Common World Frame
    logger.info("[STEP 3/10] Calibrating 6 camera poses & unified 3D coordinate system...")
    aligner = CameraAlignment()
    # Estimate rough initial extents from bounding boxes
    calibrated = aligner.calibrate_views(segmented_views)

    # STEP 4: Image-Driven 3D Geometry Reconstruction (Continuous SDF + Marching Cubes + MeshRefiner)
    logger.info("[STEP 4/10] Reconstructing 3D object geometry from photographs (IMAGE-DRIVEN)...")
    reconstructor = GeometryReconstructor(blender_bin=blender_bin)
    raw_models = reconstructor.reconstruct_from_views(
        segmented_views=segmented_views,
        output_dir=model_dir,
        calibrations=calibrated,
        grid_resolution=200,
        bounds_padding=1.06,
        target_faces=14000,
        refinement_iterations=8,
    )
    dim = raw_models["dimensions"]
    obj_diameter = max(dim[0], dim[1])
    obj_height = dim[2]
    logger.info(f"Reconstructed 3D dimensions: X={dim[0]:.3f}, Y={dim[1]:.3f}, Z={dim[2]:.3f}")

    # CRITICAL CAMERA REQUIREMENT: Preserve existing calibrated cameras consistently
    # Do NOT recalibrate cameras between reconstruction and validation.

    # STEP 5: Pre-Texture Geometry Silhouette Validation (Initial vs Refined)
    logger.info("[STEP 5/10] Validating reconstructed 3D mesh against input silhouettes...")
    geo_validator = GeometryValidator(blender_bin=blender_bin)

    # 5A. Validate INITIAL mesh (Marching Cubes raw visual hull)
    logger.info("Computing INITIAL mesh silhouette validation metrics...")
    initial_geo_report = geo_validator.validate_geometry(
        mesh_path=raw_models["initial_obj"],
        segmented_views=segmented_views,
        calibrations=calibrated,
        output_dir=model_dir / "initial_validation",
    )

    # 5B. Validate REFINED mesh (After multi-view refinement & QEM decimation)
    logger.info("Computing REFINED mesh silhouette validation metrics...")
    refined_geo_report = geo_validator.validate_geometry(
        mesh_path=raw_models["obj"],
        segmented_views=segmented_views,
        calibrations=calibrated,
        output_dir=model_dir / "refined_validation",
    )

    # Master comparison report
    refinement_metrics_file = output_dir / "refinement_metrics.json"
    comparison_summary = {
        "initial": initial_geo_report,
        "refined": refined_geo_report,
        "refinement_history": raw_models.get("refinement_report", {}),
    }
    refinement_metrics_file.write_text(json.dumps(comparison_summary, indent=2))
    logger.info(f"Saved master refinement metrics to {refinement_metrics_file}")

    logger.info(
        f"INITIAL vs REFINED Mean IoU: {initial_geo_report['mean_iou']:.4f} -> {refined_geo_report['mean_iou']:.4f} "
        f"(Delta: {(refined_geo_report['mean_iou'] - initial_geo_report['mean_iou']):+.4f})"
    )

    if skip_texture:
        logger.info("[STEP 6/6] Exporting clean, simple untextured 3D model (no colors/textures)...")
        clay_glb_path = final_dir / f"{product_name}.glb"
        clay_blend_path = final_dir / f"{product_name}.blend"
        clay_obj_path = final_dir / f"{product_name}.obj"

        clean_script = f'''
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=r"{raw_models['obj']}", up_axis="Z", forward_axis="Y")
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
obj.name = "{product_name}"

# Simple clean matte clay material (neutral off-white, no colors, no textures)
mat = bpy.data.materials.new(name="SimpleClayMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.88, 0.88, 0.88, 1.0)
bsdf.inputs["Roughness"].default_value = 0.50
bsdf.inputs["Metallic"].default_value = 0.0
obj.data.materials.clear()
obj.data.materials.append(mat)

for p in obj.data.polygons:
    p.use_smooth = True

bpy.ops.export_scene.gltf(filepath=r"{clay_glb_path}", export_format='GLB')
bpy.ops.wm.obj_export(filepath=r"{clay_obj_path}", up_axis="Z", forward_axis="Y")
bpy.ops.wm.save_as_mainfile(filepath=r"{clay_blend_path}")
print("SIMPLE_MODEL_EXPORT_SUCCESS")
'''
        temp_py = final_dir / "export_clean_simple.py"
        temp_py.write_text(clean_script)
        try:
            cmd = [blender_bin, "--background", "--python", str(temp_py)]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        finally:
            if temp_py.exists():
                temp_py.unlink()

        final_assets = {
            "glb": clay_glb_path,
            "blend": clay_blend_path,
            "obj": clay_obj_path,
        }

        # Component 11: Render 6 cardinal clay views and 5 intermediate turntable angles
        logger.info("Rendering 11 validation views (6 cardinal + 5 intermediate turntable angles in matte clay)...")
        renders = geo_validator.render_intermediate_views(
            mesh_path=final_assets["obj"],
            output_dir=renders_dir,
            resolution=512,
        )

        elapsed = time.time() - total_start
        logger.info("=================================================================")
        logger.info(f"REFINED 3D MODEL GENERATED SUCCESSFULLY IN {elapsed:.2f}s!")
        logger.info(f"Clean Untextured GLB: {final_assets['glb']}")
        logger.info("=================================================================")

        return {
            "models": raw_models,
            "initial_geometry_validation": initial_geo_report,
            "refined_geometry_validation": refined_geo_report,
            "refinement_metrics_file": refinement_metrics_file,
            "final": final_assets,
            "renders": renders,
            "elapsed_seconds": elapsed,
        }

    # STEP 6: Universal Topology-Agnostic UV Unwrapping
    logger.info("[STEP 6/10] Creating universal, low-fragmentation UV layout (UniversalUVMapper)...")
    uv_mapper = UniversalUVMapper(blender_bin=blender_bin)
    uv_assets = uv_mapper.create_uv_layout(
        mesh_blend_path=raw_models["blend"],
        output_dir=model_dir,
        resolution=2048,
        angle_limit=66.0,
        island_margin=0.003,
    )
    logger.info(f"Universal UV layout generated: {uv_assets['uv_layout']}")

    # STEP 7: Extract original cropped photographs
    logger.info("[STEP 7/10] Extracting cropped photographic views...")
    projector = TextureProjector()
    cropped_views = projector.extract_cropped_views(segmented_views)

    # STEP 8: Visibility-Aware 3D Texture Projection onto Reconstructed Mesh
    logger.info("[STEP 8/10] True 3D geometry texture projection (Winner-Takes-All, no ghosting)...")
    blender = TextureBlender(target_resolution=texture_resolution)
    texture_assets = blender.blend_textures(
        cropped_views=cropped_views,
        output_dir=texture_dir,
        camera_calibration=calibrated,
        mesh_obj_path=uv_assets["obj"],
        segmented_views=segmented_views,
    )
    logger.info(f"Master BaseColor texture generated: {texture_assets['basecolor']}")

    # STEP 9: Final PBR Material Assembly & Standalone GLB Export
    logger.info("[STEP 9/10] Assembling PBR material and exporting standalone GLB...")
    pipeline = BlenderPipeline(blender_bin=blender_bin)
    final_assets = pipeline.export_final_assets(
        can_uv_blend_path=uv_assets["blend"],
        texture_basecolor_path=texture_assets["basecolor"],
        output_dir=final_dir,
        texture_normal_path=texture_assets.get("normal"),
        texture_roughness_path=texture_assets.get("roughness"),
        texture_metallic_path=texture_assets.get("metallic"),
        product_name=product_name,
    )
    logger.info(f"Final 3D assets ready in: {final_dir}")
    logger.info(f"  GLB:  {final_assets['glb']} ({final_assets['glb'].stat().st_size:,} bytes)")

    # STEP 10: Validation Renders & Final Report
    logger.info("[STEP 10/10] Rendering 6 validation views and evaluating fidelity...")
    validator = ModelValidator(blender_bin=blender_bin)
    renders = validator.render_validation_views(
        final_glb_path=final_assets["glb"],
        output_dir=renders_dir,
        resolution=512,
    )
    validation_report = validator.validate_results(renders)

    elapsed = time.time() - total_start
    logger.info("=================================================================")
    logger.info(f"GENERIC PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s!")
    logger.info(f"Final Textured GLB: {final_assets['glb']}")
    logger.info("=================================================================")

    return {
        "models": raw_models,
        "geometry_validation": geo_report,
        "uv": uv_assets,
        "texture": texture_assets,
        "final": final_assets,
        "renders": renders,
        "validation_report": validation_report,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generic 6-Photo Image-to-3D Product Reconstruction Backend"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input directory containing 6 photos")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--res", "-r", type=int, default=4096, help="Texture resolution (default: 4096)")
    parser.add_argument(
        "--blender",
        type=str,
        default="/Applications/Blender.app/Contents/MacOS/Blender",
        help="Path to Blender binary",
    )
    parser.add_argument("--name", type=str, default="reconstructed_product", help="Product output filename")

    args = parser.parse_args()
    run_reconstruction(
        input_dir=args.input,
        output_dir=args.output,
        blender_bin=args.blender,
        texture_resolution=args.res,
        product_name=args.name,
    )


if __name__ == "__main__":
    main()
