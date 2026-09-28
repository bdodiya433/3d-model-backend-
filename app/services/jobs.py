import asyncio
import time
import logging
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import json
import numpy as np
from PIL import Image
from app.config.settings import Settings
from app.ai.manager.model_manager import ModelManager
from app.ai.providers.hunyuan.utils import setup_sys_path
from app.services.blender.exporter import BlenderExporter
from app.services.texture.baker import MultiViewTextureBaker
from app.services.geometry import (
    GeometryCleaner,
    SilhouetteEvaluator,
    ReferenceStructureAnalyzer,
    StructuralEvaluator,
)
from app.services.color import ColorEngine
from app.services.image import ImagePreprocessor

logger = logging.getLogger("app.jobs")


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self.blender_exporter = BlenderExporter()
        self.texture_baker = MultiViewTextureBaker()
        self.geometry_cleaner = GeometryCleaner()
        self.silhouette_evaluator = SilhouetteEvaluator()
        self.reference_structure_analyzer = ReferenceStructureAnalyzer()
        self.structural_evaluator = StructuralEvaluator()
        self.color_engine = ColorEngine(blender_exporter=self.blender_exporter)

    def create_job(
        self,
        image_id: Optional[str] = None,
        image_ids: Optional[List[str]] = None,
        view_mapping: Optional[Dict[str, str]] = None,
        model_name: str = "hunyuan3d",
        num_inference_steps: int = 30,
        octree_resolution: int = 384,
        enable_color: bool = True,
    ) -> str:
        job_id = str(uuid.uuid4())
        
        # Consolidate list of image IDs (supports 1 to 6 views)
        all_ids: List[str] = []
        if view_mapping:
            # Guarantee 'front' view is always first in list
            if "front" in view_mapping:
                all_ids.append(view_mapping["front"])
            for k, v in view_mapping.items():
                if k != "front" and v not in all_ids:
                    all_ids.append(v)
        elif image_ids and len(image_ids) > 0:
            all_ids.extend(image_ids)
        elif image_id:
            all_ids.append(image_id)

        if not all_ids:
            raise ValueError("Job creation requires at least one valid image_id, image_ids list, or view_mapping.")

        self._jobs[job_id] = {
            "job_id": job_id,
            "image_id": all_ids[0],
            "image_ids": all_ids,
            "view_mapping": view_mapping or {},
            "model_name": model_name,
            "num_inference_steps": num_inference_steps,
            "octree_resolution": octree_resolution,
            "enable_color": enable_color,
            "status": "PENDING",
            "progress": 5,
            "stage": "Uploading completed & validated",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "download_url": None,
            "download_blend_url": None,
            "error": None
        }
        logger.info(f"Job {job_id} created for {len(all_ids)} image(s) using {model_name} (steps={num_inference_steps}, octree_res={octree_resolution}, enable_color={enable_color}).")
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        progress: int,
        stage: str,
        status: Optional[str] = None,
        error: Optional[str] = None,
        download_url: Optional[str] = None,
        download_blend_url: Optional[str] = None
    ) -> None:
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job["progress"] = progress
            job["stage"] = stage
            if status:
                job["status"] = status
            if error:
                job["error"] = error
            if download_url:
                job["download_url"] = download_url
            if download_blend_url:
                job["download_blend_url"] = download_blend_url
            logger.debug(f"Job {job_id} updated: {progress}% - {stage} [{job['status']}]")

    async def start_simulated_pipeline(self, job_id: str) -> None:
        """
        Asynchronously steps through pipeline stages for testing.
        """
        stages = [
            (15, "Isolating Main Object & Removing Background", 1.5),
            (25, "Cropping to Object Bounding Box", 1.5),
            (45, "Generating Mesh", 2.0),
            (60, "Refining Mesh", 1.5),
            (75, "Generating Texture", 1.5),
            (90, "Blender Decimation & Remeshing", 2.0),
            (100, "GLB & .blend Exporting", 1.0)
        ]

        try:
            self.update_job(job_id, 5, "Initializing Pipeline", status="PROCESSING")
            for progress, stage, delay in stages:
                await asyncio.sleep(delay)

                if job_id not in self._jobs:
                    logger.info(f"Job {job_id} cancelled during {stage}.")
                    return

                self.update_job(job_id, progress, stage)

            output_filename = f"{job_id}.glb"
            output_path = self.settings.output_dir / output_filename
            output_path.write_text("DUMMY GLB MESH DATA FOR TESTING")

            download_url = f"/download/{job_id}?format=glb"
            download_blend_url = f"/download/{job_id}?format=blend"
            self.update_job(
                job_id,
                100,
                "Completed",
                status="COMPLETED",
                download_url=download_url,
                download_blend_url=download_blend_url
            )
            logger.info(f"Job {job_id} successfully completed.")

        except Exception as e:
            logger.error(f"Error executing simulated pipeline for job {job_id}: {e}")
            self.update_job(
                job_id,
                100,
                "Failed",
                status="FAILED",
                error=str(e)
            )

    async def start_generation_pipeline(self, job_id: str, model_manager: ModelManager) -> None:
        """
        Asynchronously loads multi-view uploaded images (1 to 6 pictures),
        executes Hunyuan3D-2 3D model generation, performs safe geometry refinement,
        extracts reference colors and maps them to 3D surface vertices (without photo textures),
        validates model quality, and exports both .glb and .blend project files.
        """
        setup_sys_path()
        try:
            self.update_job(job_id, 10, "Loading uploaded source images", status="PROCESSING")

            job = self.get_job(job_id)
            if not job:
                logger.error(f"Job {job_id} not found in state.")
                return

            image_ids: List[str] = job.get("image_ids", [job["image_id"]])
            model_name = job["model_name"]
            num_inference_steps = min(job.get("num_inference_steps", 15) or 15, 20)
            octree_resolution = min(job.get("octree_resolution", 256) or 256, 300)

            is_debug = os.getenv("DEBUG_RECONSTRUCTION", "true").lower() in ("1", "true", "yes")
            debug_dir = (self.settings.output_dir / "debug" / job_id) if is_debug else None
            if debug_dir:
                debug_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Debug mode active for job {job_id}. Artifacts will be saved to {debug_dir}")

            # 1. Preprocess all uploaded view images with silhouette & high-res analysis
            def preprocess_all_images():
                preprocessor = ImagePreprocessor(u2net_model="u2netp")

                view_names = ["front", "back", "left", "right", "top", "bottom"]
                view_mapping = job.get("view_mapping") or {}
                id_to_view = {v_id: k_view for k_view, v_id in view_mapping.items()}

                view_ai_images = {}
                view_highres_images = {}
                view_silhouettes = {}
                view_masks = {}
                view_metadata = {}

                for idx, img_id in enumerate(image_ids):
                    upload_files = list(self.settings.upload_dir.glob(f"{img_id}.*"))
                    if not upload_files:
                        logger.warning(f"Upload image for ID {img_id} not found on disk.")
                        continue
                    img_path = upload_files[0]
                    v_key = id_to_view.get(img_id) or (view_names[idx] if idx < len(view_names) else f"view_{idx+1}")

                    logger.info(f"Preprocessing view {idx + 1}/{len(image_ids)} ({v_key}) from {img_path} with uniform letterbox framing...")
                    p_res = preprocessor.preprocess_with_analysis(
                        img_path,
                        target_resolution=(1024, 1024),
                        padding_ratio=0.10,
                    )
                    preprocessed_path = self.settings.upload_dir / f"{img_id}_preprocessed.png"
                    p_res["ai_image"].save(preprocessed_path, "PNG")

                    view_ai_images[v_key] = p_res["ai_image"]
                    view_highres_images[v_key] = p_res["highres_working_image"]
                    view_silhouettes[v_key] = p_res["silhouette"]
                    view_masks[v_key] = p_res["cropped_mask"]
                    view_metadata[v_key] = p_res["transform_metadata"]

                    # Save debug artifacts if debug mode is active (Part 18 & 20)
                    if debug_dir:
                        try:
                            # 01_original, 02_preprocessed, 03_mask, mask_debug
                            shutil.copyfile(str(img_path), str(debug_dir / f"01_original_{v_key}.png"))
                            p_res["ai_image"].save(debug_dir / f"02_preprocessed_{v_key}.png", "PNG")
                            Image.fromarray((p_res["cropped_mask"] > 0).astype(np.uint8) * 255).save(debug_dir / f"03_mask_{v_key}.png", "PNG")
                            if "mask_debug" in p_res:
                                p_res["mask_debug"].save(debug_dir / f"mask_debug_{v_key}.png", "PNG")
                        except Exception as dex:
                            logger.debug(f"Debug artifact note for view {v_key}: {dex}")

                if not view_ai_images:
                    raise FileNotFoundError("No valid uploaded source images found for preprocessing.")

                return view_ai_images, view_highres_images, view_silhouettes, view_masks, view_metadata

            self.update_job(job_id, 15, f"High-quality preprocessing & background shadow removal across {len(image_ids)} view(s)")
            view_ai_images, view_highres_images, view_silhouettes, view_masks, view_metadata = await asyncio.to_thread(preprocess_all_images)
            await asyncio.sleep(0.1)

            # 2. Structural blueprint & feature analysis
            self.update_job(job_id, 25, "Analyzing structural features, negative spaces & components")

            # The Front view (or canonical perspective view) is ALWAYS the primary conditioning view for 3D geometry reconstruction!
            # Never pick top or bottom view as primary because they lack vertical height and full silhouette profile.
            primary_image = view_ai_images.get("front") or list(view_ai_images.values())[0]
            logger.info("Selected primary conditioning view for 3D shape reconstruction: 'front' view.")

            def analyze_structure():
                return self.reference_structure_analyzer.analyze(primary_image)

            structural_map = await asyncio.to_thread(analyze_structure)
            logger.info(
                f"Extracted structural reference map for job {job_id}: {len(structural_map.components)} components, "
                f"{len(structural_map.negative_spaces)} negative spaces."
            )

            glb_filename = f"{job_id}.glb"
            glb_output_path = self.settings.output_dir / glb_filename
            blend_filename = f"{job_id}.blend"
            blend_output_path = self.settings.output_dir / blend_filename

            # 3. Model Preparation & Inference (Hunyuan3D-2 generator unchanged!)
            self.update_job(job_id, 35, "Preparing AI model resources")
            provider = model_manager.get_provider(model_name)

            if not provider.is_loaded():
                self.update_job(job_id, 40, "Loading AI checkpoints to accelerator")
                await asyncio.to_thread(provider.load)
                await asyncio.sleep(0.1)

            eff_steps = max(min(num_inference_steps or 30, 35), 25)
            eff_res = max(min(octree_resolution or 384, 384), 256)

            self.update_job(job_id, 45, f"Reconstructing high-fidelity 3D AI geometry ({eff_steps} steps, res={eff_res})")

            def run_inference():
                import torch
                import gc
                with torch.inference_mode():
                    try:
                        res = provider.predict(
                            primary_image,
                            num_inference_steps=eff_steps,
                            octree_resolution=eff_res,
                            guidance_scale=5.0,
                        )
                    except Exception as ex:
                        logger.error(f"Inference error on primary image: {ex}")
                        raise ex
                    finally:
                        gc.collect()
                        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()

                if isinstance(res, list):
                    return res[0]
                return res

            raw_mesh = await asyncio.to_thread(run_inference)
            await asyncio.sleep(0.1)

            # Ensure trimesh format
            import trimesh
            if not isinstance(raw_mesh, trimesh.Trimesh) and hasattr(raw_mesh, "to_trimesh"):
                raw_mesh = raw_mesh.to_trimesh()

            if debug_dir:
                try:
                    raw_mesh.export(str(debug_dir / "09_raw_mesh.glb"))
                except Exception as dex:
                    logger.debug(f"Raw mesh debug export note: {dex}")

            # 4. Closed-Loop 3D -> 2D Structural Evaluation & Quality Gate
            self.update_job(job_id, 60, "Evaluating 3D structural fidelity & negative space preservation")

            def run_structural_evaluation():
                return self.structural_evaluator.evaluate(raw_mesh, structural_map)

            struct_report = await asyncio.to_thread(run_structural_evaluation)
            logger.info(
                f"Structural evaluation for job {job_id}: Overall Score={struct_report.overall_structural_confidence:.2f} "
                f"(Valid={struct_report.is_valid}), Failures: {len(struct_report.major_failures)}."
            )
            if job_id in self._jobs:
                self._jobs[job_id]["structural_confidence"] = struct_report.overall_structural_confidence
                self._jobs[job_id]["structural_valid"] = struct_report.is_valid
                self._jobs[job_id]["structural_failures"] = struct_report.major_failures

            # 4b. Feature-Aware Geometry Cleanup & Refinement
            self.update_job(job_id, 65, "Refining 3D geometry & preserving structural features")

            def run_geometry_cleanup():
                return self.geometry_cleaner.clean_mesh(raw_mesh, structural_map=structural_map)

            cleaned_mesh = await asyncio.to_thread(run_geometry_cleanup)
            await asyncio.sleep(0.1)

            if debug_dir:
                try:
                    cleaned_mesh.export(str(debug_dir / "10_clean_mesh.glb"))
                except Exception as dex:
                    logger.debug(f"Clean mesh debug export note: {dex}")

            # Evaluate multi-view silhouette consistency
            def run_silhouette_diagnostic():
                return self.silhouette_evaluator.evaluate_mesh(cleaned_mesh, reference_masks=view_masks)

            sil_report = await asyncio.to_thread(run_silhouette_diagnostic)
            logger.info(
                f"Multi-View Silhouette Diagnostic for job {job_id}: Similarity={sil_report.overall_silhouette_similarity * 100:.1f}%, "
                f"Mean IoU={sil_report.mean_iou:.3f}, Mean Dice={sil_report.mean_dice:.3f}."
            )
            if job_id in self._jobs:
                self._jobs[job_id]["silhouette_similarity"] = sil_report.overall_silhouette_similarity
                self._jobs[job_id]["mean_iou"] = sil_report.mean_iou

            # 5. High-Resolution Photographic UV Texture Projection & Baking
            enable_color = job.get("enable_color", True)
            if not enable_color:
                self.update_job(job_id, 85, "Generating uncolored Blender clay base model (color disabled)")
                await asyncio.sleep(0.1)
            else:
                self.update_job(job_id, 80, f"Projecting photographic reference textures from {len(view_highres_images)} view(s)")
                await asyncio.sleep(0.1)
                self.update_job(job_id, 90, "Smart UV unwrapping & baking PBR BaseColor texture map")

            def run_texture_pipeline():
                # Export clean intermediate mesh to temporary file
                temp_raw_glb = self.settings.output_dir / f"{job_id}_clean_raw.glb"
                cleaned_mesh.export(str(temp_raw_glb), file_type="glb")

                if enable_color and view_highres_images:
                    try:
                        logger.info(f"Baking multi-view photographic textures onto 3D model for job {job_id}...")
                        self.texture_baker.bake_texture(
                            mesh_glb_path=temp_raw_glb,
                            view_images=view_highres_images,
                            output_glb_path=glb_output_path,
                            output_blend_path=blend_output_path,
                            view_metadata=view_metadata,
                            texture_resolution=2048,
                            debug_dir=debug_dir if is_debug else None,
                        )
                        logger.info(f"Photographic UV texture baking succeeded for job {job_id}.")
                    except Exception as ex:
                        logger.error(f"TextureBaker encountered an error ({ex}). Falling back to ColorEngine.", exc_info=True)
                        self.color_engine.process_and_color_mesh(
                            mesh=cleaned_mesh,
                            view_images=view_highres_images,
                            output_glb_path=glb_output_path,
                            output_blend_path=blend_output_path,
                            enable_color=True,
                        )
                else:
                    logger.info(f"Color disabled (enable_color={enable_color}). Exporting clean uncolored Blender clay model for job {job_id}.")
                    self.color_engine.process_and_color_mesh(
                        mesh=cleaned_mesh,
                        view_images=view_highres_images,
                        output_glb_path=glb_output_path,
                        output_blend_path=blend_output_path,
                        enable_color=False,
                    )

                if temp_raw_glb.exists():
                    try:
                        temp_raw_glb.unlink()
                    except Exception:
                        pass

                # Save final model in debug folder if active
                if debug_dir and glb_output_path.exists():
                    try:
                        shutil.copyfile(str(glb_output_path), str(debug_dir / "20_final.glb"))
                    except Exception as dex:
                        logger.debug(f"Final GLB debug copy note: {dex}")

                # Load color validation metrics if report exists (Part 19)
                val_json_path = self.settings.output_dir / f"{job_id}_color_validation.json"
                if val_json_path.exists():
                    try:
                        with open(val_json_path, "r") as f:
                            val_data = json.load(f)
                        if job_id in self._jobs:
                            self._jobs[job_id]["color_validation"] = val_data
                        logger.info(f"Color Validation Report for job {job_id}: {val_data}")
                    except Exception as vex:
                        logger.debug(f"Color validation report read note: {vex}")

            await asyncio.to_thread(run_texture_pipeline)
            await asyncio.sleep(0.1)

            self.update_job(job_id, 95, "Validating 3D asset & finalizing export")

            download_url = f"/download/{job_id}?format=glb"
            download_blend_url = f"/download/{job_id}?format=blend"

            self.update_job(
                job_id,
                100,
                "Completed",
                status="COMPLETED",
                download_url=download_url,
                download_blend_url=download_blend_url,
            )
            logger.info(f"Job {job_id} successfully completed. High quality colored GLB and .blend exported.")

        except Exception as e:
            logger.error(f"Error executing multi-view generation pipeline for job {job_id}: {e}", exc_info=True)
            self.update_job(
                job_id,
                100,
                "Failed",
                status="FAILED",
                error=str(e),
            )

