import logging
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union, Any
from PIL import Image
import trimesh

logger = logging.getLogger("app.services.texture.baker")


class MultiViewTextureBaker:
    """
    Service for projecting multi-view photographs (front, back, left, right, top, bottom)
    onto 3D reconstructed meshes, baking them into seamless UV PBR texture maps,
    and exporting both standalone textured GLB assets and native Blender .blend project files.
    """

    def __init__(self, blender_path: Optional[str] = None):
        self.blender_path = blender_path or self._find_blender_executable()
        self.blender_bake_script = Path(__file__).parent / "blender_bake.py"

    def _find_blender_executable(self) -> str:
        macos_path = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        if macos_path.exists():
            return str(macos_path)
        env_path = os.getenv("BLENDER_PATH")
        if env_path and Path(env_path).exists():
            return env_path
        return "blender"

    def bake_texture(
        self,
        mesh_glb_path: Union[str, Path],
        view_images: Dict[str, Union[str, Path, Image.Image]],
        output_glb_path: Union[str, Path],
        output_blend_path: Optional[Union[str, Path]] = None,
        view_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        texture_resolution: int = 2048,
        debug_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Executes high-fidelity texel-level multi-view texture projection with C++ occlusion testing,
        perceptual view scoring, confidence-weighted blending, and color-space safe baking.
        
        :param mesh_glb_path: Path to the raw untextured reconstructed GLB mesh.
        :param view_images: Dict mapping view keys ('front', 'back', 'left', 'right', 'top', 'bottom')
                            to image paths or PIL Image objects.
        :param output_glb_path: Path where the textured GLB will be saved.
        :param output_blend_path: Path where the native .blend file will be saved.
        :param view_metadata: Dict mapping view keys to transform metadata from preprocessor.
        :param texture_resolution: Texture resolution (default 2048x2048).
        :param debug_dir: Optional directory where intermediate debug images are saved.
        """
        mesh_glb_path = Path(mesh_glb_path).resolve()
        output_glb_path = Path(output_glb_path).resolve()
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)

        if output_blend_path:
            output_blend_path = Path(output_blend_path).resolve()
            output_blend_path.parent.mkdir(parents=True, exist_ok=True)

        if not mesh_glb_path.exists():
            raise FileNotFoundError(f"Input GLB mesh not found: {mesh_glb_path}")

        logger.info(f"Starting multi-view texture projection on {mesh_glb_path.name}...")
        logger.info(f"Available view images: {list(view_images.keys())}")

        # Prepare temporary working directory for views manifest and images
        temp_dir = Path(tempfile.mkdtemp(prefix="tex_bake_"))
        views_manifest: Dict[str, Any] = {}

        try:
            for view_name, img_data in view_images.items():
                if img_data is None:
                    continue

                if isinstance(img_data, (str, Path)):
                    src_path = Path(img_data)
                    if not src_path.exists():
                        continue
                    pil_img = Image.open(src_path).convert("RGBA")
                elif isinstance(img_data, Image.Image):
                    pil_img = img_data.convert("RGBA")
                else:
                    continue

                view_img_path = temp_dir / f"{view_name}.png"
                pil_img.save(view_img_path, "PNG")

                meta = (view_metadata or {}).get(view_name, {})
                views_manifest[view_name] = {
                    "image_path": str(view_img_path),
                    "metadata": meta,
                }

            if not views_manifest:
                raise ValueError("No valid view images available for texture projection.")

            manifest_path = temp_dir / "views.json"
            with open(manifest_path, "w") as f:
                json.dump(views_manifest, f, indent=2)

            # Execute Blender headless baking script
            cmd = [
                self.blender_path,
                "--background",
                "--python",
                str(self.blender_bake_script),
                "--",
                str(mesh_glb_path),
                str(output_glb_path),
                str(output_blend_path) if output_blend_path else "None",
                str(texture_resolution),
                str(manifest_path),
                str(debug_dir) if debug_dir else "None"
            ]

            logger.info(f"Executing Blender texture projection CLI: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)

            if result.returncode != 0:
                logger.error(
                    f"Blender baking failed (exit code {result.returncode}):\n"
                    f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )
                raise RuntimeError(f"Blender bake process failed with return code {result.returncode}")

            logger.info(f"Successfully baked photographic texture to {output_glb_path}")

            # Inspect validation report if generated
            val_report_path = output_glb_path.parent / f"{output_glb_path.stem}_color_validation.json"
            if val_report_path.exists():
                try:
                    with open(val_report_path, "r") as f:
                        val_data = json.load(f)
                    logger.info(f"Texture Validation Report: {val_data}")
                except Exception:
                    pass

            return output_glb_path

        finally:
            # Clean up temporary images directory
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
