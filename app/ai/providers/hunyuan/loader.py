import os
import gc
import logging
from pathlib import Path
import torch

from app.ai.providers.hunyuan.utils import setup_sys_path, get_torch_device, get_torch_dtype

# Inject third-party library folder into sys.path
setup_sys_path()

try:
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
except ImportError as e:
    logging.getLogger("app.ai.hunyuan.loader").error(
        f"Failed to import from third_party/Hunyuan3D-2. Check path configurations. Error: {e}"
    )
    raise e

logger = logging.getLogger("app.ai.hunyuan.loader")


class HunyuanModelLoader:
    """
    Handles memory resident lifetime of the Tencent Hunyuan3D-2 model.
    Loads checkpoints once and handles device mapping and memory cleanup.
    """

    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self.ckpt_path = weights_dir / "model.fp16.safetensors"
        self.config_path = weights_dir / "config.yaml"
        self.pipeline = None
        self.device = get_torch_device()
        self.dtype = get_torch_dtype(self.device)

    def load_model(self) -> Hunyuan3DDiTFlowMatchingPipeline:
        """
        Loads the Flow Matching pipeline into RAM/VRAM using single safetensors file.
        """
        if self.pipeline is not None:
            logger.info("Hunyuan3D-2 model is already loaded.")
            return self.pipeline

        logger.info(f"Loading Hunyuan3D-2 model on device={self.device}, dtype={self.dtype}...")
        
        if self.ckpt_path.exists() and self.config_path.exists():
            try:
                logger.info(f"Loading local weights checkpoint from {self.ckpt_path}...")
                self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                    ckpt_path=str(self.ckpt_path),
                    config_path=str(self.config_path),
                    device=self.device,
                    dtype=self.dtype,
                    use_safetensors=True
                )
                logger.info("Successfully loaded local Hunyuan3D-2 pipeline.")
                return self.pipeline
            except Exception as e:
                logger.warning(f"Failed loading local checkpoint ({e}). Trying HuggingFace pretrained weights...")

        # Fallback to loading pretrained Hugging Face checkpoint if local file missing
        try:
            logger.info("Loading Hunyuan3D-2 model from Hugging Face ('tencent/Hunyuan3D-2mini')...")
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mini",
                subfolder="hunyuan3d-dit-v2-mini-turbo",
                device=self.device,
                dtype=self.dtype,
                use_safetensors=True
            )
            logger.info("Successfully loaded Hunyuan3D-2 pipeline from Hugging Face.")
            return self.pipeline
        except Exception as ex:
            logger.error(f"Failed to load Hunyuan3D-2 model: {ex}")
            raise ex

    def unload_model(self) -> None:
        """
        Offloads model from GPU/MPS memory to CPU and garbage collects resources.
        """
        if self.pipeline is not None:
            logger.info("Unloading Hunyuan3D-2 pipeline...")
            try:
                self.pipeline.to("cpu")
            except Exception as ex:
                logger.warning(f"Failed to move pipeline to CPU during unload: {ex}")
            
            del self.pipeline
            self.pipeline = None
            
            # Request Python garbage collection
            gc.collect()
            
            # Clear device cache if supported
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Hunyuan3D-2 pipeline unloaded from active memory.")
