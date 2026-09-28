import logging
import time
from pathlib import Path
from typing import Any
from app.services.ai.base.model import AIModel

logger = logging.getLogger("app.ai.providers.triposr")


class TripoSRProvider(AIModel):
    """
    Model provider implementation for TripoSR (fast single-image reconstruction).
    """
    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_path = model_dir / "triposr"
        self.device = device
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            logger.warning("TripoSR model is already loaded.")
            return

        logger.info(f"Loading TripoSR weights from {self.model_path} into memory on device: {self.device}...")
        start_time = time.perf_counter()
        
        # Ensure directories exist
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Simulated weights loading time
        time.sleep(0.2)
        self._loaded = True
        
        duration = time.perf_counter() - start_time
        logger.info(f"TripoSR model initialized on {self.device} in {duration:.4f}s.")

    def unload(self) -> None:
        if not self._loaded:
            logger.warning("TripoSR model is not loaded; skip unloading.")
            return

        logger.info("Unloading TripoSR weights from RAM/VRAM...")
        self._loaded = False
        logger.info("TripoSR weights successfully cleared from memory.")

    def generate(self, input_data: Any, **kwargs: Any) -> Any:
        if not self._loaded:
            raise RuntimeError("Cannot execute inference. TripoSR model is not loaded.")
        logger.info("Running simulated TripoSR 3D generation...")
        return {
            "model": "triposr",
            "device": self.device,
            "output_format": "glb",
            "mesh_data": "simulated_triposr_mesh_data"
        }

    def is_loaded(self) -> bool:
        return self._loaded

    def health(self) -> bool:
        return self.model_path.exists()
