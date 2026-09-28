import os
import logging
import psutil
import torch
from typing import Dict, Any, List
from pathlib import Path

from app.config.settings import Settings
from app.ai.base.ai_model import AIModel
from app.ai.providers.hunyuan.provider import Hunyuan3DModel
from app.ai.providers.hunyuan.utils import get_torch_device

logger = logging.getLogger("app.ai.manager")


class ModelManager:
    """
    Orchestrates the lifecycle of active AI models in the system.
    Tracks loaded models, hardware mapping, memory footprints, and health diagnostics.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # Register hunyuan3d provider with its weights folder path
        self._providers: Dict[str, AIModel] = {
            "hunyuan3d": Hunyuan3DModel(settings.model_dir / "hunyuan3d")
        }
        self._loaded: Dict[str, bool] = {
            "hunyuan3d": False
        }

    @property
    def device(self) -> str:
        """
        Returns the resolved active acceleration device (mps, cuda, or cpu).
        """
        return get_torch_device()

    def get_provider(self, name: str) -> AIModel:
        """
        Returns the model provider instance.
        """
        if name not in self._providers:
            raise ValueError(f"AI Model provider '{name}' is not registered.")
        return self._providers[name]

    def load(self, name: str) -> None:
        """
        Loads the specified model provider into memory.
        """
        logger.info(f"ModelManager: requesting load for '{name}'...")
        provider = self.get_provider(name)
        provider.load()
        self._loaded[name] = True
        logger.info(f"ModelManager: '{name}' successfully loaded.")

    def unload(self, name: str) -> None:
        """
        Unloads the specified model provider to free system VRAM/RAM.
        """
        logger.info(f"ModelManager: requesting unload for '{name}'...")
        provider = self.get_provider(name)
        provider.unload()
        self._loaded[name] = False
        logger.info(f"ModelManager: '{name}' successfully unloaded.")

    def is_loaded(self, name: str) -> bool:
        """
        Checks if the provider is currently loaded and resident.
        """
        if name not in self._providers:
            return False
        return self._loaded.get(name, False) and self._providers[name].is_loaded()

    def get_memory_usage(self) -> str:
        """
        Retrieves VRAM and host RAM allocation metrics formatted as a readable string.
        """
        ram_mb = 0.0
        vram_mb = 0.0

        # Measure host process memory
        try:
            process = psutil.Process(os.getpid())
            ram_mb = process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not read process memory: {e}")

        # Measure PyTorch CUDA allocation
        if torch.cuda.is_available():
            try:
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass

        return f"Host RAM: {ram_mb:.1f} MB | VRAM: {vram_mb:.1f} MB"

    def get_status(self, name: str) -> Dict[str, Any]:
        """
        Returns diagnostic status info for a single provider.
        """
        loaded = self.is_loaded(name)
        return {
            "provider": name,
            "loaded": loaded,
            "device": self.device,
            "memory": self.get_memory_usage(),
            "status": "ready" if loaded else "unloaded"
        }
