import logging
import platform
import time
from pathlib import Path
from typing import Dict, Any, Optional
import psutil

from app.config.settings import Settings
from app.services.ai.base.model import AIModel
from app.services.ai.providers.hunyuan.provider import HunyuanProvider
from app.services.ai.providers.triposr.provider import TripoSRProvider
from app.services.ai.providers.instantmesh.provider import InstantMeshProvider

logger = logging.getLogger("app.ai.manager")


class ModelManager:
    """
    ModelManager orchestrates the lifecycle (loading, unloading, caching)
    and memory footprints of enabled AI model providers.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._detect_device()
        self._models: Dict[str, AIModel] = {}
        self._load_durations: Dict[str, float] = {}
        logger.info(f"ModelManager initialized with device: {self.device}")

    def _detect_device(self) -> str:
        """
        Auto-detects active hardware accelerator:
        1. PyTorch CUDA
        2. PyTorch MPS
        3. Platform-based Apple Silicon (fallback)
        4. CPU (fallback)
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        # Platform check fallback for macOS ARM64
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return "mps"
        return "cpu"

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Captures memory metrics. Handles Unified Memory on Apple Silicon.
        """
        virtual_mem = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info().rss

        info = {
            "system_total_gb": round(virtual_mem.total / (1024 ** 3), 2),
            "system_available_gb": round(virtual_mem.available / (1024 ** 3), 2),
            "system_used_gb": round(virtual_mem.used / (1024 ** 3), 2),
            "system_percent": virtual_mem.percent,
            "process_rss_gb": round(process_mem / (1024 ** 3), 2),
            "device": self.device,
            "architecture": "Unified Memory Architecture (UMA)" if self.device == "mps" else "Standard Memory Architecture"
        }

        # Query CUDA memory if running on Nvidia GPU
        if self.device == "cuda":
            try:
                import torch
                info["cuda_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
                info["cuda_cached_gb"] = round(torch.cuda.memory_reserved() / (1024 ** 3), 2)
            except Exception:
                pass

        return info

    def load_model(self, model_name: str) -> None:
        """
        Loads the specified model provider if enabled in settings.
        """
        model_name = model_name.lower()
        if model_name in self._models:
            logger.info(f"Model '{model_name}' is already loaded in memory.")
            return

        model_dir = self.settings.model_dir

        if model_name == "hunyuan":
            if not self.settings.ENABLE_HUNYUAN:
                logger.warning("Hunyuan model loading bypassed (disabled in settings).")
                return
            provider = HunyuanProvider(model_dir, self.device)
        elif model_name == "triposr":
            if not self.settings.ENABLE_TRIPOSR:
                logger.warning("TripoSR model loading bypassed (disabled in settings).")
                return
            provider = TripoSRProvider(model_dir, self.device)
        elif model_name == "instantmesh":
            if not self.settings.ENABLE_INSTANTMESH:
                logger.warning("InstantMesh model loading bypassed (disabled in settings).")
                return
            provider = InstantMeshProvider(model_dir, self.device)
        else:
            raise ValueError(f"Unknown AI Model provider: {model_name}")

        start_time = time.perf_counter()
        provider.load()
        duration = time.perf_counter() - start_time

        self._models[model_name] = provider
        self._load_durations[model_name] = duration
        logger.info(f"Model '{model_name}' loaded in {duration:.4f}s.")

    def load_all_enabled(self) -> None:
        """
        Loads all models that are marked enabled in settings.
        """
        logger.info("Initializing enabled models startup load...")
        if self.settings.ENABLE_HUNYUAN:
            self.load_model("hunyuan")
        if self.settings.ENABLE_TRIPOSR:
            self.load_model("triposr")
        if self.settings.ENABLE_INSTANTMESH:
            self.load_model("instantmesh")

    def unload_model(self, model_name: str) -> None:
        """
        Unloads model provider from memory.
        """
        model_name = model_name.lower()
        if model_name in self._models:
            self._models[model_name].unload()
            del self._models[model_name]
            if model_name in self._load_durations:
                del self._load_durations[model_name]
            logger.info(f"Model '{model_name}' successfully removed from manager cache.")
        else:
            logger.warning(f"Model '{model_name}' was not loaded; skip unload.")

    def unload_all(self) -> None:
        """
        Triggers release of resources for all active providers.
        """
        logger.info("Gracefully unloading all cached AI models...")
        active_models = list(self._models.keys())
        for model in active_models:
            self.unload_model(model)

    def get_model(self, model_name: str) -> AIModel:
        """
        Retrieves a loaded provider. Supports lazy loading.
        """
        model_name = model_name.lower()
        if model_name not in self._models:
            # Check enablement status
            is_enabled = False
            if model_name == "hunyuan" and self.settings.ENABLE_HUNYUAN:
                is_enabled = True
            elif model_name == "triposr" and self.settings.ENABLE_TRIPOSR:
                is_enabled = True
            elif model_name == "instantmesh" and self.settings.ENABLE_INSTANTMESH:
                is_enabled = True

            if is_enabled:
                logger.info(f"Model '{model_name}' requested but not cached. Triggering lazy load...")
                self.load_model(model_name)
            else:
                raise RuntimeError(f"Requested model '{model_name}' is disabled or unrecognized.")

        return self._models[model_name]

    def get_status(self) -> Dict[str, Any]:
        """
        Compiles structural configuration and loading statistics for all providers.
        """
        status = {}
        for name in ["hunyuan", "triposr", "instantmesh"]:
            is_enabled = getattr(self.settings, f"ENABLE_{name.upper()}")
            is_loaded = name in self._models
            status[name] = {
                "enabled": is_enabled,
                "loaded": is_loaded,
                "load_time_seconds": self._load_durations.get(name) if is_loaded else None,
                "health": self._models[name].health() if is_loaded else False
            }
        return status
