from pathlib import Path
from typing import Any
from app.ai.base.ai_model import AIModel
from app.ai.providers.hunyuan.loader import HunyuanModelLoader
from app.ai.providers.hunyuan.predictor import HunyuanPredictor
from app.ai.providers.hunyuan.exporter import HunyuanExporter


class Hunyuan3DModel(AIModel):
    """
    Concrete implementation of AIModel for Tencent Hunyuan3D-2.
    Integrates loading, prediction, and exporting capabilities.
    """

    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self.loader = HunyuanModelLoader(weights_dir)
        self.predictor = None
        self.exporter = HunyuanExporter()

    def load(self) -> None:
        """
        Loads the model weights and instantiates the predictor.
        """
        pipeline = self.loader.load_model()
        self.predictor = HunyuanPredictor(pipeline)

    def unload(self) -> None:
        """
        Unloads model checkpoints and frees GPU/MPS resources.
        """
        self.loader.unload_model()
        self.predictor = None

    def is_loaded(self) -> bool:
        """
        Returns whether the model is loaded into memory.
        """
        return self.loader.pipeline is not None

    def predict(self, input_data: Any, **kwargs: Any) -> Any:
        """
        Executes model prediction.
        """
        if not self.is_loaded() or self.predictor is None:
            raise RuntimeError("Hunyuan3D-2 model is not loaded in memory.")
        return self.predictor.predict(input_data, **kwargs)

    def export(self, mesh: Any, output_path: Path, format: str = "glb") -> None:
        """
        Delegates the mesh export task to HunyuanExporter.
        """
        self.exporter.export(mesh, output_path, format=format)
