import logging
from typing import Any
from PIL import Image

logger = logging.getLogger("app.ai.hunyuan.predictor")


class HunyuanPredictor:
    """
    Executes forward pass/inference on the loaded Hunyuan3D-2 pipeline.
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def predict(
        self,
        image: Image.Image,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        **kwargs: Any
    ) -> Any:
        """
        Runs the Hunyuan3D pipeline to generate 3D mesh output from an input image.
        """
        if self.pipeline is None:
            raise RuntimeError("Hunyuan3D-2 pipeline has not been loaded.")

        logger.info("Executing Hunyuan3D-2 forward pass...")
        try:
            # Call the pipeline __call__ method
            mesh = self.pipeline(
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            )
            return mesh
        except Exception as e:
            logger.error(f"Hunyuan3D-2 inference failed: {e}")
            raise e
