"""
texture_projection.py - Module for unwarping and projecting original reference photo pixels
onto the canonical cylindrical and planar UV unwrapped texture regions.
Preserves original printed text, typography, barcodes, droplets, and metal details.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image

logger = logging.getLogger("backend.texture_projection")


class TextureProjector:
    """
    Extracts foreground pixels from the 6 segmented reference photographs
    and prepares rectified projection layers for texture blending.
    """

    def __init__(self):
        pass

    def extract_cropped_views(
        self,
        segmented_views: Dict[str, Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """
        Crops each photograph tightly to the active foreground can region
        using the computed bounding box and mask.
        """
        cropped_views = {}

        for view_name, data in segmented_views.items():
            img_path = Path(data["path"])
            pil_img = Image.open(img_path).convert("RGBA")
            arr = np.array(pil_img)
            bbox = data["bbox"]
            mask = data["mask"]

            xmin, ymin, xmax, ymax = bbox
            h_crop, w_crop = ymax - ymin, xmax - xmin

            # Add tiny 1% inset to eliminate background fringes
            xmin_in = min(arr.shape[1] - 1, max(0, int(xmin + 0.01 * w_crop)))
            xmax_in = max(0, min(arr.shape[1], int(xmax - 0.01 * w_crop)))
            ymin_in = min(arr.shape[0] - 1, max(0, int(ymin + 0.005 * h_crop)))
            ymax_in = max(0, min(arr.shape[0], int(ymax - 0.005 * h_crop)))

            crop = arr[ymin_in:ymax_in, xmin_in:xmax_in].copy()
            # If transparent background mask exists, apply it
            if mask is not None:
                mask_crop = mask[ymin_in:ymax_in, xmin_in:xmax_in]
                crop[mask_crop == 0, 3] = 0

            cropped_views[view_name] = crop
            logger.info(f"Extracted cropped view '{view_name}': shape={crop.shape}")

        return cropped_views
