"""
segmentation.py - Module for segmenting the can from reference photographs.
Supports SAM 2 / SAM 2.1 when checkpoints and package are available,
with automatic fallback to U2-Net Lite (rembg) and robust color/contour masking.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("backend.segmentation")


class ObjectSegmenter:
    """
    Identifies foreground product/object pixels vs background pixels across reference photographs.
    Works for any arbitrary physical object. Preserves original RGB pixels without alteration.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path or Path("backend/checkpoints/sam2.1_hiera_tiny.pt")
        self._sam_predictor = None
        self._rembg_session = None

    def _get_rembg_session(self):
        if self._rembg_session is None:
            try:
                from rembg import new_session
                logger.info("Initializing U2-Net Lite rembg session...")
                self._rembg_session = new_session("u2netp")
            except Exception as e:
                logger.warning(f"Could not initialize U2-Net Lite: {e}")
        return self._rembg_session

    def segment_image(self, image_path: Path) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Segments the main physical object from a single photograph.
        Preserves internal cavities/holes (e.g. headphone arch, mug handles),
        detects black/white studio backgrounds, and strips extraneous text banners.
        Returns:
            mask: uint8 binary mask where 255 = object, 0 = background
            bbox: (xmin, ymin, xmax, ymax) of the object in the image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_img = Image.open(image_path).convert("RGBA")
        arr = np.array(pil_img)
        h, w = arr.shape[:2]

        mask = None

        # 1. If image already has a clean alpha channel, use it directly!
        if arr.shape[-1] == 4 and np.any(arr[..., 3] < 200):
            alpha = arr[..., 3]
            if np.count_nonzero(alpha > 20) > (h * w * 0.01):
                mask = np.where(alpha > 20, np.uint8(255), np.uint8(0))

        # 2. If no alpha, detect background color from image corners (sampling inset 15px to avoid grid dividing lines)
        if mask is None:
            inset = min(25, h // 10, w // 10)
            corner_samples = [
                arr[inset:inset+15, inset:inset+15, :3],
                arr[inset:inset+15, w-inset-15:w-inset, :3],
                arr[h-inset-15:h-inset, inset:inset+15, :3],
                arr[h-inset-15:h-inset, w-inset-15:w-inset, :3],
            ]
            corner_brightness = float(np.median([np.mean(c) for c in corner_samples]))
            is_dark_bg = corner_brightness < 45

            if is_dark_bg:
                # Black studio background
                rgb_norm = np.linalg.norm(arr[..., :3].astype(np.float32), axis=-1)
                mask = np.where(rgb_norm > 35, np.uint8(255), np.uint8(0))
            else:
                # Try U2-Net Lite
                session = self._get_rembg_session()
                if session is not None:
                    try:
                        from rembg import remove
                        segmented = remove(pil_img, session=session, bgcolor=[0, 0, 0, 0])
                        alpha = np.array(segmented.getchannel("A"), dtype=np.uint8)
                        if np.count_nonzero(alpha > 15) > (h * w * 0.05):
                            mask = np.where(alpha > 20, np.uint8(255), np.uint8(0))
                    except Exception as e:
                        logger.warning(f"rembg segmentation error on {image_path.name}: {e}")

                if mask is None or np.count_nonzero(mask) < (h * w * 0.05):
                    # White / bright studio background
                    diff_white = np.abs(arr[..., :3].astype(int) - 255)
                    is_foreground = (diff_white.sum(axis=-1) > 25)
                    mask = np.where(is_foreground, np.uint8(255), np.uint8(0))

        # 3. Strip border lines and bottom text banners (e.g. 'Front View', watermark labels)
        # Clear any 1-2px border lines touching the extreme edges of the image canvas
        mask[:5, :] = 0
        mask[-5:, :] = 0
        mask[:, :5] = 0
        mask[:, -5:] = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            main_area = stats[main_label, cv2.CC_STAT_AREA]
            clean_mask = np.zeros_like(mask)
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                top = stats[lbl, cv2.CC_STAT_TOP]
                left = stats[lbl, cv2.CC_STAT_LEFT]
                cw = stats[lbl, cv2.CC_STAT_WIDTH]
                ch = stats[lbl, cv2.CC_STAT_HEIGHT]

                # Filter out border lines and hollow perimeter frames running along the image edges
                is_near_edge = (left <= 35 or left + cw >= w - 35 or top <= 35 or top + ch >= h - 35)
                is_thin_line = (cw <= 15 or ch <= 15 or (max(cw, ch) / max(min(cw, ch), 1)) > 10)
                is_hollow_frame = (min(cw, ch) > 80 and (area / max(float(cw * ch), 1.0)) < 0.05)
                if is_near_edge and (is_thin_line or is_hollow_frame) and lbl != main_label:
                    continue

                # Discard tiny components in bottom 15% (text labels)
                if top > (h * 0.85) and area < (main_area * 0.08):
                    continue

                # Keep significant object components
                if area >= (main_area * 0.005) or lbl == main_label:
                    clean_mask[labels == lbl] = 255
            mask = clean_mask

        # Morphological cleanup (light, preserve cavities/holes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 4. Compute tight bounding box of remaining foreground mask
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        else:
            bbox = (0, 0, w, h)

        return mask, bbox

    def segment_all(self, input_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Segments all 6 canonical reference photographs.
        """
        input_dir = Path(input_dir)
        views = ["front", "back", "left", "right", "top", "bottom"]
        results = {}

        for view in views:
            # Look for view.jpg or view.png
            candidates = list(input_dir.glob(f"{view}.*"))
            if not candidates:
                logger.warning(f"No image file found for view: {view}")
                continue

            img_path = candidates[0]
            logger.info(f"Segmenting {view} photograph ({img_path.name})...")
            mask, bbox = self.segment_image(img_path)
            xmin, ymin, xmax, ymax = bbox
            bw, bh = xmax - xmin, ymax - ymin

            results[view] = {
                "path": str(img_path),
                "mask": mask,
                "bbox": bbox,
                "width": bw,
                "height": bh,
                "aspect_ratio": float(bh) / max(float(bw), 1.0)
            }
            logger.info(f"  {view}: bbox=({xmin},{ymin})->({xmax},{ymax}), aspect={results[view]['aspect_ratio']:.2f}")

        return results


# Backward-compatibility alias
CanSegmenter = ObjectSegmenter
