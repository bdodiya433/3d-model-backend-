import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from app.services.image.exceptions import ImageProcessingError, ObjectDetectionError

logger = logging.getLogger("app.services.image.preprocessor")


class ImagePreprocessor:
    """
    Service class responsible for preparing single-object input images
    for Hunyuan3D-2 3D mesh generation.

    Pipeline logic:
      1. Load & validate input image into RGBA mode.
      2. Detect if the image already features a usable alpha transparency mask.
         - If YES: Extract binary mask directly from the alpha channel.
         - If NO:  Run U2-Net Lite segmentation to isolate the object and generate a mask.
      3. Locate single main object bounding box (handling multi-component noise).
      4. Crop the object tightly.
      5. Add configurable square padding (10-15%) and center on transparent canvas.
      6. Resize to target model resolution (default 512x512).
    """

    def __init__(self, u2net_model: str = "u2netp") -> None:
        """
        :param u2net_model: rembg model name to use for segmentation ('u2netp' for U2-Net Lite, 'u2net' for standard).
        """
        self.u2net_model = u2net_model
        self._rembg_session = None

    def _get_rembg_session(self):
        """Lazy loader for rembg session to minimize startup latency and memory overhead."""
        if self._rembg_session is None:
            try:
                from rembg import new_session
                logger.info(f"Initializing rembg session with model '{self.u2net_model}'...")
                self._rembg_session = new_session(self.u2net_model)
            except Exception as e:
                logger.warning(f"Failed to initialize rembg session '{self.u2net_model}': {e}. Falling back to default 'u2net'.")
                from rembg import new_session
                self._rembg_session = new_session("u2net")
        return self._rembg_session

    def load_image(self, input_data: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """
        Loads input data from various sources (file path, raw bytes, or PIL Image)
        and standardizes format to RGBA.
        """
        try:
            if isinstance(input_data, Image.Image):
                image = input_data.copy()
            elif isinstance(input_data, bytes):
                image = Image.open(io.BytesIO(input_data))
            elif isinstance(input_data, (str, Path)):
                path = Path(input_data)
                if not path.exists():
                    raise FileNotFoundError(f"Image path does not exist: {path}")
                image = Image.open(path)
            else:
                raise ValueError(f"Unsupported input_data type: {type(input_data)}")

            image.load()  # Ensure image data is loaded into memory
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            return image
        except (FileNotFoundError, ValueError) as ve:
            raise ImageProcessingError(message=str(ve))
        except Exception as e:
            logger.error(f"Failed to load image: {e}", exc_info=True)
            raise ImageProcessingError(message=f"Failed to load input image: {str(e)}")

    def has_transparency(
        self,
        image: Image.Image,
        alpha_threshold: int = 10,
        min_transparent_ratio: float = 0.02,
    ) -> bool:
        """
        Determines whether the image possesses a usable, non-trivial alpha channel transparency mask.
        Returns True ONLY if the image mode supports alpha AND contains both background transparent
        pixels and foreground opaque pixels.
        """
        if image.mode not in ("RGBA", "LA", "PA"):
            return False

        alpha = np.array(image.getchannel("A"), dtype=np.uint8)
        total_pixels = alpha.size
        if total_pixels == 0:
            return False

        transparent_pixels = np.count_nonzero(alpha <= alpha_threshold)
        opaque_pixels = total_pixels - transparent_pixels

        transparent_ratio = transparent_pixels / total_pixels
        

        # Require a minimal non-trivial fraction of transparent pixels AND some opaque object pixels
        if transparent_ratio >= min_transparent_ratio and opaque_pixels >= 10:
            return True

        return False

    def extract_alpha_mask(self, image: Image.Image, alpha_threshold: int = 10) -> np.ndarray:
        """
        Extracts a binary mask (uint8 array with values 0 or 255) directly from the image's alpha channel.
        """
        if "A" in image.getbands():
            alpha = np.array(image.getchannel("A"), dtype=np.uint8)
        else:
            alpha = np.array(image.convert("RGBA").getchannel("A"), dtype=np.uint8)

        mask = np.where(alpha > alpha_threshold, np.uint8(255), np.uint8(0))
        return mask

    def detect_negative_spaces_from_background(
        self,
        image: Image.Image,
        candidate_mask: np.ndarray,
        bg_delta_threshold: float = 12.0,
        min_hole_area: int = 30,
    ) -> np.ndarray:
        """
        Detects background negative spaces (e.g. handle loops, through-holes, hollow openings)
        that may have been falsely included inside candidate foreground mask by coarse segmenters.
        """
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        border_thickness = max(4, min(h, w) // 50)
        top_strip = arr[:border_thickness, :]
        left_strip = arr[:, :border_thickness]
        right_strip = arr[:, -border_thickness:]
        perimeter_pixels = np.vstack([
            top_strip.reshape(-1, 3),
            left_strip.reshape(-1, 3),
            right_strip.reshape(-1, 3),
        ])
        bg_color = np.median(perimeter_pixels, axis=0)

        color_dist = np.linalg.norm(arr.astype(float) - bg_color, axis=2)
        matches_bg = color_dist < bg_delta_threshold

        # Region inside mask that matches background color
        internal_bg = np.logical_and(candidate_mask > 0, matches_bg)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            internal_bg.astype(np.uint8), connectivity=8
        )
        
        carved = candidate_mask.copy()
        carved_count = 0
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= min_hole_area:
                carved[labels == lbl] = 0
                carved_count += 1

        if carved_count > 0:
            logger.info(f"Carved {carved_count} negative space / through-hole region(s) from foreground mask.")
        return carved

    def generate_mask_with_u2net(self, image: Image.Image) -> np.ndarray:
        """
        Executes background removal to isolate foreground and preserves genuine
        negative spaces (through-holes, handle loops, openings).
        """
        try:
            from rembg import remove
            session = self._get_rembg_session()

            logger.info("Executing background removal with negative space preservation...")
            output_img = remove(image, session=session, bgcolor=[0, 0, 0, 0])

            alpha = np.array(output_img.getchannel("A"), dtype=np.uint8)
            base_mask = np.where(alpha > 10, np.uint8(255), np.uint8(0))

            # Carve out background negative spaces if coarse segmenter merged them
            mask = self.detect_negative_spaces_from_background(image, base_mask)
            return mask
        except Exception as e:
            logger.error(f"Background segmentation failed: {e}", exc_info=True)
            raise ImageProcessingError(message=f"Background segmentation failed: {str(e)}")

    def compute_bounding_box(
        self,
        mask: np.ndarray,
        min_area_ratio: float = 0.0005,
        min_pixels: int = 50,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Computes the bounding box (xmin, ymin, xmax, ymax) of the single main object in the binary mask.
        Handles edge cases such as multiple disconnected components by isolating the largest component contour,
        and filters out tiny background artifacts/noise.
        """
        if mask is None or np.count_nonzero(mask) < min_pixels:
            return None

        total_area = mask.shape[0] * mask.shape[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter contours by minimum area threshold to eliminate noise
        valid_contours = [
            c for c in contours if cv2.contourArea(c) >= (total_area * min_area_ratio)
        ]

        if not valid_contours:
            # Fallback to absolute largest contour if all fall below area ratio
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < min_pixels:
                return None
        else:
            # Isolate single main object: pick contour with maximum area
            largest_contour = max(valid_contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        # Enforce minimum size sanity check
        if w < 5 or h < 5:
            return None

        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w)
        ymax = int(y + h)

        return (xmin, ymin, xmax, ymax)

    def crop_object(
        self,
        image: Image.Image,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        apply_color_bleed: bool = True,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Applies the binary mask to the alpha channel without destroying original photographic RGB values.
        Extends edge colors into transparent border using distance transform (alpha bleed)
        so that bilinear/Lanczos resampling near the silhouette never blends with black or grey.
        """
        img_arr = np.array(image.convert("RGBA"))
        h, w = img_arr.shape[:2]

        # Enforce exact alpha channel from segmentation mask
        binary_mask = (mask > 0).astype(bool)

        if apply_color_bleed and np.any(binary_mask) and not np.all(binary_mask):
            try:
                from scipy.ndimage import distance_transform_edt
                # Find nearest foreground pixel for every background pixel
                _, indices = distance_transform_edt(~binary_mask, return_indices=True)
                # Propagate genuine foreground RGB into the transparent background
                img_arr[..., :3] = img_arr[indices[0], indices[1], :3]
            except Exception as e:
                logger.debug(f"Alpha bleed note: {e}")

        # Strictly set alpha channel from mask (transparent pixels have alpha=0, but preserve foreground RGB)
        img_arr[..., 3] = np.where(binary_mask, 255, 0).astype(np.uint8)
        masked_img = Image.fromarray(img_arr, mode="RGBA")

        xmin, ymin, xmax, ymax = bbox
        cropped_img = masked_img.crop((xmin, ymin, xmax, ymax))
        cropped_mask = mask[ymin:ymax, xmin:xmax]

        return cropped_img, cropped_mask

    def compute_adaptive_padding(self, mask: np.ndarray) -> float:
        """
        Computes adaptive padding ratio between 0.06 and 0.10 based on object shape complexity.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return 0.10
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.10
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            return 0.10
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        padding = 0.06 + 0.04 * np.clip(circularity, 0.0, 1.0)
        return float(np.clip(padding, 0.06, 0.10))

    def pad_to_square(
        self,
        cropped_image: Image.Image,
        padding_ratio: float = 0.10,
        cropped_mask: Optional[np.ndarray] = None,
        return_offsets: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, int, int, int]]:
        """
        Centers the cropped object on a square transparent canvas with uniform fit-center letterboxing.
        If return_offsets is True, returns (square_canvas, offset_x, offset_y, canvas_size).
        Otherwise returns square_canvas for backward compatibility.
        """
        width, height = cropped_image.size
        max_dim = max(width, height)

        padding = int(round(max_dim * padding_ratio))
        canvas_size = max_dim + 2 * padding

        square_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

        # Calculate centering offsets
        offset_x = padding + (max_dim - width) // 2
        offset_y = padding + (max_dim - height) // 2

        square_canvas.paste(cropped_image, (offset_x, offset_y), mask=cropped_image)
        if return_offsets:
            return square_canvas, offset_x, offset_y, canvas_size
        return square_canvas

    def resize(
        self,
        square_image: Image.Image,
        target_resolution: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """
        Resizes the square transparent image to target resolution using high quality Lanczos filter.
        """
        return square_image.resize(target_resolution, Image.Resampling.LANCZOS)

    def refine_mask(
        self,
        mask: np.ndarray,
        max_hole_area: int = 15,
        min_speck_area: int = 25,
        iterations: int = 1,
    ) -> np.ndarray:
        """
        Cleans up raw segmentation mask conservatively:
        - Fills internal pinholes inside foreground object via contour hierarchy (cv2.RETR_CCOMP),
          preserving 100% of thin external protrusions, antennas, fins, and handles.
        - Prunes tiny isolated external noise specks (< min_speck_area) via connected components.
        - Does NOT use destructive morphological opening (MORPH_OPEN) which erodes thin features.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return mask

        # 1. Fill internal pinholes using 2-level contour hierarchy
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled = mask.copy()
        if hierarchy is not None and len(hierarchy) > 0:
            for i, h in enumerate(hierarchy[0]):
                # h[3] != -1 indicates an internal hole (child contour of an external boundary)
                if h[3] != -1:
                    hole_area = cv2.contourArea(contours[i])
                    if hole_area <= max_hole_area:
                        cv2.drawContours(filled, contours, i, 255, -1)

        # 2. Prune small disconnected background noise specks
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        refined = np.zeros_like(filled)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_speck_area:
                refined[labels == lbl] = 255

        # Fallback: if all components were filtered, return original
        if np.count_nonzero(refined) == 0:
            return filled

        return refined

    def analyze_silhouette(
        self,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> dict:
        """
        Extracts silhouette features from the binary mask and bounding box:
        - Bounding box dimensions & aspect ratio (w/h)
        - Foreground area and fill ratio
        - Center of mass / centroid
        - Horizontal symmetry estimate
        """
        xmin, ymin, xmax, ymax = bbox
        w = max(1, xmax - xmin)
        h = max(1, ymax - ymin)
        aspect_ratio = float(w) / float(h)

        cropped_mask = mask[ymin:ymax, xmin:xmax]
        fg_area = int(np.count_nonzero(cropped_mask))
        bbox_area = w * h
        fill_ratio = float(fg_area) / float(max(1, bbox_area))

        # Center of mass calculation
        moments = cv2.moments(cropped_mask)
        if moments["m00"] > 0:
            cx = (moments["m10"] / moments["m00"]) / float(w)
            cy = (moments["m01"] / moments["m00"]) / float(h)
        else:
            cx, cy = 0.5, 0.5

        # Horizontal symmetry estimation (flip cropped mask horizontally and compute IoU)
        flipped_mask = cv2.flip(cropped_mask, 1)
        intersection = np.logical_and(cropped_mask > 0, flipped_mask > 0).sum()
        union = np.logical_or(cropped_mask > 0, flipped_mask > 0).sum()
        symmetry_score = float(intersection) / float(max(1, union))

        return {
            "bbox": (xmin, ymin, xmax, ymax),
            "width": w,
            "height": h,
            "aspect_ratio": round(aspect_ratio, 4),
            "foreground_area": fg_area,
            "fill_ratio": round(fill_ratio, 4),
            "center_of_mass": (round(cx, 4), round(cy, 4)),
            "symmetry_score": round(symmetry_score, 4),
            "orientation": "horizontal" if aspect_ratio > 1.2 else ("vertical" if aspect_ratio < 0.8 else "square"),
        }

    def preprocess_with_analysis(
        self,
        input_data: Union[str, Path, bytes, Image.Image],
        target_resolution: Tuple[int, int] = (1024, 1024),
        padding_ratio: Optional[float] = None,
    ) -> dict:
        """
        Executes enhanced preprocessing returning both:
        1. 'highres_working_image': Segmented, cropped object preserving native resolution and fine color details.
        2. 'ai_image': Standardized square canvas with controlled adaptive padding for Hunyuan3D-2 geometry generator.
        3. 'mask': Binary segmentation mask.
        4. 'silhouette': Detailed silhouette analysis dictionary.
        5. 'bbox': Bounding box coordinates.
        """
        # 1. Load image
        img = self.load_image(input_data)

        # 2. Transparency check & mask generation
        if self.has_transparency(img):
            logger.info("Input image contains usable transparency. Extracting alpha mask.")
            raw_mask = self.extract_alpha_mask(img)
        else:
            logger.info("Input image is opaque. Running U2-Net Lite segmentation...")
            raw_mask = self.generate_mask_with_u2net(img)

        # 3. Refine mask conservatively (preserving 100% of thin features)
        mask = self.refine_mask(raw_mask)

        # 4. Compute Bounding Box
        bbox = self.compute_bounding_box(mask)

        # Fallback if initial mask failed
        if bbox is None and self.has_transparency(img):
            logger.warning("Alpha channel mask yielded no valid object. Retrying with U2-Net Lite...")
            raw_mask = self.generate_mask_with_u2net(img)
            mask = self.refine_mask(raw_mask)
            bbox = self.compute_bounding_box(mask)

        if bbox is None:
            raise ObjectDetectionError(
                message="No valid object detected in the image. Please upload a clear image containing a single main object."
            )

        orig_w, orig_h = img.size

        # 5. Extract silhouette features
        silhouette = self.analyze_silhouette(mask, bbox)

        # 6. Crop object with edge color bleeding (preserving authentic RGB without black fringe)
        cropped_img, cropped_mask = self.crop_object(img, mask, bbox, apply_color_bleed=True)
        w_crop, h_crop = cropped_img.size

        # 7. Letterbox Canvas Framing: Standardize to uniform 10% padding to keep multi-view scales consistent
        effective_padding = padding_ratio if padding_ratio is not None else 0.10
        silhouette["padding_ratio"] = round(effective_padding, 4)

        square_img, offset_x_raw, offset_y_raw, canvas_size = self.pad_to_square(
            cropped_img, padding_ratio=effective_padding, cropped_mask=cropped_mask, return_offsets=True
        )

        ai_image = self.resize(square_img, target_resolution=target_resolution)

        # High-res working image for texture baking (preserve 2048 or target_resolution)
        tex_res = max(target_resolution[0], 2048) if canvas_size >= 1500 else target_resolution[0]
        highres_working_img = self.resize(square_img, target_resolution=(tex_res, tex_res))

        # Calculate exact transform metadata for camera projection alignment
        scale = float(target_resolution[0]) / float(canvas_size)
        offset_x = int(round(offset_x_raw * scale))
        offset_y = int(round(offset_y_raw * scale))

        transform_metadata = {
            "original_width": int(orig_w),
            "original_height": int(orig_h),
            "processed_width": int(target_resolution[0]),
            "processed_height": int(target_resolution[1]),
            "scale": round(scale, 4),
            "offset_x": offset_x,
            "offset_y": offset_y,
            "bbox": [int(x) for x in bbox],
            "crop_width": int(w_crop),
            "crop_height": int(h_crop),
            "padding_ratio": round(effective_padding, 4),
        }

        # Build mask_debug showing foreground (green), preserved negative spaces/holes (red), background (black)
        mask_debug_arr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        mask_debug_arr[mask > 0] = [40, 220, 60]
        if "raw_mask" in locals() and raw_mask is not None:
            carved_holes = np.logical_and(raw_mask > 0, mask == 0)
            mask_debug_arr[carved_holes] = [230, 40, 40]
        mask_debug = Image.fromarray(mask_debug_arr, mode="RGB")

        logger.info(
            f"Preprocessed image: Canvas {ai_image.size}, Scale: {transform_metadata['scale']}, "
            f"Offset: ({offset_x}, {offset_y}), Aspect: {silhouette['aspect_ratio']}, Fill: {silhouette['fill_ratio']}."
        )

        return {
            "ai_image": ai_image,
            "highres_working_image": highres_working_img,
            "cropped_image": cropped_img,
            "mask": mask,
            "cropped_mask": cropped_mask,
            "mask_debug": mask_debug,
            "transform_metadata": transform_metadata,
            "bbox": bbox,
            "silhouette": silhouette,
        }

    def preprocess(
        self,
        input_data: Union[str, Path, bytes, Image.Image],
        target_resolution: Tuple[int, int] = (512, 512),
        padding_ratio: Optional[float] = None,
    ) -> Image.Image:
        """
        Executes full automated preprocessing pipeline.
        Maintains backwards compatibility by returning the resized AI conditioning image.
        """
        res = self.preprocess_with_analysis(
            input_data=input_data,
            target_resolution=target_resolution,
            padding_ratio=padding_ratio,
        )
        return res["ai_image"]

