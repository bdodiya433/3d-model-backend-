import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger("app.services.color.extractor")


@dataclass
class ColorItem:
    """Represents an extracted color with RGB, LAB, Hex, and distribution metrics."""
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float]
    hex_code: str
    weight: float  # Fraction of object area (0.0 - 1.0)
    confidence: float = 1.0

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, weight: float = 1.0, confidence: float = 1.0) -> "ColorItem":
        r = int(np.clip(r, 0, 255))
        g = int(np.clip(g, 0, 255))
        b = int(np.clip(b, 0, 255))
        rgb_f = np.array([[[r, g, b]]], dtype=np.float32) / 255.0
        lab_f = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2LAB)[0, 0]
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        return cls(
            rgb=(r, g, b),
            lab=(float(lab_f[0]), float(lab_f[1]), float(lab_f[2])),
            hex_code=hex_code,
            weight=round(float(weight), 4),
            confidence=round(float(confidence), 4),
        )


@dataclass
class RegionalColorZone:
    """Represents a localized color region (stripe, panel, logo, patch)."""
    color: ColorItem
    pixel_count: int
    bbox: Tuple[float, float, float, float]  # Normalized (ymin, xmin, ymax, xmax)
    centroid: Tuple[float, float]           # Normalized (y, x)
    confidence: float


@dataclass
class ColorExtractionResult:
    """Comprehensive result of perceptual color analysis."""
    dominant_colors: List[ColorItem]
    primary_color: ColorItem
    regional_zones: List[RegionalColorZone]
    denoised_rgba: np.ndarray             # High-res clean image with denoised colors
    object_mask: np.ndarray               # Boolean mask of valid object pixels
    color_map: np.ndarray                 # Quantized/clustered 2D RGB map of the object
    color_confidence: float = 1.0         # Confidence score [0.0, 1.0]
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)



class ColorExtractor:
    """
    Perceptual color extraction engine using CIE-LAB color space and spatial clustering.
    
    Principles:
    1. Color Authority: Source RGB values are the ground truth. No artificial contrast/saturation boost.
    2. Object-Only: Excludes background, alpha fringes, and shadow artifacts while preserving
       authentic dark/black object materials and white/light object materials.
    3. Multi-Color Preservation: Extracts global dominant palette AND local spatial color regions
       (stripes, panels, buttons, two-tone sections) rather than averaging everything into one color.
    4. Perceptual Accuracy: Uses float32 CIE-LAB for clustering and Delta-E distance metrics.
    """

    def __init__(
        self,
        max_clusters: int = 6,
        min_cluster_ratio: float = 0.02,
        alpha_threshold: int = 30,
    ):
        self.max_clusters = max_clusters
        self.min_cluster_ratio = min_cluster_ratio
        self.alpha_threshold = alpha_threshold

    def _load_rgba_array(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Standardizes input into an (H, W, 4) uint8 RGBA numpy array."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGBA")
            return np.array(pil_img, dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGBA"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                # Grayscale
                rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                alpha = np.full(image_input.shape, 255, dtype=np.uint8)
                return np.dstack([rgb, alpha])
            elif image_input.shape[2] == 3:
                alpha = np.full(image_input.shape[:2], 255, dtype=np.uint8)
                return np.dstack([image_input, alpha])
            elif image_input.shape[2] == 4:
                return image_input.astype(np.uint8)
            else:
                raise ValueError(f"Invalid array shape for image input: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def compute_object_mask(self, rgba: np.ndarray, external_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Derives an accurate object mask:
        - Excludes alpha transparency (< alpha_threshold).
        - Excludes obvious transparent-border shadow fringes.
        - Preserves genuine dark/black pixels (alpha == 255 with dark RGB).
        - Preserves genuine light/white pixels.
        """
        h, w = rgba.shape[:2]
        alpha = rgba[..., 3]
        base_mask = alpha > self.alpha_threshold

        if external_mask is not None:
            if external_mask.shape[:2] != (h, w):
                resized_ext = cv2.resize(
                    external_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ) > 0
                base_mask = np.logical_and(base_mask, resized_ext)
            else:
                base_mask = np.logical_and(base_mask, external_mask > 0)

        # Remove border fringe: pixels near the transparent boundary that have low saturation
        # and very low alpha (< 100) are likely anti-aliased background bleed
        fringe_mask = (alpha < 100) & (alpha > 0)
        # Only discard fringe pixels if they border empty space
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(base_mask.astype(np.uint8), kernel, iterations=1)
        boundary = np.logical_and(base_mask, ~eroded.astype(bool))
        
        # Discard low-alpha boundary fringe
        clean_mask = base_mask.copy()
        clean_mask[np.logical_and(boundary, fringe_mask)] = False

        if np.count_nonzero(clean_mask) < 20:
            return base_mask

        return clean_mask

    def extract_colors(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        external_mask: Optional[np.ndarray] = None,
    ) -> ColorExtractionResult:
        """
        Executes perceptual color extraction on the input image.
        Returns global palette, regional zones, and spatial color map.
        """
        rgba = self._load_rgba_array(image_input)
        h, w = rgba.shape[:2]

        object_mask = self.compute_object_mask(rgba, external_mask)
        fg_coords = np.argwhere(object_mask)

        if len(fg_coords) == 0:
            logger.warning("ColorExtractor: No valid object pixels detected. Returning neutral fallback.")
            neutral = ColorItem.from_rgb(200, 200, 200, weight=1.0)
            return ColorExtractionResult(
                dominant_colors=[neutral],
                primary_color=neutral,
                regional_zones=[],
                denoised_rgba=rgba,
                object_mask=object_mask,
                color_map=rgba[..., :3].copy(),
            )

        fg_rgb = rgba[object_mask, :3]  # (N, 3) uint8
        n_pixels = len(fg_rgb)

        # Convert to float32 CIE-LAB for clustering
        fg_rgb_f = fg_rgb.astype(np.float32) / 255.0
        # cv2.cvtColor expects (1, N, 3)
        fg_lab_f = cv2.cvtColor(fg_rgb_f[np.newaxis, ...], cv2.COLOR_RGB2LAB)[0]

        # 1. Determine optimal cluster count K (between 1 and max_clusters)
        # Measure LAB color variance to decide cluster granularity
        lab_std = np.std(fg_lab_f, axis=0)
        total_std = np.sum(lab_std)

        if total_std < 8.0:
            # Monochromatic / single-color object
            target_k = 1
        elif total_std < 18.0:
            target_k = min(2, self.max_clusters)
        elif total_std < 32.0:
            target_k = min(3, self.max_clusters)
        else:
            # Multi-color / complex object
            target_k = min(self.max_clusters, max(3, int(total_std / 10.0)))

        # Subsample pixels for fast K-Means if image is large
        max_samples = 40000
        if n_pixels > max_samples:
            idx = np.random.choice(n_pixels, max_samples, replace=False)
            train_pixels = fg_lab_f[idx]
        else:
            train_pixels = fg_lab_f

        # 2. Run K-Means in float32 CIE-LAB space
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        _, _, centers_lab = cv2.kmeans(
            train_pixels,
            target_k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS
        )

        # Compute distances of all foreground pixels to cluster centers
        # shape: (N, K)
        dists = np.linalg.norm(fg_lab_f[:, np.newaxis, :] - centers_lab[np.newaxis, :, :], axis=2)
        pixel_labels = np.argmin(dists, axis=1)

        # 3. Build ColorItem dictionary keyed by original cluster index k
        cluster_map: Dict[int, ColorItem] = {}
        for k in range(target_k):
            mask_k = (pixel_labels == k)
            count_k = int(np.count_nonzero(mask_k))
            weight_k = count_k / float(n_pixels)

            if weight_k < self.min_cluster_ratio and target_k > 1:
                continue

            # Robust cluster center: median RGB of assigned pixels (avoids outlier influence)
            assigned_rgb = fg_rgb[mask_k]
            median_rgb = np.median(assigned_rgb, axis=0).astype(int)
            confidence = min(1.0, float(weight_k * 2.5 + 0.3))

            item = ColorItem.from_rgb(
                r=median_rgb[0],
                g=median_rgb[1],
                b=median_rgb[2],
                weight=weight_k,
                confidence=confidence,
            )
            cluster_map[k] = item

        # If all clusters fell below threshold, keep at least the largest one
        if not cluster_map:
            median_rgb = np.median(fg_rgb, axis=0).astype(int)
            cluster_map[0] = ColorItem.from_rgb(median_rgb[0], median_rgb[1], median_rgb[2], weight=1.0)

        # Sort dominant colors by weight descending for reporting
        cluster_items = sorted(cluster_map.values(), key=lambda c: c.weight, reverse=True)
        # Normalize weights so sum is 1.0
        total_w = sum(c.weight for c in cluster_items)
        if total_w > 0:
            for c in cluster_items:
                c.weight = round(c.weight / total_w, 4)

        primary_color = cluster_items[0]

        # 4. Regional Color Zone Detection (stripes, panels, badges, spots)
        regional_zones: List[RegionalColorZone] = []
        
        # Create full 2D label map
        full_label_map = np.full((h, w), -1, dtype=np.int32)
        full_label_map[object_mask] = pixel_labels

        # Create quantized 2D color map
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        # Background gets neutral/primary fallback
        color_map[:] = primary_color.rgb

        for orig_k, item in cluster_map.items():
            cluster_mask = (full_label_map == orig_k)
            color_map[cluster_mask] = item.rgb

            if not np.any(cluster_mask):
                continue

            # Find connected components for this color cluster
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cluster_mask.astype(np.uint8), connectivity=8
            )

            for comp_idx in range(1, num_labels):
                comp_area = stats[comp_idx, cv2.CC_STAT_AREA]
                if comp_area < (n_pixels * 0.015):  # Ignore tiny specks (< 1.5% area)
                    continue

                cx = centroids[comp_idx][0] / float(w)
                cy = centroids[comp_idx][1] / float(h)
                x = stats[comp_idx, cv2.CC_STAT_LEFT] / float(w)
                y = stats[comp_idx, cv2.CC_STAT_TOP] / float(h)
                bw = stats[comp_idx, cv2.CC_STAT_WIDTH] / float(w)
                bh = stats[comp_idx, cv2.CC_STAT_HEIGHT] / float(h)

                regional_zones.append(
                    RegionalColorZone(
                        color=item,
                        pixel_count=int(comp_area),
                        bbox=(round(y, 4), round(x, 4), round(y + bh, 4), round(x + bw, 4)),
                        centroid=(round(cy, 4), round(cx, 4)),
                        confidence=min(1.0, float(comp_area / n_pixels * 3.0 + 0.2)),
                    )
                )

        # 5. Denoised RGBA (bilateral filter on color map to smooth noise while keeping sharp edges)
        # Apply bilateral filter to RGB channels of valid object region
        filtered_rgb = cv2.bilateralFilter(color_map, d=5, sigmaColor=30, sigmaSpace=15)
        denoised_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        denoised_rgba[..., :3] = filtered_rgb
        denoised_rgba[..., 3] = np.where(object_mask, 255, 0).astype(np.uint8)

        # 6. Compute Color Confidence Score
        color_confidence, confidence_breakdown = self.compute_color_confidence(
            rgba=rgba,
            object_mask=object_mask,
            cluster_items=cluster_items,
            pixel_labels=pixel_labels,
            centers_lab=centers_lab,
            fg_lab_f=fg_lab_f,
        )

        logger.info(
            f"Extracted {len(cluster_items)} dominant color(s) and {len(regional_zones)} regional zone(s). "
            f"Primary: {primary_color.hex_code} (weight={primary_color.weight}). "
            f"Confidence: {color_confidence:.2f}."
        )

        return ColorExtractionResult(
            dominant_colors=cluster_items,
            primary_color=primary_color,
            regional_zones=regional_zones,
            denoised_rgba=denoised_rgba,
            object_mask=object_mask,
            color_map=color_map,
            color_confidence=color_confidence,
            confidence_breakdown=confidence_breakdown,
        )

    def compute_color_confidence(
        self,
        rgba: np.ndarray,
        object_mask: np.ndarray,
        cluster_items: List[ColorItem],
        pixel_labels: np.ndarray,
        centers_lab: np.ndarray,
        fg_lab_f: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes an objective, explainable Color Confidence Score [0.0, 1.0] based on:
        1. Usable object pixel count & fill ratio (resolution & coverage).
        2. Lighting consistency & shadow contamination (fraction of under-exposed L* < 18 pixels).
        3. Specular highlight contamination (fraction of blown-out L* > 95 pixels).
        4. Color cluster stability (compactness/variance of clusters in LAB space).
        5. Image sharpness vs blur/compression (Laplacian variance).
        """
        h, w = rgba.shape[:2]
        fg_pixels = int(np.count_nonzero(object_mask))

        if fg_pixels < 50:
            return 0.1, {"error": 1.0, "overall_confidence": 0.1}

        # 1. Pixel Count Score: >= 25,000 pixels is optimal (1.0), <= 1,000 pixels is poor (0.2)
        pixel_score = float(np.clip((fg_pixels - 1000) / 24000.0, 0.2, 1.0))

        # 2. Lighting / Shadow Contamination:
        # L* channel in fg_lab_f: 0 (black) to 100 (white)
        l_channel = fg_lab_f[:, 0]
        shadow_ratio = float(np.count_nonzero(l_channel < 18.0)) / float(fg_pixels)
        shadow_penalty = float(np.clip(shadow_ratio * 1.8, 0.0, 0.6))
        shadow_score = float(1.0 - shadow_penalty)

        # 3. Specular Highlight Contamination:
        # Over-exposed pixels (L* > 95) with low chroma
        chroma = np.sqrt(fg_lab_f[:, 1] ** 2 + fg_lab_f[:, 2] ** 2)
        specular_ratio = float(np.count_nonzero((l_channel > 95.0) & (chroma < 10.0))) / float(fg_pixels)
        specular_penalty = float(np.clip(specular_ratio * 2.0, 0.0, 0.5))
        specular_score = float(1.0 - specular_penalty)

        # 4. Color Cluster Stability & Separation:
        assigned_centers = centers_lab[pixel_labels]
        cluster_dists = np.linalg.norm(fg_lab_f - assigned_centers, axis=1)
        mean_cluster_dist = float(np.mean(cluster_dists))
        cluster_stability_score = float(np.clip(1.0 - (mean_cluster_dist - 10.0) / 30.0, 0.4, 1.0))

        # 5. Image Sharpness & Compression (Laplacian Variance):
        gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = float(np.clip((laplacian_var - 20.0) / 130.0, 0.3, 1.0))

        # Weighted aggregate confidence
        weights = {
            "pixel_resolution": 0.20,
            "shadow_quality": 0.25,
            "specular_quality": 0.15,
            "cluster_stability": 0.25,
            "sharpness": 0.15,
        }

        confidence = (
            weights["pixel_resolution"] * pixel_score
            + weights["shadow_quality"] * shadow_score
            + weights["specular_quality"] * specular_score
            + weights["cluster_stability"] * cluster_stability_score
            + weights["sharpness"] * sharpness_score
        )
        confidence = float(np.clip(confidence, 0.1, 1.0))

        breakdown = {
            "overall_confidence": round(confidence, 4),
            "pixel_score": round(pixel_score, 4),
            "shadow_score": round(shadow_score, 4),
            "specular_score": round(specular_score, 4),
            "cluster_stability_score": round(cluster_stability_score, 4),
            "sharpness_score": round(sharpness_score, 4),
            "shadow_ratio": round(shadow_ratio, 4),
            "specular_ratio": round(specular_ratio, 4),
            "laplacian_var": round(laplacian_var, 2),
        }

        return round(confidence, 4), breakdown
