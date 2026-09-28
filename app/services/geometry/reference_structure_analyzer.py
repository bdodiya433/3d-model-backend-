import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger("app.services.geometry.reference_structure_analyzer")


@dataclass
class StructuralFeature:
    """Generic geometry descriptor for a detected physical component."""
    feature_id: str
    type: str                          # "main_body", "protrusion", "elongated_region", "recessed_pocket", "loop_arch", "ring_rim"
    bbox: Tuple[float, float, float, float]  # Normalized [ymin, xmin, ymax, xmax] in [0, 1]
    center: Tuple[float, float]        # Normalized [cy, cx] in [0, 1]
    area_ratio: float                  # Fraction of total foreground area
    orientation: float                 # Principal axis angle in degrees [-90.0, 90.0]
    aspect_ratio: float                # Length / Width (>= 1.0)
    edge_strength: float               # Mean gradient strength along boundary [0, 1]
    boundary_strength: float           # Distinctness from neighboring components [0, 1]
    depth_cue: float                   # Relative elevation estimate [0, 1]
    symmetry: float                    # Bilateral symmetry score [0, 1]
    confidence: float                  # Overall detection confidence [0, 1]


@dataclass
class NegativeSpaceRegion:
    """Generic descriptor for an internal hole, channel, or opening."""
    region_id: str
    bbox: Tuple[float, float, float, float]  # Normalized [ymin, xmin, ymax, xmax]
    center: Tuple[float, float]        # Normalized [cy, cx]
    area_ratio: float                  # Relative area compared to foreground
    is_through_hole: bool              # True if connects to background
    aspect_ratio: float                # Elongation length / width
    orientation: float                 # Principal orientation angle
    pixel_area: int = 0
    contour: Optional[np.ndarray] = None


@dataclass
class StructuralRelationship:
    """Pairwise relationship between two structural features."""
    feature_a_id: str
    feature_b_id: str
    distance: float                    # Normalized centroid distance (relative to main body scale)
    relative_angle: float              # Direction angle in degrees
    adjacency: bool                    # Boundaries touch or are in close proximity
    alignment: float                   # Cosine of angle difference [-1, 1]
    containment: bool                  # One component is inside/enclosed by another


@dataclass
class StructuralReferenceMap:
    """Complete structural blueprint extracted from reference image."""
    outer_silhouette: np.ndarray       # (H, W) uint8 binary mask (0 or 255)
    structural_edges: np.ndarray       # (H, W) uint8 binary edge map
    components: List[StructuralFeature]
    negative_spaces: List[NegativeSpaceRegion]
    relationships: List[StructuralRelationship]
    aspect_ratio: float                # Overall object width / height
    symmetry_score: float              # Horizontal bilateral symmetry
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReferenceStructureAnalyzer:
    """
    Generic Reference Structure Analyzer.
    
    Extracts structural geometry blueprints from reference images WITHOUT
    hardcoded product rules. Differentiates true structural boundaries from
    surface texture and identifies negative spaces (holes, channels) and
    major components with their spatial relationships.
    """

    def __init__(
        self,
        min_component_area_ratio: float = 0.015,
        min_negative_space_area: int = 20,
        gradient_threshold_ratio: float = 0.18,
    ):
        self.min_component_area_ratio = min_component_area_ratio
        self.min_negative_space_area = min_negative_space_area
        self.gradient_threshold_ratio = gradient_threshold_ratio

    def _standardize_inputs(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        mask_input: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and standardizes RGB image array and binary mask."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGBA")
            arr = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGBA"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                arr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGBA)
            elif image_input.shape[2] == 3:
                arr = cv2.cvtColor(image_input, cv2.COLOR_RGB2RGBA)
            else:
                arr = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        rgb = arr[..., :3]
        if mask_input is not None:
            mask = (mask_input > 10).astype(np.uint8) * 255
        elif arr.shape[2] == 4 and np.any(arr[..., 3] < 250):
            mask = np.where(arr[..., 3] > 10, 255, 0).astype(np.uint8)
        else:
            # Fallback: simple Otsu on grayscale
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return rgb, mask

    def classify_structural_edges(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Differentiates genuine physical structural boundaries from lighting falloffs
        and superficial surface textures.
        """
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        fg_mask = (mask > 10).astype(np.uint8)

        # 1. Outer Silhouette Boundary
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        silhouette_edges = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(silhouette_edges, contours, -1, 255, 1)

        # 2. Gradient Magnitude & Direction via Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_mag = np.clip(grad_mag / (grad_mag.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)

        # 3. Adaptive Canny Edge Detection
        v_median = np.median(gray[fg_mask > 0]) if np.any(fg_mask > 0) else 128
        lower_t = int(max(20, (1.0 - 0.33) * v_median))
        upper_t = int(min(240, (1.0 + 0.33) * v_median))
        canny_edges = cv2.Canny(gray, lower_t, upper_t)
        canny_edges[fg_mask == 0] = 0

        # 4. Filter structural boundaries by contour length and gradient strength
        # True structural boundaries form longer continuous chains, while textures form short fragments
        edge_contours, _ = cv2.findContours(canny_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        structural_internal = np.zeros((h, w), dtype=np.uint8)
        texture_detail = np.zeros((h, w), dtype=np.uint8)

        min_structural_length = max(12, int(min(h, w) * 0.04))
        for cnt in edge_contours:
            length = cv2.arcLength(cnt, False)
            # Sample gradient magnitude along contour
            pts = cnt.reshape(-1, 2)
            pts_x = np.clip(pts[:, 0], 0, w - 1)
            pts_y = np.clip(pts[:, 1], 0, h - 1)
            mean_grad = np.mean(grad_mag[pts_y, pts_x])

            if length >= min_structural_length and mean_grad >= 35.0:
                cv2.drawContours(structural_internal, [cnt], -1, 255, 1)
            else:
                cv2.drawContours(texture_detail, [cnt], -1, 255, 1)

        # 5. Combined Structural Edge Blueprint
        combined_structural = cv2.bitwise_or(silhouette_edges, structural_internal)

        edge_breakdown = {
            "outer_silhouette": silhouette_edges,
            "structural_internal": structural_internal,
            "surface_texture": texture_detail,
            "gradient_magnitude": grad_mag,
        }
        return combined_structural, edge_breakdown

    def detect_negative_spaces(
        self,
        mask: np.ndarray,
        rgb: np.ndarray,
    ) -> List[NegativeSpaceRegion]:
        """
        Detects internal negative spaces (holes, openings, channels, recessed pockets).
        """
        h, w = mask.shape[:2]
        fg_mask = (mask > 10).astype(np.uint8)
        total_fg_pixels = max(1, int(np.count_nonzero(fg_mask)))

        # Detect internal hole contours via 2-level hierarchy
        contours, hierarchy = cv2.findContours(fg_mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        negative_spaces: List[NegativeSpaceRegion] = []

        if hierarchy is not None and len(hierarchy) > 0:
            for i, h_elem in enumerate(hierarchy[0]):
                # h_elem[3] != -1 denotes an internal hole inside an external boundary
                if h_elem[3] != -1:
                    cnt = contours[i]
                    area = cv2.contourArea(cnt)
                    if area < self.min_negative_space_area:
                        continue

                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))

                    # Principal orientation via moments
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        mu20 = M["mu20"] / M["m00"]
                        mu02 = M["mu02"] / M["m00"]
                        mu11 = M["mu11"] / M["m00"]
                        angle = 0.5 * np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02))
                    else:
                        cx = x + bw / 2.0
                        cy = y + bh / 2.0
                        angle = 0.0

                    # Check if region matches perimeter background
                    region_id = f"negative_space_{len(negative_spaces) + 1:02d}"
                    negative_spaces.append(
                        NegativeSpaceRegion(
                            region_id=region_id,
                            bbox=(round(y / h, 4), round(x / w, 4), round((y + bh) / h, 4), round((x + bw) / w, 4)),
                            center=(round(cy / h, 4), round(cx / w, 4)),
                            area_ratio=round(area / total_fg_pixels, 4),
                            is_through_hole=True,
                            aspect_ratio=round(aspect, 2),
                            orientation=round(angle, 1),
                            pixel_area=int(area),
                            contour=cnt,
                        )
                    )

        # Also detect deep recessed pockets (e.g. rectangular channels)
        # Recessed channels have strong internal boundaries and characteristic aspect ratio
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        internal_edges = cv2.Canny(gray, 40, 100)
        internal_edges[fg_mask == 0] = 0

        # Morphological closing to find enclosed channel pockets
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(internal_edges, cv2.MORPH_CLOSE, kernel)
        pockets, p_hier = cv2.findContours(closed_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if p_hier is not None and len(p_hier) > 0:
            for i, p_elem in enumerate(p_hier[0]):
                if p_elem[3] != -1:
                    p_cnt = pockets[i]
                    p_area = cv2.contourArea(p_cnt)
                    # Significant area and elongated aspect ratio indicates a recessed slot/channel
                    if p_area >= 150:
                        px, py, pbw, pbh = cv2.boundingRect(p_cnt)
                        p_aspect = float(max(pbw, pbh)) / float(max(1, min(pbw, pbh)))
                        if p_aspect >= 2.0:
                            # Avoid duplicates with already detected through-holes
                            overlap = False
                            for ns in negative_spaces:
                                (ny1, nx1, ny2, nx2) = ns.bbox
                                if abs(py / h - ny1) < 0.05 and abs(px / w - nx1) < 0.05:
                                    overlap = True
                                    break
                            if not overlap:
                                negative_spaces.append(
                                    NegativeSpaceRegion(
                                        region_id=f"recessed_channel_{len(negative_spaces) + 1:02d}",
                                        bbox=(round(py / h, 4), round(px / w, 4), round((py + pbh) / h, 4), round((px + pbw) / w, 4)),
                                        center=(round((py + pbh / 2.0) / h, 4), round((px + pbw / 2.0) / w, 4)),
                                        area_ratio=round(p_area / total_fg_pixels, 4),
                                        is_through_hole=False,
                                        aspect_ratio=round(p_aspect, 2),
                                        orientation=round(np.degrees(np.arctan2(pbh, pbw)), 1),
                                        pixel_area=int(p_area),
                                        contour=p_cnt,
                                    )
                                )

        logger.info(f"Detected {len(negative_spaces)} negative space / opening region(s).")
        return negative_spaces

    def decompose_structural_components(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        structural_edges: np.ndarray,
        negative_spaces: List[NegativeSpaceRegion],
    ) -> List[StructuralFeature]:
        """
        Decomposes the object into visually distinct structural components using
        distance transform and watershed along structural boundaries.
        """
        h, w = mask.shape[:2]
        fg_mask = (mask > 10).astype(np.uint8)
        total_fg_pixels = max(1, int(np.count_nonzero(fg_mask)))

        # 1. Non-edge foreground region
        non_edge_fg = cv2.subtract(fg_mask, structural_edges)

        # 2. Distance Transform on non-edge regions
        dist = cv2.distanceTransform(non_edge_fg, cv2.DIST_L2, 5)

        # 3. Locate Component Seeds (Local Maxima of Distance Transform)
        dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
        max_dist = dist_smooth.max()
        if max_dist <= 0:
            return []

        # Peak markers
        _, sure_fg = cv2.threshold(dist_smooth, 0.25 * max_dist, 255, 0)
        sure_fg = np.uint8(sure_fg)

        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[fg_mask == 0] = 0

        # 4. Watershed Segmentation
        cv2.watershed(rgb, markers)

        # 5. Extract Feature Descriptors for each identified component
        components: List[StructuralFeature] = []
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sobel_grad = cv2.Sobel(gray, cv2.CV_32F, 1, 1)
        sobel_grad = np.abs(sobel_grad)

        for marker_id in range(2, num_markers + 1):
            comp_mask = (markers == marker_id).astype(np.uint8)
            comp_pixels = int(np.count_nonzero(comp_mask))
            area_ratio = float(comp_pixels) / float(total_fg_pixels)

            if area_ratio < self.min_component_area_ratio:
                continue

            # Bounding box & Center
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest_cnt = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest_cnt)

            # Oriented bounding box & principal angle
            if len(largest_cnt) >= 5:
                (ocx, ocy), (ow, oh), o_angle = cv2.minAreaRect(largest_cnt)
                aspect = float(max(ow, oh)) / float(max(1.0, min(ow, oh)))
                orientation = float(o_angle)
            else:
                aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
                orientation = 0.0

            # Centroid
            M = cv2.moments(largest_cnt)
            cx = M["m10"] / M["m00"] if M["m00"] > 0 else (x + bw / 2.0)
            cy = M["m01"] / M["m00"] if M["m00"] > 0 else (y + bh / 2.0)

            # Edge strength along boundary
            comp_boundary = cv2.Canny(comp_mask * 255, 100, 200)
            edge_pts = np.where(comp_boundary > 0)
            edge_str = float(np.mean(sobel_grad[edge_pts])) / 255.0 if len(edge_pts[0]) > 0 else 0.5

            # Generic type classification based on geometric properties
            feat_type = "main_body"
            if area_ratio >= 0.40:
                feat_type = "main_body"
            elif aspect >= 3.0:
                feat_type = "elongated_region"
            elif any(cv2.pointPolygonTest(largest_cnt, (ns.center[1] * w, ns.center[0] * h), False) >= 0 for ns in negative_spaces):
                feat_type = "loop_arch"
            elif cy < h * 0.35 and bw > w * 0.4:
                feat_type = "ring_rim"
            else:
                feat_type = "protrusion"

            # Relative vertical depth cue (higher up in perspective is often top/rear)
            depth_cue = float(1.0 - (cy / float(h)))

            # Bilateral symmetry
            comp_crop = comp_mask[y:y+bh, x:x+bw]
            flipped = cv2.flip(comp_crop, 1)
            inter = np.logical_and(comp_crop, flipped).sum()
            union = np.logical_or(comp_crop, flipped).sum()
            symmetry = float(inter) / float(max(1, union))

            feat_id = f"feature_{len(components) + 1:02d}"
            components.append(
                StructuralFeature(
                    feature_id=feat_id,
                    type=feat_type,
                    bbox=(round(y / h, 4), round(x / w, 4), round((y + bh) / h, 4), round((x + bw) / w, 4)),
                    center=(round(cy / h, 4), round(cx / w, 4)),
                    area_ratio=round(area_ratio, 4),
                    orientation=round(orientation, 1),
                    aspect_ratio=round(aspect, 2),
                    edge_strength=round(float(np.clip(edge_str, 0.1, 1.0)), 3),
                    boundary_strength=round(float(np.clip(edge_str * 1.2, 0.2, 1.0)), 3),
                    depth_cue=round(depth_cue, 3),
                    symmetry=round(symmetry, 3),
                    confidence=round(min(1.0, 0.5 + area_ratio * 1.5), 3),
                )
            )

        # Sort components: main_body first, followed by descending area
        components.sort(key=lambda c: (c.type != "main_body", -c.area_ratio))
        logger.info(f"Decomposed object into {len(components)} generic structural component(s).")
        return components

    def build_relationship_graph(
        self,
        components: List[StructuralFeature],
    ) -> List[StructuralRelationship]:
        """
        Builds relative spatial & topological relationship graph between structural components.
        """
        relationships: List[StructuralRelationship] = []
        n = len(components)

        for i in range(n):
            c_a = components[i]
            for j in range(i + 1, n):
                c_b = components[j]

                # Normalized Euclidean distance
                dy = c_b.center[0] - c_a.center[0]
                dx = c_b.center[1] - c_a.center[1]
                dist = float(np.sqrt(dx**2 + dy**2))

                # Angle between centroids
                rel_angle = float(np.degrees(np.arctan2(dy, dx)))

                # Orientation alignment
                rad_a = np.radians(c_a.orientation)
                rad_b = np.radians(c_b.orientation)
                alignment = float(np.cos(rad_a - rad_b))

                # Adjacency: bounding boxes overlap or within 8% margin
                ay1, ax1, ay2, ax2 = c_a.bbox
                by1, bx1, by2, bx2 = c_b.bbox
                margin = 0.08
                adjacent = not (ax2 + margin < bx1 or bx2 + margin < ax1 or ay2 + margin < by1 or by2 + margin < ay1)

                # Containment
                containment = (ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2) or (
                    bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2
                )

                relationships.append(
                    StructuralRelationship(
                        feature_a_id=c_a.feature_id,
                        feature_b_id=c_b.feature_id,
                        distance=round(dist, 4),
                        relative_angle=round(rel_angle, 1),
                        adjacency=adjacent,
                        alignment=round(alignment, 3),
                        containment=containment,
                    )
                )

        return relationships

    def analyze(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> StructuralReferenceMap:
        """
        Executes full generic reference structure analysis.
        Returns complete StructuralReferenceMap blueprint.
        """
        rgb, bin_mask = self._standardize_inputs(image, mask)
        h, w = rgb.shape[:2]

        # 1. Structural Edge Classification
        structural_edges, edge_details = self.classify_structural_edges(rgb, bin_mask)

        # 2. Negative Space & Openings Detection
        negative_spaces = self.detect_negative_spaces(bin_mask, rgb)

        # 3. Structural Component Decomposition
        components = self.decompose_structural_components(rgb, bin_mask, structural_edges, negative_spaces)

        # 4. Relationship Graph
        relationships = self.build_relationship_graph(components)

        # 5. Overall Object Metrics
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            _, _, ow, oh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            obj_aspect = float(ow) / float(max(1, oh))
        else:
            obj_aspect = 1.0

        # Bilateral symmetry across entire silhouette
        flipped_mask = cv2.flip(bin_mask, 1)
        sym_inter = np.logical_and(bin_mask > 0, flipped_mask > 0).sum()
        sym_union = np.logical_or(bin_mask > 0, flipped_mask > 0).sum()
        symmetry_score = float(sym_inter) / float(max(1, sym_union))

        ref_map = StructuralReferenceMap(
            outer_silhouette=bin_mask,
            structural_edges=structural_edges,
            components=components,
            negative_spaces=negative_spaces,
            relationships=relationships,
            aspect_ratio=round(obj_aspect, 4),
            symmetry_score=round(symmetry_score, 4),
            metadata={
                "image_dimensions": (h, w),
                "edge_details": edge_details,
            },
        )
        logger.info(
            f"ReferenceStructureAnalyzer: Built map with {len(components)} components, "
            f"{len(negative_spaces)} negative spaces, {len(relationships)} relationships, aspect={obj_aspect:.2f}."
        )
        return ref_map
