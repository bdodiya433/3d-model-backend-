import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import trimesh

from app.services.geometry.reference_structure_analyzer import (
    StructuralReferenceMap,
    StructuralFeature,
    NegativeSpaceRegion,
)

logger = logging.getLogger("app.services.geometry.structural_evaluator")


@dataclass
class StructuralEvaluationReport:
    """Detailed diagnostic report evaluating generated 3D model against reference structure."""
    silhouette_iou: float                   # [0.0, 1.0]
    silhouette_dice: float                  # [0.0, 1.0]
    component_score: float                 # [0.0, 1.0] (1.0 = perfect match of components)
    component_position_error: float        # Normalized centroid displacement
    component_scale_error: float           # Relative area discrepancy
    orientation_error: float               # Mean orientation angle error in degrees
    negative_space_score: float            # [0.0, 1.0] (1.0 = all holes/channels preserved)
    negative_space_error: float            # Fraction of reference openings filled or missed
    structural_boundary_error: float       # Edge alignment discrepancy [0.0, 1.0]
    overall_structural_confidence: float   # [0.0, 1.0] Combined structural score
    is_valid: bool                         # Passes structural quality gate
    major_failures: List[str] = field(default_factory=list)
    feature_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    difference_visualization: Optional[np.ndarray] = None


class StructuralEvaluator:
    """
    Closed-loop 3D -> 2D Projection & Structural Evaluator.
    
    Projects the generated 3D mesh back to the reference camera view and performs
    rigorous validation of:
    1. Outer silhouette fidelity
    2. Negative space preservation (holes, handle loops, openings)
    3. Structural component presence, position, scale, and orientation
    4. Internal crease edges and boundary continuity
    """

    def __init__(
        self,
        canvas_size: int = 512,
        padding_fraction: float = 0.08,
        crease_angle_degrees: float = 35.0,
        min_pass_threshold: float = 0.70,
    ):
        self.canvas_size = canvas_size
        self.padding_fraction = padding_fraction
        self.crease_angle_degrees = crease_angle_degrees
        self.min_pass_threshold = min_pass_threshold

    def project_mesh_geometry(
        self,
        mesh: trimesh.Trimesh,
        view_name: str = "front",
    ) -> Dict[str, np.ndarray]:
        """
        Projects 3D mesh faces, crease edges, and silhouette onto 2D camera plane.
        Returns:
            - "silhouette": (H, W) uint8 binary mask
            - "crease_edges": (H, W) uint8 edge map of sharp structural features
            - "depth_map": (H, W) float32 normalized depth map
        """
        sz = self.canvas_size
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return {
                "silhouette": np.zeros((sz, sz), dtype=np.uint8),
                "crease_edges": np.zeros((sz, sz), dtype=np.uint8),
                "depth_map": np.zeros((sz, sz), dtype=np.float32),
            }

        # Center mesh at origin
        verts = mesh.vertices - mesh.centroid

        # Camera coordinate mapping (Hunyuan3D: +Z up, +Y front, +X right)
        v_name = view_name.lower()
        if v_name == "front":
            u = verts[:, 0]
            v = verts[:, 2]
            depth = verts[:, 1]
        elif v_name == "back":
            u = -verts[:, 0]
            v = verts[:, 2]
            depth = -verts[:, 1]
        elif v_name == "left":
            u = -verts[:, 1]
            v = verts[:, 2]
            depth = verts[:, 0]
        elif v_name == "right":
            u = verts[:, 1]
            v = verts[:, 2]
            depth = -verts[:, 0]
        elif v_name == "top":
            u = verts[:, 0]
            v = verts[:, 1]
            depth = verts[:, 2]
        else:
            u = verts[:, 0]
            v = verts[:, 2]
            depth = verts[:, 1]

        # Standardized canvas scaling preserving aspect ratio
        span_u = float(u.max() - u.min())
        span_v = float(v.max() - v.min())
        max_span = max(span_u, span_v, 1e-6)

        scale = float(sz * (1.0 - 2.0 * self.padding_fraction)) / max_span
        center_u = float(u.max() + u.min()) / 2.0
        center_v = float(v.max() + v.min()) / 2.0
        offset = float(sz) / 2.0

        px = np.round((u - center_u) * scale + offset).astype(np.int32)
        py = np.round(sz - ((v - center_v) * scale + offset)).astype(np.int32)

        px = np.clip(px, 0, sz - 1)
        py = np.clip(py, 0, sz - 1)
        pts2d = np.stack([px, py], axis=1)

        # 1. Rasterize Solid Silhouette
        silhouette_canvas = np.zeros((sz, sz), dtype=np.uint8)
        triangles2d = pts2d[mesh.faces]
        for tri in triangles2d:
            cv2.fillConvexPoly(silhouette_canvas, tri, 255)

        # 2. Extract Crease Edges (sharp structural feature lines)
        crease_canvas = np.zeros((sz, sz), dtype=np.uint8)
        try:
            # Dihedral angle between adjacent faces
            face_angles = mesh.face_adjacency_angles
            sharp_mask = face_angles > np.radians(self.crease_angle_degrees)
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]

            for edge in sharp_edges:
                p1 = tuple(pts2d[edge[0]])
                p2 = tuple(pts2d[edge[1]])
                cv2.line(crease_canvas, p1, p2, 255, 1)
        except Exception as e:
            logger.warning(f"Crease edge extraction skipped: {e}")

        # Also add outer boundary of projected silhouette to edges
        contours, _ = cv2.findContours(silhouette_canvas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crease_canvas, contours, -1, 255, 1)

        # 3. Normalized Depth Map
        d_min, d_max = depth.min(), depth.max()
        d_span = max(d_max - d_min, 1e-6)
        depth_norm = (depth - d_min) / d_span

        return {
            "silhouette": silhouette_canvas,
            "crease_edges": crease_canvas,
            "pts2d": pts2d,
        }

    def _normalize_mask_to_canvas(self, mask: np.ndarray) -> np.ndarray:
        """Centers and normalizes binary mask to canvas preserving aspect ratio."""
        sz = self.canvas_size
        bin_mask = (mask > 10).astype(np.uint8)
        if np.count_nonzero(bin_mask) == 0:
            return np.zeros((sz, sz), dtype=np.uint8)

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((sz, sz), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cropped = bin_mask[y:y+h, x:x+w]

        max_dim = max(w, h, 1)
        avail_sz = sz * (1.0 - 2.0 * self.padding_fraction)
        scale = avail_sz / max_dim

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((sz, sz), dtype=np.uint8)
        off_x = (sz - new_w) // 2
        off_y = (sz - new_h) // 2
        canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized * 255
        return canvas

    def evaluate_negative_spaces(
        self,
        projected_silhouette: np.ndarray,
        reference_spaces: List[NegativeSpaceRegion],
    ) -> Tuple[float, float, List[str]]:
        """
        Verifies that detected reference negative spaces (holes, channels) actually
        exist in the 3D mesh projection rather than being filled into solid geometry.
        """
        sz = self.canvas_size
        if not reference_spaces:
            return 1.0, 0.0, []

        # Find internal holes in projected 3D silhouette
        proj_bin = (projected_silhouette > 10).astype(np.uint8)
        contours, hierarchy = cv2.findContours(proj_bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        proj_holes_mask = np.zeros((sz, sz), dtype=np.uint8)
        proj_holes_count = 0
        if hierarchy is not None and len(hierarchy) > 0:
            for i, h_elem in enumerate(hierarchy[0]):
                if h_elem[3] != -1:  # Internal hole
                    cv2.drawContours(proj_holes_mask, contours, i, 255, -1)
                    proj_holes_count += 1

        missed_holes = 0
        total_weight = 0.0
        weighted_errors = 0.0
        failures: List[str] = []

        for ns in reference_spaces:
            weight = max(0.2, ns.area_ratio * 5.0)
            total_weight += weight

            # Map normalized bbox to canvas coordinates
            ny1, nx1, ny2, nx2 = ns.bbox
            cy1, cx1 = int(round(ny1 * (sz - 1))), int(round(nx1 * (sz - 1)))
            cy2, cx2 = int(round(ny2 * (sz - 1))), int(round(nx2 * (sz - 1)))
            ch, cw = max(1, cy2 - cy1), max(1, cx2 - cx1)

            # Check if projected mesh left a hole in this region
            region_proj_hole = proj_holes_mask[cy1:cy2, cx1:cx2]
            hole_pixels = np.count_nonzero(region_proj_hole)
            expected_pixels = ch * cw

            # Check if projected silhouette is solid in this region
            region_solid = proj_bin[cy1:cy2, cx1:cx2]
            solid_ratio = float(np.count_nonzero(region_solid)) / float(expected_pixels)

            if ns.is_through_hole and solid_ratio > 0.85 and hole_pixels < (expected_pixels * 0.15):
                # Critical structural error: through-hole was filled into solid mass!
                missed_holes += 1
                weighted_errors += weight * 1.0
                fail_msg = (
                    f"MAJOR_NEGATIVE_SPACE_FILLED: Reference hole '{ns.region_id}' (area_ratio={ns.area_ratio:.3f}) "
                    f"at center ({ns.center[0]:.2f}, {ns.center[1]:.2f}) was filled into solid geometry."
                )
                failures.append(fail_msg)
                logger.error(fail_msg)
            elif hole_pixels > 0 or solid_ratio < 0.60:
                # Hole correctly preserved
                pass
            else:
                weighted_errors += weight * 0.4

        neg_error = float(weighted_errors / max(1e-6, total_weight))
        neg_score = float(np.clip(1.0 - neg_error, 0.0, 1.0))
        return round(neg_score, 4), round(neg_error, 4), failures

    def evaluate_structural_components(
        self,
        projected_silhouette: np.ndarray,
        projected_creases: np.ndarray,
        reference_components: List[StructuralFeature],
    ) -> Tuple[float, float, float, float, List[str], Dict[str, Dict[str, Any]]]:
        """
        Evaluates presence, position, scale, and orientation fidelity of structural components.
        """
        sz = self.canvas_size
        if not reference_components:
            return 1.0, 0.0, 0.0, 0.0, [], {}

        # Decompose projected 3D silhouette into components
        proj_bin = (projected_silhouette > 10).astype(np.uint8)
        non_edge = cv2.subtract(proj_bin, projected_creases)
        dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 5)

        failures: List[str] = []
        feature_metrics: Dict[str, Dict[str, Any]] = {}

        pos_errors = []
        scale_errors = []
        angle_errors = []

        total_fg_pixels = max(1, int(np.count_nonzero(proj_bin)))

        for feat in reference_components:
            # Map normalized center to canvas
            ref_cy = int(round(feat.center[0] * (sz - 1)))
            ref_cx = int(round(feat.center[1] * (sz - 1)))

            # Sample local 3D presence around center
            radius = max(8, int(round(np.sqrt(feat.area_ratio) * sz * 0.4)))
            y1 = max(0, ref_cy - radius)
            y2 = min(sz, ref_cy + radius)
            x1 = max(0, ref_cx - radius)
            x2 = min(sz, ref_cx + radius)

            local_patch = proj_bin[y1:y2, x1:x2]
            local_area = np.count_nonzero(local_patch)

            # Position & presence check
            if local_area == 0:
                pos_err = 1.0
                scale_err = 1.0
                angle_err = 45.0
                fail_msg = f"CRITICAL_COMPONENT_MISSING: Component '{feat.feature_id}' ({feat.type}) at center {feat.center} not found in 3D projection."
                failures.append(fail_msg)
            else:
                # Find centroid of local patch
                py, px_coords = np.where(local_patch > 0)
                mean_cy = (y1 + np.mean(py)) / sz
                mean_cx = (x1 + np.mean(px_coords)) / sz
                dy = mean_cy - feat.center[0]
                dx = mean_cx - feat.center[1]
                pos_err = float(np.sqrt(dx**2 + dy**2))

                # Scale error
                patch_ratio = float(local_area) / float(total_fg_pixels)
                scale_err = abs(patch_ratio - feat.area_ratio) / max(0.01, feat.area_ratio)

                # Orientation error (if elongated)
                angle_err = 0.0
                if feat.aspect_ratio >= 2.2 and len(py) >= 5:
                    pts = np.stack([px_coords, py], axis=1)
                    (_, _), (w_box, h_box), o_ang = cv2.minAreaRect(pts)
                    angle_err = abs(o_ang - feat.orientation)
                    if angle_err > 90.0:
                        angle_err = abs(angle_err - 180.0)

            pos_errors.append(pos_err)
            scale_errors.append(min(1.0, scale_err))
            angle_errors.append(angle_err)

            feature_metrics[feat.feature_id] = {
                "type": feat.type,
                "position_error": round(pos_err, 4),
                "scale_error": round(min(1.0, scale_err), 4),
                "orientation_error_deg": round(angle_err, 1),
                "present": local_area > 0,
            }

        mean_pos_err = float(np.mean(pos_errors)) if pos_errors else 0.0
        mean_scale_err = float(np.mean(scale_errors)) if scale_errors else 0.0
        mean_angle_err = float(np.mean(angle_errors)) if angle_errors else 0.0

        comp_score = float(np.clip(1.0 - (0.45 * mean_pos_err + 0.35 * mean_scale_err + 0.20 * (mean_angle_err / 90.0)), 0.0, 1.0))

        return round(comp_score, 4), round(mean_pos_err, 4), round(mean_scale_err, 4), round(mean_angle_err, 1), failures, feature_metrics

    def evaluate(
        self,
        mesh: trimesh.Trimesh,
        reference_map: StructuralReferenceMap,
        view_name: str = "front",
    ) -> StructuralEvaluationReport:
        """
        Executes comprehensive 3D -> 2D structural evaluation of the generated mesh
        against the reference structural map.
        """
        # 1. Project 3D Mesh
        projections = self.project_mesh_geometry(mesh, view_name=view_name)
        proj_sil = projections["silhouette"]
        proj_creases = projections["crease_edges"]

        # Standardize and normalize outer silhouettes
        p_norm = self._normalize_mask_to_canvas(proj_sil)
        r_norm = self._normalize_mask_to_canvas(reference_map.outer_silhouette)

        p_bin = p_norm > 10
        r_bin = r_norm > 10

        # 2. Silhouette IoU & Dice
        inter = int(np.logical_and(p_bin, r_bin).sum())
        union = int(np.logical_or(p_bin, r_bin).sum())
        sil_iou = float(inter) / float(max(1, union))
        sil_dice = (2.0 * float(inter)) / float(max(1, int(p_bin.sum()) + int(r_bin.sum())))

        # 3. Negative Space Evaluation
        neg_score, neg_error, neg_failures = self.evaluate_negative_spaces(
            projected_silhouette=proj_sil,
            reference_spaces=reference_map.negative_spaces,
        )

        # 4. Structural Component Evaluation
        (
            comp_score,
            pos_err,
            scale_err,
            angle_err,
            comp_failures,
            feature_metrics,
        ) = self.evaluate_structural_components(
            projected_silhouette=proj_sil,
            projected_creases=proj_creases,
            reference_components=reference_map.components,
        )

        # 5. Structural Boundary Discrepancy (Chamfer / distance transform)
        # Distance of projected crease edges from reference structural edges
        ref_edges = self._normalize_mask_to_canvas(reference_map.structural_edges)
        proj_edges_norm = self._normalize_mask_to_canvas(proj_creases)

        dist_ref = cv2.distanceTransform(cv2.bitwise_not(ref_edges), cv2.DIST_L2, 3)
        edge_pts = np.where(proj_edges_norm > 0)
        if len(edge_pts[0]) > 0:
            mean_dist = float(np.mean(dist_ref[edge_pts]))
            boundary_error = float(np.clip(mean_dist / 25.0, 0.0, 1.0))
        else:
            boundary_error = 0.5

        # 6. Combined Overall Structural Score
        overall_score = (
            0.25 * sil_iou
            + 0.30 * comp_score
            + 0.25 * neg_score
            + 0.20 * (1.0 - boundary_error)
        )
        overall_score = float(np.clip(overall_score, 0.0, 1.0))

        all_failures = neg_failures + comp_failures
        is_valid = (overall_score >= self.min_pass_threshold) and (len(all_failures) == 0)

        # 7. Difference Visualization
        diff_img = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        diff_img[np.logical_and(p_bin, r_bin)] = [35, 205, 35]    # Green = Matched
        diff_img[np.logical_and(~p_bin, r_bin)] = [220, 35, 35]   # Red = Missing
        diff_img[np.logical_and(p_bin, ~r_bin)] = [35, 110, 230]  # Blue = Excess
        # Overlay crease edges in yellow
        diff_img[proj_edges_norm > 0] = [255, 255, 0]

        report = StructuralEvaluationReport(
            silhouette_iou=round(sil_iou, 4),
            silhouette_dice=round(sil_dice, 4),
            component_score=round(comp_score, 4),
            component_position_error=round(pos_err, 4),
            component_scale_error=round(scale_err, 4),
            orientation_error=round(angle_err, 1),
            negative_space_score=round(neg_score, 4),
            negative_space_error=round(neg_error, 4),
            structural_boundary_error=round(boundary_error, 4),
            overall_structural_confidence=round(overall_score, 4),
            is_valid=is_valid,
            major_failures=all_failures,
            feature_metrics=feature_metrics,
            difference_visualization=diff_img,
        )

        logger.info(
            f"StructuralEvaluator: Overall Score={overall_score:.2f} (Valid={is_valid}). "
            f"Sil IoU={sil_iou:.2f}, Comp Score={comp_score:.2f}, Neg Space Score={neg_score:.2f}, "
            f"Failures count={len(all_failures)}."
        )
        return report
