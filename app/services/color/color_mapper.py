import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import trimesh

from app.services.color.color_extractor import ColorExtractor, ColorExtractionResult, ColorItem

logger = logging.getLogger("app.services.color.mapper")


@dataclass
class ColorMappingResult:
    """Result of mapping extracted image colors onto a 3D mesh."""
    mesh: trimesh.Trimesh
    vertex_colors: np.ndarray      # (N, 4) uint8 RGBA
    coverage_ratio: float          # Fraction of vertices with direct view visibility
    active_views: List[str]
    dominant_mapped_colors: List[ColorItem]
    observed_views: List[str] = field(default_factory=list)
    inferred_views: List[str] = field(default_factory=list)



class ColorMapper:
    """
    Service responsible for mapping extracted reference colors onto 3D mesh surfaces.
    
    Principles:
    1. Multi-View Projection: Supports 1 view (Front) or 2-6 views (Front, Back, Left, Right, Top, Bottom).
    2. Confidence Weighting: Uses surface normal dot view direction (max(0, n · v_dir)^p).
    3. Perceptual Seam Blending: Blends overlapping view projections in float32 CIE-LAB color space
       to eliminate harsh color boundaries.
    4. Occlusion & Hidden Surface Propagation: For unseen/hidden surfaces (especially in single-image mode),
       propagates neighboring visible colors smoothly across mesh topology rather than inventing arbitrary colors.
    5. Standard glTF Vertex Colors: Exports pure vertex colors (COLOR_0) and PBR materials.
       STRICT: Does NOT embed the raw reference photograph as a texture image.
    """

    # View camera forward vectors in Hunyuan3D coordinate space
    # (+Z is Top, -Z is Bottom, +Y is Front, -Y is Back, -X is Left, +X is Right)
    VIEW_VECTORS: Dict[str, np.ndarray] = {
        "front":  np.array([0.0,  1.0,  0.0], dtype=np.float32),
        "back":   np.array([0.0, -1.0,  0.0], dtype=np.float32),
        "left":   np.array([-1.0, 0.0,  0.0], dtype=np.float32),
        "right":  np.array([1.0,  0.0,  0.0], dtype=np.float32),
        "top":    np.array([0.0,  0.0,  1.0], dtype=np.float32),
        "bottom": np.array([0.0,  0.0, -1.0], dtype=np.float32),
    }

    def __init__(
        self,
        normal_power: float = 1.6,
        subdivide_threshold: int = 35000,
        enable_subdivision: bool = True,
    ):
        self.normal_power = normal_power
        self.subdivide_threshold = subdivide_threshold
        self.enable_subdivision = enable_subdivision

    def _project_coordinates(
        self,
        view_name: str,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps normalized 3D mesh coordinates (vx, vy, vz in [0, 1]) to 2D image coordinates (u, v in [0, 1]).
        """
        if view_name == "front":
            u = vx
            v = vz
        elif view_name == "back":
            u = 1.0 - vx
            v = vz
        elif view_name == "left":
            u = vy
            v = vz
        elif view_name == "right":
            u = 1.0 - vy
            v = vz
        elif view_name == "top":
            u = vx
            v = vy
        elif view_name == "bottom":
            u = vx
            v = 1.0 - vy
        else:
            # Default to front projection
            u = vx
            v = vz

        return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)

    def _sample_from_colormap(
        self,
        color_map: np.ndarray,
        object_mask: np.ndarray,
        u_arr: np.ndarray,
        v_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples RGB colors from a 2D color map and validates against object mask.
        Returns:
            sampled_rgb: (N, 3) uint8
            valid_mask: (N,) bool (True if inside object mask)
        """
        h, w = color_map.shape[:2]
        u_px = np.clip(np.round(u_arr * (w - 1)), 0, w - 1).astype(int)
        # In image space, v=1.0 is top (row 0), v=0.0 is bottom (row H-1)
        v_px = np.clip(np.round((1.0 - v_arr) * (h - 1)), 0, h - 1).astype(int)

        sampled_rgb = color_map[v_px, u_px]
        is_in_mask = object_mask[v_px, u_px]

        return sampled_rgb, is_in_mask

    def propagate_hidden_colors(
        self,
        mesh: trimesh.Trimesh,
        colors_lab: np.ndarray,
        valid_mask: np.ndarray,
        fallback_lab: np.ndarray,
        iterations: int = 8,
    ) -> np.ndarray:
        """
        Smoothly diffuses visible colors across mesh topology for hidden/occluded surfaces
        (e.g., backside of a single-view reconstruction).
        Uses vertex adjacency graph to propagate neighboring colors.
        """
        n_verts = len(mesh.vertices)
        if np.all(valid_mask):
            return colors_lab

        # Build adjacency list
        edges = mesh.edges_unique
        adj: List[List[int]] = [[] for _ in range(n_verts)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        filled_lab = colors_lab.copy()
        current_valid = valid_mask.copy()

        # Step-by-step neighbor diffusion
        for _ in range(iterations):
            unfilled_indices = np.argwhere(~current_valid).flatten()
            if len(unfilled_indices) == 0:
                break

            updates: Dict[int, np.ndarray] = {}
            for idx in unfilled_indices:
                neighbors = adj[idx]
                valid_neighbors = [nb for nb in neighbors if current_valid[nb]]
                if valid_neighbors:
                    avg_lab = np.mean(filled_lab[valid_neighbors], axis=0)
                    updates[idx] = avg_lab

            if not updates:
                break

            for idx, val in updates.items():
                filled_lab[idx] = val
                current_valid[idx] = True

        # Any remaining disconnected vertices get the fallback primary color
        still_unfilled = ~current_valid
        if np.any(still_unfilled):
            filled_lab[still_unfilled] = fallback_lab

        return filled_lab

    def map_colors_to_mesh(
        self,
        mesh: trimesh.Trimesh,
        view_color_results: Dict[str, ColorExtractionResult],
    ) -> ColorMappingResult:
        """
        Maps reference-derived colors onto 3D mesh vertices.
        """
        if not view_color_results:
            raise ValueError("map_colors_to_mesh requires at least one view ColorExtractionResult.")

        active_views = [v for v in self.VIEW_VECTORS.keys() if v in view_color_results]
        if not active_views:
            active_views = list(view_color_results.keys())

        logger.info(f"ColorMapper: Mapping colors from {len(active_views)} view(s): {active_views} onto mesh.")

        # Subdivide mesh if density is too coarse to capture sharp color transitions
        mesh_copy = mesh.copy()
        if self.enable_subdivision and len(mesh_copy.vertices) < self.subdivide_threshold:
            try:
                mesh_copy = mesh_copy.subdivide()
                logger.info(f"Subdivided mesh to {len(mesh_copy.vertices)} vertices for clean color boundary resolution.")
            except Exception as e:
                logger.warning(f"Mesh subdivision skipped: {e}")

        n_verts = len(mesh_copy.vertices)
        coords = mesh_copy.vertices.copy()
        normals = mesh_copy.vertex_normals.copy()

        # Compute normalized bounding box coordinates in [0, 1]
        bounds = mesh_copy.bounds
        min_b, max_b = bounds[0], bounds[1]
        dim = np.maximum(max_b - min_b, 1e-6)

        vx = (coords[:, 0] - min_b[0]) / dim[0]
        vy = (coords[:, 1] - min_b[1]) / dim[1]
        vz = (coords[:, 2] - min_b[2]) / dim[2]

        # Containers for multi-view scoring and color blending in float32 CIE-LAB space
        # shape: (n_verts, n_views)
        n_views = len(active_views)
        view_weights = np.zeros((n_verts, n_views), dtype=np.float32)
        view_lab_colors = np.zeros((n_verts, n_views, 3), dtype=np.float32)

        # Primary fallback color from the primary view (usually 'front' or first available)
        primary_key = "front" if "front" in view_color_results else active_views[0]
        primary_res = view_color_results[primary_key]
        fallback_lab = np.array(primary_res.primary_color.lab, dtype=np.float32)

        # Evaluate each camera view
        for v_idx, v_name in enumerate(active_views):
            c_res = view_color_results[v_name]
            v_dir = self.VIEW_VECTORS.get(v_name, np.array([0.0, 1.0, 0.0], dtype=np.float32))

            # Dot product of vertex normal with camera direction
            # Shape: (n_verts,)
            dot_prod = np.dot(normals, v_dir)
            # Only surfaces facing the camera are visible
            cos_theta = np.maximum(0.0, dot_prod)

            # Project coordinates
            u_proj, v_proj = self._project_coordinates(v_name, vx, vy, vz)

            # Sample from the clean 2D color map
            sampled_rgb, is_in_mask = self._sample_from_colormap(
                color_map=c_res.color_map,
                object_mask=c_res.object_mask,
                u_arr=u_proj,
                v_arr=v_proj,
            )

            # Convert sampled RGB to float32 CIE-LAB
            sampled_rgb_f = (sampled_rgb.astype(np.float32) / 255.0).reshape(1, -1, 3)
            sampled_lab_f = cv2.cvtColor(sampled_rgb_f, cv2.COLOR_RGB2LAB)[0]

            # Compute visibility confidence weight
            # High weight when looking straight at camera; zero when angled away or outside mask
            raw_weight = (cos_theta ** self.normal_power) * is_in_mask.astype(np.float32)
            view_weights[:, v_idx] = raw_weight
            view_lab_colors[:, v_idx] = sampled_lab_f

        # Multi-view confidence blending in CIE-LAB color space
        weight_sum = np.sum(view_weights, axis=1, keepdims=True)  # (n_verts, 1)
        has_direct_view = (weight_sum[:, 0] > 0.02)
        coverage_ratio = float(np.count_nonzero(has_direct_view)) / float(n_verts)

        blended_lab = np.zeros((n_verts, 3), dtype=np.float32)

        # For directly visible vertices: weighted average in CIE-LAB space
        norm_weights = np.where(weight_sum > 1e-6, view_weights / np.maximum(weight_sum, 1e-6), 0.0)
        for c in range(3):
            blended_lab[:, c] = np.sum(norm_weights * view_lab_colors[:, :, c], axis=1)

        # For occluded / hidden surfaces (e.g. back side of single image):
        # Propagate colors smoothly across mesh topology
        logger.info(f"Direct view color coverage: {coverage_ratio * 100:.1f}%. Propagating hidden surfaces...")
        final_lab = self.propagate_hidden_colors(
            mesh=mesh_copy,
            colors_lab=blended_lab,
            valid_mask=has_direct_view,
            fallback_lab=fallback_lab,
            iterations=12,
        )

        # Convert final CIE-LAB back to RGB uint8
        final_lab_f = final_lab.reshape(1, -1, 3)
        final_rgb_f = cv2.cvtColor(final_lab_f, cv2.COLOR_LAB2RGB)[0]
        final_rgb = np.clip(np.round(final_rgb_f * 255.0), 0, 255).astype(np.uint8)

        # Create RGBA vertex colors array (Alpha = 255)
        vertex_colors = np.full((n_verts, 4), 255, dtype=np.uint8)
        vertex_colors[:, :3] = final_rgb

        # Assign to mesh visual with standard neutral PBR material
        mesh_visual = trimesh.visual.color.ColorVisuals(
            mesh=mesh_copy,
            vertex_colors=vertex_colors,
        )
        try:
            mesh_visual.material = trimesh.visual.material.PBRMaterial(
                name="PBR_Material",
                metallicFactor=0.1,
                roughnessFactor=0.4,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                doubleSided=True,
            )
        except Exception as e:
            logger.debug(f"PBR material attachment note: {e}")

        mesh_copy.visual = mesh_visual

        # Extract mapped dominant colors for telemetry / diagnostics
        unique_colors = np.unique(final_rgb // 16 * 16, axis=0)  # Quantize to 16-level buckets
        mapped_dominants: List[ColorItem] = []
        for uc in unique_colors[:5]:
            mapped_dominants.append(ColorItem.from_rgb(uc[0], uc[1], uc[2]))

        # Determine observed vs inferred canonical views (Section 17)
        observed_views = [v.lower() for v in active_views if v.lower() in self.VIEW_VECTORS]
        inferred_views = [v for v in self.VIEW_VECTORS if v not in observed_views]

        logger.info(
            f"ColorMapper finished: {n_verts} vertices colored. "
            f"Coverage: {coverage_ratio * 100:.1f}%. Mapped to standard COLOR_0. "
            f"Observed: {observed_views}, Inferred: {inferred_views}."
        )

        return ColorMappingResult(
            mesh=mesh_copy,
            vertex_colors=vertex_colors,
            coverage_ratio=round(coverage_ratio, 4),
            active_views=active_views,
            dominant_mapped_colors=mapped_dominants,
            observed_views=observed_views,
            inferred_views=inferred_views,
        )

    def fuse_color_maps(
        self,
        user_color_map: np.ndarray,
        user_mask: np.ndarray,
        ref_color_map: np.ndarray,
        ref_mask: np.ndarray,
        user_weight: float = 0.60,
        ref_weight: float = 0.40,
    ) -> np.ndarray:
        """
        Fuses user color map and verified external reference color map with lighting decoupling:
        1. Resizes reference color map to match user color map resolution.
        2. Converts both to float32 CIE-LAB.
        3. Separates lighting (L*) from chromaticity (a*, b*).
        4. Fuses chromaticity: (a*, b*) = w_u * (a*, b*)_user + w_r * (a*, b*)_ref.
        5. In shadowed regions (L*_user < 28), uses reference lighting L*_ref to restore detail.
        6. In well-lit regions, preserves user lighting.
        7. Converts back to uint8 RGB.
        """
        h, w = user_color_map.shape[:2]
        ref_resized = cv2.resize(ref_color_map, (w, h), interpolation=cv2.INTER_LINEAR)
        ref_mask_resized = cv2.resize(ref_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        u_rgb_f = user_color_map.astype(np.float32) / 255.0
        r_rgb_f = ref_resized.astype(np.float32) / 255.0

        u_lab = cv2.cvtColor(u_rgb_f, cv2.COLOR_RGB2LAB)
        r_lab = cv2.cvtColor(r_rgb_f, cv2.COLOR_RGB2LAB)

        fused_lab = u_lab.copy()

        # Both masks active
        valid_both = np.logical_and(user_mask, ref_mask_resized)

        # Fused chromaticity (a*, b*)
        fused_lab[valid_both, 1] = (
            user_weight * u_lab[valid_both, 1] + ref_weight * r_lab[valid_both, 1]
        )
        fused_lab[valid_both, 2] = (
            user_weight * u_lab[valid_both, 2] + ref_weight * r_lab[valid_both, 2]
        )

        # Lighting handling: If user region is in heavy shadow (L* < 28) and reference has better lighting,
        # lift shadow gently towards reference luminance
        shadow_mask = np.logical_and(valid_both, u_lab[..., 0] < 28.0)
        fused_lab[shadow_mask, 0] = (
            0.5 * u_lab[shadow_mask, 0] + 0.5 * np.maximum(u_lab[shadow_mask, 0], r_lab[shadow_mask, 0])
        )

        fused_rgb_f = cv2.cvtColor(fused_lab, cv2.COLOR_LAB2RGB)
        fused_rgb = np.clip(np.round(fused_rgb_f * 255.0), 0, 255).astype(np.uint8)
        return fused_rgb
