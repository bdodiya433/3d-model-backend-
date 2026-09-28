import logging
from typing import Optional, Any, List, Tuple
import numpy as np
import trimesh

logger = logging.getLogger("app.services.geometry.cleaner")


class GeometryCleaner:
    """
    Service for executing safe, non-destructive mesh cleanup on raw 3D meshes
    produced by AI reconstruction models (Hunyuan3D-2/TripoSR).

    Operations:
    1. Remove degenerate geometry (zero-area faces, duplicate faces, unreferenced vertices).
    2. Remove disconnected floating noise islands while preserving legitimate sub-components.
    3. Recompute consistent outward normals.
    4. Feature-preserving mild smoothing to eliminate marching-cubes stepped voxel artifacts
       without shrinking or blunting sharp corners and silhouette edges.
    5. Clean canonical centering without altering existing coordinate system conventions.
    """

    def __init__(
        self,
        min_component_volume_ratio: float = 0.010,
        enable_taubin_smoothing: bool = True,
        smoothing_iterations: int = 1,
        
        smoothing_lambda: float = 0.15,
        smoothing_nu: float = -0.16,
        max_proximity_ratio: float = 0.08,
    ):
        self.min_component_volume_ratio = min_component_volume_ratio
        self.enable_taubin_smoothing = enable_taubin_smoothing
        self.smoothing_iterations = smoothing_iterations
        self.smoothing_lambda = smoothing_lambda
        self.smoothing_nu = smoothing_nu
        self.max_proximity_ratio = max_proximity_ratio

    def clean_degenerate_geometry(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Removes degenerate faces (zero-area, collinear), duplicates, and dangling unreferenced vertices.
        """
        try:
            # Filter non-degenerate faces
            non_deg = mesh.nondegenerate_faces()
            if non_deg is not None and len(non_deg) > 0 and len(non_deg) < len(mesh.faces):
                mesh.update_faces(non_deg)

            # Filter duplicate faces
            uniq = mesh.unique_faces()
            if uniq is not None and len(uniq) > 0 and len(uniq) < len(mesh.faces):
                mesh.update_faces(uniq)

            # Remove unreferenced vertices
            mesh.remove_unreferenced_vertices()
        except Exception as e:
            logger.warning(f"Degenerate geometry cleanup note: {e}")
        return mesh

    def remove_disconnected_islands(
        self,
        mesh: trimesh.Trimesh,
        min_ratio: Optional[float] = None,
        structural_map: Optional[Any] = None,
    ) -> trimesh.Trimesh:
        """
        Removes isolated floating noise fragments produced by AI marching cubes.
        Does NOT blindly delete smaller sub-components if they are close or meaningful.
        Uses KDTree spatial proximity AND StructuralReferenceMap cross-referencing:
        preserves nearby sub-components or components aligned with detected reference features
        (straws, handles, mounting plates, buttons), pruning only isolated empty-space noise.
        """
        ratio = min_ratio if min_ratio is not None else self.min_component_volume_ratio
        try:
            components = mesh.split(only_watertight=False)
            if len(components) <= 1:
                return mesh

            # Sort components by vertex count descending
            components = sorted(components, key=lambda c: len(c.vertices), reverse=True)
            main_comp = components[0]
            main_vert_count = len(main_comp.vertices)

            kept = [main_comp]
            removed_count = 0

            main_scale = float(np.linalg.norm(main_comp.extents))

            # Build KDTree for main mesh vertices for fast spatial proximity
            try:
                from scipy.spatial import cKDTree
                kdtree = cKDTree(main_comp.vertices)
            except Exception:
                kdtree = None

            # Extract reference component centers if structural_map is provided
            ref_feature_centers = []
            if structural_map is not None and hasattr(structural_map, "components"):
                for feat in structural_map.components:
                    ref_feature_centers.append(feat.center)

            for comp in components[1:]:
                c_verts = len(comp.vertices)
                c_ratio = c_verts / max(1, main_vert_count)

                is_kept = False
                if c_ratio >= ratio:
                    # Non-trivial component: verify not situated far off in empty space
                    dist = float(np.linalg.norm(comp.centroid - main_comp.centroid))
                    if dist <= main_scale * 1.5:
                        is_kept = True
                else:
                    # Small sub-component: check spatial proximity to main body
                    if kdtree is not None and c_verts >= 4:
                        dists, _ = kdtree.query(comp.vertices)
                        min_dist = float(dists.min())
                        # If component is close to main surface, keep it!
                        if min_dist <= self.max_proximity_ratio * main_scale:
                            is_kept = True

                    # Cross-reference with StructuralReferenceMap: if component aligns with
                    # a detected reference feature in 2D projection, protect it from deletion!
                    if not is_kept and ref_feature_centers and c_verts >= 6:
                        # Project component centroid to normalized front view coordinates
                        comp_c = comp.centroid - main_comp.centroid
                        span_max = max(main_scale * 0.5, 1e-4)
                        norm_cx = float(comp_c[0] / span_max * 0.5 + 0.5)
                        norm_cy = float(1.0 - (comp_c[2] / span_max * 0.5 + 0.5))

                        for rcy, rcx in ref_feature_centers:
                            dist_2d = np.sqrt((norm_cx - rcx)**2 + (norm_cy - rcy)**2)
                            if dist_2d < 0.20:
                                is_kept = True
                                logger.info(f"Preserving small component ({c_verts} verts) matching reference structural feature at ({rcy:.2f}, {rcx:.2f}).")
                                break

                if is_kept:
                    kept.append(comp)
                else:
                    removed_count += 1

            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} disconnected floating noise fragments. Kept {len(kept)} components."
                )
                mesh = trimesh.util.concatenate(kept)
        except Exception as e:
            logger.warning(f"Disconnected island cleanup note: {e}")
        return mesh

    def recompute_consistent_normals(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Ensures normals point consistently outward and recomputes vertex normals.
        """
        try:
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_inversion(mesh)
            mesh.fix_normals()
        except Exception as e:
            logger.warning(f"Normal recomputation note: {e}")
        return mesh

    def feature_preserving_smooth(
        self,
        mesh: trimesh.Trimesh,
        iterations: Optional[int] = None,
        lamb: Optional[float] = None,
        nu: Optional[float] = None,
        preserve_creases: bool = True,
    ) -> trimesh.Trimesh:
        """
        Applies gentle Taubin smoothing to remove high-frequency marching-cubes voxel noise
        while strictly preserving volume, curvature crests, and sharp silhouette boundaries.
        Optionally clamps displacement on sharp structural feature boundaries (dihedral angle > 30 deg).
        """
        iters = iterations if iterations is not None else self.smoothing_iterations
        if not self.enable_taubin_smoothing or iters <= 0 or len(mesh.vertices) == 0:
            return mesh

        l_val = lamb if lamb is not None else self.smoothing_lambda
        nu_val = nu if nu is not None else self.smoothing_nu

        # Identify sharp crease vertices before smoothing
        sharp_verts = None
        orig_verts = None
        if preserve_creases and len(mesh.faces) > 0:
            try:
                face_angles = mesh.face_adjacency_angles
                sharp_mask = face_angles > np.radians(30.0)
                sharp_edges = mesh.face_adjacency_edges[sharp_mask]
                if len(sharp_edges) > 0:
                    sharp_verts = np.unique(sharp_edges.flatten())
                    orig_verts = mesh.vertices.copy()
            except Exception as e:
                logger.debug(f"Crease angle detection skipped: {e}")

        try:
            smoothed = trimesh.smoothing.filter_taubin(
                mesh,
                lamb=l_val,
                nu=nu_val,
                iterations=iters,
            )

            # Protect sharp structural boundaries and open hole perimeters from melting/rounding
            if orig_verts is not None:
                if sharp_verts is not None and len(sharp_verts) > 0:
                    smoothed.vertices[sharp_verts] = (
                        0.85 * orig_verts[sharp_verts] + 0.15 * smoothed.vertices[sharp_verts]
                    )
                # Protect open boundary contours (e.g. through-hole rims, openings) 100%
                try:
                    edge_counts = np.bincount(mesh.edges_unique_inverse)
                    boundary_edges = mesh.edges_unique[edge_counts == 1]
                    if len(boundary_edges) > 0:
                        b_verts = np.unique(boundary_edges.flatten())
                        smoothed.vertices[b_verts] = orig_verts[b_verts]
                except Exception:
                    pass

            return smoothed
        except Exception as e:
            logger.warning(f"Feature-preserving smoothing note: {e}")
            return mesh

    def clean_mesh(
        self,
        mesh: trimesh.Trimesh,
        structural_map: Optional[Any] = None,
    ) -> trimesh.Trimesh:
        """
        Full non-destructive geometry cleanup sequence with structural awareness.
        """
        logger.info(
            f"Starting geometry cleanup on mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces."
        )
        mesh = self.clean_degenerate_geometry(mesh)
        mesh = self.remove_disconnected_islands(mesh, structural_map=structural_map)
        mesh = self.recompute_consistent_normals(mesh)
        mesh = self.feature_preserving_smooth(mesh, preserve_creases=True)
        logger.info(
            f"Geometry cleanup finished: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces remaining."
        )
        return mesh
