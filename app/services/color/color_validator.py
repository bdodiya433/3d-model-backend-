import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import trimesh

from app.services.color.color_extractor import ColorExtractionResult, ColorItem
from app.services.color.color_mapper import ColorMappingResult

logger = logging.getLogger("app.services.color.validator")


@dataclass
class ValidationReport:
    """Detailed quality validation report for geometry and color integrity."""
    is_valid: bool
    geometry_valid: bool
    color_valid: bool
    mean_delta_e: float
    max_delta_e: float
    color_fidelity_score: float  # 0.0 to 1.0 (1.0 = perfect match)
    vertex_count: int
    face_count: int
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ColorValidator:
    """
    Automated pre-export validation service for verifying 3D geometry integrity,
    vertex color assignment, and perceptual color fidelity against reference imagery.
    """

    def __init__(
        self,
        max_acceptable_mean_delta_e: float = 35.0,
        min_vertices: int = 10,
        min_faces: int = 10,
    ):
        self.max_acceptable_mean_delta_e = max_acceptable_mean_delta_e
        self.min_vertices = min_vertices
        self.min_faces = min_faces

    def validate(
        self,
        mapping_result: ColorMappingResult,
        reference_results: Dict[str, ColorExtractionResult],
    ) -> ValidationReport:
        """
        Runs comprehensive validation checks on the colored 3D mesh.
        """
        mesh = mapping_result.mesh
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        # ---------------- 1. GEOMETRY VALIDATION ----------------
        geometry_valid = True
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)

        metrics["vertex_count"] = n_verts
        metrics["face_count"] = n_faces

        if n_verts < self.min_vertices:
            geometry_valid = False
            warnings.append(f"Geometry check failed: Mesh has only {n_verts} vertices (min: {self.min_vertices}).")

        if n_faces < self.min_faces:
            geometry_valid = False
            warnings.append(f"Geometry check failed: Mesh has only {n_faces} faces (min: {self.min_faces}).")

        # Check for NaN / Inf in coordinates
        if not np.all(np.isfinite(mesh.vertices)):
            geometry_valid = False
            warnings.append("Geometry check failed: Mesh contains NaN or Infinite vertex coordinates.")

        # Check bounding box sanity
        extents = mesh.extents
        metrics["extents"] = [round(float(e), 4) for e in extents]
        if np.any(extents < 1e-4):
            geometry_valid = False
            warnings.append(f"Geometry check warning: Mesh appears collapsed along an axis: {extents}.")
        elif np.any(extents > 1000.0):
            warnings.append(f"Geometry check warning: Mesh dimensions unusually large: {extents}.")

        # ---------------- 2. COLOR VALIDATION ----------------
        color_valid = True
        vcolors = mapping_result.vertex_colors

        if vcolors is None or len(vcolors) != n_verts:
            color_valid = False
            warnings.append("Color check failed: Vertex colors array missing or does not match vertex count.")
            vcolors = np.full((n_verts, 4), 200, dtype=np.uint8)

        # Check alpha channel coverage (should be 255 for opaque 3D objects)
        if np.any(vcolors[:, 3] == 0):
            warnings.append("Color check warning: Some vertices have 0 alpha (fully transparent).")

        # Check for unintended dropout: pure black (0, 0, 0)
        pure_black_ratio = float(np.count_nonzero(np.all(vcolors[:, :3] == 0, axis=1))) / float(max(1, n_verts))
        metrics["pure_black_ratio"] = round(pure_black_ratio, 4)

        # Check if reference actually contained black
        ref_has_black = False
        for r_res in reference_results.values():
            for d in r_res.dominant_colors:
                if np.max(d.rgb) < 30:
                    ref_has_black = True
                    break

        if pure_black_ratio > 0.10 and not ref_has_black:
            warnings.append(
                f"Color check warning: {pure_black_ratio * 100:.1f}% of vertices are pure black (0,0,0), "
                f"but reference image has no significant black component."
            )

        # Check for unintended dropout: pure white (255, 255, 255)
        pure_white_ratio = float(np.count_nonzero(np.all(vcolors[:, :3] == 255, axis=1))) / float(max(1, n_verts))
        metrics["pure_white_ratio"] = round(pure_white_ratio, 4)

        ref_has_white = False
        for r_res in reference_results.values():
            for d in r_res.dominant_colors:
                if np.min(d.rgb) > 230:
                    ref_has_white = True
                    break

        if pure_white_ratio > 0.15 and not ref_has_white:
            warnings.append(
                f"Color check warning: {pure_white_ratio * 100:.1f}% of vertices are pure white (255,255,255), "
                f"potential background leakage onto 3D geometry."
            )

        # ---------------- 3. COLOR FIDELITY (CIE-LAB DELTA-E) ----------------
        # Compare mapped visible colors against reference primary and dominant colors
        vcolors_rgb_f = (vcolors[:, :3].astype(np.float32) / 255.0).reshape(1, -1, 3)
        vcolors_lab_f = cv2.cvtColor(vcolors_rgb_f, cv2.COLOR_RGB2LAB)[0]  # (N, 3)

        # Collect all reference dominant LAB colors
        ref_labs: List[np.ndarray] = []
        for r_res in reference_results.values():
            for d in r_res.dominant_colors:
                ref_labs.append(np.array(d.lab, dtype=np.float32))

        if not ref_labs:
            ref_labs.append(np.array([50.0, 0.0, 0.0], dtype=np.float32))

        ref_labs_arr = np.array(ref_labs)  # (K, 3)

        # For each vertex, find Euclidean distance in LAB (Delta-E) to closest reference dominant color
        # Shape: (N, K)
        delta_es = np.linalg.norm(vcolors_lab_f[:, np.newaxis, :] - ref_labs_arr[np.newaxis, :, :], axis=2)
        min_delta_e_per_vert = np.min(delta_es, axis=1)

        mean_delta_e = float(np.mean(min_delta_e_per_vert))
        max_delta_e = float(np.max(min_delta_e_per_vert))

        # Normalize score: 0 to 1 (Delta-E < 10 is imperceptible/excellent; Delta-E > 40 is noticeable)
        color_fidelity_score = max(0.0, min(1.0, 1.0 - (mean_delta_e / 50.0)))

        metrics["mean_delta_e"] = round(mean_delta_e, 2)
        metrics["max_delta_e"] = round(max_delta_e, 2)
        metrics["color_fidelity_score"] = round(color_fidelity_score, 4)

        if mean_delta_e > self.max_acceptable_mean_delta_e:
            warnings.append(
                f"Color fidelity warning: Mean Delta-E ({mean_delta_e:.1f}) exceeds threshold ({self.max_acceptable_mean_delta_e:.1f})."
            )

        is_valid = geometry_valid and color_valid

        for w in warnings:
            logger.warning(f"Validation Note: {w}")

        logger.info(
            f"Validation result: Valid={is_valid} (Geo={geometry_valid}, Col={color_valid}), "
            f"Fidelity={color_fidelity_score:.2f}, Mean Delta-E={mean_delta_e:.2f}, Warnings={len(warnings)}."
        )

        return ValidationReport(
            is_valid=is_valid,
            geometry_valid=geometry_valid,
            color_valid=color_valid,
            mean_delta_e=round(mean_delta_e, 2),
            max_delta_e=round(max_delta_e, 2),
            color_fidelity_score=round(color_fidelity_score, 4),
            vertex_count=n_verts,
            face_count=n_faces,
            warnings=warnings,
            metrics=metrics,
        )
