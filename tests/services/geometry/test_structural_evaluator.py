import unittest
import numpy as np
import trimesh
import cv2

from app.services.geometry.reference_structure_analyzer import (
    StructuralReferenceMap,
    StructuralFeature,
    NegativeSpaceRegion,
)
from app.services.geometry.structural_evaluator import (
    StructuralEvaluator,
    StructuralEvaluationReport,
)


class TestStructuralEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = StructuralEvaluator(
            canvas_size=256,
            min_pass_threshold=0.65,
        )

    def test_project_mesh_geometry(self):
        # Create a torus (mesh with an internal hole)
        torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
        projections = self.evaluator.project_mesh_geometry(torus, view_name="top")

        self.assertIn("silhouette", projections)
        self.assertIn("crease_edges", projections)
        sil = projections["silhouette"]
        self.assertEqual(sil.shape, (256, 256))
        self.assertGreater(np.count_nonzero(sil), 1000)

        # Torus projected from top should have an internal hole in the center
        center_pixel = sil[128, 128]
        self.assertEqual(center_pixel, 0, "Torus center should be empty negative space")

    def test_detect_filled_negative_space_failure(self):
        # Create a reference map that specifies a prominent through-hole in the center
        ref_sil = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(ref_sil, (128, 128), 90, 255, -1)
        cv2.circle(ref_sil, (128, 128), 35, 0, -1)  # Hole in center

        hole_region = NegativeSpaceRegion(
            region_id="center_hole_01",
            bbox=(0.36, 0.36, 0.64, 0.64),
            center=(0.50, 0.50),
            area_ratio=0.15,
            is_through_hole=True,
            aspect_ratio=1.0,
            orientation=0.0,
            pixel_area=3800,
        )

        main_feat = StructuralFeature(
            feature_id="body_01",
            type="main_body",
            bbox=(0.15, 0.15, 0.85, 0.85),
            center=(0.50, 0.50),
            area_ratio=0.85,
            orientation=0.0,
            aspect_ratio=1.0,
            edge_strength=0.8,
            boundary_strength=0.9,
            depth_cue=0.5,
            symmetry=1.0,
            confidence=0.95,
        )

        ref_map = StructuralReferenceMap(
            outer_silhouette=ref_sil,
            structural_edges=cv2.Canny(ref_sil, 100, 200),
            components=[main_feat],
            negative_spaces=[hole_region],
            relationships=[],
            aspect_ratio=1.0,
            symmetry_score=1.0,
        )

        # Now evaluate against a SOLID cylinder (where the hole was erroneously filled!)
        solid_cylinder = trimesh.creation.cylinder(radius=1.0, height=0.5, sections=32)
        report = self.evaluator.evaluate(solid_cylinder, ref_map, view_name="top")

        self.assertIsInstance(report, StructuralEvaluationReport)
        # Should flag a MAJOR failure because the hole was filled into solid geometry
        self.assertFalse(report.is_valid, "Filled negative space must fail quality gate")
        self.assertGreater(len(report.major_failures), 0)
        self.assertTrue(any("MAJOR_NEGATIVE_SPACE_FILLED" in f for f in report.major_failures))

    def test_passing_evaluation_with_preserved_structure(self):
        # Create a torus mesh and matching reference map with hole
        torus = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
        proj = self.evaluator.project_mesh_geometry(torus, view_name="top")
        sil = proj["silhouette"]

        hole_region = NegativeSpaceRegion(
            region_id="torus_hole_01",
            bbox=(0.35, 0.35, 0.65, 0.65),
            center=(0.50, 0.50),
            area_ratio=0.18,
            is_through_hole=True,
            aspect_ratio=1.0,
            orientation=0.0,
            pixel_area=4000,
        )

        main_feat = StructuralFeature(
            feature_id="torus_ring",
            type="ring_rim",
            bbox=(0.10, 0.10, 0.90, 0.90),
            center=(0.50, 0.50),
            area_ratio=0.82,
            orientation=0.0,
            aspect_ratio=1.0,
            edge_strength=0.8,
            boundary_strength=0.9,
            depth_cue=0.5,
            symmetry=1.0,
            confidence=0.95,
        )

        ref_map = StructuralReferenceMap(
            outer_silhouette=sil,
            structural_edges=proj["crease_edges"],
            components=[main_feat],
            negative_spaces=[hole_region],
            relationships=[],
            aspect_ratio=1.0,
            symmetry_score=1.0,
        )

        report = self.evaluator.evaluate(torus, ref_map, view_name="top")
        self.assertGreaterEqual(report.overall_structural_confidence, 0.70)
        self.assertAlmostEqual(report.silhouette_iou, 1.0, delta=0.05)
        self.assertEqual(len(report.major_failures), 0)
        self.assertTrue(report.is_valid)


if __name__ == "__main__":
    unittest.main()
