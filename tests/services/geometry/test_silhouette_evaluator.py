import unittest
import numpy as np
import trimesh
import cv2

from app.services.geometry.silhouette_evaluator import SilhouetteEvaluator


class TestSilhouetteEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = SilhouetteEvaluator(raster_size=128)

    def test_project_canonical_views(self):
        cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        projections = self.evaluator.project_all_canonical_views(cube, canvas_size=128)

        self.assertEqual(len(projections), 6)
        for view in ("front", "back", "left", "right", "top", "bottom"):
            self.assertIn(view, projections)
            mask = projections[view]
            self.assertEqual(mask.shape, (128, 128))
            # Cube projected from canonical orthogonal views should have significant area in center
            self.assertGreater(np.count_nonzero(mask), 1000)

    def test_evaluate_view_silhouette_perfect_match(self):
        # Create identical mask
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:98, 30:98] = 255

        metrics = self.evaluator.evaluate_view_silhouette(mask, mask, view_name="front")
        self.assertAlmostEqual(metrics.iou, 1.0, delta=0.01)
        self.assertAlmostEqual(metrics.dice, 1.0, delta=0.01)
        self.assertAlmostEqual(metrics.missing_ratio, 0.0, delta=0.01)
        self.assertAlmostEqual(metrics.excess_ratio, 0.0, delta=0.01)
        self.assertIsNotNone(metrics.difference_mask)

    def test_evaluate_mesh_multi_view(self):
        cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        front_proj = self.evaluator.project_view(cube, "front", canvas_size=128)

        report = self.evaluator.evaluate_mesh(cube, reference_masks={"front": front_proj})
        self.assertGreater(report.overall_silhouette_similarity, 0.90)
        self.assertIn("front", report.per_view_metrics)
        self.assertEqual(len(report.canonical_projections), 6)


if __name__ == "__main__":
    unittest.main()
