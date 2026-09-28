import unittest
import numpy as np
from PIL import Image
import cv2

from app.services.geometry.reference_structure_analyzer import (
    ReferenceStructureAnalyzer,
    StructuralReferenceMap,
    StructuralFeature,
    NegativeSpaceRegion,
)


class TestReferenceStructureAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ReferenceStructureAnalyzer(
            min_component_area_ratio=0.01,
            min_negative_space_area=20,
        )

    def test_classify_structural_edges_vs_texture(self):
        # Create image with a strong structural boundary (sharp square)
        # and fine high-frequency surface noise (texture)
        img = np.full((200, 200, 3), 240, dtype=np.uint8)
        # Structural boundary: dark square
        cv2.rectangle(img, (40, 40), (160, 160), (30, 30, 30), thickness=3)
        # Surface texture: fine subtle dots/lines inside
        for y in range(60, 140, 6):
            cv2.line(img, (60, y), (140, y), (210, 210, 210), thickness=1)

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[35:165, 35:165] = 255

        edges, details = self.analyzer.classify_structural_edges(img, mask)
        self.assertGreater(np.count_nonzero(edges), 100)
        self.assertIn("outer_silhouette", details)
        self.assertIn("structural_internal", details)
        self.assertIn("surface_texture", details)

    def test_detect_negative_spaces_through_hole(self):
        # Create an object mask with a prominent hole (like a handle arch or washer)
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Outer body
        cv2.circle(mask, (100, 100), 70, 255, -1)
        # Internal hole (negative space)
        cv2.circle(mask, (100, 100), 25, 0, -1)

        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        # Hole matches background color (white)
        img[mask == 0] = [255, 255, 255]

        negative_spaces = self.analyzer.detect_negative_spaces(mask, img)
        self.assertGreaterEqual(len(negative_spaces), 1)

        hole = negative_spaces[0]
        self.assertTrue(hole.is_through_hole)
        self.assertGreater(hole.area_ratio, 0.03)
        # Centroid should be near (0.5, 0.5)
        self.assertAlmostEqual(hole.center[0], 0.5, delta=0.1)
        self.assertAlmostEqual(hole.center[1], 0.5, delta=0.1)

    def test_decompose_structural_components_and_graph(self):
        # Create object with two distinct connected regions: main body and an elongated protrusion
        mask = np.zeros((250, 250), dtype=np.uint8)
        # Main body
        cv2.circle(mask, (125, 140), 60, 255, -1)
        # Elongated protrusion on top (like a handle or straw)
        cv2.rectangle(mask, (115, 20), (135, 90), 255, -1)

        img = np.full((250, 250, 3), 180, dtype=np.uint8)
        # Distinct boundary between protrusion and main body
        cv2.line(img, (110, 85), (140, 85), (40, 40, 40), 2)

        ref_map = self.analyzer.analyze(img, mask)
        self.assertIsInstance(ref_map, StructuralReferenceMap)
        self.assertGreaterEqual(len(ref_map.components), 1)
        self.assertGreater(ref_map.aspect_ratio, 0.0)

        # Check relationships
        if len(ref_map.components) >= 2:
            self.assertGreater(len(ref_map.relationships), 0)
            rel = ref_map.relationships[0]
            self.assertIsNotNone(rel.distance)
            self.assertIsNotNone(rel.alignment)


if __name__ == "__main__":
    unittest.main()
