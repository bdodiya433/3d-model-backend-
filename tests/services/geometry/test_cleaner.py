import unittest
import numpy as np
import trimesh

from app.services.geometry.cleaner import GeometryCleaner


class TestGeometryCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = GeometryCleaner()

    def test_remove_disconnected_islands(self):
        main_box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        speck = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
        speck.apply_translation([10.0, 10.0, 10.0])

        combined = trimesh.util.concatenate([main_box, speck])
        self.assertEqual(len(combined.split(only_watertight=False)), 2)

        cleaned = self.cleaner.remove_disconnected_islands(combined)
        components = cleaned.split(only_watertight=False)
        self.assertEqual(len(components), 1)
        self.assertEqual(len(cleaned.vertices), len(main_box.vertices))

    def test_clean_degenerate_geometry(self):
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        extra_verts = np.vstack([box.vertices, np.array([[5.0, 5.0, 5.0]])])
        dirty_mesh = trimesh.Trimesh(vertices=extra_verts, faces=box.faces, process=False)
        self.assertEqual(len(dirty_mesh.vertices), len(box.vertices) + 1)

        cleaned = self.cleaner.clean_degenerate_geometry(dirty_mesh)
        self.assertEqual(len(cleaned.vertices), len(box.vertices))

    def test_recompute_consistent_normals(self):
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        cleaned = self.cleaner.recompute_consistent_normals(box)
        self.assertTrue(np.all(np.isfinite(cleaned.vertex_normals)))
        norms = np.linalg.norm(cleaned.vertex_normals, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-3))

    def test_feature_preserving_smooth(self):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        orig_vol = sphere.volume
        smoothed = self.cleaner.feature_preserving_smooth(sphere, iterations=1)
        new_vol = smoothed.volume
        # Gentle Taubin smoothing preserves > 98% volume
        self.assertGreater(new_vol / orig_vol, 0.98)

    def test_proximity_preserves_nearby_subcomponents(self):
        """Nearby small subcomponents (buttons/eyes) should be preserved, while far noise is pruned."""
        main_box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        # Small button attached close to surface (0.05 from surface)
        button = trimesh.creation.box(extents=(0.1, 0.1, 0.05))
        button.apply_translation([1.05, 0.0, 0.0])
        # Far floating noise
        noise = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
        noise.apply_translation([8.0, 8.0, 8.0])

        combined = trimesh.util.concatenate([main_box, button, noise])
        self.assertEqual(len(combined.split(only_watertight=False)), 3)

        cleaned = self.cleaner.remove_disconnected_islands(combined)
        components = cleaned.split(only_watertight=False)
        # Should keep main_box AND nearby button, pruning only far noise
        self.assertEqual(len(components), 2)


if __name__ == "__main__":
    unittest.main()

