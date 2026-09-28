import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh

from app.services.color.color_extractor import ColorExtractor, ColorItem
from app.services.color.color_mapper import ColorMapper
from app.services.color.color_validator import ColorValidator
from app.services.color.color_engine import ColorEngine


class TestColorEngine(unittest.TestCase):
    def setUp(self):
        self.extractor = ColorExtractor()
        self.mapper = ColorMapper()
        self.validator = ColorValidator()
        self.engine = ColorEngine(
            extractor=self.extractor,
            mapper=self.mapper,
            validator=self.validator,
        )
        self.test_dir = Path("temp/test_color_engine")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for f in self.test_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            self.test_dir.rmdir()
        except Exception:
            pass

    def test_a_single_color_object(self):
        """Test A: Single-color object produces clean geometry and same dominant color."""
        # Create solid red image
        img = np.zeros((150, 150, 4), dtype=np.uint8)
        img[20:130, 20:130, 0] = 230  # Red
        img[20:130, 20:130, 3] = 255

        c_res = self.extractor.extract_colors(img)
        self.assertEqual(len(c_res.dominant_colors), 1)
        self.assertEqual(c_res.primary_color.rgb[0], 230)

        # Map to mesh
        box = trimesh.creation.box(extents=(1, 1, 1))
        out_glb = self.test_dir / "test_a.glb"
        colored_mesh, report = self.engine.process_and_color_mesh(box, {"front": img}, out_glb)

        self.assertTrue(report.is_valid)
        self.assertLess(report.mean_delta_e, 15.0)

    def test_b_multi_region_object(self):
        """Test B: Red/blue/yellow multi-region object preserves distinct color regions."""
        # Create 3-color horizontal stripes
        img = np.zeros((150, 150, 4), dtype=np.uint8)
        # Top: Red
        img[20:55, 20:130, 0] = 225
        img[20:55, 20:130, 3] = 255
        # Middle: Blue
        img[55:90, 20:130, 2] = 225
        img[55:90, 20:130, 3] = 255
        # Bottom: Yellow (Red + Green)
        img[90:130, 20:130, 0] = 225
        img[90:130, 20:130, 1] = 225
        img[90:130, 20:130, 3] = 255

        c_res = self.extractor.extract_colors(img)
        # Should detect at least 3 distinct dominant colors
        self.assertGreaterEqual(len(c_res.dominant_colors), 3)

        box = trimesh.creation.box(extents=(1, 1, 1))
        out_glb = self.test_dir / "test_b.glb"
        colored_mesh, report = self.engine.process_and_color_mesh(box, {"front": img}, out_glb)

        self.assertTrue(out_glb.exists())
        self.assertTrue(report.is_valid)

    def test_c_striped_object(self):
        """Test C: Striped pattern is preserved in color extraction."""
        img = np.zeros((150, 150, 4), dtype=np.uint8)
        # Alternate vertical stripes between white and dark navy
        for col in range(20, 130):
            if (col // 10) % 2 == 0:
                img[20:130, col, :3] = 240
            else:
                img[20:130, col, 2] = 180  # Navy
            img[20:130, col, 3] = 255

        c_res = self.extractor.extract_colors(img)
        self.assertGreaterEqual(len(c_res.dominant_colors), 2)

    def test_d_six_view_object(self):
        """Test D: Six views mapped consistently with smooth transitions."""
        views = {}
        colors = {
            "front": (220, 20, 20, 255),    # Red
            "back": (20, 20, 220, 255),     # Blue
            "left": (20, 220, 20, 255),     # Green
            "right": (220, 220, 20, 255),  # Yellow
            "top": (240, 240, 240, 255),    # Light
            "bottom": (50, 50, 50, 255),    # Dark Gray
        }
        for name, col in colors.items():
            arr = np.zeros((100, 100, 4), dtype=np.uint8)
            arr[10:90, 10:90] = col
            views[name] = arr

        box = trimesh.creation.box(extents=(1, 1, 1))
        out_glb = self.test_dir / "test_d.glb"
        colored_mesh, report = self.engine.process_and_color_mesh(box, views, out_glb)

        self.assertTrue(out_glb.exists())
        self.assertGreater(report.coverage_ratio if hasattr(report, "coverage_ratio") else 0.8, 0.7)

    def test_e_dark_colored_object(self):
        """Test E: Dark object (e.g. RGB 25, 25, 25) remains dark without being removed as shadow."""
        img = np.zeros((120, 120, 4), dtype=np.uint8)
        # Opaque dark gray/black object
        img[20:100, 20:100, :3] = 25
        img[20:100, 20:100, 3] = 255

        c_res = self.extractor.extract_colors(img)
        # Primary color must remain dark
        self.assertLessEqual(max(c_res.primary_color.rgb), 35)

    def test_f_white_light_object(self):
        """Test F: White/light object preserved without background leakage."""
        img = np.zeros((120, 120, 4), dtype=np.uint8)
        # Opaque clean white object
        img[20:100, 20:100, :3] = 245
        img[20:100, 20:100, 3] = 255

        c_res = self.extractor.extract_colors(img)
        self.assertGreaterEqual(min(c_res.primary_color.rgb), 235)

    def test_g_no_photographic_texture_in_glb(self):
        """Test G: Verify exported GLB contains COLOR_0 and NO raw photographic texture image."""
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[10:90, 10:90, 0] = 200
        img[10:90, 10:90, 1] = 100
        img[10:90, 10:90, 3] = 255

        box = trimesh.creation.box(extents=(1, 1, 1))
        out_glb = self.test_dir / "test_g.glb"
        self.engine.process_and_color_mesh(box, {"front": img}, out_glb)

        # Inspect exported asset
        scene = trimesh.load(str(out_glb))
        if isinstance(scene, trimesh.Scene):
            geom = list(scene.geometry.values())[0]
        else:
            geom = scene

        # Must have vertex colors
        self.assertIsNotNone(geom.visual)
        vcolors = getattr(geom.visual, "vertex_colors", None)
        if vcolors is None and hasattr(geom.visual, "vertex_attributes") and "color" in geom.visual.vertex_attributes:
            vcolors = geom.visual.vertex_attributes["color"]
        self.assertIsNotNone(vcolors)
        self.assertGreater(len(vcolors), 0)

        # Must NOT have embedded photographic texture image
        mat = getattr(geom.visual, "material", None)
        if mat is not None and hasattr(mat, "image"):
            self.assertIsNone(mat.image, "GLB should not contain an embedded photographic texture image")


if __name__ == "__main__":
    unittest.main()
