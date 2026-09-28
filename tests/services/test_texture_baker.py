import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import trimesh

from app.services.texture.baker import MultiViewTextureBaker


class TestMultiViewTextureBaker(unittest.TestCase):
    def setUp(self):
        self.baker = MultiViewTextureBaker()
        self.temp_dir = Path("temp/test_baker")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Create a simple test mesh (cylinder or box)
        self.test_mesh = trimesh.creation.box(extents=(1.0, 1.0, 2.0))
        self.test_glb_path = self.temp_dir / "test_box.glb"
        self.test_mesh.export(str(self.test_glb_path))

        # Create dummy colored view images
        self.view_images = {}
        colors = {
            "front": (255, 0, 0, 255),    # Red
            "back": (0, 0, 255, 255),     # Blue
            "left": (0, 255, 0, 255),     # Green
            "right": (255, 255, 0, 255),  # Yellow
            "top": (255, 255, 255, 255),  # White
            "bottom": (50, 50, 50, 255)   # Dark Gray
        }
        for name, col in colors.items():
            img = Image.new("RGBA", (256, 256), color=col)
            p = self.temp_dir / f"view_{name}.png"
            img.save(p)
            self.view_images[name] = p

    def tearDown(self):
        # Cleanup files in temp_dir
        for f in self.temp_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            self.temp_dir.rmdir()
        except Exception:
            pass

    def test_baker_creates_textured_glb_and_blend(self):
        out_glb = self.temp_dir / "out.glb"
        out_blend = self.temp_dir / "out.blend"

        res = self.baker.bake_texture(
            mesh_glb_path=self.test_glb_path,
            view_images=self.view_images,
            output_glb_path=out_glb,
            output_blend_path=out_blend,
            texture_resolution=512
        )

        self.assertTrue(out_glb.exists(), "Output GLB was not created")
        self.assertTrue(out_blend.exists(), "Output .blend was not created")
        self.assertGreater(out_glb.stat().st_size, 5000, "GLB size is too small")

        # Verify mesh has texture visual
        scene = trimesh.load(str(out_glb))
        if isinstance(scene, trimesh.Scene):
            geom = list(scene.geometry.values())[0]
        else:
            geom = scene

        self.assertIsNotNone(geom.visual)


if __name__ == "__main__":
    unittest.main()
