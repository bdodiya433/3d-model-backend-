import unittest
import numpy as np
from PIL import Image
from app.services.image import ImagePreprocessor, ImageProcessingError, ObjectDetectionError


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor()

    def create_dummy_opaque_image(self, width=200, height=100) -> Image.Image:
        """Creates a simple RGB image with a red box on black background."""
        img = Image.new("RGB", (width, height), (0, 0, 0))
        # Draw red box in middle
        for x in range(50, 150):
            for y in range(25, 75):
                img.putpixel((x, y), (255, 0, 0))
        return img

    def create_dummy_transparent_image(self, width=200, height=100) -> Image.Image:
        """Creates an RGBA image with a transparent background and a solid blue box."""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        x_start, x_end = int(width * 0.3), int(width * 0.7)
        y_start, y_end = int(height * 0.3), int(height * 0.7)
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                img.putpixel((x, y), (0, 0, 255, 255))
        return img

    def test_load_image(self):
        dummy = self.create_dummy_opaque_image()
        loaded = self.preprocessor.load_image(dummy)
        self.assertEqual(loaded.mode, "RGBA")
        self.assertEqual(loaded.size, (200, 100))

    def test_has_transparency_opaque(self):
        dummy_opaque = self.create_dummy_opaque_image().convert("RGBA")
        self.assertFalse(self.preprocessor.has_transparency(dummy_opaque))

    def test_has_transparency_transparent(self):
        dummy_trans = self.create_dummy_transparent_image()
        self.assertTrue(self.preprocessor.has_transparency(dummy_trans))

    def test_extract_alpha_mask(self):
        dummy = self.create_dummy_transparent_image(width=100, height=100)
        mask = self.preprocessor.extract_alpha_mask(dummy)
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)
        # Center should be 255, top-left should be 0
        self.assertEqual(mask[50, 50], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_compute_bounding_box_single_object(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255
        bbox = self.preprocessor.compute_bounding_box(mask)
        self.assertIsNotNone(bbox)
        xmin, ymin, xmax, ymax = bbox
        self.assertEqual((xmin, ymin, xmax, ymax), (30, 20, 70, 80))

    def test_compute_bounding_box_multiple_components(self):
        """Should select the largest component for a single main object."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Small noise component
        mask[5:10, 5:10] = 255
        # Large main object component
        mask[50:150, 60:160] = 255

        bbox = self.preprocessor.compute_bounding_box(mask)
        self.assertIsNotNone(bbox)
        xmin, ymin, xmax, ymax = bbox
        self.assertEqual((xmin, ymin, xmax, ymax), (60, 50, 160, 150))

    def test_compute_bounding_box_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = self.preprocessor.compute_bounding_box(mask)
        self.assertIsNone(bbox)

    def test_crop_object(self):
        dummy = self.create_dummy_transparent_image(100, 100)
        mask = self.preprocessor.extract_alpha_mask(dummy)
        bbox = self.preprocessor.compute_bounding_box(mask)
        self.assertIsNotNone(bbox)
        cropped_img, cropped_mask = self.preprocessor.crop_object(dummy, mask, bbox)
        self.assertEqual(cropped_img.size, (40, 40))
        self.assertEqual(cropped_mask.shape, (40, 40))

    def test_pad_to_square(self):
        dummy = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        padded = self.preprocessor.pad_to_square(dummy, padding_ratio=0.10)
        # max_dim = 100, padding = 10 -> canvas = 100 + 20 = 120
        self.assertEqual(padded.size, (120, 120))

    def test_resize(self):
        dummy = Image.new("RGBA", (120, 120), (255, 0, 0, 255))
        resized = self.preprocessor.resize(dummy, (512, 512))
        self.assertEqual(resized.size, (512, 512))

    def test_preprocess_transparent_png(self):
        dummy_trans = self.create_dummy_transparent_image()
        result = self.preprocessor.preprocess(dummy_trans, target_resolution=(512, 512), padding_ratio=0.15)
        self.assertEqual(result.size, (512, 512))
        self.assertEqual(result.mode, "RGBA")

    def test_preprocess_empty_image_raises_error(self):
        empty_img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        with self.assertRaises(ObjectDetectionError):
            self.preprocessor.preprocess(empty_img)

    def test_preprocess_with_analysis(self):
        dummy_trans = self.create_dummy_transparent_image(300, 150)
        res = self.preprocessor.preprocess_with_analysis(dummy_trans, target_resolution=(1024, 1024))
        self.assertIn("ai_image", res)
        self.assertIn("highres_working_image", res)
        self.assertIn("silhouette", res)
        self.assertEqual(res["ai_image"].size, (1024, 1024))
        # Highres preserves true aspect ratio
        self.assertAlmostEqual(res["silhouette"]["aspect_ratio"], 2.0, delta=0.1)

    def test_refine_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        # Introduce a pinhole inside mask
        mask[50, 50] = 0
        refined = self.preprocessor.refine_mask(mask)
        # Pinhole should be closed
        self.assertEqual(refined[50, 50], 255)

    def test_thin_protrusion_preserved(self):
        """Verifies that 1-2px thin antenna/fin structures survive refine_mask completely."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Main body
        mask[30:70, 30:70] = 255
        # Thin 2px antenna protrusion
        mask[10:30, 49:51] = 255
        # Small pinhole
        mask[48:50, 48:50] = 0

        refined = self.preprocessor.refine_mask(mask)
        # Antenna must not be eroded
        self.assertTrue(np.all(refined[10:30, 49:51] == 255))
        # Pinhole must be filled
        self.assertTrue(np.all(refined[48:50, 48:50] == 255))

    def test_compute_adaptive_padding(self):
        """Verifies adaptive padding scales properly between 0.06 and 0.10 based on complexity."""
        # Simple solid square
        square_mask = np.zeros((100, 100), dtype=np.uint8)
        square_mask[20:80, 20:80] = 255
        pad_square = self.preprocessor.compute_adaptive_padding(square_mask)
        self.assertGreaterEqual(pad_square, 0.06)
        self.assertLessEqual(pad_square, 0.10)

        # Complex slender shape with thin antennas
        complex_mask = square_mask.copy()
        complex_mask[0:20, 25:27] = 255
        complex_mask[0:20, 75:77] = 255
        complex_mask[80:100, 25:27] = 255
        complex_mask[80:100, 75:77] = 255
        pad_complex = self.preprocessor.compute_adaptive_padding(complex_mask)
        self.assertGreaterEqual(pad_complex, 0.06)
        self.assertLessEqual(pad_complex, 0.10)
        # Complex shape gets smaller or equal padding to maximize AI resolution
        self.assertLessEqual(pad_complex, pad_square)


if __name__ == "__main__":
    unittest.main()


