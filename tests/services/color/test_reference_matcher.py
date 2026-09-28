import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh

from app.services.color.color_extractor import ColorExtractor, ColorItem
from app.services.color.color_mapper import ColorMapper
from app.services.color.reference_search import ReferenceSearch, ReferenceCandidate
from app.services.color.reference_matcher import ReferenceMatcher, MatchResult
from app.services.color.color_engine import ColorEngine, ColorDecision


class TestArbitraryImageColorSystem(unittest.TestCase):
    def setUp(self):
        self.extractor = ColorExtractor()
        self.searcher = ReferenceSearch(cache_dir="temp/test_ref_cache")
        self.matcher = ReferenceMatcher(match_threshold=0.85, extractor=self.extractor)
        self.mapper = ColorMapper()

    def create_color_band_image(self, top_rgb, mid_rgb, bot_rgb, w=120, h=120, shadow=False) -> Image.Image:
        """Creates a synthetic 3-band object on transparent background."""
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        # Bounding box of object: x in [20, 100], y in [10, 110]
        y_step = 33
        arr[10:10+y_step, 20:100, :3] = top_rgb
        arr[10:10+y_step, 20:100, 3] = 255

        arr[10+y_step:10+2*y_step, 20:100, :3] = mid_rgb
        arr[10+y_step:10+2*y_step, 20:100, 3] = 255

        arr[10+2*y_step:110, 20:100, :3] = bot_rgb
        arr[10+2*y_step:110, 20:100, 3] = 255

        if shadow:
            # Darken bottom half significantly to simulate harsh lighting shadow
            arr[60:110, 20:100, :3] = (arr[60:110, 20:100, :3].astype(float) * 0.05).astype(np.uint8)

        return Image.fromarray(arr, mode="RGBA")

    def test_high_confidence_image(self):
        """Clean, well-lit image should have high confidence."""
        clean_img = self.create_color_band_image(
            top_rgb=(220, 30, 30),
            mid_rgb=(30, 30, 220),
            bot_rgb=(220, 220, 30),
            w=200,
            h=200,
            shadow=False,
        )
        res = self.extractor.extract_colors(clean_img)
        self.assertGreaterEqual(res.color_confidence, 0.65)
        self.assertIn("shadow_score", res.confidence_breakdown)

    def test_low_confidence_dark_shadow_image(self):
        """Heavily shadowed image should trigger lower confidence."""
        shadow_img = self.create_color_band_image(
            top_rgb=(220, 30, 30),
            mid_rgb=(30, 30, 220),
            bot_rgb=(220, 220, 30),
            w=100,
            h=100,
            shadow=True,
        )
        res = self.extractor.extract_colors(shadow_img)
        # Shadow ratio should be detected and confidence lower than clean
        self.assertGreater(res.confidence_breakdown["shadow_ratio"], 0.3)
        self.assertLess(res.color_confidence, 0.75)

    def test_reference_matcher_same_object_accepted(self):
        """Identical / same-structure candidate should be accepted."""
        user_img = self.create_color_band_image((220, 30, 30), (30, 30, 220), (220, 220, 30))
        ref_img = self.create_color_band_image((230, 35, 35), (25, 25, 230), (225, 225, 25))

        user_ext = self.extractor.extract_colors(user_img)
        cand = ReferenceCandidate(
            candidate_id="match_cand_1",
            image=ref_img,
            source_origin="test_catalog",
            priority_level=2,
            source_type="same_product",
        )

        match_res = self.matcher.verify_candidate(user_img, user_ext, cand)
        self.assertTrue(match_res.is_match)
        self.assertTrue(match_res.spatial_region_match)
        self.assertGreaterEqual(match_res.match_confidence, 0.85)

    def test_reference_matcher_inverted_spatial_regions_rejected(self):
        """Candidate with inverted colors (Top=Yellow, Bot=Red vs Top=Red, Bot=Yellow) MUST be rejected."""
        user_img = self.create_color_band_image((220, 30, 30), (30, 30, 220), (220, 220, 30))
        # Inverted vertical layout: Top=Yellow, Mid=Blue, Bot=Red
        inverted_img = self.create_color_band_image((220, 220, 30), (30, 30, 220), (220, 30, 30))

        user_ext = self.extractor.extract_colors(user_img)
        cand = ReferenceCandidate(
            candidate_id="inverted_cand",
            image=inverted_img,
            source_origin="test_catalog",
            priority_level=4,
            source_type="visual_match",
        )

        match_res = self.matcher.verify_candidate(user_img, user_ext, cand)
        # Strict Section 11 rule: Must reject!
        self.assertFalse(match_res.is_match)
        self.assertFalse(match_res.spatial_region_match)
        self.assertIsNotNone(match_res.rejection_reason)
        self.assertIn("Spatial color region mismatch", match_res.rejection_reason)

    def test_reference_search_network_resilience(self):
        """ReferenceSearch should handle errors gracefully and never throw exceptions."""
        searcher = ReferenceSearch(enabled=True, timeout_seconds=0.01)
        dummy = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        # Passing invalid candidate should not crash
        results = searcher.search_candidates(
            user_image=dummy,
            candidate_pool=["non_existent_file_path_xyz.png"],
        )
        self.assertIsInstance(results, list)

    def test_color_engine_decision_object(self):
        """ColorEngine should generate an explicit explainable ColorDecision object."""
        cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        clean_img = self.create_color_band_image((220, 30, 30), (30, 30, 220), (220, 220, 30))

        engine = ColorEngine(confidence_threshold=0.50)
        out_glb = Path("temp/test_ref_cache/test_decision.glb")

        colored_mesh, report = engine.process_and_color_mesh(
            mesh=cube,
            view_images={"front": clean_img},
            output_glb_path=out_glb,
        )

        self.assertIn("color_decision", report.metrics)
        decision: ColorDecision = report.metrics["color_decision"]
        self.assertEqual(decision.source, "uploaded_image")
        self.assertFalse(decision.reference_used)
        self.assertIn("front", decision.observed_views)
        self.assertIn("back", decision.inferred_views)

    def test_color_disabled_generates_blender_clay_base_model(self):
        """When enable_color is False, ColorEngine must create an uncolored Blender clay base model."""
        cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        img = self.create_color_band_image((220, 30, 30), (30, 30, 220), (220, 220, 30))

        engine = ColorEngine()
        out_glb = Path("temp/test_ref_cache/test_clay_base.glb")

        colored_mesh, report = engine.process_and_color_mesh(
            mesh=cube,
            view_images={"front": img},
            output_glb_path=out_glb,
            enable_color=False,
        )

        decision: ColorDecision = report.metrics["color_decision"]
        self.assertEqual(decision.source, "disabled")
        self.assertFalse(decision.reference_used)
        self.assertEqual(report.metrics.get("mode"), "blender_clay_base")

        # Verify vertex colors are Blender default clay grey (all channels ~205)
        vcols = colored_mesh.visual.vertex_colors
        self.assertIsNotNone(vcols)
        self.assertTrue(np.all(vcols[:, 0] == 205))
        self.assertTrue(np.all(vcols[:, 1] == 205))
        self.assertTrue(np.all(vcols[:, 2] == 205))


if __name__ == "__main__":
    unittest.main()

