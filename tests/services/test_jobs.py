import unittest
from unittest.mock import MagicMock, patch
import asyncio
from pathlib import Path
from PIL import Image

from app.config.settings import Settings
from app.services.jobs import JobManager


class TestJobManager(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.job_manager = JobManager(self.settings)

    def test_create_job_single_image(self):
        job_id = self.job_manager.create_job(image_id="img_123", model_name="hunyuan3d")
        job = self.job_manager.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["image_id"], "img_123")
        self.assertEqual(job["image_ids"], ["img_123"])
        self.assertEqual(job["num_inference_steps"], 30)
        self.assertEqual(job["octree_resolution"], 384)

    def test_create_job_multi_image(self):
        ids = ["img_1", "img_2", "img_3", "img_4", "img_5", "img_6"]
        job_id = self.job_manager.create_job(image_ids=ids, num_inference_steps=50, octree_resolution=512)
        job = self.job_manager.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(len(job["image_ids"]), 6)
        self.assertEqual(job["num_inference_steps"], 50)
        self.assertEqual(job["octree_resolution"], 512)

    def test_create_job_no_image_raises(self):
        with self.assertRaises(ValueError):
            self.job_manager.create_job()


if __name__ == "__main__":
    unittest.main()
