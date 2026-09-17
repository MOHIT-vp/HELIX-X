import math
import tempfile
import zipfile
import unittest
from dataclasses import replace
from pathlib import Path

from backend.app.config import Settings
from backend.app.service import EntityNotFound, InferenceService


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = InferenceService()
        cls.service.load()

    def test_recovered_checkpoint_enables_demo_inference(self):
        self.assertTrue(self.service.ready)
        result = self.service.prediction(0, 0)
        self.assertTrue(math.isfinite(result["score"]))
        self.assertEqual(result["score_type"], "model_score")
        self.assertEqual(result["compound"]["name"], "Metformin")

    def test_candidate_order_is_deterministic(self):
        first = self.service.candidates(0, "diseases", 5)
        second = self.service.candidates(0, "diseases", 5)
        self.assertEqual(first, second)
        self.assertEqual([candidate["rank"] for candidate in first["candidates"]], [1, 2, 3, 4, 5])

    def test_unknown_entity_is_cleanly_rejected(self):
        with self.assertRaises(EntityNotFound):
            self.service.prediction(999999, 0)
        with self.assertRaises(EntityNotFound):
            self.service.prediction(0, 999999)

    def test_evidence_graph_edges_have_valid_endpoints(self):
        graph = self.service.prediction(0, 0)["evidence_graph"]
        node_ids = {node["id"] for node in graph["nodes"]}
        self.assertTrue(all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"]))

    def test_extracted_checkpoint_fallback_is_supported(self):
        settings = Settings.from_env(Path.cwd())
        with tempfile.TemporaryDirectory() as temporary_dir:
            with zipfile.ZipFile(settings.weights_path) as archive:
                archive.extractall(temporary_dir)
            payload_root = next(Path(temporary_dir).iterdir())
            fallback_path = Path(temporary_dir) / "model_epoch8_score0.3792.bin"
            settings = replace(settings, weights_path=fallback_path)
            service = InferenceService(settings.root, settings)
            service.load()
            self.assertTrue(service.ready)

    def test_missing_checkpoint_disables_inference(self):
        settings = Settings.from_env(Path.cwd())
        settings = replace(settings, weights_path=settings.root / "assets" / "missing-checkpoint.pt")
        service = InferenceService(settings.root, settings)
        service.load()
        self.assertFalse(service.model_loaded)
        self.assertFalse(service.embeddings_cached)
        self.assertFalse(service.ready)


if __name__ == "__main__":
    unittest.main()
