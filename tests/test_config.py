import os
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.config import Settings


class ConfigTests(unittest.TestCase):
    def test_defaults_resolve_from_repository_root(self):
        settings = Settings.from_env(Path.cwd())
        self.assertEqual(settings.port, 8000)
        self.assertEqual(settings.weights_path.name, "model_epoch8_score0.3792.pt")

    def test_environment_overrides(self):
        with patch.dict(os.environ, {"GRAPHRAG_PORT": "8010", "GRAPHRAG_MODE": "production", "GRAPHRAG_DEVICE": "cpu"}):
            settings = Settings.from_env(Path.cwd())
        settings.validate()
        self.assertEqual(settings.port, 8010)
        self.assertEqual(settings.mode, "production")
        self.assertEqual(settings.device, "cpu")

    def test_invalid_mode_is_rejected(self):
        with patch.dict(os.environ, {"GRAPHRAG_MODE": "invalid"}):
            settings = Settings.from_env(Path.cwd())
        with self.assertRaises(ValueError):
            settings.validate()


if __name__ == "__main__":
    unittest.main()
