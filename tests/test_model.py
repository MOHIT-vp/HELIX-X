import unittest

import torch

from src.graph_loader import build_demo_graph
from src.model import GraphRAGLinkPredictor


class ModelTests(unittest.TestCase):
    def test_architecture_materializes_and_checkpoint_matches(self):
        data, _ = build_demo_graph()
        model = GraphRAGLinkPredictor()
        with torch.inference_mode():
            model.encode({nt: data[nt].x for nt in data.node_types}, {et: data[et].edge_index for et in data.edge_types})
        state = torch.load("assets/model_epoch8_score0.3792.pt", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 14505395)


if __name__ == "__main__":
    unittest.main()
