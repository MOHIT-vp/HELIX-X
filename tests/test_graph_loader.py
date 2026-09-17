import unittest

from src.graph_loader import auto_load, build_demo_graph, validate_graph


class GraphLoaderTests(unittest.TestCase):
    def test_demo_graph_validates(self):
        data, mappings = build_demo_graph()
        validate_graph(data, mappings)
        self.assertEqual(len(data.node_types), 11)
        self.assertEqual(len(data.edge_types), 48)

    def test_missing_real_artifacts_can_be_disabled(self):
        with self.assertRaises(FileNotFoundError):
            auto_load("/tmp/no-hetionet-data", allow_demo=False)

    def test_demo_entity_mappings_are_complete(self):
        data, mappings = build_demo_graph()
        self.assertEqual(mappings["Compound"][0], "Metformin")
        self.assertEqual(mappings["Disease"][0], "Alzheimer's disease")
        self.assertEqual(len(mappings["Compound"]), data["Compound"].num_nodes)


if __name__ == "__main__":
    unittest.main()
