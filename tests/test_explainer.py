import unittest

from src.explainer import find_paths
from src.graph_loader import build_demo_graph


class ExplainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.mappings = build_demo_graph()

    def test_paths_are_deterministic_and_bounded(self):
        first = find_paths(self.data, self.mappings, 0, 0, max_paths=3)
        second = find_paths(self.data, self.mappings, 0, 0, max_paths=3)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 3)
        for path in first:
            self.assertEqual(path["hops"], len(path["node_indices"]) - 1)

    def test_zero_path_case_is_supported(self):
        paths = find_paths(self.data, self.mappings, 0, 999, max_paths=1)
        self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
