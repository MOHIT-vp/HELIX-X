import unittest

from pydantic import ValidationError

from backend.app import main
from backend.app.schemas import CandidateRequest, EvidenceRequest, PredictionRequest


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.service.load()

    def test_status_and_metadata_routes(self):
        status = main.status()
        self.assertTrue(status["inference_ready"])
        self.assertEqual(main.health()["status"], "ready")
        self.assertEqual(main.methodology()["explanation"]["llm"], "Not used")
        self.assertGreater(len(main.compounds()), 0)
        self.assertGreater(len(main.diseases()), 0)

    def test_prediction_and_all_evidence_aliases(self):
        request = PredictionRequest(compound_id=0, disease_id=0)
        prediction = main.predict(request)
        self.assertEqual(prediction["compound"]["id"], 0)
        evidence_request = EvidenceRequest(compound_id=0, disease_id=0, max_paths=3)
        graphs = [main.evidence(evidence_request), main.evidence_path(evidence_request), main.evidence_graph(evidence_request), main.graph_subgraph(evidence_request)]
        self.assertTrue(all(graph == graphs[0] for graph in graphs))

    def test_validation_bounds(self):
        with self.assertRaises(ValidationError):
            PredictionRequest(compound_id=-1, disease_id=0)
        with self.assertRaises(ValidationError):
            CandidateRequest(entity_id=0, limit=51)
        with self.assertRaises(ValidationError):
            EvidenceRequest(compound_id=0, disease_id=0, max_paths=13)

    def test_candidate_routes(self):
        diseases = main.candidate_diseases(CandidateRequest(entity_id=0, limit=3))
        compounds = main.candidate_compounds(CandidateRequest(entity_id=0, limit=3))
        self.assertEqual(diseases["direction"], "diseases")
        self.assertEqual(compounds["direction"], "compounds")
        self.assertEqual(len(diseases["candidates"]), 3)
        self.assertEqual(len(compounds["candidates"]), 3)


if __name__ == "__main__":
    unittest.main()
