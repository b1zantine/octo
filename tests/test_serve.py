import json
from pathlib import Path
import tempfile
import unittest

from octo import Encoder, Limits, StructuralTokens
from test_octo import model, tokenizer
try:
    from fastapi.testclient import TestClient
    from octo.serve import CheckpointService, Request, create_app, default_checkpoint, to_record
    SERVE_AVAILABLE = True
except ImportError:
    SERVE_AVAILABLE = False


@unittest.skipUnless(SERVE_AVAILABLE, "install the serve extra to run playground API tests")
class ServeTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.encoder = Encoder(tokenizer(), StructuralTokens(2, 3, 4, 5, 6),
                               Limits(max_questions=8, max_choice_candidates=255), 1)
        self.client = TestClient(create_app(CheckpointService(model(), self.encoder, "colab", "cpu")))
        self.request = {"state": {"report": "broken"}, "model": "octo-latest", "questions": {
            "team": {"type": "choice", "instructions": "Which team", "criteria": {
                "billing": "payment", "engineering": "product functionality"}},
            "rating": {"type": "score", "instructions": "Which", "criteria": ["minor", "major"]},
            "broken": {"type": "noul", "instructions": "report broken"}}}

    def test_predictions_and_separate_agree(self):
        packed = self.client.post("/v1/systemone", json=self.request)
        separate = self.client.post("/v1/systemone/separate", json=self.request)
        self.assertEqual(packed.status_code, 200, packed.text)
        self.assertEqual(separate.status_code, 200, separate.text)
        a, b = packed.json(), separate.json()
        self.assertEqual(set(a["answers"]), {"team", "rating", "broken"})
        self.assertEqual(a["answers"]["team"]["choice"], b["answers"]["team"]["choice"])
        for key, p in a["answers"]["team"]["probabilities"].items():
            self.assertAlmostEqual(p, b["answers"]["team"]["probabilities"][key], places=5)
        self.assertAlmostEqual(a["answers"]["rating"]["score"], b["answers"]["rating"]["score"], places=5)
        self.assertAlmostEqual(a["answers"]["broken"]["noul"], b["answers"]["broken"]["noul"], places=5)
        self.assertGreater(b["usage"]["input_tokens"], a["usage"]["input_tokens"])

    def test_permutation_keeps_identities(self):
        response = self.client.post("/v1/systemone/permute", json={
            "request": self.request, "question": "team", "n_perm": 6})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(result["runs"]), 6)
        self.assertTrue(result["argmax_stable"])
        self.assertTrue(all(p < 1e-5 for p in result["spread"].values()))

    def test_invalid_requests(self):
        for patch in ({"model": "missing"}, {"questions": {}}, {"questions": {
                "one": {"type": "choice", "instructions": "Which", "criteria": {"a": None}}}},
                {"questions": {"unknown": {"type": "bad", "instructions": "Which"}}}):
            response = self.client.post("/v1/systemone", json={**self.request, **patch})
            self.assertEqual(response.status_code, 422, response.text)
        response = self.client.post("/v1/systemone/permute", json={
            "request": self.request, "question": "team", "n_perm": 0})
        self.assertEqual(response.status_code, 422)

    def test_chess_candidate_count_and_noul_criteria(self):
        request = Request(state="chess", questions={"move": {
            "type": "choice", "instructions": "Which", "criteria": {str(i): "minor" for i in range(20)}}})
        record = to_record(request, self.encoder.limits)
        self.assertEqual(len(record.questions[0].candidates), 20)
        self.encoder.encode(record)
        request = Request(state="report", questions={"flag": {
            "type": "noul", "instructions": "broken", "criteria": {"true": "major"}}})
        self.assertIn("major", to_record(request, self.encoder.limits).questions[0].instruction)

    def test_colab_path_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "artifacts/runs/octo_v1_colab_1"
            checkpoint = run / "checkpoints/step-00014877-epoch-3"
            checkpoint.mkdir(parents=True)
            (run / "best_checkpoint.json").write_text(json.dumps({
                "path": "/content/octo/artifacts/runs/octo_v1_colab_1/checkpoints/step-00014877-epoch-3"}))
            with self.assertRaises(ValueError):
                default_checkpoint(directory)
            for file in ("octo.json", "pointer.pt", "backbone", "tokenizer"):
                (checkpoint / file).touch()
            self.assertEqual(default_checkpoint(directory), checkpoint)


if __name__ == "__main__":
    unittest.main()
