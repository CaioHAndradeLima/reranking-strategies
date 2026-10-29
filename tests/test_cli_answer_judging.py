import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from reranking_strategies import cli


class AnswerJudgingCliTests(unittest.TestCase):
    def test_judge_answers_writes_a_report_with_scores_and_usage(self):
        response_json = json.dumps({
            "correctness": 4,
            "groundedness": 4,
            "assertiveness_calibration": 4,
            "abstained": False,
            "unsupported_claims": [],
            "rationale": "The answer is supported by the cited passage.",
        })

        class Responses:
            def create(self, **kwargs):
                return types.SimpleNamespace(
                    output_text=response_json,
                    usage=types.SimpleNamespace(input_tokens=80, output_tokens=12),
                )

        client = types.SimpleNamespace(responses=Responses())
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = lambda: client
        answers = [{
            "query_id": "q-003",
            "strategy": "dense+heuristic",
            "answer": "Security audit events are retained for one year.",
            "evidence_ids": ["doc-008"],
        }]

        with tempfile.TemporaryDirectory() as directory:
            answers_path = Path(directory) / "answers.json"
            report_path = Path(directory) / "report.json"
            answers_path.write_text(json.dumps(answers))
            argv = [
                "rerank-lab", "judge-answers", "--answers", str(answers_path),
                "--output", str(report_path), "--max-llm-calls", "1",
            ]
            stdout = io.StringIO()
            with patch.dict(sys.modules, {"openai": fake_openai}), patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    cli.main()
            report = json.loads(report_path.read_text())

        self.assertEqual(report["summary"]["dense+heuristic"]["correctness"], 4)
        self.assertEqual(report["summary"]["dense+heuristic"]["request_count"], 1)
        self.assertEqual(report["summary"]["dense+heuristic"]["input_tokens"], 80)
        self.assertTrue(json.loads(stdout.getvalue())["report"].endswith("report.json"))


if __name__ == "__main__":
    unittest.main()
