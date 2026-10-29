import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reranking_strategies import cli


class EdgeBenchmarkCliTests(unittest.TestCase):
    def test_edge_benchmark_compares_retrieval_and_reranking(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "edge-report.json"
            argv = ["rerank-lab", "benchmark-edges", "--output", str(output)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            report = json.loads(output.read_text())

        self.assertEqual(report["experiment"]["corpus_type"], "graph_edges")
        self.assertEqual(report["experiment"]["edge_count"], 19)
        self.assertIn("retrieval", report["retrieval_modes"]["bm25"]["summary"])
        self.assertIn("heuristic", report["retrieval_modes"]["bm25"]["summary"])
        self.assertIn("mmr", report["retrieval_modes"]["bm25"]["summary"])
        self.assertEqual(report["retrieval_modes"]["bm25"]["query_count"], 6)


if __name__ == "__main__":
    unittest.main()
