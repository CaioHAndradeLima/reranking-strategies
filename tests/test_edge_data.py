import json
import tempfile
import unittest
from pathlib import Path

from reranking_strategies.data import load_documents, load_edges, load_queries, validate_dataset
from reranking_strategies.models import Document
from reranking_strategies.models import Hit
from reranking_strategies.rerankers import EdgeStructureReranker


ROOT = Path(__file__).resolve().parents[1]


class GraphEdgeDataTests(unittest.TestCase):
    def test_sample_edges_are_evidence_backed_and_become_searchable_points(self):
        documents = load_documents(ROOT / "data/documents.json")
        edges = load_edges(ROOT / "data/edges.json", documents)
        edge_documents = [edge.as_document() for edge in edges]
        queries = load_queries(ROOT / "data/edge_golden.json")

        validate_dataset(edge_documents, queries)

        self.assertEqual(len(edges), 19)
        edge = next(edge for edge in edges if edge.id == "edge-001")
        point = edge.as_document()
        self.assertIn("retry worker", point.searchable_text)
        self.assertIn("SCHEDULES", point.searchable_text)
        self.assertEqual(point.metadata["evidence_ids"], ["doc-002"])
        self.assertEqual(point.metadata["kind"], "edge")

    def test_edge_loader_rejects_missing_evidence_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.json"
            edge = {
                "id": "e1", "source_id": "s", "source_label": "source",
                "relation": "RELATES_TO", "target_id": "t", "target_label": "target",
                "description": "source relates to target", "evidence_ids": ["missing"],
            }
            path.write_text(json.dumps([edge]))
            docs = [Document("known", "Known", "Known evidence.")]
            with self.assertRaisesRegex(ValueError, "unknown evidence"):
                load_edges(path, docs)

            edge["evidence_ids"] = ["known"]
            path.write_text(json.dumps([edge, edge]))
            with self.assertRaisesRegex(ValueError, "duplicate edge id"):
                load_edges(path, docs)

    def test_edge_structure_reranker_uses_endpoints_and_relation(self):
        docs = load_documents(ROOT / "data/documents.json")
        edges = load_edges(ROOT / "data/edges.json", docs)
        selected = [edge.as_document() for edge in edges if edge.id in {"edge-019", "edge-006"}]
        hits = [Hit(document, 1.0, "dense") for document in selected]

        result = EdgeStructureReranker().rerank("Who owns catalog product descriptions?", hits)

        self.assertEqual(result[0].document.id, "edge-019")
        self.assertTrue(all(hit.source == "edge_structure" for hit in result))
        with self.assertRaisesRegex(ValueError, "requires edge documents"):
            EdgeStructureReranker().rerank(
                "query", [Hit(Document("d", "title", "text"), 1.0, "test")]
            )

    def test_edge_structure_reranker_does_not_match_query_function_words(self):
        docs = load_documents(ROOT / "data/documents.json")
        edges = load_edges(ROOT / "data/edges.json", docs)
        selected = [edge.as_document() for edge in edges if edge.id in {"edge-003", "edge-009", "edge-010"}]
        hits = [Hit(document, 1.0, "dense") for document in selected]

        result = EdgeStructureReranker().rerank("What happened in INC-4821?", hits)

        self.assertEqual(result[0].document.id, "edge-009")


if __name__ == "__main__":
    unittest.main()
