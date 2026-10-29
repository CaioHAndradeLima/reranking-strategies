import json
import tempfile
import unittest
from pathlib import Path

from reranking_strategies.data import load_documents, load_queries, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def write_json(self, directory, filename, value):
        path = Path(directory) / filename
        path.write_text(json.dumps(value))
        return path

    def test_loaders_reject_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            documents = [{"id": "d", "title": "One", "text": "text"},
                         {"id": "d", "title": "Two", "text": "text"}]
            queries = [{"id": "q", "query": "one", "relevance": {}},
                       {"id": "q", "query": "two", "relevance": {}}]

            with self.assertRaisesRegex(ValueError, "duplicate document id"):
                load_documents(self.write_json(directory, "documents.json", documents))
            with self.assertRaisesRegex(ValueError, "duplicate query id"):
                load_queries(self.write_json(directory, "queries.json", queries))

    def test_loaders_reject_invalid_json_shapes_and_grades(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "JSON array"):
                load_documents(self.write_json(directory, "documents.json", {"id": "d"}))
            bad_queries = [{"id": "q", "query": "one", "relevance": {"d": True}}]
            with self.assertRaisesRegex(ValueError, "integers from 0 to 3"):
                load_queries(self.write_json(directory, "queries.json", bad_queries))

    def test_loaders_reject_inconsistent_query_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            bad_queries = [{"id": "q", "query": "one", "relevance": {"d": 1},
                            "answerable": False}]
            with self.assertRaisesRegex(ValueError, "cannot have relevance labels"):
                load_queries(self.write_json(directory, "queries.json", bad_queries))

    def test_dataset_rejects_relevance_for_unknown_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            documents = load_documents(self.write_json(
                directory, "documents.json", [{"id": "known", "title": "Known", "text": "text"}]
            ))
            queries = load_queries(self.write_json(
                directory, "queries.json", [{"id": "q", "query": "find", "relevance": {"missing": 2}}]
            ))

            with self.assertRaisesRegex(ValueError, "unknown document IDs"):
                validate_dataset(documents, queries)


if __name__ == "__main__":
    unittest.main()
