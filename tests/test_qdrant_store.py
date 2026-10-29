import os
import unittest
import uuid

from reranking_strategies.models import Document
from reranking_strategies.qdrant_store import QdrantDenseStore


class FakeModels:
    class Distance:
        COSINE = "cosine"

    @staticmethod
    def VectorParams(**kwargs):
        return kwargs

    @staticmethod
    def Document(**kwargs):
        return kwargs


class FakeClient:
    def __init__(self, collection_exists=True):
        self.exists = collection_exists
        self.calls = []

    def collection_exists(self, collection):
        return self.exists

    def delete_collection(self, collection):
        self.calls.append(("delete", collection))
        self.exists = False

    def create_collection(self, **kwargs):
        self.calls.append(("create", kwargs))
        self.exists = True

    def get_embedding_size(self, model):
        return 384

    def upload_collection(self, **kwargs):
        self.calls.append(("upload", kwargs))


class QdrantStoreUnitTests(unittest.TestCase):
    def test_reindex_replaces_existing_collection_snapshot(self):
        client = FakeClient()
        store = QdrantDenseStore(client=client, models=FakeModels)

        store.index([Document("new", "New", "current corpus")])

        self.assertEqual([call[0] for call in client.calls], ["delete", "create", "upload"])
        self.assertEqual(client.calls[-1][1]["ids"], [0])
        self.assertEqual(client.calls[-1][1]["payload"][0]["document_id"], "new")

    def test_empty_reindex_clears_existing_collection(self):
        client = FakeClient()
        QdrantDenseStore(client=client, models=FakeModels).index([])

        self.assertEqual([call[0] for call in client.calls], ["delete", "create"])

    def test_duplicate_document_ids_are_rejected_before_mutation(self):
        client = FakeClient()
        store = QdrantDenseStore(client=client, models=FakeModels)
        docs = [Document("same", "One", "first"), Document("same", "Two", "second")]

        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            store.index(docs)

        self.assertEqual(client.calls, [])


@unittest.skipUnless(os.getenv("RERANKING_QDRANT_URL"), "set RERANKING_QDRANT_URL for Docker integration")
class QdrantDockerIntegrationTests(unittest.TestCase):
    def test_reindex_removes_stale_points_from_qdrant(self):
        url = os.environ["RERANKING_QDRANT_URL"]
        collection = f"reranking_test_{uuid.uuid4().hex}"
        store = QdrantDenseStore(url=url, collection=collection)
        docs = [
            Document("first", "First", "First integration test document"),
            Document("second", "Second", "Second integration test document"),
        ]
        try:
            store.index(docs)
            store.index(docs[:1])
            count = store.client.count(collection_name=collection, exact=True).count
            self.assertEqual(count, 1)
            hits = store.search("First integration test document", {"first": docs[0]}, limit=1)
            self.assertEqual([hit.document.id for hit in hits], ["first"])
        finally:
            if store.client.collection_exists(collection):
                store.client.delete_collection(collection)


if __name__ == "__main__":
    unittest.main()
