"""Optional local dense retrieval adapter powered by Qdrant + FastEmbed."""

from .models import Document, Hit

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class QdrantDenseStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "reranking_demo",
        client=None,
        models=None,
    ):
        if client is None or models is None:
            try:
                from qdrant_client import QdrantClient, models as qdrant_models
            except ImportError as exc:
                raise RuntimeError("Install with: pip install -e '.[qdrant]'") from exc
            if client is None:
                client = QdrantClient(url=url)
            if models is None:
                models = qdrant_models
        self.client, self.models, self.collection = client, models, collection

    def index(self, documents: list[Document]) -> None:
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("Document IDs must be unique when indexing a Qdrant collection")

        # This adapter owns the collection as a snapshot of the supplied corpus.
        # Recreate it so a smaller or changed corpus cannot leave stale points behind.
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=self.models.VectorParams(
                size=self.client.get_embedding_size(MODEL),
                distance=self.models.Distance.COSINE,
            ),
        )
        if documents:
            self.client.upload_collection(
                collection_name=self.collection,
                vectors=[self.models.Document(text=doc.searchable_text, model=MODEL) for doc in documents],
                payload=[{"document_id": doc.id, "title": doc.title, "text": doc.text,
                          "metadata": doc.metadata} for doc in documents],
                ids=list(range(len(documents))),
            )

    def search(self, query: str, documents: dict[str, Document], limit: int = 10) -> list[Hit]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.models.Document(text=query, model=MODEL), limit=limit,
        )
        hits = []
        for point in response.points:
            doc_id = point.payload["document_id"]
            hits.append(Hit(documents[doc_id], float(point.score), "dense"))
        return hits
