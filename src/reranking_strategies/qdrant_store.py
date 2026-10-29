"""Optional local dense retrieval adapter powered by Qdrant + FastEmbed."""

from .models import Document, Hit

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class QdrantDenseStore:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "reranking_demo"):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[qdrant]'") from exc
        self.client, self.models, self.collection = QdrantClient(url=url), models, collection

    def index(self, documents: list[Document]) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self.models.VectorParams(
                    size=self.client.get_embedding_size(MODEL),
                    distance=self.models.Distance.COSINE,
                ),
            )
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
