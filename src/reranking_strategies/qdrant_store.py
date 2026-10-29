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

    def index(
        self,
        documents: list[Document],
        indexed_payload_fields: tuple[str, ...] | list[str] = (),
    ) -> None:
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
        for field in indexed_payload_fields:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=self.models.PayloadSchemaType.KEYWORD,
            )
        if documents:
            self.client.upload_collection(
                collection_name=self.collection,
                vectors=[self.models.Document(text=doc.searchable_text, model=MODEL) for doc in documents],
                payload=[{
                    **doc.metadata,
                    "kind": doc.metadata.get("kind", "document"),
                    "document_id": doc.id,
                    "title": doc.title,
                    "text": doc.text,
                    "metadata": doc.metadata,
                } for doc in documents],
                ids=list(range(len(documents))),
            )

    def search(
        self,
        query: str,
        documents: dict[str, Document],
        limit: int = 10,
        filters: dict[str, str | int | bool] | None = None,
    ) -> list[Hit]:
        query_filter = None
        if filters:
            if any(not isinstance(key, str) or not key for key in filters):
                raise ValueError("Payload filter keys must be non-empty strings")
            query_filter = self.models.Filter(must=[
                self.models.FieldCondition(
                    key=key,
                    match=self.models.MatchValue(value=value),
                )
                for key, value in filters.items()
            ])
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.models.Document(text=query, model=MODEL), limit=limit,
            query_filter=query_filter,
        )
        hits = []
        for point in response.points:
            doc_id = point.payload["document_id"]
            hits.append(Hit(documents[doc_id], float(point.score), "dense"))
        return hits
