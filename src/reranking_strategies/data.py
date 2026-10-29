import json
from pathlib import Path
from typing import Any

from .models import Document, QueryCase


def _load_rows(path: str | Path, kind: str) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        rows = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} contains invalid JSON: {exc.msg}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"{source} must contain a JSON array of {kind}")
    if not rows:
        raise ValueError(f"{source} must contain at least one {kind}")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{source}: {kind} row {index} must be a JSON object")
    return rows


def _required_text(row: dict[str, Any], field: str, source: Path, kind: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {kind} row {index} requires a non-empty string '{field}'")
    return value


def load_documents(path: str | Path) -> list[Document]:
    source = Path(path)
    rows = _load_rows(source, "documents")
    documents = []
    seen_ids = set()
    allowed_fields = {"id", "title", "text", "metadata"}
    for index, row in enumerate(rows, start=1):
        unexpected = set(row) - allowed_fields
        if unexpected:
            raise ValueError(f"{source}: document row {index} has unknown fields: {sorted(unexpected)}")
        document_id = _required_text(row, "id", source, "document", index)
        if document_id in seen_ids:
            raise ValueError(f"{source}: duplicate document id '{document_id}'")
        seen_ids.add(document_id)
        title = _required_text(row, "title", source, "document", index)
        text = _required_text(row, "text", source, "document", index)
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}: document row {index} field 'metadata' must be an object")
        documents.append(Document(document_id, title, text, metadata))
    return documents


def load_queries(path: str | Path) -> list[QueryCase]:
    source = Path(path)
    rows = _load_rows(source, "queries")
    queries = []
    seen_ids = set()
    allowed_fields = {"id", "query", "relevance", "tags", "answerable"}
    for index, row in enumerate(rows, start=1):
        unexpected = set(row) - allowed_fields
        if unexpected:
            raise ValueError(f"{source}: query row {index} has unknown fields: {sorted(unexpected)}")
        query_id = _required_text(row, "id", source, "query", index)
        if query_id in seen_ids:
            raise ValueError(f"{source}: duplicate query id '{query_id}'")
        seen_ids.add(query_id)
        query = _required_text(row, "query", source, "query", index)
        relevance = row.get("relevance")
        if not isinstance(relevance, dict):
            raise ValueError(f"{source}: query row {index} field 'relevance' must be an object")
        for document_id, grade in relevance.items():
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(f"{source}: query row {index} relevance keys must be non-empty strings")
            if type(grade) is not int or not 0 <= grade <= 3:
                raise ValueError(f"{source}: query row {index} relevance grades must be integers from 0 to 3")
        tags = row.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError(f"{source}: query row {index} field 'tags' must be an array of non-empty strings")
        answerable = row.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"{source}: query row {index} field 'answerable' must be a boolean")
        if not answerable and relevance:
            raise ValueError(f"{source}: unanswerable query '{query_id}' cannot have relevance labels")
        queries.append(QueryCase(query_id, query, relevance, tuple(tags), answerable))
    return queries


def validate_dataset(documents: list[Document], queries: list[QueryCase]) -> None:
    """Validate cross-file references before evaluating a corpus and its queries."""
    if not documents:
        raise ValueError("The document corpus must not be empty")
    if not queries:
        raise ValueError("The query set must not be empty")
    document_ids = {document.id for document in documents}
    for query in queries:
        unknown_ids = set(query.relevance) - document_ids
        if unknown_ids:
            raise ValueError(
                f"Query '{query.id}' references unknown document IDs: {sorted(unknown_ids)}"
            )
