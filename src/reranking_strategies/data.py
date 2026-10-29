import json
from pathlib import Path

from .models import Document, QueryCase


def load_documents(path: str | Path) -> list[Document]:
    return [Document(**row) for row in json.loads(Path(path).read_text())]


def load_queries(path: str | Path) -> list[QueryCase]:
    return [QueryCase(id=row["id"], query=row["query"], relevance=row["relevance"],
                      tags=tuple(row.get("tags", [])), answerable=row.get("answerable", True))
            for row in json.loads(Path(path).read_text())]
