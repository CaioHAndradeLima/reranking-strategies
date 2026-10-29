from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float
    source: str


@dataclass(frozen=True)
class QueryCase:
    id: str
    query: str
    relevance: dict[str, int]
    tags: tuple[str, ...] = ()
    answerable: bool = True
    reference_answer: str | None = None
