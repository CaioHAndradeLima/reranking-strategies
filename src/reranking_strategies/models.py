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
class GraphEdge:
    """A directed, evidence-backed relationship stored as a searchable point."""

    id: str
    source_id: str
    source_label: str
    relation: str
    target_id: str
    target_label: str
    description: str
    evidence_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_document(self) -> Document:
        return Document(
            id=self.id,
            title=f"{self.source_label} — {self.relation} → {self.target_label}",
            text=self.description,
            metadata={
                **self.metadata,
                "kind": "edge",
                "source_id": self.source_id,
                "source_label": self.source_label,
                "relation": self.relation,
                "target_id": self.target_id,
                "target_label": self.target_label,
                "evidence_ids": list(self.evidence_ids),
            },
        )


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
