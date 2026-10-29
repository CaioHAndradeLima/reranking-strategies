import math
import re
from collections import Counter
from collections.abc import Iterable

from .models import Document, Hit

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.I)


def _validate_limit(limit: int) -> None:
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a non-negative integer")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Small dependency-free BM25 implementation for the demo corpus."""

    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        if isinstance(k1, bool) or not isinstance(k1, (int, float)) or not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        document_ids = [doc.id for doc in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Document IDs must be unique for BM25 retrieval")
        self.k1, self.b = k1, b
        self.tokens = [tokenize(doc.searchable_text) for doc in self.documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequency.update(frequencies.keys())

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        _validate_limit(limit)
        terms, count = set(tokenize(query)), len(self.documents)
        scored = []
        for doc, frequencies, length in zip(self.documents, self.term_frequencies, self.lengths):
            score = 0.0
            for term in terms:
                tf = frequencies.get(term, 0)
                if tf:
                    df = self.document_frequency[term]
                    idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                    norm = tf + self.k1 * (1 - self.b + self.b * length / max(1, self.average_length))
                    score += idf * tf * (self.k1 + 1) / norm
            scored.append(Hit(doc, score, "bm25"))
        scored.sort(key=lambda hit: (-hit.score, hit.document.id))
        return [hit for hit in scored[:limit] if hit.score > 0]


def rrf_fuse(rankings: Iterable[Iterable[Hit]], limit: int = 10, k: int = 60) -> list[Hit]:
    """Fuse ranked lists by reciprocal rank; ranks are one-based."""
    _validate_limit(limit)
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for hit in ranking:
            doc_id = hit.document.id
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank = len(seen)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            documents[doc_id] = hit.document
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:limit]
    return [Hit(documents[doc_id], scores[doc_id], "rrf") for doc_id in ordered]
