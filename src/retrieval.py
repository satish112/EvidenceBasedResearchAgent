from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocumentChunk:
    doc_id: str
    source: str
    text: str
    chunk_id: int


@dataclass
class RetrievedChunk:
    doc_id: str
    source: str
    text: str
    score: float
    chunk_id: int


class LocalRetriever:
    """Simple local retrieval layer for prototype use.

    This uses TF-IDF similarity so the project runs without paid APIs.
    It can be replaced later with sentence embeddings + FAISS/Chroma.
    """

    def __init__(self, docs_path: str | Path, chunk_size: int = 120, overlap: int = 30):
        self.docs_path = Path(docs_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[DocumentChunk] = self._load_chunks()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    def _load_chunks(self) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for file_path in sorted(self.docs_path.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")
            words = text.split()
            step = max(1, self.chunk_size - self.overlap)
            for i in range(0, len(words), step):
                block = " ".join(words[i : i + self.chunk_size])
                if len(block.split()) < 20:
                    continue
                chunks.append(
                    DocumentChunk(
                        doc_id=file_path.stem,
                        source=file_path.name,
                        text=block,
                        chunk_id=len(chunks),
                    )
                )
        if not chunks:
            raise ValueError(f"No .txt documents found in {self.docs_path}")
        return chunks

    def search(self, query: str, top_k: int = 4, min_score: float = 0.05) -> List[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]
        results: List[RetrievedChunk] = []
        for idx in ranked_indices[: max(top_k * 2, top_k)]:
            score = float(scores[idx])
            if score < min_score:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                    chunk_id=chunk.chunk_id,
                )
            )
            if len(results) >= top_k:
                break
        return results
