"""Text embeddings for retrieval.

- Default model: BAAI/bge-small-en-v1.5 (384-d). Small enough to run on CPU in
  the demo, strong on retrieval benchmarks for its size.
- Embeddings are L2-normalized, so cosine similarity = dot product, which matches
  the pgvector `vector_cosine_ops` index.
- bge v1.5 recommends an instruction prefix on *queries* (not passages) for
  short-query -> passage retrieval.
- Passages are embedded with a "title > section" header so a chunk like
  "values were higher in older adults" keeps its context.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.db.models import EMBEDDING_DIM

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def passage_text(title: str, section: str | None, text: str) -> str:
    """Text actually embedded for a chunk: context header + chunk text."""
    header = f"{title} > {section}" if section else title
    return f"{header}\n{text}"


class Embedder(Protocol):
    """What the ingestion and retrieval code needs from an embedding model."""

    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of unit vectors."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) unit vector."""
        ...

    def count_tokens(self, text: str) -> int:
        """Number of model tokens in `text` (without special tokens)."""
        ...


class BgeEmbedder:
    """sentence-transformers wrapper for bge models (CPU)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self.model.get_embedding_dimension())
        self.batch_size = batch_size
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages in batches."""
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query (with the bge instruction prefix)."""
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens with the model's own tokenizer."""
        # verbose=False: counting long texts is fine; only *embedding* them would truncate.
        ids = self.model.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        return len(ids)


class HashingEmbedder:
    """Deterministic, model-free embedder for tests (feature hashing of words).

    Texts sharing words get similar vectors, which is enough to test storage
    and search plumbing without downloading a model.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)  # noqa: S324 - not security
            v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages."""
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query."""
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        """Whitespace token count."""
        return len(text.split())


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> Embedder:
    """Load (once) the configured embedding model."""
    return BgeEmbedder(model_name)
