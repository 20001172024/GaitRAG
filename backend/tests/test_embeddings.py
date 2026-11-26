"""Tests for the embedding helpers (model-free)."""

import numpy as np

from app.db.models import EMBEDDING_DIM
from app.rag.embeddings import HashingEmbedder, passage_text


def test_passage_text_adds_context_header() -> None:
    assert passage_text("Gait norms", "Results", "Cadence was 110.") == (
        "Gait norms > Results\nCadence was 110."
    )
    assert passage_text("Gait norms", None, "x") == "Gait norms\nx"


def test_hashing_embedder_returns_unit_vectors_of_db_dimension() -> None:
    e = HashingEmbedder()
    vecs = e.embed_passages(["stride time variability", "insole pressure sensor"])
    assert vecs.shape == (2, EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_hashing_embedder_ranks_word_overlap_higher() -> None:
    e = HashingEmbedder()
    q = e.embed_query("stride time variability")
    near, far = e.embed_passages(["stride time variability was low", "the insole has sensors"])
    assert q @ near > q @ far
