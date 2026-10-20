"""Opt-in tests that download a model or call a paid external provider."""
import numpy as np
import pytest

pytestmark = pytest.mark.live


def test_real_embeddings():
    from app.embedding_service import generate_embeddings
    vectors = np.asarray(generate_embeddings(['Solar panels produce renewable energy.',
                                              'Clean electricity comes from solar energy.',
                                              'A recipe for chocolate cake.']))
    assert vectors.shape == (3, 384)
    assert np.isfinite(vectors).all()
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    assert normalized[0] @ normalized[1] > normalized[0] @ normalized[2]


def test_real_gemini():
    from app.config import GEMINI_API_KEY
    from app.llm import generate_answer
    if not GEMINI_API_KEY:
        pytest.skip('GEMINI_API_KEY is required for the live Gemini test')
    answer = generate_answer('In which year did the project start?', '[S1] Document: sample.pdf; page: 1. Project Aurora started in 2024.')
    assert '2024' in answer
