import numpy as np
from app.rag.vector_math import cosine_similarity

def test_cosine_similarity_identical_vectors():
    v = np.array([1,2,3])

    sim = cosine_similarity(v,v)
    assert np.isclose(sim, 1.0)

def test_cosine_similarity_orthogonal_vectors():
    v1 = np.array([1,0])
    v2 = np.array([0,1])

    sim = cosine_similarity( v1, v2)
    assert np.isclose(sim, 0.0)

def test_cosine_similarity_zero_vector():
    v1 = np.array([0,0,0])
    v2 = np.array([1,2,3])

    sim=cosine_similarity(v1,v2)

    assert sim == 0.0