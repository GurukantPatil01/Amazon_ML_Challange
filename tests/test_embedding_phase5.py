import pytest
import numpy as np
import faiss
from src.phase5a_embedding_experiments import (
    detect_unicode_script,
    build_dual_representations,
    EMBEDDING_DIM,
)


def test_detect_unicode_script():
    assert detect_unicode_script("Reliance Retail Limited") == "Latin"
    assert detect_unicode_script("रियल टेक प्राइवेट लिमिटेड") == "Devanagari (Hindi/Marathi)"
    assert detect_unicode_script("ಡೈನಾಮಿಕ್ ಪ್ರೊಡಕ್ಟ್ಸ್") == "Kannada"
    assert detect_unicode_script("హై ఎనర్జీ") == "Telugu"
    assert detect_unicode_script("12345") == "Latin"


def test_build_dual_representations():
    df_sample = [
        {"entity_id": "S1_1", "business_name": "ABC Corp", "business_address": "123 Main St", "name_norm": "abc corp", "address_norm": "123 main st"},
        {"entity_id": "S1_2", "business_name": "XYZ Ltd", "business_address": "", "name_norm": "xyz ltd", "address_norm": ""},
    ]
    import pandas as pd
    df = pd.DataFrame(df_sample)
    names, combos = build_dual_representations(df)
    assert len(names) == 2
    assert len(combos) == 2
    assert names[0] == "abc corp"
    assert combos[0] == "abc corp | 123 main st"
    assert names[1] == "xyz ltd"
    # When address is missing, fallback to name only
    assert combos[1] == "xyz ltd"


def test_faiss_inner_product_cosine():
    # Unit vectors
    dim = 64
    np.random.seed(42)
    vecs = np.random.randn(10, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    query = vecs[0:1]  # Query exact same vector as index 0
    scores, indices = index.search(query, k=3)

    assert indices[0][0] == 0
    assert pytest.approx(scores[0][0], rel=1e-4) == 1.0
