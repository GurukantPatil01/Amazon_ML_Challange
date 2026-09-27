"""Unit tests for Phase 4 evidence-backed blocking strategies."""

import pytest
import pandas as pd

from src.blocking_phase4 import (
    CandidateGeneratorPhase4,
    compute_name_core_v2,
    compute_name_web_collapsed,
    compute_name_web_norm,
    generate_candidates_phase4,
)
from src.normalize import normalize_dataframe


def test_name_web_normalization():
    # 1. URL separator and suffix removal
    raw_url = "fortuneprecisionhigh.com"
    collapsed = compute_name_web_collapsed(raw_url)
    assert collapsed == "fortuneprecisionhigh"

    raw_spaced = "Fortune Precision High Inc"
    collapsed_spaced = compute_name_web_collapsed(raw_spaced)
    # Both should collapse to the same canonical representation
    assert collapsed == collapsed_spaced

    # 2. Mills & Le case
    raw_m1 = "millsle.com"
    raw_m2 = "Mills & Le LLC"
    assert compute_name_web_collapsed(raw_m1) == compute_name_web_collapsed(raw_m2)


def test_name_core_v2_leading_prefixes():
    # Leading 'The'
    assert compute_name_core_v2("the valley coalition") == "valley coalition"
    assert compute_name_core_v2("valley coalition") == "valley coalition"

    # Leading 'M/S' and 'T/A'
    assert compute_name_core_v2("m/s apex enterprises") == "apex enterprises"
    assert compute_name_core_v2("t/a bay charities") == "bay charities"

    # Interior 'the' must NOT be stripped
    assert compute_name_core_v2("inn of the mountain") == "inn of the mountain"


@pytest.fixture
def sample_phase4_data():
    s1 = pd.DataFrame([
        {
            "entity_id": "S1-101",
            "business_name": "Valley Coalition",
            "business_address": "9628 Laurel Lane, Scottsdale, AZ",
            "country": "US",
        },
        {
            "entity_id": "S1-102",
            "business_name": "Fortune Precision High Inc",
            "business_address": "1659 N Temple Street, Salt Lake City, UT",
            "country": "US",
        },
        {
            "entity_id": "S1-103",
            "business_name": "Bay Charities LP",
            "business_address": "362 Scott Street, Hubbard, OH",
            "country": "US",
        },
    ])

    s2 = pd.DataFrame([
        {
            "entity_id": "S2-201",
            "business_name": "The Valley Coalition",
            "business_address": "9628 Laurel Lane, Scottsdale, AZ",
            "country": "US",
        },
        {
            "entity_id": "S2-202",
            "business_name": "Fortuneprecisionhigh.com",
            "business_address": "1659 1/2 N Temple St, Salt Lake City, UT",
            "country": "US",
        },
        {
            "entity_id": "S2-203",
            "business_name": "Fayekor t/a Bay Charities LP",
            "business_address": "362 Scott Street, Hubbard, OH",
            "country": "US",
        },
    ])

    return normalize_dataframe(s1), normalize_dataframe(s2)


def test_phase4_candidate_generation_strategies(sample_phase4_data):
    s1_norm, s2_norm = sample_phase4_data
    gen = CandidateGeneratorPhase4("source2", s2_norm)

    s1_records = s1_norm.to_dict(orient="records")

    # S1-101 (Valley Coalition) -> matches S2-201 via name_core_v2
    cands_v2 = gen.block_name_core_v2("us", s1_records[0])
    assert "S2-201" in cands_v2

    # S1-102 (Fortune Precision High Inc) -> matches S2-202 via web_norm
    cands_web = gen.block_web_norm("us", s1_records[1])
    assert "S2-202" in cands_web

    # S1-103 (Bay Charities LP) -> matches S2-203 via house_token (house=362, token=charities)
    cands_house = gen.block_house_token("us", s1_records[2])
    assert "S2-203" in cands_house


def test_generate_candidates_phase4_interface(sample_phase4_data):
    s1_norm, s2_norm = sample_phase4_data
    # Empty s3 for test
    s3_norm = pd.DataFrame(columns=s2_norm.columns)

    cands_df = generate_candidates_phase4(
        s1_norm, s2_norm, s3_norm, budget=50
    )

    required_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocking_sources",
        "block_count",
        "rank",
    ]
    for col in required_cols:
        assert col in cands_df.columns

    # Verify all matches captured
    pairs = set(zip(cands_df["source1_entity_id"], cands_df["candidate_entity_id"]))
    assert ("S1-101", "S2-201") in pairs
    assert ("S1-102", "S2-202") in pairs
    assert ("S1-103", "S2-203") in pairs
