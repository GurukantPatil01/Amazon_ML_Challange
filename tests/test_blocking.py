"""Unit tests for blocking and candidate generation (Phase 2)."""

import pytest
import pandas as pd

from src.block_index import InvertedIndex
from src.blocking import (
    CandidateGenerator,
    generate_candidates,
    run_strategy_on_s1,
)
from src.normalize import normalize_dataframe


@pytest.fixture
def mock_datasets():
    s1 = pd.DataFrame([
        {
            "entity_id": "S1-101",
            "business_name": "Celestial Memorial Trust",
            "business_address": "No. 35 Brentwood Apartments, Bangalore",
            "country": "India",
        },
        {
            "entity_id": "S1-102",
            "business_name": "Strategic Praetorian",
            "business_address": "702 N Street, Washington, DC",
            "country": "US",
        },
        {
            "entity_id": "S1-103",
            "business_name": "ZNB Club",
            "business_address": "5 bis Rue Pierre Dignac, Bordeaux",
            "country": "France",
        },
        {
            "entity_id": "S1-104",  # Singleton
            "business_name": "Unique Lone Entity Inc",
            "business_address": "123 Remote Road, Alaska",
            "country": "US",
        },
    ])

    s2 = pd.DataFrame([
        {
            "entity_id": "S2-201",  # True match for S1-101 (word order + suffix difference)
            "business_name": "Celestial Memorial",
            "business_address": "35 Brentwood Apartments, Bangalore",
            "country": "India",
        },
        {
            "entity_id": "S2-202",  # True match for S1-102 (suffix added)
            "business_name": "Strategic Praetorian Inc",
            "business_address": "N St, Washington, DC",
            "country": "US",
        },
        {
            "entity_id": "S2-203",  # True match for S1-103 (French)
            "business_name": "ZNB Club SARL",
            "business_address": "5 bis Rue Pierre Dignac, Bordeaux",
            "country": "France",
        },
        {
            "entity_id": "S2-204",  # Same name, different country (must NOT match S1-102)
            "business_name": "Strategic Praetorian",
            "business_address": "10 MG Road, Bangalore",
            "country": "India",
        },
    ])

    s3 = pd.DataFrame([
        {
            "entity_id": "S3-301",
            "business_name": "Celestial Memorial Private Limited",
            "business_address": "Bangalore, Karnataka",
            "country": "India",
        },
    ])

    return normalize_dataframe(s1), normalize_dataframe(s2), normalize_dataframe(s3)


class TestInvertedIndex:
    def test_inverted_index_add_and_query(self):
        idx = InvertedIndex("test", min_token_length=2)
        idx.add("us", "acme", "S2-1")
        idx.add("us", "acme", "S2-2")
        idx.add("india", "acme", "S2-3")

        assert idx.query("us", "acme") == ["S2-1", "S2-2"]
        assert idx.query("india", "acme") == ["S2-3"]
        assert idx.query("france", "acme") == []

    def test_frequency_pruning(self):
        idx = InvertedIndex("test_prune", max_df=2, max_df_ratio=None)
        idx.add("us", "common", "S2-1")
        idx.add("us", "common", "S2-2")
        idx.add("us", "common", "S2-3")
        idx.add("us", "rare", "S2-4")

        idx.finalize()

        # 'common' had 3 documents > max_df (2), so it is pruned
        assert idx.query("us", "common") == []
        # 'rare' is preserved
        assert idx.query("us", "rare") == ["S2-4"]


class TestBlockingStrategies:
    def test_country_isolation(self, mock_datasets):
        s1, s2, _ = mock_datasets
        gen = CandidateGenerator("source2", s2)

        # S1-102 is in US; S2-204 has same name but in India
        pairs = run_strategy_on_s1("exact_name", s1, gen)
        cands_s1_102 = [cid for (sid, cid) in pairs if sid == "S1-102"]

        assert "S2-204" not in cands_s1_102

    def test_exact_name_core(self, mock_datasets):
        s1, s2, _ = mock_datasets
        gen = CandidateGenerator("source2", s2)

        # S1-102 ('Strategic Praetorian') matches S2-202 ('Strategic Praetorian Inc') via name_core
        pairs = run_strategy_on_s1("name_core", s1, gen)
        assert ("S1-102", "S2-202") in pairs

    def test_french_entity_blocking(self, mock_datasets):
        s1, s2, _ = mock_datasets
        gen = CandidateGenerator("source2", s2)

        # S1-103 ('ZNB Club') matches S2-203 ('ZNB Club SARL') via name_core in France
        pairs = run_strategy_on_s1("name_core", s1, gen)
        assert ("S1-103", "S2-203") in pairs

    def test_char_3gram_retrieval(self, mock_datasets):
        s1, s2, _ = mock_datasets
        gen = CandidateGenerator("source2", s2)

        pairs = run_strategy_on_s1("char_3gram_k5", s1, gen)
        # S1-101 ('celestial memorial trust') matches S2-201 ('celestial memorial')
        assert ("S1-101", "S2-201") in pairs

    def test_generate_candidates_api(self, mock_datasets):
        s1, s2, s3 = mock_datasets
        cand_df = generate_candidates(s1, s2, s3)

        assert isinstance(cand_df, pd.DataFrame)
        expected_cols = {"source1_entity_id", "candidate_entity_id", "candidate_source", "blocking_sources"}
        assert expected_cols.issubset(set(cand_df.columns))

        # Check candidate sources
        assert set(cand_df["candidate_source"].unique()).issubset({"source2", "source3"})

        # Check provenance
        assert all(len(srcs) > 0 for srcs in cand_df["blocking_sources"])
