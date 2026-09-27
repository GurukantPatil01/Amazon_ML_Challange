"""Unit tests for the normalization module (src/normalize.py)."""

import pytest
import pandas as pd

from src.normalize import (
    extract_address_number_tokens,
    extract_house_number,
    extract_name_alnum,
    extract_name_core,
    extract_postal_code,
    extract_sorted_tokens,
    generate_char_3grams,
    normalize_address_base,
    normalize_business_address,
    normalize_business_name,
    normalize_business_name_base,
    normalize_country,
    normalize_dataframe,
    normalize_record,
    normalize_unicode_nfkc,
    strip_accents,
)


class TestUnicodeAndCasing:
    def test_unicode_nfkc_casing(self):
        assert normalize_unicode_nfkc("CAFÉ & CO") == "café & co"
        assert normalize_unicode_nfkc("  ABC  DEF  ") == "abc  def"
        assert normalize_unicode_nfkc(None) == ""
        assert normalize_unicode_nfkc("") == ""

    def test_strip_accents(self):
        assert strip_accents("café") == "cafe"
        assert strip_accents("maison de santé") == "maison de sante"
        assert strip_accents("bordeaux") == "bordeaux"
        assert strip_accents("") == ""


class TestPunctuationAndAmpersand:
    def test_ampersand_expansion(self):
        assert normalize_business_name_base("Barnes & Noble") == "barnes and noble"
        assert normalize_business_name_base("B & W Corp") == "b and w corp"
        assert normalize_business_name_base("Thermal & Fils SASU") == "thermal and fils sasu"

    def test_apostrophe_handling(self):
        assert normalize_business_name_base("McDonald's") == "mcdonalds"
        assert normalize_business_name_base("Orelee's Barbershop") == "orelees barbershop"
        assert normalize_business_name_base("Levi’s") == "levis"

    def test_hyphen_and_punctuation(self):
        assert normalize_business_name_base("7-Eleven") == "7 eleven"
        assert normalize_business_name_base("ABC, Inc.") == "abc inc"
        assert normalize_business_name_base("<< Team Ecole >>") == "team ecole"
        assert normalize_business_name_base("B+ Retail Inc") == "b retail inc"

    def test_whitespace_collapsing(self):
        assert normalize_business_name_base("  Too    Many   Spaces  ") == "too many spaces"


class TestLegalSuffixes:
    def test_name_core_stripping(self):
        # US / UK suffixes
        assert extract_name_core("abc corporation") == "abc"
        assert extract_name_core("abc inc") == "abc"
        assert extract_name_core("abc llc") == "abc"
        assert extract_name_core("custom wealth services llc") == "custom wealth services"

        # Indian suffixes
        assert extract_name_core("consulting nyasa nursing private limited") == "consulting nyasa nursing"
        assert extract_name_core("shree ram pvt ltd") == "shree ram"

        # French suffixes
        assert extract_name_core("thermal and fils sasu") == "thermal and fils"
        assert extract_name_core("saint herblain societe sarl") == "saint herblain societe"
        assert extract_name_core("elephant centre eurl") == "elephant centre"

    def test_safeguard_suffix_not_stripped_if_alone(self):
        # When name is only the suffix itself, do not reduce to empty string
        assert extract_name_core("the company") == "the"  # 'the' is retained
        assert extract_name_core("inc") == "inc"
        assert extract_name_core("corp") == "corp"


class TestTokenizationAndNgrams:
    def test_sorted_tokens(self):
        # Word order inversion
        assert extract_sorted_tokens("sofie greenman") == "greenman sofie"
        assert extract_sorted_tokens("greenman sofie") == "greenman sofie"
        assert extract_sorted_tokens("xx apex nippon") == "apex nippon xx"
        assert extract_sorted_tokens("xx nippon apex") == "apex nippon xx"

    def test_short_tokens_preserved(self):
        res = normalize_business_name("3M Company")
        assert "3m" in res["name_tokens"].split()
        res_a1 = normalize_business_name("A1 Cleaners")
        assert "a1" in res_a1["name_tokens"].split()

    def test_char_3grams(self):
        ngrams = generate_char_3grams("abc")
        # '  abc  ' -> '  a', ' ab', 'abc', 'bc ', 'c  '
        assert "abc" in ngrams.split()
        assert len(ngrams.split()) == 5
        assert generate_char_3grams("") == ""


class TestAddressNormalization:
    def test_abbreviation_standardization(self):
        # US thoroughfares
        assert normalize_address_base("1795 Westchester Drive") == "1795 westchester dr"
        assert normalize_address_base("702 N Street") == "702 n st"
        assert normalize_address_base("5559 Orville Avenue") == "5559 orville ave"
        assert normalize_address_base("Highway 101") == "hwy 101"

        # Units
        assert normalize_address_base("Suite 400") == "ste 400"
        assert normalize_address_base("Apartment 2B") == "apt 2b"
        assert normalize_address_base("Building 5, Floor 3") == "bldg 5 fl 3"

        # French terms
        assert normalize_address_base("154 BD du President Wilson") == "154 blvd du president wilson"
        assert normalize_address_base("21 Chemin des Reunis") == "21 chem des reunis"
        assert normalize_address_base("5 Impasse Jean Baptiste") == "5 imp jean baptiste"


class TestAddressComponentExtraction:
    def test_postal_code_extraction(self):
        # US ZIP (5 and 9 digit)
        assert extract_postal_code("Springfield, IL 62704") == "62704"
        assert extract_postal_code("New York, NY 10001-1234") == "10001-1234"

        # India PIN (6 digit)
        assert extract_postal_code("Gurgaon, Haryana 122001") == "122001"
        assert extract_postal_code("Bangalore, Karnataka 560001") == "560001"

        # France Code Postal (5 digit)
        assert extract_postal_code("12 Rue de la Paix, 75002 Paris") == "75002"
        assert extract_postal_code("Bordeaux, 33000") == "33000"

        # No postal code
        assert extract_postal_code("1795 Westchester Drive, High Point, NC") == ""
        assert extract_postal_code("") == ""
        assert extract_postal_code(None) == ""

    def test_house_number_extraction(self):
        # Leading numbers
        assert extract_house_number("1795 Westchester Drive") == "1795"
        assert extract_house_number("123 Main St") == "123"

        # Explicit prefix (Indian / international patterns)
        assert extract_house_number("No. 35, Brentwood Apartments") == "35"
        assert extract_house_number("Door No 4-4 C Ashwamegh") == "4-4"
        assert extract_house_number("Plot 12-B, Sector 14") == "12-B"

        # French patterns
        assert extract_house_number("5 bis Rue Pierre Dignac") == "5 bis"
        assert extract_house_number("20 Rue Parmentier") == "20"

        # Number preceding street
        assert extract_house_number("OH, Columbus, 5559 Orville Avenue") == "5559"

        # Empty / None
        assert extract_house_number("") == ""
        assert extract_house_number(None) == ""

    def test_numeric_tokens(self):
        nums = extract_address_number_tokens("Plot 12-B, Sector 14, PIN 122001")
        assert "12-B" in nums or "12" in nums
        assert "14" in nums
        assert "122001" in nums


class TestMissingValuesAndRobustness:
    def test_none_and_empty_inputs(self):
        rec = {
            "entity_id": "S2-0001",
            "business_name": None,
            "business_address": None,
            "country": None,
        }
        res = normalize_record(rec)
        assert res["business_name"] is None  # Original preserved
        assert res["name_norm"] == ""
        assert res["name_core"] == ""
        assert res["address_norm"] == ""
        assert res["postal_code"] == ""
        assert res["house_number"] == ""
        assert res["country_norm"] == ""

    def test_whitespace_only(self):
        rec = {
            "entity_id": "S3-0002",
            "business_name": "   ",
            "business_address": "\t\n  ",
            "country": "  US  ",
        }
        res = normalize_record(rec)
        assert res["name_norm"] == ""
        assert res["address_norm"] == ""
        assert res["country_norm"] == "us"


class TestDataFrameNormalization:
    def test_dataframe_preserves_raw_and_appends_columns(self):
        df = pd.DataFrame([
            {
                "entity_id": "S1-1",
                "business_name": "Acme Corp.",
                "business_address": "123 Main Street, Unit 4, Dallas, TX 75201",
                "country": "US",
            },
            {
                "entity_id": "S2-2",
                "business_name": "McDonald's & Sons",
                "business_address": None,
                "country": "France",
            },
        ])

        norm_df = normalize_dataframe(df)

        # Original columns must be preserved exactly
        assert list(norm_df["business_name"]) == ["Acme Corp.", "McDonald's & Sons"]
        assert pd.isna(norm_df.loc[1, "business_address"])
        assert list(norm_df["country"]) == ["US", "France"]

        # Normalized columns must exist
        assert norm_df.loc[0, "name_norm"] == "acme corp"
        assert norm_df.loc[0, "name_core"] == "acme"
        assert norm_df.loc[0, "address_norm"] == "123 main st unit 4 dallas tx 75201"
        assert norm_df.loc[0, "postal_code"] == "75201"
        assert norm_df.loc[0, "house_number"] == "123"

        # Missing address safely handled
        assert norm_df.loc[1, "address_norm"] == ""
        assert norm_df.loc[1, "postal_code"] == ""
        assert norm_df.loc[1, "house_number"] == ""
        assert norm_df.loc[1, "name_norm"] == "mcdonalds and sons"
        assert norm_df.loc[1, "country_norm"] == "france"
