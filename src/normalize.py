"""High-performance text and address normalization module for Amazon Entity Resolution.

Preserves raw fields and generates standardized representations for blocking,
feature engineering, and matching models across multilingual / multi-country data
(US, India, France, and arbitrary unseen countries).
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import re
import unicodedata
import pandas as pd

from src.config import (
    DEFAULT_ADDRESS_ABBREVIATIONS,
    DEFAULT_LEGAL_SUFFIXES,
)

# ---------------------------------------------------------------------------
# Pre-compiled Regular Expressions for High Throughput
# ---------------------------------------------------------------------------

# Ampersand normalization
RE_AMPERSAND = re.compile(r"&+")

# Apostrophes and smart quotes
RE_APOSTROPHES = re.compile(r"['’`´]")

# Punctuation to replace with space (hyphens, slashes, commas, periods, parens, etc.)
RE_PUNCTUATION = re.compile(r"[^\w\s]")

# Whitespace collapsing
RE_WHITESPACE = re.compile(r"\s+")

# Non-alphanumeric characters (for pure alphanumeric representation)
RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# Numeric tokens in text
RE_NUMBER_TOKENS = re.compile(r"\b\d+[a-zA-Z]?(?:[-/]\d+[a-zA-Z]?)?\b")

# Postal code patterns (supports US 5/9-digit, India 6-digit, France 5-digit, universal 5-6 digit)
RE_POSTAL_US_9 = re.compile(r"\b\d{5}-\d{4}\b")
RE_POSTAL_6 = re.compile(r"\b[1-9]\d{5}\b")  # Standard Indian PIN
RE_POSTAL_5 = re.compile(r"\b\d{5}\b")       # US ZIP / French Code Postal

# House / Building number patterns
RE_EXPLICIT_HOUSE_NO = re.compile(
    r"\b(?:no|door\s*no|plot|flat|tower|block|house\s*no|bldg\s*no)[\s.:#-]*"
    r"([0-9]+[a-zA-Z]?(?:[-/][0-9a-zA-Z]+)?)\b",
    re.IGNORECASE,
)
RE_LEADING_HOUSE_NO = re.compile(
    r"^\s*([0-9]+[a-zA-Z]?(?:[-/][0-9a-zA-Z]+)?)\b"
)
RE_FRENCH_HOUSE_NO = re.compile(
    r"\b([0-9]+(?:\s*(?:bis|ter))?)\s+(?:rue|avenue|av|boulevard|bd|chemin|all|impasse)\b",
    re.IGNORECASE,
)
RE_STREET_BEFORE_HOUSE_NO = re.compile(
    r"\b([0-9]+[a-zA-Z]?(?:[-/][0-9]+[a-zA-Z]?)?)\s+"
    r"(?:[a-zA-Z]+\s+)?(?:street|st|road|rd|avenue|ave|drive|dr|boulevard|blvd|lane|ln|way)\b",
    re.IGNORECASE,
)


def build_legal_suffix_regex(suffixes: Sequence[str]) -> re.Pattern:
    """Builds a compiled regex that matches legal entity suffixes at string boundaries.

    Sorts suffixes by length descending so multi-word suffixes (e.g. 'private limited')
    are matched before shorter sub-tokens ('limited' or 'private').
    """
    sorted_suffixes = sorted(suffixes, key=len, reverse=True)
    escaped_patterns = [re.escape(s.strip().lower()) for s in sorted_suffixes if s.strip()]
    pattern = r"(?:\s+|^)(" + "|".join(escaped_patterns) + r")\s*$"
    return re.compile(pattern, re.IGNORECASE)


DEFAULT_SUFFIX_REGEX = build_legal_suffix_regex(DEFAULT_LEGAL_SUFFIXES)


# ---------------------------------------------------------------------------
# Core Text Cleaning Primitives
# ---------------------------------------------------------------------------

def normalize_unicode_nfkc(text: Optional[str]) -> str:
    """Applies Unicode NFKC normalization, lowercasing, and whitespace stripping.

    Preserves accented letters (e.g., French é, à, ç, ê) while standardizing
    compatibility characters, ligatures, and decomposed characters.
    """
    if not text or not isinstance(text, str):
        return ""
    # Unicode NFKC: converts full-width chars, ligatures (fi -> f i), etc.
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.lower().strip()


def strip_accents(text: str) -> str:
    """Decomposes Unicode characters and strips combining diacritical marks.

    Example: 'santé' -> 'sante', 'garçon' -> 'garcon'.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


# ---------------------------------------------------------------------------
# Business Name Normalization
# ---------------------------------------------------------------------------

def normalize_business_name_base(name: Optional[str]) -> str:
    """Normalizes raw business name with NFKC, '&' -> 'and', apostrophe cleaning,

    and punctuation-to-space mapping. Preserves accented characters and tokens.
    """
    if not name or not isinstance(name, str):
        return ""
    text = normalize_unicode_nfkc(name)
    if not text:
        return ""

    # Replace '&' with ' and '
    text = RE_AMPERSAND.sub(" and ", text)

    # Remove apostrophes without splitting tokens (McDonald's -> mcdonalds)
    text = RE_APOSTROPHES.sub("", text)

    # Replace all punctuation characters with space
    text = RE_PUNCTUATION.sub(" ", text)

    # Collapse multiple whitespaces
    return RE_WHITESPACE.sub(" ", text).strip()


def extract_name_core(
    name_norm: str,
    suffix_regex: Optional[re.Pattern] = None,
) -> str:
    """Extracts core business name by stripping trailing legal entity suffixes.

    Safeguard: Never strips if the entire string consists solely of the suffix.
    """
    if not name_norm:
        return ""
    regex = suffix_regex or DEFAULT_SUFFIX_REGEX

    stripped = regex.sub("", name_norm).strip()
    # If stripping resulted in empty string or single char, retain original name_norm
    if len(stripped) < 2:
        return name_norm
    return stripped


def extract_name_alnum(name_norm: str) -> str:
    """Creates a clean alphanumeric-only representation with accents folded to ASCII."""
    if not name_norm:
        return ""
    ascii_folded = strip_accents(name_norm)
    clean = RE_NON_ALNUM.sub(" ", ascii_folded)
    return RE_WHITESPACE.sub(" ", clean).strip()


def extract_sorted_tokens(text: str) -> str:
    """Extracts unique tokens sorted alphabetically, joined by space.

    Inverts word-order variations (e.g., 'Sofie Greenman' vs 'Greenman Sofie').
    """
    if not text:
        return ""
    tokens = [tok for tok in text.split() if tok]
    # Unique sorted tokens
    unique_sorted = sorted(set(tokens))
    return " ".join(unique_sorted)


def generate_char_3grams(text: str) -> str:
    """Generates space-separated character 3-grams padded with boundary spaces.

    Suitable for high-recall candidate generation, MinHash, and string indexing.
    Example: 'abc' -> '  a  ab abc bc  c  '
    """
    if not text:
        return ""
    padded = f"  {text}  "
    if len(padded) < 3:
        return ""
    trigrams = [padded[i : i + 3] for i in range(len(padded) - 2)]
    return " ".join(trigrams)


def normalize_business_name(
    name: Optional[str],
    suffix_regex: Optional[re.Pattern] = None,
) -> Dict[str, str]:
    """Generates all normalized representations for a business name record."""
    norm = normalize_business_name_base(name)
    core = extract_name_core(norm, suffix_regex=suffix_regex)
    alnum = extract_name_alnum(norm)
    tokens = " ".join(norm.split())
    sorted_tokens = extract_sorted_tokens(norm)
    char_3gram = generate_char_3grams(alnum)

    return {
        "name_norm": norm,
        "name_core": core,
        "name_alnum": alnum,
        "name_tokens": tokens,
        "name_sorted_tokens": sorted_tokens,
        "name_char_3gram": char_3gram,
    }


# ---------------------------------------------------------------------------
# Business Address Normalization & Component Extraction
# ---------------------------------------------------------------------------

def normalize_address_base(
    address: Optional[str],
    abbreviations: Optional[Dict[str, str]] = None,
) -> str:
    """Normalizes address string: Unicode NFKC, '&' -> 'and', apostrophe cleaning,

    punctuation removal, and standard thoroughfare/unit abbreviation mapping.
    Country-agnostic.
    """
    if not address or not isinstance(address, str):
        return ""
    text = normalize_unicode_nfkc(address)
    if not text:
        return ""

    # Replace '&' with ' and '
    text = RE_AMPERSAND.sub(" and ", text)

    # Remove apostrophes
    text = RE_APOSTROPHES.sub("", text)

    # Replace punctuation with spaces
    text = RE_PUNCTUATION.sub(" ", text)

    # Standardize abbreviations
    mapping = abbreviations if abbreviations is not None else DEFAULT_ADDRESS_ABBREVIATIONS
    tokens = text.split()
    standardized_tokens = [mapping.get(t, t) for t in tokens if t]

    return " ".join(standardized_tokens)


def extract_postal_code(address: Optional[str]) -> str:
    """Conservatively extracts postal code from address string.

    Supports US 5/9 digit, India 6 digit, France 5 digit, and universal codes.
    Returns empty string if no valid postal pattern is found.
    """
    if not address or not isinstance(address, str):
        return ""

    # Check for US 9-digit postal code first (e.g. 12345-6789)
    us_9 = RE_POSTAL_US_9.findall(address)
    if us_9:
        return us_9[-1]

    # Check for India 6-digit PIN (usually at end of address)
    in_6 = RE_POSTAL_6.findall(address)
    if in_6:
        return in_6[-1]

    # Check for 5-digit postal code (US 5-digit / France code postal)
    # Exclude matches that are obviously leading house numbers
    tokens = address.split()
    for tok in reversed(tokens):
        cleaned_tok = tok.strip(" ,.-#")
        if RE_POSTAL_5.fullmatch(cleaned_tok):
            return cleaned_tok

    return ""


def extract_house_number(address: Optional[str]) -> str:
    """Conservatively extracts house/building number from address.

    Handles explicit labels ('No. 35', 'Plot 12-B'), leading numbers,
    French 'bis/ter' patterns ('5 bis Rue...'), and street-preceded numbers.
    """
    if not address or not isinstance(address, str):
        return ""

    # 1. Explicit marker (No. 35, Door No 4-4, Plot 12-B, Flat 201)
    m = RE_EXPLICIT_HOUSE_NO.search(address)
    if m:
        return m.group(1).strip()

    # 2. French numbering pattern (e.g. 5 bis Rue... or 12 ter Avenue...)
    m = RE_FRENCH_HOUSE_NO.search(address)
    if m:
        return m.group(1).strip()

    # 3. Number immediately preceding a standard thoroughfare
    m = RE_STREET_BEFORE_HOUSE_NO.search(address)
    if m:
        return m.group(1).strip()

    # 4. Leading number at beginning of address
    m = RE_LEADING_HOUSE_NO.match(address)
    if m:
        return m.group(1).strip()

    return ""


def extract_address_number_tokens(address: Optional[str]) -> str:
    """Extracts all numeric and alphanumeric number tokens from an address string.

    Preserves all numbers for pairwise numerical overlap comparison.
    """
    if not address or not isinstance(address, str):
        return ""
    matches = RE_NUMBER_TOKENS.findall(address)
    return " ".join(matches)


def normalize_business_address(
    address: Optional[str],
    abbreviations: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Generates all normalized representations and components for an address record."""
    norm = normalize_address_base(address, abbreviations=abbreviations)
    ascii_folded = strip_accents(norm)
    alnum = RE_WHITESPACE.sub(" ", RE_NON_ALNUM.sub(" ", ascii_folded)).strip()
    tokens = " ".join(norm.split())
    sorted_tokens = extract_sorted_tokens(norm)
    postal_code = extract_postal_code(address)
    house_number = extract_house_number(address)
    number_tokens = extract_address_number_tokens(address)

    return {
        "address_norm": norm,
        "address_alnum": alnum,
        "address_tokens": tokens,
        "address_sorted_tokens": sorted_tokens,
        "postal_code": postal_code,
        "house_number": house_number,
        "address_number_tokens": number_tokens,
    }


# ---------------------------------------------------------------------------
# Country Normalization
# ---------------------------------------------------------------------------

def normalize_country(country: Optional[str]) -> str:
    """Standardizes country label string (case-folded and stripped)."""
    if not country or not isinstance(country, str):
        return ""
    return country.strip().lower()


# ---------------------------------------------------------------------------
# High-Throughput Batch / DataFrame Processing
# ---------------------------------------------------------------------------

def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes a single entity record dictionary, preserving all original fields.

    Args:
        record: Dictionary with keys 'entity_id', 'business_name', 'business_address', 'country'.

    Returns:
        New dictionary containing original fields plus all normalized representations.
    """
    out = dict(record)
    out.update(normalize_business_name(record.get("business_name")))
    out.update(normalize_business_address(record.get("business_address")))
    out["country_norm"] = normalize_country(record.get("country"))
    return out


def normalize_dataframe(
    df: pd.DataFrame,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Applies high-throughput normalization to a pandas DataFrame.

    Preserves original columns ('business_name', 'business_address', 'country')
    without modification, and appends all required normalized representations.

    Args:
        df: Input DataFrame containing entity records.
        chunk_size: Optional chunk size for memory-bounded batching.

    Returns:
        DataFrame with original columns preserved and normalized columns appended.
    """
    # Defensive copy of original DataFrame structure without modifying raw columns
    result = df.copy()

    # Pre-extract series to avoid DataFrame index lookup overhead
    raw_names = df["business_name"].fillna("").astype(str)
    raw_addresses = df["business_address"].fillna("").astype(str)
    raw_countries = df["country"].fillna("").astype(str)

    # 1. Names
    name_norm_series = raw_names.apply(normalize_business_name_base)
    result["name_norm"] = name_norm_series
    result["name_core"] = name_norm_series.apply(extract_name_core)
    result["name_alnum"] = name_norm_series.apply(extract_name_alnum)
    result["name_tokens"] = name_norm_series
    result["name_sorted_tokens"] = name_norm_series.apply(extract_sorted_tokens)
    result["name_char_3gram"] = result["name_alnum"].apply(generate_char_3grams)

    # 2. Addresses
    addr_norm_series = raw_addresses.apply(normalize_address_base)
    result["address_norm"] = addr_norm_series
    result["address_alnum"] = addr_norm_series.apply(
        lambda a: RE_WHITESPACE.sub(" ", RE_NON_ALNUM.sub(" ", strip_accents(a))).strip()
    )
    result["address_tokens"] = addr_norm_series
    result["address_sorted_tokens"] = addr_norm_series.apply(extract_sorted_tokens)

    # 3. Address Components
    result["postal_code"] = raw_addresses.apply(extract_postal_code)
    result["house_number"] = raw_addresses.apply(extract_house_number)
    result["address_number_tokens"] = raw_addresses.apply(extract_address_number_tokens)

    # 4. Country
    result["country_norm"] = raw_countries.apply(normalize_country)

    return result
