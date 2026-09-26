"""Text and Address Normalization Module for Amazon Entity Resolution.

Planned Phase 1 components:
- Business name standardization (stripping legal abbreviations: INC, LLC, PVT LTD, CORP, etc.)
- Special character & punctuation cleaning
- Lowercasing and whitespace collapsing
- Address field normalization (standardizing St/Street, Rd/Road, Ave/Avenue, etc.)
- Multi-country compatibility (US, India, and unseen France)
"""

from typing import Optional
import re


def clean_text(text: Optional[str]) -> str:
    """Basic string cleaner for entity resolution text fields.

    Args:
        text: Raw input text.

    Returns:
        Stripped, lowercased string with collapsed whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: Optional[str]) -> str:
    """Normalizes business name strings.

    Args:
        name: Raw business name.

    Returns:
        Normalized business name.
    """
    return clean_text(name)


def normalize_business_address(address: Optional[str]) -> str:
    """Normalizes business address strings.

    Args:
        address: Raw address string.

    Returns:
        Normalized address string.
    """
    return clean_text(address)
