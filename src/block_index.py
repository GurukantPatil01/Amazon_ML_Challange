"""High-performance inverted indexing engine for scalable entity resolution blocking.

Provides memory-compact inverted indices mapping blocking keys to entity IDs,
with document frequency (DF) tracking, high-frequency token suppression,
and country-partitioned candidate lookup.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import array


class InvertedIndex:
    """Memory-efficient inverted index with frequency filtering and country partitioning.

    Uses integer indexing internally and compact arrays/lists of integer IDs
    to minimize Python object overhead when indexing millions of records.
    """

    def __init__(
        self,
        name: str = "index",
        min_token_length: int = 2,
        max_df: Optional[int] = None,
        max_df_ratio: Optional[float] = 0.01,
        max_postings_per_key: int = 500,
    ) -> None:
        """Initializes the inverted index.

        Args:
            name: Identifier name for the index.
            min_token_length: Minimum character length for indexed string tokens.
            max_df: Absolute maximum document frequency allowed for a key.
            max_df_ratio: Maximum fraction of total documents allowed for a key.
            max_postings_per_key: Hard cap on postings list length to prevent blowout.
        """
        self.name = name
        self.min_token_length = min_token_length
        self.max_df = max_df
        self.max_df_ratio = max_df_ratio
        self.max_postings_per_key = max_postings_per_key

        # Master mapping: (country, key) -> list of integer record IDs
        self.index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        # Entity ID registry: int_id <-> str_id
        self.id_to_str: List[str] = []
        self.str_to_id: Dict[str, int] = {}
        # Document frequencies for tokens
        self.doc_frequencies: Counter = Counter()
        self.num_documents: int = 0
        self._pruned: bool = False

    def register_entity(self, entity_id: str) -> int:
        """Registers a string entity ID and returns its compact integer representation."""
        if entity_id in self.str_to_id:
            return self.str_to_id[entity_id]
        int_id = len(self.id_to_str)
        self.id_to_str.append(entity_id)
        self.str_to_id[entity_id] = int_id
        return int_id

    def get_entity_str(self, int_id: int) -> str:
        """Returns the original string entity ID for an internal integer ID."""
        return self.id_to_str[int_id]

    def add(self, country: str, key: str, entity_id: str) -> None:
        """Adds a single key posting to the index.

        Args:
            country: Normalized country string (e.g. 'us', 'india', 'france').
            key: Blocking key (e.g. normalized name, token, postal code).
            entity_id: Target entity identifier.
        """
        if not key or not country:
            return
        if len(key) < self.min_token_length:
            return

        int_id = self.register_entity(entity_id)
        self.index[(country, key)].append(int_id)
        self.doc_frequencies[key] += 1

    def finalize(self, total_documents: Optional[int] = None) -> None:
        """Prunes over-frequent postings lists and caps giant blocks to enforce selectivity."""
        if self._pruned:
            return
        self.num_documents = total_documents or len(self.id_to_str)
        effective_max_df = self.max_df
        if self.max_df_ratio is not None and self.num_documents > 0:
            ratio_df = max(1, int(self.max_df_ratio * self.num_documents))
            if effective_max_df is not None:
                effective_max_df = min(effective_max_df, ratio_df)
            else:
                effective_max_df = ratio_df

        if effective_max_df is not None or self.max_postings_per_key is not None:
            keys_to_remove = []
            for (country, key), postings in self.index.items():
                # Filter out overly common universal keys
                if effective_max_df is not None and len(postings) > effective_max_df:
                    keys_to_remove.append((country, key))
                elif self.max_postings_per_key and len(postings) > self.max_postings_per_key:
                    # Truncate to maximum cap
                    self.index[(country, key)] = postings[: self.max_postings_per_key]

            for k in keys_to_remove:
                del self.index[k]

        self._pruned = True

    def query(self, country: str, key: str) -> List[str]:
        """Queries the index for candidate string entity IDs sharing the (country, key)."""
        if not key or not country:
            return []
        int_ids = self.index.get((country, key), [])
        return [self.id_to_str[i] for i in int_ids]

    def query_int_ids(self, country: str, key: str) -> List[int]:
        """Queries the index returning raw integer IDs for high-speed inner loop operations."""
        if not key or not country:
            return []
        return self.index.get((country, key), [])

    def __len__(self) -> int:
        return len(self.index)
