"""Candidate Generation and Blocking Module for Amazon Entity Resolution.

Implements 9 scalable blocking strategies, multi-strategy unioning,
deduplication, and strategy provenance tracking without dense Cartesian matrices.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pandas as pd

from src.block_index import InvertedIndex
from src.config import PipelineConfig, DEFAULT_CONFIG
from src.normalize import (
    extract_name_core,
    generate_char_3grams,
    normalize_dataframe,
)
from src.utils import setup_logger, timer

logger = setup_logger("blocking")


@dataclass
class BlockingResult:
    """Holds generated candidate pairs and metadata."""
    pairs: Dict[Tuple[str, str], Set[str]]  # (s1_id, cand_id) -> set of strategy names
    strategy_name: str
    num_queries: int


class CandidateGenerator:
    """Manages inverted index building and candidate retrieval for a target source (S2 or S3)."""

    def __init__(
        self,
        target_name: str,
        target_df: pd.DataFrame,
        max_token_df_ratio: float = 0.005,  # Max 0.5% of dataset per token block
        max_postings_per_key: int = 500,
    ) -> None:
        """Initializes indices over the target pool (Source 2 or Source 3).

        Args:
            target_name: 'source2' or 'source3'.
            target_df: Normalized target DataFrame.
            max_token_df_ratio: Document frequency cutoff to avoid common token explosions.
            max_postings_per_key: Cap on maximum candidates retrieved per key.
        """
        self.target_name = target_name
        self.target_df = target_df
        self.total_records = len(target_df)

        logger.info(f"Initializing CandidateGenerator for {target_name} ({self.total_records:,} records)...")

        # 1. Exact Name & Core Name Indices
        self.idx_exact_name = InvertedIndex(f"{target_name}_exact_name", min_token_length=2, max_postings_per_key=1000)
        self.idx_name_core = InvertedIndex(f"{target_name}_name_core", min_token_length=2, max_postings_per_key=1000)

        # 2. Token Inverted Indices
        self.idx_name_token = InvertedIndex(
            f"{target_name}_name_token",
            min_token_length=3,
            max_df_ratio=max_token_df_ratio,
            max_postings_per_key=max_postings_per_key,
        )

        # 3. Character 3-Gram Index
        self.idx_char_3gram = InvertedIndex(
            f"{target_name}_char_3gram",
            min_token_length=3,
            max_df_ratio=0.05,  # 3-grams are naturally frequent
            max_postings_per_key=1000,
        )

        # 4. Postal Code Index
        self.idx_postal = InvertedIndex(f"{target_name}_postal", min_token_length=4, max_postings_per_key=500)

        # 5. Composite Blocks: (country, house_number, name_token)
        self.idx_house_name = InvertedIndex(f"{target_name}_house_name", min_token_length=2, max_postings_per_key=200)

        # 6. Address Token Index (selective)
        self.idx_addr_token = InvertedIndex(
            f"{target_name}_addr_token",
            min_token_length=4,
            max_df_ratio=0.002,  # Rare address tokens only
            max_postings_per_key=300,
        )

        # 7. Combined: (country, postal_code, name_token)
        self.idx_postal_name = InvertedIndex(f"{target_name}_postal_name", min_token_length=2, max_postings_per_key=200)

        self._build_indices()

    def _build_indices(self) -> None:
        """Populates all inverted indices in a single pass over target records using high-speed zipped iteration."""
        with timer(f"Building indices for {self.target_name}", logger):
            cids = self.target_df["entity_id"].tolist()
            countries = self.target_df["country_norm"].tolist()
            name_norms = self.target_df["name_norm"].tolist() if "name_norm" in self.target_df.columns else [""] * len(self.target_df)
            name_cores = self.target_df["name_core"].tolist() if "name_core" in self.target_df.columns else [""] * len(self.target_df)
            postals = self.target_df["postal_code"].tolist() if "postal_code" in self.target_df.columns else [""] * len(self.target_df)
            houses = self.target_df["house_number"].tolist() if "house_number" in self.target_df.columns else [""] * len(self.target_df)
            name_tok_strs = self.target_df["name_tokens"].tolist() if "name_tokens" in self.target_df.columns else [""] * len(self.target_df)
            char_3gram_strs = self.target_df["name_char_3gram"].tolist() if "name_char_3gram" in self.target_df.columns else [""] * len(self.target_df)
            addr_tok_strs = self.target_df["address_tokens"].tolist() if "address_tokens" in self.target_df.columns else [""] * len(self.target_df)

            for cid, country, name_norm, name_core, postal, house, name_tok_str, c_3gram_str, addr_tok_str in zip(
                cids, countries, name_norms, name_cores, postals, houses, name_tok_strs, char_3gram_strs, addr_tok_strs
            ):
                if not country:
                    continue

                # Strategy 1 & 2: Exact Name & Core
                if name_norm:
                    self.idx_exact_name.add(country, name_norm, cid)
                if name_core:
                    self.idx_name_core.add(country, name_core, cid)

                # Strategy 3 & 4: Name tokens
                name_toks = name_tok_str.split()
                for tok in set(name_toks):
                    if len(tok) >= 3:
                        self.idx_name_token.add(country, tok, cid)

                # Strategy 5: Character 3-grams
                c_3grams = c_3gram_str.split()
                for gram in set(c_3grams):
                    self.idx_char_3gram.add(country, gram, cid)

                # Strategy 6: Postal code
                if postal:
                    self.idx_postal.add(country, postal, cid)

                # Strategy 7: House number + rare/primary name token
                if house and name_toks:
                    first_sub = next((t for t in name_toks if len(t) >= 3), "")
                    if first_sub:
                        self.idx_house_name.add(country, f"{house}#{first_sub}", cid)

                # Strategy 8: Address tokens
                addr_toks = addr_tok_str.split()
                for atok in set(addr_toks):
                    if len(atok) >= 4 and not atok.isdigit():
                        self.idx_addr_token.add(country, atok, cid)

                # Strategy 9: Postal + name token
                if postal and name_toks:
                    first_sub = next((t for t in name_toks if len(t) >= 3), "")
                    if first_sub:
                        self.idx_postal_name.add(country, f"{postal}#{first_sub}", cid)

            # Finalize / prune over-frequent tokens
            self.idx_exact_name.finalize(self.total_records)
            self.idx_name_core.finalize(self.total_records)
            self.idx_name_token.finalize(self.total_records)
            self.idx_char_3gram.finalize(self.total_records)
            self.idx_postal.finalize(self.total_records)
            self.idx_house_name.finalize(self.total_records)
            self.idx_addr_token.finalize(self.total_records)
            self.idx_postal_name.finalize(self.total_records)

        logger.info(f"Indices built successfully for {self.target_name}.")

    # -----------------------------------------------------------------------
    # Individual Blocking Query Methods
    # -----------------------------------------------------------------------

    def block_exact_name(self, country: str, row: Dict[str, Any]) -> List[str]:
        """Strategy 1: Exact Normalized Name."""
        name_norm = row.get("name_norm", "")
        return self.idx_exact_name.query(country, name_norm)

    def block_exact_name_core(self, country: str, row: Dict[str, Any]) -> List[str]:
        """Strategy 2: Exact Name Core."""
        name_core = row.get("name_core", "")
        return self.idx_name_core.query(country, name_core)

    def block_name_tokens(self, country: str, row: Dict[str, Any], max_cands_per_query: int = 50) -> List[str]:
        """Strategy 3: Informative Name Tokens."""
        name_toks = [t for t in row.get("name_tokens", "").split() if len(t) >= 3]
        if not name_toks:
            return []
        candidates: List[str] = []
        seen: Set[str] = set()
        for tok in name_toks:
            for cid in self.idx_name_token.query(country, tok):
                if cid not in seen:
                    seen.add(cid)
                    candidates.append(cid)
                    if len(candidates) >= max_cands_per_query:
                        return candidates
        return candidates

    def block_rare_tokens(
        self,
        country: str,
        row: Dict[str, Any],
        top_k_tokens: int = 2,
        max_cands_per_query: int = 50,
    ) -> List[str]:
        """Strategy 4: Rare Token Intersection (highest IDF tokens)."""
        name_toks = [t for t in row.get("name_tokens", "").split() if len(t) >= 3]
        if not name_toks:
            return []

        # Sort tokens by document frequency ascending (rarest first)
        def token_frequency(tok: str) -> int:
            return self.idx_name_token.doc_frequencies.get(tok, 999_999)

        sorted_tokens = sorted(name_toks, key=token_frequency)[:top_k_tokens]

        candidates: List[str] = []
        seen: Set[str] = set()
        for tok in sorted_tokens:
            for cid in self.idx_name_token.query(country, tok):
                if cid not in seen:
                    seen.add(cid)
                    candidates.append(cid)
                    if len(candidates) >= max_cands_per_query:
                        return candidates
        return candidates

    def block_char_3gram(
        self,
        country: str,
        row: Dict[str, Any],
        top_k: int = 10,
    ) -> List[str]:
        """Strategy 5: Character 3-Gram Sparse Retrieval."""
        query_grams = set(row.get("name_char_3gram", "").split())
        if not query_grams:
            return []

        # Count shared 3-grams
        match_counts: Counter = Counter()
        for gram in query_grams:
            for int_id in self.idx_char_3gram.query_int_ids(country, gram):
                match_counts[int_id] += 1

        if not match_counts:
            return []

        # Return top-K candidates by shared gram count
        top_cands = [self.idx_char_3gram.get_entity_str(int_id) for int_id, _ in match_counts.most_common(top_k)]
        return top_cands

    def block_postal(self, country: str, row: Dict[str, Any], max_cands: int = 50) -> List[str]:
        """Strategy 6: Postal Code Block."""
        postal = row.get("postal_code", "")
        if not postal:
            return []
        return self.idx_postal.query(country, postal)[:max_cands]

    def block_house_and_name(self, country: str, row: Dict[str, Any], max_cands: int = 50) -> List[str]:
        """Strategy 7: House Number + Name Token."""
        house = row.get("house_number", "")
        name_toks = [t for t in row.get("name_tokens", "").split() if len(t) >= 3]
        if not house or not name_toks:
            return []
        first_sub = name_toks[0]
        return self.idx_house_name.query(country, f"{house}#{first_sub}")[:max_cands]

    def block_address_tokens(self, country: str, row: Dict[str, Any], max_cands: int = 30) -> List[str]:
        """Strategy 8: Address Token Blocking."""
        addr_toks = [t for t in row.get("address_tokens", "").split() if len(t) >= 4 and not t.isdigit()]
        if not addr_toks:
            return []
        candidates: List[str] = []
        seen: Set[str] = set()
        for atok in addr_toks:
            for cid in self.idx_addr_token.query(country, atok):
                if cid not in seen:
                    seen.add(cid)
                    candidates.append(cid)
                    if len(candidates) >= max_cands:
                        return candidates
        return candidates

    def block_combined_postal_name(self, country: str, row: Dict[str, Any], max_cands: int = 50) -> List[str]:
        """Strategy 9: Combined Name + Postal Code."""
        postal = row.get("postal_code", "")
        name_toks = [t for t in row.get("name_tokens", "").split() if len(t) >= 3]
        if not postal or not name_toks:
            return []
        first_sub = name_toks[0]
        return self.idx_postal_name.query(country, f"{postal}#{first_sub}")[:max_cands]


# ---------------------------------------------------------------------------
# Multi-Strategy Orchestration & Candidate Unions
# -----------------------------------------------------------------------

AVAILABLE_STRATEGIES: Dict[str, str] = {
    "exact_name": "Strategy 1: Exact Normalized Name",
    "name_core": "Strategy 2: Exact Name Core",
    "name_token": "Strategy 3: Informative Name Tokens",
    "rare_token": "Strategy 4: Rare Token Intersection (IDF)",
    "char_3gram_k5": "Strategy 5: Character 3-Gram (K=5)",
    "char_3gram_k10": "Strategy 5: Character 3-Gram (K=10)",
    "char_3gram_k20": "Strategy 5: Character 3-Gram (K=20)",
    "postal": "Strategy 6: Postal Code Block",
    "house_name": "Strategy 7: House Number + Name Token",
    "address_token": "Strategy 8: Selective Address Token",
    "combined_postal_name": "Strategy 9: Postal Code + Name Token",
}


def run_strategy_on_s1(
    strategy: str,
    s1_df: pd.DataFrame,
    generator: CandidateGenerator,
    char_k: int = 10,
) -> Dict[Tuple[str, str], Set[str]]:
    """Executes a specific blocking strategy for all S1 queries against a CandidateGenerator.

    Returns:
        Mapping from (s1_id, candidate_id) to set of strategy names.
    """
    pairs: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    s1_records = s1_df.to_dict(orient="records")

    for row in s1_records:
        s1_id = row["entity_id"]
        country = row.get("country_norm", "")
        if not country:
            continue

        cands: List[str] = []
        if strategy == "exact_name":
            cands = generator.block_exact_name(country, row)
        elif strategy == "name_core":
            cands = generator.block_exact_name_core(country, row)
        elif strategy == "name_token":
            cands = generator.block_name_tokens(country, row)
        elif strategy == "rare_token":
            cands = generator.block_rare_tokens(country, row, top_k_tokens=2)
        elif strategy.startswith("char_3gram"):
            k = char_k
            if "k5" in strategy:
                k = 5
            elif "k10" in strategy:
                k = 10
            elif "k20" in strategy:
                k = 20
            cands = generator.block_char_3gram(country, row, top_k=k)
        elif strategy == "postal":
            cands = generator.block_postal(country, row)
        elif strategy == "house_name":
            cands = generator.block_house_and_name(country, row)
        elif strategy == "address_token":
            cands = generator.block_address_tokens(country, row)
        elif strategy == "combined_postal_name":
            cands = generator.block_combined_postal_name(country, row)

        for cid in cands:
            pairs[(s1_id, cid)].add(strategy)

    return pairs


def generate_candidate_pairs(
    source1_df: pd.DataFrame,
    candidates_pool_df: pd.DataFrame,
    max_candidates_per_s1: int = 50,
    strategies: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Standardized blocking generator meeting Phase 2 interface.

    Returns:
        Mapping from source1_entity_id to candidate entity IDs list.
    """
    target_source = "source2" if "S2" in candidates_pool_df["entity_id"].iloc[0] else "source3"
    gen = CandidateGenerator(target_source, candidates_pool_df)

    active_strategies = strategies or ["exact_name", "name_core", "rare_token", "char_3gram_k10"]
    combined_pairs: Dict[str, List[str]] = defaultdict(list)
    seen_per_s1: Dict[str, Set[str]] = defaultdict(set)

    for strat in active_strategies:
        strat_pairs = run_strategy_on_s1(strat, source1_df, gen)
        for (s1_id, cid) in strat_pairs:
            if cid not in seen_per_s1[s1_id]:
                seen_per_s1[s1_id].add(cid)
                combined_pairs[s1_id].append(cid)

    # Enforce maximum candidates per S1
    for s1_id in list(combined_pairs.keys()):
        combined_pairs[s1_id] = combined_pairs[s1_id][:max_candidates_per_s1]

    return combined_pairs


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: Optional[PipelineConfig] = None,
    strategies: Optional[List[str]] = None,
    max_candidates_per_s1: int = 50,
) -> pd.DataFrame:
    """Public competition API for full end-to-end candidate generation.

    Generates candidates separately for S1 -> S2 and S1 -> S3 with country constraints,
    unions the candidates, tracks strategy provenance, and returns a deduplicated table.

    Args:
        source1: Source 1 DataFrame (deduplicated reference).
        source2: Source 2 DataFrame.
        source3: Source 3 DataFrame.
        config: Optional pipeline configuration.
        strategies: Optional list of blocking strategies to activate.
        max_candidates_per_s1: Cap on total candidates per S1 entity.

    Returns:
        DataFrame with columns:
        ['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocking_sources']
    """
    cfg = config or DEFAULT_CONFIG
    active_strategies = strategies or ["exact_name", "name_core", "rare_token", "char_3gram_k10", "house_name"]

    # Ensure normalized fields exist
    s1_norm = source1 if "name_norm" in source1.columns else normalize_dataframe(source1)
    s2_norm = source2 if "name_norm" in source2.columns else normalize_dataframe(source2)
    s3_norm = source3 if "name_norm" in source3.columns else normalize_dataframe(source3)

    records: List[Dict[str, Any]] = []

    for target_name, target_df in [("source2", s2_norm), ("source3", s3_norm)]:
        logger.info(f"Generating candidate pairs for S1 -> {target_name}...")
        gen = CandidateGenerator(target_name, target_df)

        pairs_with_sources: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for strat in active_strategies:
            strat_pairs = run_strategy_on_s1(strat, s1_norm, gen)
            for pair, strats in strat_pairs.items():
                pairs_with_sources[pair].update(strats)

        # Organize by S1 and apply per-entity cap
        s1_cands: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for (s1_id, cid), strats in pairs_with_sources.items():
            s1_cands[s1_id].append((cid, ",".join(sorted(strats))))

        for s1_id, cand_list in s1_cands.items():
            for cid, strat_str in cand_list[:max_candidates_per_s1]:
                records.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "candidate_source": target_name,
                    "blocking_sources": strat_str,
                })

    cand_df = pd.DataFrame(records)
    logger.info(f"Candidate generation complete: {len(cand_df):,} total candidate pairs generated.")
    return cand_df
