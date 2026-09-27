"""Phase 4 Evidence-Backed Candidate Generation Engine.

Extends the locked Phase 3 blocking system with four evidence-backed strategies
derived from rigorous error mining on the 53 validation candidate misses:
1. name_web_norm: URL/domain normalization (.com/.net/.org stripped, domain concatenation collapsed)
2. char_3gram_adaptive: Query-side adaptive character 3-gram retrieval (K=30/50 for difficult queries)
3. house_token: House number + any meaningful shared name token or address token
4. name_core_v2: Leading article and prefix-insensitive core name (the, a, an, m/s, t/a stripped)

Architecture:
- Preserves all Phase 3 blocks (Union E) unchanged.
- Multi-block support ranking with deterministic tie-breaking.
- Tracks full provenance: block_names, block_count, best_block, candidate_rank.
- Zero ground truth leakage: adaptive thresholds and query logic rely strictly on query-side observables.
"""

from collections import Counter, defaultdict
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from src.block_index import InvertedIndex
from src.blocking import CandidateGenerator, run_strategy_on_s1
from src.config import DEFAULT_CONFIG, PipelineConfig
from src.normalize import normalize_dataframe
from src.utils import setup_logger

logger = setup_logger("blocking_phase4")

# Extended strategy priority weights for Phase 4
STRATEGY_PRIORITY_PHASE4: Dict[str, float] = {
    "exact_name": 8.0,
    "name_core": 7.0,
    "name_core_v2": 6.5,
    "name_web_norm": 6.0,
    "house_name": 5.5,
    "house_token": 5.0,
    "combined_postal_name": 4.5,
    "rare_token": 3.5,
    "char_3gram_adaptive": 2.5,
    "char_3gram_k10": 2.0,
    "address_token": 1.0,
}

LEGAL_STOPWORDS = {
    "limited", "private", "pvt", "ltd", "inc", "corp", "corporation",
    "llc", "llp", "co", "company", "enterprises", "services", "industries",
    "group", "holdings", "trading", "retail", "solutions",
}

COMMON_DOMAIN_SUFFIXES = {"com", "net", "org", "co", "in", "us", "io", "biz", "info"}
LEADING_ARTICLES_PREFIXES = {"the", "a", "an", "ms", "m/s", "ta", "t/a"}


def compute_name_web_norm(text: str) -> str:
    """Computes experimental web/URL normalized representation.

    Replaces URL separators, strips standalone domain suffixes, and collapses spaces.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    # Replace URL separators and punctuation with space
    text = re.sub(r"[\./\-_&]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # Strip common domain suffixes
    tokens = [t for t in tokens if t not in COMMON_DOMAIN_SUFFIXES]
    # Strip legal stopwords
    tokens = [t for t in tokens if t not in LEGAL_STOPWORDS]
    return " ".join(tokens)


def compute_name_web_collapsed(text: str) -> str:
    """Computes collapsed web string without whitespace (e.g. fortuneprecisionhigh)."""
    norm = compute_name_web_norm(text)
    return re.sub(r"\s+", "", norm)


def compute_name_core_v2(name_core: str) -> str:
    """Strips leading articles and corporate prefixes from name_core."""
    if not name_core:
        return ""
    tokens = name_core.split()
    if tokens and tokens[0] in LEADING_ARTICLES_PREFIXES:
        tokens = tokens[1:]
    return " ".join(tokens)


class CandidateGeneratorPhase4(CandidateGenerator):
    """Extended candidate generator supporting Phase 4 evidence-backed blocking strategies."""

    def __init__(self, target_name: str, target_df: pd.DataFrame, config: Optional[PipelineConfig] = None):
        # Add experimental fields if missing
        df = target_df.copy()
        if "name_web_collapsed" not in df.columns:
            df["name_web_collapsed"] = [compute_name_web_collapsed(n) for n in df["business_name"]]
        if "name_core_v2" not in df.columns:
            df["name_core_v2"] = [compute_name_core_v2(c) for c in df.get("name_core", df["business_name"])]

        # Initialize base CandidateGenerator
        super().__init__(target_name, df)

        # Build additional indices for Phase 4
        self._build_phase4_indices(df)

    def _build_phase4_indices(self, df: pd.DataFrame) -> None:
        """Builds specialized inverted indices for new Phase 4 strategies."""
        logger.info(f"Building specialized Phase 4 inverted indices for {self.target_name}...")
        self.idx_web_collapsed = InvertedIndex(f"{self.target_name}_web_collapsed", max_df_ratio=0.005)
        self.idx_name_core_v2 = InvertedIndex(f"{self.target_name}_core_v2", max_df_ratio=0.005)
        self.idx_house_token = InvertedIndex(f"{self.target_name}_house_token", max_df_ratio=0.005)

        countries = df["country_norm"].tolist()
        e_ids = df["entity_id"].tolist()
        webs = df["name_web_collapsed"].tolist()
        core_v2s = df["name_core_v2"].tolist()
        houses = df.get("house_number", [""] * len(df)).tolist()
        name_tokens_list = df.get("name_tokens", [""] * len(df)).tolist()
        addr_tokens_list = df.get("address_norm", [""] * len(df)).tolist()

        for c, eid, web, core2, house, n_toks_str, a_toks_str in zip(
            countries, e_ids, webs, core_v2s, houses, name_tokens_list, addr_tokens_list
        ):
            if web:
                self.idx_web_collapsed.add(c, web, eid)
            if core2:
                self.idx_name_core_v2.add(c, core2, eid)
            if house:
                # Add house + meaningful name tokens
                n_toks = [t for t in n_toks_str.split() if len(t) >= 3 and t not in LEGAL_STOPWORDS]
                for tok in n_toks[:4]:
                    self.idx_house_token.add(c, f"{house}_{tok}", eid)
                # Add house + address tokens (for multi-script or co-located matches)
                a_toks = [t for t in a_toks_str.split() if len(t) >= 4 and not t.isdigit()]
                for tok in a_toks[:2]:
                    self.idx_house_token.add(c, f"addr_{house}_{tok}", eid)

        self.idx_web_collapsed.finalize()
        self.idx_name_core_v2.finalize()
        self.idx_house_token.finalize()
        logger.info(f"Phase 4 indices built successfully for {self.target_name}.")

    def block_web_norm(self, country: str, row: Dict[str, Any], max_cands: int = 50) -> List[str]:
        """Strategy: URL/Domain-collapsed Exact Match."""
        web = compute_name_web_collapsed(row.get("business_name", ""))
        if not web or len(web) < 4:
            return []
        return self.idx_web_collapsed.query(country, web)[:max_cands]

    def block_name_core_v2(self, country: str, row: Dict[str, Any], max_cands: int = 50) -> List[str]:
        """Strategy: Leading Article / Prefix-Insensitive Name Core."""
        core2 = compute_name_core_v2(row.get("name_core", ""))
        if not core2:
            return []
        return self.idx_name_core_v2.query(country, core2)[:max_cands]

    def block_house_token(self, country: str, row: Dict[str, Any], max_cands: int = 50) -> List[str]:
        """Strategy: House Number + Any Shared Name Token or Address Token."""
        house = row.get("house_number", "")
        if not house:
            return []
        cands: List[str] = []
        seen: Set[str] = set()

        # Try house + name tokens
        n_toks = [t for t in row.get("name_tokens", "").split() if len(t) >= 3 and t not in LEGAL_STOPWORDS]
        for tok in n_toks[:4]:
            key = f"{house}_{tok}"
            for cid in self.idx_house_token.query(country, key):
                if cid not in seen:
                    seen.add(cid)
                    cands.append(cid)
                    if len(cands) >= max_cands:
                        return cands

        # Try house + address tokens
        a_toks = [t for t in row.get("address_norm", "").split() if len(t) >= 4 and not t.isdigit()]
        for tok in a_toks[:2]:
            key = f"addr_{house}_{tok}"
            for cid in self.idx_house_token.query(country, key):
                if cid not in seen:
                    seen.add(cid)
                    cands.append(cid)
                    if len(cands) >= max_cands:
                        return cands

        return cands

    def block_char_3gram_adaptive(
        self,
        country: str,
        row: Dict[str, Any],
        base_k: int = 10,
        boosted_k: int = 35,
    ) -> List[str]:
        """Strategy: Query-Side Adaptive Character 3-Gram Retrieval.

        Increases retrieval depth to boosted_k strictly based on query-side observable features:
        - Query name has >= 8 characters
        - Query has low token count or rare character composition
        """
        name = row.get("name_norm", "")
        # Query-side trigger: longer names with potential character distortion
        use_boost = len(name) >= 8 and (len(name.split()) <= 2 or not name.isalnum())
        k = boosted_k if use_boost else base_k
        return self.block_char_3gram(country, row, top_k=k)


def run_strategy_phase4_on_s1(
    strategy: str,
    s1_df: pd.DataFrame,
    generator: CandidateGeneratorPhase4,
) -> Dict[Tuple[str, str], Set[str]]:
    """Runs a specific blocking strategy for all S1 queries against CandidateGeneratorPhase4."""
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
        elif strategy == "rare_token":
            cands = generator.block_rare_tokens(country, row, top_k_tokens=2)
        elif strategy == "char_3gram_k10":
            cands = generator.block_char_3gram(country, row, top_k=10)
        elif strategy == "char_3gram_k20":
            cands = generator.block_char_3gram(country, row, top_k=20)
        elif strategy == "char_3gram_k50":
            cands = generator.block_char_3gram(country, row, top_k=50)
        elif strategy == "char_3gram_adaptive":
            cands = generator.block_char_3gram_adaptive(country, row, base_k=10, boosted_k=35)
        elif strategy == "house_name":
            cands = generator.block_house_and_name(country, row)
        elif strategy == "combined_postal_name":
            cands = generator.block_combined_postal_name(country, row)
        elif strategy == "address_token":
            cands = generator.block_address_tokens(country, row)
        elif strategy == "name_web_norm":
            cands = generator.block_web_norm(country, row)
        elif strategy == "name_core_v2":
            cands = generator.block_name_core_v2(country, row)
        elif strategy == "house_token":
            cands = generator.block_house_token(country, row)

        for cid in cands:
            pairs[(s1_id, cid)].add(strategy)

    return pairs


def generate_candidates_phase4(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    budget: int = 200,
    active_strategies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Production candidate generation interface for Phase 4 meeting all contract specifications."""
    default_strategies = [
        "exact_name",
        "name_core",
        "rare_token",
        "char_3gram_k10",
        "house_name",
        "combined_postal_name",
        "address_token",
        "name_web_norm",
        "name_core_v2",
        "house_token",
        "char_3gram_adaptive",
    ]
    strategies = active_strategies or default_strategies

    s1_norm = source1_df if "name_norm" in source1_df.columns else normalize_dataframe(source1_df)
    s2_norm = source2_df if "name_norm" in source2_df.columns else normalize_dataframe(source2_df)
    s3_norm = source3_df if "name_norm" in source3_df.columns else normalize_dataframe(source3_df)

    output_rows: List[Dict[str, Any]] = []

    for target_name, target_df in [("source2", s2_norm), ("source3", s3_norm)]:
        logger.info(f"Generating Phase 4 candidate pool for S1 -> {target_name} ({len(target_df):,} records)...")
        gen = CandidateGeneratorPhase4(target_name, target_df)

        s1_candidate_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        for strat in strategies:
            strat_pairs = run_strategy_phase4_on_s1(strat, s1_norm, gen)
            for (s1_id, cid), strats in strat_pairs.items():
                s1_candidate_map[s1_id][cid].update(strats)

        for s1_id, cand_dict in s1_candidate_map.items():
            cand_items = []
            for cid, sources in cand_dict.items():
                support_count = len(sources)
                strength_sum = sum(STRATEGY_PRIORITY_PHASE4.get(s, 1.0) for s in sources)
                max_strength = max(STRATEGY_PRIORITY_PHASE4.get(s, 1.0) for s in sources)
                best_block = max(sources, key=lambda s: STRATEGY_PRIORITY_PHASE4.get(s, 1.0))
                cand_items.append((cid, support_count, strength_sum, max_strength, best_block, sources))

            # Multi-block support ranking
            cand_items.sort(key=lambda x: (x[1], x[2], x[3], x[0]), reverse=True)
            pruned_cands = cand_items[:budget]

            for rank_idx, (cid, sup_count, str_sum, _, best_block, sources) in enumerate(pruned_cands, start=1):
                output_rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "candidate_source": target_name,
                    "block_names": "|".join(sorted(sources)),
                    "blocking_sources": ",".join(sorted(sources)),
                    "block_count": sup_count,
                    "best_block": best_block,
                    "candidate_rank": rank_idx,
                    "rank": rank_idx,
                })

    return pd.DataFrame(output_rows)
