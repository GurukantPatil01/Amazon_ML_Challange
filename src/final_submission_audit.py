"""Final Submission Audit Script for Amazon ML Challenge 2026.

Performs rigorous consistency and compliance checks across:
1. matching_results.tsv
2. candidate_pairs.tsv
3. dataset/test/test_source1.tsv, test_source2.tsv, test_source3.tsv
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple


def audit_submission(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
) -> bool:
    print("=" * 70)
    print("STARTING FINAL SUBMISSION AUDIT")
    print("=" * 70)

    all_passed = True
    errors = []
    warnings = []

    # 1. Load Test S1 IDs
    s1_test_path = test_dir / "test_source1.tsv"
    if not s1_test_path.exists():
        print(f"FAIL: {s1_test_path} does not exist.")
        return False

    print("Loading test Source 1 IDs...")
    test_s1_ids = set()
    with open(s1_test_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.strip()
            if line:
                test_s1_ids.add(line.split("\t", 1)[0].strip())
    print(f"Total Test S1 Entities: {len(test_s1_ids):,}")

    # 2. Audit Matching Results File
    print(f"\nAuditing Matching Results: {matching_path}...")
    if not matching_path.exists():
        errors.append(f"Matching file not found: {matching_path}")
        return False

    matching_s1_seen = set()
    matching_predictions: Dict[str, Set[str]] = {}
    total_matching_pairs = 0
    empty_matches = 0

    with open(matching_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            errors.append("Matching file is empty.")
            return False
        header = [c.strip().lower() for c in header_line.rstrip("\n").split("\t")]
        if header != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"Matching file invalid header: {header}. Expected ['source1_entity_id', 'matched_entity_ids']")

        for line_idx, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 1:
                continue
            s1 = parts[0].strip()
            if s1 in matching_s1_seen:
                errors.append(f"Duplicate S1 ID in matching file at line {line_idx}: {s1}")
            matching_s1_seen.add(s1)

            matched_str = parts[1].strip() if len(parts) > 1 else ""
            if not matched_str:
                empty_matches += 1
                matching_predictions[s1] = set()
            else:
                targets = [t.strip() for t in matched_str.split(",") if t.strip()]
                # Check intra-row duplicates
                if len(targets) != len(set(targets)):
                    errors.append(f"Duplicate target IDs in row {line_idx} for S1 {s1}")
                # Check for S1 self-matches
                for t in targets:
                    if t.startswith("S1") or t == s1:
                        errors.append(f"Illegal S1 target match in row {line_idx}: {t}")
                    elif not (t.startswith("S2_") or t.startswith("S3_") or "S2" in t or "S3" in t):
                        errors.append(f"Invalid target prefix in row {line_idx}: {t}")
                matching_predictions[s1] = set(targets)
                total_matching_pairs += len(targets)

    print(f"- Total Matching Rows: {len(matching_s1_seen):,}")
    print(f"- Empty Matches (Singletons): {empty_matches:,} ({empty_matches / len(matching_s1_seen) * 100:.2f}%)")
    print(f"- Non-Empty Predicted Matches: {total_matching_pairs:,}")

    # Check S1 coverage
    if len(matching_s1_seen) != len(test_s1_ids):
        errors.append(f"Matching row count ({len(matching_s1_seen):,}) does not match Test S1 count ({len(test_s1_ids):,})")
    missing_s1 = test_s1_ids - matching_s1_seen
    if missing_s1:
        errors.append(f"Missing {len(missing_s1)} Test S1 entities in matching file!")

    # 3. Audit Candidate Pairs File
    print(f"\nAuditing Candidate Pairs: {candidate_path}...")
    candidate_s1_seen = set()
    candidate_map: Dict[str, Set[str]] = {}
    total_candidate_pairs = 0

    if not candidate_path.exists():
        errors.append(f"Candidate pairs file not found: {candidate_path}")
        return False

    with open(candidate_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            errors.append("Candidate file is empty.")
            return False
        header = [c.strip().lower() for c in header_line.rstrip("\n").split("\t")]
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"Candidate file invalid header: {header}. Expected ['source1_entity_id', 'candidate_entity_ids']")

        for line_idx, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 1:
                continue
            s1 = parts[0].strip()
            if s1 in candidate_s1_seen:
                errors.append(f"Duplicate S1 ID in candidate file at line {line_idx}: {s1}")
            candidate_s1_seen.add(s1)

            cands_str = parts[1].strip() if len(parts) > 1 else ""
            if cands_str:
                targets = [t.strip() for t in cands_str.split(",") if t.strip()]
                if len(targets) != len(set(targets)):
                    errors.append(f"Duplicate candidate IDs in row {line_idx} for S1 {s1}")
                candidate_map[s1] = set(targets)
                total_candidate_pairs += len(targets)
            else:
                candidate_map[s1] = set()

    print(f"- Total Candidate Rows: {len(candidate_s1_seen):,}")
    print(f"- Total Candidate Edges: {total_candidate_pairs:,}")
    if candidate_s1_seen:
        print(f"- Avg Candidates / S1: {total_candidate_pairs / len(candidate_s1_seen):.2f}")

    if len(candidate_s1_seen) != len(test_s1_ids):
        errors.append(f"Candidate row count ({len(candidate_s1_seen):,}) does not match Test S1 count ({len(test_s1_ids):,})")

    # 4. Cross-File Consistency: Every Prediction must be in Candidates
    print("\nAuditing Cross-File Consistency (Predictions ⊆ Candidates)...")
    violations = 0
    violation_examples = []
    for s1, preds in matching_predictions.items():
        cands = candidate_map.get(s1, set())
        diff = preds - cands
        if diff:
            violations += len(diff)
            if len(violation_examples) < 5:
                violation_examples.append(f"S1 {s1} has predictions not in candidate set: {diff}")

    if violations > 0:
        errors.append(f"Found {violations} predicted matches that DO NOT EXIST in candidate set!")
        for ex in violation_examples:
            errors.append(f"  Example: {ex}")
    else:
        print("PASS: 100% of predicted matches are contained within the candidate set!")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUBMISSION AUDIT SUMMARY")
    print("=" * 70)
    if errors:
        print(f"AUDIT RESULT: FAIL ({len(errors)} errors found)")
        for e in errors[:10]:
            print(f"  [ERROR] {e}")
        return False
    else:
        print("AUDIT RESULT: PASS")
        print("  - Matching Results: VALID & 100% S1 Coverage")
        print("  - Candidate Pairs: VALID & 100% S1 Coverage")
        print("  - Cross-File Consistency: 100% Predictions ∈ Candidates")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Final ER Submission Files")
    parser.add_argument("--matching", type=Path, default=Path("output/matching_results.tsv"), help="Path to matching results TSV")
    parser.add_argument("--candidate", type=Path, default=Path("output/candidate_pairs.tsv"), help="Path to candidate pairs TSV")
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"), help="Path to test dataset directory")

    args = parser.parse_args()
    success = audit_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
