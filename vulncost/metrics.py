"""
Pairwise metrics calculator for PrimeVul-style paired runs.

Reads a calls.jsonl log (one CallRecord per line, pairs on adjacent rows)
and reports the four PrimeVul pairwise outcomes:

  P-C  both correct            (vuln->VULNERABLE, benign->SAFE)
  P-V  both called vulnerable  (model over-predicts vulnerability)
  P-B  both called safe        (model misses the vulnerability)
  P-R  reversed                (vuln->SAFE, benign->VULNERABLE)

Any pair containing an UNKNOWN / unparseable prediction is counted
separately as UNRESOLVED, so the four real metrics stay honest.

Usage:
    python -m vulncost.metrics path/to/calls.jsonl
"""

import argparse
import json
from collections import Counter

VULN = "VULNERABLE"
SAFE = "SAFE"

METRICS = [
    ("P-C", "both correct"),
    ("P-V", "both called vulnerable"),
    ("P-B", "both called safe"),
    ("P-R", "predictions reversed"),
    ("UNRESOLVED", "contains UNKNOWN / bad pair"),
]


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def classify_pair(vuln_pred, benign_pred):
    """Given the prediction on the vulnerable function and on the benign
    function, return which of the four pairwise categories it falls into."""
    if vuln_pred == VULN and benign_pred == SAFE:
        return "P-C"
    if vuln_pred == VULN and benign_pred == VULN:
        return "P-V"
    if vuln_pred == SAFE and benign_pred == SAFE:
        return "P-B"
    if vuln_pred == SAFE and benign_pred == VULN:
        return "P-R"
    return "UNRESOLVED"  # at least one prediction was UNKNOWN / not V or S


def compute_metrics(records):
    counts = Counter()
    warnings = []
    total_pairs = 0

    # pairs are adjacent rows: (0,1), (2,3), ...
    for i in range(0, len(records) - 1, 2):
        a, b = records[i], records[i + 1]
        total_pairs += 1

        # identify the two halves by LABEL, not by position (more robust)
        pair = [a, b]
        vuln = [r for r in pair if r.get("label") == 1]
        benign = [r for r in pair if r.get("label") == 0]

        if len(vuln) != 1 or len(benign) != 1:
            counts["UNRESOLVED"] += 1
            warnings.append(
                f"  pair at rows {i},{i+1}: expected one vuln + one benign "
                f"(labels were {a.get('label')}, {b.get('label')}) "
                f"- pairing assumption may be broken"
            )
            continue

        counts[classify_pair(vuln[0]["prediction"], benign[0]["prediction"])] += 1

    if len(records) % 2 == 1:
        warnings.append(
            f"  odd record count ({len(records)}): last row has no pair, ignored"
        )

    return counts, total_pairs, warnings


def print_table(counts, total_pairs, source):
    print(f"\nPairwise metrics  -  {source}")
    print(f"Pairs: {total_pairs}   (denominator for all percentages below)\n")

    header = f"{'Metric':<11}{'Meaning':<30}{'Count':>6}{'%':>8}   Bar"
    print(header)
    print("-" * len(header))

    bar_width = 30  # chars for 100%
    for key, meaning in METRICS:
        n = counts.get(key, 0)
        pct = (100.0 * n / total_pairs) if total_pairs else 0.0
        bar = "#" * round(bar_width * pct / 100)
        print(f"{key:<11}{meaning:<30}{n:>6}{pct:>7.1f}%   {bar}")

    total_counted = sum(counts.get(k, 0) for k, _ in METRICS)
    print("-" * len(header))
    print(f"{'TOTAL':<11}{'':<30}{total_counted:>6}{100.0 if total_pairs else 0:>7.1f}%")

    resolved = total_pairs - counts.get("UNRESOLVED", 0)
    if resolved and counts.get("UNRESOLVED", 0):
        pc_resolved = 100.0 * counts.get("P-C", 0) / resolved
        print(f"\nP-C over resolved pairs only ({resolved}): {pc_resolved:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compute PrimeVul pairwise metrics from a calls.jsonl log.")
    parser.add_argument("results_file", help="path to a calls.jsonl run log")
    args = parser.parse_args()

    records = load_records(args.results_file)
    counts, total_pairs, warnings = compute_metrics(records)
    print_table(counts, total_pairs, args.results_file)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()