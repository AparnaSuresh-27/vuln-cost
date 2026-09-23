"""
Full metrics for a calls.jsonl log: standard, pairwise (PrimeVul P-C/P-V/P-B/P-R)
and token cost per context_config.

    python -m vulncost.full_metrics logs/<run>/calls.jsonl
    python -m vulncost.full_metrics logs/<run>/calls.jsonl --exclude results/overlap_primevul_train.txt

Only COMPLETE pairs are scored: a sample_id counts only if every config has
exactly its two halves logged with status "ok". So a partial log (run still
going), a skipped over-length sample, or a duplicated idx drops the whole pair
from every config, and all configs are always compared on the same pairs.
"""
import argparse
import json
import re
from collections import Counter, defaultdict

NO_CONTEXT_MARKERS = ("// ---- Context: none resolved for this function ----",
                      "(no resolved caller/callee context)")  # new runs, old MLX runs


def label_str(label: int) -> str:
    return "VULNERABLE" if label == 1 else "SAFE"


def load_calls(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def complete_ids(rows, exclude):
    per = defaultdict(Counter)   # sample_id -> Counter(config -> ok records)
    bad = set()
    for r in rows:
        if r["status"] != "ok":
            bad.add(r["sample_id"])
        per[r["sample_id"]][r["context_config"]] += 1
    configs = {r["context_config"] for r in rows}
    ok = {sid for sid, c in per.items()
          if sid not in bad and all(c[k] == 2 for k in configs)}
    return ok - exclude, {"logged": len(per), "failed": len(bad),
                          "incomplete_or_duplicate": len(per) - len(bad) - len(ok - bad),
                          "excluded": len(ok & exclude)}


def pairs_of(records):
    out = []
    for i in range(0, len(records) - 1, 2):
        a, b = records[i], records[i + 1]
        assert a["sample_id"] == b["sample_id"], f"misaligned pair at row {i}"
        out.append((a, b) if a["label"] == 1 else (b, a))  # (vuln, safe)
    return out


def standard_metrics(records):
    tp = fp = tn = fn = unknown = correct = 0
    for r in records:
        if r["prediction"] == "UNKNOWN":
            unknown += 1        # counted as a non-VULNERABLE call below
        pred_vuln = r["prediction"] == "VULNERABLE"
        true_vuln = r["label"] == 1
        tp += pred_vuln and true_vuln
        fp += pred_vuln and not true_vuln
        tn += (not pred_vuln) and (not true_vuln)
        fn += (not pred_vuln) and true_vuln
        correct += r["prediction"] == label_str(r["label"])
    n = len(records)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"n": n, "accuracy": correct / n if n else 0, "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "unknown": unknown}


def pairwise_metrics(records):
    c = Counter()
    for vuln, safe in pairs_of(records):
        v, s = vuln["prediction"], safe["prediction"]
        if v == "VULNERABLE" and s == "SAFE":
            c["P-C"] += 1
        elif v == "VULNERABLE" and s == "VULNERABLE":
            c["P-V"] += 1
        elif v == "SAFE" and s == "SAFE":
            c["P-B"] += 1
        elif v == "SAFE" and s == "VULNERABLE":
            c["P-R"] += 1
        else:
            c["unresolved"] += 1  # a half is UNKNOWN
    pairs = len(records) // 2
    return {"pairs": pairs, **{k: c[k] for k in ("P-C", "P-V", "P-B", "P-R", "unresolved")}}


def token_stats(records):
    inp = [r["input_tokens"] for r in records]
    out = [r["output_tokens"] for r in records]
    return {"total_input": sum(inp), "total_output": sum(out),
            "avg_input": sum(inp) / len(inp) if inp else 0,
            "avg_output": sum(out) / len(out) if out else 0}


def report(name, records):
    m, p, t = standard_metrics(records), pairwise_metrics(records), token_stats(records)
    n = p["pairs"] or 1
    print(f"\n{'=' * 55}\n{name}  (n={m['n']}, {p['pairs']} pairs)\n{'=' * 55}")
    print(f"  Accuracy {m['accuracy']:.3f}  Precision {m['precision']:.3f}  "
          f"Recall {m['recall']:.3f}  F1 {m['f1']:.3f}")
    print(f"  Confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}"
          f"  (UNKNOWN answers: {m['unknown']})")
    print("  Pairwise:")
    print(f"    P-C both correct         : {p['P-C'] / n:.3f} ({p['P-C']}/{p['pairs']})")
    print(f"    P-V both -> VULNERABLE   : {p['P-V'] / n:.3f} ({p['P-V']}/{p['pairs']})")
    print(f"    P-B both -> SAFE         : {p['P-B'] / n:.3f} ({p['P-B']}/{p['pairs']})")
    print(f"    P-R reversed             : {p['P-R'] / n:.3f} ({p['P-R']}/{p['pairs']})")
    if p["unresolved"]:
        print(f"    unresolved (UNKNOWN)     : {p['unresolved']}")
    print(f"  Tokens: input {t['total_input']} ({t['avg_input']:.0f}/sample), "
          f"output {t['total_output']} ({t['avg_output']:.1f}/sample)")
    return t, p


def subset_ids(by_config):
    """Pair-level properties read straight from the logged prompts."""
    norm = lambda s: re.sub(r"\s+", "", s)
    identical, no_ctx = set(), set()
    for vuln, safe in pairs_of(by_config.get("function_only", [])):
        if norm(vuln["prompt"]) == norm(safe["prompt"]):
            identical.add(vuln["sample_id"])
    for vuln, safe in pairs_of(by_config.get("plus_caller_callee", [])):
        if all(any(m in r["prompt"] for m in NO_CONTEXT_MARKERS) for r in (vuln, safe)):
            no_ctx.add(vuln["sample_id"])
    return identical, no_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--exclude", help="file of sample_ids (one per line) to drop")
    args = ap.parse_args()

    rows = load_calls(args.path)
    exclude = set()
    if args.exclude:
        exclude = {l.strip() for l in open(args.exclude) if l.strip()}
    keep, why = complete_ids(rows, exclude)

    by_config = defaultdict(list)
    for r in rows:
        if r["sample_id"] in keep:
            by_config[r["context_config"]].append(r)

    print(f"Loaded {len(rows)} records from {args.path}")
    print(f"sample_ids logged {why['logged']} | scored {len(keep)} | dropped: failed "
          f"{why['failed']}, incomplete/duplicate {why['incomplete_or_duplicate']}, "
          f"excluded {why['excluded']}")

    totals = {cfg: report(cfg, recs)[0] for cfg, recs in by_config.items()}

    if "function_only" in totals and "plus_caller_callee" in totals:
        fo, cc = totals["function_only"]["total_input"], totals["plus_caller_callee"]["total_input"]
        print(f"\nTOKEN COST RATIO (caller_callee / function_only): {cc / fo:.2f}x")

        identical, no_ctx = subset_ids(by_config)
        n = len(keep)
        print(f"\nPair subsets (of {n} scored):")
        print(f"  identical target function in both halves: {len(identical)} "
              f"-> function-only P-C ceiling {(n - len(identical)) / n:.3f}")
        print(f"  no caller/callee context in either half : {len(no_ctx)}")
        print(f"\n{'subset':<30}{'pairs':>6}{'fo P-C':>9}{'cc P-C':>9}{'cc/fo tok':>11}")
        subsets = {"all": keep, "distinguishable targets": keep - identical,
                   "has context": keep - no_ctx,
                   "identical targets": identical}
        for name, ids in subsets.items():
            if not ids:
                continue
            row = []
            tok = []
            for cfg in ("function_only", "plus_caller_callee"):
                recs = [r for r in by_config[cfg] if r["sample_id"] in ids]
                row.append(pairwise_metrics(recs)["P-C"] / (len(recs) // 2))
                tok.append(token_stats(recs)["total_input"])
            print(f"{name:<30}{len(ids):>6}{row[0]:>9.3f}{row[1]:>9.3f}{tok[1] / tok[0]:>10.2f}x")


if __name__ == "__main__":
    main()