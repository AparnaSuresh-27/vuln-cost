import json
import sys
from collections import defaultdict


def label_str(label: int) -> str:
    return "VULNERABLE" if label == 1 else "SAFE"


def load_calls(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def standard_metrics(records):
    tp = fp = tn = fn = 0
    correct = 0
    for r in records:
        pred_vuln = r["prediction"] == "VULNERABLE"
        true_vuln = r["label"] == 1
        if pred_vuln and true_vuln:
            tp += 1
        elif pred_vuln and not true_vuln:
            fp += 1
        elif not pred_vuln and not true_vuln:
            tn += 1
        else:
            fn += 1
        if r["prediction"] == label_str(r["label"]):
            correct += 1

    n = len(records)
    acc = correct / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return {
        "n": n, "accuracy": acc, "precision": precision, "recall": recall,
        "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def pairwise_metrics(records):
    pc = pv = pb = pr = 0
    pairs = 0
    for i in range(0, len(records) - 1, 2):
        a, b = records[i], records[i + 1]
        vuln = a if a["label"] == 1 else b
        safe = b if a["label"] == 1 else a
        vuln_pred = vuln["prediction"]
        safe_pred = safe["prediction"]
        pairs += 1
        if vuln_pred == "VULNERABLE" and safe_pred == "SAFE":
            pc += 1
        elif vuln_pred == "VULNERABLE" and safe_pred == "VULNERABLE":
            pv += 1
        elif vuln_pred == "SAFE" and safe_pred == "SAFE":
            pb += 1
        elif vuln_pred == "SAFE" and safe_pred == "VULNERABLE":
            pr += 1
    return {
        "pairs": pairs,
        "P-C": pc / pairs if pairs else 0,
        "P-V": pv / pairs if pairs else 0,
        "P-B": pb / pairs if pairs else 0,
        "P-R": pr / pairs if pairs else 0,
        "P-C_count": pc, "P-V_count": pv, "P-B_count": pb, "P-R_count": pr,
    }


def token_stats(records):
    inp = [r["input_tokens"] for r in records]
    out = [r["output_tokens"] for r in records]
    return {
        "total_input": sum(inp),
        "total_output": sum(out),
        "avg_input": sum(inp) / len(inp) if inp else 0,
        "avg_output": sum(out) / len(out) if out else 0,
    }


def report(name, records):
    m = standard_metrics(records)
    p = pairwise_metrics(records)
    t = token_stats(records)
    print(f"\n{'='*55}\n{name}  (n={m['n']})\n{'='*55}")
    print("Standard metrics:")
    print(f"  Accuracy : {m['accuracy']:.3f}")
    print(f"  Precision: {m['precision']:.3f}")
    print(f"  Recall   : {m['recall']:.3f}")
    print(f"  F1       : {m['f1']:.3f}")
    print(f"  Confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print("Pairwise metrics:")
    print(f"  P-C (both correct)      : {p['P-C']:.3f} ({p['P-C_count']}/{p['pairs']})")
    print(f"  P-V (only vuln correct) : {p['P-V']:.3f} ({p['P-V_count']}/{p['pairs']})")
    print(f"  P-B (both -> vuln)      : {p['P-B']:.3f} ({p['P-B_count']}/{p['pairs']})")
    print(f"  P-R (reversed)          : {p['P-R']:.3f} ({p['P-R_count']}/{p['pairs']})")
    print("Token cost:")
    print(f"  Input : {t['total_input']} total, {t['avg_input']:.0f}/sample")
    print(f"  Output: {t['total_output']} total, {t['avg_output']:.0f}/sample")
    return t


def main(path):
    rows = load_calls(path)
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["context_config"]].append(r)

    print(f"Loaded {len(rows)} records from {path}")
    print(f"Configs found: {dict((k, len(v)) for k, v in by_config.items())}")

    token_totals = {}
    for config, records in by_config.items():
        token_totals[config] = report(config, records)

    if "function_only" in token_totals and "plus_caller_callee" in token_totals:
        fo = token_totals["function_only"]["total_input"]
        cc = token_totals["plus_caller_callee"]["total_input"]
        print(f"\n{'='*55}")
        print(f"TOKEN COST RATIO (caller_callee / function_only): {cc/fo:.2f}x")
        print(f"{'='*55}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m vulncost.full_metrics <path/to/calls.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
    