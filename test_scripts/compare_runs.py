"""
Compare conditions on the SAME pairs, with exact McNemar tests on P-C.

Each condition is name=path/to/calls.jsonl:context_config. Only pairs that are
complete and ok in EVERY condition are used, so rows are directly comparable.

    python3 -m test_scripts.compare_runs \
        ft_fo=logs/<ft_norm>/calls.jsonl:function_only \
        ft_cc=logs/<ft_norm>/calls.jsonl:plus_caller_callee \
        ft_placebo=logs/<placebo>/calls.jsonl:plus_caller_callee \
        base_fo=logs/<base_norm>/calls.jsonl:function_only \
        base_cc=logs/<base_norm>/calls.jsonl:plus_caller_callee \
        [--exclude results/overlap_primevul_train.txt]

McNemar: for two conditions A and B, b = pairs correct (P-C) under A only,
c = under B only. Exact two-sided p = 2 * P(X <= min(b, c)), X ~ Bin(b + c, 0.5).
"""
import argparse
import json
from collections import defaultdict
from math import comb

COMPARISONS = [("ft_fo", "ft_cc", "context, fine-tuned (total)"),
               ("ft_fo", "ft_placebo", "layout only, fine-tuned"),
               ("ft_placebo", "ft_cc", "context itself, fine-tuned"),
               ("base_fo", "base_cc", "context, base model"),
               ("base_fo", "ft_fo", "fine-tuning, function-only"),
               ("base_cc", "ft_cc", "fine-tuning, caller/callee")]


def load(spec):
    name, rest = spec.split("=", 1)
    path, config = rest.rsplit(":", 1)
    pairs = defaultdict(dict)
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["context_config"] != config:
            continue
        slot = pairs[r["sample_id"]]
        if r["status"] != "ok" or r["label"] in slot:
            slot["bad"] = True
        slot[r["label"]] = r
    good = {sid: (p[1], p[0]) for sid, p in pairs.items()
            if "bad" not in p and 1 in p and 0 in p}   # (vuln, safe)
    return name, good


def outcome(vuln, safe):
    v, s = vuln["prediction"], safe["prediction"]
    return {("VULNERABLE", "SAFE"): "P-C", ("VULNERABLE", "VULNERABLE"): "P-V",
            ("SAFE", "SAFE"): "P-B", ("SAFE", "VULNERABLE"): "P-R"}.get((v, s), "unres")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="+")
    ap.add_argument("--exclude")
    args = ap.parse_args()

    runs = dict(load(s) for s in args.conditions)
    common = set.intersection(*(set(g) for g in runs.values()))
    if args.exclude:
        common -= {l.strip() for l in open(args.exclude) if l.strip()}
    n = len(common)
    print(f"pairs common to all conditions: {n}\n")

    print(f"{'condition':<12}{'P-C':>12}{'P-V':>8}{'P-B':>8}{'P-R':>8}{'tok/sample':>12}")
    correct = {}
    for name, g in runs.items():
        outs = [outcome(*g[sid]) for sid in common]
        tok = sum(g[sid][0]["input_tokens"] + g[sid][1]["input_tokens"] for sid in common)
        correct[name] = {sid for sid in common if outcome(*g[sid]) == "P-C"}
        pc = outs.count("P-C")
        print(f"{name:<12}{pc / n:>7.3f} ({pc:>3}){outs.count('P-V') / n:>8.3f}"
              f"{outs.count('P-B') / n:>8.3f}{outs.count('P-R') / n:>8.3f}{tok / (2 * n):>12.0f}")

    print(f"\n{'comparison':<30}{'A':<12}{'B':<12}{'A only':>7}{'B only':>7}{'p':>8}")
    for a, b, label in COMPARISONS:
        if a in correct and b in correct:
            only_a = len(correct[a] - correct[b])
            only_b = len(correct[b] - correct[a])
            print(f"{label:<30}{a:<12}{b:<12}{only_a:>7}{only_b:>7}{mcnemar(only_a, only_b):>8.3f}")


if __name__ == "__main__":
    main()