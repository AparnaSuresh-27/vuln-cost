"""
Metrics for the ReposVul ladder (unpaired data, so no P-C).

Per level: accuracy, balanced accuracy, precision, recall, F1, share predicted
VULNERABLE, with 95% bootstrap CIs (resampling samples, the same resample used for
every level so the comparison stays matched). Between levels: exact McNemar test on
per-sample correctness, and the token ratio vs function-only with a bootstrap CI.
Cost: mean input tokens, GPU-seconds per call, and (optionally) dollars.

Only complete cases are scored: a sample counts only if every level has a parsed
prediction, so all levels are compared on exactly the same samples.

    python3 -m vulncost.reposvul_metrics logs/<run>/calls.jsonl \
        --exclude results/reposvul_leak_function.txt results/reposvul_leak_commit.txt \
        --gpu-usd-per-hour 1.21
"""
import argparse
import json
import random
from collections import defaultdict
from math import comb

LEVEL_ORDER = ["function", "slice", "full"]


def load(path):
    by = defaultdict(dict)
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            by[r["idx"]][r["context_config"]] = r
    return by


def scores(recs):
    tp = sum(r["prediction"] == "VULNERABLE" and r["label"] == 1 for r in recs)
    fp = sum(r["prediction"] == "VULNERABLE" and r["label"] == 0 for r in recs)
    tn = sum(r["prediction"] == "SAFE" and r["label"] == 0 for r in recs)
    fn = sum(r["prediction"] == "SAFE" and r["label"] == 1 for r in recs)
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": n,
        "acc": (tp + tn) / n if n else 0.0,
        "bal_acc": (rec + spec) / 2,
        "prec": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "pred_vuln": (tp + fp) / n if n else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def ci(vals, alpha=0.05):
    v = sorted(vals)
    lo = v[int(alpha / 2 * len(v))]
    hi = v[int((1 - alpha / 2) * len(v)) - 1]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--gpu-usd-per-hour", type=float, default=None)
    ap.add_argument("--api-usd-per-mtok", type=float, default=None,
                    help="input price in USD per million tokens, for an API-equivalent cost")
    args = ap.parse_args()

    by = load(args.calls)
    excl = set()
    for p in args.exclude:
        excl |= {int(x) for x in open(p).read().split()}
    levels = [l for l in LEVEL_ORDER if any(l in v for v in by.values())]

    status = defaultdict(lambda: defaultdict(int))
    for v in by.values():
        for l, r in v.items():
            status[l][r["status"] if r["prediction"] not in (None, "UNPARSED") else
                      (r["status"] if r["status"] != "ok" else "unparsed")] += 1
    print("call status by level:", {l: dict(s) for l, s in status.items()})

    ids = sorted(i for i, v in by.items()
                 if i not in excl and all(l in v and v[l]["prediction"] in ("VULNERABLE", "SAFE")
                                          for l in levels))
    print(f"samples: {len(by)} logged, {len(excl & set(by))} excluded, "
          f"{len(ids)} complete cases across {levels}\n")
    if not ids:
        return

    point = {l: scores([by[i][l] for i in ids]) for l in levels}
    rng = random.Random(0)
    boots = defaultdict(lambda: defaultdict(list))
    tok_ratio_boot = defaultdict(list)
    for _ in range(args.boot):
        s = [rng.choice(ids) for _ in ids]
        for l in levels:
            m = scores([by[i][l] for i in s])
            for k in ("acc", "bal_acc", "f1", "recall", "pred_vuln"):
                boots[l][k].append(m[k])
        base = sum(by[i]["function"]["input_tokens"] for i in s)
        for l in levels:
            tok_ratio_boot[l].append(sum(by[i][l]["input_tokens"] for i in s) / base)

    print(f"{'level':9s} {'n':>4s} {'bal_acc [95% CI]':>22s} {'F1 [95% CI]':>22s} "
          f"{'recall':>7s} {'%predV':>7s}  tp fp tn fn")
    for l in levels:
        m = point[l]
        ba, f1 = ci(boots[l]["bal_acc"]), ci(boots[l]["f1"])
        print(f"{l:9s} {m['n']:4d} {m['bal_acc']:6.3f} [{ba[0]:.3f},{ba[1]:.3f}]     "
              f"{m['f1']:6.3f} [{f1[0]:.3f},{f1[1]:.3f}]     {m['recall']:6.3f} "
              f"{m['pred_vuln']:6.3f}  {m['tp']} {m['fp']} {m['tn']} {m['fn']}")

    print("\nMcNemar (exact) on per-sample correctness:")
    for a in range(len(levels)):
        for b in range(a + 1, len(levels)):
            la, lb = levels[a], levels[b]
            ca = [by[i][la]["prediction"] == ("VULNERABLE" if by[i][la]["label"] else "SAFE") for i in ids]
            cb = [by[i][lb]["prediction"] == ("VULNERABLE" if by[i][lb]["label"] else "SAFE") for i in ids]
            only_a = sum(x and not y for x, y in zip(ca, cb))
            only_b = sum(y and not x for x, y in zip(ca, cb))
            print(f"  {la} vs {lb}: right only in {la}={only_a}, only in {lb}={only_b}, "
                  f"p={mcnemar_exact(only_a, only_b):.4f}")

    print("\nCost per call (input tokens; ratio vs function-only with 95% CI):")
    for l in levels:
        toks = [by[i][l]["input_tokens"] for i in ids]
        lat = [by[i][l]["latency_s"] for i in ids if by[i][l]["latency_s"] is not None]
        mean_t = sum(toks) / len(toks)
        med_t = sorted(toks)[len(toks) // 2]
        r = sum(toks) / sum(by[i]["function"]["input_tokens"] for i in ids)
        lo, hi = ci(tok_ratio_boot[l])
        line = (f"  {l:9s} mean {mean_t:8.0f}  median {med_t:7d}  ratio {r:5.2f}x "
                f"[{lo:.2f},{hi:.2f}]")
        if lat:
            mlat = sum(lat) / len(lat)
            line += f"  GPU {mlat:6.2f}s/call"
            if args.gpu_usd_per_hour:
                line += f"  ${mlat * args.gpu_usd_per_hour / 3600 * 1000:6.3f}/1k calls (GPU)"
        if args.api_usd_per_mtok:
            line += f"  ${mean_t * args.api_usd_per_mtok / 1e6 * 1000:6.3f}/1k calls (API input)"
        print(line)


if __name__ == "__main__":
    main()
