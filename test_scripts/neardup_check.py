"""
Near-duplicate leakage check for the v3 adapter.

v3's training set excludes every JITVul CVE and every JITVul body, but PrimeVul
often contains OTHER versions of the same function (other commits, other CVEs).
If v3 is recognising functions it saw in training rather than reading the code,
P-C should be much higher on JITVul pairs whose function (same project + same
function name) appears in the training set than on pairs whose function doesn't.

Rebuilds the exact v3 training set with the same function and arguments.

    python3 -m test_scripts.neardup_check logs/<ft3 run>/calls.jsonl \
        --paired data/primevul_train_paired.jsonl
"""
import argparse
import json
import re
from collections import defaultdict

from finetune_vd import build_data, _read, _func_target
from vulncost.data_loader import JitVulLoader

KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "defined"}


def func_name(code):
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", code or ""):
        if m.group(1) not in KEYWORDS:
            return m.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("--train", default="data/primevul_train.jsonl")
    ap.add_argument("--paired")
    ap.add_argument("--jitvul", default="data/final_benchmark.jsonl")
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--max-safe", type=int, default=12000)
    args = ap.parse_args()

    # normalised code -> projects, for every PrimeVul function
    proj_of = defaultdict(set)
    for path in [args.train] + ([args.paired] if args.paired else []):
        for r in _read(path):
            func, _ = _func_target(r)
            if func:
                proj_of[JitVulLoader.normalize(func)].add(r.get("project"))

    data, stats = build_data(args.train, args.oversample, args.max_safe,
                             args.jitvul, args.paired, True)
    seen = set()
    for ex in data:
        for p in proj_of.get(ex["code"], ()):
            seen.add((p, func_name(ex["code"])))
    print(f"training set rebuilt: {json.dumps(stats)}")
    print(f"distinct (project, function name) in training: {len(seen)}")

    rows = [json.loads(l) for l in open(args.calls) if l.strip()]
    fo = defaultdict(dict)
    for r in rows:
        if r["context_config"] == "function_only" and r["status"] == "ok":
            fo[r["sample_id"]][r["label"]] = r["prediction"]

    buckets = {True: [0, 0], False: [0, 0]}   # flagged -> [pairs, P-C]
    for pf, _ in JitVulLoader(args.jitvul, clean=True, dedupe=True,
                              normalize_ws=True).get_samples():
        if pf.target != 1 or len(fo.get(pf.idx, {})) != 2:
            continue
        flag = (pf.project, func_name(pf.func)) in seen
        buckets[flag][0] += 1
        buckets[flag][1] += fo[pf.idx][1] == "VULNERABLE" and fo[pf.idx][0] == "SAFE"

    print("\nfunction-only P-C by whether the same function (project + name) is in training:")
    for flag, (n, pc) in buckets.items():
        if n:
            print(f"  {'SEEN in training' if flag else 'not seen':<18} {pc:>4}/{n:<4} = {pc / n:.3f}")


if __name__ == "__main__":
    main()