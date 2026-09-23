"""
Train/eval leakage check: which JITVul pairs appear in a PrimeVul split?
Run once per split; train is the one that matters (the adapter saw it).

    python -m test_scripts.check_overlap_jitvul --primevul data/primevul_train.jsonl

Three levels, strongest first:
  func_hash : JITVul's vulnerable function has the same PrimeVul func_hash
  body      : either half's body (whitespace-stripped) appears verbatim
  cve       : same CVE, possibly different functions (weaker, reported only)
Writes the idx of every pair matching at func_hash or body level to
results/overlap_<split>.txt, ready for full_metrics --exclude.
Near-duplicates (renamed variables, reformatted code) are NOT caught.
"""
import argparse
import json
import os
import re

from vulncost.data_loader import JitVulLoader

norm = lambda s: re.sub(r"\s+", "", s or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul", required=True)
    ap.add_argument("--jitvul", default="data/final_benchmark.jsonl")
    args = ap.parse_args()

    bodies, hashes, cves, n = set(), set(), set(), 0
    with open(args.primevul) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            bodies.add(norm(r.get("func") or r.get("function") or r.get("code")))
            if r.get("func_hash") is not None:
                hashes.add(str(r["func_hash"]))
            if r.get("cve"):
                cves.add(r["cve"])
    print(f"{args.primevul}: {n} functions, {len(cves)} CVEs")

    samples = list(JitVulLoader(args.jitvul, clean=True, dedupe=True).get_samples())
    hit_hash, hit_vuln, hit_safe, hit_cve, flagged = set(), set(), set(), set(), set()
    for vuln, _ in samples[0::2]:
        if str(vuln.func_hash) in hashes:
            hit_hash.add(vuln.idx)
        if norm(vuln.func) in bodies:
            hit_vuln.add(vuln.idx)
        if vuln.cve in cves:
            hit_cve.add(vuln.idx)
    for safe, _ in samples[1::2]:
        if norm(safe.func) in bodies:
            hit_safe.add(safe.idx)
    flagged = hit_hash | hit_vuln | hit_safe
    total = len(samples) // 2

    print(f"\nJITVul pairs: {total}")
    for name, s in [("vuln func_hash in split", hit_hash),
                    ("vuln body in split", hit_vuln),
                    ("patched body in split", hit_safe),
                    ("FLAGGED (any of the above)", flagged),
                    ("same CVE (not flagged)", hit_cve)]:
        print(f"  {name:<28}{len(s):>5}  ({len(s) / total:.1%})")

    os.makedirs("results", exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.primevul))[0]
    out = f"results/overlap_{stem}.txt"
    with open(out, "w") as f:
        f.writelines(f"{i}\n" for i in sorted(flagged, key=int))
    print(f"\nflagged idx -> {out}")


if __name__ == "__main__":
    main()