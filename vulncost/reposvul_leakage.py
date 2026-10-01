"""
Leakage check: does any ReposVul sample appear in the detector's training data?

Two levels, same logic as the JITVul check:
  function-level  the extracted target function matches a PrimeVul train function
                  after removing all whitespace
  commit-level    the ReposVul commit hash (from its url) is a PrimeVul train commit

    python3 -m vulncost.reposvul_leakage \
        --train data/primevul_train.jsonl data/primevul_train_paired.jsonl

Writes results/reposvul_leak_function.txt and results/reposvul_leak_commit.txt
(one idx per line) for the metrics script's --exclude option.
"""
import argparse
import json
import os
import re


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/reposvul_ladder.jsonl")
    ap.add_argument("--train", nargs="+", required=True)
    args = ap.parse_args()

    funcs, commits = set(), set()
    for path in args.train:
        n = 0
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            funcs.add(squash(r.get("func") or r.get("function") or r.get("code") or ""))
            c = r.get("commit_id")
            if c:
                commits.add(c.lower()[:12])
            n += 1
        print(f"{path}: {n} records")

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    f_leak, c_leak = [], []
    for r in rows:
        if squash(r["function"]) in funcs:
            f_leak.append(r["idx"])
        m = re.search(r"/commit/([0-9a-f]{7,40})", r.get("url") or "")
        if m and m.group(1).lower()[:12] in commits:
            c_leak.append(r["idx"])

    os.makedirs("results", exist_ok=True)
    for name, ids in [("function", f_leak), ("commit", c_leak)]:
        with open(f"results/reposvul_leak_{name}.txt", "w") as f:
            f.write("\n".join(map(str, ids)) + ("\n" if ids else ""))
    both = set(f_leak) | set(c_leak)
    print(f"samples: {len(rows)}")
    print(f"function-level overlap: {len(f_leak)}")
    print(f"commit-level overlap:   {len(c_leak)}")
    print(f"either:                 {len(both)}  -> clean: {len(rows) - len(both)}")


if __name__ == "__main__":
    main()
