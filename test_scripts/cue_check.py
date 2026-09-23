"""
Does the detector's answer track FORMATTING rather than code?
Reads a calls.jsonl, joins each function-only prediction back to the raw
JITVul body, and cross-tabs the prediction against a formatting feature
(does the body end with a newline) separately for each true label. If the
answer moves with the newline while the label is held fixed, the model is
reading formatting.

    python3 -m test_scripts.format_cue_check logs/<run>/calls.jsonl
"""
import argparse
import json
import re
from collections import Counter

from vulncost.data_loader import JitVulLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("--data", default="data/final_benchmark.jsonl")
    args = ap.parse_args()

    # raw bodies as the RUN saw them (clean + dedupe, no whitespace normalisation)
    raw = {(pf.idx, pf.target): pf.func for pf, _ in
           JitVulLoader(args.data, clean=True, dedupe=True).get_samples()}
    rows = [json.loads(l) for l in open(args.calls) if l.strip()]
    fo = [r for r in rows if r["context_config"] == "function_only" and r["status"] == "ok"]

    tab = Counter()
    for r in fo:
        body = raw[(r["sample_id"], r["label"])]
        tab[(r["label"], body.endswith("\n"), r["prediction"])] += 1

    print("function-only: share predicted VULNERABLE, by true label and trailing newline")
    for label in (1, 0):
        for nl in (True, False):
            v = tab[(label, nl, "VULNERABLE")]
            n = v + tab[(label, nl, "SAFE")] + tab[(label, nl, "UNKNOWN")]
            if n:
                print(f"  truth={'VULN' if label else 'SAFE'}  ends_with_newline={str(nl):<5}"
                      f"  {v}/{n} = {v / n:.2f}")

    # pairs whose code is identical apart from whitespace: a formatting-blind
    # detector must give both halves the same answer
    by_id = {}
    for r in fo:
        by_id.setdefault(r["sample_id"], {})[r["label"]] = r["prediction"]
    nows = lambda s: re.sub(r"\s+", "", s)
    same = [i for i, d in by_id.items() if len(d) == 2
            and nows(raw[(i, 1)]) == nows(raw[(i, 0)])]
    split = sum(by_id[i][1] != by_id[i][0] for i in same)
    print(f"\ncode-identical pairs: {len(same)}; given DIFFERENT answers: {split}")


if __name__ == "__main__":
    main()