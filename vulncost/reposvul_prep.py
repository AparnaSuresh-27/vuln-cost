"""
Build the ReposVul context ladder from LLMxCPG's released test set.

Source: github.com/qcri/llmxcpg  data/reposvul_test.json  (120 samples: 60 vuln, 60 safe).
Each record already ships two context levels:
    input -> the CPG slice produced by LLMxCPG's own query model (LLMxCPG-Q + Joern)
    code  -> the full source file(s) the slice was cut from
We add a third, function-only, by taking the top-level function in `code` that
contains the most slice lines (an approximation of ReposVul's labelled target).

Output: data/reposvul_ladder.jsonl, one row per sample, with all three levels
side by side so every level is scored on exactly the same samples.

    python3 -m vulncost.reposvul_prep
"""
import json
import re
import sys

SRC = "data/reposvul_test.json"
OUT = "data/reposvul_ladder.jsonl"


def slice_body(inp: str) -> str:
    """Strip LLMxCPG's '### Source Code:\\n```' wrapper and closing fence."""
    s = inp.split("```", 1)[1] if "```" in inp else inp
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s


def _top_level_blocks(code: str):
    """Spans (open, close) of every top-level {...} block, skipping comments and literals."""
    blocks, depth, j, n, open_at = [], 0, 0, len(code), None
    while j < n:
        c = code[j]
        if c == "/" and code.startswith("//", j):
            j = code.find("\n", j); j = n if j < 0 else j
            continue
        if c == "/" and code.startswith("/*", j):
            j = code.find("*/", j + 2); j = n if j < 0 else j + 2
            continue
        if c in "\"'":
            q, j = c, j + 1
            while j < n and code[j] != q and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            j += 1
            continue
        if c == "{":
            if depth == 0:
                open_at = j
            depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                blocks.append((open_at, j))
        j += 1
    return blocks


def _functions(code: str):
    """(start, end) spans of top-level function definitions: a top-level {...} block
    whose header since the previous top-level item contains ')'."""
    blocks = _top_level_blocks(code)
    out, prev_end = [], -1
    for o, c in blocks:
        head = code[prev_end + 1:o]
        prev_end = c
        semi = head.rfind(";")
        if semi >= 0:
            head = head[semi + 1:]
        if ")" not in head:                          # struct / enum / initializer
            continue
        lines = head.split("\n")
        while lines and (not lines[0].strip()
                         or lines[0].lstrip().startswith(("#", "//", "/*", "*"))):
            lines.pop(0)
        out.append(("\n".join(lines), o, c))
    return out


def extract_function(code: str, slice_text: str):
    """Return the function the slice is mostly drawn from.

    LLMxCPG slices start at the target function but can pull in callers/callees, so
    the slice's first line is not always inside the target. We score every top-level
    function in the file by how many distinct, non-trivial slice lines it contains and
    take the best; ties go to the earliest. This is an approximation of ReposVul's
    labelled target function and is documented as such in the methods.
    """
    sl = {ln.strip() for ln in slice_text.splitlines() if len(ln.strip()) > 12}
    best, best_score = None, 0
    for head, o, c in _functions(code):
        body = code[o:c + 1]
        fn_lines = {ln.strip() for ln in (head + body).splitlines()}
        score = len(sl & fn_lines)
        if score > best_score:
            best, best_score = head + body, score
    return best, best_score, len(sl)


def main():
    data = json.load(open(SRC))
    rows, failed = [], []
    for k, r in enumerate(data):
        sl = slice_body(r["input"])
        fn, hit, tot = extract_function(r["code"], sl)
        if fn is None:
            failed.append(k)
            continue
        rows.append({
            "idx": k,
            "label": 1 if r["output"].strip().lower() == "yes" else 0,
            "cwe": r.get("cwe"), "project": r.get("project"), "url": r.get("url"),
            "file_name": r.get("file_name"),
            "function": fn, "slice": sl, "full": r["code"],
            "fn_slice_overlap": round(hit / tot, 3) if tot else None,
        })
    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}; function extraction failed for {len(failed)}: {failed}")
    v = sum(r["label"] for r in rows)
    print(f"labels: {v} vulnerable, {len(rows) - v} safe")


if __name__ == "__main__":
    main()
