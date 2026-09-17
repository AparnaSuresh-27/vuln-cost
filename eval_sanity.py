#!/usr/bin/env python3
"""
Sanity-check the fine-tuned adapter on PrimeVul TEST before any ladder work.
Confirms the model learned to detect (beats chance, non-trivial P-C) rather than
collapsing to always-VULNERABLE or always-SAFE.

Uses the SAME PROMPT as finetune_vd.py (must match, or the model is off-format).
Run on the same box, after training:
    python eval_sanity.py --adapter qwen-coder-7b-vd-lora --test data/primevul_test_paired.jsonl
"""
import argparse
import json
import re
from collections import Counter

PROMPT = """You are a security analyst. Analyze the following C/C++ code for security vulnerabilities.

{code}

Is this code VULNERABLE or SAFE? Answer with exactly one word: VULNERABLE or SAFE."""

_CLASS = re.compile(r"\b(VULNERABLE|SAFE)\b")


def verdict(text):
    m = _CLASS.findall(text.upper())
    return m[-1] if m else "UNKNOWN"


def label_str(t):
    return "VULNERABLE" if int(t) == 1 else "SAFE"


def load(path, limit):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        func = r.get("func") or r.get("function") or r.get("code")
        tgt = r.get("target")
        if tgt is None:
            tgt = r.get("label")
        if not func or tgt is None:
            continue
        rows.append((func, int(tgt)))
        if limit and len(rows) >= limit:
            break
    return rows


def p_c(results):
    """results: [(target, pred)] in pair order (test_paired is vuln/safe consecutive)."""
    c = 0
    for i in range(0, len(results) - 1, 2):
        t1, p1 = results[i]
        t2, p2 = results[i + 1]
        if p1 == label_str(t1) and p2 == label_str(t2):
            c += 1
    return c, len(results) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="qwen-coder-7b-vd-lora")
    ap.add_argument("--test", default="data/primevul_test_paired.jsonl")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=8192, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    rows = load(args.test, args.limit)
    print(f"eval samples: {len(rows)}")

    results = []
    dist = Counter()
    correct = 0
    for func, tgt in rows:
        msgs = [{"role": "user", "content": PROMPT.format(code=func[:16000])}]
        inputs = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(inputs, max_new_tokens=8, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        pred = verdict(text)
        results.append((tgt, pred))
        dist[pred] += 1
        if pred == label_str(tgt):
            correct += 1

    n = len(results)
    acc = correct / max(n, 1)
    c, pairs = p_c(results)
    print(f"\nprediction distribution: {dict(dist)}")
    print(f"accuracy: {acc:.3f}   P-C: {c}/{pairs} = {c/max(pairs,1):.3f}")
    print("\nRead this as:")
    print("- distribution heavily one-sided  -> model collapsed to one label, retrain")
    print("- acc ~0.5 and P-C ~0             -> at chance, not usable, retrain")
    print("- acc > 0.6 and P-C clearly > 0   -> learned to detect, good ladder model")


if __name__ == "__main__":
    main()

