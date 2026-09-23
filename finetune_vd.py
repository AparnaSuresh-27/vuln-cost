#!/usr/bin/env python3
"""
QLoRA fine-tune of Qwen2.5-Coder-7B for function-level vulnerability detection
on PrimeVul train. Produces the LoRA adapter used as the FIXED detector across
the context ladder. Fits on one A10G (24 GB).

v3 changes (v1/v2 were trained without these):
  * loss on the ANSWER ONLY (train_on_responses_only). v2 computed loss on every
    token of prompt + code, so the one-token label was a tiny part of the signal.
  * JITVul held out: every PrimeVul function from a JITVul CVE, and any function
    whose body matches a JITVul body, is removed (--exclude-jitvul).
  * code is whitespace-normalised exactly as at evaluation time.
  * examples longer than --maxlen are dropped, not truncated (truncation would
    cut off the answer, i.e. the only token that carries loss).
  * optional --paired: patched counterparts from PrimeVul's paired train file
    are always included as SAFE, so training contains the near-identical
    distinctions the pairwise evaluation tests.

The PROMPT below MUST NOT change: the runners import it from here.

    # check data + masking only, no training (a few minutes)
    python3 finetune_vd.py --exclude-jitvul data/final_benchmark.jsonl \
        --paired data/primevul_train_paired.jsonl --check-only
    # real run
    python3 finetune_vd.py --exclude-jitvul data/final_benchmark.jsonl \
        --paired data/primevul_train_paired.jsonl --epochs 1 --out qwen-coder-7b-vd-lora-v3
"""
import argparse
import json
import random
import re

# === the one template, reused in every rung runner ===
PROMPT = """You are a security analyst. Analyze the following C/C++ code for security vulnerabilities.

{code}

Is this code VULNERABLE or SAFE? Answer with exactly one word: VULNERABLE or SAFE."""


def _nows(s):
    return re.sub(r"\s+", "", s or "")


def _read(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _func_target(r):
    func = r.get("func") or r.get("function") or r.get("code")
    tgt = r.get("target")
    if tgt is None:
        tgt = r.get("label")
    return func, (None if tgt is None else int(tgt))


def jitvul_holdout(path):
    """CVEs and whitespace-free bodies of every JITVul function (both halves)."""
    from vulncost.data_loader import JitVulLoader
    cves, bodies = set(), set()
    for pf, _ in JitVulLoader(path, clean=True, dedupe=False).get_samples():
        if pf.cve:
            cves.add(pf.cve)
        bodies.add(_nows(pf.func))
    return cves, bodies


def build_data(train, oversample_vuln, max_safe, exclude_jitvul=None,
               paired=None, normalize=True, seed=0):
    from vulncost.data_loader import JitVulLoader
    norm = JitVulLoader.normalize if normalize else (lambda s: s)
    hold_cves, hold_bodies = jitvul_holdout(exclude_jitvul) if exclude_jitvul else (set(), set())

    stats = {"raw": 0, "dropped_cve": 0, "dropped_body": 0}

    def keep(r, func):
        if r.get("cve") in hold_cves:
            stats["dropped_cve"] += 1
            return False
        if _nows(func) in hold_bodies:
            stats["dropped_body"] += 1
            return False
        return True

    vuln, safe, seen = [], [], set()
    for r in _read(train):
        func, tgt = _func_target(r)
        if not func or tgt is None:
            continue
        stats["raw"] += 1
        if not keep(r, func):
            continue
        seen.add(_nows(func))
        ex = {"code": norm(func), "label": "VULNERABLE" if tgt == 1 else "SAFE"}
        (vuln if tgt == 1 else safe).append(ex)

    paired_safe = []
    if paired:
        for r in _read(paired):
            func, tgt = _func_target(r)
            if not func or tgt != 0 or not keep(r, func):
                continue
            paired_safe.append({"code": norm(func), "label": "SAFE"})
        # these are usually already in train; take them out of the random pool
        ps = {_nows(e["code"]) for e in paired_safe}
        safe = [e for e in safe if _nows(e["code"]) not in ps]

    rng = random.Random(seed)
    if max_safe and len(safe) > max_safe:
        safe = rng.sample(safe, max_safe)
    data = paired_safe + safe + vuln * oversample_vuln
    rng.shuffle(data)
    stats.update(vuln=len(vuln), paired_safe=len(paired_safe), random_safe=len(safe),
                 total=len(data))
    return data, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/primevul_train.jsonl")
    ap.add_argument("--exclude-jitvul", help="final_benchmark.jsonl: hold out its CVEs and bodies")
    ap.add_argument("--paired", help="primevul_train_paired.jsonl: always include patched SAFE")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--max-safe", type=int, default=12000)
    ap.add_argument("--out", default="qwen-coder-7b-vd-lora-v3")
    ap.add_argument("--check-only", action="store_true",
                    help="build data, apply masking, print checks, then exit")
    args = ap.parse_args()

    raw, stats = build_data(args.train, args.oversample, args.max_safe,
                            args.exclude_jitvul, args.paired, not args.no_normalize)
    print("data:", json.dumps(stats))

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        max_seq_length=args.maxlen, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")

    def to_text(ex):
        msgs = [{"role": "user", "content": PROMPT.format(code=ex["code"])},
                {"role": "assistant", "content": ex["label"]}]
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    # drop, don't truncate: a truncated example loses its answer
    texts, labels, too_long = [], [], 0
    for ex in raw:
        t = to_text(ex)
        if len(tokenizer(t, add_special_tokens=False)["input_ids"]) > args.maxlen:
            too_long += 1
            continue
        texts.append(t)
        labels.append(ex["label"])
    n_v = labels.count("VULNERABLE")
    print(f"dropped {too_long} examples over {args.maxlen} tokens; "
          f"training on {len(texts)} ({n_v} VULNERABLE, {len(texts) - n_v} SAFE)")

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=args.maxlen,
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            num_train_epochs=args.epochs, learning_rate=2e-4, warmup_ratio=0.05,
            weight_decay=0.01, lr_scheduler_type="cosine", logging_steps=20,
            save_strategy="epoch", output_dir=args.out, bf16=True,
            optim="adamw_8bit", seed=0))
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n")

    # verify the mask: only the answer (+ end token) may carry loss
    counts = []
    for i in range(min(50, len(trainer.train_dataset))):
        lab = trainer.train_dataset[i]["labels"]
        counts.append(sum(1 for x in lab if x != -100))
    lab0 = trainer.train_dataset[0]["labels"]
    shown = tokenizer.decode([x for x in lab0 if x != -100])
    print(f"loss tokens per example (first 50): min {min(counts)}, max {max(counts)}")
    print(f"example 0 loss text: {shown!r}")
    if max(counts) > 8 or min(counts) == 0:
        raise SystemExit("MASK CHECK FAILED: loss is not restricted to the answer. Not training.")
    print("mask check passed")
    if args.check_only:
        return

    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(f"{args.out}/train_meta.json", "w") as f:
        json.dump({**stats, "too_long_dropped": too_long, "trained_on": len(texts),
                   "vulnerable": n_v, "args": vars(args)}, f, indent=2)
    print(f"\nSaved LoRA adapter to {args.out}/")


if __name__ == "__main__":
    main()