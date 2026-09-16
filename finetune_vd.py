#!/usr/bin/env python3
"""
QLoRA fine-tune of Qwen2.5-Coder-7B for function-level vulnerability detection
on PrimeVul train. Produces a LoRA adapter used as the FIXED detector across the
whole context ladder. Fits on one A10G (24 GB).

Recipe follows standard VD LoRA setups (rank 16, ~2 epochs, oversample the
vulnerable class). The PROMPT below MUST be reused verbatim in every rung runner
(function-only, slice, full-file, explanation) so the model always sees the format
it was trained on -- the rungs differ only in what goes in {code}.

Run on g5.xlarge (1x A10G):
    pip install unsloth
    export HF_HOME=/dev/shm/hf          # if disk is tight; else omit
    python finetune_vd.py --train data/primevul_train.jsonl

NOTE: unsloth / trl APIs drift between versions. If SFTConfig/SFTTrainer args
error, paste the traceback -- it's usually a one-line arg rename, not a real problem.
"""
import argparse
import json
import random

# === the one template, reused in every rung runner ===
PROMPT = """You are a security analyst. Analyze the following C/C++ code for security vulnerabilities.

{code}

Is this code VULNERABLE or SAFE? Answer with exactly one word: VULNERABLE or SAFE."""


def load_primevul(path, oversample_vuln):
    vuln, safe = [], []
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
        ex = {"code": func, "label": "VULNERABLE" if int(tgt) == 1 else "SAFE"}
        (vuln if int(tgt) == 1 else safe).append(ex)
    print(f"raw: {len(vuln)} vulnerable, {len(safe)} safe")
    data = safe + vuln * oversample_vuln          # push minority class toward balance
    random.seed(0)
    random.shuffle(data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/primevul_train.jsonl")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--oversample", type=int, default=10)
    ap.add_argument("--out", default="qwen-coder-7b-vd-lora")
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        max_seq_length=args.maxlen,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    raw = load_primevul(args.train, args.oversample)
    print(f"training examples (after oversampling): {len(raw)}")

    def to_text(ex):
        msgs = [
            {"role": "user", "content": PROMPT.format(code=ex["code"])},
            {"role": "assistant", "content": ex["label"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    ds = Dataset.from_list(raw).map(to_text)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.maxlen,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            logging_steps=20,
            save_strategy="epoch",
            output_dir=args.out,
            bf16=True,
            optim="adamw_8bit",
            seed=0,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved LoRA adapter to {args.out}/")
    print("Next: sanity-eval on PrimeVul TEST before running the ladder.")


if __name__ == "__main__":
    main()