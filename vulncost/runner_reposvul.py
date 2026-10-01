"""
Run the fixed detector over the ReposVul ladder: function-only -> CPG slice -> full file.

Every level uses the SAME prompt template as training (PROMPT from finetune_vd.py) and
the same whitespace normalisation as the JITVul runs; only what goes into {code} changes.

    # smoke test: 3 samples, all levels
    python3 -m vulncost.runner_reposvul --adapter qwen-coder-7b-vd-lora-v3 --limit 3 --label rv_smoke
    # full runs
    python3 -m vulncost.runner_reposvul --adapter qwen-coder-7b-vd-lora-v3 --label rv_v3
    python3 -m vulncost.runner_reposvul --adapter base --label rv_base

Output: logs/<timestamp>_<label>/calls.jsonl, one row per (sample, level).
Use --resume <run_dir> to continue an interrupted run without redoing finished calls.
"""
import argparse
import json
import os
import time
from datetime import datetime

from finetune_vd import PROMPT

BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
LEVELS = ["function", "slice", "full"]


def get_normalizer():
    """Use the exact normaliser from the JITVul pipeline so inputs match training."""
    try:
        from vulncost.data_loader import JitVulLoader
        print("normalisation: JitVulLoader.normalize")
        return JitVulLoader.normalize
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not import JitVulLoader.normalize ({e}). "
                         "Fix the import rather than running un-normalised inputs.")


def parse_prediction(text: str) -> str:
    t = text.strip().upper()
    if t.startswith("VULNERABLE"):
        return "VULNERABLE"
    if t.startswith("SAFE"):
        return "SAFE"
    if "VULNERABLE" in t:
        return "VULNERABLE"
    if "SAFE" in t:
        return "SAFE"
    return "UNPARSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/reposvul_ladder.jsonl")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'base' for the untrained model")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--max-input-tokens", type=int, default=16384,
                    help="skip prompts longer than this (same cap as the JITVul runs)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default="reposvul")
    ap.add_argument("--resume", default=None, help="existing run dir to append to")
    args = ap.parse_args()

    import torch
    from unsloth import FastLanguageModel

    levels = args.levels.split(",")
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        # take vulnerable and safe samples alternately so a smoke test sees both labels
        v = [r for r in rows if r["label"] == 1]
        s = [r for r in rows if r["label"] == 0]
        rows = [x for pair in zip(v, s) for x in pair][:args.limit]

    if args.resume:
        run_dir = args.resume
    else:
        run_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.label)
        os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "calls.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            if l.strip():
                d = json.loads(l)
                done.add((d["idx"], d["context_config"]))
    print(f"run dir: {run_dir}  ({len(done)} calls already done)")

    model_name = BASE_MODEL if args.adapter == "base" else args.adapter
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=args.max_input_tokens + 64, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    normalize = get_normalizer()

    total = len(rows) * len(levels)
    n_done, t0 = len(done), time.time()
    with open(out_path, "a") as f:
        for r in rows:                      # all levels of a sample together -> complete cases
            for lvl in levels:
                if (r["idx"], lvl) in done:
                    continue
                code = normalize(r[lvl])
                msgs = [{"role": "user", "content": PROMPT.format(code=code)}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              return_tensors="pt")
                n_in = ids.shape[-1]
                rec = {"idx": r["idx"], "label": r["label"], "context_config": lvl,
                       "adapter": args.adapter, "input_tokens": n_in, "output_tokens": 0,
                       "latency_s": None, "raw_output": "", "prediction": None,
                       "status": "ok", "cwe": r.get("cwe"), "project": r.get("project")}
                if n_in > args.max_input_tokens:
                    rec["status"] = "too_long"
                else:
                    try:
                        ids = ids.to(model.device)
                        torch.cuda.synchronize()
                        ts = time.time()
                        with torch.no_grad():
                            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                                 max_new_tokens=8, do_sample=False,
                                                 pad_token_id=tok.eos_token_id)
                        torch.cuda.synchronize()
                        rec["latency_s"] = round(time.time() - ts, 3)
                        new = gen[0, n_in:]
                        rec["output_tokens"] = int(new.shape[-1])
                        rec["raw_output"] = tok.decode(new, skip_special_tokens=True)
                        rec["prediction"] = parse_prediction(rec["raw_output"])
                    except torch.cuda.OutOfMemoryError:
                        rec["status"] = "oom"
                        torch.cuda.empty_cache()
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_done += 1
                el = time.time() - t0
                print(f"[{n_done}/{total}] idx={r['idx']} {lvl:8s} in={n_in:6d} "
                      f"label={r['label']} pred={rec['prediction']} {rec['status']} "
                      f"elapsed={el/60:.1f}m", flush=True)
    print(f"done {run_dir}")


if __name__ == "__main__":
    main()
