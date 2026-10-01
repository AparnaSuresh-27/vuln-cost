"""
ReposVul context ladder with the SAME fixed detector and inference path as the
JITVul runs: function-only -> CPG slice -> full file.

Uses HFInference / parse_prediction / MAX_NEW_TOKENS from hf_inference.py and the
PROMPT from finetune_vd.py unchanged, and normalises every input with
JitVulLoader.normalize, so the only thing that differs from JITVul is the data.

    # smoke test: 3 samples x 3 levels = 9 inferences
    python -m vulncost.runner_reposvul --adapter <v3 adapter dir> --limit 3 --label rv_smoke
    # full runs (119 samples x 3 levels = 357 inferences each)
    nohup python -m vulncost.runner_reposvul --adapter <v3 adapter dir> --label rv_v3 > logs/rv_v3.out 2>&1 &
    nohup python -m vulncost.runner_reposvul --adapter base --label rv_base > logs/rv_base.out 2>&1 &

Writes logs/<timestamp>_<label>/calls.jsonl (one row per inference, flushed as it
goes) and run_meta.json. Analyse with vulncost.reposvul_metrics.
--resume <run_dir> continues an interrupted run without redoing finished calls.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime

from finetune_vd import PROMPT as FT_TEMPLATE
from vulncost.data_loader import JitVulLoader
from vulncost.hf_inference import HFInference, parse_prediction, MAX_NEW_TOKENS

BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
LEVELS = ["function", "slice", "full"]
TAGS = {"function": "fn", "slice": "sl", "full": "file"}
SOURCE = "data/reposvul_test.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="LoRA adapter dir, or 'base' for the untrained control")
    ap.add_argument("--data", default="data/reposvul_ladder.jsonl")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--max-seq", type=int, default=16384, help="same cap as the JITVul runs")
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    ap.add_argument("--label", default="reposvul")
    ap.add_argument("--resume", default=None, help="existing run dir to continue")
    args = ap.parse_args()
    levels = args.levels.split(",")

    if args.adapter == "base":
        model_path, model_name = BASE_MODEL, "qwen2.5-coder-7b-instruct-4bit"
    else:
        model_path = os.path.expanduser(args.adapter)
        if not os.path.isfile(os.path.join(model_path, "adapter_config.json")):
            raise SystemExit(f"no adapter_config.json in {model_path}")
        model_name = "qwen2.5-coder-7b-instruct-4bit+" + os.path.basename(model_path.rstrip("/"))

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        # alternate vulnerable / safe so a smoke test sees both labels
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

    engine = HFInference(model_path, args.max_seq)

    import torch
    meta = {
        "started": datetime.now().isoformat(),
        "model_name": model_name, "model_path": model_path,
        "max_seq_length": args.max_seq, "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False)",
        "prompt_template_sha256": hashlib.sha256(FT_TEMPLATE.encode()).hexdigest(),
        "normalisation": "JitVulLoader.normalize on every input",
        "levels": levels,
        "data_path": args.data, "data_sha256": sha256(args.data),
        "source_path": SOURCE,
        "source_sha256": sha256(SOURCE) if os.path.exists(SOURCE) else None,
        "source_origin": "github.com/qcri/llmxcpg data/reposvul_test.json @ fd7af0d",
        "samples": len(rows), "resumed_calls": len(done),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "host": platform.node(),
    }
    meta_name = "run_meta.json" if not args.resume else f"run_meta_resume_{datetime.now():%H%M%S}.json"
    with open(os.path.join(run_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)

    total = len(rows) * len(levels)
    print(f"run {run_dir}: {len(rows)} samples x {len(levels)} levels = {total} inferences "
          f"({len(done)} already done)")

    t_start, n_new = time.time(), 0
    todo = total - len(done)
    with open(out_path, "a") as f:
        for i, r in enumerate(rows):            # all levels of a sample together
            line = []
            for lvl in levels:
                if (r["idx"], lvl) in done:
                    continue
                prompt = FT_TEMPLATE.format(code=JitVulLoader.normalize(r[lvl]))
                t0 = time.time()
                response, in_tok, out_tok, status = engine.run(prompt)
                dur = time.time() - t0
                pred = parse_prediction(response) if status == "ok" else "UNKNOWN"
                f.write(json.dumps({
                    "idx": r["idx"], "label": r["label"], "context_config": lvl,
                    "model": model_name, "input_tokens": in_tok, "output_tokens": out_tok,
                    "latency_s": round(dur, 3) if status == "ok" else None,
                    "response": response, "prediction": pred, "status": status,
                    "cwe": r.get("cwe"), "project": r.get("project"),
                }) + "\n")
                f.flush()
                n_new += 1
                line.append(f"{TAGS[lvl]}={pred}({in_tok}{'' if status == 'ok' else ' ' + status})")
            if line:
                eta = (time.time() - t_start) / n_new * (todo - n_new) / 60
                truth = "VULNERABLE" if r["label"] == 1 else "SAFE"
                print(f"[{i + 1}/{len(rows)}] idx={r['idx']} truth={truth} "
                      f"{' '.join(line)}  eta {eta:.0f} min", flush=True)

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    print(f"analyse: python -m vulncost.reposvul_metrics {out_path}")


if __name__ == "__main__":
    main()