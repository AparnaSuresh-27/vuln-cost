"""
JITVul context ladder with ONE fixed detector: function-only vs + caller/callee.
Runs on the RACE GPU box, from the repo root.

    # smoke test: 3 pairs = 12 inferences
    python -m vulncost.runner_jitvul_ft --pairs 3 --label jitvul_ft_smoke
    # full run (800 unique pairs), survives a DCV disconnect
    nohup python -m vulncost.runner_jitvul_ft --label jitvul_ft_full > logs/jitvul_ft_full.out 2>&1 &
    # untrained control through the identical pipeline and prompt
    nohup python -m vulncost.runner_jitvul_ft --adapter base --label jitvul_base_full > logs/jitvul_base_full.out 2>&1 &

Writes logs/<timestamp>_<label>/calls.jsonl (one row per inference, flushed
as it goes) and run_meta.json (what produced it). Analyse with full_metrics.
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
from vulncost.prompt_builder_ft import build_ft_prompt
from vulncost.hf_inference import HFInference, parse_prediction, MAX_NEW_TOKENS
from vulncost.logger import RunLogger, CallRecord
from vulncost.enums.approach import Approach
from vulncost.enums.context_config import ContextConfig

DEFAULT_ADAPTER = "~/studies/swb-s4130302-personal-study/vd-lora-v2"
BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
RUNGS = (ContextConfig.function_only, ContextConfig.plus_caller_callee)


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


def write_meta(run_dir, args, model_path, model_name, n_pairs):
    import torch
    meta = {
        "started": datetime.now().isoformat(),
        "model_name": model_name,
        "model_path": model_path,
        "max_seq_length": args.max_seq,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False)",
        "prompt_template_sha256": hashlib.sha256(FT_TEMPLATE.encode()).hexdigest(),
        "data_path": args.data,
        "data_sha256": sha256(args.data),
        "loader": "JitVulLoader(clean=True, dedupe=True)",
        "pairs_requested": args.pairs or "all",
        "pairs_available": n_pairs,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "host": platform.node(),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help="LoRA adapter dir, or 'base' for the untrained control")
    ap.add_argument("--data", default="data/final_benchmark.jsonl")
    ap.add_argument("--pairs", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--max-seq", type=int, default=16384)
    ap.add_argument("--label", default="jitvul_ft")
    args = ap.parse_args()

    if args.adapter == "base":
        model_path, model_name = BASE_MODEL, "qwen2.5-coder-7b-instruct-4bit"
    else:
        model_path = os.path.expanduser(args.adapter)
        if not os.path.isfile(os.path.join(model_path, "adapter_config.json")):
            raise SystemExit(f"no adapter_config.json in {model_path}")
        model_name = "qwen2.5-coder-7b-instruct-4bit+" + os.path.basename(model_path.rstrip("/"))

    loader = JitVulLoader(args.data, clean=True, dedupe=True)
    samples = list(loader.get_samples())  # small; lets us print an ETA
    n_pairs = len(samples) // 2
    if args.pairs:
        samples = samples[: 2 * args.pairs]
        n_pairs = min(n_pairs, args.pairs)

    engine = HFInference(model_path, args.max_seq)
    logger = RunLogger("logs", args.label)
    run_dir = os.path.dirname(str(logger.path))
    write_meta(run_dir, args, model_path, model_name, n_pairs)
    print(f"run {logger.run_id}: {n_pairs} pairs, {len(samples) * len(RUNGS)} inferences")

    t_start = time.time()
    for i, (sample, extracted) in enumerate(samples):
        line = []
        for cfg in RUNGS:
            prompt = build_ft_prompt(cfg, sample.func, extracted)
            t0 = time.time()
            response, in_tok, out_tok, status = engine.run(prompt)
            dur = time.time() - t0
            pred = parse_prediction(response) if status == "ok" else "UNKNOWN"
            logger.log(CallRecord(
                model=model_name, approach=Approach.context, context_config=cfg,
                agent=None, sample_id=sample.idx, repeat=0, call_number=1,
                prompt=prompt, response=response if status == "ok" else status.upper(),
                input_tokens=in_tok, output_tokens=out_tok, prediction=pred,
                label=sample.target, status="ok" if status == "ok" else "failed",
                duration_s=dur, temperature=0.0, seed=0,
            ))
            tag = "fo" if cfg == ContextConfig.function_only else "cc"
            line.append(f"{tag}={pred}({in_tok}{'' if status == 'ok' else ' ' + status})")

        done = i + 1
        eta = (time.time() - t_start) / done * (len(samples) - done) / 60
        truth = "VULNERABLE" if sample.target == 1 else "SAFE"
        print(f"[{done}/{len(samples)}] {sample.idx} truth={truth} "
              f"{' '.join(line)}  eta {eta:.0f} min", flush=True)

    logger.close()
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    print(f"analyse: python -m vulncost.full_metrics {logger.path}")


if __name__ == "__main__":
    main()