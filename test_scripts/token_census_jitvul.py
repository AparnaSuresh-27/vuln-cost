"""
Token census for the JITVul ladder, BEFORE spending GPU time. No model needed,
only the tokenizer, so it runs on the Mac.

    python -m test_scripts.token_census_jitvul
    python -m test_scripts.token_census_jitvul --tokenizer ~/studies/swb-s4130302-personal-study/vd-lora-v2   # on RACE, exact

Counts are chat-template-formatted input tokens, the same quantity the runner logs.
"""
import argparse
import os
import statistics as st

from transformers import AutoTokenizer

from vulncost.data_loader import JitVulLoader
from vulncost.enums.context_config import ContextConfig
from vulncost.prompt_builder_ft import build_ft_prompt

LIMITS = (4096, 8192, 16384)  # 4096 = fine-tuning max length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/final_benchmark.jsonl")
    ap.add_argument("--tokenizer", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer))

    def count(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    counts = {ContextConfig.function_only: [], ContextConfig.plus_caller_callee: []}
    for sample, extracted in JitVulLoader(args.data, clean=True, dedupe=True).get_samples():
        for cfg in counts:
            counts[cfg].append(count(build_ft_prompt(cfg, sample.func, extracted)))

    print(f"{len(counts[ContextConfig.function_only]) // 2} pairs\n")
    print(f"{'rung':<20}{'mean':>7}{'median':>8}{'p95':>7}{'max':>8}"
          + "".join(f"{'>' + str(l):>8}" for l in LIMITS))
    for cfg, xs in counts.items():
        s = sorted(xs)
        p95 = s[int(0.95 * (len(s) - 1))]
        print(f"{cfg.value:<20}{st.mean(xs):>7.0f}{st.median(xs):>8.0f}{p95:>7}{max(xs):>8}"
              + "".join(f"{sum(x > l for x in xs):>8}" for l in LIMITS))
    fo = sum(counts[ContextConfig.function_only])
    cc = sum(counts[ContextConfig.plus_caller_callee])
    print(f"\ntotal input tokens: fo {fo}, cc {cc}, ratio {cc / fo:.2f}x")


if __name__ == "__main__":
    main()