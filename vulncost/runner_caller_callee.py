import json
import time
from pathlib import Path

from vulncost.data_loader import PairedFunctionLoader
from vulncost.prompt_builder import make_prompt_builder
from vulncost.inference import MLXInference, parse_prediction
from vulncost.logger import RunLogger, CallRecord
from vulncost.enums.approach import Approach
from vulncost.enums.context_config import ContextConfig

MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DATA_PATH = "data/primevul_valid_paired.jsonl"
EXTRACT_DIR = Path("data/extracted")


def label_str(target: int) -> str:
    return "VULNERABLE" if target == 1 else "SAFE"


def load_extracted(idx: str):
    path = EXTRACT_DIR / f"{idx}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def run_one(engine, logger, sample, context_config, extracted=None):
    prompt = make_prompt_builder(
        Approach.context, context_config, sample.func, extracted=extracted
    ).build()

    start = time.time()
    response, in_tok, out_tok = engine.run(prompt)
    duration = time.time() - start
    prediction = parse_prediction(response)

    logger.log(CallRecord(
        model=MODEL_ID, approach=Approach.context, context_config=context_config,
        agent=None, sample_id=sample.idx, repeat=0, call_number=1,
        prompt=prompt, response=response, input_tokens=in_tok, output_tokens=out_tok,
        prediction=prediction, label=sample.target, status="ok",
        duration_s=duration, temperature=0.0, seed=0,
    ))
    return prediction, in_tok


def p_c(results):
    # results: list of (target, prediction) in pair order
    count = 0
    for i in range(0, len(results) - 1, 2):
        t1, p1 = results[i]
        t2, p2 = results[i + 1]
        if p1 == label_str(t1) and p2 == label_str(t2):
            count += 1
    pairs = len(results) // 2
    return count, pairs


def main():
    loader = PairedFunctionLoader(DATA_PATH)
    engine = MLXInference(MODEL_ID)
    logger = RunLogger("logs", "caller_callee_comparison")

    fo_results, cc_results = [], []      # (target, prediction) per rung
    fo_tokens, cc_tokens = 0, 0

    for sample in loader.get_samples():
        extracted = load_extracted(sample.idx)
        if extracted is None:
            continue  # only samples we have extracted context for

        # rung 1: function only
        fo_pred, fo_in = run_one(engine, logger, sample, ContextConfig.function_only)
        fo_results.append((sample.target, fo_pred))
        fo_tokens += fo_in

        # rung 2: + caller/callee
        cc_pred, cc_in = run_one(engine, logger, sample, ContextConfig.plus_caller_callee,
                                 extracted=extracted)
        cc_results.append((sample.target, cc_pred))
        cc_tokens += cc_in

        print(f"{sample.idx}: fo={fo_pred}({fo_in}tok) "
              f"cc={cc_pred}({cc_in}tok) truth={label_str(sample.target)}")

    logger.close()

    n = len(fo_results)
    fo_c, pairs = p_c(fo_results)
    cc_c, _ = p_c(cc_results)

    print("\n" + "=" * 55)
    print(f"MATCHED COMPARISON on {n} samples ({pairs} pairs)")
    print("=" * 55)
    print(f"Function-only:      P-C {fo_c}/{pairs} = {fo_c/max(pairs,1):.3f} | "
          f"{fo_tokens} input tokens ({fo_tokens/max(n,1):.0f}/sample)")
    print(f"+ Caller/Callee:    P-C {cc_c}/{pairs} = {cc_c/max(pairs,1):.3f} | "
          f"{cc_tokens} input tokens ({cc_tokens/max(n,1):.0f}/sample)")
    if fo_tokens:
        print(f"\nToken cost ratio (cc / fo): {cc_tokens/fo_tokens:.2f}x")


if __name__ == "__main__":
    main()