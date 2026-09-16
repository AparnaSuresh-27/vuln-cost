import time
from vulncost.data_loader import JitVulLoader
from vulncost.prompt_builder import make_prompt_builder
from vulncost.inference import MLXInference, parse_prediction
from vulncost.logger import RunLogger, CallRecord
from vulncost.enums.approach import Approach
from vulncost.enums.context_config import ContextConfig

MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DATA_PATH = "data/final_benchmark.jsonl"
SUBSET_PAIRS = 300 # each pair = 2 samples; 2 rungs each = 4 inferences/pair


def label_str(t: int) -> str:
    return "VULNERABLE" if t == 1 else "SAFE"


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
    count = 0
    for i in range(0, len(results) - 1, 2):
        t1, p1 = results[i]
        t2, p2 = results[i + 1]
        if p1 == label_str(t1) and p2 == label_str(t2):
            count += 1
    return count, len(results) // 2


def main():
    loader = JitVulLoader(DATA_PATH)
    engine = MLXInference(MODEL_ID)
    logger = RunLogger("logs", "jitvul_comparison")

    fo_results, cc_results = [], []
    fo_tokens = cc_tokens = 0
    samples_done = 0
    target_samples = SUBSET_PAIRS * 2 if SUBSET_PAIRS else None

    for sample, extracted in loader.get_samples():
        if target_samples and samples_done >= target_samples:
            break

        fo_pred, fo_in = run_one(engine, logger, sample, ContextConfig.function_only)
        fo_results.append((sample.target, fo_pred))
        fo_tokens += fo_in

        cc_pred, cc_in = run_one(engine, logger, sample, ContextConfig.plus_caller_callee,
                                 extracted=extracted)
        cc_results.append((sample.target, cc_pred))
        cc_tokens += cc_in

        print(f"{sample.idx}[t={sample.target}]: fo={fo_pred}({fo_in}) "
              f"cc={cc_pred}({cc_in}) truth={label_str(sample.target)}")
        samples_done += 1

    logger.close()

    n = len(fo_results)
    fo_c, pairs = p_c(fo_results)
    cc_c, _ = p_c(cc_results)

    print("\n" + "=" * 55)
    print(f"JITVUL COMPARISON on {n} samples ({pairs} pairs)")
    print("=" * 55)
    print(f"Function-only:    P-C {fo_c}/{pairs} = {fo_c/max(pairs,1):.3f} | "
          f"{fo_tokens} input tokens ({fo_tokens/max(n,1):.0f}/sample)")
    print(f"+ Caller/Callee:  P-C {cc_c}/{pairs} = {cc_c/max(pairs,1):.3f} | "
          f"{cc_tokens} input tokens ({cc_tokens/max(n,1):.0f}/sample)")
    if fo_tokens:
        print(f"\nToken cost ratio (cc / fo): {cc_tokens/fo_tokens:.2f}x")


if __name__ == "__main__":
    main()